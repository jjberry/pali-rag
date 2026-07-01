"""Local, single-user web UI: read saved answers, ask, and chat in a browser.

Stdlib ThreadingHTTPServer + Jinja2 templates + markdown-it-py rendering. Heavy
singletons (the embed-model Retriever and the Anthropic client) are created once
and shared across requests; pipeline/chat work is serialized under a lock since
it's a single-user tool. Binds to 127.0.0.1 only.
"""
from __future__ import annotations

import os
import sys
import threading
import uuid
from datetime import datetime
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from jinja2 import Environment, FileSystemLoader, select_autoescape  # noqa: E402
from rag import pipeline  # noqa: E402
from rag.chat import ChatSession  # noqa: E402
from web.render import markdown_to_html  # noqa: E402

HERE = Path(__file__).resolve().parent
env = Environment(
    loader=FileSystemLoader(str(HERE / "templates")),
    autoescape=select_autoescape(["html"]),
)

# --- shared heavy singletons (built lazily, guarded by _lock) --------------
_lock = threading.Lock()
_retriever = None
_client = None
_chats: dict[str, ChatSession] = {}
_high_quality = False


def _get_retriever():
    global _retriever
    if _retriever is None:
        from rag.retriever import Retriever
        _retriever = Retriever()
    return _retriever


def _get_client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic()
    return _client


def _model_name() -> str:
    return config.GEN_MODEL_HQ if _high_quality else config.GEN_MODEL


def _have_api_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


# --- saved-answer listing --------------------------------------------------
def _list_answers() -> list[dict]:
    if not config.ANSWERS_DIR.exists():
        return []
    items = []
    for p in config.ANSWERS_DIR.glob("*.md"):
        items.append({
            "name": p.name,
            "title": _title_of(p),
            "modified": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            "mtime": p.stat().st_mtime,
        })
    items.sort(key=lambda a: a["mtime"], reverse=True)
    return items


def _title_of(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
        if line.strip():
            return line.strip()
    return path.stem


def _answer_path(name: str) -> Path | None:
    """Resolve a requested answer file safely (no path traversal)."""
    if not name.endswith(".md") or "/" in name or "\\" in name:
        return None
    p = config.ANSWERS_DIR / name
    return p if p.is_file() else None


class Handler(BaseHTTPRequestHandler):
    server_version = "pali-rag-web"

    def log_message(self, fmt, *args):  # quieter than the default
        sys.stderr.write("  %s - %s\n" % (self.address_string(), fmt % args))

    # --- response helpers --------------------------------------------------
    def _send(self, code: int, body: str, ctype="text/html; charset=utf-8",
              cookie: str | None = None):
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.end_headers()
        self.wfile.write(data)

    def _render(self, template: str, code=200, cookie=None, **ctx):
        self._send(code, env.get_template(template).render(**ctx), cookie=cookie)

    def _form(self) -> dict[str, str]:
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n).decode("utf-8") if n else ""
        return {k: v[0] for k, v in parse_qs(raw).items()}

    def _cookie(self, key: str) -> str | None:
        c = SimpleCookie(self.headers.get("Cookie", ""))
        return c[key].value if key in c else None

    # --- routing -----------------------------------------------------------
    def do_GET(self):
        path = urlparse(self.path).path
        query = parse_qs(urlparse(self.path).query)
        if path == "/":
            return self._render("index.html", title="Read", section="read",
                                answers=_list_answers())
        if path == "/static/style.css":
            css = (HERE / "static" / "style.css").read_text(encoding="utf-8")
            return self._send(200, css, ctype="text/css; charset=utf-8")
        if path.startswith("/answers/"):
            return self._get_answer(unquote(path[len("/answers/"):]))
        if path == "/ask":
            return self._render("ask.html", title="Ask", section="ask")
        if path == "/chat":
            return self._get_chat(new="new" in query)
        return self._send(404, "<h1>404</h1>")

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/ask":
            return self._post_ask()
        if path == "/chat":
            return self._post_chat()
        return self._send(404, "<h1>404</h1>")

    # --- handlers ----------------------------------------------------------
    def _get_answer(self, name: str):
        p = _answer_path(name)
        if p is None:
            return self._send(404, "<h1>404 — no such answer</h1>")
        return self._render("answer.html", title=_title_of(p), section="read",
                            html=markdown_to_html(p.read_text(encoding="utf-8")))

    def _post_ask(self):
        form = self._form()
        question = (form.get("question") or "").strip()
        hq = "hq" in form
        if not question:
            return self._render("ask.html", title="Ask", section="ask",
                                error="Please enter a question.")
        if not _have_api_key():
            return self._render("ask.html", title="Ask", section="ask",
                                question=question, hq=hq,
                                error="ANTHROPIC_API_KEY is not set; needed to generate answers.")
        try:
            with _lock:
                text = pipeline.answer(question, high_quality=hq,
                                       retriever=_get_retriever(), client=_get_client())
                model = config.GEN_MODEL_HQ if hq else config.GEN_MODEL
                saved = pipeline.save_answer(question, text, model)
        except SystemExit as e:  # pipeline uses sys.exit for API/index errors
            return self._render("ask.html", title="Ask", section="ask",
                                question=question, hq=hq, error=str(e))
        return self._render("ask.html", title="Ask", section="ask",
                            question=question, hq=hq,
                            html=markdown_to_html(text), saved=saved)

    def _get_chat(self, new: bool):
        sid = None if new else self._cookie("sid")
        conv = _chats.get(sid) if sid else None
        cookie = None
        if sid is None:
            sid = uuid.uuid4().hex[:8]
            cookie = f"sid={sid}; Path=/; SameSite=Lax"
        return self._render("chat.html", title="Chat", section="chat",
                            sid=sid, turns=_turns(conv), cookie=cookie)

    def _post_chat(self):
        form = self._form()
        sid = (form.get("sid") or self._cookie("sid") or uuid.uuid4().hex[:8])
        question = (form.get("question") or "").strip()
        if not question:
            return self._get_chat(new=False)
        if not _have_api_key():
            return self._render("chat.html", title="Chat", section="chat", sid=sid,
                                turns=_turns(_chats.get(sid)),
                                error="ANTHROPIC_API_KEY is not set; needed to generate answers.")
        try:
            with _lock:
                conv = _chats.get(sid)
                if conv is None:
                    conv = ChatSession(high_quality=_high_quality,
                                       retriever=_get_retriever(), client=_get_client())
                    _chats[sid] = conv
                conv.ask(question)
                conv.save(sid)
                conv.export_markdown(str(config.ANSWERS_DIR / f"{sid}.md"))
        except SystemExit as e:
            return self._render("chat.html", title="Chat", section="chat", sid=sid,
                                turns=_turns(_chats.get(sid)), error=str(e))
        return self._render("chat.html", title="Chat", section="chat",
                            sid=sid, turns=_turns(conv))


def _turns(conv: ChatSession | None) -> list[dict]:
    """Render a session's dialogue for the chat template."""
    if conv is None:
        return []
    out = []
    for t in conv.dialogue:
        if t["role"] == "user":
            out.append({"role": "user", "text": t["text"]})
        else:
            out.append({
                "role": "assistant",
                "html": markdown_to_html(t["text"]),
                "sources": ", ".join(t.get("sources", [])),
            })
    return out


def serve(port: int = 8000, high_quality: bool = False) -> int:
    global _high_quality
    _high_quality = high_quality
    config.ANSWERS_DIR.mkdir(parents=True, exist_ok=True)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Pāli Canon RAG web UI on {url}  (Ctrl-C to stop)")
    if not _have_api_key():
        print("  note: ANTHROPIC_API_KEY not set — Read works, Ask/Chat will error.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(serve())
