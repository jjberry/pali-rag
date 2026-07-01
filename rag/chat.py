"""Multi-turn, history-aware RAG chat over the Pāli Canon.

Each turn: condense the (possibly referential) follow-up into a standalone
search query using the conversation so far, retrieve fresh passages for it
(DPD-expanded + title-fused, same as `ask`), then answer with the full message
history so the conversation stays coherent. Sessions can be saved and resumed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from rag import pipeline, prompts  # noqa: E402
from rag.retriever import Retriever  # noqa: E402

EXIT_WORDS = {"exit", "quit", ":q"}


class ChatSession:
    def __init__(self, high_quality: bool = False, retriever=None, client=None) -> None:
        self.model = config.GEN_MODEL_HQ if high_quality else config.GEN_MODEL
        self.high_quality = high_quality
        if client is None:
            import anthropic
            client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.client = client
        self.retriever = retriever or Retriever()
        self.messages: list[dict] = []          # API history (passages inline)
        self.dialogue: list[dict] = []           # clean Q/A for condense + display

    def _condense(self, question: str) -> str:
        """Rewrite a follow-up into a standalone search query. The first turn is
        already standalone, so this is only called from turn 2 on."""
        history = "\n".join(
            f"{t['role'].upper()}: {t['text']}" for t in self.dialogue[-6:]
        )
        msg = f"Conversation so far:\n{history}\n\nFollow-up: {question}"
        candidate = pipeline.complete(
            self.client, config.GEN_MODEL, prompts.CONDENSE_SYSTEM,
            [{"role": "user", "content": msg}], max_tokens=120,
            warn_truncation=False,
        ).strip()
        # The model occasionally answers instead of rewriting; a real query is a
        # single short line. If it misfired, fall back to the raw follow-up.
        if not candidate or "\n" in candidate or len(candidate) > 200:
            return question
        return candidate

    def ask(self, question: str) -> dict:
        search_query = question if not self.dialogue else self._condense(question)
        chunks = pipeline.retrieve(self.retriever, search_query)

        self.messages.append(pipeline.user_turn(question, chunks))
        body = pipeline.complete(
            self.client, self.model, prompts.SYSTEM_PROMPT, self.messages
        )
        self.messages.append({"role": "assistant", "content": body})

        sources = pipeline.source_uids(chunks)
        self.dialogue.append({"role": "user", "text": question})
        self.dialogue.append({"role": "assistant", "text": body, "sources": sources})
        return {
            "answer": body,
            "sources": sources,
            "search_query": search_query,
        }

    # --- persistence ------------------------------------------------------
    def save(self, name: str) -> None:
        config.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        path = config.SESSIONS_DIR / f"{name}.json"
        path.write_text(json.dumps({
            "high_quality": self.high_quality,
            "messages": self.messages,
            "dialogue": self.dialogue,
        }, ensure_ascii=False, indent=2))

    def export_markdown(self, path: str | None = None) -> Path:
        """Write the conversation as a readable Markdown transcript for later
        re-reading (distinct from the resumable JSON session). Auto-names under
        `config.ANSWERS_DIR` from the first question when no path is given."""
        from datetime import datetime

        now = datetime.now()
        first_q = next((t["text"] for t in self.dialogue if t["role"] == "user"),
                       "chat")
        if path is None:
            config.ANSWERS_DIR.mkdir(parents=True, exist_ok=True)
            stamp = now.strftime("%Y-%m-%d-%H%M%S")
            out = config.ANSWERS_DIR / f"{stamp}-chat-{pipeline._slug(first_q)}.md"
        else:
            out = Path(path)
            out.parent.mkdir(parents=True, exist_ok=True)

        lines = [f"# Pāli Canon chat — {now.strftime('%Y-%m-%d %H:%M:%S')} "
                 f"({self.model})\n"]
        for turn in self.dialogue:
            if turn["role"] == "user":
                lines.append(f"\n## {turn['text']}\n")
            else:
                lines.append(f"{turn['text']}\n")
        out.write_text("\n".join(lines), encoding="utf-8")
        return out

    @classmethod
    def load(cls, name: str, retriever=None, client=None) -> "ChatSession":
        path = config.SESSIONS_DIR / f"{name}.json"
        if not path.exists():
            sys.exit(f"no saved session '{name}' at {path}")
        data = json.loads(path.read_text())
        conv = cls(high_quality=data.get("high_quality", False),
                   retriever=retriever, client=client)
        conv.messages = data.get("messages", [])
        conv.dialogue = data.get("dialogue", [])
        return conv


def run_repl(session: str | None = None, resume: str | None = None,
             high_quality: bool = False) -> int:
    pipeline.require_api_key()

    # The REPL needs a real terminal; under a non-interactive stdin (e.g. a
    # harness `!` shell or /dev/null) input() returns EOF immediately. Check
    # before the (slow) model load so it fails fast.
    if not sys.stdin.isatty():
        print(
            "chat needs an interactive terminal (stdin is not a TTY). Run it in "
            "a real shell, or use `cli.py ask \"<question>\"` for a one-shot answer.",
            file=sys.stderr,
        )
        return 1

    if resume:
        conv = ChatSession.load(resume)
        name = resume
        print(f"Resumed session '{resume}' — {len(conv.dialogue)//2} prior turn(s):")
        for turn in conv.dialogue:
            if turn["role"] == "user":
                print(f"  > {turn['text']}")
    else:
        conv = ChatSession(high_quality=high_quality)
        name = session

    where = f" (saving to '{name}')" if name else " (not saved — use --session NAME)"
    print(f"\nPāli Canon chat{where}. Type a question; 'exit' to quit.")
    print("Commands: ':save [path]' export a Markdown transcript; 'exit' to quit.")

    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in EXIT_WORDS:
            break
        if question.split(" ", 1)[0] == ":save":
            if not conv.dialogue:
                print("Nothing to save yet — ask something first.", file=sys.stderr)
                continue
            arg = question[len(":save"):].strip()
            out = conv.export_markdown(arg or None)
            print(f"[transcript saved to {out}]", file=sys.stderr)
            continue

        result = conv.ask(question)
        if result["search_query"] != question:
            print(f"[searched: {result['search_query']}]", file=sys.stderr)
        print(f"\n{result['answer']}")
        print(f"\nSources retrieved: {', '.join(result['sources'])}")
        if name:
            conv.save(name)

    if name and conv.dialogue:  # don't write an empty session file
        conv.save(name)
        # Also leave a human-readable transcript, named by session so a
        # resume-and-exit overwrites it in place rather than piling up.
        transcript = conv.export_markdown(str(config.ANSWERS_DIR / f"{name}.md"))
        print(f"Saved session '{name}' (transcript: {transcript}). "
              f"Resume with: cli.py chat --resume {name}")
    return 0
