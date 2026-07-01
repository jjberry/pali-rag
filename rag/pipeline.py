"""End-to-end query -> retrieved context -> grounded Claude answer.

Wires the Retriever to the Anthropic API using the prompt templates. The
shared helpers here (`complete`, `source_uids`, `retrieve`, `user_turn`) are
also used by the multi-turn `rag.chat` session.
"""
from __future__ import annotations

import os
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from rag import prompts  # noqa: E402
from rag.retriever import Retriever  # noqa: E402
from scripts import term_lookup  # noqa: E402


def require_api_key() -> None:
    """Fail fast before the (slow) embedding model load if we can't generate."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set; required for `ask`/`chat`.")


def complete(client, model: str, system: str, messages: list[dict],
             max_tokens: int = 4096, warn_truncation: bool = True) -> str:
    """One Anthropic call, with the project's clean error handling."""
    import anthropic

    try:
        resp = client.messages.create(
            model=model, max_tokens=max_tokens, system=system, messages=messages
        )
    except anthropic.AuthenticationError:
        sys.exit("Anthropic auth failed (401): the ANTHROPIC_API_KEY is invalid or revoked.")
    except anthropic.APIStatusError as e:
        sys.exit(f"Anthropic API error ({e.status_code}): {e.message}")
    except anthropic.APIConnectionError as e:
        sys.exit(f"Could not reach the Anthropic API: {e}")
    if warn_truncation and resp.stop_reason == "max_tokens":
        print(f"[warning] answer truncated at max_tokens={max_tokens}; "
              "raise the cap for a complete response.", file=sys.stderr)
    return "".join(block.text for block in resp.content if block.type == "text")


def source_uids(chunks: list[dict]) -> list[str]:
    """De-duplicated retrieved sutta UIDs, in first-seen order."""
    seen: list[str] = []
    for c in chunks:
        uid = c.get("sutta_uid", "?")
        if uid not in seen:
            seen.append(uid)
    return seen


def _sources_footer(chunks: list[dict]) -> str:
    return "Sources retrieved: " + ", ".join(source_uids(chunks))


def retrieve(retriever: Retriever, query: str, k: int = config.TOP_K,
             announce: bool = True) -> list[dict]:
    """DPD-expand a query (bridging the English-only index) and retrieve."""
    search_text, glossed = term_lookup.expand_query(query)
    if glossed and announce:
        note = ", ".join(f"{t} → {'; '.join(g)}" for t, g in glossed.items())
        print(f"[query expanded via DPD] {note}", file=sys.stderr)
    return retriever.query(search_text, k=k)


def user_turn(question: str, chunks: list[dict]) -> dict:
    """Build the user message that carries the retrieved passages."""
    context = prompts.format_context(chunks)
    return {
        "role": "user",
        "content": f"Question: {question}\n\nRetrieved passages:\n\n{context}",
    }


def answer(question: str, k: int = config.TOP_K, high_quality: bool = False,
           retriever: Retriever | None = None, client=None) -> str:
    """Grounded one-shot answer. `retriever`/`client` may be injected so a
    long-lived host (e.g. the web server) reuses one embed model + API client
    across requests instead of reloading per call."""
    require_api_key()

    chunks = retrieve(retriever or Retriever(), question, k=k)
    if not chunks:
        return "No passages were retrieved for that question; the index may be empty."

    if client is None:
        import anthropic
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    model = config.GEN_MODEL_HQ if high_quality else config.GEN_MODEL
    body = complete(client, model, prompts.SYSTEM_PROMPT, [user_turn(question, chunks)])
    return f"{body}\n\n{_sources_footer(chunks)}"


def _slug(text: str, max_len: int = 60) -> str:
    """A filesystem-safe stub of the question for auto-named answer files."""
    stub = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return stub[:max_len].rstrip("-") or "answer"


def save_answer(question: str, answer_text: str, model: str,
                path: str | Path | None = None) -> Path:
    """Write a rendered Markdown record of an answer for later re-reading.

    With no `path`, auto-names a timestamped file under `config.ANSWERS_DIR`.
    Returns the path written.
    """
    now = datetime.now()
    if path is None:
        config.ANSWERS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = now.strftime("%Y-%m-%d-%H%M%S")
        path = config.ANSWERS_DIR / f"{stamp}-{_slug(question)}.md"
    else:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

    doc = (
        f"# {question}\n\n"
        f"*{now.strftime('%Y-%m-%d %H:%M:%S')} — {model}*\n\n"
        f"{answer_text}\n"
    )
    path.write_text(doc, encoding="utf-8")
    return path
