"""End-to-end query -> retrieved context -> grounded Claude answer.

Wires the Retriever to the Anthropic API using the prompt templates.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from rag import prompts  # noqa: E402
from rag.retriever import Retriever  # noqa: E402


def _sources_footer(chunks: list[dict]) -> str:
    """De-duplicated list of retrieved sutta UIDs, in first-seen order."""
    seen: list[str] = []
    for c in chunks:
        uid = c.get("sutta_uid", "?")
        if uid not in seen:
            seen.append(uid)
    return "Sources retrieved: " + ", ".join(seen)


def answer(question: str, k: int = config.TOP_K, high_quality: bool = False) -> str:
    # Fail fast before the (slow) embedding model load if we can't generate.
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set; required for `ask`.")

    chunks = Retriever().query(question, k=k)
    if not chunks:
        return "No passages were retrieved for that question; the index may be empty."

    context = prompts.format_context(chunks)

    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    model = config.GEN_MODEL_HQ if high_quality else config.GEN_MODEL
    resp = client.messages.create(
        model=model,
        max_tokens=1500,
        system=prompts.SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Question: {question}\n\n"
                f"Retrieved passages:\n\n{context}",
            }
        ],
    )
    body = "".join(block.text for block in resp.content if block.type == "text")
    return f"{body}\n\n{_sources_footer(chunks)}"
