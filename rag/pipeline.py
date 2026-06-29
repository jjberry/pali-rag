"""End-to-end query -> retrieved context -> grounded Claude answer.

Stub: wires the Retriever to the Anthropic API using the prompt templates.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from rag import prompts  # noqa: E402
from rag.retriever import Retriever  # noqa: E402


def answer(question: str, k: int = config.TOP_K, high_quality: bool = False) -> str:
    chunks = Retriever().query(question, k=k)
    context = prompts.format_context(chunks)

    # TODO:
    #   import anthropic
    #   client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY
    #   model = config.GEN_MODEL_HQ if high_quality else config.GEN_MODEL
    #   resp = client.messages.create(
    #       model=model, max_tokens=1500, system=prompts.SYSTEM_PROMPT,
    #       messages=[{"role": "user",
    #                  "content": f"Question: {question}\n\n"
    #                             f"Retrieved passages:\n\n{context}"}])
    #   return resp.content[0].text
    raise NotImplementedError("pipeline.answer is a stub — wire up Anthropic API.")
