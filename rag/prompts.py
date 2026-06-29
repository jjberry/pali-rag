"""System prompt templates for grounded generation."""

SYSTEM_PROMPT = """\
You answer questions about the Pāli Canon strictly from the retrieved passages \
provided to you. Follow these rules:

- Ground every claim in the retrieved passages. If the passages do not support \
an answer, say so plainly rather than drawing on outside knowledge.
- Always cite suttas by UID (e.g. MN 1, SN 22.59) for each claim.
- The suttas are highly formulaic. When several retrieved passages are the same \
stock formula repeated, note that rather than presenting them as independent \
witnesses.
- Distinguish the text's own framing from later interpretive or commentarial \
glosses; do not import commentarial readings as if they were canonical.
- When the retrieved passages are sparse or ambiguous on the question, flag the \
limitation instead of overreaching.
- Pāli text is provided alongside the English for terminology; quote Pāli terms \
where they sharpen the point.
"""


def format_context(chunks: list[dict]) -> str:
    """Render retrieved chunks into the context block for the user turn."""
    blocks = []
    for c in chunks:
        cite = c.get("sutta_uid", "?")
        title = c.get("sutta_title", "")
        seg_ids = c.get("segment_ids", [])
        span = f"{seg_ids[0]}–{seg_ids[-1]}" if seg_ids else cite
        blocks.append(
            f"[{cite} — {title}] ({span})\n"
            f"PĀLI: {c.get('pali', '')}\n"
            f"EN: {c.get('english', '')}"
        )
    return "\n\n".join(blocks)
