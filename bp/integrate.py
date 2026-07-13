"""Glue for hosting the BP secondary-sources answer beside the v1 Pāli answer.

Kept in bp/ (which already depends on the v1 project config) so that rag/ never
has to import bp — the v1 pipeline stays usable with the BP pilot absent. The
entry points that legitimately know about both worlds (cli.py, web/app.py) call
in here; ChatSession takes `secondary_section`-shaped callables by injection.
"""
from __future__ import annotations

from . import config as C
from . import pilot
from .client import BPClient

PALI_HEADER = "## From the Pāli Canon"
SECONDARY_HEADER = "## From the secondary literature (Bibliotheca Polyglotta)"


def secondary_section(question: str, *, anthropic_client, embed_model,
                      bp: BPClient, hq: bool = False) -> str:
    """The BP answer as a Markdown H2 section. **Never raises**: a live-fetch,
    parse, or API failure degrades to an italic note so it can't sink the Pāli
    answer it runs beside. Reuses the caller's shared embed model + client."""
    try:
        md = pilot.answer_markdown(
            question, anthropic_client=anthropic_client, embed_model=embed_model,
            bp=bp, texts=C.TEXTS, hq=hq,
        )
    except Exception as e:  # network/parse/API — isolate from the Pāli answer
        return f"{SECONDARY_HEADER}\n\n*Secondary-source lookup failed: {e}*"
    if not md:
        return f"{SECONDARY_HEADER}\n\n*No relevant secondary-source passages were found.*"
    return f"{SECONDARY_HEADER}\n\n{md}"


def combine(pali_text: str, secondary_md: str) -> str:
    """Assemble the two-section answer (used when secondary sources are on)."""
    return f"{PALI_HEADER}\n\n{pali_text}\n\n{secondary_md}"


def make_secondary_fn(*, anthropic_client, embed_model, bp: BPClient,
                      hq: bool = False):
    """A one-arg `question -> section` callable for injection into ChatSession,
    binding the shared singletons so each chat turn reuses them."""
    def _fn(question: str) -> str:
        return secondary_section(question, anthropic_client=anthropic_client,
                                 embed_model=embed_model, bp=bp, hq=hq)
    return _fn
