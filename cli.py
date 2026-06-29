#!/usr/bin/env python3
"""Pāli Canon RAG — command-line entry point.

Subcommands:
    check                 verify local data layout (sc-data, dpd.db)
    ask "<question>"      retrieve + generate a grounded answer  [needs index]
    term <pali-word>      DPD term-archaeology lookup            [needs dpd.db]

The data pipeline (extract -> chunk -> embed_and_index) lives in scripts/.
"""
from __future__ import annotations

import argparse
import sys

import config


def cmd_check(_args) -> int:
    problems = config.check_data()
    if problems:
        print("Data layout problems:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("Local data OK:")
    print(f"  Pāli root:  {config.PALI_ROOT}")
    print(f"  Sujato EN:  {config.EN_SUJATO}")
    print(f"  DPD DB:     {config.DPD_DB}")
    return 0


def cmd_ask(args) -> int:
    from rag.pipeline import answer
    print(answer(args.question, high_quality=args.hq))
    return 0


def cmd_term(args) -> int:
    import subprocess
    return subprocess.call([sys.executable, "scripts/term_lookup.py", args.word])


def main() -> int:
    parser = argparse.ArgumentParser(description="Pāli Canon RAG")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="verify local data layout").set_defaults(func=cmd_check)

    p_ask = sub.add_parser("ask", help="grounded RAG answer")
    p_ask.add_argument("question")
    p_ask.add_argument("--hq", action="store_true", help="use the high-quality model")
    p_ask.set_defaults(func=cmd_ask)

    p_term = sub.add_parser("term", help="DPD term lookup")
    p_term.add_argument("word")
    p_term.set_defaults(func=cmd_term)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
