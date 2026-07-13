#!/usr/bin/env python3
"""Pāli Canon RAG — command-line entry point.

Subcommands:
    check                 verify local data layout (sc-data, dpd.db)
    ask "<question>"      retrieve + generate a grounded answer  [needs index]
    chat                  interactive multi-turn grounded conversation
    web                   browser UI: read saved answers, ask, chat
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
    from rag.pipeline import answer, require_api_key, save_answer
    if not args.secondary:
        text = answer(args.question, high_quality=args.hq)
    else:
        # Run the Pāli and secondary (Bibliotheca Polyglotta) pipelines
        # concurrently, sharing one embed model + client, so BP's rate-limited
        # network work overlaps the Pāli answer rather than adding to it.
        import concurrent.futures

        import anthropic

        from bp import integrate as bp_integrate
        from bp.client import BPClient
        from rag.retriever import Retriever

        require_api_key()
        retriever, client, bp = Retriever(), anthropic.Anthropic(), BPClient()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            f_pali = ex.submit(answer, args.question, high_quality=args.hq,
                               retriever=retriever, client=client)
            f_bp = ex.submit(bp_integrate.secondary_section, args.question,
                             anthropic_client=client, embed_model=retriever.model,
                             bp=bp, hq=args.hq)
            text = bp_integrate.combine(f_pali.result(), f_bp.result())
    print(text)
    if args.save is not None:
        # --save with no value -> auto-name; --save PATH -> that path.
        path = save_answer(args.question, text,
                           config.GEN_MODEL_HQ if args.hq else config.GEN_MODEL,
                           path=args.save or None)
        print(f"\n[saved to {path}]", file=sys.stderr)
    return 0


def cmd_chat(args) -> int:
    from rag.chat import run_repl
    return run_repl(session=args.session, resume=args.resume,
                    high_quality=args.hq, secondary=args.secondary)


def cmd_web(args) -> int:
    from web.app import serve
    return serve(port=args.port, high_quality=args.hq)


def cmd_term(args) -> int:
    import subprocess
    cmd = [sys.executable, "scripts/term_lookup.py", args.word]
    if args.compounds:
        cmd.append("--compounds")
    return subprocess.call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description="Pāli Canon RAG")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="verify local data layout").set_defaults(func=cmd_check)

    p_ask = sub.add_parser("ask", help="grounded RAG answer")
    p_ask.add_argument("question")
    p_ask.add_argument("--hq", action="store_true", help="use the high-quality model")
    p_ask.add_argument("--save", nargs="?", const="", default=None, metavar="PATH",
                       help="save the answer as Markdown; PATH optional "
                            "(default: data/answers/<timestamp>-<question>.md)")
    p_ask.add_argument("--secondary", action="store_true",
                       help="also answer from secondary literature "
                            "(Bibliotheca Polyglotta); slower, needs network")
    p_ask.set_defaults(func=cmd_ask)

    p_chat = sub.add_parser("chat", help="interactive multi-turn RAG conversation")
    p_chat.add_argument("--hq", action="store_true", help="use the high-quality model")
    p_chat.add_argument("--session", help="name this conversation (saved for --resume)")
    p_chat.add_argument("--resume", help="resume a saved conversation by name")
    p_chat.add_argument("--secondary", action="store_true",
                        help="also answer from secondary literature "
                             "(Bibliotheca Polyglotta); slower, needs network")
    p_chat.set_defaults(func=cmd_chat)

    p_web = sub.add_parser("web", help="serve the browser UI (read/ask/chat)")
    p_web.add_argument("--port", type=int, default=8000, help="port (default 8000)")
    p_web.add_argument("--hq", action="store_true", help="use the high-quality model")
    p_web.set_defaults(func=cmd_web)

    p_term = sub.add_parser("term", help="DPD term lookup")
    p_term.add_argument("word")
    p_term.add_argument("--compounds", action="store_true",
                        help="also match sandhi-fused compounds containing the term")
    p_term.set_defaults(func=cmd_term)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
