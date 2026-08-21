from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from slim_wegovy.config import load_settings
from slim_wegovy.harness import L2Harness
from slim_wegovy.patient import PatientSimulator


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Run the Slim Wegovy L2 harness.")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="Ask one question.")
    ask.add_argument("question")
    ask.add_argument("--json", action="store_true", help="Print full JSON result.")

    sim = sub.add_parser("simulate", help="Run patient simulator conversation.")
    sim.add_argument("--turns", type=int, default=3)
    sim.add_argument("--out", type=Path, default=None)

    serve = sub.add_parser("serve", help="Start the web playground.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()
    settings = load_settings()

    if args.command == "ask":
        result = L2Harness(settings).answer(args.question)
        if args.json:
            print(result.model_dump_json(indent=2))
        else:
            print(result.answer)
        return

    if args.command == "simulate":
        harness = L2Harness(settings)
        patient = PatientSimulator(settings)
        history: list[dict[str, str]] = []
        first_question_exact = None
        for idx in range(max(args.turns, 0)):
            question = patient.next_question(history)
            if idx == 0:
                first_question_exact = question
            print(f"\n[user {idx + 1}]\n{question}")
            result = harness.answer(question, history=history)
            print(f"\n[assistant {idx + 1}]\n{result.answer}")
            history.extend(
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": result.answer},
                ]
            )
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                json.dumps(
                    {"first_question_exact": first_question_exact, "messages": history},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        return

    if args.command == "serve":
        import uvicorn

        uvicorn.run("slim_wegovy.web:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
