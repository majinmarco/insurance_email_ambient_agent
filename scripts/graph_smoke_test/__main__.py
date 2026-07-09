"""CLI entry point.

Examples:
    uv run python -m scripts.graph_smoke_test --n 1 --attachments off
    uv run python -m scripts.graph_smoke_test --n 5 --seed 42
    uv run python -m scripts.graph_smoke_test --email-type renewal --attachments on --verbose
    uv run python -m scripts.graph_smoke_test --no-llm --keep-pdfs ./out
"""

from __future__ import annotations

import argparse
import os
import random
import sys

from insurance_email_agent.schemas import EmailCategory

from .client_run import make_client, run_graph, studio_url
from .payload import build_email_payload, decode_attachments
from .scenario import generate_scenario
from .taxonomy import build_skeleton
from .verify import verify


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="scripts.graph_smoke_test", description=__doc__)
    p.add_argument("--n", "--runs", dest="n", type=int, default=1, help="number of scenarios")
    p.add_argument("--seed", type=int, default=None, help="seed structural randomness")
    p.add_argument("--url", default="http://127.0.0.1:2024", help="langgraph dev URL")
    p.add_argument("--assistant-id", default="insurance_email_agent")
    p.add_argument("--attachments", choices=["on", "off", "auto"], default="auto")
    p.add_argument(
        "--email-type",
        choices=[c.value for c in EmailCategory],
        default=None,
        help="force the email type instead of random",
    )
    p.add_argument("--max-docs", type=int, default=2, help="max docs bundled per attachment")
    p.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    p.add_argument("--model", default=None, help="override generation model")
    p.add_argument("--no-llm", action="store_true", help="use deterministic content (no API call)")
    p.add_argument("--verbose", action="store_true", help="stream node-by-node updates")
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--no-smoke", action="store_true", help="skip the 0-attachment model gate")
    p.add_argument("--keep-pdfs", metavar="DIR", default=None, help="also write generated PDFs")
    return p.parse_args(argv)


def _print_scenario(skeleton, scenario) -> None:
    print(f"  email type : {skeleton.email_type.value}")
    plans = [[d.value for d in a.docs] for a in skeleton.attachments]
    print(f"  attachments: {plans or '(none)'}")
    print(f"  subject    : {scenario.email.subject}")


def _print_result(result: dict | None) -> None:
    result = result or {}
    cls = result.get("classification") or {}
    print(f"  -> classification: {cls.get('category')}  ({cls.get('rationale', '')[:100]})")
    ext = result.get("email_extraction") or {}
    if ext:
        print(f"  -> email_extraction: named_insured={ext.get('named_insured')} "
              f"policy={ext.get('policy_number')} changes={str(ext.get('requested_changes'))[:60]}")
    for att in result.get("document_data") or []:
        segs = att.get("segments") or []
        print(f"  -> {att.get('filename')}: {len(segs)} segment(s)")
        for seg in segs:
            extr = seg.get("extraction")
            extr_type = extr.get("doc_type") if isinstance(extr, dict) else None
            print(f"       [{seg.get('category')}] pages={seg.get('page_indices')} "
                  f"extraction={'yes' if extr else 'none'}{f' ({extr_type})' if extr_type else ''}")


def _write_pdfs(directory: str, run_idx: int, payload: dict) -> None:
    os.makedirs(directory, exist_ok=True)
    for fn, data in decode_attachments(payload):
        path = os.path.join(directory, f"run{run_idx}_{fn}")
        with open(path, "wb") as fh:
            fh.write(data)
        print(f"  wrote {path} ({len(data)} bytes)")


def smoke_gate(client, args) -> bool:
    """One deterministic 0-attachment run to prove the graph + models resolve."""
    print("== smoke gate: 0-attachment run ==")
    sk = build_skeleton(random.Random(0), email_type=EmailCategory.RENEWAL, attachments_mode="off")
    sc = generate_scenario(sk, use_llm=False)
    payload = build_email_payload(sk, sc)
    try:
        tid, result, err = run_graph(
            client, args.assistant_id, payload, {"smoke_test": True, "phase": "gate"},
            verbose=args.verbose,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  ! could not reach the dev server at {args.url}: {exc!r}")
        print("    Start it with:  uv run langgraph dev")
        return False
    if err:
        print(f"  ! gate run failed: {err}")
        print("    Most likely the graph's model IDs (gpt-5.4-*) don't resolve on your")
        print("    OPENAI_API_KEY. Check src/insurance_email_agent/graph.py lines 58/75/82.")
        return False
    print(f"  ok — classification={result.get('classification', {}).get('category')}")
    print(f"  studio: {studio_url(args.url, tid)}")
    return True


def _load_env() -> None:
    """Load the repo ``.env`` so the generation LLM sees OPENAI_API_KEY, exactly
    like ``langgraph dev`` does for the server (langgraph.json ``env: ./.env``)."""
    try:
        from dotenv import load_dotenv

        load_dotenv()  # searches cwd upward for .env
    except Exception:  # noqa: BLE001 — dotenv optional; env may be set another way
        pass


def main(argv=None) -> int:
    args = parse_args(argv)
    _load_env()
    rng = random.Random(args.seed)
    client = make_client(args.url)

    if not args.no_smoke:
        if not smoke_gate(client, args):
            return 1
        print()

    forced_type = EmailCategory(args.email_type) if args.email_type else None
    passed = 0
    for i in range(1, args.n + 1):
        print(f"== run {i}/{args.n} ==")
        skeleton = build_skeleton(
            rng,
            email_type=forced_type,
            attachments_mode=args.attachments,
            max_docs_per_attachment=args.max_docs,
        )
        scenario = generate_scenario(
            skeleton, provider=args.provider, model=args.model, use_llm=not args.no_llm
        )
        _print_scenario(skeleton, scenario)
        payload = build_email_payload(skeleton, scenario)
        if args.keep_pdfs:
            _write_pdfs(args.keep_pdfs, i, payload)

        metadata = {
            "smoke_test": True,
            "expected_email_type": skeleton.email_type.value,
            "expected_doc_types": [d.value for d in skeleton.flat_doc_types],
        }
        try:
            tid, result, err = run_graph(
                client, args.assistant_id, payload, metadata, verbose=args.verbose
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  ! run errored: {exc!r}")
            continue

        print(f"  studio: {studio_url(args.url, tid)}")
        if err:
            print(f"  ! run failed: {err}")
            continue
        _print_result(result)

        if not args.no_verify:
            checks = verify(skeleton, result)
            for c in checks:
                print(f"  [{'PASS' if c.ok else 'FAIL'}] {c.name}: {c.detail}")
            if all(c.ok for c in checks):
                passed += 1
        print()

    if not args.no_verify:
        print(f"== {passed}/{args.n} runs passed all checks ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
