"""Optional persistence of per-run smoke results.

The harness previously wrote nothing to disk (stdout + a Studio URL only), so there
was no way to measure how realism affects the pipeline. With ``--results-dir`` set,
each run is appended to ``runs.jsonl`` (skeleton, the full seeded noise manifest,
checks, findings, pass/fail) and an aggregate ``summary.json`` is written — pass-rate
sliced by realism tier, correctness failures by check, and graph-limitation counts by
type (e.g. how many multi-doc bundles collapsed).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass


@dataclass
class RunRecord:
    run_index: int
    tier: str
    effective_seed: int
    run_label: str | None
    email_type: str
    attachment_doc_types: list          # list[list[str]] — doc types per attachment
    noise_manifest: dict | None         # asdict(NoisePlan) — exactly what was injected
    checks: list                        # [{name, ok, detail, kind}] — gate pass/fail
    findings: list                      # [{name, ok, detail, kind}] — info / graph_limitation
    interrupted: bool
    thread_id: str | None
    studio_url: str | None
    passed: bool
    error: str | None = None


def build_record(*, run_index, realism, skeleton, thread_id, studio_url, result,
                 checks, findings, error, passed, run_label) -> RunRecord:
    result = result or {}
    return RunRecord(
        run_index=run_index,
        tier=realism.tier,
        effective_seed=realism.seed,
        run_label=run_label,
        email_type=skeleton.email_type.value,
        attachment_doc_types=[[d.value for d in a.docs] for a in skeleton.attachments],
        noise_manifest=asdict(skeleton.noise) if skeleton.noise else None,
        checks=[asdict(c) for c in (checks or [])],
        findings=[asdict(f) for f in (findings or [])],
        interrupted=bool(result.get("__interrupt__")),
        thread_id=thread_id,
        studio_url=studio_url,
        passed=bool(passed),
        error=error,
    )


def append(directory: str, record: RunRecord) -> None:
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "runs.jsonl"), "a") as fh:
        fh.write(json.dumps(asdict(record), default=str) + "\n")


def write_summary(directory: str, records: list[RunRecord]) -> dict:
    os.makedirs(directory, exist_ok=True)
    by_tier: dict[str, dict] = {}
    check_fail: dict[str, int] = {}
    limitations: dict[str, int] = {}
    errored = 0
    for r in records:
        t = by_tier.setdefault(r.tier, {"runs": 0, "passed": 0})
        t["runs"] += 1
        t["passed"] += 1 if r.passed else 0
        if r.error:
            errored += 1
        for c in r.checks:
            if not c.get("ok"):
                check_fail[c["name"]] = check_fail.get(c["name"], 0) + 1
        for f in r.findings:
            if f.get("kind") == "graph_limitation":
                limitations[f["name"]] = limitations.get(f["name"], 0) + 1

    summary = {
        "total_runs": len(records),
        "errored_runs": errored,
        "by_tier": {
            t: {**v, "pass_rate": round(v["passed"] / v["runs"], 3) if v["runs"] else 0.0}
            for t, v in by_tier.items()
        },
        "correctness_failures_by_check": check_fail,
        "graph_limitations_by_type": limitations,
    }
    with open(os.path.join(directory, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary
