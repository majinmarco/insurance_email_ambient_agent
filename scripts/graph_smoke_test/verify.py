"""Lenient expected-vs-actual checks. Mismatches are findings, never crashes.

With the realism overlay, some mismatches are *expected* under the tier: when a
multi-doc bundle's ``# TITLE`` boundary is dropped (``messy``), the graph — which
splits only on ``#`` H1 lines — merges the whole PDF into one segment, so some doc
types legitimately disappear. Those surface as ``graph_limitation`` findings (honest
signal about what the graph can't do on realistic input), kept separate from the
``correctness`` checks that gate pass/fail. ``clean`` has no overlay, so it behaves
exactly as before.
"""

from __future__ import annotations

from dataclasses import dataclass

from insurance_email_agent.schemas import DocumentCategory

from .taxonomy import Skeleton

CORRECTNESS = "correctness"
GRAPH_LIMITATION = "graph_limitation"
INFO = "info"


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    kind: str = CORRECTNESS


def _actual_segment_types(result: dict) -> list[str]:
    out = []
    for att in (result or {}).get("document_data") or []:
        for seg in att.get("segments") or []:
            cat = seg.get("category")
            if cat is not None:
                out.append(cat)
    return out


def verify(skeleton: Skeleton, result: dict | None, realism=None) -> tuple[list[Check], list[Check]]:
    """Return ``(checks, findings)``.

    ``checks`` gate pass/fail (a run passes iff every check is ok). ``findings`` are
    informational / graph-limitation observations that are surfaced but never fail a
    run — they exist so realism the graph legitimately can't handle doesn't read as a
    false failure.
    """
    result = result or {}
    expected = skeleton.noise.expected if skeleton.noise else None
    checks: list[Check] = []
    findings: list[Check] = []

    # The graph's needs_review interrupt is currently disabled; if it is ever
    # re-enabled, note the paused run rather than crashing on partial state.
    if result.get("__interrupt__"):
        findings.append(Check("interrupt", True, "run paused awaiting human review", kind=INFO))

    # 1. Email classification matches the intended type — unless the email was made
    #    intentionally ambiguous (body-only / body-vs-attachment mismatch), where any
    #    classification is defensible, so we report rather than fail.
    actual_email = (result.get("classification") or {}).get("category")
    expected_email = skeleton.email_type.value
    if expected is None or expected.email_type_hard:
        checks.append(
            Check("email_type", actual_email == expected_email, f"expected={expected_email} actual={actual_email}")
        )
    else:
        findings.append(
            Check("email_type", True, f"expected~{expected_email} actual={actual_email} (intentionally ambiguous)", kind=INFO)
        )

    # 2. One document_data entry per attachment sent. Collapse merges *segments*
    #    within an attachment, not the per-attachment entries, so this stays hard.
    n_actual = len(result.get("document_data") or [])
    n_expected = len(skeleton.attachments)
    checks.append(Check("attachment_count", n_actual == n_expected, f"expected={n_expected} actual={n_actual}"))

    # 3. Each non-needs_review doc type appears in a segment — hard for attachments
    #    whose '#' boundary was kept; a finding for collapsed bundles (dropped '#'),
    #    which legitimately merge into one segment and lose the other types.
    actual = set(_actual_segment_types(result))
    collapse_flags = expected.per_attachment_collapse if expected else []
    scanned_flags = expected.per_attachment_scanned if expected else []
    expected_hard: set[str] = set()
    collapse_types: set[str] = set()
    scanned_types: set[str] = set()
    for a_i, att in enumerate(skeleton.attachments):
        types = {d.value for d in att.docs if d != DocumentCategory.NEEDS_REVIEW}
        if a_i < len(scanned_flags) and scanned_flags[a_i]:
            scanned_types |= types              # no text layer → type can't be recovered
        elif a_i < len(collapse_flags) and collapse_flags[a_i]:
            collapse_types |= types             # merged into one segment
        else:
            expected_hard |= types

    missing_hard = sorted(expected_hard - actual)
    checks.append(
        Check("doc_types_present", not missing_hard, f"missing={missing_hard or 'none'} actual={sorted(actual) or 'none'}")
    )

    if any(collapse_flags):
        n = sum(1 for f in collapse_flags if f)
        merged = sorted((collapse_types - actual) - expected_hard)
        findings.append(Check(
            "multi_doc_collapse", not merged,
            f"{n} multi-doc bundle(s) lost their '#' boundary; types merged away={merged or 'none'}",
            kind=GRAPH_LIMITATION))

    if any(scanned_flags):
        n = sum(1 for f in scanned_flags if f)
        missed = sorted((scanned_types - actual) - expected_hard)
        findings.append(Check(
            "scanned_no_text", not missed,
            f"{n} scanned attachment(s) have no text layer (graph has no OCR); types missed={missed or 'none'}",
            kind=GRAPH_LIMITATION))

    return checks, findings
