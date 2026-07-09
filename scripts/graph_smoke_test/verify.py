"""Lenient expected-vs-actual checks. Mismatches are findings, never crashes."""

from __future__ import annotations

from dataclasses import dataclass

from .taxonomy import Skeleton


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def _actual_segment_types(result: dict) -> list[str]:
    out = []
    for att in (result or {}).get("document_data") or []:
        for seg in att.get("segments") or []:
            cat = seg.get("category")
            if cat is not None:
                out.append(cat)
    return out


def verify(skeleton: Skeleton, result: dict | None) -> list[Check]:
    result = result or {}
    checks: list[Check] = []

    # 1. Email classification matches the intended type.
    actual_email = (result.get("classification") or {}).get("category")
    expected_email = skeleton.email_type.value
    checks.append(
        Check(
            "email_type",
            actual_email == expected_email,
            f"expected={expected_email} actual={actual_email}",
        )
    )

    # 2. One document_data entry per attachment sent.
    n_actual = len((result.get("document_data") or []))
    n_expected = len(skeleton.attachments)
    checks.append(
        Check(
            "attachment_count",
            n_actual == n_expected,
            f"expected={n_expected} actual={n_actual}",
        )
    )

    # 3. Each non-needs_review doc type appears in at least one segment (subset,
    #    not exact multiset — multi-doc PDFs often collapse to one segment).
    actual = set(_actual_segment_types(result))
    expected = {d.value for d in skeleton.expected_segment_types}
    missing = sorted(expected - actual)
    checks.append(
        Check(
            "doc_types_present",
            not missing,
            f"missing={missing or 'none'} actual={sorted(actual) or 'none'}",
        )
    )
    return checks
