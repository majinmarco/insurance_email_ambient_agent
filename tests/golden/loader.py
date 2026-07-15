"""Deterministic loader for the frozen golden dataset (MAR-9).

This is the entry point the eval suite (MAR-10) imports. It reads the committed
fixtures under ``tests/golden/`` — never regenerating anything — and returns, per case,
the exact frozen graph input payload plus the parsed, schema-validated ground-truth
labels.

    from tests.golden.loader import load_cases
    for case in load_cases(tier="messy"):
        result = runner.run(case.input["email"], local=True)   # drive the graph
        score(result, case.labels)                              # compare vs ground truth

The pydantic models below also *validate* the committed JSON, so a malformed or
hand-edited label file fails loudly at load time rather than silently skewing metrics.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel

GOLDEN_DIR = Path(__file__).resolve().parent
CASES_DIR = GOLDEN_DIR / "cases"
MANIFEST_PATH = GOLDEN_DIR / "manifest.json"


# --------------------------------------------------------------------------- #
# Label schema (mirrors scripts/golden/labels.py output)
# --------------------------------------------------------------------------- #
class DocumentLabel(BaseModel):
    index: int
    doc_type: str
    page_start: int
    page_end: int
    extraction: dict[str, Any] | None = None


class AttachmentLabel(BaseModel):
    filename: str
    render_format: str
    n_pages: int
    documents: list[DocumentLabel]
    boundary_page_offsets: list[int]
    n_documents: int
    expected_segment_types: list[str]
    expected_collapse: bool
    expected_scanned: bool
    expected_interrupt: bool


class EmailLabel(BaseModel):
    email_type: str
    email_type_hard: bool
    extraction: dict[str, Any]


class GoldenLabels(BaseModel):
    case_id: str
    tier: str
    email: EmailLabel
    attachments: list[AttachmentLabel]


class GoldenCase(BaseModel):
    """One frozen labeled scenario."""

    case_id: str
    tier: str
    email_type: str
    #: Frozen graph input: ``{"email": {...}}`` with base64 attachments — feed to the graph.
    input: dict[str, Any]
    #: Ground-truth labels the eval scores against.
    labels: GoldenLabels

    @property
    def email(self) -> dict[str, Any]:
        """Convenience: the inner email dict (what ``handler.runner.run`` expects)."""
        return self.input["email"]


class ManifestEntry(BaseModel):
    case_id: str
    tier: str
    seed: int
    email_type: str
    n_attachments: int
    attachment_doc_types: list[list[str]]
    has_multidoc: bool
    render_formats: list[str]
    content_source: str
    dir: str


class Manifest(BaseModel):
    schema_version: int
    builder_git_sha: str | None = None
    content_source: str
    n_cases: int
    email_categories: list[str]
    cases: list[ManifestEntry]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


@lru_cache(maxsize=1)
def load_manifest() -> Manifest:
    """Parse and validate ``manifest.json``."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"golden manifest not found at {MANIFEST_PATH}. "
            "Build it with: uv run python -m scripts.build_golden_dataset"
        )
    return Manifest.model_validate(_read_json(MANIFEST_PATH))


def load_case(case_id: str) -> GoldenCase:
    """Load one case by id (validated)."""
    case_dir = CASES_DIR / case_id
    payload = _read_json(case_dir / "input.json")
    labels = GoldenLabels.model_validate(_read_json(case_dir / "labels.json"))
    return GoldenCase(
        case_id=case_id,
        tier=labels.tier,
        email_type=labels.email.email_type,
        input=payload,
        labels=labels,
    )


def load_cases(tier: str | None = None) -> list[GoldenCase]:
    """All golden cases (optionally filtered by realism tier), in manifest order."""
    manifest = load_manifest()
    cases = [load_case(e.case_id) for e in manifest.cases]
    if tier is not None:
        cases = [c for c in cases if c.tier == tier]
    return cases
