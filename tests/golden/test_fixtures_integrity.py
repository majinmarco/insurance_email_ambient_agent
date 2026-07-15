"""Integrity guardrails for the frozen golden dataset (MAR-9).

These are *not* the eval suite (MAR-10 adds the real model metrics). They guarantee the
committed fixtures are well-formed, self-consistent, and decodable, so a bad edit or a
botched rebuild fails here instead of silently corrupting downstream metrics.
"""

from __future__ import annotations

import base64
import io

import pytest
from pdfminer.pdfpage import PDFPage

from insurance_email_agent.schemas import DocumentCategory, EmailCategory
from scripts.golden import label_studio as ls
from tests.golden.loader import GoldenCase, load_cases, load_manifest

_EMAIL_VALUES = {e.value for e in EmailCategory}
_DOC_VALUES = {d.value for d in DocumentCategory}

CASES = load_cases()
CASE_IDS = [c.case_id for c in CASES]


def test_manifest_matches_cases() -> None:
    manifest = load_manifest()
    assert manifest.n_cases == len(CASES)
    assert {e.case_id for e in manifest.cases} == set(CASE_IDS)


def test_dataset_covers_tiers_and_email_types() -> None:
    tiers = {c.tier for c in CASES}
    assert tiers == {"clean", "mixed", "messy"}
    covered = {c.email_type for c in CASES}
    assert covered == _EMAIL_VALUES, f"missing email types: {_EMAIL_VALUES - covered}"


def test_dataset_covers_boundary_and_edge_structure() -> None:
    # Multi-doc bundles are required for boundary metrics (WindowDiff/Pk/boundary-F1).
    assert any(
        any(a.n_documents > 1 for a in c.labels.attachments) for c in CASES
    ), "no multi-doc bundle in the golden set"
    formats = {f for c in CASES for a in c.labels.attachments for f in [a.render_format]}
    assert "scanned_pdf" in formats, "no scanned edge case"
    assert "html" in formats, "no HTML edge case"


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_email_type_valid(case: GoldenCase) -> None:
    assert case.labels.email.email_type in _EMAIL_VALUES
    assert case.labels.email.email_type == case.email_type


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_doc_types_valid(case: GoldenCase) -> None:
    for att in case.labels.attachments:
        for doc in att.documents:
            assert doc.doc_type in _DOC_VALUES
        # expected_segment_types = physical docs minus needs_review, order preserved
        expected = [
            d.doc_type
            for d in att.documents
            if d.doc_type != DocumentCategory.NEEDS_REVIEW.value
        ]
        assert att.expected_segment_types == expected


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_boundary_invariants(case: GoldenCase) -> None:
    for att in case.labels.attachments:
        # doc count consistency
        assert att.n_documents == len(att.documents)
        # contiguous, non-overlapping page spans starting at 0
        cursor = 0
        for doc in att.documents:
            assert doc.page_start == cursor, f"{case.case_id}/{att.filename} gap at doc {doc.index}"
            assert doc.page_end >= doc.page_start
            cursor = doc.page_end + 1
        # spans tile the whole attachment
        assert cursor == att.n_pages
        # boundary offsets are exactly the internal cut points
        assert att.boundary_page_offsets == [d.page_end + 1 for d in att.documents[:-1]]
        for off in att.boundary_page_offsets:
            assert 0 < off < att.n_pages


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_input_payload_shape_and_decodes(case: GoldenCase) -> None:
    email = case.email
    assert email["id"] == case.case_id  # deterministic, stable id
    for key in ("subject", "body", "sender", "recipient", "date_received", "attachments"):
        assert key in email
    # one payload attachment per labeled attachment, same filenames/order
    payload_atts = email["attachments"]
    assert len(payload_atts) == len(case.labels.attachments)
    for p_att, l_att in zip(payload_atts, case.labels.attachments):
        assert p_att["filename"] == l_att.filename
        raw = base64.b64decode(p_att["content"])
        assert raw, "empty attachment content"
        if l_att.render_format in ("pdf", "scanned_pdf"):
            n_pages = len(list(PDFPage.get_pages(io.BytesIO(raw))))
            assert n_pages == l_att.n_pages, (
                f"{case.case_id}/{l_att.filename}: frozen PDF has {n_pages} pages, "
                f"label says {l_att.n_pages}"
            )


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_extraction_labels_present_per_type(case: GoldenCase) -> None:
    for att in case.labels.attachments:
        for doc in att.documents:
            if doc.doc_type == DocumentCategory.NEEDS_REVIEW.value:
                assert doc.extraction is None  # graph emits no extraction for needs_review
            else:
                assert doc.extraction is not None
                assert "named_insured" in doc.extraction


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_label_studio_roundtrip_is_lossless(case: GoldenCase) -> None:
    """Exporting a case to a Label Studio task and re-importing its (unedited) pre-
    annotations must reproduce the labels exactly — the hand-correction round-trip is
    lossless, so an unchanged review never mutates the ground truth."""
    labels = case.labels.model_dump()
    task = ls.case_to_task(case.input, labels)
    assert ls.apply_annotations(task, labels) == labels


def test_label_studio_config_and_export_valid() -> None:
    config = ls.labeling_config()
    assert "<Repeater" in config and "email_type" in config
    assert "<HyperText" in config and "doc_pdf_{{idx}}" in config  # per-document PDF viewer
    for choice in ("new_submission", "renewal", "certificate of insurance"):
        assert choice in config
    # every task is JSON-serializable and carries derived pre-annotations
    import json

    for case in CASES:
        task = ls.case_to_task(case.input, case.labels.model_dump())
        json.dumps(task)
        assert task["predictions"][0]["result"], case.case_id


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_label_studio_pdf_embed_present_and_wellformed(case: GoldenCase) -> None:
    """Every task document carries a rendered-PDF embed so a reviewer can eyeball the source
    against the labels. localfiles URLs must be well-formed and point at the dumped path."""
    for mode in ("localfiles", "embed"):
        task = ls.case_to_task(case.input, case.labels.model_dump(), pdf_mode=mode)
        docs = task["data"]["documents"]
        assert len(docs) == sum(len(a.documents) for a in case.labels.attachments)
        for datum in docs:
            embed = datum.get("pdf", "")
            assert embed.startswith("<embed src=") and "width=" in embed
            if mode == "localfiles":
                assert f"/data/local-files/?d={ls.LOCALFILES_PREFIX}/{case.case_id}/" in embed
            else:
                assert "src=\"data:" in embed and ";base64," in embed
    # 'none' mode omits the viewer entirely
    task_none = ls.case_to_task(case.input, case.labels.model_dump(), pdf_mode="none")
    assert all("pdf" not in d for d in task_none["data"]["documents"])
