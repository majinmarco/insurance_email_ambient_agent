"""Derive frozen ground-truth labels from a generated scenario.

The smoke generator already *knows* the truth of every scenario: the email type and
per-attachment doc-type sequence come from the structural :class:`~scripts.graph_smoke_test.taxonomy.Skeleton`
(never the LLM), and the field values come from the generated
:class:`~scripts.graph_smoke_test.scenario.Scenario`. This module turns that knowledge
into the label schema the eval suite (MAR-10) scores against:

* **email type** + whether it is a *hard* target (ambiguous emails are soft),
* **per-attachment doc-type sequence** and ``expected_segment_types`` (the graph never
  emits a segment for a ``needs_review`` doc),
* **document boundaries** at *page* granularity — derived by rendering each document
  alone and counting its pages (a bundle inserts a ``PageBreak`` between documents, so
  per-document pagination is independent), giving exact page spans + boundary offsets
  for boundary-F1 / WindowDiff / Pk,
* **key extraction fields** per document (mapped onto the graph's extraction schemas in
  ``insurance_email_agent.schemas``) and for the email body,
* **expected-limitation flags** (collapse / scanned / interrupt) reused from the noise
  plan so the eval doesn't score honest graph limits as errors.

Everything here is deterministic given the frozen scenario; the emitted labels are
committed as human-readable JSON and are the artifact a curator hand-corrects.
"""

from __future__ import annotations

import io
import re
from typing import Any

from pdfminer.pdfpage import PDFPage

from insurance_email_agent.schemas import DocumentCategory, EmailCategory
from scripts.graph_smoke_test.realism import DELIM_HASH
from scripts.graph_smoke_test.render import build_html, build_pdf, build_scanned_pdf
from scripts.graph_smoke_test.scenario import (
    CoverageLine,
    DocumentContent,
    Scenario,
    SharedFacts,
)
from scripts.graph_smoke_test.taxonomy import Skeleton

DC = DocumentCategory
EC = EmailCategory


# --------------------------------------------------------------------------- #
# Value normalizers — map generated string content onto the graph's typed fields.
# --------------------------------------------------------------------------- #
def parse_money(value: str | None) -> float | None:
    """``"$18,935.00"`` -> ``18935.0``; ``None``/unparseable -> ``None``."""
    if value is None:
        return None
    cleaned = re.sub(r"[^0-9.\-]", "", str(value))
    if cleaned in ("", "-", ".", "-."):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_date(value: str | None) -> str | None:
    """Normalize ``MM/DD/YYYY`` or ``YYYY-MM-DD`` to an ISO ``YYYY-MM-DD`` string.

    Returns ``None`` when the value is missing or not a recognizable date, so the
    label stays honest rather than inventing a date.
    """
    if not value:
        return None
    s = str(value).strip()
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m:
        y, mo, d = m.groups()
        return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})", s)
    if m:
        mo, d, y = m.groups()
        return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
    return None


def _canonical_policy(shared: SharedFacts, doc: DocumentContent) -> str | None:
    """Prefer the shared policy number; fall back to a policy number printed on the
    document's primary coverage line (certificates carry it per coverage row)."""
    if shared.policy_number:
        return shared.policy_number
    for line in (doc.coverage_lines or []):
        if line.policy_number:
            return line.policy_number
    return None


def _limits_from_coverage_lines(lines: list[CoverageLine] | None) -> dict[str, str]:
    """Flatten ACORD coverage rows into a ``{limit label: amount}`` dict, matching the
    shape of ``CertificateExtraction.limits`` / ``DeclarationsExtraction.coverages``."""
    out: dict[str, str] = {}
    for line in (lines or []):
        for lim in (line.limits or []):
            if lim.coverage and lim.limit and lim.coverage not in out:
                out[lim.coverage] = lim.limit
    return out


def _coverages_from_parts(parts: list[CoverageLine] | None) -> dict[str, str]:
    """Declarations coverage parts -> ``{coverage_type: per-part premium}``."""
    out: dict[str, str] = {}
    for part in (parts or []):
        if part.coverage_type and part.premium and part.coverage_type not in out:
            out[part.coverage_type] = part.premium
    return out


# --------------------------------------------------------------------------- #
# Per-document extraction labels (keyed onto the graph's extraction schemas).
# --------------------------------------------------------------------------- #
def document_extraction_label(
    doc_type: DocumentCategory, shared: SharedFacts, doc: DocumentContent
) -> dict[str, Any] | None:
    """Ground-truth extraction fields for one document, or ``None`` for needs_review
    (the graph emits no extraction for it)."""
    named = shared.named_insured
    if doc_type == DC.CERTIFICATE:
        primary = (doc.coverage_lines or [None])[0]
        eff = parse_date(getattr(primary, "effective_date", None)) or parse_date(shared.effective_date)
        exp = parse_date(getattr(primary, "expiration_date", None)) or parse_date(shared.expiration_date)
        return {
            "named_insured": named,
            "policy_number": _canonical_policy(shared, doc),
            "certificate_holder": doc.certificate_holder,
            "effective_date": eff,
            "expiration_date": exp,
            "limits": _limits_from_coverage_lines(doc.coverage_lines),
        }
    if doc_type == DC.INVOICE:
        return {
            "named_insured": named,
            "policy_number": _canonical_policy(shared, doc),
            "amount_due": parse_money(doc.amount_due),
            "billing_period": doc.billing_period,
            "policy_reference": _canonical_policy(shared, doc),
            "due_date": parse_date(doc.due_date),
        }
    if doc_type == DC.DECLARATIONS:
        return {
            "named_insured": named,
            "policy_number": _canonical_policy(shared, doc),
            "coverages": _coverages_from_parts(doc.coverage_parts),
            "total_premium": parse_money(doc.total_premium),
        }
    if doc_type == DC.ENDORSEMENT:
        return {
            "named_insured": named,
            "policy_number": _canonical_policy(shared, doc),
            "change_description": doc.change_description,
            "effective_date": parse_date(doc.effective_date),
        }
    # NEEDS_REVIEW: no graph extraction schema.
    return None


def email_extraction_label(skeleton: Skeleton, scenario: Scenario) -> dict[str, Any]:
    """Ground-truth ``EmailExtraction`` for the email *body*.

    The body — not the attachments — is the source, so when the overlay makes the email
    body-only ("See attached."), the extractable fields are genuinely null. Otherwise the
    generated/fallback bodies state the insured, the policy (except brand-new
    submissions), and a requested effective date (except FNOL, which reports a loss rather
    than requesting a date).
    """
    shared = scenario.shared
    et = skeleton.email_type
    noise = skeleton.noise
    body_only = bool(noise and noise.email.body_only)
    if body_only:
        return {
            "named_insured": None,
            "policy_number": None,
            "requested_effective_date": None,
            "requested_changes": None,
            "contact_name": None,
            "contact_phone": None,
        }

    policy = None if et == EC.NEW_SUBMISSION else shared.policy_number
    # Only intents that inherently *request* a future effective date state one in the
    # body (new submission, mid-term endorsement). Renewals cite the expiration; FNOL
    # reports a loss; needs_review states nothing actionable. (Audit per case.)
    requested_eff = (
        parse_date(shared.effective_date)
        if et in (EC.NEW_SUBMISSION, EC.ENDORSEMENT)
        else None
    )
    # requested_changes is a free-text field (LLM-judged in MAR-10); provide a short
    # per-intent reference summary rather than a brittle exact string.
    requested_changes = {
        EC.NEW_SUBMISSION: "Quote/bind a new commercial package policy.",
        EC.RENEWAL: "Prepare a renewal quote for the expiring policy.",
        EC.ENDORSEMENT: "Endorse the in-force policy (mid-term change).",
        EC.CLAIM_FNOL: "Report a new loss / open a claim.",
        EC.NEEDS_REVIEW: None,
    }[et]
    return {
        "named_insured": shared.named_insured,
        "policy_number": policy,
        "requested_effective_date": requested_eff,
        "requested_changes": requested_changes,
        "contact_name": shared.contact_name,
        "contact_phone": shared.contact_phone,
    }


# --------------------------------------------------------------------------- #
# Document boundaries (page granularity) — rendered, graph-independent truth.
# --------------------------------------------------------------------------- #
def _pdf_page_count(pdf_bytes: bytes) -> int:
    return len(list(PDFPage.get_pages(io.BytesIO(pdf_bytes))))


def _render_single_doc_pages(
    shared: SharedFacts,
    doc_type: DocumentCategory,
    doc: DocumentContent,
    *,
    render_format: str,
    delimiter_mode: str,
    watermark: str | None,
    layout: int,
) -> int:
    """Page count of one document rendered *alone*, using the same knobs the bundle used
    so the per-document counts sum to the bundle's page count."""
    if render_format == "html":
        return 1  # HTML has no pagination; treated as a single unit
    if render_format == "scanned_pdf":
        return _pdf_page_count(build_scanned_pdf(shared, doc_type, doc))
    pdf = build_pdf(
        shared,
        [(doc_type, doc)],
        doc_modes=[delimiter_mode],
        watermarks=[watermark],
        layouts=[layout],
    )
    return _pdf_page_count(pdf)


def attachment_boundary_label(
    skeleton: Skeleton,
    scenario: Scenario,
    attachment_index: int,
    flat_offset: int,
) -> dict[str, Any]:
    """Boundary + per-document labels for one attachment.

    ``flat_offset`` is the attachment's start index into ``skeleton.flat_doc_types`` /
    ``scenario.documents`` / ``noise.docs`` (payload order).
    """
    plan = skeleton.attachments[attachment_index]
    noise = skeleton.noise
    docs_out: list[dict[str, Any]] = []
    page_cursor = 0
    for local_i, doc_type in enumerate(plan.docs):
        flat_i = flat_offset + local_i
        content = scenario.documents[flat_i]
        dn = noise.docs[flat_i] if noise else None
        render_format = dn.render_format if dn else "pdf"
        pages = _render_single_doc_pages(
            scenario.shared,
            doc_type,
            content,
            render_format=render_format,
            delimiter_mode=dn.delimiter_mode if dn else DELIM_HASH,
            watermark=dn.watermark if dn else None,
            layout=dn.layout_variant if dn else 0,
        )
        docs_out.append(
            {
                "index": local_i,
                "doc_type": doc_type.value,
                "page_start": page_cursor,
                "page_end": page_cursor + pages - 1,
                "extraction": document_extraction_label(doc_type, scenario.shared, content),
            }
        )
        page_cursor += pages

    n_pages = page_cursor
    # Boundary offsets = the page index at which each non-final document ends + 1
    # (i.e. the cut points between documents), for WindowDiff / Pk / boundary-F1.
    boundary_page_offsets = [d["page_end"] + 1 for d in docs_out[:-1]]

    expected = noise.expected if noise else None
    ai = attachment_index
    return {
        "filename": (noise.attachments[ai].filename if noise else f"attachment_{ai + 1}.pdf"),
        "render_format": (noise.docs[flat_offset].render_format if (noise and len(plan.docs) == 1) else "pdf"),
        "n_pages": n_pages,
        "documents": docs_out,
        "boundary_page_offsets": boundary_page_offsets,
        "n_documents": len(plan.docs),
        "expected_segment_types": [
            d.value for d in plan.docs if d != DC.NEEDS_REVIEW
        ],
        "expected_collapse": bool(expected.per_attachment_collapse[ai])
        if (expected and ai < len(expected.per_attachment_collapse)) else False,
        "expected_scanned": bool(expected.per_attachment_scanned[ai])
        if (expected and ai < len(expected.per_attachment_scanned)) else False,
        "expected_interrupt": bool(expected.per_attachment_expect_interrupt[ai])
        if (expected and ai < len(expected.per_attachment_expect_interrupt)) else False,
    }


# --------------------------------------------------------------------------- #
# Top-level: assemble the full label record for one scenario.
# --------------------------------------------------------------------------- #
def build_labels(case_id: str, tier: str, skeleton: Skeleton, scenario: Scenario) -> dict[str, Any]:
    """Full ground-truth label record for one frozen case."""
    noise = skeleton.noise
    email_type_hard = not (noise and (noise.email.body_only or noise.email.inconsistency))

    attachments: list[dict[str, Any]] = []
    flat_offset = 0
    for ai, plan in enumerate(skeleton.attachments):
        attachments.append(attachment_boundary_label(skeleton, scenario, ai, flat_offset))
        flat_offset += len(plan.docs)

    return {
        "case_id": case_id,
        "tier": tier,
        "email": {
            "email_type": skeleton.email_type.value,
            "email_type_hard": email_type_hard,
            "extraction": email_extraction_label(skeleton, scenario),
        },
        "attachments": attachments,
    }
