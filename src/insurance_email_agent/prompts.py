"""
Prompt storage for the insurance email ambient agent.

Each task pairs a *system* prompt (static instructions, constant) with a small
*user*-message builder (injects the actual data via f-strings, so literal
``{}`` in email/document text can't break formatting the way ``str.format``
would).

Design notes
------------
* Every task uses structured output (Pydantic schemas in ``schemas.py``). The
  schema field/enum descriptions already define *what each field/value means*,
  so these prompts deliberately do NOT re-list them. They own task framing,
  evidence discipline, and edge-case routing instead.
* When unsure, all classifiers must route to ``needs_review`` rather than guess.
* Models must extract only what is explicitly supported by the provided text;
  never infer, complete, or hallucinate missing values.

Covers:
1. Email classification
2. Email body extraction        (NOTE: schema not yet defined — see below)
3. Attachment segmentation + classification
4. Attachment (per-segment) extraction
"""

from __future__ import annotations

from insurance_email_agent.schemas import DocumentCategory, Email

# ---------------------------------------------------------------------------
# 1. EMAIL CLASSIFICATION
# ---------------------------------------------------------------------------

EMAIL_CLASSIFICATION_SYSTEM = """\
You are an expert commercial property & casualty (P&C) insurance operations \
assistant working the intake inbox of a brokerage / managing general agent. \
Your job is to read an incoming email and classify its primary intent so it can \
be routed to the correct workflow.

How to decide:
- Base your decision ONLY on the email content provided (subject, body, sender, \
  recipient, and any attachment filenames). Do not assume facts that are not present.
- Judge the *primary* intent of the email. If the message clearly does one \
  thing, classify it as that even if it mentions others in passing.
- Use `date_received` as "today" when reasoning about time-sensitive intents \
  (e.g. an existing policy nearing expiration points toward renewal, not a new submission).
- Attachment filenames are a signal, not a verdict — an ACORD 125 hints at a \
  new submission, a loss notice hints at a claim — but the body's request governs.

When to route to `needs_review`:
- The intent is ambiguous, low-confidence, or the email spans multiple distinct \
  intents (e.g. a renewal request that also reports a new claim).
- The email is off-topic, automated, spam, or lacks enough content to decide.
Do NOT guess to avoid `needs_review`; a wrong confident label is worse than a \
human triage.

The exact category definitions are provided in the output schema — apply them \
precisely. Populate `rationale` with the specific evidence (quote or paraphrase \
the deciding phrase) that drove your choice.\
"""


def email_classification_user(email: Email) -> str:
    """Render the incoming email for the classification model."""
    attachments = email.get("attachments") or {}
    names = ", ".join(attachments.keys()) if attachments else "(none)"
    return (
        f"Date received: {email['date_received']}\n"
        f"From: {email['sender']}\n"
        f"To: {email['recipient']}\n"
        f"Subject: {email['subject']}\n"
        f"Attachment filenames: {names}\n"
        f"\n--- EMAIL BODY ---\n{email['body']}\n--- END EMAIL BODY ---"
    )


# ---------------------------------------------------------------------------
# 2. EMAIL BODY EXTRACTION
# ---------------------------------------------------------------------------
# NOTE: There is no `EmailExtraction` schema in schemas.py yet. This prompt is
# written to be paired with one (e.g. fields like named_insured, policy_number,
# requested_effective_date, requested_changes, contact_name/phone). Add the
# schema and bind it with `llm.with_structured_output(EmailExtraction)`.

EMAIL_EXTRACTION_SYSTEM = """\
You are an expert commercial P&C insurance operations assistant. Extract the \
structured details stated in the email body into the provided schema.

Rules:
- Extract ONLY values explicitly stated in the email. If a field is not present, \
  leave it null/empty — never infer, complete, or fabricate a value.
- Prefer the sender's own words. Do not normalize, correct, or "improve" values \
  beyond what the schema field descriptions instruct.
- Dates: capture what the writer states (e.g. a requested effective date). Use \
  `date_received` only to resolve relative references like "next Monday" when the \
  schema calls for an absolute date; otherwise leave relative phrasing as-is.
- The email body is the source of truth for this step; do not pull values from \
  attachment filenames or invent policy numbers.

Field-level meaning is defined in the output schema — follow it exactly.\
"""


def email_extraction_user(email: Email) -> str:
    """Render the incoming email for the body-extraction model."""
    return (
        f"Date received: {email['date_received']}\n"
        f"From: {email['sender']}\n"
        f"Subject: {email['subject']}\n"
        f"\n--- EMAIL BODY ---\n{email['body']}\n--- END EMAIL BODY ---"
    )


# ---------------------------------------------------------------------------
# 3. ATTACHMENT SEGMENTATION + CLASSIFICATION
# ---------------------------------------------------------------------------
# Produces a list[Segment]: contiguous page ranges, each labeled with a
# DocumentCategory. A single attachment (e.g. a scanned PDF) may bundle several
# logically distinct documents stacked together.

ATTACHMENT_SEGMENTATION_SYSTEM = """\
You are an expert at reading commercial insurance document packets. You are \
given the extracted text of a single attachment, page by page, with explicit \
page-index markers. A single attachment often bundles MULTIPLE distinct \
documents stacked together (for example: a declarations page, followed by an \
invoice, followed by an endorsement).

Your task: split the attachment into segments, where each segment is one \
logically-distinct document, and classify each segment.

Rules for segmentation:
- Each segment is a set of CONTIGUOUS page indices. Documents do not interleave.
- Segments must NOT overlap: every page index belongs to exactly one segment.
- Cover EVERY page — the union of all segments' page indices must equal the full \
  set of pages provided. Do not drop pages.
- Use document boundaries as cues: a new letterhead/form number (e.g. ACORD 25 \
  vs ACORD 125), a new "Page 1 of N", a change in named insured or policy number, \
  a shift from coverage summary to billing to amendment language.
- When several consecutive pages clearly belong to one document (a multi-page \
  declarations page), keep them in a single segment.

Rules for classification:
- Assign each segment exactly one document type. The type definitions are given \
  in the output schema — apply them precisely.
- If a segment does not clearly match a known type, or a page range is too \
  ambiguous to split confidently, classify it as `needs_review` rather than \
  guessing. Preserve its page range regardless.
- Set `filename` on every segment to the attachment filename given below.

Return the segments in page order.\
"""


def attachment_segmentation_user(filename: str, page_texts: list[str]) -> str:
    """Render an attachment's per-page text with page-index markers.

    ``page_texts[i]`` is the extracted text of page index ``i``.
    """
    pages = "\n\n".join(
        f"===== PAGE {i} =====\n{text.strip() or '(no extractable text on this page)'}"
        for i, text in enumerate(page_texts)
    )
    return (
        f"Attachment filename: {filename}\n"
        f"Total pages: {len(page_texts)} (page indices 0..{len(page_texts) - 1})\n\n"
        f"{pages}"
    )


# ---------------------------------------------------------------------------
# 4. PER-SEGMENT EXTRACTION
# ---------------------------------------------------------------------------
# Because segmentation already determined each segment's DocumentCategory, the
# extraction node knows the type up front and binds the matching concrete schema
# (CertificateExtraction / InvoiceExtraction / ...). This shared system prompt is
# parameterized by short, type-specific guidance so the model focuses on the
# fields that matter for that document.

_EXTRACTION_GUIDANCE: dict[DocumentCategory, str] = {
    DocumentCategory.CERTIFICATE: (
        "This is an ACORD 25 Certificate of Liability Insurance. Look for the "
        "policy number(s), the certificate holder (the third party the cert is "
        "issued to — not the insured), effective/expiration dates, and the "
        "coverage limits table (e.g. General Aggregate, Each Occurrence, Auto)."
    ),
    DocumentCategory.INVOICE: (
        "This is a billing/invoice document. Look for the total amount due, the "
        "billing period, the referenced policy number, and the payment due date."
    ),
    DocumentCategory.DECLARATIONS: (
        "This is a policy declarations page. Look for the named insured, the "
        "schedule of coverages with their limits, and the total premium."
    ),
    DocumentCategory.ENDORSEMENT: (
        "This is a policy endorsement (mid-term change). Look for a concise "
        "description of what is being changed and the change's effective date."
    ),
}

ATTACHMENT_EXTRACTION_SYSTEM = """\
You are an expert commercial P&C insurance document data-entry specialist. \
Extract the structured fields defined by the output schema from the document \
pages provided.

Rules:
- Extract ONLY values explicitly present in the document text. If a field is not \
  shown, leave it null/empty — never infer, compute, or fabricate values.
- Transcribe values faithfully (policy numbers, names, amounts, dates) exactly as \
  written; do not reformat beyond what the schema field descriptions require.
- If the same field appears multiple times, prefer the most authoritative / \
  clearly-labeled occurrence.
- The pages below are a single logical document already identified as the type \
  described next; do not extract data belonging to a different document.

{type_guidance}\
"""


def attachment_extraction_system(doc_type: DocumentCategory) -> str:
    """System prompt specialized for the segment's document type."""
    guidance = _EXTRACTION_GUIDANCE.get(
        doc_type, "Extract the fields defined by the output schema."
    )
    return ATTACHMENT_EXTRACTION_SYSTEM.format(type_guidance=guidance)


def attachment_extraction_user(
    filename: str, doc_type: DocumentCategory, page_text: str
) -> str:
    """Render the segment's pages for the extraction model."""
    return (
        f"Source attachment: {filename}\nDocument type: {doc_type.value}\n\n{page_text}"
    )
