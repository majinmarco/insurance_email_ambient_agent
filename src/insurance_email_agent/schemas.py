"""
Storage of pydantic/typing data schemas for:
1. Email classification
2. Email body extraction
3. Attachment classification
4. Attachment extraction
5. States
"""

from datetime import datetime
from enum import Enum
from typing import Annotated, Literal, TypedDict, Union

from pydantic import BaseModel, Field


class EmailCategory(str, Enum):
    """
    Custom class for EmailCategory to maintain consistent category naming across the codebase
    """

    NEW_SUBMISSION = "new_submission"
    RENEWAL = "renewal"
    ENDORSEMENT = "endorsement"
    CLAIM_FNOL = "claim_fnol"
    NEEDS_REVIEW = "needs_review"


class EmailClassification(BaseModel):
    category: EmailCategory = Field(
        description=(
            "The intent of the incoming email. Choose exactly one:\n"
            "- new_submission: A request to quote or bind a brand-new policy, "
            "typically including an application packet (e.g., ACORD 125/126/130) "
            "for a risk not currently on the books.\n"
            "- renewal: Correspondence about an existing policy approaching its "
            "expiration, including renewal quote requests, updated exposures, or "
            "intent to renew.\n"
            "- endorsement: A request to change an active, in-force policy mid-term "
            "(e.g., add/remove a vehicle or location, adjust limits, update the "
            "named insured or address).\n"
            "- claim_fnol: A first notice of loss or any report of a new claim, "
            "incident, or damage event under an existing policy.\n"
            "- needs_review: Anything that does not clearly fit the above, is "
            "ambiguous, spans multiple intents, or is low-confidence. Route here "
            "instead of guessing so a human can triage."
        )
    )
    rationale: str = Field(
        description="Describe your reasoning for the chosen category in 2-3 sentences."
    )


class DocumentCategory(str, Enum):
    """
    Custom class for DocumentCategory to maintain consistent category naming across the codebase
    """

    CERTIFICATE = "certificate of insurance"
    INVOICE = "insurance invoice"
    DECLARATIONS = "insurance declaration"
    ENDORSEMENT = "insurance endorsement"
    NEEDS_REVIEW = "unrelated or unclear document"


class DocumentClassification(BaseModel):
    doc_type: DocumentCategory = Field(
        description=(
            "The type of insurance document. Choose exactly one:\n"
            "- certificate: An ACORD 25 Certificate of Liability Insurance — a "
            "one-page proof-of-coverage summary issued to a third party, showing "
            "policy numbers, limits, effective dates, and the certificate holder.\n"
            "- invoice: A billing document requesting payment, showing premium "
            "amounts due, billing period, policy reference, and payment terms.\n"
            "- declarations: A policy declarations page — the issued policy's "
            "summary of named insured, coverages, limits, premiums, and forms; "
            "confirms bound coverage.\n"
            "- endorsement: A document amending an in-force policy (adding/removing "
            "coverage, insureds, or locations, or changing limits/terms)."
        )
    )


# --- one concrete schema per document type ---


class CertificateExtraction(BaseModel):
    doc_type: Literal[DocumentCategory.CERTIFICATE] = DocumentCategory.CERTIFICATE
    named_insured: str | None = Field(
        None, description="Named insured / policyholder shown on the document"
    )
    policy_number: str | None = Field(
        None, description="Policy number shown on the ACORD 25"
    )
    certificate_holder: str | None = Field(
        None, description="Name of the certificate holder"
    )
    effective_date: datetime | None = None
    expiration_date: datetime | None = None
    limits: dict[str, str] = Field(
        default_factory=dict, description="Coverage → limit amount"
    )


class InvoiceExtraction(BaseModel):
    doc_type: Literal[DocumentCategory.INVOICE] = DocumentCategory.INVOICE
    named_insured: str | None = Field(
        None, description="Named insured / policyholder shown on the document"
    )
    policy_number: str | None = Field(
        None, description="Policy number shown on the document"
    )
    amount_due: float | None = Field(None, description="Total premium due")
    billing_period: str | None = None
    policy_reference: str | None = None
    due_date: datetime | None = None


class DeclarationsExtraction(BaseModel):
    doc_type: Literal[DocumentCategory.DECLARATIONS] = DocumentCategory.DECLARATIONS
    named_insured: str | None = Field(
        None, description="Named insured / policyholder shown on the document"
    )
    policy_number: str | None = Field(
        None, description="Policy number shown on the document"
    )
    coverages: dict[str, str] = Field(default_factory=dict)
    total_premium: float | None = None


class EndorsementExtraction(BaseModel):
    doc_type: Literal[DocumentCategory.ENDORSEMENT] = DocumentCategory.ENDORSEMENT
    named_insured: str | None = Field(
        None, description="Named insured / policyholder shown on the document"
    )
    policy_number: str | None = Field(
        None, description="Policy number shown on the document"
    )
    change_description: str | None = None
    effective_date: datetime | None = None


# --- the single "type" you referenced ---

Extraction = Annotated[  # Pipes are for "Union" type
    CertificateExtraction
    | InvoiceExtraction
    | DeclarationsExtraction
    | EndorsementExtraction,
    Field(discriminator="doc_type"),
]


class Segment(BaseModel):
    """
    Classification and segmentation data from an attachment for a specific segment
    """

    category: DocumentClassification = Field(description="Document class of segment")
    filename: str = Field(
        description="Filename corresponding to attachment that this segment belongs to"
    )
    page_indices: list[int] = Field(
        description="List of page indices corresponding to this segment"
    )
    text: str = Field(description="Text turned to markdown from the segment of text")
    extraction: Extraction | None = Field(
        None, description="Extraction result for this segment"
    )


class Attachment(TypedDict):
    filename: str
    content: bytes
    segments: list[Segment]


class Email(TypedDict):
    id: str  # randomly generated uuid
    subject: str
    date_received: datetime
    body: str
    sender: str
    recipient: str
    attachments: list[Attachment]


class EmailExtraction(BaseModel):
    """
    Structured details extracted from the email body itself (as opposed to
    attachments). All fields are optional — populate only what the sender
    explicitly states, leave the rest null.
    """

    named_insured: str | None = Field(
        None,
        description=(
            "The named insured / applicant the email concerns, exactly as written "
            "(company or person). Null if not stated."
        ),
    )
    policy_number: str | None = Field(
        None,
        description=(
            "An existing policy number referenced in the email (for renewals, "
            "endorsements, or claims). Null for brand-new submissions or if absent."
        ),
    )
    requested_effective_date: datetime | None = Field(
        None,
        description=(
            "The effective date the sender is requesting for the new policy, "
            "renewal, or endorsement change. Only populate when an absolute date "
            "can be determined; leave null for vague phrasing like 'asap'."
        ),
    )
    requested_changes: str | None = Field(
        None,
        description=(
            "A concise summary of what the sender is asking for — e.g. the "
            "endorsement change, the coverage requested, or the reason for the "
            "email. Null if the email states no actionable request."
        ),
    )
    contact_name: str | None = Field(
        None,
        description="Name of the person to contact about this email, if stated.",
    )
    contact_phone: str | None = Field(
        None,
        description="Phone number provided for follow-up contact, if stated.",
    )
