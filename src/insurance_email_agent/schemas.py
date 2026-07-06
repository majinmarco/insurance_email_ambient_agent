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


class Email(TypedDict):
    id: str  # randomly generated uuid
    subject: str
    date_received: datetime
    body: str
    sender: str
    recipient: str
    attachments: dict[str, bytes]


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


class DocumentCategory(str, Enum):
    """
    Custom class for DocumentCategory to maintain consistent category naming across the codebase
    """

    CERTIFICATE = "certificate"
    INVOICE = "invoice"
    DECLARATIONS = "declarations"
    ENDORSEMENT = "endorsement"
    NEEDS_REVIEW = "needs_review"


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


# --- one concrete schema per document type ---


class CertificateExtraction(BaseModel):
    doc_type: Literal[DocumentCategory.CERTIFICATE] = DocumentCategory.CERTIFICATE
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
    amount_due: float | None = Field(None, description="Total premium due")
    billing_period: str | None = None
    policy_reference: str | None = None
    due_date: datetime | None = None


class DeclarationsExtraction(BaseModel):
    doc_type: Literal[DocumentCategory.DECLARATIONS] = DocumentCategory.DECLARATIONS
    named_insured: str | None = None
    coverages: dict[str, str] = Field(default_factory=dict)
    total_premium: float | None = None


class EndorsementExtraction(BaseModel):
    doc_type: Literal[DocumentCategory.ENDORSEMENT] = DocumentCategory.ENDORSEMENT
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
