"""Turn a :class:`Skeleton` into concrete, coherent test content.

One structured-output LLM call fills a single :class:`Scenario` so the email and
every attached document reference the *same* named insured / policy / dates.
Falls back to a deterministic template scenario when ``--no-llm`` is set or the
LLM call fails, so the pipeline is always exercisable (and free) offline.

The generation schemas are intentionally rich (producer/insurer blocks, ACORD
coverage lines, forms schedules, itemized charges, endorsement provisions) so the
reportlab templates can render authentic, real-world-looking insurance documents.
Everything is optional so the model fills only the fields relevant to each slot.

Doc types are NOT taken from the LLM — they are fixed by the skeleton (the source
of truth for the "expected" side of verification). The scenario only carries the
per-document *field values*, mapped back to types positionally.
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field

from insurance_email_agent.schemas import DocumentCategory, EmailCategory

from .taxonomy import Skeleton

EC = EmailCategory
DC = DocumentCategory


# --------------------------------------------------------------------------- #
# Generation schemas (kept separate from the graph's own schemas). Structured-
# output-safe: only primitives and list[BaseModel] — no dicts, tuples, or unions.
# --------------------------------------------------------------------------- #
class LimitLine(BaseModel):
    coverage: str = Field(description="Limit label, e.g. 'Each Occurrence' or 'General Aggregate'")
    limit: str = Field(description="Limit amount, e.g. '$1,000,000'")


class Producer(BaseModel):
    """The broker / agency that produced the account (ACORD 'PRODUCER' box)."""

    agency: str | None = None
    address: str | None = None
    contact_name: str | None = None
    phone: str | None = None
    fax: str | None = None
    email: str | None = None


class Insurer(BaseModel):
    """One carrier in the ACORD 25 'INSURER(S) AFFORDING COVERAGE' table."""

    letter: str | None = Field(None, description="Insurer letter A–F")
    name: str | None = None
    naic: str | None = Field(None, description="5-digit NAIC number, as a string")


class CoverageLine(BaseModel):
    """A row in the ACORD 25 COVERAGES grid or a declarations coverage part."""

    insurer_letter: str | None = Field(None, description="Links to an Insurer.letter")
    coverage_type: str | None = Field(
        None,
        description="e.g. 'Commercial General Liability', 'Automobile Liability', "
        "'Umbrella Liability', 'Workers Compensation'",
    )
    policy_number: str | None = None
    effective_date: str | None = Field(None, description="MM/DD/YYYY on the certificate")
    expiration_date: str | None = None
    addl_insured: bool | None = None
    subro_waived: bool | None = None
    limits: list[LimitLine] | None = Field(None, description="Standard limit label→amount rows")
    premium: str | None = Field(None, description="Per-coverage premium (declarations)")


class FormLine(BaseModel):
    form_number: str | None = Field(None, description="e.g. 'CG 00 01 04 13'")
    title: str | None = None


class ChargeLine(BaseModel):
    description: str | None = Field(
        None, description="e.g. 'Policy Premium', 'Terrorism (TRIA)', 'State Tax', 'Broker Fee'"
    )
    amount: str | None = Field(None, description="e.g. '$1,234.00'")


class ScheduleRow(BaseModel):
    label: str | None = None
    value: str | None = None


class TableRow(BaseModel):
    """A generic data-table row (used by the structured needs_review document)."""

    cells: list[str] = Field(default_factory=list, description="Cell values, in column order")


class SharedFacts(BaseModel):
    """Facts shared by the email and every attachment, so they stay coherent."""

    named_insured: str
    policy_number: str | None = Field(
        None, description="Existing policy number; leave null for brand-new submissions"
    )
    insured_address: str
    contact_name: str
    contact_email: str
    contact_phone: str
    carrier_name: str
    effective_date: str = Field(description="ISO date, YYYY-MM-DD")
    expiration_date: str = Field(description="ISO date, YYYY-MM-DD")
    coverage_limits: list[LimitLine]
    total_premium: str = Field(description="e.g. '$18,750'")
    # richer shared identity used across documents
    producer: Producer | None = None
    insurers: list[Insurer] | None = Field(None, description="INSURER(S) AFFORDING COVERAGE, A–F")
    business_description: str | None = Field(None, description="e.g. 'Long-haul motor freight trucking'")
    entity_type: str | None = Field(None, description="e.g. 'Limited Liability Company', 'Corporation'")
    fein: str | None = Field(None, description="Federal EIN, 'XX-XXXXXXX'")
    naic_carrier: str | None = None


class GeneratedEmail(BaseModel):
    subject: str
    body: str = Field(description="Email body written in the chosen intent's voice")
    sender: str = Field(description="From line, e.g. 'Jane Broker <jane@acme.com>'")
    recipient: str


class DocumentContent(BaseModel):
    """Flat, all-optional per-document fields (only the relevant ones filled)."""

    title: str = Field(description="Short document title / header line")

    # --- certificate (ACORD 25) ---
    certificate_holder: str | None = None
    coverage_lines: list[CoverageLine] | None = None
    description_of_operations: str | None = None
    authorized_representative: str | None = None
    certificate_number: str | None = None
    revision_number: str | None = None

    # --- invoice ---
    invoice_number: str | None = None
    invoice_date: str | None = None
    account_number: str | None = None
    line_items: list[ChargeLine] | None = None
    subtotal: str | None = None
    taxes: str | None = None
    fees: str | None = None
    prior_balance: str | None = None
    amount_due: str | None = None
    billing_period: str | None = None
    due_date: str | None = None
    payment_methods: str | None = None
    remit_to: str | None = None

    # --- declarations ---
    coverage_parts: list[CoverageLine] | None = None
    forms_schedule: list[FormLine] | None = None
    premium_summary: list[ChargeLine] | None = None
    countersignature: str | None = None
    countersignature_date: str | None = None

    # --- endorsement ---
    form_number: str | None = None
    endorsement_title: str | None = None
    endorsement_number: str | None = None
    schedule_rows: list[ScheduleRow] | None = None
    provisions: list[str] | None = None
    additional_premium: str | None = None
    change_description: str | None = Field(None, description="One-line summary of the change")

    # --- shared / legacy ---
    coverages: list[LimitLine] | None = Field(None, description="Simple limits table (fallback)")
    total_premium: str | None = None
    effective_date: str | None = None
    expiration_date: str | None = None

    # --- needs_review (insurance-adjacent, but not one of the four modeled types) ---
    nr_kind: str | None = Field(
        None, description="Subtype/title, e.g. 'Loss Run Report', 'Commercial Insurance Application'"
    )
    nr_header_rows: list[ScheduleRow] | None = Field(
        None, description="Fielded header info (Insured, Policy No., Valuation Date, Prepared By, …)"
    )
    nr_columns: list[str] | None = Field(None, description="Data-table column headers, in order")
    nr_rows: list[TableRow] | None = Field(None, description="Data-table rows (cells match nr_columns order)")
    unrelated_subtitle: str | None = Field(None, description="Subtitle / reference line")
    unrelated_body: str | None = Field(None, description="Short intro / notes paragraph(s)")


class Scenario(BaseModel):
    shared: SharedFacts
    email: GeneratedEmail
    documents: list[DocumentContent] = Field(
        description="One entry per requested document, IN THE GIVEN ORDER"
    )


# --------------------------------------------------------------------------- #
# LLM generation
# --------------------------------------------------------------------------- #
_EMAIL_GUIDANCE: dict[EmailCategory, str] = {
    EC.NEW_SUBMISSION: "a broker requesting a quote/bind for a BRAND-NEW policy (no existing policy number)",
    EC.RENEWAL: "correspondence about an existing policy nearing expiration — a renewal quote request or intent to renew",
    EC.ENDORSEMENT: "a request to change an active in-force policy mid-term (add/remove a vehicle or location, adjust limits, update the named insured)",
    EC.CLAIM_FNOL: "a first notice of loss reporting a new claim/incident/damage under an existing policy",
    EC.NEEDS_REVIEW: "an ambiguous message that does not clearly fit one intent, spans multiple intents, or is low-signal",
}

_DOC_GUIDANCE: dict[DocumentCategory, str] = {
    DC.CERTIFICATE: (
        "ACORD 25 Certificate of Liability Insurance. Fill certificate_holder (a THIRD PARTY, "
        "not the insured); coverage_lines — one per line of business (Commercial General Liability, "
        "Automobile Liability, Umbrella Liability, Workers Compensation), each with insurer_letter "
        "(matching a shared.insurers letter), policy_number, MM/DD/YYYY effective_date/expiration_date, "
        "addl_insured, subro_waived, and the STANDARD limits label set (CGL: Each Occurrence, Damage to "
        "Rented Premises, Med Exp, Personal & Adv Injury, General Aggregate, Products-Comp/Op Agg; Auto: "
        "Combined Single Limit; WC: E.L. Each Accident / E.L. Disease-Ea Employee / E.L. Disease-Policy "
        "Limit); description_of_operations (additional-insured / primary & non-contributory / waiver "
        "language citing form numbers); authorized_representative; certificate_number."
    ),
    DC.INVOICE: (
        "Premium invoice. Fill invoice_number, invoice_date, account_number; line_items (Policy Premium, "
        "Terrorism/TRIA, state tax or surcharge, broker fee, prior balance); subtotal, taxes, fees, "
        "amount_due (these MUST reconcile — line items sum to amount_due); due_date, billing_period, "
        "payment_methods, remit_to."
    ),
    DC.DECLARATIONS: (
        "Policy declarations page. Fill coverage_parts (each coverage part with its limits and a per-part "
        "premium); forms_schedule (real ISO/carrier form numbers with editions and titles); "
        "premium_summary (premium, TRIA, taxes/fees, total); total_premium; countersignature and "
        "countersignature_date; effective_date/expiration_date."
    ),
    DC.ENDORSEMENT: (
        "Policy change endorsement. Fill form_number, endorsement_title, endorsement_number; schedule_rows "
        "(the changed items, e.g. added vehicle Year/Make/Model/VIN); provisions — 2 to 4 amendatory "
        "paragraphs in real policy language, the last ending 'All other terms and conditions of this "
        "policy remain unchanged.'; additional_premium; effective_date; a one-line change_description."
    ),
    DC.NEEDS_REVIEW: (
        "A STRUCTURED, insurance-adjacent document that is deliberately NOT one of the four modeled types "
        "(not a certificate, invoice, declaration, or endorsement). Pick ONE subtype and set nr_kind to it: "
        "'Loss Run Report' (claims history), 'Commercial Insurance Application (ACORD 125)', 'Statement of "
        "Values', or 'Submission Cover Letter'. Fill unrelated_subtitle (a reference line), nr_header_rows "
        "(fielded header — Insured, Policy No., Valuation Date, Prepared By, etc.), nr_columns + nr_rows (a "
        "data table whose cells line up with the columns), and a short unrelated_body intro. Reference the "
        "shared insured/policy for coherence. Use real insurance vocabulary, but do NOT format it as a "
        "certificate/invoice/declaration/endorsement — it must read as a different kind of document so the "
        "classifier routes it to review."
    ),
}

_SYSTEM = (
    "You generate realistic synthetic US commercial property & casualty (P&C) "
    "insurance test data. Everything is fictional but internally consistent: the "
    "email and every attached document reference the SAME named insured, policy "
    "number, carrier, dates, and producer from `shared`. Write like real ACORD "
    "forms and carrier documents:\n"
    "- Use authentic commercial P&C vocabulary and real ISO/ACORD form numbers with "
    "editions (e.g. CG 00 01 04 13, IL 00 17 11 98, CP 00 10 10 12, CG 20 10 04 13, "
    "CA 00 01 11 20) and valid 5-digit NAIC numbers.\n"
    "- Use the STANDARD limit labels for each coverage; format money as '$1,234,567'.\n"
    "- Any itemized charges or premium parts MUST sum to the stated total.\n"
    "- A needs_review document is insurance-adjacent but clearly NOT a certificate, "
    "invoice, declaration, or endorsement (e.g. a loss run, application, or cover letter).\n"
    "- IMPORTANT: never begin any field value with the character '#'; write 'No.' "
    "not '#'. Do not add commentary."
)


def _doc_slot_lines(skeleton: Skeleton) -> str:
    lines = []
    idx = 0
    for a_i, att in enumerate(skeleton.attachments):
        for d in att.docs:
            idx += 1
            lines.append(f"  {idx}. (attachment {a_i + 1}) {d.value} — {_DOC_GUIDANCE[d]}")
    return "\n".join(lines) or "  (no attachments)"


def _build_prompt(skeleton: Skeleton) -> str:
    n_docs = len(skeleton.flat_doc_types)
    return (
        f"Email intent to write: {skeleton.email_type.value} — {_EMAIL_GUIDANCE[skeleton.email_type]}.\n\n"
        "First invent `shared`: the named insured + address, the producer (agency, address, contact, "
        "phone, email), the insurers A–F (name + NAIC) affording coverage, business_description, "
        "entity_type, fein, carrier, policy number, and policy dates. Keep these CONSISTENT across every "
        "document (a certificate's coverage_lines[].insurer_letter must match a shared.insurers[].letter; "
        "the same policy number appears everywhere; certificate dates in MM/DD/YYYY).\n\n"
        "Then write the `email` in the intent's voice (mention the named insured and, where applicable, "
        "the policy number and a requested effective date).\n\n"
        f"Then produce EXACTLY {n_docs} document(s) in `documents`, in this order, filling ONLY the fields "
        f"noted for each (leave the rest null):\n{_doc_slot_lines(skeleton)}\n\n"
        "Give each document a `title` suitable as a header line (no leading '#')."
    )


def _make_llm(provider: str, model: str | None, temperature: float):
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic  # optional dep

        return ChatAnthropic(model=model or "claude-sonnet-5", temperature=temperature)
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model or os.getenv("SCENARIO_LLM_MODEL", "gpt-5.4-mini"),
        temperature=temperature,
    )


def generate_scenario(
    skeleton: Skeleton,
    *,
    provider: str = "openai",
    model: str | None = None,
    temperature: float = 0.7,
    use_llm: bool = True,
) -> Scenario:
    """Generate content for ``skeleton``; deterministic fallback on any failure."""
    if not use_llm:
        return _fallback_scenario(skeleton)

    n_docs = len(skeleton.flat_doc_types)
    prompt = _build_prompt(skeleton)
    structured = _make_llm(provider, model, temperature).with_structured_output(Scenario)

    for _ in range(2):
        try:
            sc = structured.invoke([("system", _SYSTEM), ("human", prompt)])
            if len(sc.documents) == n_docs:
                return sc
        except Exception as exc:  # noqa: BLE001 — any LLM/transport failure → fallback
            print(f"  [scenario] LLM generation failed ({exc!r}); retrying/falling back")
    return _fallback_scenario(skeleton)


# --------------------------------------------------------------------------- #
# Deterministic fallback (no LLM) — rich, coherent, always correct count
# --------------------------------------------------------------------------- #
_FALLBACK_SUBJECT: dict[EmailCategory, str] = {
    EC.NEW_SUBMISSION: "New submission — GL/Auto quote request for Northgate Logistics LLC",
    EC.RENEWAL: "Renewal — CPP-2291043 expiring 08/01, please quote",
    EC.ENDORSEMENT: "Endorsement request — add scheduled vehicle to CPP-2291043",
    EC.CLAIM_FNOL: "FNOL — rear-end collision, unit 12, policy CPP-2291043",
    EC.NEEDS_REVIEW: "Following up + a couple of unrelated questions",
}

_FALLBACK_BODY: dict[EmailCategory, str] = {
    EC.NEW_SUBMISSION: (
        "Hi team,\n\nWe'd like to quote a brand-new commercial package for "
        "Northgate Logistics LLC (1450 Harbor Blvd, Suite 200, Oakland, CA 94607). "
        "Requested effective date is 2026-08-01. Application packet attached.\n\n"
        "Thanks,\nDana Whitfield\n(510) 555-0142"
    ),
    EC.RENEWAL: (
        "Hello,\n\nPolicy CPP-2291043 for Northgate Logistics LLC expires "
        "2026-08-01. Please prepare a renewal quote; expiring declarations and "
        "the latest invoice are attached.\n\nBest,\nDana Whitfield"
    ),
    EC.ENDORSEMENT: (
        "Hi,\n\nPlease endorse policy CPP-2291043 to add a 2025 Freightliner "
        "Cascadia (VIN 3AKJHHDR9PSNL1234) effective 2026-08-15. Signed endorsement "
        "request attached.\n\nThanks,\nDana Whitfield"
    ),
    EC.CLAIM_FNOL: (
        "Reporting a new loss under CPP-2291043 for Northgate Logistics LLC. "
        "On 2026-07-06 our unit 12 was rear-ended at a stoplight; minor bumper "
        "damage, no injuries. Please open a claim. Contact Dana at (510) 555-0142."
    ),
    EC.NEEDS_REVIEW: (
        "Hey — circling back on our last call. Also, is the office newsletter "
        "still going out monthly, and did you get my note about the holiday party? "
        "Attaching a doc, not sure it's the right one.\n\nDana"
    ),
}


def _fallback_doc(doc_type: DocumentCategory, shared: SharedFacts) -> DocumentContent:
    pol = shared.policy_number or "PENDING"
    if doc_type == DC.CERTIFICATE:
        return DocumentContent(
            title="Certificate of Liability Insurance (ACORD 25)",
            certificate_holder="Port of Oakland\n530 Water Street\nOakland, CA 94607",
            certificate_number="CERT-2026-0142",
            revision_number="0",
            authorized_representative="Priya Anand",
            description_of_operations=(
                "The Port of Oakland is included as an Additional Insured with respect to General "
                "Liability per form CG 20 10 04 13 and CG 20 37 04 13, on a Primary and Non-Contributory "
                "basis per CG 20 01 04 13. Waiver of Subrogation applies in favor of the certificate "
                "holder per CG 24 04 05 09 and WC 00 03 13. Umbrella follows form. RE: terminal access "
                "and drayage operations at Berth 24."
            ),
            coverage_lines=[
                CoverageLine(
                    insurer_letter="A",
                    coverage_type="Commercial General Liability",
                    policy_number=pol,
                    effective_date="08/01/2026",
                    expiration_date="08/01/2027",
                    addl_insured=True,
                    subro_waived=True,
                    limits=[
                        LimitLine(coverage="Each Occurrence", limit="$1,000,000"),
                        LimitLine(coverage="Damage to Rented Premises (Ea occurrence)", limit="$100,000"),
                        LimitLine(coverage="Med Exp (Any one person)", limit="$5,000"),
                        LimitLine(coverage="Personal & Adv Injury", limit="$1,000,000"),
                        LimitLine(coverage="General Aggregate", limit="$2,000,000"),
                        LimitLine(coverage="Products-Comp/Op Agg", limit="$2,000,000"),
                    ],
                ),
                CoverageLine(
                    insurer_letter="A",
                    coverage_type="Automobile Liability",
                    policy_number="BAP-5510388",
                    effective_date="08/01/2026",
                    expiration_date="08/01/2027",
                    addl_insured=True,
                    subro_waived=False,
                    limits=[LimitLine(coverage="Combined Single Limit (Ea accident)", limit="$1,000,000")],
                ),
                CoverageLine(
                    insurer_letter="B",
                    coverage_type="Umbrella Liability",
                    policy_number="CUP-7741220",
                    effective_date="08/01/2026",
                    expiration_date="08/01/2027",
                    limits=[
                        LimitLine(coverage="Each Occurrence", limit="$5,000,000"),
                        LimitLine(coverage="Aggregate", limit="$5,000,000"),
                    ],
                ),
                CoverageLine(
                    insurer_letter="C",
                    coverage_type="Workers Compensation",
                    policy_number="WC-3390017",
                    effective_date="08/01/2026",
                    expiration_date="08/01/2027",
                    subro_waived=True,
                    limits=[
                        LimitLine(coverage="E.L. Each Accident", limit="$1,000,000"),
                        LimitLine(coverage="E.L. Disease - Ea Employee", limit="$1,000,000"),
                        LimitLine(coverage="E.L. Disease - Policy Limit", limit="$1,000,000"),
                    ],
                ),
            ],
        )
    if doc_type == DC.INVOICE:
        return DocumentContent(
            title="Premium Invoice",
            invoice_number="INV-2026-88417",
            invoice_date="2026-07-10",
            account_number="ACCT-NGL-0142",
            line_items=[
                ChargeLine(description="Policy Premium", amount="$18,750.00"),
                ChargeLine(description="Terrorism Risk Insurance (TRIA)", amount="$185.00"),
                ChargeLine(description="CA State Surcharge", amount="$9.38"),
                ChargeLine(description="Broker Fee", amount="$250.00"),
                ChargeLine(description="Prior Balance", amount="$0.00"),
            ],
            subtotal="$18,935.00",
            taxes="$9.38",
            fees="$250.00",
            prior_balance="$0.00",
            amount_due="$19,194.38",
            billing_period="2026-08-01 to 2027-08-01",
            due_date="2026-07-25",
            payment_methods=(
                "Payable to Continental Casualty Company. Pay by check, ACH, or online at "
                "billpay.cna.com. Please reference the invoice and policy number on all payments."
            ),
            remit_to="Continental Casualty Company\nPO Box 74007619\nChicago, IL 60674-7619",
        )
    if doc_type == DC.DECLARATIONS:
        return DocumentContent(
            title="Commercial Package Policy Declarations",
            effective_date=shared.effective_date,
            expiration_date=shared.expiration_date,
            coverage_parts=[
                CoverageLine(
                    coverage_type="Commercial Property",
                    premium="$6,200.00",
                    limits=[
                        LimitLine(coverage="Building", limit="$4,500,000"),
                        LimitLine(coverage="Business Personal Property", limit="$750,000"),
                    ],
                ),
                CoverageLine(
                    coverage_type="Commercial General Liability",
                    premium="$7,850.00",
                    limits=[
                        LimitLine(coverage="Each Occurrence", limit="$1,000,000"),
                        LimitLine(coverage="General Aggregate", limit="$2,000,000"),
                        LimitLine(coverage="Products-Comp/Op Agg", limit="$2,000,000"),
                    ],
                ),
                CoverageLine(
                    coverage_type="Commercial Auto",
                    premium="$3,900.00",
                    limits=[LimitLine(coverage="Combined Single Limit", limit="$1,000,000")],
                ),
                CoverageLine(
                    coverage_type="Commercial Umbrella",
                    premium="$800.00",
                    limits=[
                        LimitLine(coverage="Each Occurrence", limit="$5,000,000"),
                        LimitLine(coverage="Aggregate", limit="$5,000,000"),
                    ],
                ),
            ],
            forms_schedule=[
                FormLine(form_number="IL 00 17 11 98", title="Common Policy Conditions"),
                FormLine(form_number="IL 00 21 09 08", title="Nuclear Energy Liability Exclusion"),
                FormLine(form_number="CG 00 01 04 13", title="Commercial General Liability Coverage Form"),
                FormLine(form_number="CG 20 10 04 13", title="Additional Insured - Owners, Lessees or Contractors"),
                FormLine(form_number="CG 24 04 05 09", title="Waiver of Transfer of Rights of Recovery Against Others"),
                FormLine(form_number="CP 00 10 10 12", title="Building and Personal Property Coverage Form"),
                FormLine(form_number="CP 10 30 10 12", title="Causes of Loss - Special Form"),
                FormLine(form_number="CA 00 01 11 20", title="Business Auto Coverage Form"),
            ],
            premium_summary=[
                ChargeLine(description="Total Policy Premium", amount="$18,750.00"),
                ChargeLine(description="Terrorism Risk Insurance (TRIA)", amount="$185.00"),
            ],
            total_premium="$18,935.00",
            countersignature="Continental Casualty Company",
            countersignature_date=shared.effective_date,
        )
    if doc_type == DC.ENDORSEMENT:
        return DocumentContent(
            title="Policy Change Endorsement",
            form_number="G-140148-A",
            endorsement_title="Additional Scheduled Auto — Amendment of Covered Autos",
            endorsement_number="1",
            effective_date="2026-08-15",
            change_description=(
                "Add 2025 Freightliner Cascadia (VIN 3AKJHHDR9PSNL1234) as a scheduled Covered "
                "Auto; Combined Single Limit $1,000,000 each accident."
            ),
            schedule_rows=[
                ScheduleRow(label="Year", value="2025"),
                ScheduleRow(label="Make & Model", value="Freightliner Cascadia"),
                ScheduleRow(label="VIN", value="3AKJHHDR9PSNL1234"),
                ScheduleRow(label="Gross Vehicle Weight", value="80,000 lbs"),
                ScheduleRow(label="Radius of Operation", value="500+ miles"),
                ScheduleRow(label="Cost New", value="$185,000"),
                ScheduleRow(label="Garaging Location", value="1450 Harbor Blvd, Oakland, CA 94607"),
                ScheduleRow(label="Coverages", value="Liability; Physical Damage (Comp & Collision)"),
            ],
            provisions=[
                "In consideration of the premium charged, it is hereby agreed and understood that the "
                "vehicle described in the Schedule above is added as a Covered Auto under the Business "
                "Auto Coverage Form (CA 00 01 11 20), effective on the endorsement effective date shown "
                "above.",
                "The Combined Single Limit for Bodily Injury and Property Damage Liability applicable to "
                "the added Covered Auto is $1,000,000 each accident. Physical Damage coverage applies with "
                "a $1,000 Comprehensive deductible and a $2,500 Collision deductible.",
                "Item Three (Schedule of Covered Autos You Own) of the Business Auto Declarations is "
                "amended to include the vehicle described herein, resulting in the additional premium shown "
                "below, calculated pro-rata for the remainder of the policy period.",
            ],
            additional_premium="$1,240.00",
        )
    # NEEDS_REVIEW: a structured, insurance-adjacent document that is NOT one of the
    # four modeled types (here a loss run / claims-history report). Insurance vocab,
    # but shaped so the classifier stays below the 0.5 gate on every modeled type.
    return DocumentContent(
        title="Loss Run Report",
        nr_kind="Loss Run Report",
        unrelated_subtitle="Claims Experience — Valued as of 06/30/2026",
        nr_header_rows=[
            ScheduleRow(label="Insured", value=shared.named_insured),
            ScheduleRow(label="Policy No.", value=pol),
            ScheduleRow(label="Line of Business", value="Commercial Auto / General Liability"),
            ScheduleRow(label="Valuation Date", value="06/30/2026"),
            ScheduleRow(label="Prepared By", value=shared.producer.agency if shared.producer else shared.carrier_name),
        ],
        nr_columns=["Date of Loss", "Claim No.", "Description", "Paid", "Reserve", "Status"],
        nr_rows=[
            TableRow(cells=["03/14/2025", "CLM-2025-00417", "Rear-end collision, unit 12", "$8,420", "$0", "Closed"]),
            TableRow(cells=["07/22/2025", "CLM-2025-01188", "Cargo water damage in transit", "$14,900", "$5,000", "Open"]),
            TableRow(cells=["11/03/2025", "CLM-2025-02043", "Slip-and-fall, loading dock", "$2,150", "$0", "Closed"]),
            TableRow(cells=["02/09/2026", "CLM-2026-00231", "Windshield / glass, unit 7", "$640", "$0", "Closed"]),
        ],
        unrelated_body=(
            "The following loss experience is provided for underwriting and renewal review. "
            "Amounts reflect incurred losses (paid plus reserves) as of the valuation date and "
            "are subject to change as open claims develop. This report is not a bill, a policy, "
            "or confirmation of coverage."
        ),
    )


def _fallback_scenario(skeleton: Skeleton) -> Scenario:
    shared = SharedFacts(
        named_insured="Northgate Logistics LLC",
        policy_number=None if skeleton.email_type == EC.NEW_SUBMISSION else "CPP-2291043",
        insured_address="1450 Harbor Blvd, Suite 200, Oakland, CA 94607",
        contact_name="Dana Whitfield",
        contact_email="dana.whitfield@northgatelog.com",
        contact_phone="(510) 555-0142",
        carrier_name="Continental Casualty Company",
        effective_date="2026-08-01",
        expiration_date="2027-08-01",
        coverage_limits=[
            LimitLine(coverage="General Liability - Each Occurrence", limit="$1,000,000"),
            LimitLine(coverage="General Liability - General Aggregate", limit="$2,000,000"),
            LimitLine(coverage="Automobile Liability - Combined Single Limit", limit="$1,000,000"),
        ],
        total_premium="$18,750",
        producer=Producer(
            agency="Meridian Risk Partners",
            address="88 Kearny Street, Suite 1400, San Francisco, CA 94108",
            contact_name="Priya Anand",
            phone="(415) 555-0188",
            fax="(415) 555-0190",
            email="certs@meridianrisk.com",
        ),
        insurers=[
            Insurer(letter="A", name="Continental Casualty Company", naic="20443"),
            Insurer(letter="B", name="Transportation Insurance Company", naic="20494"),
            Insurer(letter="C", name="Valley Forge Insurance Company", naic="20508"),
        ],
        business_description="Long-haul motor freight trucking and warehousing",
        entity_type="Limited Liability Company",
        fein="47-3921885",
        naic_carrier="20443",
    )
    email = GeneratedEmail(
        subject=_FALLBACK_SUBJECT[skeleton.email_type],
        body=_FALLBACK_BODY[skeleton.email_type],
        sender="Dana Whitfield <dana.whitfield@northgatelog.com>",
        recipient="intake@meridianrisk.com",
    )
    documents = [_fallback_doc(dt, shared) for dt in skeleton.flat_doc_types]
    return Scenario(shared=shared, email=email, documents=documents)
