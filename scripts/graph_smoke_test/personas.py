"""A rotating pool of realistic US commercial P&C insured identities.

The original harness reused a single hardcoded insured ("Northgate Logistics") on
the ``--no-llm`` path, and the LLM path invented a fresh identity each run with no
control. This pool gives deterministic, seeded identity rotation: ``realism`` records
a ``persona_index`` on the skeleton; ``scenario`` builds the offline ``SharedFacts``
from it and nudges the LLM prompt with it, so email + attachments stay coherent while
the *who* varies across runs.

A static, hand-written table (no ``faker`` dependency) — 10 distinct industries,
entity types, regions, carriers, and producers. ``len(POOL)`` must equal
``realism.PERSONA_POOL_SIZE``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Persona:
    named_insured: str
    entity_type: str
    business_description: str
    insured_address: str
    contact_name: str
    contact_email: str
    contact_phone: str
    carrier_name: str
    naic_carrier: str
    policy_number: str            # used when the email is not a brand-new submission
    fein: str
    total_premium: str
    effective_date: str           # ISO YYYY-MM-DD
    expiration_date: str
    producer_agency: str
    producer_address: str
    producer_contact: str
    producer_phone: str
    producer_email: str
    #: (letter, name, NAIC) rows for the ACORD "INSURER(S) AFFORDING COVERAGE" table.
    insurers: tuple[tuple[str, str, str], ...]

    @property
    def intake_recipient(self) -> str:
        return self.producer_email

    def to_shared_facts(self, *, new_submission: bool):
        """Build a ``scenario.SharedFacts`` (lazy import avoids an import cycle)."""
        from .scenario import Insurer, LimitLine, Producer, SharedFacts

        return SharedFacts(
            named_insured=self.named_insured,
            policy_number=None if new_submission else self.policy_number,
            insured_address=self.insured_address,
            contact_name=self.contact_name,
            contact_email=self.contact_email,
            contact_phone=self.contact_phone,
            carrier_name=self.carrier_name,
            effective_date=self.effective_date,
            expiration_date=self.expiration_date,
            coverage_limits=[
                LimitLine(coverage="General Liability - Each Occurrence", limit="$1,000,000"),
                LimitLine(coverage="General Liability - General Aggregate", limit="$2,000,000"),
                LimitLine(coverage="Automobile Liability - Combined Single Limit", limit="$1,000,000"),
            ],
            total_premium=self.total_premium,
            producer=Producer(
                agency=self.producer_agency,
                address=self.producer_address,
                contact_name=self.producer_contact,
                phone=self.producer_phone,
                email=self.producer_email,
            ),
            insurers=[Insurer(letter=l, name=n, naic=naic) for (l, n, naic) in self.insurers],
            business_description=self.business_description,
            entity_type=self.entity_type,
            fein=self.fein,
            naic_carrier=self.naic_carrier,
        )


POOL: list[Persona] = [
    Persona(
        named_insured="Riverside Manufacturing Co.",
        entity_type="Corporation",
        business_description="Sheet-metal fabrication and industrial machining",
        insured_address="2200 Foundry Road, Cleveland, OH 44115",
        contact_name="Karen Delgado",
        contact_email="kdelgado@riversidemfg.com",
        contact_phone="(216) 555-0173",
        carrier_name="The Hartford",
        naic_carrier="19682",
        policy_number="SBA-4471902",
        fein="34-2918765",
        total_premium="$42,300",
        effective_date="2026-09-01",
        expiration_date="2027-09-01",
        producer_agency="Great Lakes Insurance Advisors",
        producer_address="410 Euclid Ave, Suite 900, Cleveland, OH 44114",
        producer_contact="Tom Bransford",
        producer_phone="(216) 555-0110",
        producer_email="submissions@greatlakesins.com",
        insurers=(("A", "Hartford Fire Insurance Company", "19682"),
                  ("B", "Twin City Fire Insurance Company", "29459"),
                  ("C", "Ohio Casualty Insurance Company", "24074")),
    ),
    Persona(
        named_insured="Blue Harbor Restaurant Group LLC",
        entity_type="Limited Liability Company",
        business_description="Full-service seafood restaurants (4 locations)",
        insured_address="18 Commercial Wharf, Boston, MA 02110",
        contact_name="Marco Bianchi",
        contact_email="marco@blueharborgroup.com",
        contact_phone="(617) 555-0146",
        carrier_name="Liberty Mutual Insurance",
        naic_carrier="23043",
        policy_number="BOP-7789341",
        fein="04-3781220",
        total_premium="$27,900",
        effective_date="2026-08-15",
        expiration_date="2027-08-15",
        producer_agency="Beacon Hill Risk Partners",
        producer_address="75 State Street, Boston, MA 02109",
        producer_contact="Alison Pierce",
        producer_phone="(617) 555-0102",
        producer_email="intake@beaconhillrisk.com",
        insurers=(("A", "Liberty Mutual Fire Insurance Company", "23035"),
                  ("B", "Ohio Security Insurance Company", "24082")),
    ),
    Persona(
        named_insured="Summit Ridge Construction Inc.",
        entity_type="Corporation",
        business_description="Commercial general contractor (ground-up)",
        insured_address="9400 E Arapahoe Rd, Centennial, CO 80112",
        contact_name="Derek Olsson",
        contact_email="dolsson@summitridgegc.com",
        contact_phone="(303) 555-0188",
        carrier_name="Travelers Indemnity Company",
        naic_carrier="25658",
        policy_number="CPP-6612078",
        fein="84-2201947",
        total_premium="$88,150",
        effective_date="2026-10-01",
        expiration_date="2027-10-01",
        producer_agency="Front Range Commercial Insurance",
        producer_address="1550 Market St, Denver, CO 80202",
        producer_contact="Rachel Nguyen",
        producer_phone="(303) 555-0120",
        producer_email="newbusiness@frontrangeins.com",
        insurers=(("A", "Travelers Property Casualty Company of America", "25674"),
                  ("B", "Charter Oak Fire Insurance Company", "25615"),
                  ("C", "Travelers Indemnity Company of Connecticut", "25682")),
    ),
    Persona(
        named_insured="Evergreen Home Health Services LLC",
        entity_type="Limited Liability Company",
        business_description="In-home nursing and personal care services",
        insured_address="1201 SW Fifth Ave, Portland, OR 97204",
        contact_name="Priya Raman",
        contact_email="praman@evergreenhh.com",
        contact_phone="(503) 555-0159",
        carrier_name="CNA Financial",
        naic_carrier="20443",
        policy_number="HMP-3320815",
        fein="93-4417802",
        total_premium="$51,600",
        effective_date="2026-08-01",
        expiration_date="2027-08-01",
        producer_agency="Cascade Professional Insurance",
        producer_address="805 SW Broadway, Portland, OR 97205",
        producer_contact="Kevin Doyle",
        producer_phone="(503) 555-0134",
        producer_email="submissions@cascadepro.com",
        insurers=(("A", "Continental Casualty Company", "20443"),
                  ("B", "Valley Forge Insurance Company", "20508")),
    ),
    Persona(
        named_insured="Lone Star Freight Systems LP",
        entity_type="Limited Partnership",
        business_description="Regional less-than-truckload motor carrier",
        insured_address="4500 Irving Blvd, Dallas, TX 75247",
        contact_name="Wade Carrington",
        contact_email="wcarrington@lonestarfreight.com",
        contact_phone="(214) 555-0191",
        carrier_name="Great West Casualty Company",
        naic_carrier="11371",
        policy_number="CA-9948120",
        fein="75-3392014",
        total_premium="$134,700",
        effective_date="2026-11-01",
        expiration_date="2027-11-01",
        producer_agency="Southwest Transportation Insurance",
        producer_address="600 N Pearl St, Dallas, TX 75201",
        producer_contact="Monica Salas",
        producer_phone="(214) 555-0117",
        producer_email="trucking@swtransins.com",
        insurers=(("A", "Great West Casualty Company", "11371"),
                  ("B", "Zurich American Insurance Company", "16535")),
    ),
    Persona(
        named_insured="Pinnacle Staffing Solutions Inc.",
        entity_type="Corporation",
        business_description="Temporary clerical and light-industrial staffing",
        insured_address="303 Peachtree St NE, Atlanta, GA 30308",
        contact_name="Danielle Foster",
        contact_email="dfoster@pinnaclestaffing.com",
        contact_phone="(404) 555-0163",
        carrier_name="AmTrust Financial",
        naic_carrier="30210",
        policy_number="WC-5580372",
        fein="58-2790143",
        total_premium="$63,400",
        effective_date="2026-09-15",
        expiration_date="2027-09-15",
        producer_agency="Peachtree Commercial Brokers",
        producer_address="1100 Peachtree St NE, Atlanta, GA 30309",
        producer_contact="Brian Whitaker",
        producer_phone="(404) 555-0128",
        producer_email="accounts@peachtreebrokers.com",
        insurers=(("A", "Technology Insurance Company", "42376"),
                  ("B", "Wesco Insurance Company", "25011")),
    ),
    Persona(
        named_insured="Coastal Property Management LLC",
        entity_type="Limited Liability Company",
        business_description="Commercial real estate management (retail centers)",
        insured_address="500 N Westshore Blvd, Tampa, FL 33609",
        contact_name="Hector Ramos",
        contact_email="hramos@coastalpm.com",
        contact_phone="(813) 555-0177",
        carrier_name="Nationwide Mutual Insurance Company",
        naic_carrier="23787",
        policy_number="CPP-2214569",
        fein="59-3810276",
        total_premium="$71,250",
        effective_date="2026-08-01",
        expiration_date="2027-08-01",
        producer_agency="Gulf Coast Insurance Group",
        producer_address="401 E Jackson St, Tampa, FL 33602",
        producer_contact="Sandra Lee",
        producer_phone="(813) 555-0139",
        producer_email="submissions@gulfcoastig.com",
        insurers=(("A", "Nationwide Mutual Insurance Company", "23787"),
                  ("B", "Scottsdale Insurance Company", "41297")),
    ),
    Persona(
        named_insured="Ironclad Security Services Inc.",
        entity_type="Corporation",
        business_description="Contract security guard services",
        insured_address="2801 N Central Ave, Phoenix, AZ 85004",
        contact_name="Gloria Tran",
        contact_email="gtran@ironcladsecurity.com",
        contact_phone="(602) 555-0154",
        carrier_name="Berkshire Hathaway GUARD",
        naic_carrier="20044",
        policy_number="GL-4471083",
        fein="86-1129055",
        total_premium="$38,800",
        effective_date="2026-10-15",
        expiration_date="2027-10-15",
        producer_agency="Desert Sky Insurance Advisors",
        producer_address="2 N Central Ave, Phoenix, AZ 85004",
        producer_contact="Manuel Ortega",
        producer_phone="(602) 555-0123",
        producer_email="newbiz@desertskyins.com",
        insurers=(("A", "AmGUARD Insurance Company", "42390"),
                  ("B", "EastGUARD Insurance Company", "14702")),
    ),
    Persona(
        named_insured="Golden Gate Hospitality Group LLC",
        entity_type="Limited Liability Company",
        business_description="Boutique hotel operator (3 properties)",
        insured_address="655 Montgomery St, San Francisco, CA 94111",
        contact_name="Olivia Bennett",
        contact_email="obennett@gghospitality.com",
        contact_phone="(415) 555-0169",
        carrier_name="Chubb Group",
        naic_carrier="20281",
        policy_number="CPP-8830471",
        fein="94-3729108",
        total_premium="$96,900",
        effective_date="2026-12-01",
        expiration_date="2027-12-01",
        producer_agency="Pacific Union Insurance Services",
        producer_address="50 California St, San Francisco, CA 94111",
        producer_contact="Nathan Cole",
        producer_phone="(415) 555-0141",
        producer_email="submissions@pacunionins.com",
        insurers=(("A", "Federal Insurance Company", "20281"),
                  ("B", "Great Northern Insurance Company", "20303"),
                  ("C", "Vigilant Insurance Company", "20397")),
    ),
    Persona(
        named_insured="Midwest AgriSupply Cooperative",
        entity_type="Cooperative",
        business_description="Farm supply, grain handling, and agronomy services",
        insured_address="1750 NE Broadway Ave, Des Moines, IA 50313",
        contact_name="Roger Halvorsen",
        contact_email="rhalvorsen@midwestagri.coop",
        contact_phone="(515) 555-0182",
        carrier_name="Nationwide Agribusiness",
        naic_carrier="26093",
        policy_number="FARM-1120934",
        fein="42-1980553",
        total_premium="$58,700",
        effective_date="2026-09-01",
        expiration_date="2027-09-01",
        producer_agency="Prairie States Insurance Cooperative",
        producer_address="2900 University Ave, West Des Moines, IA 50266",
        producer_contact="Julie Andersen",
        producer_phone="(515) 555-0115",
        producer_email="commercial@prairiestatesins.com",
        insurers=(("A", "Nationwide Agribusiness Insurance Company", "26093"),
                  ("B", "Farmland Mutual Insurance Company", "13897")),
    ),
]


def get(index: int | None) -> Persona | None:
    if index is None:
        return None
    return POOL[index % len(POOL)]
