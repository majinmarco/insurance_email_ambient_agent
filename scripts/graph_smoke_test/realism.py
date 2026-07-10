"""Seeded realism overlay for the smoke test.

Real production emails/attachments are far messier than the pristine, ACORD-perfect
PDFs this harness generated originally — most importantly, real PDFs contain no
markdown ``#`` header, yet the graph's *only* way to split a bundled PDF into
separate documents is to split on single ``#`` H1 lines (``graph.py`` 86-89, 255).
Generation planting that delimiter is teaching-to-the-test.

This module adds a third, *seeded* layer between the two kinds of randomness that
already exist implicitly:

  * **structure** — email type, attachment/doc counts (already seeded in
    ``taxonomy.build_skeleton``);
  * **noise** (this module) — a deterministic :class:`NoisePlan` recorded on the
    ``Skeleton`` and consumed by scenario/render/payload/verify;
  * **content** — LLM prose (unseeded, temp 0.7) or the deterministic fallback.

``clean`` is a no-op: ``build_skeleton`` never calls :meth:`Realism.draw_noise_plan`
for it, so the RNG stream and rendered PDFs are byte-identical to the pre-realism
pipeline. ``mixed`` injects realism the graph can still handle (drops ``#`` only on
single-doc PDFs, which stay one chunk regardless). ``messy`` drops ``#`` everywhere
and turns up the noise, so multi-doc bundles collapse — the honest signal that the
graph's segmentation depends on a delimiter reality does not provide.

Determinism: every decision is drawn from a *path-addressed child RNG*
(:meth:`Realism.rng`), so decisions are order-independent — adding a new noise
dimension never shifts the values of existing ones, and concurrency in the graph
cannot perturb generation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from insurance_email_agent.schemas import DocumentCategory

if TYPE_CHECKING:
    from .taxonomy import AttachmentPlan, Skeleton

DC = DocumentCategory

TIERS = ("clean", "mixed", "messy")

#: Keep in sync with ``len(personas.POOL)``; ``draw_noise_plan`` only stores an index
#: so this module never imports ``personas`` (avoids a cycle and keeps it self-contained).
PERSONA_POOL_SIZE = 10

# Delimiter modes understood by ``render._title_flowable``.
DELIM_HASH = "hash"        # "# TITLE" — the splittable boundary (today's behavior)
DELIM_BOLD_CAPS = "bold_caps"
DELIM_PLAIN = "plain"
_NONHASH_MODES = (DELIM_BOLD_CAPS, DELIM_PLAIN, DELIM_BOLD_CAPS)  # weighted toward ACORD-style caps

_WATERMARKS = ("DRAFT", "COPY", "SPECIMEN", "DUPLICATE")
_SUBJECT_STYLES = ("allcaps", "typo", "ticket", "urgent", "empty")
_SCANNY = ("scan{n:04d}", "IMG_{n:04d}", "Scan_{n:04d}", "20260708{n:04d}")
_PACKET_STEMS = ("Submission Packet", "Insurance Documents", "Policy Documents", "Renewal Package")
_DOCTYPE_STEM = {
    DC.CERTIFICATE: "ACORD 25 - Certificate of Insurance",
    DC.INVOICE: "Premium Invoice",
    DC.DECLARATIONS: "Policy Declarations",
    DC.ENDORSEMENT: "Endorsement Request",
    DC.NEEDS_REVIEW: "Loss Run Report",
}


@dataclass(frozen=True)
class Profile:
    """Per-tier probabilities/switches. All default to the ``clean`` (zero) row."""

    # email overlay
    thread_p: float = 0.0
    signature_p: float = 0.0
    subject_noise_p: float = 0.0
    multi_recipient_p: float = 0.0
    backdate_p: float = 0.0
    body_only_p: float = 0.0
    inconsistency_p: float = 0.0
    # attachment / filename
    filename_style_p: float = 0.0        # P(realistic name instead of attachment_N.pdf)
    filename_corrupt_ext_p: float = 0.0  # P(wrong/missing ext | realistic name) — messy only
    # per-document
    layout_variant_p: float = 0.0
    watermark_p: float = 0.0
    content_noise_p: float = 0.0
    content_noise_level: float = 0.0     # intensity 0..1 handed to render/scenario
    # delimiter behavior (the core knob)
    drop_delim_singledoc: bool = False   # drop "#" on single-doc PDFs (safe: still one chunk)
    drop_delim_multidoc: bool = False    # drop "#" on multi-doc PDFs (→ segmentation collapse)
    # deferred track (only honored when the corresponding CLI flag is set)
    scanned_p: float = 0.0
    nonpdf_p: float = 0.0


PROFILES: dict[str, Profile] = {
    "clean": Profile(),
    "mixed": Profile(
        thread_p=0.35, signature_p=0.5, subject_noise_p=0.4, multi_recipient_p=0.25,
        backdate_p=1.0, body_only_p=0.0, inconsistency_p=0.15,
        filename_style_p=1.0, filename_corrupt_ext_p=0.0,
        layout_variant_p=0.4, watermark_p=0.2,
        content_noise_p=0.25, content_noise_level=0.15,
        drop_delim_singledoc=True, drop_delim_multidoc=False,
        scanned_p=0.0, nonpdf_p=0.0,
    ),
    "messy": Profile(
        thread_p=0.6, signature_p=0.7, subject_noise_p=0.7, multi_recipient_p=0.5,
        backdate_p=1.0, body_only_p=0.25, inconsistency_p=0.35,
        filename_style_p=1.0, filename_corrupt_ext_p=0.3,
        layout_variant_p=0.8, watermark_p=0.5,
        content_noise_p=0.6, content_noise_level=0.35,
        drop_delim_singledoc=True, drop_delim_multidoc=True,
        scanned_p=0.4, nonpdf_p=0.3,
    ),
}


# --------------------------------------------------------------------------- #
# Recorded plan (the ground truth for what was injected)
# --------------------------------------------------------------------------- #
@dataclass
class DocNoise:
    """Per-document (flat, in ``skeleton.flat_doc_types`` order) injected noise."""

    delimiter_mode: str = DELIM_HASH
    layout_variant: int = 0
    watermark: str | None = None
    content_noise: bool = False
    content_noise_level: float = 0.0
    is_scanned: bool = False
    #: "pdf" (default), "scanned_pdf" (image-only, no text layer), or "html".
    #: Non-PDF/scanned are restricted to single-doc attachments (deferred track).
    render_format: str = "pdf"


@dataclass
class AttachmentNoise:
    filename: str


@dataclass
class EmailNoise:
    persona_index: int | None = None
    thread: bool = False
    signature: bool = False
    subject_style: str | None = None      # one of _SUBJECT_STYLES, or None
    multi_recipient: bool = False
    body_only: bool = False
    inconsistency: bool = False
    date_received: str | None = None      # concrete ISO string when backdated; None → payload uses now()


@dataclass
class ExpectedOutcome:
    """What verify should treat as *expected* under this tier (so realism that the
    graph legitimately cannot handle surfaces as findings, not false failures)."""

    email_type_hard: bool = True                                  # False under body_only/inconsistency
    per_attachment_collapse: list[bool] = field(default_factory=list)        # multi-doc + dropped "#"
    per_attachment_scanned: list[bool] = field(default_factory=list)         # image-only, no text layer
    per_attachment_expect_interrupt: list[bool] = field(default_factory=list)  # needs_review or scanned


@dataclass
class NoisePlan:
    email: EmailNoise
    attachments: list[AttachmentNoise]
    docs: list[DocNoise]
    expected: ExpectedOutcome


# --------------------------------------------------------------------------- #
# The overlay engine
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Realism:
    tier: str
    seed: int
    profile: Profile
    delimiter_override: str = "auto"   # "auto" | "keep" | "drop"
    scanned_enabled: bool = False
    nonpdf_enabled: bool = False

    @property
    def active(self) -> bool:
        return self.tier != "clean"

    @classmethod
    def resolve(cls, args) -> "Realism":
        """Build a :class:`Realism` from parsed CLI args, resolving an *effective*
        seed even when ``--seed`` is omitted so every run is reproducible/printable."""
        tier = (getattr(args, "realism", None) or "clean")
        if tier not in PROFILES:
            tier = "clean"
        seed = getattr(args, "seed", None)
        if seed is None:
            seed = random.Random().randrange(2**31)
        return cls(
            tier=tier,
            seed=int(seed),
            profile=PROFILES[tier],
            delimiter_override=(getattr(args, "delimiter", None) or "auto"),
            scanned_enabled=bool(getattr(args, "scanned", False)),
            nonpdf_enabled=bool(getattr(args, "nonpdf", False)),
        )

    def rng(self, *path) -> random.Random:
        """A deterministic child RNG addressed by ``path``. String-seeded so it is
        reproducible across processes (``random`` hashes str seeds with SHA-512, not
        the salted builtin ``hash``)."""
        return random.Random("::".join(str(x) for x in (self.seed, *path)))

    def _hit(self, prob: float, *path) -> bool:
        if prob <= 0.0:
            return False
        if prob >= 1.0:
            return True
        return self.rng(*path).random() < prob

    # -- individual dimension draws (each from its own child RNG → additive) -- #
    def _choose_delimiter(self, is_multidoc: bool, *path) -> str:
        if self.delimiter_override == "keep":
            return DELIM_HASH
        r = self.rng(*path)
        if self.delimiter_override == "drop":
            return r.choice(_NONHASH_MODES)
        drop = self.profile.drop_delim_multidoc if is_multidoc else self.profile.drop_delim_singledoc
        return r.choice(_NONHASH_MODES) if drop else DELIM_HASH

    def _draw_watermark(self, *path) -> str | None:
        r = self.rng(*path)
        return r.choice(_WATERMARKS) if r.random() < self.profile.watermark_p else None

    def _draw_layout(self, *path) -> int:
        r = self.rng(*path)
        return (1 + r.randrange(2)) if r.random() < self.profile.layout_variant_p else 0

    def _draw_subject_style(self) -> str | None:
        r = self.rng("email", "subject")
        return r.choice(_SUBJECT_STYLES) if r.random() < self.profile.subject_noise_p else None

    def _draw_date(self) -> str | None:
        r = self.rng("email", "date")
        if not (r.random() < self.profile.backdate_p):
            return None
        # Backdate relative to today's midnight (UTC) with a seeded offset: reproducible
        # within a day, and never stale (clean already uses wall-clock now()).
        base = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        days_ago = r.randrange(1, 22)
        into_day = timedelta(hours=r.randrange(7, 19), minutes=r.randrange(0, 60))
        return (base - timedelta(days=days_ago) + into_day).isoformat()

    def _choose_format(self, is_multidoc: bool, flat_i: int) -> str:
        """Per-doc render format. Scanned/non-PDF are single-doc only (a bundle is a
        born-digital packet); both are gated by their CLI flag."""
        if is_multidoc:
            return "pdf"
        if self.scanned_enabled and self._hit(self.profile.scanned_p, "doc", flat_i, "scan"):
            return "scanned_pdf"
        if self.nonpdf_enabled and self._hit(self.profile.nonpdf_p, "doc", flat_i, "nonpdf"):
            return "html"
        return "pdf"

    def _draw_filename(self, plan: "AttachmentPlan", fmt: str, *path) -> str:
        a_i = path[-1]
        r = self.rng(*path)
        if not (r.random() < self.profile.filename_style_p):
            return f"attachment_{a_i + 1}{'.html' if fmt == 'html' else '.pdf'}"
        if r.random() < 0.4:
            stem = r.choice(_SCANNY).format(n=r.randrange(1, 9000))
        elif len(plan.docs) > 1:
            stem = r.choice(_PACKET_STEMS)
        else:
            stem = _DOCTYPE_STEM.get(plan.docs[0], "Document")
            if r.random() < 0.3:
                stem = f"{stem} ({r.randrange(1, 4)})"
        if fmt == "html":
            return f"{stem}.html"
        ext = ".pdf"
        # Visually-"wrong" but MIME-safe (splitext still yields '.pdf', so MarkItDown
        # still parses). Truly-broken extensions (missing/.bin) cause hard MarkItDown
        # errors that bypass verify — intentionally out of scope.
        if r.random() < self.profile.filename_corrupt_ext_p:
            ext = ".pdf.pdf"
        return f"{stem}{ext}"

    def draw_noise_plan(self, skeleton: "Skeleton") -> NoisePlan:
        """Draw the full deterministic overlay for ``skeleton``. Only called for
        active (non-clean) tiers, after all structural draws are complete."""
        email = EmailNoise(
            persona_index=self.rng("persona").randrange(PERSONA_POOL_SIZE),
            thread=self._hit(self.profile.thread_p, "email", "thread"),
            signature=self._hit(self.profile.signature_p, "email", "signature"),
            subject_style=self._draw_subject_style(),
            multi_recipient=self._hit(self.profile.multi_recipient_p, "email", "recipient"),
            body_only=self._hit(self.profile.body_only_p, "email", "bodyonly"),
            inconsistency=self._hit(self.profile.inconsistency_p, "email", "inconsistency"),
            date_received=self._draw_date(),
        )

        attachments: list[AttachmentNoise] = []
        docs: list[DocNoise] = []
        per_att_collapse: list[bool] = []
        per_att_scanned: list[bool] = []
        per_att_interrupt: list[bool] = []
        flat_i = 0
        for a_i, att in enumerate(skeleton.attachments):
            is_multidoc = len(att.docs) > 1
            att_docs: list[DocNoise] = []
            dropped_any = False
            scanned_any = False
            for cat in att.docs:
                fmt = self._choose_format(is_multidoc, flat_i)
                mode = self._choose_delimiter(is_multidoc, "doc", flat_i, "delim")
                dn = DocNoise(
                    delimiter_mode=mode,
                    layout_variant=self._draw_layout("doc", flat_i, "layout"),
                    watermark=self._draw_watermark("doc", flat_i, "wm"),
                    content_noise=(cat != DC.NEEDS_REVIEW)
                    and self._hit(self.profile.content_noise_p, "doc", flat_i, "cnoise"),
                    content_noise_level=self.profile.content_noise_level,
                    is_scanned=(fmt == "scanned_pdf"),
                    render_format=fmt,
                )
                att_docs.append(dn)
                dropped_any = dropped_any or (mode != DELIM_HASH)
                scanned_any = scanned_any or dn.is_scanned
                flat_i += 1
            docs.extend(att_docs)
            fmt0 = att_docs[0].render_format if len(att_docs) == 1 else "pdf"
            attachments.append(AttachmentNoise(filename=self._draw_filename(att, fmt0, "file", a_i)))
            per_att_collapse.append(is_multidoc and dropped_any)
            per_att_scanned.append(scanned_any)
            per_att_interrupt.append(any(c == DC.NEEDS_REVIEW for c in att.docs) or scanned_any)

        expected = ExpectedOutcome(
            email_type_hard=not (email.body_only or email.inconsistency),
            per_attachment_collapse=per_att_collapse,
            per_attachment_scanned=per_att_scanned,
            per_attachment_expect_interrupt=per_att_interrupt,
        )
        return NoisePlan(email=email, attachments=attachments, docs=docs, expected=expected)
