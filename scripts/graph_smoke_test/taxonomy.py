"""Random *structural* choices for a test scenario.

We keep randomness here (seedable via a ``random.Random``) and let the LLM only
fill in field *values* later — so a run is reproducible in shape even when the
generated prose varies. Doc-type weights are biased by the email type so the
attachments stay plausible for the email (e.g. renewals carry declarations /
invoices, endorsement requests carry an endorsement doc).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from insurance_email_agent.schemas import DocumentCategory, EmailCategory

if TYPE_CHECKING:
    from .realism import NoisePlan, Realism

EC = EmailCategory
DC = DocumentCategory

# (doc type, weight) per email type — governs which doc types tend to appear.
_DOC_WEIGHTS: dict[EmailCategory, list[tuple[DocumentCategory, float]]] = {
    EC.NEW_SUBMISSION: [
        (DC.CERTIFICATE, 3), (DC.DECLARATIONS, 3), (DC.INVOICE, 1),
        (DC.ENDORSEMENT, 0.5), (DC.NEEDS_REVIEW, 0.5),
    ],
    EC.RENEWAL: [
        (DC.DECLARATIONS, 3), (DC.INVOICE, 3), (DC.CERTIFICATE, 1),
        (DC.ENDORSEMENT, 0.5), (DC.NEEDS_REVIEW, 0.5),
    ],
    EC.ENDORSEMENT: [
        (DC.ENDORSEMENT, 4), (DC.DECLARATIONS, 1), (DC.INVOICE, 0.5),
        (DC.CERTIFICATE, 0.5), (DC.NEEDS_REVIEW, 0.5),
    ],
    EC.CLAIM_FNOL: [
        (DC.NEEDS_REVIEW, 2), (DC.CERTIFICATE, 1), (DC.DECLARATIONS, 1),
        (DC.INVOICE, 0.5), (DC.ENDORSEMENT, 0.5),
    ],
    EC.NEEDS_REVIEW: [
        (DC.CERTIFICATE, 1), (DC.DECLARATIONS, 1), (DC.INVOICE, 1),
        (DC.ENDORSEMENT, 1), (DC.NEEDS_REVIEW, 2),
    ],
}


@dataclass
class AttachmentPlan:
    """One PDF file, bundling ``docs`` distinct documents in order."""

    docs: list[DocumentCategory]


@dataclass
class Skeleton:
    """The structural blueprint for one test scenario (no content yet)."""

    email_type: EmailCategory
    attachments: list[AttachmentPlan]
    #: Seeded realism overlay (personas, delimiter modes, filenames, expected outcomes).
    #: ``None`` at the ``clean`` tier — the structural blueprint alone is ground truth.
    noise: "NoisePlan | None" = None

    @property
    def flat_doc_types(self) -> list[DocumentCategory]:
        """All doc types across all attachments, in payload order."""
        return [d for a in self.attachments for d in a.docs]

    @property
    def expected_segment_types(self) -> list[DocumentCategory]:
        """Doc types we expect the graph to emit a *segment* for.

        ``needs_review`` docs never produce a segment (the graph skips them at
        graph.py:271), so they are excluded from the expected set.
        """
        return [d for d in self.flat_doc_types if d != DC.NEEDS_REVIEW]


def _weighted_pick(rng: random.Random, weights) -> DocumentCategory:
    cats = [c for c, _ in weights]
    w = [x for _, x in weights]
    return rng.choices(cats, weights=w, k=1)[0]


def build_skeleton(
    rng: random.Random,
    email_type: EmailCategory | None = None,
    attachments_mode: str = "auto",
    max_attachments: int = 3,
    max_docs_per_attachment: int = 2,
    realism: "Realism | None" = None,
) -> Skeleton:
    """Draw a random scenario shape.

    ``attachments_mode``: ``"off"`` → 0 attachments, ``"on"`` → 1..N,
    ``"auto"`` → ~30% get 0 else 1..N.

    All *structural* draws happen first (unchanged order), so the ``clean`` tier's
    RNG stream is identical to the pre-realism pipeline. The seeded noise overlay is
    then drawn from ``realism``'s own path-addressed RNGs (never off ``rng``), so it
    never perturbs structure and is a strict no-op for ``clean``.
    """
    if email_type is None:
        email_type = rng.choice(list(EmailCategory))

    if attachments_mode == "off":
        n_att = 0
    elif attachments_mode == "on":
        n_att = rng.randint(1, max_attachments)
    else:  # auto
        n_att = 0 if rng.random() < 0.3 else rng.randint(1, max_attachments)

    weights = _DOC_WEIGHTS[email_type]
    attachments: list[AttachmentPlan] = []
    for _ in range(n_att):
        n_docs = rng.randint(1, max_docs_per_attachment)
        docs = [_weighted_pick(rng, weights) for _ in range(n_docs)]
        attachments.append(AttachmentPlan(docs=docs))

    skeleton = Skeleton(email_type=email_type, attachments=attachments)
    if realism is not None and realism.active:
        skeleton.noise = realism.draw_noise_plan(skeleton)
    return skeleton
