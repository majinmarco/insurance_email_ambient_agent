"""Assemble the graph input payload from a skeleton + generated scenario.

Attachment content is base64-encoded (JSON has no bytes type; the graph's
patched segmentation node decodes it). Filenames end in ``.pdf`` because the
graph derives the MIME type from the extension.
"""

from __future__ import annotations

import base64
import uuid
from datetime import datetime, timezone

from .render import build_html, build_pdf, build_scanned_pdf
from .scenario import Scenario
from .taxonomy import Skeleton


def build_email_payload(skeleton: Skeleton, scenario: Scenario, realism=None) -> dict:
    """Return ``{"email": {...}}`` ready for ``client.runs.wait(input=...)``.

    The seeded realism overlay (``skeleton.noise``) drives realistic filenames, the
    per-document ``# TITLE`` delimiter mode, and a backdated ``date_received``. When
    there is no overlay (``clean`` tier) every field falls back to today's values, so
    the payload is byte-identical to the pre-realism pipeline.
    """
    typed = list(zip(skeleton.flat_doc_types, scenario.documents))
    noise = skeleton.noise

    attachments = []
    cursor = 0
    for a_i, plan in enumerate(skeleton.attachments):
        n = len(plan.docs)
        group = typed[cursor : cursor + n]
        dslice = noise.docs[cursor : cursor + n] if noise else None
        cursor += n
        # Deferred single-doc formats (off unless --scanned/--nonpdf) override the PDF path.
        fmt = dslice[0].render_format if (dslice and n == 1) else "pdf"
        if fmt == "html":
            content_bytes = build_html(scenario.shared, *group[0])
        elif fmt == "scanned_pdf":
            content_bytes = build_scanned_pdf(scenario.shared, *group[0])
        else:
            doc_modes = [d.delimiter_mode for d in dslice] if dslice else None
            watermarks = [d.watermark for d in dslice] if dslice else None
            layouts = [d.layout_variant for d in dslice] if dslice else None
            content_bytes = build_pdf(scenario.shared, group, doc_modes=doc_modes, watermarks=watermarks, layouts=layouts)
        filename = noise.attachments[a_i].filename if noise else f"attachment_{a_i + 1}.pdf"
        attachments.append(
            {
                "filename": filename,
                "content": base64.b64encode(content_bytes).decode("ascii"),
                "segments": [],
            }
        )

    date_received = (
        noise.email.date_received
        if (noise and noise.email.date_received)
        else datetime.now(timezone.utc).isoformat()
    )
    email = {
        "id": str(uuid.uuid4()),
        "subject": scenario.email.subject,
        # ISO string, never a datetime object — orjson in the SDK can't serialize
        # arbitrary datetimes, and the graph only string-interpolates this field.
        "date_received": date_received,
        "body": scenario.email.body,
        "sender": scenario.email.sender,
        "recipient": scenario.email.recipient,
        "attachments": attachments,
    }
    return {"email": email}


def decode_attachments(payload: dict) -> list[tuple[str, bytes]]:
    """Decode a payload's attachments back to ``(filename, pdf_bytes)`` pairs."""
    out = []
    for att in payload["email"]["attachments"]:
        out.append((att["filename"], base64.b64decode(att["content"])))
    return out
