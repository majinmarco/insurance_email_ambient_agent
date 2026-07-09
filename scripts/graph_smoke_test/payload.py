"""Assemble the graph input payload from a skeleton + generated scenario.

Attachment content is base64-encoded (JSON has no bytes type; the graph's
patched segmentation node decodes it). Filenames end in ``.pdf`` because the
graph derives the MIME type from the extension.
"""

from __future__ import annotations

import base64
import uuid
from datetime import datetime, timezone

from .render import build_pdf
from .scenario import Scenario
from .taxonomy import Skeleton


def build_email_payload(skeleton: Skeleton, scenario: Scenario) -> dict:
    """Return ``{"email": {...}}`` ready for ``client.runs.wait(input=...)``."""
    typed = list(zip(skeleton.flat_doc_types, scenario.documents))

    attachments = []
    cursor = 0
    for a_i, plan in enumerate(skeleton.attachments):
        n = len(plan.docs)
        group = typed[cursor : cursor + n]
        cursor += n
        pdf_bytes = build_pdf(scenario.shared, group)
        attachments.append(
            {
                "filename": f"attachment_{a_i + 1}.pdf",
                "content": base64.b64encode(pdf_bytes).decode("ascii"),
                "segments": [],
            }
        )

    email = {
        "id": str(uuid.uuid4()),
        "subject": scenario.email.subject,
        # ISO string, never a datetime object — orjson in the SDK can't serialize
        # arbitrary datetimes, and the graph only string-interpolates this field.
        "date_received": datetime.now(timezone.utc).isoformat(),
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
