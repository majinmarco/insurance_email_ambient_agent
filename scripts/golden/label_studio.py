"""Label Studio adapter for the golden dataset (MAR-9).

The canonical ground truth stays in ``tests/golden/cases/<id>/labels.json``. This module
projects those labels into `Label Studio`_'s import format so a curator can load the set,
**correct** the labels in the UI, export, and re-import the corrections back into
``labels.json`` losslessly.

What is editable in Label Studio here:

* email intent (``email_type``) — a single-choice control,
* email-body extraction fields — one editable JSON block,
* per-document type (``doc_type``) — a single-choice control per document,
* per-document extraction fields — one editable JSON block per document.

Boundaries / page spans / ``expected_*`` limitation flags are *derived* ground truth
(the smoke generator knows them exactly), so they are shown as read-only context and are
preserved verbatim on re-import rather than being hand-labeled.

Two entry points:

* :func:`case_to_task` / :func:`export_tasks` — build the import file (tasks with
  pre-annotations) + :func:`labeling_config` for the project's labeling interface.
* :func:`apply_annotations` — merge a Label-Studio-exported task's annotations back onto
  the original label dict (lossless round-trip when nothing was edited).

.. _Label Studio: https://labelstud.io/guide/predictions
"""

from __future__ import annotations

import base64
import io
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from pdfminer.high_level import extract_text

from insurance_email_agent.schemas import DocumentCategory, EmailCategory

GOLDEN_DIR = Path(__file__).resolve().parents[2] / "tests" / "golden"
CASES_DIR = GOLDEN_DIR / "cases"
LS_DIR = GOLDEN_DIR / "label_studio"

EMAIL_CHOICES = [e.value for e in EmailCategory]
DOC_CHOICES = [d.value for d in DocumentCategory]

MODEL_VERSION = "golden-derived"
_MAX_DOC_TEXT_CHARS = 6000  # keep tasks light; enough to judge type + key fields


# --------------------------------------------------------------------------- #
# Labeling interface (project config)
# --------------------------------------------------------------------------- #
def labeling_config() -> str:
    """The Label Studio ``<View>`` labeling config matching the exported tasks.

    Uses a ``<Repeater>`` over ``$documents`` so a single static config handles a variable
    number of documents per task.
    """
    email_choices = "\n      ".join(f'<Choice value="{c}"/>' for c in EMAIL_CHOICES)
    doc_choices = "\n          ".join(f'<Choice value="{c}"/>' for c in DOC_CHOICES)
    return f"""<View>
  <Header value="$case_id  ·  tier: $tier"/>

  <View style="background:#f8f9fb;padding:1em;border-radius:6px">
    <Header value="Email"/>
    <Text name="email_subject" value="$email_subject"/>
    <Text name="email_body" value="$email_body"/>
  </View>

  <Header value="Email intent"/>
  <Choices name="email_type" toName="email_body" choice="single" showInLine="true">
      {email_choices}
  </Choices>

  <Header value="Email extraction (edit JSON to correct)"/>
  <TextArea name="email_extraction" toName="email_body" rows="7"
            editable="true" maxSubmissions="1"/>

  <Header value="Documents"/>
  <Repeater on="$documents" indexFlag="{{{{idx}}}}">
    <View style="border:1px solid #dcdce3;border-radius:6px;padding:1em;margin-bottom:1em">
      <Text name="doc_file_{{{{idx}}}}" value="$documents[{{{{idx}}}}].filename"/>
      <Text name="doc_meta_{{{{idx}}}}" value="$documents[{{{{idx}}}}].meta"/>
      <Text name="doc_text_{{{{idx}}}}" value="$documents[{{{{idx}}}}].text"/>
      <Choices name="doc_type_{{{{idx}}}}" toName="doc_text_{{{{idx}}}}" choice="single" showInLine="true">
          {doc_choices}
      </Choices>
      <TextArea name="doc_extraction_{{{{idx}}}}" toName="doc_text_{{{{idx}}}}" rows="6"
                editable="true" maxSubmissions="1"/>
    </View>
  </Repeater>
</View>
"""


# --------------------------------------------------------------------------- #
# Export: labels -> Label Studio task (with pre-annotations)
# --------------------------------------------------------------------------- #
def _pdf_text(pdf_bytes: bytes, page_start: int, page_end: int) -> str:
    try:
        txt = extract_text(io.BytesIO(pdf_bytes), page_numbers=list(range(page_start, page_end + 1)))
    except Exception:  # noqa: BLE001 — never let text extraction break the export
        txt = ""
    return (txt or "").strip()


def _document_display_text(payload_email: dict, att_index: int, att_label: dict, doc_label: dict) -> str:
    """Human-readable source text for one document, from the frozen attachment bytes."""
    fmt = att_label["render_format"]
    raw = base64.b64decode(payload_email["attachments"][att_index]["content"])
    if fmt == "html":
        text = raw.decode("utf-8", "replace")
    elif fmt == "scanned_pdf":
        text = "[scanned image — no text layer; label from context]"
    else:
        text = _pdf_text(raw, doc_label["page_start"], doc_label["page_end"])
    return text[:_MAX_DOC_TEXT_CHARS]


def _flat_documents(labels: dict) -> list[tuple[int, int, dict, dict]]:
    """Flatten to ``(attachment_index, doc_index, attachment_label, doc_label)`` in payload
    order — the stable index the Repeater / control names use."""
    out = []
    for ai, att in enumerate(labels["attachments"]):
        for di, doc in enumerate(att["documents"]):
            out.append((ai, di, att, doc))
    return out


def _dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True)


def case_to_task(payload: dict, labels: dict) -> dict:
    """Build one Label Studio task (``data`` + pre-annotated ``predictions``) for a case."""
    email = payload["email"]
    flat = _flat_documents(labels)

    documents_data: list[dict] = []
    result: list[dict] = []

    # email-level pre-annotations
    result.append({
        "from_name": "email_type", "to_name": "email_body", "type": "choices",
        "value": {"choices": [labels["email"]["email_type"]]},
    })
    result.append({
        "from_name": "email_extraction", "to_name": "email_body", "type": "textarea",
        "value": {"text": [_dumps(labels["email"]["extraction"])]},
    })

    # per-document data + pre-annotations
    for idx, (ai, di, att, doc) in enumerate(flat):
        meta = (
            f"[{att['filename']}  ·  format={att['render_format']}  ·  "
            f"pages {doc['page_start']}–{doc['page_end']}  ·  "
            f"expected_collapse={att['expected_collapse']} expected_scanned={att['expected_scanned']}]"
        )
        documents_data.append({
            "filename": att["filename"],
            "meta": meta,
            "text": _document_display_text(email, ai, att, doc),
            "attachment_index": ai,
            "doc_index": di,
        })
        result.append({
            "from_name": f"doc_type_{idx}", "to_name": f"doc_text_{idx}", "type": "choices",
            "value": {"choices": [doc["doc_type"]]},
        })
        result.append({
            "from_name": f"doc_extraction_{idx}", "to_name": f"doc_text_{idx}", "type": "textarea",
            "value": {"text": [_dumps(doc["extraction"])]},
        })

    return {
        "data": {
            "case_id": labels["case_id"],
            "tier": labels["tier"],
            "email_subject": email.get("subject", ""),
            "email_body": email.get("body", ""),
            "documents": documents_data,
        },
        "predictions": [{"model_version": MODEL_VERSION, "result": result}],
    }


def export_tasks(cases: list[tuple[dict, dict]]) -> list[dict]:
    """``[(payload, labels), ...]`` -> list of Label Studio tasks."""
    return [case_to_task(payload, labels) for payload, labels in cases]


# --------------------------------------------------------------------------- #
# Import: Label Studio annotations -> corrected labels (round-trip)
# --------------------------------------------------------------------------- #
def _result_items(task: dict) -> list[dict]:
    """Prefer human annotations; fall back to predictions (unedited round-trip)."""
    anns = task.get("annotations") or []
    if anns:
        return anns[-1].get("result") or []
    preds = task.get("predictions") or []
    if preds:
        return preds[-1].get("result") or []
    return []


def _first_text(value: dict) -> str:
    txt = value.get("text") or [""]
    return txt[0] if txt else ""


def apply_annotations(task: dict, original_labels: dict) -> dict:
    """Overlay a Label-Studio task's (corrected) annotations onto ``original_labels``.

    Only the Label-Studio-editable fields (email type + extraction, per-document type +
    extraction) are overwritten; boundaries, page spans and ``expected_*`` flags are kept
    from ``original_labels``. Returns a new dict; the input is not mutated. Round-trips to
    an identical dict when the task carries only the derived pre-annotations.
    """
    corrected = deepcopy(original_labels)
    flat = _flat_documents(corrected)

    for item in _result_items(task):
        name = item.get("from_name", "")
        value = item.get("value") or {}
        if name == "email_type":
            choices = value.get("choices") or []
            if choices:
                corrected["email"]["email_type"] = choices[0]
        elif name == "email_extraction":
            corrected["email"]["extraction"] = json.loads(_first_text(value))
        elif name.startswith("doc_type_"):
            idx = int(name[len("doc_type_"):])
            choices = value.get("choices") or []
            if choices and idx < len(flat):
                ai, di, _, _ = flat[idx]
                corrected["attachments"][ai]["documents"][di]["doc_type"] = choices[0]
        elif name.startswith("doc_extraction_"):
            idx = int(name[len("doc_extraction_"):])
            if idx < len(flat):
                ai, di, _, _ = flat[idx]
                corrected["attachments"][ai]["documents"][di]["extraction"] = json.loads(_first_text(value))

    return corrected


# --------------------------------------------------------------------------- #
# Filesystem helpers + CLI: (re)generate the Label Studio import files from the
# committed fixtures, and re-import corrections back into labels.json.
# --------------------------------------------------------------------------- #
def _load_committed_pairs() -> list[tuple[str, dict, dict]]:
    """``(case_id, payload, labels)`` for every committed case, in manifest order."""
    manifest = json.loads((GOLDEN_DIR / "manifest.json").read_text())
    out = []
    for entry in manifest["cases"]:
        cid = entry["case_id"]
        payload = json.loads((CASES_DIR / cid / "input.json").read_text())
        labels = json.loads((CASES_DIR / cid / "labels.json").read_text())
        out.append((cid, payload, labels))
    return out


def write_import_files() -> Path:
    """Generate ``tests/golden/label_studio/{tasks.json,config.xml}`` from the fixtures."""
    LS_DIR.mkdir(parents=True, exist_ok=True)
    pairs = _load_committed_pairs()
    tasks = export_tasks([(p, l) for _, p, l in pairs])
    (LS_DIR / "tasks.json").write_text(json.dumps(tasks, indent=2) + "\n")
    (LS_DIR / "config.xml").write_text(labeling_config())
    return LS_DIR


def reimport_corrections(export_path: str) -> int:
    """Merge a Label Studio *export* (JSON list of annotated tasks) back into each case's
    ``labels.json``. Matches tasks to cases by ``data.case_id``. Returns the count updated."""
    exported = json.loads(Path(export_path).read_text())
    by_id = {t["data"]["case_id"]: t for t in exported if t.get("data", {}).get("case_id")}
    updated = 0
    for cid, _payload, labels in _load_committed_pairs():
        task = by_id.get(cid)
        if task is None:
            continue
        corrected = apply_annotations(task, labels)
        if corrected != labels:
            (CASES_DIR / cid / "labels.json").write_text(
                json.dumps(corrected, indent=2, default=str) + "\n"
            )
            updated += 1
    return updated


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="scripts.golden.label_studio", description=__doc__)
    ap.add_argument("--reimport", metavar="EXPORT_JSON", default=None,
                    help="merge a Label Studio export back into labels.json instead of exporting")
    args = ap.parse_args(argv)
    if args.reimport:
        n = reimport_corrections(args.reimport)
        print(f"re-imported corrections into {n} labels.json file(s)")
        return 0
    path = write_import_files()
    print(f"wrote {path}/tasks.json + config.xml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
