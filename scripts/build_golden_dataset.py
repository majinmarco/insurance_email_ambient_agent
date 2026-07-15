"""Build the frozen golden dataset (MAR-9).

Enumerates a *pinned* case matrix (tier x email-type x structure), generates each
scenario once through the existing smoke generator, freezes the exact rendered graph
payload, derives hand-correctable ground-truth labels, and writes committed fixtures
under ``tests/golden/``. The committed fixtures are authoritative — because rendered
PDFs are not byte-reproducible (reportlab timestamps) and the generation LLM runs at
temperature 0.7, regenerating would not reproduce them. This script is a one-time /
refresh tool, not run at eval time.

Usage:
    uv run python -m scripts.build_golden_dataset               # LLM content (needs OPENAI_API_KEY)
    uv run python -m scripts.build_golden_dataset --no-llm      # deterministic offline content
    uv run python -m scripts.build_golden_dataset --limit 2 --no-llm   # quick smoke of the build path

The builder never imports or runs the graph (generation + render only); it needs the
*generation* LLM key only when ``--no-llm`` is absent.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from insurance_email_agent.schemas import EmailCategory
from scripts.golden.labels import build_labels
from scripts.graph_smoke_test import personas
from scripts.graph_smoke_test import scenario as S
from scripts.graph_smoke_test.payload import build_email_payload
from scripts.graph_smoke_test.realism import PROFILES, Realism
from scripts.graph_smoke_test.scenario import Scenario, augment_scenario
from scripts.graph_smoke_test.taxonomy import Skeleton, build_skeleton

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "tests" / "golden"
CASES_DIR = OUT_DIR / "cases"


# --------------------------------------------------------------------------- #
# Pinned case matrix (~30). Explicit seeds so the set is reproducible & curated.
#   - 3 tiers x 5 email types baseline (attachments on) for classification coverage
#   - multi-doc bundles (max_docs=3) at clean & messy for boundary metrics + collapse
#   - email-only (attachments off) cases for classification-only signal
#   - scanned (image-only, no text layer) and HTML single-doc edge cases
# force_format targets attachment 0 (guaranteed single-doc via max_att=1/max_docs=1).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Case:
    id: str
    tier: str
    seed: int
    email_type: str
    attachments: str = "on"          # on | off | auto
    max_docs: int = 2
    max_att: int = 3
    force_format: str | None = None  # None | "scanned_pdf" | "html"


_ET = [e.value for e in EmailCategory]

CASES: list[Case] = [
    # --- baseline: every tier x every email type, with attachments -----------
    Case("clean-new_submission-a", "clean", 101, "new_submission"),
    Case("clean-renewal-a", "clean", 102, "renewal"),
    Case("clean-endorsement-a", "clean", 103, "endorsement"),
    Case("clean-claim_fnol-a", "clean", 104, "claim_fnol"),
    Case("clean-needs_review-a", "clean", 105, "needs_review"),
    Case("mixed-new_submission-a", "mixed", 201, "new_submission"),
    Case("mixed-renewal-a", "mixed", 202, "renewal"),
    Case("mixed-endorsement-a", "mixed", 203, "endorsement"),
    Case("mixed-claim_fnol-a", "mixed", 204, "claim_fnol"),
    Case("mixed-needs_review-a", "mixed", 205, "needs_review"),
    Case("messy-new_submission-a", "messy", 301, "new_submission"),
    Case("messy-renewal-a", "messy", 302, "renewal"),
    Case("messy-endorsement-a", "messy", 303, "endorsement"),
    Case("messy-claim_fnol-a", "messy", 304, "claim_fnol"),
    Case("messy-needs_review-a", "messy", 305, "needs_review"),
    # --- multi-doc bundles (boundary metrics; clean split vs messy collapse) --
    Case("clean-new_submission-multi", "clean", 11, "new_submission", max_docs=3),
    Case("clean-renewal-multi", "clean", 5, "renewal", max_docs=3),
    Case("mixed-new_submission-multi", "mixed", 11, "new_submission", max_docs=3),
    Case("messy-new_submission-multi", "messy", 11, "new_submission", max_docs=3),
    Case("messy-renewal-multi", "messy", 5, "renewal", max_docs=3),
    # --- email-only (no attachments): classification-only signal -------------
    Case("clean-claim_fnol-emailonly", "clean", 410, "claim_fnol", attachments="off"),
    Case("mixed-needs_review-emailonly", "mixed", 411, "needs_review", attachments="off"),
    Case("messy-renewal-emailonly", "messy", 412, "renewal", attachments="off"),
    Case("clean-new_submission-emailonly", "clean", 413, "new_submission", attachments="off"),
    Case("mixed-endorsement-emailonly", "mixed", 414, "endorsement", attachments="off"),
    # --- scanned (image-only PDF, no text layer): no-OCR honest-limit signal --
    Case("mixed-new_submission-scanned", "mixed", 501, "new_submission",
         max_docs=1, max_att=1, force_format="scanned_pdf"),
    Case("messy-claim_fnol-scanned", "messy", 502, "claim_fnol",
         max_docs=1, max_att=1, force_format="scanned_pdf"),
    Case("messy-renewal-scanned", "messy", 503, "renewal",
         max_docs=1, max_att=1, force_format="scanned_pdf"),
    # --- HTML single-doc (text-extractable non-PDF format) -------------------
    Case("mixed-renewal-html", "mixed", 601, "renewal",
         max_docs=1, max_att=1, force_format="html"),
    Case("messy-endorsement-html", "messy", 602, "endorsement",
         max_docs=1, max_att=1, force_format="html"),
]

#: Seed floor for auto-generated ``--extra`` cases, kept clear of the pinned seeds above
#: so an extra case never collides with a curated one.
_EXTRA_SEED_BASE = 100_000


def extra_cases(n: int, *, seed_base: int = _EXTRA_SEED_BASE) -> list[Case]:
    """Generate ``n`` additional cases beyond the pinned matrix, for more eval volume.

    Deterministic: the id/seed/tier/email-type of extra case *i* depend only on *i*, so the
    set is reproducible in shape (LLM prose still varies). Tiers rotate fastest and email
    types on a co-prime cycle, so coverage stays balanced across ``clean/mixed/messy`` x the
    five intents. All are attachment-bearing PDFs (the pinned set already covers the
    scanned/HTML/email-only edges)."""
    tiers = ("clean", "mixed", "messy")
    out: list[Case] = []
    for i in range(max(0, n)):
        tier = tiers[i % 3]
        email_type = _ET[i % len(_ET)]
        # Vary structure a little so extras aren't all identical shape: every 3rd is a
        # multi-doc bundle (more boundary-metric material).
        max_docs = 3 if i % 3 == 2 else 2
        out.append(Case(
            id=f"extra-{i:03d}-{tier}-{email_type}",
            tier=tier, seed=seed_base + i, email_type=email_type,
            attachments="on", max_docs=max_docs,
        ))
    return out


# --------------------------------------------------------------------------- #
# Content generation with an explicit persona (so identities vary across ALL
# tiers, including clean where the smoke path would default to one identity).
# --------------------------------------------------------------------------- #
def generate_with_persona(
    skeleton: Skeleton, persona: "personas.Persona | None", *, use_llm: bool
) -> tuple[Scenario, str]:
    """Return ``(scenario, content_source)`` where content_source is 'llm' or 'fallback'.

    Mirrors ``scenario.generate_scenario`` but takes an explicit persona instead of
    reading it off the (possibly absent) noise plan.
    """
    n_docs = len(skeleton.flat_doc_types)
    if not use_llm:
        return S._fallback_scenario(skeleton, persona=persona), "fallback"

    prompt = S._build_prompt(skeleton, persona=persona)
    structured = S._make_llm("openai", None, 0.7).with_structured_output(Scenario)
    for _ in range(2):
        try:
            sc = structured.invoke([("system", S._SYSTEM), ("human", prompt)])
            if len(sc.documents) == n_docs:
                return sc, "llm"
        except Exception as exc:  # noqa: BLE001 — any LLM/transport failure -> fallback
            print(f"    [scenario] LLM generation failed ({exc!r}); retrying/falling back")
    return S._fallback_scenario(skeleton, persona=persona), "fallback"


def _force_single_doc_format(skeleton: Skeleton, fmt: str) -> None:
    """Override attachment 0 (single-doc) to a scanned/HTML render format and recompute
    its expected-limitation flags. Guarantees the curated edge case exists deterministically
    rather than relying on the probabilistic draw."""
    noise = skeleton.noise
    assert noise is not None, "scanned/html cases must be a non-clean tier (noise present)"
    assert len(skeleton.attachments[0].docs) == 1, "force_format requires a single-doc attachment 0"
    dn = noise.docs[0]
    dn.render_format = fmt
    dn.is_scanned = fmt == "scanned_pdf"
    exp = noise.expected
    if exp.per_attachment_scanned:
        exp.per_attachment_scanned[0] = fmt == "scanned_pdf"
    if exp.per_attachment_expect_interrupt:
        exp.per_attachment_expect_interrupt[0] = exp.per_attachment_expect_interrupt[0] or (fmt == "scanned_pdf")
    if fmt == "html" and noise.attachments:
        fn = noise.attachments[0].filename
        if not fn.endswith(".html"):
            noise.attachments[0].filename = fn.rsplit(".", 1)[0] + ".html"


# --------------------------------------------------------------------------- #
# Build one case
# --------------------------------------------------------------------------- #
def build_case(case: Case, case_index: int, *, use_llm: bool) -> dict:
    realism = Realism(
        tier=case.tier,
        seed=case.seed,
        profile=PROFILES[case.tier],
        scanned_enabled=case.force_format == "scanned_pdf",
        nonpdf_enabled=case.force_format == "html",
    )
    rng = random.Random(case.seed)
    skeleton = build_skeleton(
        rng,
        email_type=EmailCategory(case.email_type),
        attachments_mode=case.attachments,
        max_attachments=case.max_att,
        max_docs_per_attachment=case.max_docs,
        realism=realism,
    )
    if case.force_format:
        _force_single_doc_format(skeleton, case.force_format)

    # Force identity variety across all tiers: use the drawn persona when there is a
    # noise plan, else a deterministic per-case persona so clean cases aren't all one insured.
    persona_index = (
        skeleton.noise.email.persona_index
        if skeleton.noise
        else case_index % len(personas.POOL)
    )
    persona = personas.get(persona_index)

    scenario, content_source = generate_with_persona(skeleton, persona, use_llm=use_llm)
    scenario = augment_scenario(scenario, skeleton, realism)

    payload = build_email_payload(skeleton, scenario, realism=realism)
    payload["email"]["id"] = case.id  # deterministic id for a stable frozen fixture

    labels = build_labels(case.id, case.tier, skeleton, scenario)

    provenance = {
        "case_id": case.id,
        "tier": case.tier,
        "seed": case.seed,
        "email_type": case.email_type,
        "attachments_mode": case.attachments,
        "max_docs_per_attachment": case.max_docs,
        "max_attachments": case.max_att,
        "force_format": case.force_format,
        "persona_index": persona_index,
        "content_source": content_source,
        "noise_manifest": asdict(skeleton.noise) if skeleton.noise else None,
        "raw_scenario": scenario.model_dump(mode="json"),
    }

    case_dir = CASES_DIR / case.id
    case_dir.mkdir(parents=True, exist_ok=True)
    _write_json(case_dir / "input.json", payload)
    _write_json(case_dir / "labels.json", labels)
    _write_json(case_dir / "provenance.json", provenance)

    doc_types = [[d.value for d in a.docs] for a in skeleton.attachments]
    return {
        "case_id": case.id,
        "tier": case.tier,
        "seed": case.seed,
        "email_type": case.email_type,
        "n_attachments": len(skeleton.attachments),
        "attachment_doc_types": doc_types,
        "has_multidoc": any(len(a.docs) > 1 for a in skeleton.attachments),
        "render_formats": [a["render_format"] for a in labels["attachments"]],
        "content_source": content_source,
        "dir": f"cases/{case.id}",
    }


def _manifest_entry_from_disk(case: Case) -> dict:
    """Rebuild a manifest entry for an already-built (cached) case, without regenerating."""
    case_dir = CASES_DIR / case.id
    labels = json.loads((case_dir / "labels.json").read_text())
    prov = json.loads((case_dir / "provenance.json").read_text())
    doc_types = [[d["doc_type"] for d in a["documents"]] for a in labels["attachments"]]
    return {
        "case_id": case.id,
        "tier": case.tier,
        "seed": case.seed,
        "email_type": case.email_type,
        "n_attachments": len(labels["attachments"]),
        "attachment_doc_types": doc_types,
        "has_multidoc": any(len(a["documents"]) > 1 for a in labels["attachments"]),
        "render_formats": [a["render_format"] for a in labels["attachments"]],
        "content_source": prov.get("content_source", "unknown"),
        "dir": f"cases/{case.id}",
    }


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return None


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:  # noqa: BLE001
        pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="scripts.build_golden_dataset", description=__doc__)
    ap.add_argument("--no-llm", action="store_true", help="deterministic offline content (no API call)")
    ap.add_argument("--limit", type=int, default=None, help="only build the first N cases (smoke)")
    ap.add_argument("--extra", type=int, default=0, metavar="N",
                    help="append N auto-generated cases beyond the pinned matrix (more eval volume)")
    ap.add_argument("--clean", action="store_true", help="wipe tests/golden/cases before building")
    args = ap.parse_args(argv)
    _load_env()

    use_llm = not args.no_llm
    if use_llm and not os.getenv("OPENAI_API_KEY"):
        print("! OPENAI_API_KEY not set; falling back to --no-llm deterministic content.")
        use_llm = False

    all_cases = list(CASES) + extra_cases(args.extra)
    cases = all_cases[: args.limit] if args.limit else all_cases
    if args.clean and CASES_DIR.exists():
        shutil.rmtree(CASES_DIR)
    CASES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"building {len(cases)} case(s)  content={'llm' if use_llm else 'fallback'}", flush=True)
    entries: list[dict] = []
    for i, case in enumerate(cases):
        case_dir = CASES_DIR / case.id
        complete = all((case_dir / f).exists() for f in ("input.json", "labels.json", "provenance.json"))
        if complete and not args.clean:
            # Resume: reuse the already-built case (content is per-case independent).
            entries.append(_manifest_entry_from_disk(case))
            print(f"== [{i + 1}/{len(cases)}] {case.id}  (cached, skipped)", flush=True)
            continue
        print(f"== [{i + 1}/{len(cases)}] {case.id}  (tier={case.tier} seed={case.seed} type={case.email_type})", flush=True)
        entry = build_case(case, i, use_llm=use_llm)
        print(f"   attachments={entry['attachment_doc_types'] or '(none)'}  formats={entry['render_formats']}  content={entry['content_source']}", flush=True)
        entries.append(entry)

    manifest = {
        "schema_version": 1,
        "builder_git_sha": _git_sha(),
        "content_source": "llm" if use_llm else "fallback",
        "n_cases": len(entries),
        "email_categories": _ET,
        "cases": entries,
    }
    _write_json(OUT_DIR / "manifest.json", manifest)

    # Keep the Label Studio import files in sync with the fixtures on every build, so the
    # set can always be loaded into Label Studio for hand-correction.
    if not args.limit:
        from scripts.golden.label_studio import write_import_files

        ls_dir = write_import_files()
        print(f"== wrote Label Studio import files -> {ls_dir}/")

    tiers = {}
    for e in entries:
        tiers[e["tier"]] = tiers.get(e["tier"], 0) + 1
    print(f"== wrote {len(entries)} cases -> {OUT_DIR}  by-tier={tiers}  multidoc={sum(e['has_multidoc'] for e in entries)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
