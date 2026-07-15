# Golden dataset (frozen, labeled) — MAR-9

Frozen, hand-correctable ground truth for the eval suite ([MAR-10]). Each case pairs the
**exact graph input** with **labels** for every metric MAR-10 measures — email
classification, per-type document classification, segmentation boundaries, and field
extraction — sliced by the `clean` / `mixed` / `messy` realism tiers.

## Layout

```
tests/golden/
  manifest.json                 # index of all cases + builder git sha + content source
  cases/<case_id>/
    input.json                  # frozen {"email": {...}} graph payload (base64 attachments) — feed to the graph
    labels.json                 # ground truth (hand-correctable)
    provenance.json             # seed, tier, noise manifest, raw generated scenario
  loader.py                     # deterministic loader + validated pydantic models
  test_fixtures_integrity.py    # guardrails (not the eval suite)
```

## Loading (MAR-10)

```python
from tests.golden.loader import load_cases
from insurance_email_agent.handler import runner

for case in load_cases(tier="messy"):        # or load_cases() for all
    result = runner.run(case.email, local=True)   # drive the graph on the frozen input
    # compare result vs case.labels (email_type, per-attachment docs, boundaries, extraction)
```

`load_cases()` reads and **validates** the committed JSON (via `GoldenLabels`); a
malformed or mis-edited label fails at load time rather than skewing metrics.

## What the labels contain

`labels.json` (`GoldenLabels`):

- `email.email_type` — ground-truth classification; `email_type_hard` is `False` for
  intentionally ambiguous emails (body-only / body-vs-attachment mismatch), where any
  classification is defensible — score those softly.
- `email.extraction` — ground-truth `EmailExtraction` fields (from the body).
- `attachments[]`:
  - `documents[]` — the true ordered doc-type sequence, each with its `page_start` /
    `page_end` span and per-document `extraction` ground truth (`None` for
    `needs_review`, which the graph never extracts).
  - `boundary_page_offsets` — internal document cut points (page indices), the reference
    segmentation for **boundary-F1 / WindowDiff / Pk**.
  - `expected_segment_types` — physical docs minus `needs_review` (what the graph should
    emit a segment for).
  - `expected_collapse` / `expected_scanned` / `expected_interrupt` — honest
    graph-limitation flags (dropped `#` boundary merges a multi-doc bundle; a scanned
    image-only PDF has no text layer for the OCR-less graph to read). Use these so a
    limitation the graph legitimately can't handle isn't scored as a model error.

**Boundary granularity is the PDF page** — chosen because it's graph-independent
(`build_pdf` inserts a `PageBreak` between documents, so per-document pagination is
exact and the per-doc page spans tile the frozen bundle exactly). MAR-10 maps the graph's
`Segment.chunk_indices` onto these page spans to score segmentation.

## Reproducibility & why fixtures are frozen

The **committed fixtures are authoritative.** Rendered PDFs are *not* byte-reproducible
(reportlab embeds timestamps/IDs) and the generation LLM runs at temperature 0.7, so
re-running the builder produces *different* bytes — which is exactly why the rendered
payloads are frozen rather than regenerated at eval time. `provenance.json` records the
seed, the full noise manifest, and the raw generated `Scenario` for audit and to let
labels be re-derived without re-calling the LLM.

## Rebuilding / refreshing

```
uv run python -m scripts.build_golden_dataset            # LLM content (needs OPENAI_API_KEY)
uv run python -m scripts.build_golden_dataset --no-llm   # deterministic offline content
uv run python -m scripts.build_golden_dataset --limit 2 --no-llm   # quick smoke of the build path
```

The pinned case matrix lives in `scripts/build_golden_dataset.py` (`CASES`). Labels are
derived by `scripts/golden/labels.py` from the generator's known structure. Rebuilding
overwrites `cases/` and `manifest.json`; **re-run the hand-correction audit** afterward,
since new content changes the derived values.

## Hand-correction

Labels are machine-derived from the generator's ground truth, so they start correct — but
they are meant to be audited and edited by hand (that's why `labels.json` is kept
separate from the base64-heavy `input.json`). Pay closest attention to:

- `email.extraction` free-text/optional fields (`requested_effective_date`,
  `requested_changes`) — populated from intent, not parsed from the exact body text.
- extraction dict fields (`limits`, `coverages`) — best-effort flattening of the source.

Edit `labels.json` directly, **or** use Label Studio (below). After any edit,
`uv run pytest tests/golden` re-checks structural invariants.

### Label Studio round-trip

The set ships as a Label Studio import in `label_studio/`, regenerated on every build:

- `label_studio/config.xml` — the project's labeling interface (email intent + per-document
  type as single-choice controls; email/document extraction as editable JSON blocks; a
  `<Repeater>` handles the variable document count). Paste it into
  *Project → Settings → Labeling Interface*.
- `label_studio/tasks.json` — one task per case, with the current labels pre-loaded as
  `predictions` so you correct pre-annotations instead of labeling from scratch. Each task
  shows the email and the per-document source text (extracted from the frozen attachment).

Workflow: create a project with `config.xml`, import `tasks.json`, correct, then
*Export* as JSON and merge the corrections back into `labels.json`:

```
uv run python -m scripts.golden.label_studio --reimport path/to/export.json
```

Re-import is **lossless** and only overwrites the Label-Studio-editable fields (intent,
doc type, extraction); boundaries, page spans and `expected_*` flags are preserved. Run
`uv run python -m scripts.golden.label_studio` on its own to regenerate the import files
from the current fixtures without rebuilding.

[MAR-10]: https://linear.app/marco-nardone-guerra/issue/MAR-10
