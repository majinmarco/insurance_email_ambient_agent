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

### Generating more than 30 cases

Two ways to grow the set:

- **Auto-generate extra volume** — append `N` cases beyond the pinned matrix, balanced across
  tiers × email types (every 3rd a multi-doc bundle), with deterministic ids/seeds
  (`extra-000-…`, seeds from `100000`):

  ```
  make golden EXTRA=20                                        # 30 pinned + 20 extra (LLM)
  make golden-offline EXTRA=50                                # + 50 extra, offline
  uv run python -m scripts.build_golden_dataset --extra 20    # raw equivalent
  ```

  The build is **resumable** — existing cases are skipped, so raising `EXTRA` only builds the
  new ones. (Lowering it leaves the old `extra-*` dirs on disk but drops them from
  `manifest.json`, so the loader ignores them; `--clean` removes them.)

- **Add curated cases** — edit the `CASES` list in `scripts/build_golden_dataset.py` (pin a
  specific tier/seed/email-type/structure, e.g. more scanned or multi-doc edges), then rebuild.

Either way the extra fixtures carry the same labels + provenance and load through the same
`load_cases()` loader; run `uv run pytest tests/golden` to validate them.

## Hand-correction

Labels are machine-derived from the generator's ground truth, so they start correct — but
they are meant to be audited and edited by hand (that's why `labels.json` is kept
separate from the base64-heavy `input.json`). Pay closest attention to:

- `email.extraction` free-text/optional fields (`requested_effective_date`,
  `requested_changes`) — populated from intent, not parsed from the exact body text.
- extraction dict fields (`limits`, `coverages`) — best-effort flattening of the source.

Edit `labels.json` directly, **or** use Label Studio (below). After any edit,
`uv run pytest tests/golden` re-checks structural invariants.

### Label Studio round-trip (with rendered PDFs)

The set projects into a Label Studio import in `label_studio/`:

- `label_studio/config.xml` (committed) — the project's labeling interface. Email intent +
  per-document type as single-choice controls; email/document extraction as editable JSON
  blocks; **each document's rendered PDF shown inline** via a `<HyperText>` viewer; the
  extracted text tucked in a collapsible panel. A `<Repeater>` handles the variable document
  count. Paste it into *Project → Settings → Labeling Interface*.
- `label_studio/tasks.json` (generated, git-ignored) — one task per case, current labels
  pre-loaded as `predictions` so you correct pre-annotations instead of labeling from scratch.

**Fastest path — the launcher** (dumps PDFs, wires local-file serving, starts Label Studio):

```
make golden-labelstudio          # = uv run python -m scripts.golden.label_studio --serve
```

It prints the one-time project setup (paste `config.xml`; add a **Local files** source =
the printed document root; import `tasks.json`). Then every case shows its real PDF next to
the labels — scanned cases show the image pages, so you can eyeball the source against the
ground truth.

**Correcting & merging back.** Correct labels in the UI → *Export* as JSON → merge:

```
make golden-reimport EXPORT=path/to/export.json
# = uv run python -m scripts.golden.label_studio --reimport path/to/export.json
```

Re-import is **lossless** and only overwrites the editable fields (intent, doc type,
extraction); boundaries, page spans and `expected_*` flags are preserved.

**PDF display modes** (`--pdf-mode`, default `localfiles`):

- `localfiles` — PDFs served from disk (`--serve` sets this up). Reliable rendering; tiny
  `tasks.json`. This is what the launcher uses.
- `embed` — PDFs travel inside `tasks.json` as base64 `data:` URIs; zero setup, works on a
  plain import, but Label Studio's HTML sanitizer may block it in some versions.
- `none` — text-only review (no PDF viewer).

**Other commands:**

```
uv run python -m scripts.golden.label_studio                 # regenerate config.xml + tasks.json
uv run python -m scripts.golden.label_studio --pdf-mode embed  # self-contained tasks.json
uv run python -m scripts.golden.label_studio --dump-pdfs DIR    # decode frozen PDFs to DIR/<case>/<file>
make golden-pdfs                                             # dump PDFs for viewing in any viewer
```

[MAR-10]: https://linear.app/marco-nardone-guerra/issue/MAR-10
