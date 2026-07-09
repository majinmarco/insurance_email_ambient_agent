# graph_smoke_test

A smoke-test harness for the `insurance_email_agent` LangGraph graph. It randomly
generates insurance emails (one of the five `EmailCategory` types) with relevant
reportlab-rendered PDF attachments (one or more `DocumentCategory` types, several
optionally bundled into a single PDF), fires them at the graph running under
`uv run langgraph dev`, and prints a lenient expected-vs-actual summary. Runs land
in LangGraph Studio.

## How it works

1. `taxonomy.py` — `random` (seedable) picks the email type + attachment structure;
   doc-type weights are biased by email type so attachments stay plausible.
2. `scenario.py` — one structured-output LLM call fills a coherent `Scenario`
   (shared facts + email + per-doc field values). `--no-llm` uses a deterministic
   template instead (offline, free). Doc types come from the skeleton, not the LLM.
3. `render.py` — reportlab templates, one per doc type. Multiple docs are
   concatenated into one PDF (page breaks between them). Each doc opens with a
   literal `# TITLE` line so MarkItDown's PDF→markdown keeps a header the graph's
   `MarkdownHeaderTextSplitter` can split multi-doc PDFs on.
4. `payload.py` — base64-encodes each PDF (JSON has no bytes type) and builds the
   `{"email": {...}}` input. Filenames end in `.pdf` (the graph derives MIME type
   from the extension).
5. `client_run.py` — `get_sync_client` → create a thread → `runs.wait` (or stream
   with `--verbose`). Prints a Studio URL per run.
6. `verify.py` — checks email type, attachment count, and that each non-`needs_review`
   doc type appears in at least one segment. Mismatches are findings, not crashes.

## Prerequisites

- `uv add reportlab` (already added).
- The graph must be running: `uv run langgraph dev` (serves at `http://127.0.0.1:2024`).
- `OPENAI_API_KEY` in `./.env` (auto-loaded, same as the server). For Claude
  generation instead: `uv add langchain-anthropic`, set `ANTHROPIC_API_KEY`, pass
  `--provider anthropic`.
- Requires the four graph patches (B1–B4) described in the top-level plan; without
  them attachment runs crash and no-attachment runs fail with `KeyError: '__end__'`.

## Usage

```bash
# 0-attachment run only (cheap gate: proves the graph + models resolve)
uv run python -m scripts.graph_smoke_test --n 1 --attachments off

# mixed random scenarios (seeded), verified
uv run python -m scripts.graph_smoke_test --n 5 --seed 42

# force a type, force attachments, stream node-by-node, keep the PDFs
uv run python -m scripts.graph_smoke_test --email-type renewal --attachments on --verbose --keep-pdfs ./out

# fully offline generation (no generation API cost; graph still calls its own LLM)
uv run python -m scripts.graph_smoke_test --n 3 --no-llm
```

Key flags: `--n`, `--seed`, `--url`, `--assistant-id`, `--attachments {on,off,auto}`,
`--email-type`, `--max-docs`, `--provider {openai,anthropic}`, `--model`, `--no-llm`,
`--verbose`, `--no-verify`, `--no-smoke`, `--keep-pdfs DIR`. Override the generation
model with `SCENARIO_LLM_MODEL` (default `gpt-5.4-mini`).

> If `langgraph dev` reports port 2024 in use and picks another port, pass
> `--url http://127.0.0.1:<port>` (shown in the dev-server startup banner).
