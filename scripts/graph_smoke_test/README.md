# graph_smoke_test

A smoke-test harness for the `insurance_email_agent` LangGraph graph. It randomly
generates insurance emails (one of the five `EmailCategory` types) with relevant
reportlab-rendered PDF attachments (one or more `DocumentCategory` types, several
optionally bundled into a single PDF), runs them **through the same handler the
production poller uses** (`handler/runner.py`), and prints a lenient
expected-vs-actual summary.

By default the graph runs **in-process** — no server required. Pass `--remote` to
instead drive a running `uv run langgraph dev` server over the LangGraph SDK (so runs
show up in LangGraph Studio).

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
   from the extension). The in-process runner accepts the base64 string directly.
5. `client_run.py` — a thin dispatcher over `handler/runner.py::run`. In-process it
   calls the compiled graph (`graph.invoke`, or `graph.stream` with `--verbose`);
   with `--remote` it drives `langgraph dev` over the SDK. Returns
   `(thread_id, result, error)`.
6. `verify.py` — checks email type, attachment count, and that each non-`needs_review`
   doc type appears in at least one segment. Mismatches are findings, not crashes.

### Human-in-the-loop interrupts

A low-confidence attachment chunk (zero-shot score < 0.5) is classified
`needs_review`, which raises a LangGraph `interrupt` (agent-inbox schema). By default
the runner **auto-resolves** each interrupt with the model's best guess so unattended
runs complete. Pass `--manual-review` to hand the decision to a human instead — a
console prompt in-process, or the **Agent Inbox / Studio** when `--remote` (open the
printed `studio:` URL to accept/edit/ignore, and the run resumes on the server).

## Prerequisites

- `uv add reportlab` (already added).
- `OPENAI_API_KEY` in `./.env` (auto-loaded). The graph's own LLMs run **in this
  process**, so the key is required even with `--no-llm` (that flag only skips the
  *generation* LLM, not the graph). For Claude generation instead: `uv add
  langchain-anthropic`, set `ANTHROPIC_API_KEY`, pass `--provider anthropic`.
- The first in-process run loads the HuggingFace zero-shot model
  (`MoritzLaurer/deberta-v3-large-zeroshot-v2.0`) once per process; set `DEVICE` to
  pick CPU/GPU.
- **Only for `--remote`:** the graph must be running via `uv run langgraph dev`
  (serves `http://127.0.0.1:2024`). Point elsewhere with `LANGGRAPH_URL`.

## Usage

```bash
# 0-attachment run only (cheap gate: proves the graph + models resolve)
uv run python -m scripts.graph_smoke_test --n 1 --attachments off

# mixed random scenarios (seeded), verified, in-process
uv run python -m scripts.graph_smoke_test --n 5 --seed 42

# force a type, force attachments, stream node-by-node, keep the PDFs
uv run python -m scripts.graph_smoke_test --email-type renewal --attachments on --verbose --keep-pdfs ./out

# run against a langgraph dev server (visible in Studio)
uv run langgraph dev            # in another terminal
uv run python -m scripts.graph_smoke_test --n 3 --remote

# hand NEEDS_REVIEW interrupts to a human instead of auto-resolving
uv run python -m scripts.graph_smoke_test --email-type claim_fnol --attachments on --manual-review

# fully offline generation (no generation API cost; the graph still calls its own LLM)
uv run python -m scripts.graph_smoke_test --n 3 --no-llm
```

Key flags: `--n`, `--seed`, `--attachments {on,off,auto}`, `--email-type`,
`--max-docs`, `--provider {openai,anthropic}`, `--model`, `--no-llm`, `--verbose`,
`--remote`, `--manual-review`, `--no-verify`, `--no-smoke`, `--keep-pdfs DIR`,
`--realism {clean,mixed,messy}`, `--results-dir DIR`. Override the generation model
with `SCENARIO_LLM_MODEL` (default `gpt-5.4-mini`); point `--remote` runs at a
different server with `LANGGRAPH_URL`.

> `--remote` only: if `langgraph dev` reports port 2024 in use and picks another
> port (shown in its startup banner), set `LANGGRAPH_URL=http://127.0.0.1:<port>`.
