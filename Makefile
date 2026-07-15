.PHONY: golden golden-offline golden-labelstudio golden-reimport golden-pdfs golden-test

# Golden dataset for evals (MAR-9). See tests/golden/README.md.

# Rebuild the frozen fixtures with LLM content (needs OPENAI_API_KEY; resumable).
golden:
	uv run python -m scripts.build_golden_dataset

# Rebuild with deterministic offline content (no API key).
golden-offline:
	uv run python -m scripts.build_golden_dataset --no-llm

# Dump frozen PDFs, wire local-file serving, and launch Label Studio for review.
golden-labelstudio:
	uv run python -m scripts.golden.label_studio --serve

# Merge a Label Studio JSON export back into labels.json:  make golden-reimport EXPORT=export.json
golden-reimport:
	uv run python -m scripts.golden.label_studio --reimport "$(EXPORT)"

# Decode every case's frozen attachments to files for viewing in any PDF viewer.
golden-pdfs:
	uv run python -m scripts.golden.label_studio --dump-pdfs

# Run the golden-fixture integrity suite.
golden-test:
	uv run pytest tests/golden
