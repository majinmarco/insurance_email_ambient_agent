"""Smoke-test harness for the `insurance_email_agent` LangGraph dev server.

Generates random insurance emails (one of the five ``EmailCategory`` types) with
relevant reportlab-rendered PDF attachments (one or more ``DocumentCategory``
types, optionally several bundled into a single PDF), fires them at the graph
running under ``uv run langgraph dev``, and prints a lenient expected-vs-actual
summary. Runs land in LangGraph Studio for inspection.

Run with:  ``uv run python -m scripts.graph_smoke_test --help``
"""
