"""Dispatch a smoke-test payload to the graph and normalize the outcome.

``run_graph`` delegates to ``handler.runner.run`` — the same entry point the poller
uses — running the graph **in-process** (``graph.invoke``/``graph.stream``) or, with
``remote=True``, against a running ``uv run langgraph dev`` server over the SDK. It
returns ``(thread_id, result, error)`` so the caller stays agnostic to the mode.
"""

from __future__ import annotations

from typing import Any


def _extract_error(result: Any) -> str | None:
    """Return an error string if the run failed, else None."""
    if result is None:
        return "no result returned"
    if isinstance(result, dict):
        # A completed run of this graph always has a classification.
        if "classification" not in result:
            return f"unexpected result shape (keys={list(result.keys())})"
    return None


def run_graph(
    payload: dict[str, dict],
    *,
    verbose: bool = False,
    remote: bool = False,
    auto_resume: bool = True,
) -> tuple[str | None, Any, str | None]:
    # Lazy import: keeps ``--help`` fast and lets ``main()`` load ``.env`` before
    # ``graph.py`` constructs its models at import time.
    from insurance_email_agent.handler import runner

    email = payload["email"]
    result = runner.run(email, local=not remote, stream=verbose, auto_resume=auto_resume)
    return email.get("id"), result, _extract_error(result)
