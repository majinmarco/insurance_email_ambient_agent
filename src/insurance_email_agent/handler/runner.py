"""
Executed by poller/webhook, calls graph
"""

import os
from typing import Any

from insurance_email_agent.schemas import Email, HumanResponse
from insurance_email_agent.states import OverallState
from insurance_email_agent.graph import build_local_graph
from uuid import uuid4
from langgraph.types import Command

# In-process graph with its own checkpointer/store, used by ``_drive_local``.
# The API-hosted ``graph`` in graph.py has no checkpointer (the platform supplies
# persistence), so it can't drive the interrupt/resume loop in-process.
graph = build_local_graph()

URL = os.getenv("LANGGRAPH_URL", "http://127.0.0.1:2024")
ASSISTANT_ID = "insurance_email_agent"
# Bound on resume rounds so a stuck/looping interrupt can't hang a run. Each
# round consumes one pending NEEDS_REVIEW interrupt, so this caps how many
# low-confidence chunks a single email may auto-resolve.
MAX_RESUME_ROUNDS = 25

def make_client(url: str):
    # Lazy import so the poller/webhook don't pull in the SDK unless remote mode is used.
    from langgraph_sdk import get_sync_client
    return get_sync_client(url=url)

def studio_url(url: str, thread_id: str) -> str:
    return f"https://smith.langchain.com/studio/thread/{thread_id}?baseUrl={url}"

def run_config(email: Email) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": email.get("id", str(uuid4()))}}

def _dump(x):
    if hasattr(x, "model_dump"):
        return x.model_dump(mode="json")
    else:
        return x

def shape_results(result: OverallState):
    final_results: dict[str, Any] = {}

    # interruption exception
    if "__interrupt__" in result:
        final_results["__interrupt__"] = result["__interrupt__"]

    final_results["classification"] = _dump(result["classification"])
    final_results["email_extraction"] = _dump(result["email_extraction"])
    # Ignore, redundancy exists only for graph to function correctly
    # final_results["extracted_segments"] = result["extracted_segments"]
    final_results["document_data"] = []
    if "document_data" in result.keys():
        for att in result["document_data"]:
            attachment_data: dict[str, Any] = {
                "filename": att["filename"],
                "segments": []
            }
            # TODO - persist documents somewhere online/locally
            # For now, remove content from each attachment in document_data so it can
            # persist in a json-like structure w/o issue
            # final_results["document_data"][-1]["content"] = None

            if att["segments"] is not None:
                for seg in att["segments"]:
                    attachment_data["segments"].append(_dump(seg))

            final_results["document_data"].append(attachment_data)

    return final_results

def default_review_response() -> HumanResponse:
    """Non-interactive auto-response for a NEEDS_REVIEW interrupt.

    ``accept`` with empty args makes ``review_chunk_category`` fall back to the
    graph's own best guess, so an unattended run (poller/webhook/smoke test)
    completes instead of pausing. Override via ``run(..., review_response=...)``
    (e.g. ``{"type": "ignore"}`` to leave low-confidence chunks as NEEDS_REVIEW).
    """
    return {"type": "accept", "args": {"action": "review_chunk_category", "args": {}}}


def _interrupt_id(intr: Any) -> Any:
    """Interrupt id from a local ``Interrupt`` object or an SDK-serialized dict."""
    if isinstance(intr, dict):
        return intr.get("id") or intr.get("interrupt_id")
    return getattr(intr, "id", None)


def _pending_interrupts(state: Any) -> list:
    """Pending interrupt objects on a paused graph result, or []."""
    if isinstance(state, dict):
        return list(state.get("__interrupt__") or [])
    return []


def _resume_map(interrupts, response: HumanResponse) -> dict:
    """Interrupt-id -> resume value. LangGraph requires an id-keyed map when more
    than one interrupt is pending (parallel per-attachment subgraphs); the
    agent-inbox contract wants the value wrapped in a list (``interrupt(...)[0]``)."""
    return {_interrupt_id(i): [response] for i in interrupts if _interrupt_id(i) is not None}


def _print_updates(chunk: dict) -> None:
    for node, upd in chunk.items():
        if node == "__interrupt__":   # skip the verbose interrupt payload
            continue
        keys = list(upd.keys()) if isinstance(upd, dict) else upd
        print(f"    · {node}: {keys}")


def _drive_local(email: Email, response: HumanResponse, stream: bool) -> Any:
    config = run_config(email)
    if not stream:
        result = graph.invoke({"email": email}, config=config)
        # rounds = 0
        # while (pending := _pending_interrupts(result)) and rounds < MAX_RESUME_ROUNDS:
        #     rounds += 1
        #     print(f"    ↻ resolving {len(pending)} interrupt(s) (round {rounds})")
        #     result = graph.invoke(Command(resume=_resume_map(pending, response)), config=config)
        return result

    # In-process streaming: print per-node updates, keep the final state, and
    # re-stream a resume whenever the run pauses on one or more interrupts.
    stream_input: Any = {"email": email}
    final: dict = {}
    rounds = 0
    while True:
        pending: dict = {}
        for mode, chunk in graph.stream(
            stream_input, config=config, stream_mode=["updates", "values"]
        ):
            if isinstance(chunk, dict) and "__interrupt__" in chunk:
                for it in chunk["__interrupt__"]:
                    pending[_interrupt_id(it)] = it
            if mode == "updates" and isinstance(chunk, dict):
                _print_updates(chunk)
            elif mode == "values":
                final = chunk
        if not pending:
            pending = {_interrupt_id(it): it for it in _pending_interrupts(final)}
        if not pending or rounds >= MAX_RESUME_ROUNDS:
            return final
        rounds += 1
        print(f"    ↻ resolving {len(pending)} interrupt(s) (round {rounds})")
        stream_input = Command(resume={k: [response] for k in pending})


def _drive_remote(email: Email, response: HumanResponse, stream: bool) -> Any:
    tid = email.get("id", str(uuid4()))
    client = make_client(URL)
    client.threads.create(thread_id=tid)
    print(f"    studio: {studio_url(URL, tid)}")

    if not stream:
        result = client.runs.wait(
            tid, ASSISTANT_ID, input={"email": email}, raise_error=True
        )
        rounds = 0
        while (pending := _pending_interrupts(result)) and rounds < MAX_RESUME_ROUNDS:
            rounds += 1
            print(f"    ↻ resolving {len(pending)} interrupt(s) (round {rounds})")
            result = client.runs.wait(
                tid, ASSISTANT_ID,
                command={"resume": _resume_map(pending, response)},
                raise_error=True,
            )
        return result

    final: dict = {}
    stream_kwargs: dict = {"input": {"email": email}}
    rounds = 0
    while True:
        pending: dict = {}
        for part in client.runs.stream(
            tid, ASSISTANT_ID, stream_mode=["updates", "values"], **stream_kwargs
        ):
            event = getattr(part, "event", None)
            data = getattr(part, "data", None)
            if isinstance(data, dict) and "__interrupt__" in data:
                for it in (data["__interrupt__"] or []):
                    pending[_interrupt_id(it)] = it
            if event == "updates" and isinstance(data, dict):
                _print_updates(data)
            elif event == "values":
                final = data
            elif event and event.startswith("error"):
                print(f"    ! error event: {data}")
        state = final or client.threads.get_state(tid).get("values")
        if not pending:
            pending = {_interrupt_id(it): it for it in _pending_interrupts(state)}
        if not pending or rounds >= MAX_RESUME_ROUNDS:
            return state
        rounds += 1
        print(f"    ↻ resolving {len(pending)} interrupt(s) (round {rounds})")
        final = {}
        stream_kwargs = {"command": {"resume": {k: [response] for k in pending}}}


def run(
    email: Email,
    local: bool = True,
    stream: bool = False,
    review_response: HumanResponse | None = None,
) -> dict[str, Any]:
    response = review_response or default_review_response()
    try:
        state = (
            _drive_local(email, response, stream)
            if local
            else _drive_remote(email, response, stream)
        )
        return shape_results(state)
    except:
        print("Invocation failed!")
        raise
