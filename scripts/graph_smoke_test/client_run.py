"""Drive the running ``langgraph dev`` graph over the SDK so runs show in Studio.

We always create an explicit thread (stable Studio URL), then either stream
node-by-node updates (``--verbose``) or wait for the final state.
"""

from __future__ import annotations

from typing import Any

from langgraph_sdk import get_sync_client


def make_client(url: str):
    return get_sync_client(url=url)


def studio_url(url: str, thread_id: str) -> str:
    return f"https://smith.langchain.com/studio/thread/{thread_id}?baseUrl={url}"


def _extract_error(result: Any) -> str | None:
    """Return an error string if the run failed, else None."""
    if result is None:
        return "no result returned"
    if isinstance(result, dict):
        if "__error__" in result:
            return str(result["__error__"])
        # A completed run of this graph always has a classification.
        if "classification" not in result:
            return f"unexpected result shape (keys={list(result.keys())})"
    return None


def run_graph(
    client,
    assistant_id: str,
    payload: dict,
    metadata: dict,
    *,
    verbose: bool = False,
) -> tuple[str, Any, str | None]:
    """Create a thread, run the graph, return ``(thread_id, final_state, error)``."""
    thread = client.threads.create(metadata=metadata)
    tid = thread["thread_id"]

    if verbose:
        final: dict = {}
        err: str | None = None
        for part in client.runs.stream(
            tid,
            assistant_id,
            input=payload,
            metadata=metadata,
            stream_mode=["updates", "values"],
        ):
            event = getattr(part, "event", None)
            data = getattr(part, "data", None)
            if event == "updates" and isinstance(data, dict):
                for node, upd in data.items():
                    keys = list(upd.keys()) if isinstance(upd, dict) else upd
                    print(f"    · {node}: {keys}")
            elif event == "values":
                final = data
            elif event and event.startswith("error"):
                err = str(data)
                print(f"    ! error event: {data}")
        result = final or client.threads.get_state(tid).get("values")
        return tid, result, err or _extract_error(result)

    result = client.runs.wait(
        tid, assistant_id, input=payload, metadata=metadata, raise_error=False
    )
    return tid, result, _extract_error(result)
