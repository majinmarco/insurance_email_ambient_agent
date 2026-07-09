"""
Executed by poller/webhook, calls graph
"""

from typing import Any

from insurance_email_agent.schemas import Email
from insurance_email_agent.states import OverallState
from insurance_email_agent.graph import graph

def run(email: Email) -> dict[str, Any]:
    try:
        result: OverallState = graph.invoke({"email": email})

        final_results: dict[str, Any] = {}

        final_results["classification"] = result["classification"].model_dump(mode="json")
        final_results["email_extraction"] = result["email_extraction"].model_dump(mode="json")
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
                        attachment_data["segments"].append(seg.model_dump(mode="json"))

                final_results["document_data"].append(attachment_data)


        return final_results
    except:
        print("Invocation failed!")
        raise
