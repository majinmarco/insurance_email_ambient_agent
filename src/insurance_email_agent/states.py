"""
Storage of states to be used in graph
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages.utils import AnyMessage

from insurance_email_agent.schemas import (
    Attachment,
    Email,
    EmailClassification,
    EmailExtraction,
    Segment,
)


class OverallState(TypedDict):
    """
    Overall state of process
    """

    email: Email
    email_extraction: EmailExtraction
    classification: EmailClassification
    extracted_segments: Annotated[list[Segment], operator.add]
    # operator.add reducer: the router fans out one Send per attachment, so
    # several segmentation subgraph runs write document_data concurrently.
    # Without a reducer LangGraph raises InvalidUpdateError on the collision.
    document_data: Annotated[list[Attachment], operator.add]


class SegmentationState(TypedDict):
    """
    Used in attachment segmentation process
    """

    attachment: Attachment
