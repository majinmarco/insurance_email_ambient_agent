"""
Storage of states to be used in graph
"""

import operator
from typing import Annotated, Literal, TypedDict
from uuid import uuid4

from langchain_core.messages.utils import AnyMessage

from insurance_email_agent.schemas import (
    Attachment,
    Email,
    EmailClassification,
    Segment,
)

# TODO - import email type
# TODO - import attachment type


class OverallState(TypedDict):
    email: Email
    classification: EmailClassification
    document_data: Annotated[list[Attachment], operator.add]


class SegmentationState(TypedDict):
    """
    Used in attachment segmentation process
    """

    messages: Annotated[list[AnyMessage], operator.add]
    segmentation_data: list[Segment]


class ExtractionState(TypedDict):
    segment_data: Segment
