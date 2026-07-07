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
    document_data: Annotated[list[Attachment], operator.add]


class SegmentationState(TypedDict):
    """
    Used in attachment segmentation process
    """

    attachment: Attachment


class ExtractionState(TypedDict):
    """
    Per-segment extraction data
    """

    segment_data: Segment
