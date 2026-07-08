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
    document_data: list[Attachment]


class SegmentationState(TypedDict):
    """
    Used in attachment segmentation process
    """

    attachment: Attachment
