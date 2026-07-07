"""
Insurance Email Ambient Agent
1. Receives email json
2. Extracts data from email body; classifies email based on body
3. Classifies each received document; extracts data from each document
"""

import io
import mimetypes
import os
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langgraph.constants import END, START, StateGraph
from langgraph.types import Send
from markitdown import MarkItDown, StreamInfo
from transformers import pipeline
from trustcall import create_extractor

from insurance_email_agent.prompts import (
    ATTACHMENT_EXTRACTION_SYSTEM,
    ATTACHMENT_SEGMENTATION_SYSTEM,
    EMAIL_CLASSIFICATION_SYSTEM,
    EMAIL_EXTRACTION_SYSTEM,
    attachment_extraction_user,
    attachment_segmentation_user,
    email_classification_user,
    email_extraction_user,
)
from insurance_email_agent.schemas import (
    Attachment,
    DocumentCategory,
    DocumentClassification,
    Email,
    EmailCategory,
    EmailClassification,
    EmailExtraction,
    Extraction,
    Segment,
)
from insurance_email_agent.states import (
    ExtractionState,
    OverallState,
    SegmentationState,
)

### LLM Wrappers ###
llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0)
email_classification_llm = llm.with_structured_output(EmailClassification)
email_extraction_llm = llm.with_structured_output(EmailExtraction)
# document_extraction_llm = llm.bind_tools([Extraction], tool_choice="Extraction")

micro_llm = ChatOpenAI(model="gpt-5.4-nano-2026-03-17", temperature=0)
document_classification_llm = micro_llm.bind_tools(
    [DocumentClassification], tool_choice="DocumentClassification"
)

### Attachment classification/extraction utils ###
md_client = MarkItDown(
    enable_plugins=True, llm_client=ChatOpenAI(), llm_model="gpt-5.4-nano-2026-03-17"
)
headers = [("#", "Header 1")]
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers, strip_headers=False
)

DEVICE = os.getenv("DEVICE", -1)

zeroshot_classifier = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
    device=DEVICE,
)
hypothesis_template = "This document is a {}"
classes_verbalized = [
    DocumentCategory.CERTIFICATE,
    DocumentCategory.DECLARATIONS,
    DocumentCategory.ENDORSEMENT,
    DocumentCategory.INVOICE,
    DocumentCategory.NEEDS_REVIEW,
]


def email_classification(state: OverallState):
    email = state["email"]

    system_msg = EMAIL_CLASSIFICATION_SYSTEM
    user_msg = email_classification_user(email)
    result = email_classification_llm.invoke(
        [SystemMessage(content=system_msg), HumanMessage(content=user_msg)]
    )

    return {"classification": result}


# TODO: future HITL layer for classifications set as "NEEDS_REVIEW"
# def ambiguous_email_cls_checker(state: OverallState) -> Literal["email_extraction",     ]


def email_extraction(state: OverallState):
    email = state["email"]

    system_msg = EMAIL_EXTRACTION_SYSTEM
    user_msg = email_extraction_user(email)

    result = email_extraction_llm.invoke(
        [SystemMessage(content=system_msg), HumanMessage(content=user_msg)]
    )

    return {"email_extraction": result}


def attachment_existence_router(state: OverallState) -> Literal[END, "segmentation"]:
    attachments = state["email"].get("attachments")

    if not attachments:
        """
        If no documents, rely **only** on email data extraction
        """
        return END

    """
    If documents, commence segmentation process for each document (one node instance per document)
    """
    return [
        Send(
            "document_segmentation",
            {
                "attachment": {
                    "filename": att["filename"],
                    "content": att["content"],
                    "segments": [],
                }
            },
        )
        for att in attachments
    ]


def get_extension_mime_type(name: str) -> tuple[str, str]:
    """
    Get the MIME type of a file based on its extension.
    """
    _, ext = os.path.splitext(name)
    return ext, mimetypes.types_map.get(ext.lower(), "application/octet-stream")


def document_segmentation(state: SegmentationState):
    attachment = state["attachment"]
    content = attachment["content"]
    ext, mtype = get_extension_mime_type(attachment["filename"])

    # convert each page to md using markitdown
    result = md_client.convert_stream(
        io.BytesIO(content), stream_info=StreamInfo(extension=ext, mimetype=mtype)
    )

    # Header-wise chunking of data
    md_text = result.markdown
    chunks = markdown_splitter.split_text(md_text)
    chunks_text = [chunk.page_content for chunk in chunks]

    # zeroshot classification for every chunk (parallelized)
    labels = [c.value for c in classes_verbalized]
    outputs = zeroshot_classifier(
        chunks_text,
        candidate_labels=labels,
        hypothesis_template=hypothesis_template,
        multi_label=False,
        batch_size=8,
    )

    # Attach text to results
    categorized_chunks = []
    for i, o in enumerate(outputs):
        text = chunks_text[i]

        # redefine as category enum
        category = (
            DocumentCategory(o["labels"][0])
            if o["scores"][0] >= 0.5
            else DocumentCategory.NEEDS_REVIEW
        )

        categorized_chunks.append(
            {
                "text": text,
                "category": category,
                # TODO: add metadata for classes
                # "cls_metadata":
                #     {
                #         k:v
                #         for k,v in o.items()
                #     }
            }
        )
