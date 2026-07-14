"""
Insurance Email Ambient Agent
1. Receives email json
2. Extracts data from email body; classifies email based on body
3. Classifies each received document; extracts data from each document
"""

import base64
import io
import mimetypes
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

import tiktoken
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt
from markitdown import MarkItDown, StreamInfo
from pydantic import BaseModel
from transformers import pipeline
from trustcall import create_extractor

from insurance_email_agent.prompts import (
    EMAIL_CLASSIFICATION_SYSTEM,
    EMAIL_EXTRACTION_SYSTEM,
    SEGMENT_STITCH_SYSTEM,
    segment_stitch_user,
    attachment_extraction_system,
    attachment_extraction_user,
    email_classification_user,
    email_extraction_user,
)
from insurance_email_agent.schemas import (
    Attachment,
    CertificateExtraction,
    DeclarationsExtraction,
    DocumentCategory,
    SegmentStitch,
    EmailClassification,
    EmailExtraction,
    EndorsementExtraction,
    Extraction,
    InvoiceExtraction,
    Segment,
    HumanInterrupt,
    HumanResponse,
)
from insurance_email_agent.states import (
    OverallState,
    SegmentationState,
)

### LLM Wrappers ###
llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0)
email_classification_llm = llm.with_structured_output(EmailClassification)
email_extraction_llm = llm.with_structured_output(EmailExtraction)

SCHEMA_BY_CATEGORY: dict[DocumentCategory, type[BaseModel]] = {
    DocumentCategory.CERTIFICATE: CertificateExtraction,
    DocumentCategory.INVOICE: InvoiceExtraction,
    DocumentCategory.DECLARATIONS: DeclarationsExtraction,
    DocumentCategory.ENDORSEMENT: EndorsementExtraction,
}

# one llm, N lightweight extractors — built once at import
EXTRACTOR_BY_CATEGORY = {
    cat: create_extractor(llm, tools=[schema], tool_choice=schema.__name__)
    for cat, schema in SCHEMA_BY_CATEGORY.items()
}

micro_llm = ChatOpenAI(model="gpt-5.4-nano-2026-03-17", temperature=0)
segment_stitching_llm = micro_llm.with_structured_output(
    SegmentStitch
)

### Attachment classification/extraction utils ###
md_client = MarkItDown(
    enable_plugins=True, llm_client=ChatOpenAI(), llm_model="gpt-5.4-nano-2026-03-17"
)
headers = [("#", "Header 1")]
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers, strip_headers=False
)

DEVICE = int(os.getenv("DEVICE", -1))

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
]

MAX_REVIEW_RETRIES = 2

### HELPERS ###
tokenizer = tiktoken.encoding_for_model("gpt-4o")


def get_token_length(text: str) -> int:
    return len(tokenizer.encode(text))


token_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4096, chunk_overlap=0, length_function=get_token_length
)


def get_extension_mime_type(name: str) -> tuple[str, str]:
    """
    Get the MIME type of a file based on its extension.
    """
    _, ext = os.path.splitext(name)
    return ext, mimetypes.types_map.get(ext.lower(), "application/octet-stream")


def extract_segment(seg: Segment) -> Extraction | None:
    extractor = EXTRACTOR_BY_CATEGORY.get(seg.category)
    if extractor is None:  # NEEDS_REVIEW → no schema
        return None

    system_msg = attachment_extraction_system(seg.category)

    result = extractor.invoke(
        {
            "messages": [
                SystemMessage(content=system_msg),
                HumanMessage(
                    content=attachment_extraction_user(
                        seg.filename, seg.category, seg.text
                    )
                ),
            ]
        }
    )
    return result["responses"][0]

def stitch_or_divide_segments(seg_A: str, seg_B: str) -> SegmentStitch:
    sys_msg = SEGMENT_STITCH_SYSTEM
    user_msg = segment_stitch_user(seg_A, seg_B)

    result: SegmentStitch = segment_stitching_llm.invoke([SystemMessage(content=sys_msg), HumanMessage(content=user_msg)])

    return result

def chunk_cat_interrupt_description(
    attachment: Attachment, chunk_text: str, cls_metadata: dict[str, float]
) -> str:
    scores = "\n".join(
        f"- {cls}: {score:.2%}" for cls, score in cls_metadata.items()
    )
    return f"""
    ## Please review the below data and assign the correct category (categories listed under "Classification scores")

    # **Filename:** {attachment['filename']}

    ------
    # **Text chunk:**
    {chunk_text}

    ------
    # **Classification scores:**
    {scores}

    ______

    """


def _coerce_category(raw: str | None) -> DocumentCategory | None:
    """Normalize free text; return the matching assignable category or None."""
    if not raw:
        return None
    norm = raw.strip().lower()
    return next((c for c in classes_verbalized if norm == c.value.lower()), None)

def review_chunk_category(request: HumanInterrupt,
                          best_guess: DocumentCategory) -> DocumentCategory:
    """Drive one NEEDS_REVIEW interrupt to a concrete category.
    accept -> best_guess; edit -> validated category (re-interrupt if invalid);
    ignore -> NEEDS_REVIEW; bounded retry then fall back to NEEDS_REVIEW."""
    current = request
    for _ in range(MAX_REVIEW_RETRIES + 1):
        response: HumanResponse = interrupt(current)[0]   # inbox returns a list
        rtype = response.get("type")
        if rtype == "ignore":
            return DocumentCategory.NEEDS_REVIEW
        if rtype == "accept":
            ar = response.get("args") or {}
            return _coerce_category((ar.get("args") or {}).get("category")) or best_guess
        if rtype == "edit":
            ar = response.get("args") or {}
            edited = (ar.get("args") or {}).get("category")
            resolved = _coerce_category(edited)
            if resolved is not None:
                return resolved
            # invalid -> re-interrupt with a corrective description next iteration
            valid = "\n".join(f"- `{v}`" for v in [c.value for c in classes_verbalized])
            current = {**request, "description":
                       f"**`{edited}` is not a valid category.** Type EXACTLY one of:\n"
                       f"{valid}\n\n" + (request.get("description") or "")}
            continue
        return DocumentCategory.NEEDS_REVIEW
    return DocumentCategory.NEEDS_REVIEW

### NODES/ROUTERS ###


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


def attachment_existence_router(
    state: OverallState,
) -> Literal[END] | list[Send]:
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
            "document_segmentation_extraction",
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


def document_segmentation_extraction(state: SegmentationState):
    attachment = state["attachment"]
    content = attachment["content"]
    # Input arriving over the langgraph HTTP API is JSON, which has no bytes
    # type, so callers pass attachment content as a base64 string. Decode it
    # back to bytes for MarkItDown. In-process callers may still pass raw bytes.
    if isinstance(content, str):
        content = base64.b64decode(content)
    ext, mtype = get_extension_mime_type(attachment["filename"])

    # convert each page to md using markitdown
    result = md_client.convert_stream(
        io.BytesIO(content), stream_info=StreamInfo(extension=ext, mimetype=mtype)
    )

    # Header-wise chunking of data
    md_text = result.markdown
    chunks = markdown_splitter.split_text(md_text)
    chunks_text = [chunk.page_content for chunk in chunks]

    # Further split large chunks and insert in-place in original spot
    chunks_text_remade: list[str] = []
    for txt in chunks_text:
        chunks_to_add: list[str] = []
        if get_token_length(txt) > 8192:
            # split into chunks of <=8192/2 tokens
            chunks_to_add = token_text_splitter.split_text(txt)
            chunks_text_remade.extend(chunks_to_add)
        else:
            chunks_text_remade.append(txt)


    # zeroshot classification for every chunk (parallelized)
    labels = [c.value for c in classes_verbalized]
    outputs = zeroshot_classifier(
        chunks_text_remade,
        candidate_labels=labels,
        hypothesis_template=hypothesis_template,
        multi_label=False,
        batch_size=8,
    )

    # Attach text to results
    categorized_chunks = []
    for i, o in enumerate(outputs):
        text = chunks_text_remade[i]

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
                "cls_metadata":
                    {
                        k:v
                        for k,v in zip(o["labels"], o["scores"])
                    }
            }
        )

    # Join together chunks of equal category unless OTHER/NEEDS_REVIEW
    segments: list[Segment] = []
    for i, chunk in enumerate(categorized_chunks):
        category = chunk["category"]
        if category == DocumentCategory.NEEDS_REVIEW:
            best_guess = DocumentCategory(next(iter(chunk["cls_metadata"])))
            request: HumanInterrupt = {
                "action_request": {
                    "action": "DocumentClassificationCorrection",
                    "args": {"category": best_guess.value},   # lowercase key, string value
                },
                "config": {"allow_ignore": True, "allow_respond": False,
                        "allow_edit": True, "allow_accept": True},
                "description": chunk_cat_interrupt_description(
                    attachment, chunk["text"], chunk["cls_metadata"]),
            }
            category = review_chunk_category(request, best_guess)
            chunk["category"] = category

        # Only the most recent (current) segment can be continued — stitching is
        # sequential, so compare this chunk against the previous chunk.
        last = segments[-1] if segments else None
        if last is not None and last.category == category:
            # analyze whether pages are connected or different
            stitch_status: SegmentStitch = stitch_or_divide_segments(
                categorized_chunks[i - 1]["text"], chunk["text"]
            )  # prev vs current chunk

            # chunk part of same document (add to current segment)
            if not stitch_status.new_document:
                last.text = last.text + "\n------\n" + chunk["text"]
                last.chunk_indices.append(i)
                last.cls_metadata.append(chunk["cls_metadata"])
                continue

        # new document, different category, or no segment yet → start a new one
        segments.append(
            Segment(
                category=category,
                filename=attachment["filename"],
                chunk_indices=[i],
                text=chunk["text"],
                extraction=None,
                cls_metadata=[chunk["cls_metadata"]]
            )
        )

    # Execute extractions here
    # --- map: extract every segment in parallel ---
    if segments:
        with ThreadPoolExecutor(max_workers=min(8, len(segments))) as pool:
            future_to_seg = {pool.submit(extract_segment, seg): seg for seg in segments}
            for future in as_completed(future_to_seg):
                seg = future_to_seg[future]
                try:
                    seg.extraction = future.result()
                except Exception as exc:
                    seg.extraction = None
                    # isolate failure — don't lose the other segments' work
                    print(
                        f"extraction failed for {seg.filename} {seg.chunk_indices}: {exc}"
                    )

    return {
        "document_data": [{**attachment, "segments": segments}]
    }  # return existing attachment properties + add segments with extractions to it


## BUILD SEGMENTATION GRAPH (different state) ##
segment_builder = StateGraph(SegmentationState, output=OverallState)

segment_builder.add_node(
    "document_segmentation_extraction", document_segmentation_extraction
)

segment_builder.add_edge(START, "document_segmentation_extraction")
segment_builder.add_edge("document_segmentation_extraction", END)

segment_subgraph = segment_builder.compile()

## BUILD OVERALL GRAPH ##

overall_builder = StateGraph(OverallState)

overall_builder.add_node("email_classification", email_classification)
overall_builder.add_node("email_extraction", email_extraction)
overall_builder.add_node("document_segmentation_extraction", segment_subgraph)

overall_builder.add_edge(START, "email_classification")
overall_builder.add_edge("email_classification", "email_extraction")
overall_builder.add_conditional_edges(
    "email_extraction",
    attachment_existence_router,
    # END must be listed: the router returns END when there are no attachments,
    # and a value returned from a conditional edge must be a declared destination
    # (otherwise LangGraph raises KeyError: '__end__').
    [END, "document_segmentation_extraction"],
)
overall_builder.add_edge("document_segmentation_extraction", END)

graph = overall_builder.compile()
