"""
Functions used to ingest emails and convert to `Email` type
"""
from email.message import EmailMessage
from email.utils import parsedate_to_datetime
from insurance_email_agent.schemas import Email, Attachment
from email import policy
from email.parser import BytesParser
from typing import Any
import re
from uuid import uuid4

from html.parser import HTMLParser
import mimetypes

def to_email(raw: bytes) -> Email:
    """
    Function that receives a base64-encoded string and converts to Email type
    """
    msg = BytesParser(policy=policy.default).parsebytes(raw)

    # obtain body
    # Requires complex parsing due to variation in formatting among different email providers
    body = extract_email_body(msg)

    # obtain attachments
    attachments, attachment_contents = extract_attachments_with_content(
                    msg
                )

    attachments_formatted = [
        Attachment(
            filename=att["filename"],
            content=attachment_contents[att["filename"]],
            segments=[]
        )
        for att in attachments
    ]

    email = Email(
        id = msg.get("Message-ID", str(uuid4())),
        subject = msg.get("Subject", ""),
        date_received = parsedate_to_datetime(msg.get("Date", "")),
        body = body,
        sender = msg.get("From", ""),
        recipient = msg.get("To", ""),
        attachments = attachments_formatted
    )

    return email

def extract_email_body(msg: EmailMessage) -> str:
    """Extract text and HTML body from email message.

    If text body is null but HTML exists, extract plain text from HTML.
    """
    body: dict[str, str | None] = {"text": None, "html": None}


    try:
        # Try modern email API first
        text_body = msg.get_body(preferencelist=("plain",))
        if text_body:
            body["text"] = text_body.get_content()

        html_body = msg.get_body(preferencelist=("html",))
        if html_body:
            body["html"] = html_body.get_content()

    except Exception as e:
        print(f"Modern email API failed, using fallback: {e}")

        # Fallback method for broader compatibility
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                disposition = part.get_content_disposition()

                # Skip attachments
                if disposition == "attachment":
                    continue

                try:
                    if content_type == "text/plain" and not body["text"]:
                        body["text"] = part.get_content()
                    elif content_type == "text/html" and not body["html"]:
                        body["html"] = part.get_content()
                except (UnicodeDecodeError, AttributeError, ValueError) as e:
                    print(f"Failed to get content for part: {e}")
                    return "PLACEHOLDER"

    # If text is null but HTML exists, extract plain text from HTML
    if not body["text"] and body["html"]:
        print(
            "Text body is null but HTML exists - extracting plain text from HTML"
        )
        body["text"] = html_to_text(body["html"])
        if body["text"]:
            print(
                f"Successfully extracted {len(body['text'])} characters of plain text from HTML"
            )
        else:
            print("HTML to text extraction resulted in empty text")

    text = body["text"]
    html = body["html"]
    if text is not None:
        return text
    if html is not None:
        return html
    return "PLACEHOLDER"


class HTMLTextExtractor(HTMLParser):
    """Simple HTML parser to extract plain text from HTML content"""

    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.skip_tags = {"script", "style", "head", "title", "meta"}
        self.current_tag = None

    def handle_starttag(self, tag, attrs):
        self.current_tag = tag

    def handle_endtag(self, tag):
        self.current_tag = None

    def handle_data(self, data):
        # Skip content from script, style, etc.
        if self.current_tag not in self.skip_tags:
            text = data.strip()
            if text:
                self.text_parts.append(text)

    def get_text(self):
        return " ".join(self.text_parts)


def html_to_text(html_content: str) -> str:
    """
    Convert HTML content to plain text by stripping tags.

    Args:
        html_content: HTML string

    Returns:
        Plain text extracted from HTML
    """
    if not html_content:
        return ""

    try:
        parser = HTMLTextExtractor()
        parser.feed(html_content)
        text = parser.get_text()

        # Clean up extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text
    except Exception as e:
        print(f"Failed to parse HTML content: {e}")
        # Fallback: simple regex-based tag removal
        try:
            text = re.sub(r"<[^>]+>", "", html_content)
            text = re.sub(r"\s+", " ", text).strip()
            return text
        except Exception as fallback_error:
            print(f"HTML parsing fallback also failed: {fallback_error}")
            return ""

def extract_attachments_with_content(
    msg,
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    """Extract both attachment metadata and content.
    """
    attachments: list[dict[str, Any]] = []
    attachment_contents: dict[str, bytes] = {}

    for part in msg.walk():
        # Skip container parts
        if part.is_multipart():
            continue

        cd = part.get_content_disposition()  # 'attachment', 'inline', or None
        filename = part.get_filename()

        if cd == "attachment":
            if not filename:
                continue  # regular attachments still require a filename (existing behavior)
        else:
            continue

        att_info, content = process_attachment_with_content(
            part,
            len(attachments) + 1
        )

        if att_info is not None and content is not None:
            attachments.append(att_info)
            attachment_contents[att_info["s3_key"]] = content

    return attachments, attachment_contents

def process_attachment_with_content(
    part, index: int
) -> tuple[dict[str, str] | None, bytes | None]:
    """Process a single attachment and return both metadata and content"""
    try:
        filename = part.get_filename()
        if not filename:
            ext = mimetypes.guess_extension(part.get_content_type()) or ""
            filename = f"attachment_{index}{ext}"

        # Get attachment content
        try:
            content = part.get_content()
            if isinstance(content, str):
                content = content.encode("utf-8")
        except (UnicodeDecodeError, AttributeError, ValueError) as e:
            print(f"Failed to get content for attachment {filename}: {e}")
            content = part.get_payload(decode=True)

        if not content:
            print(f"No content found for attachment: {filename}")
            return None, None


        attachment_info = {
            "filename": filename,
        }

        return attachment_info, content

    except (UnicodeDecodeError, AttributeError, ValueError, TypeError) as e:
        print(f"Error processing attachment {index}: {e}")
        return None, None
