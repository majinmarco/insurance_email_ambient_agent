"""Frozen golden dataset for evals (MAR-9).

Re-exports the loader API so the eval suite (MAR-10) can::

    from tests.golden import load_cases, GoldenCase
"""

from tests.golden.loader import (
    AttachmentLabel,
    DocumentLabel,
    EmailLabel,
    GoldenCase,
    GoldenLabels,
    Manifest,
    load_case,
    load_cases,
    load_manifest,
)

__all__ = [
    "AttachmentLabel",
    "DocumentLabel",
    "EmailLabel",
    "GoldenCase",
    "GoldenLabels",
    "Manifest",
    "load_case",
    "load_cases",
    "load_manifest",
]
