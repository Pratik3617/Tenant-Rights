"""
legal_pdf_parser.py  (updated — Phase 2.1 replacement)
-------------------------------------------------------
Legal PDF parser that uses GovernmentPDFOCRHandler as its extraction backend.
Replaces the original pdfplumber-only parser with full OCR support.

The parser's job is still the same:
  DocumentResult (text per page) → list[LegalSection] (structured sections)

The OCR handler's job is: PDF file → DocumentResult (text per page).
This file wires them together.
"""

import re
import logging
from dataclasses import dataclass

from ocr_handler import GovernmentPDFOCRHandler, DocumentResult
from constants import (
    BARE_SECTION_PATTERN,
    DEFINITION_BODY_MAX_LEN,
    DEFINITION_BODY_MIN_LEN,
    DEFINITION_TERM_MAX_LEN,
    DEFINITION_TERM_MIN_LEN,
    DEFINITIONS_PATTERN,
    DEFINITIONS_SCAN_CHARS,
    MIN_SECTION_MATCHES,
    MIN_SECTION_WORDS,
    RASTERIZE_DPI,
    SECTION_PATTERN,
    SECTION_REVIEW_CONFIDENCE_THRESHOLD,
)

logger = logging.getLogger(__name__)


# Data structures
@dataclass
class LegalSection:
    act: str
    jurisdiction: str
    section_number: str
    section_title: str
    content: str
    page_start: int
    last_verified: str
    ocr_confidence: float = 1.0          # new: propagated from OCR handler
    needs_review: bool = False           # new: flagged for manual review


# Updated parser
class LegalPDFParser:
    """
    Parses legal PDFs into structured LegalSection objects.
    Uses GovernmentPDFOCRHandler for text extraction — handles digital,
    scanned, and mixed PDFs automatically.

    All regex patterns and thresholds are defined in constants.py.
    """

    def __init__(self, preprocess_scans: bool = True):
        self.ocr_handler = GovernmentPDFOCRHandler(
            rasterize_dpi=RASTERIZE_DPI,
            preprocess_scans=preprocess_scans,
        )

    def parse(self, pdf_path: str, metadata: dict) -> list[LegalSection]:
        """
        Main entry point.

        metadata = {
            "act": "Maharashtra Rent Control Act 1999",
            "jurisdiction": "india_maharashtra",
            "last_verified": "2024-01",
        }
        """
        # 1. Extract text (handles scanned/digital/mixed automatically)
        doc_result: DocumentResult = self.ocr_handler.process_document(pdf_path)

        if not doc_result.full_text.strip():
            logger.error(f"No text extracted from {pdf_path}. Check file.")
            return []

        logger.info(
            f"Extracted {len(doc_result.full_text.split()):,} words from "
            f"{doc_result.total_pages} pages "
            f"({doc_result.digital_pages} digital, {doc_result.scanned_pages} scanned)"
        )

        # 2. Build page → confidence mapping for propagation
        page_confidence_map = {
            p.page_number: p.confidence for p in doc_result.pages
        }

        # 3. Extract definitions block (injected into every chunk later)
        definitions = self._extract_definitions(doc_result.full_text)
        if definitions:
            logger.info(f"Found {len(definitions)} defined terms")

        # 4. Split into sections
        sections = self._split_into_sections(
            text=doc_result.full_text,
            metadata=metadata,
            page_confidence_map=page_confidence_map,
            doc_result=doc_result,
        )

        logger.info(f"Extracted {len(sections)} sections from {pdf_path}")
        return sections

    def _split_into_sections(
        self,
        text: str,
        metadata: dict,
        page_confidence_map: dict,
        doc_result: DocumentResult,
    ) -> list[LegalSection]:
        """
        Split full document text into LegalSection objects by detecting
        section headers. Tries primary pattern first, falls back to bare
        numbered sections.
        """
        matches = list(SECTION_PATTERN.finditer(text))

        if len(matches) < MIN_SECTION_MATCHES:
            logger.warning(
                f"Primary section pattern found only {len(matches)} matches. "
                f"Trying bare-number fallback."
            )
            matches = list(BARE_SECTION_PATTERN.finditer(text))

        if not matches:
            logger.error(
                "No sections detected. Treating entire document as one block."
            )
            return [LegalSection(
                act=metadata["act"],
                jurisdiction=metadata["jurisdiction"],
                section_number="FULL",
                section_title="Full Document",
                content=text,
                page_start=1,
                last_verified=metadata["last_verified"],
                ocr_confidence=doc_result.avg_ocr_confidence,
            )]

        sections = []
        for i, match in enumerate(matches):
            # Section content = text from end of header to start of next header
            content_start = match.end()
            content_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[content_start:content_end].strip()

            # Skip near-empty sections
            if len(content.split()) < MIN_SECTION_WORDS:
                continue

            # Estimate page number from character offset
            page_num = self._estimate_page(match.start(), doc_result)

            # Get OCR confidence for that page (1.0 if digital)
            confidence = page_confidence_map.get(page_num, 1.0)

            sections.append(LegalSection(
                act=metadata["act"],
                jurisdiction=metadata["jurisdiction"],
                section_number=match.group(1),
                section_title=match.group(2).strip(),
                content=content,
                page_start=page_num,
                last_verified=metadata["last_verified"],
                ocr_confidence=confidence,
                needs_review=(confidence < SECTION_REVIEW_CONFIDENCE_THRESHOLD),
            ))

        return sections

    def _extract_definitions(self, text: str) -> dict[str, str]:
        """
        Extract term definitions from the definitions section.
        These will be injected into chunks that reference those terms.

        Returns: {"landlord": "means the person entitled...", ...}
        """
        definitions = {}

        def_match = DEFINITIONS_PATTERN.search(text)
        if not def_match:
            return definitions

        def_block = text[def_match.end(): def_match.end() + DEFINITIONS_SCAN_CHARS]

        term_pattern = re.compile(
            r'"([^"]{' + str(DEFINITION_TERM_MIN_LEN) + r',' + str(DEFINITION_TERM_MAX_LEN) + r'})"'
            r'\s+(?:means?|includes?|refers? to)\s+'
            r'([^;.]{' + str(DEFINITION_BODY_MIN_LEN) + r',' + str(DEFINITION_BODY_MAX_LEN) + r'})',
            re.IGNORECASE
        )
        for match in term_pattern.finditer(def_block):
            term = match.group(1).strip().lower()
            definition = match.group(2).strip()
            definitions[term] = definition
            logger.debug(f"  Definition: '{term}' = {definition[:60]}...")

        return definitions

    def _estimate_page(self, char_offset: int, doc_result: DocumentResult) -> int:
        """
        Estimate the page number for a character offset in the full text.
        Approximate — sufficient for metadata; not for precise page references.
        """
        total_chars = len(doc_result.full_text)
        if total_chars == 0:
            return 1
        ratio = char_offset / total_chars
        estimated = max(1, int(ratio * doc_result.total_pages))
        return estimated