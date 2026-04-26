import re
import logging
from dataclasses import dataclass, field
from ocr_handler import GovernmentPDFOCRHandler, DocumentResult

logger = logging.getLogger(__name__)

@dataclass
class LegalSection:
    act: str
    jurisdiction: str
    section_number: str
    title: str
    content: str
    page_start: int
    last_verified: str
    ocr_confidence: float = field(default=1.0)
    needs_review: bool = field(default=False)

class LegalPDFParser:
    """
        Parses legal PDFs into structured LegalSection objects.
        Uses GovernmentPDFOCRHandler for text extraction — handles digital,
        scanned, and mixed PDFs automatically.
    """
    # Matches: "Section 15A — Security Deposit"
    #          "15A. Security Deposit"
    #          "S. 15A Security Deposit"
    SECTION_PATTERN = re.compile(
        r'^(?:Section|Sec\.|S\.)\s*(\d+[A-Za-z]?)'   # "Section 15A" or "15A"
        r'(?:\s*[.—–:-]\s*|\s+)'                       # separator
        r'([A-Z][^\n]{3,80})',                          # title (3–80 chars)
        re.MULTILINE
    )

    # Fallback: bare numbered sections like "15A. Deposit refund."
    BARE_SECTION_PATTERN = re.compile(
        r'^(\d+[A-Za-z]?)\.\s+([A-Z][^\n]{3,80})',
        re.MULTILINE
    )

    # Definitions section header — used to extract definition blocks
    DEFINITIONS_PATTERN = re.compile(
        r'^(?:DEFINITIONS?|Interpretation|Meaning of terms)',
        re.MULTILINE | re.IGNORECASE
    )

    def __init__(
            self,
            rasterize_dpi: int = 200,
            preprocess_scans: bool = True,
            min_section_words: int = 10,
    ):
        self.ocr_handler = GovernmentPDFOCRHandler(
            rasterize_dpi=rasterize_dpi,
            preprocess_scans=preprocess_scans
        )
        self.min_section_words = min_section_words
        
    def parse(self, pdf_path: str, metadata: dict) -> list[LegalSection]:
        """"
            metadata = {
                "act": "Maharashtra Rent Control Act 1999",
                "jurisdiction": "india_maharashtra",
                "last_verified": "2024-01",
            }
        """
        # Extract text (handles scanned/digital/mixed PDFs)
        doc_result: DocumentResult = self.ocr_handler.process_document(pdf_path)

        if not doc_result.full_text.strip():
            logger.warning(f"No text extracted from {pdf_path}")
            return []
        
        logger.info(
            f"Extracted {len(doc_result.full_text.split()):,} words from "
            f"{doc_result.total_pages} pages "
            f"({doc_result.digital_pages} digital, {doc_result.scanned_pages} scanned)"
        )

        # Build Page -> Confidence mapping for propagation
        page_Confidence_map = {
            p.page_number: p.confidence
            for p in doc_result.pages
        }

        # Extract definitions block (injected into every chunk later)
        definitions = self._extract_definitions(doc_result.full_text)

        if definitions:
            logger.info(f"Found {len(definitions)} defined terms")

        # split into sections
        sections = self._split_into_sections(
            text = doc_result.full_text,
            metadata = metadata,
            page_confidence_map = page_Confidence_map,
            doc_result = doc_result
        )

    def split_into_sections(
            self,
            text: str,
            metadata: dict,
            page_confidence_map: dict,
            doc_result: DocumentResult
    ) -> list[LegalSection]:
        """
            Split full document text into LegalSection objects by detecting
            section headers. Tries primary pattern first, falls back to bare
            numbered sections.
        """
        matches = list(self.SECTION_PATTERN.finditer(text))

        # fallback to bare numbered sections if primary finds < 3 sections
        if len(matches) < 3:
            logger.warning(
                f"Primary section pattern found only {len(matches)} matches. "
                f"Trying bare-number fallback."
            )

            matches = list(self.BARE_SECTION_PATTERN.finditer(text))

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
            # section content = text from end of header to the start of next header (or end of doc)
            content_start = match.end()
            content_end = matches[i+1].start() if i+1 < len(matches) else len(text)
            content = text[content_start:content_end].strip()

            # skip near empty sections
            if len(content.split()) < self.min_section_words:
                continue

            # Estimate page number from character offset
            page_num = self._estimate_page_number(match.start(), doc_result)

            # Get OCR confidence for that page (1.0 if digital)
            confindence = page_confidence_map.get(page_num, 1.0)

            sections.append(LegalSection(
                act=metadata["act"],
                jurisdiction=metadata["jurisdiction"],
                section_number=match.group(1),
                title=match.group(2).strip(),
                content=content,
                page_start=page_num,
                last_verified=metadata["last_verified"],
                ocr_confidence=confindence,
                needs_review=(confindence < 0.65)
            ))

    def _extract_definitions(self, text: str) -> dict:
        """
            Extract definitions block (if present) as a dict of term -> definition.
            These will be injected into chunks that reference those terms.

            Returns: {"landlord": "means the person entitled...", ...}
        """
        definitions = {}

        # find the definitions section
        def_match = self.DEFINITIONS_PATTERN.search(text)
        if not def_match:
            return definitions
        
        # take the first 3000 chars after the header as the definitions block
        def_block = text[def_match.end():def_match.end()+3000]

        # match patterns like:
        #   "landlord" means...
        #   "premises" means...
        term_pattern = re.compile(
            r'"([^"]{3,40})"\s+(?:means?|includes?|refers? to)\s+([^;.]{20,400})',
            re.IGNORECASE
        )

        for match in term_pattern.finditer(def_block):
            term = match.group(1).strip().lower()
            definition = match.group(2).strip()
            definitions[term] = definition
            logger.debug(f"  Definition: '{term}' = {definition[:60]}...")
            return definitions
        
    def _estimate_page_number(self, char_offset: int, doc_result: DocumentResult) -> int:
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
    