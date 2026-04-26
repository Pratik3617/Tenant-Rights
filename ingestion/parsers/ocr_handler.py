"""
ocr_handler.py
--------------
Production OCR handler for government legal PDFs in the Tenant Rights RAG pipeline.

Handles:
  - Pure digital PDFs          → pdfplumber (fast, accurate)
  - Pure scanned PDFs          → PyMuPDF rasterize → Tesseract OCR
  - Mixed PDFs (most common)   → per-page detection → best strategy per page
  - Garbled encoding PDFs      → font check → rasterize fallback
  - Low-quality scans          → OpenCV preprocessing before OCR

Install requirements:
    pip install pdfplumber pymupdf pytesseract opencv-python-headless pillow
    apt-get install -y tesseract-ocr tesseract-ocr-eng poppler-utils
"""

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import cv2
import fitz                      # PyMuPDF
import numpy as np
import pdfplumber
import pytesseract
from PIL import Image

from constants import (
    ADAPTIVE_THRESH_BLOCK_SIZE,
    ADAPTIVE_THRESH_C,
    CLEANUP_PATTERNS,
    DENOISE_H,
    DIGITAL_CHAR_THRESHOLD,
    DILATE_KERNEL_SIZE,
    GARBLED_NON_ASCII_RATIO,
    GARBLED_SINGLE_CHAR_RATIO,
    HOUGH_THRESHOLD,
    HYBRID_CONFIDENCE_BOOST,
    HYBRID_CONFIDENCE_CAP,
    HYBRID_UNIQUE_WORD_MIN_LEN,
    HYBRID_UNIQUE_WORD_RATIO,
    MAX_DESKEW_ANGLE_DEG,
    MIN_DESKEW_ANGLE_DEG,
    MIXED_TEXT_MIN_CHARS,
    OCR_CHARACTER_FIXES,
    OCR_CONFIDENCE_THRESHOLD,
    RASTERIZE_DPI,
    TESSERACT_CONFIG,
)

logger = logging.getLogger(__name__)


# Data structures
class PageType(str, Enum):
    DIGITAL      = "digital"       # clean, machine-readable text
    SCANNED      = "scanned"       # image-only, needs OCR
    MIXED        = "mixed"         # partially extractable — use both
    GARBLED      = "garbled"       # has text layer but encoding broken
    EMPTY        = "empty"         # blank or separator page


@dataclass
class PageResult:
    page_number: int               # 1-based
    page_type: PageType
    text: str
    confidence: float              # 0.0–1.0 (OCR confidence or 1.0 for digital)
    extraction_method: str         # "pdfplumber" | "tesseract" | "hybrid"
    word_count: int = 0
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.word_count = len(self.text.split()) if self.text else 0


@dataclass
class DocumentResult:
    file_path: str
    pages: list[PageResult]
    full_text: str
    digital_pages: int
    scanned_pages: int
    failed_pages: list[int]
    avg_ocr_confidence: float

    @property
    def total_pages(self) -> int:
        return len(self.pages)

    @property
    def is_fully_digital(self) -> bool:
        return self.scanned_pages == 0

    @property
    def is_fully_scanned(self) -> bool:
        return self.digital_pages == 0


# Core handler
class GovernmentPDFOCRHandler:
    """
    Smart per-page PDF processor. Detects page type and routes to the
    correct extraction strategy. Designed for Indian government legal PDFs
    which are almost always mixed (some digital, some scanned pages).
    """

    def __init__(
        self,
        rasterize_dpi: int = RASTERIZE_DPI,
        ocr_confidence_threshold: float = OCR_CONFIDENCE_THRESHOLD,
        preprocess_scans: bool = True,    # apply OpenCV cleanup before OCR
    ):
        self.rasterize_dpi = rasterize_dpi
        self.ocr_confidence_threshold = ocr_confidence_threshold
        self.preprocess_scans = preprocess_scans
        self._verify_dependencies()

    # ── Public API ──────────────────────────────

    def process_document(self, pdf_path: str) -> DocumentResult:
        """
        Main entry point. Process all pages of a PDF, auto-detecting
        the best extraction strategy per page.
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Processing: {path.name}")

        pages: list[PageResult] = []
        failed_pages: list[int] = []

        # Open once with each library — keep both handles open
        with pdfplumber.open(pdf_path) as plumber_doc:
            fitz_doc = fitz.open(pdf_path)
            try:
                total = len(plumber_doc.pages)
                logger.info(f"Total pages: {total}")

                for i, plumber_page in enumerate(plumber_doc.pages):
                    page_num = i + 1
                    fitz_page = fitz_doc[i]

                    try:
                        result = self._process_page(
                            page_num, plumber_page, fitz_page
                        )
                        pages.append(result)

                        if result.warnings:
                            for w in result.warnings:
                                logger.warning(f"  Page {page_num}: {w}")

                    except Exception as e:
                        logger.error(f"  Page {page_num} FAILED: {e}")
                        failed_pages.append(page_num)
                        pages.append(PageResult(
                            page_number=page_num,
                            page_type=PageType.EMPTY,
                            text="",
                            confidence=0.0,
                            extraction_method="failed",
                            warnings=[str(e)]
                        ))
            finally:
                fitz_doc.close()

        return self._build_document_result(pdf_path, pages, failed_pages)

    def process_page(self, pdf_path: str, page_number: int) -> PageResult:
        """Process a single page (1-based). Useful for debugging."""
        with pdfplumber.open(pdf_path) as plumber_doc:
            fitz_doc = fitz.open(pdf_path)
            try:
                plumber_page = plumber_doc.pages[page_number - 1]
                fitz_page    = fitz_doc[page_number - 1]
                return self._process_page(page_number, plumber_page, fitz_page)
            finally:
                fitz_doc.close()

    # ── Page type detection ──────────────────────

    def detect_page_type(
        self,
        plumber_page,
        fitz_page
    ) -> tuple[PageType, str]:
        """
        Detect what type of page we're dealing with.
        Returns (PageType, reason string for logging).
        """
        # Step 1: try pdfplumber text extraction
        raw_text = plumber_page.extract_text() or ""
        char_count = len(raw_text.strip())

        # Step 2: check if page has embedded images (sign of a scan)
        has_images = len(fitz_page.get_images()) > 0

        # Step 3: check font encoding (garbled = broken encoding)
        if char_count >= DIGITAL_CHAR_THRESHOLD:
            if self._is_garbled(raw_text):
                return PageType.GARBLED, f"has text ({char_count} chars) but garbled encoding"
            return PageType.DIGITAL, f"clean digital text ({char_count} chars)"

        if has_images:
            if char_count > MIXED_TEXT_MIN_CHARS:
                return PageType.MIXED, f"partial text ({char_count} chars) + images"
            return PageType.SCANNED, f"image-only page (no extractable text)"

        if char_count > 0:
            return PageType.DIGITAL, f"sparse digital text ({char_count} chars), no images"

        return PageType.EMPTY, "blank page"

    # ── Per-page processing ──────────────────────

    def _process_page(
        self,
        page_num: int,
        plumber_page,
        fitz_page
    ) -> PageResult:

        page_type, reason = self.detect_page_type(plumber_page, fitz_page)
        logger.debug(f"  Page {page_num}: {page_type.value} — {reason}")

        if page_type == PageType.DIGITAL:
            return self._extract_digital(page_num, plumber_page)

        elif page_type == PageType.SCANNED:
            return self._extract_ocr(page_num, fitz_page)

        elif page_type == PageType.MIXED:
            return self._extract_hybrid(page_num, plumber_page, fitz_page)

        elif page_type == PageType.GARBLED:
            # Font encoding broken — rasterize and OCR instead
            logger.info(f"  Page {page_num}: garbled encoding, falling back to OCR")
            return self._extract_ocr(page_num, fitz_page)

        else:  # EMPTY
            return PageResult(
                page_number=page_num,
                page_type=page_type,
                text="",
                confidence=1.0,
                extraction_method="skipped_empty"
            )

    def _extract_digital(self, page_num: int, plumber_page) -> PageResult:
        """Fast path: clean digital text via pdfplumber."""
        text = plumber_page.extract_text() or ""
        text = self._clean_legal_text(text)
        return PageResult(
            page_number=page_num,
            page_type=PageType.DIGITAL,
            text=text,
            confidence=1.0,
            extraction_method="pdfplumber"
        )

    def _extract_ocr(self, page_num: int, fitz_page) -> PageResult:
        """
        Full OCR path for scanned pages.
        Rasterize with PyMuPDF → preprocess with OpenCV → OCR with Tesseract.
        """
        # 1. Rasterize to image
        pil_image = self._rasterize_page(fitz_page)

        # 2. Preprocess (deskew, denoise, binarize) — critical for old govt scans
        if self.preprocess_scans:
            pil_image = self._preprocess_scan(pil_image)

        # 3. Run Tesseract with confidence data
        ocr_data = pytesseract.image_to_data(
            pil_image,
            config=TESSERACT_CONFIG,
            output_type=pytesseract.Output.DICT
        )

        # 4. Extract text and compute mean confidence
        words, confidences = [], []
        for j, word in enumerate(ocr_data["text"]):
            conf = int(ocr_data["conf"][j])
            if conf > 0 and word.strip():  # conf=-1 means non-text block
                words.append(word)
                confidences.append(conf)

        text = " ".join(words)
        text = self._clean_legal_text(text)
        avg_conf = float(np.mean(confidences)) if confidences else 0.0

        warnings = []
        if avg_conf < self.ocr_confidence_threshold:
            warnings.append(
                f"Low OCR confidence: {avg_conf:.1f}% "
                f"(threshold: {self.ocr_confidence_threshold}%). "
                f"Manual review recommended."
            )

        return PageResult(
            page_number=page_num,
            page_type=PageType.SCANNED,
            text=text,
            confidence=avg_conf / 100.0,
            extraction_method="tesseract",
            warnings=warnings
        )

    def _extract_hybrid(
        self,
        page_num: int,
        plumber_page,
        fitz_page
    ) -> PageResult:
        """
        Hybrid strategy for mixed pages.
        Take pdfplumber text (better quality), fill gaps with OCR.
        Merge and deduplicate.
        """
        # Get digital text
        digital_text = plumber_page.extract_text() or ""

        # Get OCR text
        ocr_result = self._extract_ocr(page_num, fitz_page)

        # Merge: prefer digital text, use OCR to fill gaps
        merged = self._merge_digital_ocr(digital_text, ocr_result.text)
        merged = self._clean_legal_text(merged)

        return PageResult(
            page_number=page_num,
            page_type=PageType.MIXED,
            text=merged,
            confidence=min(HYBRID_CONFIDENCE_CAP, ocr_result.confidence + HYBRID_CONFIDENCE_BOOST),
            extraction_method="hybrid",
            warnings=ocr_result.warnings
        )

    # ── Image processing ─────────────────────────

    def _rasterize_page(self, fitz_page) -> Image.Image:
        """Rasterize a PDF page to PIL Image at configured DPI."""
        zoom = self.rasterize_dpi / 72.0           # 72 DPI is PDF default
        matrix = fitz.Matrix(zoom, zoom)
        pixmap = fitz_page.get_pixmap(matrix=matrix, alpha=False)
        img_array = np.frombuffer(pixmap.samples, dtype=np.uint8)
        img_array = img_array.reshape(pixmap.height, pixmap.width, 3)
        return Image.fromarray(img_array, "RGB")

    def _preprocess_scan(self, image: Image.Image) -> Image.Image:
        """
        OpenCV preprocessing pipeline — critical for Indian government
        scans which are often low contrast, slightly skewed, and noisy.

        Pipeline:
          RGB → Grayscale → Denoise → Binarize (Otsu) → Deskew → Dilate
        """
        # Convert PIL → OpenCV
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

        # 1. Denoise — removes scanner artifacts, smudges
        img_cv = cv2.fastNlMeansDenoising(img_cv, h=DENOISE_H)

        # 2. Adaptive threshold — handles uneven lighting across page
        #    Better than Otsu for pages with shadows at the edges
        img_cv = cv2.adaptiveThreshold(
            img_cv, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=ADAPTIVE_THRESH_BLOCK_SIZE,
            C=ADAPTIVE_THRESH_C
        )

        # 3. Deskew — correct scanner tilt (common in government docs)
        img_cv = self._deskew(img_cv)

        # 4. Dilate slightly — connects broken characters from low ink
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, DILATE_KERNEL_SIZE)
        img_cv = cv2.dilate(img_cv, kernel, iterations=1)

        return Image.fromarray(img_cv)

    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and correct rotation of scanned pages.
        Uses Hough line transform to find dominant text angle.
        Skips correction if angle is <0.5° (negligible) or >45° (probably wrong).
        """
        # Find edges
        edges = cv2.Canny(image, 50, 150, apertureSize=3)

        # Detect lines via Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=HOUGH_THRESHOLD)
        if lines is None:
            return image

        angles = []
        for line in lines:
            rho, theta = line[0]
            angle_deg = np.degrees(theta) - 90
            if -MAX_DESKEW_ANGLE_DEG < angle_deg < MAX_DESKEW_ANGLE_DEG:
                angles.append(angle_deg)

        if not angles:
            return image

        median_angle = float(np.median(angles))

        if abs(median_angle) < MIN_DESKEW_ANGLE_DEG:
            return image

        logger.debug(f"    Correcting skew: {median_angle:.2f}°")

        # Rotate to correct
        h, w = image.shape
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        corrected = cv2.warpAffine(
            image, rotation_matrix, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        return corrected

    # ── Text utilities ───────────────────────────

    def _merge_digital_ocr(self, digital: str, ocr: str) -> str:
        """
        Merge digital and OCR text for mixed pages.
        Strategy: use digital as base (higher quality), append unique
        OCR content not already present in digital text.
        This catches text in image regions that pdfplumber misses.
        """
        if not digital.strip():
            return ocr
        if not ocr.strip():
            return digital

        # Simple merge: if OCR is significantly longer, it captured more
        digital_words = set(digital.lower().split())
        ocr_words = ocr.lower().split()

        # Find OCR words not in digital (unique content from scan regions)
        unique_ocr_words = [
            w for w in ocr_words if w not in digital_words and len(w) > HYBRID_UNIQUE_WORD_MIN_LEN
        ]

        if len(unique_ocr_words) < HYBRID_UNIQUE_WORD_RATIO * len(ocr_words):
            return digital

        # Otherwise combine — digital first (more trustworthy), OCR supplements
        return digital + "\n\n" + ocr

    def _clean_legal_text(self, text: str) -> str:
        """
        Post-process OCR/digital text for legal documents.
        Fixes common OCR errors in legal text, normalizes whitespace,
        and preserves section numbering structure.
        """
        if not text:
            return ""

        for pattern, replacement in CLEANUP_PATTERNS:
            text = re.sub(pattern, replacement, text)

        for pattern, fix in OCR_CHARACTER_FIXES.items():
            text = re.sub(pattern, fix, text)

        return text.strip()

    def _is_garbled(self, text: str) -> bool:
        """
        Detect if extracted text has broken encoding (common in old govt PDFs
        with non-embedded fonts or custom CID mappings).
        Heuristic: >15% non-ASCII or >20% single-char tokens signals garbling.
        """
        if not text:
            return False
        non_ascii = sum(1 for c in text if ord(c) > 127)
        if non_ascii / len(text) > GARBLED_NON_ASCII_RATIO:
            return True
        words = text.split()
        if not words:
            return False
        single_chars = sum(1 for w in words if len(w) == 1)
        return single_chars / len(words) > GARBLED_SINGLE_CHAR_RATIO

    # ── Result assembly ──────────────────────────

    def _build_document_result(
        self,
        pdf_path: str,
        pages: list[PageResult],
        failed_pages: list[int]
    ) -> DocumentResult:
        full_text = "\n\n".join(
            p.text for p in pages if p.text.strip()
        )
        digital_pages = sum(1 for p in pages if p.page_type == PageType.DIGITAL)
        scanned_pages = sum(1 for p in pages if p.page_type == PageType.SCANNED)

        ocr_pages = [p for p in pages if "tesseract" in p.extraction_method]
        avg_ocr_conf = (
            float(np.mean([p.confidence for p in ocr_pages]))
            if ocr_pages else 1.0
        )

        result = DocumentResult(
            file_path=pdf_path,
            pages=pages,
            full_text=full_text,
            digital_pages=digital_pages,
            scanned_pages=scanned_pages,
            failed_pages=failed_pages,
            avg_ocr_confidence=avg_ocr_conf
        )

        self._log_summary(result)
        return result

    def _log_summary(self, result: DocumentResult):
        logger.info(
            f"\n{'─'*50}\n"
            f"  File    : {Path(result.file_path).name}\n"
            f"  Pages   : {result.total_pages} total | "
            f"{result.digital_pages} digital | "
            f"{result.scanned_pages} scanned\n"
            f"  OCR avg : {result.avg_ocr_confidence:.1%} confidence\n"
            f"  Words   : {len(result.full_text.split()):,}\n"
            f"  Failed  : {result.failed_pages or 'none'}\n"
            f"{'─'*50}"
        )

    def _verify_dependencies(self):
        """Fail fast with clear errors if dependencies are missing."""
        try:
            pytesseract.get_tesseract_version()
        except pytesseract.TesseractNotFoundError:
            raise EnvironmentError(
                "Tesseract not found. Install with:\n"
                "  Ubuntu/Debian: sudo apt-get install -y tesseract-ocr tesseract-ocr-eng\n"
                "  macOS:         brew install tesseract"
            )