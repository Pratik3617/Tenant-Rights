"""
constants.py
------------
All tunable constants for the ingestion pipeline — OCR handler and legal PDF parser.
Change values here; both modules pick them up automatically.

Grouped by concern:
  - Page detection thresholds
  - OCR / rasterization settings
  - Text cleanup patterns
  - Legal section parsing patterns
  - Review / quality thresholds
"""

import re


# ─────────────────────────────────────────────
# Page type detection
# ─────────────────────────────────────────────

# Minimum characters extracted by pdfplumber for a page to be
# considered digital. Pages below this are treated as scanned/mixed.
DIGITAL_CHAR_THRESHOLD: int = 100

# Minimum characters to distinguish a MIXED page (partial text + image)
# from a SCANNED page (image only). Below this → SCANNED, above → MIXED.
MIXED_TEXT_MIN_CHARS: int = 20

# Fraction of non-ASCII characters above which text is considered garbled.
# Indian government PDFs with broken CID font maps often exceed this.
GARBLED_NON_ASCII_RATIO: float = 0.15

# Fraction of single-character tokens above which text is considered garbled.
# OCR misreads and font encoding failures produce many isolated characters.
GARBLED_SINGLE_CHAR_RATIO: float = 0.20


# ─────────────────────────────────────────────
# OCR — rasterization
# ─────────────────────────────────────────────

# DPI for rasterizing PDF pages before OCR.
# 200 is the sweet spot for legal text — high enough to catch fine print,
# low enough to keep memory and CPU manageable at scale.
# Increase to 300 for very small fonts or heavily degraded scans.
RASTERIZE_DPI: int = 200

# Tesseract page segmentation mode and engine:
#   --psm 6  → assume a single uniform block of text (best for legal body text)
#   --oem 3  → use LSTM neural net engine (most accurate for English govt text)
#   -l eng   → English language model
TESSERACT_CONFIG: str = "--psm 6 --oem 3 -l eng"


# ─────────────────────────────────────────────
# OCR — image preprocessing (OpenCV)
# ─────────────────────────────────────────────

# fastNlMeansDenoising strength. Higher = more aggressive noise removal,
# but risks blurring fine strokes. 10 works for most Indian govt scans.
DENOISE_H: int = 10

# Adaptive threshold block size — must be odd. Larger values handle more
# uneven lighting (e.g. shadow at binding edges). 31 covers most cases.
ADAPTIVE_THRESH_BLOCK_SIZE: int = 31

# Adaptive threshold constant subtracted from the mean. Higher = more
# pixels become white (lighter binarization). 10 is a safe default.
ADAPTIVE_THRESH_C: int = 10

# Dilation kernel size. (1,1) does a minimal dilation that reconnects
# broken character strokes without thickening text noticeably.
DILATE_KERNEL_SIZE: tuple[int, int] = (1, 1)

# Minimum skew angle (degrees) to correct. Smaller angles are negligible
# and correction introduces more interpolation artefacts than it removes.
MIN_DESKEW_ANGLE_DEG: float = 0.5

# Maximum credible skew angle. Beyond this the Hough detection is
# probably picking up table lines or borders, not text orientation.
MAX_DESKEW_ANGLE_DEG: float = 45.0

# Hough line transform threshold — minimum votes to count as a line.
# Lower = more sensitive (catches faint lines), higher = more selective.
HOUGH_THRESHOLD: int = 100


# ─────────────────────────────────────────────
# OCR — quality thresholds
# ─────────────────────────────────────────────

# Average Tesseract word confidence below which a page is flagged for
# manual review. Tesseract returns confidence 0–100; we normalise to 0–1.
OCR_CONFIDENCE_THRESHOLD: float = 60.0  # i.e. 60% → 0.60 after normalisation

# Confidence boost applied to hybrid (mixed) pages because they combine
# the higher-quality digital text with OCR. Caps at 0.9 to remain honest.
HYBRID_CONFIDENCE_BOOST: float = 0.10
HYBRID_CONFIDENCE_CAP: float = 0.90

# Pages with OCR confidence below this are marked needs_review=True in
# LegalSection, signalling that a human should verify before going live.
SECTION_REVIEW_CONFIDENCE_THRESHOLD: float = 0.65

# Minimum word count for a merged hybrid page's OCR text to be
# considered meaningfully different from the digital text (10% unique words).
HYBRID_UNIQUE_WORD_RATIO: float = 0.10

# Minimum word length to consider a word "unique" when merging digital
# and OCR text. Short words (articles, prepositions) are too noisy.
HYBRID_UNIQUE_WORD_MIN_LEN: int = 3


# ─────────────────────────────────────────────
# Text cleanup — post-OCR normalisation
# ─────────────────────────────────────────────

# Applied in order via re.sub on every extracted page, digital or OCR.
# Fixes form feeds, trailing whitespace, excess blank lines, OCR hyphenation
# artefacts, missing spaces after section numbers, and spaced-out characters.
CLEANUP_PATTERNS: list[tuple[str, str]] = [
    (r'\f',                         '\n'),        # form feed → newline
    (r'[ \t]+\n',                   '\n'),        # trailing whitespace
    (r'\n{3,}',                     '\n\n'),      # collapse excess blank lines
    (r'(?<=[a-z])-\n(?=[a-z])',     ''),          # dehyphenate mid-word breaks
    (r'\b(\d+)\s*\.\s*([A-Z])',     r'\1. \2'),   # "15.Eviction" → "15. Eviction"
    (r'\bS\s*e\s*c\s*t\s*i\s*o\s*n\b', 'Section'),  # fix spaced OCR chars
]

# Single-character OCR misreads common in legal text.
# Applied after CLEANUP_PATTERNS.
OCR_CHARACTER_FIXES: dict[str, str] = {
    r'\bl\b(?=\s+\d)':        '1',       # lowercase L misread as 1
    r'\bO\b(?=\s+\d)':        '0',       # uppercase O misread as 0
    r'§':                     'Section', # section symbol → word
    r'(?<=\d),(?=\d{3}\b)':   '',        # remove thousand separators in refs
}


# ─────────────────────────────────────────────
# Legal section parsing
# ─────────────────────────────────────────────

# Primary section header pattern — tried first on every document.
# Matches: "Section 15A — Security deposit"
#          "Sec. 15A. Security deposit"
#          "S. 15A Security deposit"
SECTION_PATTERN: re.Pattern = re.compile(
    r'^(?:Section|Sec\.|S\.)\s*(\d+[A-Za-z]?)'
    r'(?:\s*[.—–:-]\s*|\s+)'
    r'([A-Z][^\n]{3,80})',
    re.MULTILINE
)

# Fallback pattern — used when primary finds fewer than MIN_SECTION_MATCHES.
# Catches older acts (Delhi 1958) that omit the word "Section".
# Matches: "15A. Security deposit and its refund."
BARE_SECTION_PATTERN: re.Pattern = re.compile(
    r'^(\d+[A-Za-z]?)\.\s+([A-Z][^\n]{3,80})',
    re.MULTILINE
)

_PAGE_HEADER = re.compile(
    r'^\d{4}\s*:\s*\w+[\.\]]\s*\w'   # "2000: Mah. XVIII] The..."
    r'|^\w+\s+ACT\s+No\.'            # "MAHARASHTRA ACT No. XVIII"
    r'|\[\d{4}\s*:\s*\w+'            # "...[2000 : Mah." gazette ref anywhere in line
    r'|^\d+\s+The\s+\w+',            # "2 The Maharashtra..." page footer
    re.IGNORECASE
)

# If the primary pattern finds fewer than this many matches, the fallback
# is tried. Set to 3 rather than 1 to avoid treating stray numbered
# paragraphs as the entire section structure.
MIN_SECTION_MATCHES: int = 3

# Sections with fewer words than this are skipped — they're almost always
# chapter headers, "OMITTED" placeholders, or parser noise.
MIN_SECTION_WORDS: int = 10

# Pattern to locate the definitions block within an act.
# The definitions dict extracted from this block is injected into every chunk.
DEFINITIONS_PATTERN: re.Pattern = re.compile(
    r'^(?:DEFINITIONS?|Interpretation|Meaning of terms)',
    re.MULTILINE | re.IGNORECASE
)

# How many characters after the definitions header to scan for term definitions.
# Definitions sections in Indian acts are typically 500–2000 chars.
DEFINITIONS_SCAN_CHARS: int = 3000

# Minimum and maximum character length for a defined term.
DEFINITION_TERM_MIN_LEN: int = 3
DEFINITION_TERM_MAX_LEN: int = 40

# Minimum and maximum character length for a definition body.
DEFINITION_BODY_MIN_LEN: int = 20
DEFINITION_BODY_MAX_LEN: int = 400