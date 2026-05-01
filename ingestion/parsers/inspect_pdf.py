"""
inspect_pdf.py
--------------
CLI diagnostic tool — run this on any PDF BEFORE ingesting it.
Tells you: page types, OCR confidence, section count, problem pages.

Usage:
    python inspect_pdf.py path/to/act.pdf
    python inspect_pdf.py path/to/act.pdf --page 5   # inspect single page
    python inspect_pdf.py path/to/act.pdf --save      # save extracted text
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Configure logging so you can see what's happening
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)

from ingestion.parsers.ocr_handler import GovernmentPDFOCRHandler, PageType


def inspect_document(pdf_path: str, save_text: bool = False):
    handler = GovernmentPDFOCRHandler(rasterize_dpi=200, preprocess_scans=True)
    result = handler.process_document(pdf_path)

    print(f"\n{'═'*60}")
    print(f"  DOCUMENT INSPECTION REPORT")
    print(f"{'═'*60}")
    print(f"  File       : {Path(pdf_path).name}")
    print(f"  Pages      : {result.total_pages}")
    print(f"  Digital    : {result.digital_pages} pages ({result.digital_pages/result.total_pages:.0%})")
    print(f"  Scanned    : {result.scanned_pages} pages ({result.scanned_pages/result.total_pages:.0%})")
    print(f"  OCR avg    : {result.avg_ocr_confidence:.1%}")
    print(f"  Words      : {len(result.full_text.split()):,}")
    print(f"  Failed     : {result.failed_pages or 'none'}")

    # Flag problem pages
    low_conf = [
        p for p in result.pages
        if p.confidence < 0.65 and p.page_type != PageType.EMPTY
    ]
    if low_conf:
        print(f"\n  ⚠ LOW CONFIDENCE PAGES (manual review recommended):")
        for p in low_conf:
            print(f"    Page {p.page_number}: {p.confidence:.0%} ({p.extraction_method})")

    # Show warnings
    all_warnings = [
        (p.page_number, w)
        for p in result.pages
        for w in p.warnings
    ]
    if all_warnings:
        print(f"\n  ⚠ WARNINGS:")
        for page_num, warning in all_warnings[:10]:   # cap at 10
            print(f"    Page {page_num}: {warning}")

    # Show page type breakdown
    print(f"\n  PAGE TYPE BREAKDOWN:")
    type_counts = {}
    for p in result.pages:
        type_counts[p.page_type.value] = type_counts.get(p.page_type.value, 0) + 1
    for ptype, count in sorted(type_counts.items()):
        bar = "█" * min(40, count)
        print(f"    {ptype:10s}: {count:3d}  {bar}")

    # Show first 500 chars of extracted text
    print(f"\n  FIRST 500 CHARS OF EXTRACTED TEXT:")
    print(f"  {'─'*56}")
    preview = result.full_text[:500].replace('\n', '\n  ')
    print(f"  {preview}")
    print(f"  {'─'*56}")

    if save_text:
        out_path = Path(pdf_path).with_suffix(".txt")
        out_path.write_text(result.full_text, encoding="utf-8")
        print(f"\n  ✓ Full text saved to: {out_path}")

    return result


def inspect_single_page(pdf_path: str, page_number: int):
    handler = GovernmentPDFOCRHandler(rasterize_dpi=200, preprocess_scans=True)
    result = handler.process_page(pdf_path, page_number)

    print(f"\n{'═'*60}")
    print(f"  PAGE {page_number} INSPECTION")
    print(f"{'═'*60}")
    print(f"  Type       : {result.page_type.value}")
    print(f"  Method     : {result.extraction_method}")
    print(f"  Confidence : {result.confidence:.1%}")
    print(f"  Words      : {result.word_count}")
    if result.warnings:
        print(f"  Warnings   : {'; '.join(result.warnings)}")
    print(f"\n  EXTRACTED TEXT:")
    print(f"  {'─'*56}")
    print(f"  {result.text[:1000].replace(chr(10), chr(10)+'  ')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect a government PDF before ingestion"
    )
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--page", type=int, help="Inspect a specific page number")
    parser.add_argument("--save", action="store_true", help="Save extracted text to .txt")
    args = parser.parse_args()

    if not Path(args.pdf_path).exists():
        print(f"Error: File not found: {args.pdf_path}")
        sys.exit(1)

    if args.page:
        inspect_single_page(args.pdf_path, args.page)
    else:
        inspect_document(args.pdf_path, save_text=args.save)