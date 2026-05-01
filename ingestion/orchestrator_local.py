"""
orchestrator_local.py
----------------------
Local development version of orchestrator.py.

The ONLY difference from orchestrator.py:
  - Reads PDFs from local disk instead of downloading from S3
  - No boto3 / AWS dependency

Everything else is identical:
  same parser → same chunker → same embedder → same Weaviate

When you move to production with S3, use orchestrator.py as-is.

Usage:
    python ingestion/orchestrator_local.py \
        --pdf data/acts/eng_maharashtra_rent_control_act_1999.pdf \
        --jurisdiction india_maharashtra \
        --act "Maharashtra Rent Control Act 1999" \
        --last_verified 2024-01

    # Ingest all registered PDFs at once
    python ingestion/orchestrator_local.py --all
"""

import sys
import argparse
import logging
from pathlib import Path

# ── path setup (same as orchestrator.py) ──────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ingestion" / "parsers"))
sys.path.insert(0, str(ROOT / "ingestion" / "chunkers"))
sys.path.insert(0, str(ROOT / "ingestion" / "embedders"))

from ingestion.parsers.pdf_parsers import LegalPDFParser
from ingestion.chunkers.hierarchial_chunker import HierarchialChunker
from ingestion.embedders.weaviate_embedder import WeaviateIngester

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ── same PDF registry as before ────────────────────────────────────────────────
PDF_REGISTRY = [
    {
        "filename":     "eng_maharashtra_rent_control_act_1999.pdf",
        "act":          "Maharashtra Rent Control Act 1999",
        "jurisdiction": "india_maharashtra",
        "last_verified": "2024-01",
    },
    {
        "filename":     "delhi-rent-control-act-1958.pdf",
        "act":          "Delhi Rent Control Act 1958",
        "jurisdiction": "india_delhi",
        "last_verified": "2024-01",
    },
    {
        "filename":     "60b7acb90a086Model-Tenancy-Act-English-02.06.2021.pdf",
        "act":          "Model Tenancy Act 2021",
        "jurisdiction": "india_general",
        "last_verified": "2024-01",
    },
    {
        "filename":     "2Draft_Model_Tenancy_Act.pdf",
        "act":          "Model Tenancy Act Draft",
        "jurisdiction": "india_general",
        "last_verified": "2021-06",
    },
    {
        "filename":     "uk_tenancy_act,_2021_eng.pdf",
        "act":          "Uttarakhand Tenancy Act 2021",
        "jurisdiction": "india_uttarakhand",
        "last_verified": "2024-01",
    },
    {
        "filename":     "FAQs-on-RERA(1).pdf",
        "act":          "RERA FAQs",
        "jurisdiction": "india_general",
        "last_verified": "2024-01",
    },
    {
        "filename":     "FAQs-on-RERA.pdf",
        "act":          "RERA FAQs",
        "jurisdiction": "india_general",
        "last_verified": "2024-01",
    },
]


class IngestionOrchestratorLocal:
    """
    Identical to IngestionOrchestrator except ingest_pdf()
    reads from local disk instead of S3.
    """

    def __init__(self):
        self.parser   = LegalPDFParser()
        self.chunker  = HierarchialChunker()
        self.ingester = WeaviateIngester()

    def ingest_pdf(self, local_path: str, metadata: dict):
        """
        local_path  : full path to PDF on disk
        metadata    : {"act": ..., "jurisdiction": ..., "last_verified": ...}

        Difference from orchestrator.py:
          orchestrator.py     → self.s3.download_file(bucket, s3_key, local_path)
          orchestrator_local  → uses local_path directly, no download needed
        """
        path = Path(local_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {local_path}")

        logger.info(f"Parsing:   {path.name}")
        sections    = self.parser.parse(str(path), metadata)

        if not sections:
            logger.warning(f"  No sections extracted from {path.name} — skipping")
            return

        definitions = self.parser._extract_definitions(
            " ".join(s.content for s in sections[:5])
        )

        logger.info(f"  {len(sections)} sections, {len(definitions)} definitions")

        logger.info(f"Chunking:  {path.name}")
        all_docs = []
        for section in sections:
            all_docs.extend(self.chunker.chunk(section, definitions))

        parents  = sum(1 for d in all_docs if d.metadata.get("chunk_type") == "parent")
        children = sum(1 for d in all_docs if d.metadata.get("chunk_type") == "child")
        logger.info(f"  {parents} parents + {children} children = {len(all_docs)} chunks")

        logger.info(f"Ingesting: {path.name}")
        self.ingester.ingest_batch(all_docs)
        logger.info(f"  Done — {len(all_docs)} chunks ingested\n")


def main():
    parser = argparse.ArgumentParser(
        description="Local orchestrator — ingest PDFs from disk into Weaviate"
    )
    parser.add_argument("--pdf",          type=str, help="Path to a single PDF")
    parser.add_argument("--jurisdiction", type=str, help="Jurisdiction code e.g. india_maharashtra")
    parser.add_argument("--act",          type=str, help="Act name")
    parser.add_argument("--last_verified",type=str, default="2024-01", help="YYYY-MM")
    parser.add_argument("--all",          action="store_true",
                        help="Ingest all PDFs in the registry from data/acts/")
    parser.add_argument("--acts-dir",     type=str,
                        default=str(ROOT / "data" / "acts"),
                        help="Directory containing PDFs (default: data/acts/)")
    args = parser.parse_args()

    # load .env
    env_file = ROOT / ".env"
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)

    orchestrator = IngestionOrchestratorLocal()

    if args.all:
        acts_dir = Path(args.acts_dir)
        logger.info(f"Ingesting all PDFs from {acts_dir}")
        for entry in PDF_REGISTRY:
            pdf_path = acts_dir / entry["filename"]
            if not pdf_path.exists():
                logger.warning(f"Not found, skipping: {entry['filename']}")
                continue
            metadata = {
                "act":           entry["act"],
                "jurisdiction":  entry["jurisdiction"],
                "last_verified": entry["last_verified"],
            }
            orchestrator.ingest_pdf(str(pdf_path), metadata)

    elif args.pdf:
        if not args.jurisdiction or not args.act:
            parser.error("--jurisdiction and --act are required with --pdf")
        metadata = {
            "act":           args.act,
            "jurisdiction":  args.jurisdiction,
            "last_verified": args.last_verified,
        }
        orchestrator.ingest_pdf(args.pdf, metadata)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()