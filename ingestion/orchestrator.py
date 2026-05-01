import boto3
from pathlib import Path
from ingestion.parsers.pdf_parsers import LegalPDFParser
from ingestion.chunkers.hierarchial_chunker import HierarchialChunker
from ingestion.embedders.weaviate_embedder import WeaviateIngester

class IngestionOrchestrator:
    def __init__(self):
        self.parser = LegalPDFParser()
        self.chunker = HierarchialChunker()
        self.ingester = WeaviateIngester()
        self.s3 = boto3.client('s3')

    def ingest_pdf(self, s3_key: str, metadata: dict):
        # Download from S3
        local_path = f"/tmp/{Path(s3_key).name}"
        self.s3.download_file("tenant-rag-docs", s3_key, local_path)

        # Parse into sections
        sections = self.parser.parse(local_path, metadata)
        definitions = self.parser._extract_definitions(sections)

        # chunk each sections
        all_docs = []
        for section in sections:
            all_docs.extend(self.chunker.chunk(section, definitions))
        
        # Embed and ingest into Weaviate
        self.ingester.ingest_batch(all_docs)
        print(f"Ingested {len(all_docs)} chunks from {s3_key} into Weaviate")


IngestionOrchestrator()