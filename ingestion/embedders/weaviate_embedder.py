from langchain_openai import OpenAIEmbeddings
import weaviate, weaviate.classes as wvc
from config.settings import settings

class WeaviateIngester:
    COLLECTION = settings.WEAVIATE_INDEX

    def __init__(self):
        _host = settings.WEAVIATE_URL.replace("http://", "").replace("https://", "")
        _host, _port = (_host.split(":")[0], int(_host.split(":")[1])) \
                    if ":" in _host else (_host, 8080)

        self.client = weaviate.connect_to_local(
            host=_host,    # "localhost"
            port=_port,    # 8080
        )

        self.embedder = OpenAIEmbeddings(
            model=settings.OPENAI_EMBED_MODEL,
            dimensions=settings.OPENAI_EMBED_DIMS,
            openai_api_key=settings.OPENAI_API_KEY
        )
        self._ensure_schema()

    def _ensure_schema(self):
        if not self.client.collections.exists(self.COLLECTION):
            self.client.collections.create(
                name = self.COLLECTION,
                vectorizer_config=wvc.config.Configure.Vectorizer.none(),
                properties=[
                    wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="jurisdiction", data_type=wvc.config.DataType.TEXT, index_filterable=True, index_searchable=True),
                    wvc.config.Property(name="act", data_type=wvc.config.DataType.TEXT, index_filterable=True),
                    wvc.config.Property(name="section_number", data_type=wvc.config.DataType.TEXT, index_filterable=True),
                    wvc.config.Property(name="section_id", data_type=wvc.config.DataType.TEXT, index_filterable=True, index_searchable=False),
                    wvc.config.Property(name="section_title", data_type=wvc.config.DataType.TEXT, index_filterable=False, index_searchable=True),
                    wvc.config.Property(name="chunk_type", data_type=wvc.config.DataType.TEXT, index_filterable=True),
                    wvc.config.Property(name="last_verified", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="parent_section_id", data_type=wvc.config.DataType.TEXT, index_filterable=True),
                    wvc.config.Property(name="child_index", data_type=wvc.config.DataType.INT, index_filterable=True),
                ]
            )

    def ingest_batch(self, docs: list, batch_size: int = 100):
        collection = self.client.collections.get(self.COLLECTION)
        texts = [doc.page_content for doc in docs]

        # batch embed - openAI allows upto 2048 inputs per call
        vectors = self.embedder.embed_documents(texts)

        with collection.batch.fixed_size(batch_size=batch_size) as batch:
            for doc, vector in zip(docs, vectors):
                batch.add_object(
                    properties={
                        "text": doc.page_content,
                        **doc.metadata
                    },
                    vector=vector
                )