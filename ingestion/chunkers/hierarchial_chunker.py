from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class HierarchialChunker:
    def __init__(
            self, 
            child_size: int = 400, 
            child_overlap: int = 50, 
            parent_size: int = 1500, 
            parent_overlap: int = 100
        ):
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size, 
            chunk_overlap=child_overlap,
            separators=["\n\n", "\n", ". ", " "]
        )
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size, 
            chunk_overlap=parent_overlap
        )

    def chunk(self, section, definitions: dict) -> list[Document]:
        # Prepend relevant definitions to section content
        relevant_defs = self._extract_relevant_defs(section.content, definitions)

        enriched_content = relevant_defs + "\n\n" + section.content

        base_metadata = {
            "jurisdiction": section.jurisdiction,
            "act": section.act,
            "section_number": section.section_number,
            "section_title": section.section_title,
            "last_verified": section.last_verified,
            "chunk_type": "child"
        }

        # create parent chunk (full section + definitions)
        parent_doc = Document(
            page_content=enriched_content,
            metadata={**base_metadata, "chunk_type": "parent",
                        "section_id": f"{section.act}_{section.section_number}"}
        )
        

        # create child chunks
        child_texts = self.child_splitter.split_text(section.content)
        child_docs = []
        for i, chunk in enumerate(child_texts):
            child_docs.append(Document(
                page_content=chunk,
                metadata={**base_metadata, 
                    "parent_section_id": f"{section.act}_{section.section_number}",
                        "child_index": int(i)
                }
            ))
        return [parent_doc] + child_docs

    def _extract_relevant_defs(self, content: str, definitions: dict) -> str:
        """
        Find which defined terms appear in this section's content
        and return them as a formatted string to prepend to the chunk.
        Returns empty string if no definitions match.
        """
        if not definitions:
            return ""

        matched = {}
        content_lower = content.lower()
        for term, definition in definitions.items():
            if term.lower() in content_lower:
                matched[term] = definition

        if not matched:
            return ""

        lines = ["[Definitions relevant to this section]"]
        for term, definition in matched.items():
            lines.append(f'"{term}" means {definition}')

        return "\n".join(lines)
            