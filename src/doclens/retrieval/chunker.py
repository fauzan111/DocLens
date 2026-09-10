from doclens.ingest.models import Document
from doclens.retrieval.models import Chunk


def chunk_document(document: Document, chunk_size: int = 800, overlap: int = 100) -> list[Chunk]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    for page in document.pages:
        text = page.text.strip()
        if not text:
            continue

        start = 0
        index = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.doc_id[:16]}:{page.page_number}:{index}",
                        doc_id=document.doc_id,
                        doc_title=document.title,
                        page_number=page.page_number,
                        chunk_index=index,
                        text=chunk_text,
                        source_url=document.source.source_url,
                    )
                )
                index += 1
            if end >= len(text):
                break
            start = end - overlap

    return chunks
