from pathlib import Path

from doclens.caption.cache import CaptionCache
from doclens.caption.captioner import PageCaptioner
from doclens.ingest.corpus import load_documents
from doclens.ingest.models import Document
from doclens.retrieval.models import Chunk


def _caption_chunk(document: Document, page_number: int, caption: str) -> Chunk:
    return Chunk(
        chunk_id=f"{document.doc_id[:16]}:{page_number}:caption",
        doc_id=document.doc_id,
        doc_title=document.title,
        page_number=page_number,
        chunk_index=0,
        text=caption,
        source_url=document.source.source_url,
        source_type="caption",
    )


def generate_caption_chunks(
    document: Document, cache: CaptionCache, captioner: PageCaptioner | None = None
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in document.pages:
        if cache.has(document.doc_id, page.page_number):
            caption = cache.get(document.doc_id, page.page_number)
        else:
            resolved_captioner = captioner or PageCaptioner()
            caption = resolved_captioner.caption_page(page.image_path)
            cache.set(document.doc_id, page.page_number, caption)

        if caption is not None:
            chunks.append(_caption_chunk(document, page.page_number, caption))
    return chunks


def load_all_cached_caption_chunks(corpus_dir: Path, cache: CaptionCache) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in load_documents(corpus_dir):
        for page in document.pages:
            if cache.has(document.doc_id, page.page_number):
                caption = cache.get(document.doc_id, page.page_number)
                if caption is not None:
                    chunks.append(_caption_chunk(document, page.page_number, caption))
    return chunks
