from datetime import datetime, timezone
from pathlib import Path

import pytest

from doclens.ingest.models import Document, License, Page, SourceRecord
from doclens.retrieval.chunker import chunk_document


def _document(pages: list[Page]) -> Document:
    source = SourceRecord(
        source_url="https://example.com/manual.pdf",
        license=License.PUBLIC_DOMAIN,
        retrieved_at=datetime(2026, 9, 10, tzinfo=timezone.utc),
        sha256="a" * 64,
    )
    return Document(doc_id="a" * 64, title="Test Manual", source=source, pages=pages)


def test_chunk_document_produces_one_chunk_for_short_page():
    page = Page(page_number=1, text="Torque spec: 45 Nm", text_source="extracted",
                image_path=Path("corpus/x/page-1.png"))
    chunks = chunk_document(_document([page]), chunk_size=800, overlap=100)

    assert len(chunks) == 1
    assert chunks[0].page_number == 1
    assert chunks[0].chunk_index == 0
    assert "Torque spec: 45 Nm" in chunks[0].text
    assert chunks[0].doc_title == "Test Manual"
    assert chunks[0].source_url == "https://example.com/manual.pdf"


def test_chunk_document_produces_overlapping_chunks_for_long_page():
    long_text = "word " * 500  # far longer than chunk_size
    page = Page(page_number=1, text=long_text, text_source="extracted",
                image_path=Path("corpus/x/page-1.png"))
    chunks = chunk_document(_document([page]), chunk_size=100, overlap=20)

    assert len(chunks) > 1
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    # Overlap: the tail of chunk 0 should reappear at the head of chunk 1.
    assert chunks[0].text[-10:] in chunks[1].text


def test_chunk_document_skips_empty_text_pages():
    blank_page = Page(page_number=1, text="   ", text_source="extracted",
                       image_path=Path("corpus/x/page-1.png"))
    text_page = Page(page_number=2, text="Real content here", text_source="extracted",
                      image_path=Path("corpus/x/page-2.png"))
    chunks = chunk_document(_document([blank_page, text_page]))

    assert len(chunks) == 1
    assert chunks[0].page_number == 2


def test_chunk_document_never_merges_text_across_page_boundary():
    page1 = Page(page_number=1, text="End of page one.", text_source="extracted",
                 image_path=Path("corpus/x/page-1.png"))
    page2 = Page(page_number=2, text="Start of page two.", text_source="extracted",
                 image_path=Path("corpus/x/page-2.png"))
    chunks = chunk_document(_document([page1, page2]), chunk_size=800, overlap=100)

    assert len(chunks) == 2
    assert "End of page one" in chunks[0].text
    assert "Start of page two" not in chunks[0].text
    assert chunks[0].page_number == 1
    assert chunks[1].page_number == 2


def test_chunk_document_rejects_overlap_not_smaller_than_chunk_size():
    page = Page(page_number=1, text="some text", text_source="extracted",
                image_path=Path("corpus/x/page-1.png"))
    with pytest.raises(ValueError):
        chunk_document(_document([page]), chunk_size=100, overlap=100)
