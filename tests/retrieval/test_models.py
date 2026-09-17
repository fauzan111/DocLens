from doclens.retrieval.models import Chunk


def test_chunk_source_type_defaults_to_extracted_text():
    chunk = Chunk(chunk_id="c1", doc_id="doc1", doc_title="Manual", page_number=1,
                  chunk_index=0, text="some text", source_url="https://example.com/a.pdf")
    assert chunk.source_type == "extracted_text"


def test_chunk_source_type_accepts_caption():
    chunk = Chunk(chunk_id="c1:caption", doc_id="doc1", doc_title="Manual", page_number=1,
                  chunk_index=0, text="a table showing X", source_url="https://example.com/a.pdf",
                  source_type="caption")
    assert chunk.source_type == "caption"
