from datetime import datetime, timezone
from pathlib import Path

from doclens.caption.cache import CaptionCache
from doclens.caption.chunker import generate_caption_chunks, load_all_cached_caption_chunks
from doclens.ingest.models import Document, License, Page, SourceRecord


def _document(doc_id: str, title: str, page_count: int) -> Document:
    source = SourceRecord(
        source_url=f"https://example.com/{doc_id}.pdf",
        license=License.PUBLIC_DOMAIN,
        retrieved_at=datetime(2026, 9, 17, tzinfo=timezone.utc),
        sha256=doc_id,
    )
    pages = [
        Page(page_number=i + 1, text=f"page {i + 1}", text_source="extracted",
             image_path=Path(f"corpus/{doc_id}/page-{i + 1}.png"))
        for i in range(page_count)
    ]
    return Document(doc_id=doc_id, title=title, source=source, pages=pages)


class _StubCaptioner:
    def __init__(self, captions: dict[int, str | None]):
        self.captions = captions
        self.call_count = 0

    def caption_page(self, image_path: Path) -> str | None:
        self.call_count += 1
        # image_path looks like corpus/<doc_id>/page-<N>.png
        page_number = int(image_path.stem.split("-")[1])
        return self.captions[page_number]


def test_generate_caption_chunks_produces_a_chunk_only_for_pages_with_content(tmp_path: Path):
    document = _document("a" * 64, "Manual A", page_count=3)
    cache = CaptionCache(corpus_dir=tmp_path)
    captioner = _StubCaptioner({1: "A torque table.", 2: None, 3: "A wiring diagram."})

    chunks = generate_caption_chunks(document, cache, captioner)

    assert len(chunks) == 2
    assert {c.page_number for c in chunks} == {1, 3}
    assert all(c.source_type == "caption" for c in chunks)


def test_generate_caption_chunks_persists_to_cache(tmp_path: Path):
    document = _document("a" * 64, "Manual A", page_count=1)
    cache = CaptionCache(corpus_dir=tmp_path)
    captioner = _StubCaptioner({1: "A torque table."})

    generate_caption_chunks(document, cache, captioner)

    assert cache.has("a" * 64, 1) is True
    assert cache.get("a" * 64, 1) == "A torque table."


def test_generate_caption_chunks_skips_already_cached_pages(tmp_path: Path):
    document = _document("a" * 64, "Manual A", page_count=1)
    cache = CaptionCache(corpus_dir=tmp_path)
    cache.set("a" * 64, 1, "Already cached caption.")
    captioner = _StubCaptioner({})  # would KeyError if called; proves it wasn't

    chunks = generate_caption_chunks(document, cache, captioner)

    assert captioner.call_count == 0
    assert len(chunks) == 1
    assert chunks[0].text == "Already cached caption."


def test_generate_caption_chunks_treats_empty_string_as_content_but_none_as_no_content(
    tmp_path: Path,
):
    document = _document("a" * 64, "Manual A", page_count=2)
    cache = CaptionCache(corpus_dir=tmp_path)
    captioner = _StubCaptioner({1: "", 2: None})

    chunks = generate_caption_chunks(document, cache, captioner)

    assert len(chunks) == 1
    assert chunks[0].page_number == 1
    assert chunks[0].text == ""


def test_load_all_cached_caption_chunks_reads_only_from_cache(tmp_path: Path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    document = _document("a" * 64, "Manual A", page_count=2)
    (corpus_dir / f"{document.doc_id}.json").write_text(
        document.model_dump_json(indent=2), encoding="utf-8"
    )
    cache = CaptionCache(corpus_dir=corpus_dir)
    cache.set("a" * 64, 1, "A torque table.")
    cache.set("a" * 64, 2, None)

    chunks = load_all_cached_caption_chunks(corpus_dir, cache)

    assert len(chunks) == 1
    assert chunks[0].page_number == 1
    assert chunks[0].text == "A torque table."


def test_load_all_cached_caption_chunks_returns_empty_for_uncached_corpus(tmp_path: Path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    document = _document("a" * 64, "Manual A", page_count=2)
    (corpus_dir / f"{document.doc_id}.json").write_text(
        document.model_dump_json(indent=2), encoding="utf-8"
    )
    cache = CaptionCache(corpus_dir=corpus_dir)

    chunks = load_all_cached_caption_chunks(corpus_dir, cache)

    assert chunks == []
