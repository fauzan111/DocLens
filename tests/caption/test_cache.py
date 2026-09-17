from pathlib import Path

from doclens.caption.cache import CaptionCache


def test_has_returns_false_for_never_processed_page(tmp_path: Path):
    cache = CaptionCache(corpus_dir=tmp_path)
    assert cache.has("doc1", 5) is False


def test_set_and_get_a_real_caption(tmp_path: Path):
    cache = CaptionCache(corpus_dir=tmp_path)
    cache.set("doc1", 5, "A table showing torque values by bolt size.")

    assert cache.has("doc1", 5) is True
    assert cache.get("doc1", 5) == "A table showing torque values by bolt size."


def test_set_and_get_a_no_visual_content_page(tmp_path: Path):
    cache = CaptionCache(corpus_dir=tmp_path)
    cache.set("doc1", 3, None)

    assert cache.has("doc1", 3) is True
    assert cache.get("doc1", 3) is None


def test_cache_persists_across_instances(tmp_path: Path):
    cache_a = CaptionCache(corpus_dir=tmp_path)
    cache_a.set("doc1", 5, "A caption.")

    cache_b = CaptionCache(corpus_dir=tmp_path)
    assert cache_b.has("doc1", 5) is True
    assert cache_b.get("doc1", 5) == "A caption."


def test_cache_keeps_documents_separate(tmp_path: Path):
    cache = CaptionCache(corpus_dir=tmp_path)
    cache.set("doc1", 5, "Doc1 caption.")
    cache.set("doc2", 5, "Doc2 caption.")

    assert cache.get("doc1", 5) == "Doc1 caption."
    assert cache.get("doc2", 5) == "Doc2 caption."


def test_cache_accumulates_multiple_pages_for_same_document(tmp_path: Path):
    cache = CaptionCache(corpus_dir=tmp_path)
    cache.set("doc1", 5, "Page 5 caption.")
    cache.set("doc1", 8, "Page 8 caption.")

    assert cache.get("doc1", 5) == "Page 5 caption."
    assert cache.get("doc1", 8) == "Page 8 caption."
    assert cache.has("doc1", 3) is False
