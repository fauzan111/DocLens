from datetime import datetime, timezone
from pathlib import Path

from doclens.eval.grounding import validate_grounding
from doclens.eval.models import BenchmarkQuestion, Source
from doclens.ingest.models import Document, License, Page, SourceRecord


def _write_document(corpus_dir: Path, doc_id: str, title: str, page_count: int) -> None:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    source = SourceRecord(
        source_url=f"https://example.com/{doc_id}.pdf",
        license=License.PUBLIC_DOMAIN,
        retrieved_at=datetime(2026, 9, 16, tzinfo=timezone.utc),
        sha256=doc_id,
    )
    pages = [
        Page(page_number=i + 1, text=f"page {i + 1}", text_source="extracted",
             image_path=Path(f"corpus/{doc_id}/page-{i + 1}.png"))
        for i in range(page_count)
    ]
    document = Document(doc_id=doc_id, title=title, source=source, pages=pages)
    (corpus_dir / f"{doc_id}.json").write_text(document.model_dump_json(indent=2), encoding="utf-8")


def test_validate_grounding_passes_for_a_real_doc_and_page(tmp_path: Path):
    corpus_dir = tmp_path / "corpus"
    _write_document(corpus_dir, "a" * 64, "Manual A", page_count=20)

    question = BenchmarkQuestion(
        id="bq-001", slice="answerable_text", language="en",
        question="Q?", expected_answer="A",
        sources=[Source(doc_id="a" * 64, doc_title="Manual A", page_number=13)],
    )

    issues = validate_grounding([question], corpus_dir)

    assert issues == []


def test_validate_grounding_flags_unknown_doc_id(tmp_path: Path):
    corpus_dir = tmp_path / "corpus"
    _write_document(corpus_dir, "a" * 64, "Manual A", page_count=20)

    question = BenchmarkQuestion(
        id="bq-002", slice="answerable_text", language="en",
        question="Q?", expected_answer="A",
        sources=[Source(doc_id="b" * 64, doc_title="Manual B", page_number=1)],
    )

    issues = validate_grounding([question], corpus_dir)

    assert len(issues) == 1
    assert issues[0].question_id == "bq-002"
    assert "not found" in issues[0].reason


def test_validate_grounding_flags_out_of_range_page_number(tmp_path: Path):
    corpus_dir = tmp_path / "corpus"
    _write_document(corpus_dir, "a" * 64, "Manual A", page_count=20)

    question = BenchmarkQuestion(
        id="bq-003", slice="answerable_text", language="en",
        question="Q?", expected_answer="A",
        sources=[Source(doc_id="a" * 64, doc_title="Manual A", page_number=999)],
    )

    issues = validate_grounding([question], corpus_dir)

    assert len(issues) == 1
    assert "999" in issues[0].reason
    assert "20" in issues[0].reason


def test_validate_grounding_passes_unanswerable_question_trivially(tmp_path: Path):
    corpus_dir = tmp_path / "corpus"
    _write_document(corpus_dir, "a" * 64, "Manual A", page_count=20)

    question = BenchmarkQuestion(
        id="bq-004", slice="unanswerable", language="en",
        question="What is the price?", expected_answer=None, sources=[],
    )

    issues = validate_grounding([question], corpus_dir)

    assert issues == []
