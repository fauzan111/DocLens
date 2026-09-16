from doclens.eval.dataset_card import generate_benchmark_card
from doclens.eval.models import BenchmarkQuestion


def _question(id_: str, slice_: str, language: str, split: str) -> BenchmarkQuestion:
    return BenchmarkQuestion(
        id=id_, slice=slice_, language=language, question="placeholder",
        expected_answer="A" if slice_ != "unanswerable" else None,
        sources=[] if slice_ == "unanswerable" else [
            {"doc_id": "a" * 64, "doc_title": "Manual", "page_number": 1}
        ],
        split=split,
    )


def test_dataset_card_reports_total_and_slice_breakdown():
    questions = [
        _question("bq-001", "answerable_text", "en", "dev"),
        _question("bq-002", "answerable_text", "en", "hidden"),
        _question("bq-003", "table_lookup", "it", "dev"),
    ]

    card = generate_benchmark_card(questions)

    assert "Total questions: 3" in card
    assert "answerable_text" in card
    assert "table_lookup" in card


def test_dataset_card_reports_language_breakdown():
    questions = [
        _question("bq-001", "answerable_text", "en", "dev"),
        _question("bq-002", "answerable_text", "it", "dev"),
    ]

    card = generate_benchmark_card(questions)

    assert "en: 1" in card
    assert "it: 1" in card


def test_dataset_card_documents_the_scanned_page_limitation():
    questions = [_question("bq-001", "answerable_text", "en", "dev")]

    card = generate_benchmark_card(questions)

    assert "scanned_no_text_layer" in card
    assert "Interroll" in card
