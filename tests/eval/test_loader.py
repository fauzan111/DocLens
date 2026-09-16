import json
from pathlib import Path

from doclens.eval.loader import load_benchmark_draft


def _write_draft(path: Path, entries: list[dict]) -> None:
    path.write_text(json.dumps(entries), encoding="utf-8")


def test_load_benchmark_draft_parses_answerable_and_unanswerable(tmp_path: Path):
    draft_path = tmp_path / "draft.json"
    _write_draft(draft_path, [
        {
            "id": "bq-001",
            "slice": "answerable_text",
            "language": "en",
            "question": "What temperature triggers a shutdown?",
            "expected_answer": "190 degrees F",
            "sources": [{"doc_id": "a" * 64, "doc_title": "Manual", "page_number": 13}],
            "notes": "",
        },
        {
            "id": "bq-002",
            "slice": "unanswerable",
            "language": "en",
            "question": "What is the price?",
            "expected_answer": None,
            "sources": [],
            "notes": "",
        },
    ])

    questions = load_benchmark_draft(draft_path)

    assert len(questions) == 2
    assert questions[0].id == "bq-001"
    assert questions[0].sources[0].page_number == 13
    assert questions[1].expected_answer is None


def test_load_benchmark_draft_parses_cross_document_multi_source(tmp_path: Path):
    draft_path = tmp_path / "draft.json"
    _write_draft(draft_path, [
        {
            "id": "bq-050",
            "slice": "cross_document",
            "language": "en",
            "question": "Do these share a rating?",
            "expected_answer": "Yes",
            "sources": [
                {"doc_id": "a" * 64, "doc_title": "Doc A", "page_number": 5},
                {"doc_id": "b" * 64, "doc_title": "Doc B", "page_number": 6},
            ],
            "notes": "",
        },
    ])

    questions = load_benchmark_draft(draft_path)

    assert len(questions[0].sources) == 2
