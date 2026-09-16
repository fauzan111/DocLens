import pytest
from pydantic import ValidationError

from doclens.eval.models import BenchmarkQuestion, Source


def test_source_requires_positive_page_number():
    with pytest.raises(ValidationError):
        Source(doc_id="a" * 64, doc_title="Manual", page_number=0)


def test_answerable_question_with_source_is_valid():
    question = BenchmarkQuestion(
        id="bq-001",
        slice="answerable_text",
        language="en",
        question="What temperature triggers a shutdown?",
        expected_answer="190 degrees F",
        sources=[Source(doc_id="a" * 64, doc_title="Manual", page_number=13)],
    )
    assert question.expected_answer == "190 degrees F"
    assert question.split is None


def test_unanswerable_question_must_have_no_answer_and_no_sources():
    question = BenchmarkQuestion(
        id="bq-057",
        slice="unanswerable",
        language="en",
        question="What is the maximum pressure rating?",
        expected_answer=None,
        sources=[],
    )
    assert question.expected_answer is None


def test_unanswerable_question_rejects_a_non_null_answer():
    with pytest.raises(ValidationError):
        BenchmarkQuestion(
            id="bq-057",
            slice="unanswerable",
            language="en",
            question="What is the maximum pressure rating?",
            expected_answer="100 bar",
            sources=[],
        )


def test_unanswerable_question_rejects_a_non_empty_sources_list():
    with pytest.raises(ValidationError):
        BenchmarkQuestion(
            id="bq-057",
            slice="unanswerable",
            language="en",
            question="What is the maximum pressure rating?",
            expected_answer=None,
            sources=[Source(doc_id="a" * 64, doc_title="Manual", page_number=1)],
        )


def test_cross_document_question_accepts_multiple_sources():
    question = BenchmarkQuestion(
        id="bq-050",
        slice="cross_document",
        language="en",
        question="Do these three products share an IP rating?",
        expected_answer="Yes, IP20",
        sources=[
            Source(doc_id="a" * 64, doc_title="Doc A", page_number=5),
            Source(doc_id="b" * 64, doc_title="Doc B", page_number=6),
            Source(doc_id="c" * 64, doc_title="Doc C", page_number=46),
        ],
    )
    assert len(question.sources) == 3
