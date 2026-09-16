from doclens.eval.models import BenchmarkQuestion
from doclens.eval.split import assign_splits, check_contamination


def _question(id_: str, slice_: str, question_text: str = "placeholder") -> BenchmarkQuestion:
    return BenchmarkQuestion(
        id=id_, slice=slice_, language="en", question=question_text,
        expected_answer="A" if slice_ != "unanswerable" else None,
        sources=[] if slice_ == "unanswerable" else [
            {"doc_id": "a" * 64, "doc_title": "Manual", "page_number": 1}
        ],
    )


def test_assign_splits_produces_roughly_80_20_split():
    questions = [_question(f"bq-{i:03d}", "answerable_text") for i in range(20)]

    split_questions = assign_splits(questions, hidden_fraction=0.2, seed=42)

    hidden_count = sum(1 for q in split_questions if q.split == "hidden")
    dev_count = sum(1 for q in split_questions if q.split == "dev")
    assert hidden_count == 4
    assert dev_count == 16


def test_assign_splits_is_deterministic_given_same_seed():
    questions = [_question(f"bq-{i:03d}", "answerable_text") for i in range(20)]

    first = assign_splits(questions, seed=42)
    second = assign_splits(questions, seed=42)

    assert [q.split for q in first] == [q.split for q in second]


def test_assign_splits_stratifies_per_slice():
    # A tiny slice (3 questions) should still get at least one hidden question,
    # not have all 3 luckily land in dev by chance of a global (non-stratified) split.
    questions = (
        [_question(f"bq-{i:03d}", "answerable_text") for i in range(20)]
        + [_question(f"bq-sc-{i}", "scanned_no_text_layer") for i in range(3)]
    )

    split_questions = assign_splits(questions, hidden_fraction=0.2, seed=42)

    scanned_hidden = [
        q for q in split_questions if q.slice == "scanned_no_text_layer" and q.split == "hidden"
    ]
    assert len(scanned_hidden) >= 1


def test_check_contamination_flags_near_duplicate_questions_across_splits():
    dev_question = _question(
        "bq-dev", "answerable_text",
        "What temperature must the bearing housing reach before the pump is stopped immediately?",
    )
    dev_question = dev_question.model_copy(update={"split": "dev"})
    hidden_question = _question(
        "bq-hidden", "answerable_text",
        "At what temperature must the bearing housing be before the pump is stopped immediately?",
    )
    hidden_question = hidden_question.model_copy(update={"split": "hidden"})

    issues = check_contamination([dev_question, hidden_question], overlap_threshold=0.4)

    assert len(issues) == 1
    assert issues[0].dev_id == "bq-dev"
    assert issues[0].hidden_id == "bq-hidden"


def test_check_contamination_does_not_flag_unrelated_questions():
    dev_question = _question("bq-dev", "answerable_text", "What is the bearing temperature limit?")
    dev_question = dev_question.model_copy(update={"split": "dev"})
    hidden_question = _question("bq-hidden", "table_lookup", "What torque applies to the M12 bolt?")
    hidden_question = hidden_question.model_copy(update={"split": "hidden"})

    issues = check_contamination([dev_question, hidden_question], overlap_threshold=0.5)

    assert issues == []
