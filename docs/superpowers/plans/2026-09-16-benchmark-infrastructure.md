# DocLens Benchmark Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the 66 already-drafted, content-grounded benchmark questions into the actual
versioned `doclens-bench` artifact: a typed schema, a loader, an automated grounding validator,
a frozen dev/hidden split with a contamination check, and a dataset card. This is what
Milestone 3's retrieval arms will be measured against.

**Architecture:** A typed `BenchmarkQuestion`/`Source` model (Pydantic) backs a loader that reads
the raw draft JSON. A `GroundingValidator` checks every question's cited document/page actually
exists in the ingested corpus (the cheap, reliable, automatable safety net; visual/table/diagram
correctness was already verified by direct image inspection during drafting and is out of scope
for automated re-verification here). A deterministic 80/20 dev/hidden splitter assigns each
question a `split`, and a contamination check flags near-duplicate questions that leaked across
the split. A dataset-card generator writes a markdown summary, including an explicit,
undisguised note about the corpus's scanned-page limitation. A CLI ties it together.

**Tech Stack:** Python 3.11+ (existing), Pydantic v2 (existing), no new dependencies.

## Global Constraints

- Never use the em dash character (`—`, U+2014) anywhere: code, comments, docstrings, commit
  messages, reports.
- All file paths in code are `pathlib.Path`, never raw strings (Windows dev environment).
- No Anthropic/Claude API calls anywhere in this pipeline (this task has no LLM calls at all;
  it's pure validation/data-processing code).
- The benchmark's ground truth (question text, expected answers, citations) was already
  human/agent-drafted and grounded by directly reading real corpus content during an earlier
  content-curation pass; this plan's code must not alter question text or expected answers, only
  validate, split, and report on them.
- Tests must never require network access or a real model download to pass.

---

## Input data

The consolidated draft lives at
`benchmarks/doclens-bench/_drafts/consolidated_draft.json`: a JSON array of 66 objects, each
shaped like:

```json
{
  "id": "bq-001",
  "slice": "answerable_text",
  "language": "en",
  "question": "...",
  "expected_answer": "190 degrees F (88 degrees C)",
  "sources": [
    {"doc_id": "3cba77928150108b...", "doc_title": "...", "page_number": 13}
  ],
  "notes": "..."
}
```

`slice` is one of: `answerable_text`, `table_lookup`, `diagram_lookup`,
`scanned_no_text_layer`, `cross_document`, `unanswerable`. `expected_answer` is `null` and
`sources` is `[]` for `unanswerable` questions. `sources` has 2-3 entries for
`cross_document` questions, 1 entry otherwise.

---

## File Structure

```
DocLens/
  src/doclens/
    eval/
      __init__.py
      models.py           # Task 1: Source, BenchmarkQuestion
      loader.py           # Task 1: load_benchmark_draft()
      grounding.py        # Task 2: GroundingValidator
      split.py            # Task 3: assign_splits(), check_contamination()
      dataset_card.py     # Task 4: generate_benchmark_card()
    cli.py                # Task 4: `doclens bench-validate`, `doclens bench-freeze` (extended)
  benchmarks/doclens-bench/
    _drafts/
      consolidated_draft.json   # input (already exists)
    questions.json              # Task 3/4 output: the frozen, split, versioned benchmark
    DATASET_CARD.md              # Task 4 output
  tests/
    eval/
      test_models.py       # Task 1
      test_loader.py        # Task 1
      test_grounding.py     # Task 2
      test_split.py          # Task 3
      test_dataset_card.py  # Task 4
```

---

### Task 1: Benchmark question model + loader

**Files:**
- Create: `src/doclens/eval/__init__.py` (empty)
- Create: `src/doclens/eval/models.py`
- Create: `src/doclens/eval/loader.py`
- Test: `tests/eval/test_models.py`
- Test: `tests/eval/test_loader.py`

**Interfaces:**
- Produces: `Source(doc_id: str, doc_title: str, page_number: int)` and
  `BenchmarkQuestion(id: str, slice: Literal["answerable_text", "table_lookup",
  "diagram_lookup", "scanned_no_text_layer", "cross_document", "unanswerable"], language:
  Literal["en", "it"], question: str, expected_answer: str | None, sources: list[Source],
  notes: str = "", split: Literal["dev", "hidden"] | None = None)` in
  `doclens.eval.models` (both Pydantic `BaseModel`); `load_benchmark_draft(path: Path) ->
  list[BenchmarkQuestion]` in `doclens.eval.loader`, which reads a JSON array in the shape
  described above and validates every entry. `page_number` must be `>= 1` (matches the
  `Page.page_number` constraint already enforced in `doclens.ingest.models`). A `slice ==
  "unanswerable"` question must have `expected_answer is None` and `sources == []`; the model
  enforces this invariant (an unanswerable question with a non-null answer or a non-empty
  sources list is a data-authoring bug, not something to silently accept).

- [ ] **Step 1: Write the failing tests**

```python
# tests/eval/test_models.py
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
```

```python
# tests/eval/test_loader.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/eval/test_models.py tests/eval/test_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'doclens.eval'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/doclens/eval/models.py
from typing import Literal

from pydantic import BaseModel, model_validator

Slice = Literal[
    "answerable_text",
    "table_lookup",
    "diagram_lookup",
    "scanned_no_text_layer",
    "cross_document",
    "unanswerable",
]


class Source(BaseModel):
    doc_id: str
    doc_title: str
    page_number: int

    @model_validator(mode="after")
    def page_number_must_be_positive(self) -> "Source":
        if self.page_number < 1:
            raise ValueError("page_number must be >= 1")
        return self


class BenchmarkQuestion(BaseModel):
    id: str
    slice: Slice
    language: Literal["en", "it"]
    question: str
    expected_answer: str | None
    sources: list[Source]
    notes: str = ""
    split: Literal["dev", "hidden"] | None = None

    @model_validator(mode="after")
    def unanswerable_has_no_answer_or_sources(self) -> "BenchmarkQuestion":
        if self.slice == "unanswerable":
            if self.expected_answer is not None:
                raise ValueError("an unanswerable question must have expected_answer=None")
            if self.sources:
                raise ValueError("an unanswerable question must have an empty sources list")
        return self
```

```python
# src/doclens/eval/loader.py
import json
from pathlib import Path

from doclens.eval.models import BenchmarkQuestion


def load_benchmark_draft(path: Path) -> list[BenchmarkQuestion]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [BenchmarkQuestion.model_validate(entry) for entry in raw]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/eval/test_models.py tests/eval/test_loader.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add src/doclens/eval/__init__.py src/doclens/eval/models.py src/doclens/eval/loader.py \
  tests/eval/test_models.py tests/eval/test_loader.py
git commit -m "feat(eval): add typed benchmark question model and loader"
```

---

### Task 2: Grounding validator

**Files:**
- Create: `src/doclens/eval/grounding.py`
- Test: `tests/eval/test_grounding.py`

**Interfaces:**
- Consumes: `doclens.eval.models.BenchmarkQuestion`; `doclens.ingest.corpus.load_documents`
  (already exists, returns `list[Document]`, each with `doc_id` and `pages: list[Page]`)
- Produces: `GroundingIssue(question_id: str, reason: str)` (Pydantic `BaseModel`);
  `validate_grounding(questions: list[BenchmarkQuestion], corpus_dir: Path) ->
  list[GroundingIssue]` in `doclens.eval.grounding`. For every question with at least one
  source, checks: (1) the source's `doc_id` matches a document actually present in
  `corpus_dir` (loaded via `load_documents`), and (2) the source's `page_number` is within
  that document's actual page range (`1 <= page_number <= len(document.pages)`). An
  `unanswerable` question (no sources) always passes trivially. Returns an empty list when
  everything is grounded; each failure becomes one `GroundingIssue` with a human-readable
  `reason` (e.g. `"doc_id not found in corpus"` or `"page_number 999 exceeds document's 40
  pages"`).

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_grounding.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_grounding.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'doclens.eval.grounding'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/doclens/eval/grounding.py
from pathlib import Path

from pydantic import BaseModel

from doclens.eval.models import BenchmarkQuestion
from doclens.ingest.corpus import load_documents


class GroundingIssue(BaseModel):
    question_id: str
    reason: str


def validate_grounding(questions: list[BenchmarkQuestion], corpus_dir: Path) -> list[GroundingIssue]:
    documents_by_id = {document.doc_id: document for document in load_documents(corpus_dir)}

    issues: list[GroundingIssue] = []
    for question in questions:
        for source in question.sources:
            document = documents_by_id.get(source.doc_id)
            if document is None:
                issues.append(
                    GroundingIssue(
                        question_id=question.id,
                        reason=f"doc_id not found in corpus: {source.doc_id[:16]}...",
                    )
                )
                continue
            if not (1 <= source.page_number <= len(document.pages)):
                issues.append(
                    GroundingIssue(
                        question_id=question.id,
                        reason=(
                            f"page_number {source.page_number} exceeds document's "
                            f"{len(document.pages)} pages ({document.title})"
                        ),
                    )
                )
    return issues
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_grounding.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/doclens/eval/grounding.py tests/eval/test_grounding.py
git commit -m "feat(eval): add automated grounding validator for benchmark citations"
```

---

### Task 3: Dev/hidden split + contamination check

**Files:**
- Create: `src/doclens/eval/split.py`
- Test: `tests/eval/test_split.py`

**Interfaces:**
- Consumes: `doclens.eval.models.BenchmarkQuestion`
- Produces: `assign_splits(questions: list[BenchmarkQuestion], hidden_fraction: float = 0.2,
  seed: int = 42) -> list[BenchmarkQuestion]` in `doclens.eval.split`, which returns new
  `BenchmarkQuestion` objects with `split` set to `"dev"` or `"hidden"` (does not mutate the
  input list's objects in place; Pydantic models are treated as immutable data here). The
  split is stratified per slice (each slice gets its own ~80/20 split, so a slice with few
  questions, like `scanned_no_text_layer`'s 3, still gets fair representation in both splits
  rather than all 3 landing in the same split by chance) and deterministic given the same
  `seed` (the same input always produces the same split, which matters for reproducibility).
  Also produces: `ContaminationIssue(dev_id: str, hidden_id: str, overlap_ratio: float)`
  (Pydantic `BaseModel`) and `check_contamination(questions: list[BenchmarkQuestion],
  overlap_threshold: float = 0.8) -> list[ContaminationIssue]`, which compares every dev
  question's text against every hidden question's text using word n-gram (n=4) Jaccard
  overlap, flagging any pair above `overlap_threshold` as a potential leak (a hidden question
  that's just a reworded dev question would leak information about what's being tested).

- [ ] **Step 1: Write the failing tests**

```python
# tests/eval/test_split.py
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

    issues = check_contamination([dev_question, hidden_question], overlap_threshold=0.5)

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_split.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'doclens.eval.split'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/doclens/eval/split.py
import random
import re
from collections import defaultdict

from pydantic import BaseModel

from doclens.eval.models import BenchmarkQuestion


def assign_splits(
    questions: list[BenchmarkQuestion], hidden_fraction: float = 0.2, seed: int = 42
) -> list[BenchmarkQuestion]:
    by_slice: dict[str, list[BenchmarkQuestion]] = defaultdict(list)
    for question in questions:
        by_slice[question.slice].append(question)

    result: list[BenchmarkQuestion] = []
    for slice_name in sorted(by_slice):
        slice_questions = sorted(by_slice[slice_name], key=lambda q: q.id)
        rng = random.Random(f"{seed}:{slice_name}")
        shuffled = slice_questions[:]
        rng.shuffle(shuffled)

        hidden_count = max(1, round(len(shuffled) * hidden_fraction)) if shuffled else 0
        hidden_ids = {q.id for q in shuffled[:hidden_count]}

        for question in slice_questions:
            split = "hidden" if question.id in hidden_ids else "dev"
            result.append(question.model_copy(update={"split": split}))

    order = {question.id: index for index, question in enumerate(questions)}
    result.sort(key=lambda q: order[q.id])
    return result


class ContaminationIssue(BaseModel):
    dev_id: str
    hidden_id: str
    overlap_ratio: float


def _ngrams(text: str, n: int = 4) -> set[str]:
    words = re.findall(r"\w+", text.lower())
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def check_contamination(
    questions: list[BenchmarkQuestion], overlap_threshold: float = 0.8
) -> list[ContaminationIssue]:
    dev_questions = [q for q in questions if q.split == "dev"]
    hidden_questions = [q for q in questions if q.split == "hidden"]

    issues: list[ContaminationIssue] = []
    for hidden in hidden_questions:
        hidden_grams = _ngrams(hidden.question)
        if not hidden_grams:
            continue
        for dev in dev_questions:
            dev_grams = _ngrams(dev.question)
            if not dev_grams:
                continue
            overlap = len(hidden_grams & dev_grams) / len(hidden_grams | dev_grams)
            if overlap >= overlap_threshold:
                issues.append(
                    ContaminationIssue(dev_id=dev.id, hidden_id=hidden.id, overlap_ratio=overlap)
                )
    return issues
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_split.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/doclens/eval/split.py tests/eval/test_split.py
git commit -m "feat(eval): add stratified dev/hidden split and n-gram contamination check"
```

---

### Task 4: Dataset card + CLI + final freeze

**Files:**
- Create: `src/doclens/eval/dataset_card.py`
- Modify: `src/doclens/cli.py` (add `bench-validate` and `bench-freeze` commands)
- Test: `tests/eval/test_dataset_card.py`

**Interfaces:**
- Consumes: `doclens.eval.models.BenchmarkQuestion`; `doclens.eval.grounding.GroundingIssue`;
  `doclens.eval.split.ContaminationIssue`
- Produces: `generate_benchmark_card(questions: list[BenchmarkQuestion]) -> str` in
  `doclens.eval.dataset_card`, returning a markdown string with: total question count,
  per-slice breakdown, per-language breakdown, dev/hidden counts per slice, and a fixed,
  explicit paragraph documenting the `scanned_no_text_layer` slice's limitation (only one
  real source page exists in the current corpus; the other ~28 pages flagged as OCR'd are
  confirmed-blank divider pages, independently verified twice during content curation). CLI
  commands: `doclens bench-validate --draft <path> --corpus-dir <dir>` (loads the draft, runs
  `validate_grounding`, prints any `GroundingIssue`s, exits non-zero if any exist so it can gate
  a freeze); `doclens bench-freeze --draft <path> --corpus-dir <dir> --out-dir
  benchmarks/doclens-bench` (runs grounding validation first and refuses to freeze if any
  issues exist, then runs `assign_splits`, then `check_contamination` and refuses to freeze if
  any contamination issues exist, then writes the split, validated questions to
  `<out-dir>/questions.json` and the dataset card to `<out-dir>/DATASET_CARD.md`).

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_dataset_card.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_dataset_card.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'doclens.eval.dataset_card'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/doclens/eval/dataset_card.py
from collections import Counter

from doclens.eval.models import BenchmarkQuestion

SCANNED_PAGE_LIMITATION_NOTE = """## Known limitation: scanned_no_text_layer coverage

Of the ~29 pages flagged `text_source: "ocr"` across the 28-document seed corpus, only one
(Interroll 24V Roller Conveyor manual, page 77) has genuine, non-blank scanned content. The
other pages are confirmed-blank divider/filler pages (empty extracted text, blank rendered
images, tiny file sizes), independently verified twice during content curation. This slice is
therefore populated with 3 questions, all grounded in that single real page, rather than the
originally planned 8. Growing this slice meaningfully requires sourcing documents with
genuinely scanned (non-blank) content, not just documents that happen to have an OCR-tagged
page."""


def generate_benchmark_card(questions: list[BenchmarkQuestion]) -> str:
    slice_counts = Counter(q.slice for q in questions)
    language_counts = Counter(q.language for q in questions)
    split_counts = Counter(q.split for q in questions)
    slice_split_counts = Counter((q.slice, q.split) for q in questions)

    lines = [
        "# doclens-bench Dataset Card",
        "",
        f"Total questions: {len(questions)}",
        "",
        "## Slice breakdown",
        "",
    ]
    for slice_name in sorted(slice_counts):
        dev = slice_split_counts.get((slice_name, "dev"), 0)
        hidden = slice_split_counts.get((slice_name, "hidden"), 0)
        lines.append(f"- {slice_name}: {slice_counts[slice_name]} (dev={dev}, hidden={hidden})")

    lines += ["", "## Language breakdown", ""]
    for language in sorted(language_counts):
        lines.append(f"- {language}: {language_counts[language]}")

    lines += ["", "## Split totals", ""]
    for split_name in ("dev", "hidden"):
        lines.append(f"- {split_name}: {split_counts.get(split_name, 0)}")

    lines += ["", SCANNED_PAGE_LIMITATION_NOTE]
    return "\n".join(lines)
```

Append to `src/doclens/cli.py`. Add `import json` near the top of the file alongside the
existing imports if it is not already imported (check first):

```python
from doclens.eval.dataset_card import generate_benchmark_card
from doclens.eval.grounding import validate_grounding
from doclens.eval.loader import load_benchmark_draft
from doclens.eval.split import assign_splits, check_contamination


@app.command(name="bench-validate")
def bench_validate(
    draft: Path = typer.Option(..., help="Path to the raw benchmark draft JSON"),
    corpus_dir: Path = typer.Option(Path("corpus"), help="Ingested corpus directory"),
) -> None:
    questions = load_benchmark_draft(draft)
    issues = validate_grounding(questions, corpus_dir)
    if not issues:
        typer.echo(f"All {len(questions)} questions are grounded in the corpus.")
        return
    for issue in issues:
        typer.echo(f"  {issue.question_id}: {issue.reason}")
    typer.echo(f"{len(issues)} grounding issue(s) found.")
    raise typer.Exit(code=1)


@app.command(name="bench-freeze")
def bench_freeze(
    draft: Path = typer.Option(..., help="Path to the raw benchmark draft JSON"),
    corpus_dir: Path = typer.Option(Path("corpus"), help="Ingested corpus directory"),
    out_dir: Path = typer.Option(Path("benchmarks/doclens-bench"), help="Output directory"),
) -> None:
    questions = load_benchmark_draft(draft)

    grounding_issues = validate_grounding(questions, corpus_dir)
    if grounding_issues:
        for issue in grounding_issues:
            typer.echo(f"  {issue.question_id}: {issue.reason}")
        typer.echo(f"Refusing to freeze: {len(grounding_issues)} grounding issue(s) found.")
        raise typer.Exit(code=1)

    split_questions = assign_splits(questions)
    contamination_issues = check_contamination(split_questions)
    if contamination_issues:
        for issue in contamination_issues:
            typer.echo(f"  {issue.dev_id} <-> {issue.hidden_id}: overlap {issue.overlap_ratio:.2f}")
        typer.echo(f"Refusing to freeze: {len(contamination_issues)} contamination issue(s) found.")
        raise typer.Exit(code=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    questions_path = out_dir / "questions.json"
    payload = [q.model_dump(mode="json") for q in split_questions]
    questions_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    card_path = out_dir / "DATASET_CARD.md"
    card_path.write_text(generate_benchmark_card(split_questions), encoding="utf-8")

    typer.echo(f"Froze {len(split_questions)} questions to {questions_path}")
    typer.echo(f"Wrote dataset card to {card_path}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/eval/test_dataset_card.py -v`
Expected: PASS (3 passed)

Then run the full test suite: `pytest tests/ -v -m "not slow"` Expected: all pass, no
regressions.

- [ ] **Step 5: Freeze the real benchmark**

This step runs the actual freeze against the real draft and real corpus, not a test fixture,
mirroring how Milestone 1/2 validated against real data.

```bash
python -m doclens.cli bench-validate --draft benchmarks/doclens-bench/_drafts/consolidated_draft.json --corpus-dir corpus
python -m doclens.cli bench-freeze --draft benchmarks/doclens-bench/_drafts/consolidated_draft.json --corpus-dir corpus --out-dir benchmarks/doclens-bench
```

Confirm: `bench-validate` reports all 66 questions grounded (or, if it finds real issues,
report them honestly rather than silently patching the draft to make them disappear; a
citation genuinely pointing at a nonexistent page is a real authoring bug worth surfacing,
not hiding). Confirm `bench-freeze` succeeds and produces
`benchmarks/doclens-bench/questions.json` and `benchmarks/doclens-bench/DATASET_CARD.md`.
Record the exact output in the task report.

- [ ] **Step 6: Commit**

```bash
git add src/doclens/eval/dataset_card.py src/doclens/cli.py tests/eval/test_dataset_card.py \
  benchmarks/doclens-bench/questions.json benchmarks/doclens-bench/DATASET_CARD.md
git commit -m "feat(eval): add dataset card, CLI, and freeze the doclens-bench benchmark"
```

---

## Exit criteria

- [ ] `doclens bench-validate` runs against the real draft and real corpus with zero grounding
  issues (or any real issues found are fixed in the draft data, not silently ignored).
- [ ] `doclens bench-freeze` produces a versioned `benchmarks/doclens-bench/questions.json`
  with a stratified 80/20 dev/hidden split and zero contamination issues.
- [ ] `benchmarks/doclens-bench/DATASET_CARD.md` accurately documents slice/language/split
  breakdowns and the scanned-page limitation.
- [ ] All tests in `tests/eval/` pass under `pytest -m "not slow"`, with no regressions to
  Milestones 1-2.

This closes the benchmark infrastructure sub-project. The caption-and-index and vision-embedding
retrieval arms, each with their own plan, are what get measured against this benchmark next.
