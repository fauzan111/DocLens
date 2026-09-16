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
