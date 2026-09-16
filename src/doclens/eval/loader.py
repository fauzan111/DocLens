import json
from pathlib import Path

from doclens.eval.models import BenchmarkQuestion


def load_benchmark_draft(path: Path) -> list[BenchmarkQuestion]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [BenchmarkQuestion.model_validate(entry) for entry in raw]
