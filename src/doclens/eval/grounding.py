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
