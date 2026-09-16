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
