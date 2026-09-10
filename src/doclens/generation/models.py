from pydantic import BaseModel


class Citation(BaseModel):
    doc_title: str
    page_number: int
    source_url: str


class Answer(BaseModel):
    text: str
    citations: list[Citation]
    grounded: bool
