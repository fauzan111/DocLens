from pydantic import BaseModel


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    page_number: int
    chunk_index: int
    text: str
    source_url: str
