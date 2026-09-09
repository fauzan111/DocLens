from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator, model_validator


class License(StrEnum):
    PUBLIC_DOMAIN = "public_domain"
    MANUFACTURER_PUBLIC = "manufacturer_public"
    CC_BY = "cc_by"
    CC_BY_SA = "cc_by_sa"
    PROPRIETARY_PREVIEW = "proprietary_preview"
    UNKNOWN = "unknown"


class SourceRecord(BaseModel):
    source_url: str
    license: License
    retrieved_at: datetime
    sha256: str

    @field_validator("sha256")
    @classmethod
    def sha256_must_be_64_hex_chars(cls, value: str) -> str:
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()):
            raise ValueError("sha256 must be a 64-character hex string")
        return value.lower()


class Page(BaseModel):
    page_number: int
    text: str
    text_source: Literal["extracted", "ocr"]
    image_path: Path
    language_hint: str | None = None

    @field_validator("page_number")
    @classmethod
    def page_number_must_be_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("page_number must be >= 1")
        return value


class Document(BaseModel):
    doc_id: str
    title: str
    source: SourceRecord
    pages: list[Page]

    @model_validator(mode="after")
    def doc_id_must_match_source_hash(self) -> "Document":
        if self.doc_id != self.source.sha256:
            raise ValueError("doc_id must equal source.sha256 (content-addressed)")
        return self
