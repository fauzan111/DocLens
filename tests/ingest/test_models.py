from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from doclens.ingest.models import Document, License, Page, SourceRecord


def test_source_record_requires_license_and_url():
    record = SourceRecord(
        source_url="https://example.com/manual.pdf",
        license=License.MANUFACTURER_PUBLIC,
        retrieved_at=datetime(2026, 9, 9, tzinfo=timezone.utc),
        sha256="a" * 64,
    )
    assert record.license == License.MANUFACTURER_PUBLIC
    assert len(record.sha256) == 64


def test_source_record_rejects_invalid_sha256_length():
    with pytest.raises(ValidationError):
        SourceRecord(
            source_url="https://example.com/manual.pdf",
            license=License.PUBLIC_DOMAIN,
            retrieved_at=datetime(2026, 9, 9, tzinfo=timezone.utc),
            sha256="too-short",
        )


def test_document_doc_id_matches_source_sha256():
    record = SourceRecord(
        source_url="https://example.com/manual.pdf",
        license=License.PUBLIC_DOMAIN,
        retrieved_at=datetime(2026, 9, 9, tzinfo=timezone.utc),
        sha256="b" * 64,
    )
    page = Page(
        page_number=1,
        text="Torque spec: 45 Nm",
        text_source="extracted",
        image_path=Path("corpus/b" * 1 + ".../page-1.png"),
        language_hint="en",
    )
    doc = Document(doc_id="b" * 64, title="Pump Manual", source=record, pages=[page])
    assert doc.doc_id == doc.source.sha256


def test_document_rejects_doc_id_mismatch_with_source_sha256():
    record = SourceRecord(
        source_url="https://example.com/manual.pdf",
        license=License.PUBLIC_DOMAIN,
        retrieved_at=datetime(2026, 9, 9, tzinfo=timezone.utc),
        sha256="c" * 64,
    )
    with pytest.raises(ValidationError):
        Document(doc_id="d" * 64, title="Pump Manual", source=record, pages=[])
