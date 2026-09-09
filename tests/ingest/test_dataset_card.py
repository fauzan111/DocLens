from pathlib import Path

from doclens.ingest.dataset_card import generate_dataset_card
from doclens.ingest.models import License
from doclens.ingest.pipeline import run_ingestion
from doclens.ingest.registry import SourceRegistry

import fitz  # PyMuPDF


def _make_pdf_with_text(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_dataset_card_reports_document_count_and_license_breakdown(tmp_path: Path):
    corpus_dir = tmp_path / "corpus"
    registry = SourceRegistry(registry_path=corpus_dir / "registry.json")
    run_ingestion(
        pdf_bytes=_make_pdf_with_text("Doc A"),
        source_url="https://example.com/a.pdf",
        license=License.MANUFACTURER_PUBLIC,
        title="Doc A",
        corpus_dir=corpus_dir,
        registry=registry,
    )
    run_ingestion(
        pdf_bytes=_make_pdf_with_text("Doc B"),
        source_url="https://example.com/b.pdf",
        license=License.PUBLIC_DOMAIN,
        title="Doc B",
        corpus_dir=corpus_dir,
        registry=registry,
    )

    card = generate_dataset_card(corpus_dir=corpus_dir, registry=registry)

    assert "Total documents: 2" in card
    assert "manufacturer_public: 1" in card
    assert "public_domain: 1" in card


def test_dataset_card_reports_ocr_page_fraction(tmp_path: Path):
    corpus_dir = tmp_path / "corpus"
    registry = SourceRegistry(registry_path=corpus_dir / "registry.json")
    run_ingestion(
        pdf_bytes=_make_pdf_with_text("Text page"),
        source_url="https://example.com/c.pdf",
        license=License.PUBLIC_DOMAIN,
        title="Doc C",
        corpus_dir=corpus_dir,
        registry=registry,
    )

    card = generate_dataset_card(corpus_dir=corpus_dir, registry=registry)

    assert "OCR-derived pages: 0 / 1" in card
