import json
from pathlib import Path

import fitz  # PyMuPDF

from doclens.ingest.models import License
from doclens.ingest.ocr_fallback import OcrFallback
from doclens.ingest.pdf_extractor import PdfExtractor
from doclens.ingest.pipeline import run_ingestion
from doclens.ingest.registry import SourceRegistry


def _make_pdf_with_text(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_run_ingestion_produces_document_with_extracted_text(tmp_path: Path):
    corpus_dir = tmp_path / "corpus"
    registry = SourceRegistry(registry_path=tmp_path / "registry.json")
    pdf_bytes = _make_pdf_with_text("Torque: 45 Nm")

    document = run_ingestion(
        pdf_bytes=pdf_bytes,
        source_url="https://example.com/manual.pdf",
        license=License.MANUFACTURER_PUBLIC,
        title="Pump Manual",
        corpus_dir=corpus_dir,
        registry=registry,
    )

    assert document.title == "Pump Manual"
    assert len(document.pages) == 1
    assert "Torque: 45 Nm" in document.pages[0].text
    assert document.pages[0].text_source == "extracted"


def test_run_ingestion_writes_document_json_to_corpus_dir(tmp_path: Path):
    corpus_dir = tmp_path / "corpus"
    registry = SourceRegistry(registry_path=tmp_path / "registry.json")
    pdf_bytes = _make_pdf_with_text("Voltage: 230V")

    document = run_ingestion(
        pdf_bytes=pdf_bytes,
        source_url="https://example.com/manual2.pdf",
        license=License.PUBLIC_DOMAIN,
        title="Wiring Manual",
        corpus_dir=corpus_dir,
        registry=registry,
    )

    json_path = corpus_dir / f"{document.doc_id}.json"
    assert json_path.exists()
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["title"] == "Wiring Manual"


def test_run_ingestion_uses_ocr_fallback_when_no_text_layer(tmp_path: Path, monkeypatch):
    corpus_dir = tmp_path / "corpus"
    registry = SourceRegistry(registry_path=tmp_path / "registry.json")
    doc = fitz.open()
    doc.new_page()  # blank -> no text layer
    pdf_bytes = doc.tobytes()
    doc.close()

    class StubOcr:
        def read_text(self, image_path: Path) -> str:
            return "OCR RECOVERED TEXT"

    document = run_ingestion(
        pdf_bytes=pdf_bytes,
        source_url="https://example.com/scanned.pdf",
        license=License.UNKNOWN,
        title="Scanned Sheet",
        corpus_dir=corpus_dir,
        registry=registry,
        ocr_fallback=StubOcr(),
    )

    assert document.pages[0].text == "OCR RECOVERED TEXT"
    assert document.pages[0].text_source == "ocr"


def test_run_ingestion_skips_re_extraction_for_duplicate_content(tmp_path: Path):
    corpus_dir = tmp_path / "corpus"
    registry = SourceRegistry(registry_path=tmp_path / "registry.json")
    pdf_bytes = _make_pdf_with_text("Duplicate content")

    first = run_ingestion(
        pdf_bytes=pdf_bytes,
        source_url="https://example.com/a.pdf",
        license=License.PUBLIC_DOMAIN,
        title="Original",
        corpus_dir=corpus_dir,
        registry=registry,
    )
    second = run_ingestion(
        pdf_bytes=pdf_bytes,
        source_url="https://example.com/a-mirror.pdf",
        license=License.PUBLIC_DOMAIN,
        title="Mirror",
        corpus_dir=corpus_dir,
        registry=registry,
    )

    assert first.doc_id == second.doc_id
    assert second.title == "Original"  # first-write wins; not overwritten by the mirror
