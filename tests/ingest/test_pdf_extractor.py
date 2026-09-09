from pathlib import Path

import fitz  # PyMuPDF

from doclens.ingest.pdf_extractor import PdfExtractor


def _make_pdf_with_text(pages_text: list[str]) -> bytes:
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def _make_pdf_with_blank_page() -> bytes:
    doc = fitz.open()
    doc.new_page()  # no text inserted -> no text layer
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_extract_returns_one_raw_page_per_pdf_page(tmp_path: Path):
    extractor = PdfExtractor()
    pdf_bytes = _make_pdf_with_text(["Torque: 45 Nm", "Voltage: 230V"])
    pages = extractor.extract(pdf_bytes, doc_id="doc1", image_output_dir=tmp_path)
    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert pages[1].page_number == 2


def test_extract_captures_text_and_marks_text_layer_present(tmp_path: Path):
    extractor = PdfExtractor()
    pdf_bytes = _make_pdf_with_text(["Torque: 45 Nm"])
    pages = extractor.extract(pdf_bytes, doc_id="doc1", image_output_dir=tmp_path)
    assert "Torque: 45 Nm" in pages[0].text
    assert pages[0].has_text_layer is True


def test_extract_marks_blank_page_as_no_text_layer(tmp_path: Path):
    extractor = PdfExtractor()
    pdf_bytes = _make_pdf_with_blank_page()
    pages = extractor.extract(pdf_bytes, doc_id="doc1", image_output_dir=tmp_path)
    assert pages[0].text.strip() == ""
    assert pages[0].has_text_layer is False


def test_extract_writes_page_image_files(tmp_path: Path):
    extractor = PdfExtractor()
    pdf_bytes = _make_pdf_with_text(["Torque: 45 Nm"])
    pages = extractor.extract(pdf_bytes, doc_id="doc1", image_output_dir=tmp_path)
    assert pages[0].image_path.exists()
    assert pages[0].image_path.name == "page-1.png"
    assert pages[0].image_path.suffix == ".png"
