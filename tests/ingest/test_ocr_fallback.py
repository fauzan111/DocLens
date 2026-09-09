from pathlib import Path

import fitz  # PyMuPDF
import pytest

from doclens.ingest.ocr_fallback import OcrFallback


def _render_text_to_png(text: str, out_path: Path) -> Path:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 200), text, fontsize=36)
    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    pixmap.save(str(out_path))
    doc.close()
    return out_path


@pytest.mark.slow
def test_read_text_extracts_visible_text_from_image(tmp_path: Path):
    image_path = _render_text_to_png("VOLTAGE 230", tmp_path / "page.png")
    ocr = OcrFallback(languages=["en"])
    result = ocr.read_text(image_path)
    assert "VOLTAGE" in result.upper() or "230" in result


def test_ocr_fallback_does_not_load_model_on_init():
    # Constructing OcrFallback must not attempt any network/model download.
    ocr = OcrFallback(languages=["en"])
    assert ocr._reader is None
