import os
from pathlib import Path

import pytest

from doclens.caption.captioner import (
    NO_VISUAL_CONTENT_SENTINEL,
    PageCaptioner,
    _parse_caption_response,
)


def test_parse_caption_response_returns_none_for_sentinel():
    assert _parse_caption_response(NO_VISUAL_CONTENT_SENTINEL) is None


def test_parse_caption_response_strips_whitespace_around_sentinel():
    assert _parse_caption_response(f"  {NO_VISUAL_CONTENT_SENTINEL}  \n") is None


def test_parse_caption_response_returns_stripped_text_for_real_caption():
    result = _parse_caption_response("  A table showing torque values.  ")
    assert result == "A table showing torque values."


def test_captioner_does_not_load_model_on_init():
    captioner = PageCaptioner(api_key="unused-for-this-test")
    assert captioner._model is None


def test_captioner_without_api_key_raises_on_caption_page(tmp_path: Path):
    image_path = tmp_path / "page-1.png"
    image_path.write_bytes(b"not a real png, never reached")
    captioner = PageCaptioner(api_key=None)
    with pytest.raises(RuntimeError):
        captioner.caption_page(image_path)


@pytest.mark.slow
def test_captioner_produces_a_real_caption_for_a_real_table_image():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY not set; skipping live Gemini call")

    import fitz  # PyMuPDF

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Table 1: Torque Values")
    page.insert_text((72, 100), "Bolt Size    Torque (Nm)")
    page.insert_text((72, 120), "M8           25")
    page.insert_text((72, 140), "M10          45")
    image_path = Path("test_caption_table.png")
    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    pixmap.save(str(image_path))
    doc.close()

    try:
        captioner = PageCaptioner(api_key=api_key)
        result = captioner.caption_page(image_path)
        assert result is not None
        assert "torque" in result.lower() or "45" in result or "25" in result
    finally:
        image_path.unlink(missing_ok=True)
