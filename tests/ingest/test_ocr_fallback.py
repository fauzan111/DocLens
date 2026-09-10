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


class _FakeStream:
    """Mimics enough of a text stream to exercise the reconfigure() guard."""

    def __init__(self, encoding: str):
        self.encoding = encoding
        self.reconfigure_calls: list[dict] = []

    def reconfigure(self, **kwargs):
        self.reconfigure_calls.append(kwargs)


class _FakeEasyocrReader:
    def __init__(self, languages, gpu):
        pass

    def readtext(self, image_path, detail=0):
        return ["stub"]


def _install_fake_easyocr(monkeypatch):
    import sys
    import types

    fake_module = types.ModuleType("easyocr")
    fake_module.Reader = _FakeEasyocrReader
    monkeypatch.setitem(sys.modules, "easyocr", fake_module)


def test_read_text_reconfigures_non_utf8_stdout_and_stderr(monkeypatch, tmp_path):
    _install_fake_easyocr(monkeypatch)
    fake_stdout = _FakeStream("cp1252")
    fake_stderr = _FakeStream("cp1252")
    monkeypatch.setattr("sys.stdout", fake_stdout)
    monkeypatch.setattr("sys.stderr", fake_stderr)

    ocr = OcrFallback(languages=["en"])
    ocr.read_text(tmp_path / "page.png")

    assert fake_stdout.reconfigure_calls == [{"encoding": "utf-8", "errors": "replace"}]
    assert fake_stderr.reconfigure_calls == [{"encoding": "utf-8", "errors": "replace"}]


def test_read_text_does_not_reconfigure_already_utf8_streams(monkeypatch, tmp_path):
    _install_fake_easyocr(monkeypatch)
    fake_stdout = _FakeStream("utf-8")
    fake_stderr = _FakeStream("utf-8")
    monkeypatch.setattr("sys.stdout", fake_stdout)
    monkeypatch.setattr("sys.stderr", fake_stderr)

    ocr = OcrFallback(languages=["en"])
    ocr.read_text(tmp_path / "page.png")

    assert fake_stdout.reconfigure_calls == []
    assert fake_stderr.reconfigure_calls == []
