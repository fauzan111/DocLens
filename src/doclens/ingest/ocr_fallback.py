import sys
from pathlib import Path


class OcrFallback:
    def __init__(self, languages: list[str] | None = None):
        self.languages = languages or ["en", "it"]
        self._reader = None

    def read_text(self, image_path: Path) -> str:
        if self._reader is None:
            # EasyOCR's model-download progress bar prints Unicode block
            # characters (e.g. U+2588). On Windows, stdout/stderr default to
            # a non-UTF-8 console codepage and crash on that output.
            for stream in (sys.stdout, sys.stderr):
                if getattr(stream, "encoding", "").lower() != "utf-8":
                    stream.reconfigure(encoding="utf-8", errors="replace")

            import easyocr  # heavy import; deferred until first real use

            self._reader = easyocr.Reader(self.languages, gpu=False)
        results = self._reader.readtext(str(image_path), detail=0)
        return " ".join(results)
