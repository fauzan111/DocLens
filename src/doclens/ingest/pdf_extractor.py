from pathlib import Path
from typing import NamedTuple

import fitz  # PyMuPDF


class RawPage(NamedTuple):
    page_number: int
    text: str
    has_text_layer: bool
    image_path: Path


class PdfExtractor:
    RENDER_DPI = 150

    def extract(
        self, pdf_bytes: bytes, doc_id: str, image_output_dir: Path
    ) -> list[RawPage]:
        image_output_dir.mkdir(parents=True, exist_ok=True)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages: list[RawPage] = []
        try:
            zoom = self.RENDER_DPI / 72
            matrix = fitz.Matrix(zoom, zoom)
            for index, page in enumerate(doc, start=1):
                text = page.get_text("text")
                pixmap = page.get_pixmap(matrix=matrix)
                image_path = image_output_dir / f"page-{index}.png"
                pixmap.save(str(image_path))
                pages.append(
                    RawPage(
                        page_number=index,
                        text=text,
                        has_text_layer=bool(text.strip()),
                        image_path=image_path,
                    )
                )
        finally:
            doc.close()
        return pages
