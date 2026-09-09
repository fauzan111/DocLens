from pathlib import Path

from doclens.ingest.models import Document, License, Page
from doclens.ingest.ocr_fallback import OcrFallback
from doclens.ingest.pdf_extractor import PdfExtractor
from doclens.ingest.registry import SourceRegistry


def run_ingestion(
    pdf_bytes: bytes,
    source_url: str,
    license: License,
    title: str,
    corpus_dir: Path,
    registry: SourceRegistry,
    pdf_extractor: PdfExtractor | None = None,
    ocr_fallback: OcrFallback | None = None,
) -> Document:
    pdf_extractor = pdf_extractor or PdfExtractor()
    corpus_dir.mkdir(parents=True, exist_ok=True)

    source, is_new = registry.register(
        source_url=source_url, license=license, file_bytes=pdf_bytes
    )
    doc_id = source.sha256
    json_path = corpus_dir / f"{doc_id}.json"

    if not is_new and json_path.exists():
        return Document.model_validate_json(json_path.read_text(encoding="utf-8"))

    image_dir = corpus_dir / doc_id
    raw_pages = pdf_extractor.extract(pdf_bytes, doc_id=doc_id, image_output_dir=image_dir)

    pages: list[Page] = []
    for raw_page in raw_pages:
        if raw_page.has_text_layer:
            pages.append(
                Page(
                    page_number=raw_page.page_number,
                    text=raw_page.text,
                    text_source="extracted",
                    image_path=raw_page.image_path,
                )
            )
        else:
            resolved_ocr = ocr_fallback or OcrFallback()
            ocr_text = resolved_ocr.read_text(raw_page.image_path)
            pages.append(
                Page(
                    page_number=raw_page.page_number,
                    text=ocr_text,
                    text_source="ocr",
                    image_path=raw_page.image_path,
                )
            )

    document = Document(doc_id=doc_id, title=title, source=source, pages=pages)
    json_path.write_text(document.model_dump_json(indent=2), encoding="utf-8")
    return document
