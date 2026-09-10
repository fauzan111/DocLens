from collections import Counter
from pathlib import Path

from doclens.ingest.corpus import load_documents
from doclens.ingest.registry import SourceRegistry


def generate_dataset_card(corpus_dir: Path, registry: SourceRegistry) -> str:
    records = registry.all_records()
    license_counts = Counter(record.license.value for record in records)

    total_pages = 0
    ocr_pages = 0
    for document in load_documents(corpus_dir):
        total_pages += len(document.pages)
        ocr_pages += sum(1 for page in document.pages if page.text_source == "ocr")

    lines = [
        "# DocLens Dataset Card",
        "",
        f"Total documents: {len(records)}",
        "",
        "## License breakdown",
        "",
    ]
    for license_name, count in sorted(license_counts.items()):
        lines.append(f"- {license_name}: {count}")
    lines += [
        "",
        "## Page statistics",
        "",
        f"OCR-derived pages: {ocr_pages} / {total_pages}",
    ]
    return "\n".join(lines)
