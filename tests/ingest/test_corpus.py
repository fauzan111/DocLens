import json
from pathlib import Path

from doclens.ingest.corpus import load_documents
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


def test_load_documents_skips_registry_json(tmp_path: Path):
    corpus_dir = tmp_path / "corpus"
    registry = SourceRegistry(registry_path=corpus_dir / "registry.json")
    run_ingestion(
        pdf_bytes=_make_pdf_with_text("Torque: 45 Nm"),
        source_url="https://example.com/a.pdf",
        license=License.PUBLIC_DOMAIN,
        title="Doc A",
        corpus_dir=corpus_dir,
        registry=registry,
    )

    documents = load_documents(corpus_dir)

    assert len(documents) == 1
    assert documents[0].title == "Doc A"


def test_load_documents_skips_unparseable_json_without_crashing(tmp_path: Path):
    corpus_dir = tmp_path / "corpus"
    registry = SourceRegistry(registry_path=corpus_dir / "registry.json")
    run_ingestion(
        pdf_bytes=_make_pdf_with_text("Voltage: 230V"),
        source_url="https://example.com/b.pdf",
        license=License.PUBLIC_DOMAIN,
        title="Doc B",
        corpus_dir=corpus_dir,
        registry=registry,
    )
    # Simulate a stray, non-Document JSON file landing in corpus_dir.
    (corpus_dir / "notes.json").write_text(json.dumps({"unrelated": "data"}), encoding="utf-8")

    documents = load_documents(corpus_dir)

    assert len(documents) == 1
    assert documents[0].title == "Doc B"
