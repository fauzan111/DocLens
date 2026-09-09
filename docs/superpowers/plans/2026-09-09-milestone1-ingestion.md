# DocLens Milestone 1: Corpus + Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the ingestion pipeline that turns a folder of sourced PDFs into a structured,
licensed corpus — per-page text (with OCR fallback for scanned/no-text-layer pages), per-page
images, and a generated dataset card — ready for the three retrieval systems built in Milestone 2.

**Architecture:** A typed domain model (`Document`, `Page`, `SourceRecord`) backs a small
pipeline: a `SourceRegistry` records provenance/license/hash for every sourced PDF and rejects
duplicates; a `PdfExtractor` (PyMuPDF) pulls per-page text and renders per-page PNGs; an
`OcrFallback` (EasyOCR) fills in text for pages PyMuPDF returns empty; a `CorpusWriter` persists
everything to disk as one JSON artifact per document; a `DatasetCardGenerator` aggregates the
registry into a markdown report. A CLI ties it together.

**Tech Stack:** Python 3.11+, PyMuPDF (`pymupdf`), EasyOCR (`easyocr`), Pydantic v2, pytest,
Typer (CLI), hashlib (SHA-256 content addressing).

## Global Constraints

- No Anthropic/Claude API calls anywhere in this pipeline — ingestion must run at $0 ongoing
  cost (per DESIGN.md "Stack" section).
- Corpus is bilingual IT/EN; nothing in ingestion may assume a single language.
- Every sourced document must have a recorded source URL, license, and retrieval date before it
  is usable — undocumented documents are rejected, not silently skipped.
- No confidential or employer (GEKO) data may enter the corpus.
- Content-addressed by SHA-256: re-ingesting the same file content under a different name must
  not create a duplicate entry (mirrors TrustGate's immutable-registry pattern).
- All file paths in code are `pathlib.Path`, never raw strings, for cross-platform (Windows dev
  environment) correctness.

---

## File Structure

```
DocLens/
  src/doclens/
    ingest/
      __init__.py
      models.py          # Task 1: Document, Page, SourceRecord, License
      registry.py         # Task 2: SourceRegistry (JSON-backed, hash-deduped)
      pdf_extractor.py    # Task 3: PdfExtractor (text + page images via PyMuPDF)
      ocr_fallback.py     # Task 4: OcrFallback (EasyOCR wrapper)
      pipeline.py          # Task 5: run_ingestion() orchestration + CorpusWriter
      dataset_card.py     # Task 6: DatasetCardGenerator
    cli.py                # Task 5: `doclens ingest` command (extended)
  tests/
    ingest/
      test_models.py
      test_registry.py
      test_pdf_extractor.py
      test_ocr_fallback.py
      test_pipeline.py
      test_dataset_card.py
  corpus/                 # gitignored output: <hash>.json per document, <hash>/page-N.png
```

---

### Task 1: Domain models

**Files:**
- Create: `src/doclens/ingest/models.py`
- Test: `tests/ingest/test_models.py`

**Interfaces:**
- Produces: `License` (str Enum: `PUBLIC_DOMAIN`, `MANUFACTURER_PUBLIC`, `CC_BY`, `CC_BY_SA`,
  `PROPRIETARY_PREVIEW`, `UNKNOWN`), `SourceRecord(source_url: str, license: License,
  retrieved_at: datetime, sha256: str)`, `Page(page_number: int, text: str, text_source:
  Literal["extracted", "ocr"], image_path: Path, language_hint: str | None)`, `Document(doc_id:
  str, title: str, source: SourceRecord, pages: list[Page])`. All Pydantic `BaseModel` subclasses.
  `Document.doc_id` is always the source's `sha256`.

- [ ] **Step 1: Write the failing test**

```python
# tests/ingest/test_models.py
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from doclens.ingest.models import Document, License, Page, SourceRecord


def test_source_record_requires_license_and_url():
    record = SourceRecord(
        source_url="https://example.com/manual.pdf",
        license=License.MANUFACTURER_PUBLIC,
        retrieved_at=datetime(2026, 9, 9, tzinfo=timezone.utc),
        sha256="a" * 64,
    )
    assert record.license == License.MANUFACTURER_PUBLIC
    assert len(record.sha256) == 64


def test_source_record_rejects_invalid_sha256_length():
    with pytest.raises(ValidationError):
        SourceRecord(
            source_url="https://example.com/manual.pdf",
            license=License.PUBLIC_DOMAIN,
            retrieved_at=datetime(2026, 9, 9, tzinfo=timezone.utc),
            sha256="too-short",
        )


def test_document_doc_id_matches_source_sha256():
    record = SourceRecord(
        source_url="https://example.com/manual.pdf",
        license=License.PUBLIC_DOMAIN,
        retrieved_at=datetime(2026, 9, 9, tzinfo=timezone.utc),
        sha256="b" * 64,
    )
    page = Page(
        page_number=1,
        text="Torque spec: 45 Nm",
        text_source="extracted",
        image_path=Path("corpus/b" * 1 + ".../page-1.png"),
        language_hint="en",
    )
    doc = Document(doc_id="b" * 64, title="Pump Manual", source=record, pages=[page])
    assert doc.doc_id == doc.source.sha256


def test_document_rejects_doc_id_mismatch_with_source_sha256():
    record = SourceRecord(
        source_url="https://example.com/manual.pdf",
        license=License.PUBLIC_DOMAIN,
        retrieved_at=datetime(2026, 9, 9, tzinfo=timezone.utc),
        sha256="c" * 64,
    )
    with pytest.raises(ValidationError):
        Document(doc_id="d" * 64, title="Pump Manual", source=record, pages=[])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ingest/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'doclens.ingest.models'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/doclens/ingest/models.py
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator, model_validator


class License(StrEnum):
    PUBLIC_DOMAIN = "public_domain"
    MANUFACTURER_PUBLIC = "manufacturer_public"
    CC_BY = "cc_by"
    CC_BY_SA = "cc_by_sa"
    PROPRIETARY_PREVIEW = "proprietary_preview"
    UNKNOWN = "unknown"


class SourceRecord(BaseModel):
    source_url: str
    license: License
    retrieved_at: datetime
    sha256: str

    @field_validator("sha256")
    @classmethod
    def sha256_must_be_64_hex_chars(cls, value: str) -> str:
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()):
            raise ValueError("sha256 must be a 64-character hex string")
        return value.lower()


class Page(BaseModel):
    page_number: int
    text: str
    text_source: Literal["extracted", "ocr"]
    image_path: Path
    language_hint: str | None = None

    @field_validator("page_number")
    @classmethod
    def page_number_must_be_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("page_number must be >= 1")
        return value


class Document(BaseModel):
    doc_id: str
    title: str
    source: SourceRecord
    pages: list[Page]

    @model_validator(mode="after")
    def doc_id_must_match_source_hash(self) -> "Document":
        if self.doc_id != self.source.sha256:
            raise ValueError("doc_id must equal source.sha256 (content-addressed)")
        return self
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ingest/test_models.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/doclens/ingest/models.py tests/ingest/test_models.py
git commit -m "feat(ingest): add typed domain models for documents, pages, and source records"
```

---

### Task 2: Source registry

**Files:**
- Create: `src/doclens/ingest/registry.py`
- Test: `tests/ingest/test_registry.py`

**Interfaces:**
- Consumes: `SourceRecord`, `License` from `doclens.ingest.models`
- Produces: `SourceRegistry(registry_path: Path)` with methods
  `register(source_url: str, license: License, file_bytes: bytes) -> tuple[SourceRecord, bool]`
  (returns `(record, is_new)`; `is_new=False` if the SHA-256 already exists — same content is
  never duplicated even under a different URL/name) and `all_records() -> list[SourceRecord]`.
  Registry persists to a JSON file at `registry_path` (list of `SourceRecord`, one entry per
  unique `sha256`).

- [ ] **Step 1: Write the failing test**

```python
# tests/ingest/test_registry.py
from datetime import datetime, timezone
from pathlib import Path

from doclens.ingest.models import License
from doclens.ingest.registry import SourceRegistry


def test_register_new_document_returns_is_new_true(tmp_path: Path):
    registry = SourceRegistry(registry_path=tmp_path / "registry.json")
    record, is_new = registry.register(
        source_url="https://example.com/pump-manual.pdf",
        license=License.MANUFACTURER_PUBLIC,
        file_bytes=b"%PDF-1.4 fake pdf bytes",
    )
    assert is_new is True
    assert record.source_url == "https://example.com/pump-manual.pdf"
    assert len(record.sha256) == 64


def test_register_same_bytes_twice_is_deduped(tmp_path: Path):
    registry = SourceRegistry(registry_path=tmp_path / "registry.json")
    file_bytes = b"%PDF-1.4 identical content"
    first, first_is_new = registry.register(
        source_url="https://example.com/a.pdf",
        license=License.PUBLIC_DOMAIN,
        file_bytes=file_bytes,
    )
    second, second_is_new = registry.register(
        source_url="https://example.com/a-mirror.pdf",
        license=License.PUBLIC_DOMAIN,
        file_bytes=file_bytes,
    )
    assert first_is_new is True
    assert second_is_new is False
    assert first.sha256 == second.sha256
    assert len(registry.all_records()) == 1


def test_registry_persists_across_instances(tmp_path: Path):
    registry_path = tmp_path / "registry.json"
    registry_a = SourceRegistry(registry_path=registry_path)
    registry_a.register(
        source_url="https://example.com/a.pdf",
        license=License.CC_BY,
        file_bytes=b"content-a",
    )
    registry_b = SourceRegistry(registry_path=registry_path)
    assert len(registry_b.all_records()) == 1
    assert registry_b.all_records()[0].license == License.CC_BY
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ingest/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'doclens.ingest.registry'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/doclens/ingest/registry.py
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from doclens.ingest.models import License, SourceRecord


class SourceRegistry:
    def __init__(self, registry_path: Path):
        self.registry_path = registry_path
        self._records: dict[str, SourceRecord] = {}
        if self.registry_path.exists():
            raw = json.loads(self.registry_path.read_text(encoding="utf-8"))
            for entry in raw:
                record = SourceRecord.model_validate(entry)
                self._records[record.sha256] = record

    def register(
        self, source_url: str, license: License, file_bytes: bytes
    ) -> tuple[SourceRecord, bool]:
        sha256 = hashlib.sha256(file_bytes).hexdigest()
        if sha256 in self._records:
            return self._records[sha256], False
        record = SourceRecord(
            source_url=source_url,
            license=license,
            retrieved_at=datetime.now(timezone.utc),
            sha256=sha256,
        )
        self._records[sha256] = record
        self._save()
        return record, True

    def all_records(self) -> list[SourceRecord]:
        return list(self._records.values())

    def _save(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [json.loads(r.model_dump_json()) for r in self._records.values()]
        self.registry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ingest/test_registry.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/doclens/ingest/registry.py tests/ingest/test_registry.py
git commit -m "feat(ingest): add content-addressed source registry with dedup"
```

---

### Task 3: PDF extractor (text + page images)

**Files:**
- Create: `src/doclens/ingest/pdf_extractor.py`
- Test: `tests/ingest/test_pdf_extractor.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (standalone; `Page.text_source` values are produced by
  the caller in Task 5, not here).
- Produces: `PdfExtractor()` with method
  `extract(pdf_bytes: bytes, doc_id: str, image_output_dir: Path) -> list[RawPage]`, where
  `RawPage = NamedTuple("RawPage", [("page_number", int), ("text", str), ("has_text_layer",
  bool), ("image_path", Path)])`. Renders each page to `image_output_dir / f"page-{n}.png"` at
  150 DPI. `has_text_layer=False` when extracted text is empty/whitespace-only after stripping —
  this is the signal Task 4's OCR fallback acts on.

- [ ] **Step 1: Write the failing test**

```python
# tests/ingest/test_pdf_extractor.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ingest/test_pdf_extractor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'doclens.ingest.pdf_extractor'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/doclens/ingest/pdf_extractor.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ingest/test_pdf_extractor.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/doclens/ingest/pdf_extractor.py tests/ingest/test_pdf_extractor.py
git commit -m "feat(ingest): add PyMuPDF-based text and page-image extraction"
```

---

### Task 4: OCR fallback

**Files:**
- Create: `src/doclens/ingest/ocr_fallback.py`
- Test: `tests/ingest/test_ocr_fallback.py`

**Interfaces:**
- Consumes: nothing from earlier tasks directly (operates on an image path — Task 5 wires it to
  `RawPage.image_path` for pages where `has_text_layer is False`).
- Produces: `OcrFallback(languages: list[str] = ["en", "it"])` with method
  `read_text(image_path: Path) -> str`. Lazily loads the EasyOCR reader on first call (not in
  `__init__`) so importing this module doesn't trigger a model download.

- [ ] **Step 1: Write the failing test**

```python
# tests/ingest/test_ocr_fallback.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ingest/test_ocr_fallback.py -v -m "not slow"`
Expected: FAIL with `ModuleNotFoundError: No module named 'doclens.ingest.ocr_fallback'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/doclens/ingest/ocr_fallback.py
from pathlib import Path


class OcrFallback:
    def __init__(self, languages: list[str] | None = None):
        self.languages = languages or ["en", "it"]
        self._reader = None

    def read_text(self, image_path: Path) -> str:
        if self._reader is None:
            import easyocr  # heavy import; deferred until first real use

            self._reader = easyocr.Reader(self.languages, gpu=False)
        results = self._reader.readtext(str(image_path), detail=0)
        return " ".join(results)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ingest/test_ocr_fallback.py -v -m "not slow"`
Expected: PASS (1 passed — the fast init test; the `slow` test is opt-in since it downloads
EasyOCR's model weights on first run)

Register the `slow` marker in `pyproject.toml` (Task 5's setup covers `pyproject.toml`
creation; if it doesn't exist yet by this task, add a minimal `[tool.pytest.ini_options]
markers = ["slow: downloads model weights, run explicitly"]` block now).

- [ ] **Step 5: Commit**

```bash
git add src/doclens/ingest/ocr_fallback.py tests/ingest/test_ocr_fallback.py
git commit -m "feat(ingest): add EasyOCR fallback for pages without a text layer"
```

---

### Task 5: Ingestion pipeline + CLI

**Files:**
- Create: `src/doclens/ingest/pipeline.py`
- Create: `src/doclens/cli.py`
- Create: `pyproject.toml` (if not already created in Task 4)
- Test: `tests/ingest/test_pipeline.py`

**Interfaces:**
- Consumes: `Document`, `Page`, `SourceRecord`, `License` (models.py); `SourceRegistry`
  (registry.py); `PdfExtractor`, `RawPage` (pdf_extractor.py); `OcrFallback` (ocr_fallback.py)
- Produces: `run_ingestion(pdf_bytes: bytes, source_url: str, license: License, title: str,
  corpus_dir: Path, registry: SourceRegistry, pdf_extractor: PdfExtractor | None = None,
  ocr_fallback: OcrFallback | None = None) -> Document`. Writes `corpus_dir /
  f"{doc_id}.json"` (the serialized `Document`) and `corpus_dir / doc_id / page-N.png` (via
  `PdfExtractor`). If the document is a duplicate (registry says `is_new=False`), returns the
  existing `Document` loaded from disk without re-extracting. CLI command: `doclens ingest
  --pdf <path> --source-url <url> --license <license> --title <title> --corpus-dir <dir>`.

- [ ] **Step 1: Write the failing test**

```python
# tests/ingest/test_pipeline.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ingest/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'doclens.ingest.pipeline'`

- [ ] **Step 3: Write minimal implementation**

First, if not already present, create `pyproject.toml`:

```toml
# pyproject.toml
[project]
name = "doclens"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.0",
    "pymupdf>=1.24",
    "easyocr>=1.7",
    "typer>=0.12",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[project.scripts]
doclens = "doclens.cli:app"

[tool.pytest.ini_options]
markers = ["slow: downloads model weights, run explicitly"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
```

```python
# src/doclens/ingest/pipeline.py
import json
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
```

```python
# src/doclens/cli.py
from pathlib import Path

import typer

from doclens.ingest.models import License
from doclens.ingest.pipeline import run_ingestion
from doclens.ingest.registry import SourceRegistry

app = typer.Typer()


@app.command()
def ingest(
    pdf: Path = typer.Option(..., exists=True, help="Path to the source PDF"),
    source_url: str = typer.Option(..., help="Public URL the PDF was retrieved from"),
    license: License = typer.Option(..., help="License classification"),
    title: str = typer.Option(..., help="Human-readable document title"),
    corpus_dir: Path = typer.Option(Path("corpus"), help="Output corpus directory"),
) -> None:
    registry = SourceRegistry(registry_path=corpus_dir / "registry.json")
    document = run_ingestion(
        pdf_bytes=pdf.read_bytes(),
        source_url=source_url,
        license=license,
        title=title,
        corpus_dir=corpus_dir,
        registry=registry,
    )
    typer.echo(f"Ingested '{document.title}' -> {document.doc_id} ({len(document.pages)} pages)")


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ingest/test_pipeline.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/doclens/ingest/pipeline.py src/doclens/cli.py tests/ingest/test_pipeline.py
git commit -m "feat(ingest): wire extraction, OCR fallback, and registry into ingestion pipeline + CLI"
```

---

### Task 6: Dataset card generator

**Files:**
- Create: `src/doclens/ingest/dataset_card.py`
- Modify: `src/doclens/cli.py:1-5` (add import), append new command at end of file
- Test: `tests/ingest/test_dataset_card.py`

**Interfaces:**
- Consumes: `SourceRegistry.all_records() -> list[SourceRecord]` (registry.py); `Document`
  (models.py, loaded from `corpus_dir/*.json` for page/OCR statistics)
- Produces: `generate_dataset_card(corpus_dir: Path, registry: SourceRegistry) -> str` returning
  a markdown string; CLI command `doclens dataset-card --corpus-dir <dir>` writes it to
  `corpus_dir / "DATASET_CARD.md"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/ingest/test_dataset_card.py
from pathlib import Path

from doclens.ingest.dataset_card import generate_dataset_card
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


def test_dataset_card_reports_document_count_and_license_breakdown(tmp_path: Path):
    corpus_dir = tmp_path / "corpus"
    registry = SourceRegistry(registry_path=corpus_dir / "registry.json")
    run_ingestion(
        pdf_bytes=_make_pdf_with_text("Doc A"),
        source_url="https://example.com/a.pdf",
        license=License.MANUFACTURER_PUBLIC,
        title="Doc A",
        corpus_dir=corpus_dir,
        registry=registry,
    )
    run_ingestion(
        pdf_bytes=_make_pdf_with_text("Doc B"),
        source_url="https://example.com/b.pdf",
        license=License.PUBLIC_DOMAIN,
        title="Doc B",
        corpus_dir=corpus_dir,
        registry=registry,
    )

    card = generate_dataset_card(corpus_dir=corpus_dir, registry=registry)

    assert "Total documents: 2" in card
    assert "manufacturer_public: 1" in card
    assert "public_domain: 1" in card


def test_dataset_card_reports_ocr_page_fraction(tmp_path: Path):
    corpus_dir = tmp_path / "corpus"
    registry = SourceRegistry(registry_path=corpus_dir / "registry.json")
    run_ingestion(
        pdf_bytes=_make_pdf_with_text("Text page"),
        source_url="https://example.com/c.pdf",
        license=License.PUBLIC_DOMAIN,
        title="Doc C",
        corpus_dir=corpus_dir,
        registry=registry,
    )

    card = generate_dataset_card(corpus_dir=corpus_dir, registry=registry)

    assert "OCR-derived pages: 0 / 1" in card
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ingest/test_dataset_card.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'doclens.ingest.dataset_card'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/doclens/ingest/dataset_card.py
from collections import Counter
from pathlib import Path

from doclens.ingest.models import Document
from doclens.ingest.registry import SourceRegistry


def generate_dataset_card(corpus_dir: Path, registry: SourceRegistry) -> str:
    records = registry.all_records()
    license_counts = Counter(record.license.value for record in records)

    total_pages = 0
    ocr_pages = 0
    for json_path in corpus_dir.glob("*.json"):
        document = Document.model_validate_json(json_path.read_text(encoding="utf-8"))
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
```

Append to `src/doclens/cli.py`:

```python
from doclens.ingest.dataset_card import generate_dataset_card


@app.command(name="dataset-card")
def dataset_card(
    corpus_dir: Path = typer.Option(Path("corpus"), help="Corpus directory to summarize"),
) -> None:
    registry = SourceRegistry(registry_path=corpus_dir / "registry.json")
    card = generate_dataset_card(corpus_dir=corpus_dir, registry=registry)
    output_path = corpus_dir / "DATASET_CARD.md"
    output_path.write_text(card, encoding="utf-8")
    typer.echo(f"Wrote dataset card to {output_path}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ingest/test_dataset_card.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the full Milestone 1 test suite**

Run: `pytest tests/ingest/ -v -m "not slow"`
Expected: All tests pass (14 passed, 1 deselected)

- [ ] **Step 6: Commit**

```bash
git add src/doclens/ingest/dataset_card.py src/doclens/cli.py tests/ingest/test_dataset_card.py
git commit -m "feat(ingest): add dataset card generator summarizing licenses and OCR coverage"
```

---

## Milestone 1 exit criteria (from DESIGN.md)

- [ ] `doclens ingest` runs end-to-end on a real sourced PDF and produces a `Document` JSON +
  page images in `corpus/`.
- [ ] Duplicate content (same bytes, different URL) is deduped, not re-extracted.
- [ ] Pages without a text layer are OCR'd and flagged `text_source="ocr"`.
- [ ] `doclens dataset-card` produces `corpus/DATASET_CARD.md` with license and OCR-coverage
  breakdowns.
- [ ] All tests in `tests/ingest/` pass.

This closes Milestone 1 from `DESIGN.md`. Milestone 2 (baseline text-only retrieval system) gets
its own plan once real corpus documents have been sourced and run through this pipeline.
