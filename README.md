# DocLens

**Multimodal RAG for industrial technical documents — does indexing page images and tables
actually beat text-only RAG, and where does it not?**

An Italian manufacturing SME has years of equipment manuals, datasheets, and spec sheets — mostly
scanned or mixed-format PDFs full of tables, wiring diagrams, and photos. Staff search by keyword
and miss answers that live in a table or diagram rather than a sentence.

> Research question: how much does retrieval quality improve when a RAG system indexes page
> images and tables directly, instead of only extracted text — and on which question types does
> it *not* help, given the added cost and complexity?

See [`DESIGN.md`](DESIGN.md) for the full architecture: three retrieval systems (text-only
hybrid baseline, caption-and-index, unified vision embedding) compared head-to-head on the same
bilingual (IT/EN) benchmark, with a documented failure taxonomy as the actual deliverable — not
a single accuracy number.

## Status

**Milestone 1 — corpus + ingestion (done).** A typed pipeline turns sourced PDFs into a
licensed, versioned corpus:

- Content-addressed source registry (SHA-256 dedup — the same file under a different URL is
  never re-ingested)
- PyMuPDF-based per-page text extraction + page-image rendering
- EasyOCR fallback for scanned pages with no text layer
- A dataset card summarizing license and OCR coverage

```bash
pip install -e .
doclens ingest --pdf manual.pdf --source-url https://example.com/manual.pdf \
  --license manufacturer_public --title "Pump Manual" --corpus-dir corpus
doclens dataset-card --corpus-dir corpus
```

**Next:** sourcing a real corpus of public manufacturer manuals/datasheets, then Milestone 2
(text-only hybrid retrieval baseline) and Milestone 3 (the multimodal retrieval arms + full
benchmark comparison).

## Run the tests

```bash
pip install -e ".[dev]"
pytest tests/ -v -m "not slow"
```

(The one `slow`-marked test downloads EasyOCR's model weights on first run — opt in with
`pytest tests/ -v` if you want to exercise it.)

## Layout

```
src/doclens/
  ingest/       models, source registry, PDF extraction, OCR fallback, pipeline (done)
  embed/        text + vision embedders (Milestone 2/3)
  retrieval/    the three retrieval systems compared head-to-head (Milestone 2/3)
  eval/         benchmark loader, metrics, failure-taxonomy report (Milestone 3)
  api/          FastAPI query endpoint
  cli.py        `doclens ingest`, `doclens dataset-card`
benchmarks/doclens-bench/   versioned question set, dev/hidden split (Milestone 3)
```

License: MIT (see [`LICENSE`](LICENSE)).
