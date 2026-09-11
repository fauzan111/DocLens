# DocLens

**Multimodal RAG for industrial technical documents: does indexing page images and tables
actually beat text-only RAG, and where does it not?**

An Italian manufacturing SME has years of equipment manuals, datasheets, and spec sheets, mostly
scanned or mixed-format PDFs full of tables, wiring diagrams, and photos. Staff search by keyword
and miss answers that live in a table or diagram rather than a sentence.

> Research question: how much does retrieval quality improve when a RAG system indexes page
> images and tables directly, instead of only extracted text, and on which question types does
> it *not* help, given the added cost and complexity?

See [`DESIGN.md`](DESIGN.md) for the full architecture: three retrieval systems (text-only
hybrid baseline, caption-and-index, unified vision embedding) compared head-to-head on the same
bilingual (IT/EN) benchmark, with a documented failure taxonomy as the actual deliverable, not
a single accuracy number.

## Status

**Milestone 1, corpus + ingestion (done).** A typed pipeline turns sourced PDFs into a
licensed, versioned corpus:

- Content-addressed source registry (SHA-256 dedup; the same file under a different URL is
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

A real, bilingual (IT/EN) seed corpus of public manufacturer manuals (pumps, valves,
compressors, PLCs, drives, HVAC, safety equipment) is tracked as a manifest at
[`data/seed_manifest.json`](data/seed_manifest.json) (28 documents, ~2,600 pages as of this
writing) and reproduced locally with:

```bash
python scripts/seed_corpus.py   # downloads + ingests into corpus/ (gitignored, regenerable)
```

**Milestone 2, text-only hybrid retrieval baseline (done).** Chunks the corpus page-scoped,
indexes it with BM25 (lexical) and a multilingual dense embedder, fuses both with reciprocal
rank fusion, reranks with a multilingual cross-encoder, and optionally generates a grounded,
cited answer via Gemini's free tier:

```bash
doclens build-index --corpus-dir corpus --index-dir index
doclens query "what torque should be used on the flange bolts" --corpus-dir corpus --index-dir index
doclens query "cosa fare in caso di sovraccarico del motore" --corpus-dir corpus --index-dir index --answer
```

This is the "floor" baseline: what everyone already builds, measured honestly. A real smoke test
against the seed corpus already surfaced a genuine, disclosed limitation (an English query about
flange-bolt torque returned topically-adjacent-but-imprecise results, while an Italian query
returned clearly relevant ones) rather than hiding it, exactly the kind of failure-mode evidence
this project is meant to produce.

**Next:** grow the seed corpus further, then Milestone 3 (the multimodal retrieval arms:
caption-and-index and unified vision embedding, plus the full 200-300 question benchmark
comparing all systems head-to-head).

## Run the tests

```bash
pip install -e ".[dev]"
pytest tests/ -v -m "not slow"
```

(The one `slow`-marked test downloads EasyOCR's model weights on first run; opt in with
`pytest tests/ -v` if you want to exercise it.)

## Layout

```
src/doclens/
  ingest/       models, source registry, PDF extraction, OCR fallback, pipeline (done)
  embed/        text embedder (done); vision embedder (Milestone 3)
  retrieval/    chunker, BM25, vector store, hybrid fusion, reranker, pipeline (done);
                caption-and-index + vision-embedding arms (Milestone 3)
  generation/   Gemini answer generator with citations (done)
  eval/         benchmark loader, metrics, failure-taxonomy report (Milestone 3)
  api/          FastAPI query endpoint
  cli.py        `doclens ingest`, `dataset-card`, `build-index`, `query`
benchmarks/doclens-bench/   versioned question set, dev/hidden split (Milestone 3)
```

License: MIT (see [`LICENSE`](LICENSE)).
