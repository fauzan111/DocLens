# DocLens

**A retrieval-augmented question-answering system for industrial technical documents, built to
measure whether reading page images and tables actually improves answers over reading text
alone, and where it doesn't.**

## The problem

Manufacturing companies sit on years of equipment manuals, datasheets, and spec sheets, most of
them scanned or mixed-format PDFs full of tables, wiring diagrams, and photos. Staff search by
keyword and routinely miss answers that live in a table or a diagram rather than a sentence: a
torque spec in a parts table, a wiring color code in a diagram, a limit value buried in a scanned
page that never made it through OCR cleanly.

DocLens ingests a library of these documents, indexes them, and answers natural-language
questions in English or Italian with citations back to the exact source page. Rather than just
shipping "a RAG system," the project is built to answer a specific, measurable question:

> How much does retrieval quality actually improve when a system indexes page images and tables
> directly, instead of only text extracted via OCR, and on which kinds of questions does it not
> help, given the added cost and complexity?

Most RAG demos never measure this. DocLens is built to.

## How it works

**1. Ingestion.** PDFs are parsed page by page: text is extracted where a text layer exists, and
EasyOCR fills in the gap for scanned pages that have none. Every page is also rendered to an
image. Every source document is recorded with its origin URL, license, and retrieval date, and
content-addressed by SHA-256 so the same file can never be silently duplicated under a different
name.

**2. Chunking and indexing.** Extracted text is split into page-scoped chunks (a chunk never
spans two pages, so a citation always points at one real page) and indexed two ways in parallel:
a classic BM25 lexical index, and a dense vector index built from multilingual sentence
embeddings. The two are combined with reciprocal rank fusion, then reranked with a multilingual
cross-encoder for the final result set.

**3. Answering.** The top reranked passages are handed to an LLM (Gemini, on its free tier) with
an instruction to answer only from the provided context and say so when it can't. Every answer
comes back with citations to the specific document and page it drew from.

**4. Evaluation.** The whole point of this project is to measure retrieval quality honestly, not
assume it. `doclens-bench` is a versioned set of real, source-grounded questions (every citation
independently checked against the actual corpus) split into slices that isolate exactly what
this project cares about: plain text lookups, table lookups, diagram lookups, scanned pages with
no usable text layer, questions requiring more than one document, and questions the corpus
genuinely doesn't answer. A held-out portion is frozen and never touched during development, so a
reported score can't be quietly tuned to. See [`benchmarks/doclens-bench/DATASET_CARD.md`](benchmarks/doclens-bench/DATASET_CARD.md)
for the current composition, including honestly-documented limitations.

**5. What's coming next.** The text pipeline above is the baseline, deliberately built first so
it can be measured, not assumed. The next phase adds two more ways of indexing the same
documents: captioning tables and diagrams with a vision-capable model before indexing them as
text, and embedding page images directly so retrieval never depends on text extraction succeeding
at all. All three approaches will be run against `doclens-bench` head-to-head, and the results,
including where the extra complexity does *not* pay off, will be published as part of this
repository. See [`DESIGN.md`](DESIGN.md) for the full technical design.

## Corpus

The system ships with a real, bilingual (Italian/English) seed corpus of publicly available
manufacturer manuals and datasheets: pumps, valves, compressors, PLCs and drives, motors, HVAC
equipment, and safety gear, sourced from manufacturer websites (28 documents, ~2,600 pages as of
this writing). Every document's source URL, license, and retrieval date is tracked in
[`data/seed_manifest.json`](data/seed_manifest.json), and the corpus is fully reproducible from
that manifest rather than checked into the repository directly.

## Getting started

Install the package:

```bash
pip install -e .
```

Build the seed corpus (downloads and ingests the manifest's real PDFs; this can take a few
minutes on first run):

```bash
python scripts/seed_corpus.py
```

Check what got ingested:

```bash
doclens dataset-card --corpus-dir corpus
```

Build the search index:

```bash
doclens build-index --corpus-dir corpus --index-dir index
```

Ask it a question:

```bash
doclens query "what torque should be used on the flange bolts" --corpus-dir corpus --index-dir index
doclens query "cosa fare in caso di sovraccarico del motore" --corpus-dir corpus --index-dir index --answer
```

The `--answer` flag additionally generates a grounded, cited answer via Gemini; without it, the
command returns the raw ranked passages and their source pages. Generation requires a free
`GEMINI_API_KEY` environment variable; retrieval alone does not.

To ingest your own documents instead of (or alongside) the seed corpus:

```bash
doclens ingest --pdf manual.pdf --source-url https://example.com/manual.pdf \
  --license manufacturer_public --title "Pump Manual" --corpus-dir corpus
```

## Running the tests

```bash
pip install -e ".[dev]"
pytest tests/ -v -m "not slow"
```

A handful of tests are marked `slow` because they exercise real model downloads (OCR, embeddings,
reranking) instead of stubs; they're skipped by default and can be run explicitly with
`pytest tests/ -v`.

## Project layout

```
src/doclens/
  ingest/       PDF parsing, OCR fallback, the source registry, and the ingestion pipeline
  embed/        the text embedding model wrapper
  retrieval/    chunking, BM25, the vector store, hybrid fusion, reranking, and the retrieval pipeline
  generation/   the answer generator and its citation/answer models
  eval/         the benchmark schema, grounding validator, dev/hidden split, and dataset card
  cli.py        the `doclens` command-line tool
benchmarks/doclens-bench/   the versioned, frozen evaluation question set and its dataset card
```

An API layer is planned but not yet built; see [`DESIGN.md`](DESIGN.md) for the full plan.

## License

MIT (see [`LICENSE`](LICENSE)).
