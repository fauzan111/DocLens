# DocLens: Design

**Multimodal RAG for industrial technical documents: does indexing page images and tables
actually beat text-only RAG, and where does it not?**

## Problem statement

An Italian manufacturing SME has years of equipment manuals, datasheets, and spec sheets,
mostly scanned or mixed-format PDFs full of tables, wiring diagrams, and photos. Staff search
by keyword and miss answers that live in a table or diagram rather than a sentence.

> Research question: how much does retrieval quality improve when a RAG system indexes page
> images and tables directly, instead of only extracted text, and on which question types does
> it *not* help, given the added cost and complexity?

This is a portfolio flagship in the same spirit as TrustGate and AgentSec Range: one hard
question, one benchmark, one decision report. It does not depend on either of those repos and
does not require them to make sense, though its evaluator could later plug into TrustGate's
typed evaluator contract as a second product it measures.

## Non-goals / uniqueness

- Not another energy, insurance, invoice, support, or security-domain project (already covered
  by supply-chain-risk-fabric and AgentSec Range).
- Not a retrieval-quality re-hash of TrustGate: TrustGate evaluates *any* system; DocLens is
  the system being evaluated, with its own domain-specific benchmark.
- Not a single "best pipeline" demo: the deliverable is a controlled comparison across
  retrieval architectures, with a documented failure taxonomy.

## Architecture: three retrieval systems compared head-to-head

All three answer every benchmark question independently, using the same corpus, so results are
directly comparable. No query router in v1; routing between them is a natural v2 feature, not
required for the flagship claim.

1. **Text-only baseline.** OCR/text-extraction, then chunk, then dense embedding (open
   multilingual text embedder: BGE-M3 or multilingual-e5) plus BM25 hybrid plus cross-encoder
   reranker. Represents "what everyone already builds" and sets the floor.
2. **Caption-and-index.** Gemini Flash (free tier, vision-capable) generates a structured
   caption/description for every image, table, and diagram at ingest time. Captions are embedded
   and indexed as ordinary text passages alongside extracted text chunks.
3. **Unified vision embedding.** Each page is embedded directly as an image (open model: SigLIP)
   into the same vector space used for queries, so retrieval never depends on text extraction
   succeeding; this is what should rescue scanned, no-text-layer pages.
4. **Stretch, time-permitting.** ColQwen2-style multi-vector late-interaction retrieval, run in
   short batches on free Colab/Kaggle GPU sessions, compared as a fourth arm.

## Data

- **Corpus:** real public equipment manuals, datasheets, and standards-preview PDFs sourced from
  manufacturer sites and public catalogs. Bilingual IT/EN. Target ~150-250 documents /
  1,500-3,000 pages for v1, scaled up opportunistically.
- **Dataset card:** source, license, retrieval date, and any usage restrictions recorded per
  document before ingestion. No confidential or employer (GEKO) data.
- **Benchmark (`doclens-bench`):** 200-300 questions, versioned, across slices:
  - answerable (single-page, text)
  - table-lookup
  - diagram/figure-lookup
  - scanned / no-text-layer page
  - multi-document / cross-page
  - unanswerable (corpus doesn't support an answer; correct behavior is abstention)
  - Italian, English, and mixed-language queries
  - 20% held out as a hidden split, frozen before any system sees it, with contamination checks
    against the dev split (n-gram overlap).
- **Ground truth:** each question is labeled with the correct source page(s) and a reference
  answer before evaluation begins.

## Evaluation

Per-system, per-slice:

- Recall@5, Recall@10, nDCG@10 (retrieval quality)
- Citation precision (does the cited page actually support the generated answer)
- Answer faithfulness (is the answer backed by retrieved content, not model memory)
- Abstention precision (does the system correctly refuse when the corpus has no answer)
- p50/p95 latency and cost per answered query (API calls, embedding compute)

**The failure taxonomy is the actual deliverable**, not a single headline number: a table of
which question types each of the three systems gets wrong, with concrete examples, so the report
can state plainly where multimodal retrieval earns its added complexity and where it doesn't.

## Stack (all free-tier / $0 ongoing cost)

| Layer | Choice | Why |
|---|---|---|
| Generation / VLM | Gemini Flash API (free tier) | Vision-capable, usable free quota |
| Text embeddings | BGE-M3 or multilingual-e5 (open, local CPU) | Multilingual, no API cost |
| Vision embeddings | SigLIP (open, local CPU/light GPU) | Direct page-image embedding |
| Stretch: multi-vector | ColQwen2 (open) | Late-interaction, run on free Colab/Kaggle GPU |
| Vector store | Qdrant (local) | Free, supports hybrid + multi-vector |
| BM25 | rank_bm25 or local OpenSearch | Free, simple |
| API / orchestration | Python, FastAPI | Matches existing repos' stack |
| Eval harness | Custom, typed contract (TrustGate-compatible shape) | Reusable later |
| Tests / CI | pytest, GitHub Actions | Matches existing repos |

No Claude/Anthropic API in the core pipeline (keeps cost at $0); Claude may be used ad hoc for
report/documentation polish only, outside the measured pipeline.

## Repo layout

```
DocLens/
  src/doclens/
    ingest/        # PDF/image parsing, OCR, chunking, VLM captioning
    embed/         # text embedder, vision embedder wrappers
    retrieval/     # 3 (or 4) retrieval systems behind one typed interface
    eval/          # benchmark loader, metrics, failure-taxonomy report
    api/           # FastAPI query endpoint
    cli.py
  benchmarks/doclens-bench/   # versioned question set, dev/hidden split, dataset card
  docs/            # architecture notes, dataset card, technical report
  tests/
```

## Milestones (exit-criterion based, not week-numbered; pace varies)

1. **Corpus + ingestion.** Documents sourced, licensed, and logged; text extraction with OCR
   fallback; page images stored; dataset card written.
2. **Baseline works end-to-end.** Text-only hybrid retrieval + generation, first 50 benchmark
   questions answerable, smoke-tested.
3. **Multimodal arms + full benchmark.** Caption-and-index and vision-embedding systems working;
   full 200-300 question benchmark built; hidden split frozen; contamination check passes.
4. **Comparison + failure taxonomy.** All three (or four) systems run against the full benchmark;
   metrics table and failure taxonomy report produced.
5. **Stretch: ColQwen2 arm**, if time allows.
6. **Polish.** README, three-minute demo, technical report ending in a go/no-go verdict on
   multimodal retrieval for this domain, same repository standard as TrustGate/AgentSec Range.

## Acceptance gate

A fresh reviewer can reproduce the benchmark run from a dataset tag and config. Every headline
metric is reported with its sample size and per-slice breakdown, not just an aggregate. The
report explicitly states which question types justify multimodal retrieval's added complexity
and cost, and which don't; a negative or mixed result, honestly reported, is an acceptable and
credible outcome.

## Risks / limits

- Public manufacturer PDFs can change or be taken down; record retrieval dates, cache only
  what license terms allow.
- Free-tier API quotas (Gemini) may throttle a large ingestion run; confirmed in practice for
  vision captioning, where the free tier caps `gemini-3.6-flash` at 20 requests/day, making a
  full-corpus captioning pass impractical without billing enabled. Batch and rate-limit
  ingestion, and keep a fallback path (a smaller, targeted page scope) when quota is a hard
  blocker, as this project does for its caption-and-index arm.
- Gemini model names and SDKs are not stable over time: `gemini-2.0-flash` was retired mid-
  project and the `google.generativeai` SDK package is itself deprecated in favor of
  `google.genai`. Pin a model name explicitly, expect to update it, and treat an unannounced
  model retirement as a normal operating condition, not a bug in this codebase.
- Open multilingual/vision embedding model quality is lower than paid alternatives (Voyage,
  Cohere Embed-4); this is a known, disclosed trade-off in the report, not hidden.
- OCR quality on scanned pages varies; this is expected to be exactly where the vision-embedding
  arm shows its value, and is a benchmark slice, not a bug to eliminate.
