# Seed benchmark: methodology and findings

`seed_questions.json` is a first tranche of grounded questions for `doclens-bench`, not the
full target set. Scope actually completed in this pass: **20 questions** (target for the first
full benchmark is ~50-60; DESIGN.md's eventual target is 200-300). Every question below was
produced by reading the real, already-ingested page text and, for `table_lookup` and
`diagram_lookup` items, viewing the actual page image, not generated from memory or assumption.

## Coverage by slice (this tranche)

| Slice | Count | Language split |
|---|---|---|
| table_lookup | 8 | 2 en, 6 it |
| answerable_text | 4 | 2 en, 2 it |
| diagram_lookup | 3 | 3 en |
| scanned_no_text_layer | 2 | 2 en |
| unanswerable | 2 | 2 en |
| cross_document | 1 | 1 en |

Not yet represented: a second cross-document question, more diagram_lookup and
scanned_no_text_layer coverage in Italian, and a genuinely mixed-language query. These are the
clear next additions for whoever continues this tranche toward the 50-60 target.

## A real finding, not just seed data: most "scanned" pages have no usable OCR text

While sourcing the `scanned_no_text_layer` slice, every known OCR-tagged page across the corpus
was checked directly (grep across all `corpus/*.json` for `"text_source": "ocr"`, then reading
each match). Result: **of roughly 28 pages flagged `text_source: "ocr"` across the corpus, only
one (Interroll manual, page 77) has non-empty, usable OCR text.** All the others returned an
empty string from EasyOCR.

Spot-checking the actual page image for one of the empty cases (Atlas Copco Compressed Air
Manual, page 2) confirmed the page is genuinely, entirely blank, a section-divider page with no
content at all, so an empty OCR result there is *correct*, not a bug. This was not checked for
every one of the 27 empty cases individually in this pass, only page 2 of Atlas Copco, but the
pattern (page 2 as a title-page divider, and several other empty pages recurring at low page
numbers like page 2 or 4 across multiple documents) is consistent with cover/divider pages
rather than OCR failure on real scanned content.

**Why this matters for Milestone 3:** the `scanned_no_text_layer` slice is exactly the case the
project expects the unified vision-embedding retrieval arm to rescue (DESIGN.md: "this is what
should rescue scanned, no-text-layer pages"). If most of the corpus's currently-tagged
"scanned" pages are actually blank dividers with nothing to retrieve, that slice will be
under-populated with real, answerable examples using only the current seed corpus. Two honest
options going forward, neither of which should be papered over:

1. **Source a few documents known to have genuinely scanned content pages** (old manuals,
   photocopied datasheets) specifically to populate this slice properly, rather than relying on
   whatever happened to get OCR-tagged during ingestion.
2. **Verify each of the 27 empty-OCR pages individually** (not just the one spot-checked here)
   to confirm they're really blank and not an EasyOCR failure on real scanned content. If any
   turn out to be real scanned content that OCR simply failed on, that is itself a Milestone 1
   pipeline finding worth its own investigation, and a much better scanned_no_text_layer example
   than a blank page.

This is exactly the kind of thing the project's honesty-first design (DESIGN.md's "report
negative results" instruction) is meant to catch before it silently weakens a benchmark slice.

## Process notes

- Grep was used across `corpus/*.json` to locate candidate facts (torque values, IP ratings,
  temperature limits) before opening any file, to avoid reading whole documents unnecessarily.
- Every `table_lookup` and `diagram_lookup` question's page image was actually viewed to confirm
  a genuine table or diagram is present, not inferred from extracted text alone.
- The cross-document question (q018) synthesizes three already-individually-verified facts
  (q006, q010, q016); it does not introduce any new unverified claim.
- Both unanswerable questions were checked against the full corpus document title list (q019,
  Grundfos, confirmed absent) or a full page-by-page read of the source manual (q005, PumpWorks
  PWI, no pressure rating found anywhere in its 26 pages) before being marked unanswerable, to
  avoid a false-negative ground truth label.
- A schema question is left open for whoever builds the benchmark infrastructure (typed model,
  loader, dev/hidden split): q018's `doc_id`/`page_number` fields use semicolon-separated
  multi-value strings as a placeholder representation for a multi-source citation, since the
  final `BenchmarkQuestion` schema hasn't been designed yet. This will need a real decision
  (e.g. `doc_id: list[str]` instead of `str`) when that infrastructure gets built.

## What this pass did NOT do

- Did not build the typed `BenchmarkQuestion` model, loader, dev/hidden split, or grounding
  validator described in the project's plan for this sub-project. This file and
  `seed_questions.json` are raw content, not yet validated code output.
- Did not reach the agreed ~50-60 question target; this is roughly a third of that.
- Did not get user spot-check review yet, per the earlier agreed process (agent drafts, human
  reviews a sample before it's treated as frozen ground truth).
