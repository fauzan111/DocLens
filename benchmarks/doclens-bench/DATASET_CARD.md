# doclens-bench Dataset Card

Total questions: 66

## Slice breakdown

- answerable_text: 17 (dev=14, hidden=3)
- cross_document: 7 (dev=6, hidden=1)
- diagram_lookup: 11 (dev=9, hidden=2)
- scanned_no_text_layer: 3 (dev=2, hidden=1)
- table_lookup: 18 (dev=14, hidden=4)
- unanswerable: 10 (dev=8, hidden=2)

## Language breakdown

- en: 40
- it: 26

## Split totals

- dev: 53
- hidden: 13

## Known limitation: scanned_no_text_layer coverage

Of the ~29 pages flagged `text_source: "ocr"` across the 28-document seed corpus, only one
(Interroll 24V Roller Conveyor manual, page 77) has genuine, non-blank scanned content. The
other pages are confirmed-blank divider/filler pages (empty extracted text, blank rendered
images, tiny file sizes), independently verified twice during content curation. This slice is
therefore populated with 3 questions, all grounded in that single real page, rather than the
originally planned 8. Growing this slice meaningfully requires sourcing documents with
genuinely scanned (non-blank) content, not just documents that happen to have an OCR-tagged
page.