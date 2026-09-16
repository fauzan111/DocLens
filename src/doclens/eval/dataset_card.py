from collections import Counter

from doclens.eval.models import BenchmarkQuestion

SCANNED_PAGE_LIMITATION_NOTE = """## Known limitation: scanned_no_text_layer coverage

Of the ~29 pages flagged `text_source: "ocr"` across the 28-document seed corpus, only one
(Interroll 24V Roller Conveyor manual, page 77) has genuine, non-blank scanned content. The
other pages are confirmed-blank divider/filler pages (empty extracted text, blank rendered
images, tiny file sizes), independently verified twice during content curation. This slice is
therefore populated with 3 questions, all grounded in that single real page, rather than the
originally planned 8. Growing this slice meaningfully requires sourcing documents with
genuinely scanned (non-blank) content, not just documents that happen to have an OCR-tagged
page."""


def generate_benchmark_card(questions: list[BenchmarkQuestion]) -> str:
    slice_counts = Counter(q.slice for q in questions)
    language_counts = Counter(q.language for q in questions)
    split_counts = Counter(q.split for q in questions)
    slice_split_counts = Counter((q.slice, q.split) for q in questions)

    lines = [
        "# doclens-bench Dataset Card",
        "",
        f"Total questions: {len(questions)}",
        "",
        "## Slice breakdown",
        "",
    ]
    for slice_name in sorted(slice_counts):
        dev = slice_split_counts.get((slice_name, "dev"), 0)
        hidden = slice_split_counts.get((slice_name, "hidden"), 0)
        lines.append(f"- {slice_name}: {slice_counts[slice_name]} (dev={dev}, hidden={hidden})")

    lines += ["", "## Language breakdown", ""]
    for language in sorted(language_counts):
        lines.append(f"- {language}: {language_counts[language]}")

    lines += ["", "## Split totals", ""]
    for split_name in ("dev", "hidden"):
        lines.append(f"- {split_name}: {split_counts.get(split_name, 0)}")

    lines += ["", SCANNED_PAGE_LIMITATION_NOTE]
    return "\n".join(lines)
