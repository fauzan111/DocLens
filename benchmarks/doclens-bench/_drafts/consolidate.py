"""One-off script to merge the three independently-drafted benchmark question
files into one consistent shape. Not part of the shipped codebase.
"""
import json
from pathlib import Path

drafts_dir = Path(__file__).parent
bench_dir = drafts_dir.parent

mystery = json.loads((bench_dir / "seed_questions.json").read_text(encoding="utf-8"))
text_slices = json.loads((drafts_dir / "draft_text_slices.json").read_text(encoding="utf-8"))
visual_slices = json.loads((drafts_dir / "draft_visual_slices.json").read_text(encoding="utf-8"))

# Drop the one near-exact duplicate: text-013 duplicates mystery's q007
# (both: Arduino Opta operating temperature range, page 8, same answer).
text_slices = [q for q in text_slices if q["id"] != "text-013"]


def to_sources(entry):
    """Normalize every entry's citation(s) into a uniform sources list."""
    if entry["slice"] == "unanswerable":
        return []

    # Mystery agent's cross_document entry (q018) used semicolon-separated strings.
    if ";" in str(entry.get("doc_id") or ""):
        doc_ids = entry["doc_id"].split(";")
        doc_titles = entry["doc_title"].split(";")
        pages = entry["page_number"].split(";")
        return [
            {"doc_id": d.strip(), "doc_title": t.strip(), "page_number": p.strip()}
            for d, t, p in zip(doc_ids, doc_titles, pages)
        ]

    sources = [
        {
            "doc_id": entry["doc_id"],
            "doc_title": entry["doc_title"],
            "page_number": entry["page_number"],
        }
    ]
    # My text agent's cross_document entries used doc_id_2/doc_title_2/page_number_2.
    if entry.get("doc_id_2"):
        sources.append(
            {
                "doc_id": entry["doc_id_2"],
                "doc_title": entry["doc_title_2"],
                "page_number": entry["page_number_2"],
            }
        )
    return sources


consolidated = []
for entry in mystery + text_slices + visual_slices:
    consolidated.append(
        {
            "question": entry["question"],
            "language": entry["language"],
            "slice": entry["slice"],
            "expected_answer": entry["expected_answer"],
            "sources": to_sources(entry),
            "notes": entry.get("notes", ""),
        }
    )

# Group by slice for readability, then renumber sequentially.
slice_order = [
    "answerable_text",
    "table_lookup",
    "diagram_lookup",
    "scanned_no_text_layer",
    "cross_document",
    "unanswerable",
]
consolidated.sort(key=lambda q: slice_order.index(q["slice"]))
for i, q in enumerate(consolidated, start=1):
    q["id"] = f"bq-{i:03d}"
consolidated = [
    {k: q[k] for k in ["id", "slice", "language", "question", "expected_answer", "sources", "notes"]}
    for q in consolidated
]

out_path = drafts_dir / "consolidated_draft.json"
out_path.write_text(json.dumps(consolidated, indent=2, ensure_ascii=False), encoding="utf-8")

from collections import Counter
counts = Counter(q["slice"] for q in consolidated)
lang_counts = Counter(q["language"] for q in consolidated)
print(f"Total: {len(consolidated)}")
for s in slice_order:
    print(f"  {s}: {counts[s]}")
print(f"Language split: {dict(lang_counts)}")
print(f"Written to {out_path}")
