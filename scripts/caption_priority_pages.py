"""One-off script: caption an exact list of (doc_id, page_number) pairs, spending
scarce free-tier quota precisely on pages real benchmark questions cite, rather
than sequential whole-document order. Not part of the shipped codebase.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from doclens.caption.cache import CaptionCache
from doclens.caption.captioner import PageCaptioner
from doclens.ingest.corpus import load_documents

corpus_dir = Path("corpus")
cache = CaptionCache(corpus_dir=corpus_dir)
captioner = PageCaptioner()

# Priority pages: exact (doc_id, page_number) pairs cited by real dev-split
# table_lookup/diagram_lookup benchmark questions, one per distinct document,
# ordered to maximize document diversity per quota unit spent.
priority_pairs = [
    ("041d79afdac54b1d46d4f7f2d69a5be49c51a02653b727b1bc6859a1e0fdd954", 46),  # CAREL
    ("f724d13cfbf00bddef8405d4861a790b2b2bf675b3c898c63f9e6d0ee90b29d3", 8),   # Arduino Opta
    ("65ea1b76c3b53927ca73740683d1b7166761a4992c168322693a4c068ab49027", 6),   # Danfoss
    ("f7309f2bd6935148d577ec0e828dba560b43ce0aedc3e5ddab1c783d7d014502", 15),  # ABB MNS
    ("e10e53f9c32b872e4c16ad9d1701d5d925ab39b29b9c9314d5722c359141ce3f", 16),  # Riello
    ("e10e53f9c32b872e4c16ad9d1701d5d925ab39b29b9c9314d5722c359141ce3f", 20),  # Riello
    ("f44000d8440373ab83211e1537d4b683cb39d695cc9e69cc94830c3248c502cd", 5),   # OMAL
    ("f44000d8440373ab83211e1537d4b683cb39d695cc9e69cc94830c3248c502cd", 8),   # OMAL
    ("f57f590661950ee04dca29bad40b89bd7609dd55b1d53284399f2d771d6ee49a", 15),  # Mecc Alte
    ("08c0041d03ac3519fef3b98a743835114c5b7cc523f3df805859844fe12673d2", 20),  # Bonfiglioli
    ("08c0041d03ac3519fef3b98a743835114c5b7cc523f3df805859844fe12673d2", 40),  # Bonfiglioli
    ("1506756fd352ea74ce5693c0cda8c4b7ddcaf08ec8125ce58bce29a8d1558641", 20),  # Interroll
    ("2fb2274f8ac748234a97025bd53a9d8c6238469d3a557fbda6f99454ea4b57a0", 30),  # FAAC
    ("3cba77928150108bce7e921374de04c6f0d2e24b1bd3c089a7be23755c69dcdb", 25),  # PumpWorks
    ("3d22b042d8cd054494e16ab61662a5ac2e48ad76bc78a40634a6b84f1c9e9d62", 6),   # Baltur
    ("44d688d6aafc10ca8bde3ba5b3440f33548a642af393da9a2f841b3373542caa", 15),  # Bray
    ("c6924fd792138999e805bae59215606fe4c23437e17b685a1d39165c67888885", 20),  # Legrand
    ("f724d13cfbf00bddef8405d4861a790b2b2bf675b3c898c63f9e6d0ee90b29d3", 9),   # Arduino Opta
]

documents_by_id = {doc.doc_id: doc for doc in load_documents(corpus_dir)}

done = 0
for doc_id, page_number in priority_pairs:
    if cache.has(doc_id, page_number):
        print(f"SKIP (already cached): {doc_id[:12]} page {page_number}")
        continue

    document = documents_by_id[doc_id]
    page = next(p for p in document.pages if p.page_number == page_number)

    try:
        caption = captioner.caption_page(page.image_path)
    except Exception as exc:  # noqa: BLE001 - report and stop on quota/API errors
        print(f"STOPPED at {document.title} page {page_number}: {exc}")
        break

    cache.set(doc_id, page_number, caption)
    done += 1
    preview = (caption[:70] + "...") if caption else "None (no visual content)"
    print(f"OK ({done}): {document.title} page {page_number}: {preview}")

print(f"\n{done} new pages captioned this run.")
