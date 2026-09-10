"""Download and ingest the documents listed in a seed manifest.

Usage:
    python scripts/seed_corpus.py [--manifest data/seed_manifest.json] [--corpus-dir corpus]

Each manifest entry is a JSON object: {"title": str, "source_url": str,
"license": str (a doclens.ingest.models.License value), "language": str
(informational only, not currently stored per-document)}.

Downloads are not committed to git (see .gitignore's `corpus/` entry) — this
script is what reproduces the corpus from the manifest, which IS committed.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from doclens.ingest.models import License
from doclens.ingest.pipeline import run_ingestion
from doclens.ingest.registry import SourceRegistry

USER_AGENT = "DocLens-corpus-seeder/0.1 (portfolio research project)"


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def seed(manifest_path: Path, corpus_dir: Path) -> None:
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    registry = SourceRegistry(registry_path=corpus_dir / "registry.json")

    succeeded = 0
    failed: list[tuple[str, str]] = []

    for entry in entries:
        title = entry["title"]
        source_url = entry["source_url"]
        license_value = License(entry["license"])

        print(f"[{title}]")
        try:
            pdf_bytes = download(source_url)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            print(f"  DOWNLOAD FAILED: {exc}")
            failed.append((title, str(exc)))
            continue

        try:
            document = run_ingestion(
                pdf_bytes=pdf_bytes,
                source_url=source_url,
                license=license_value,
                title=title,
                corpus_dir=corpus_dir,
                registry=registry,
            )
        except Exception as exc:  # noqa: BLE001 — report and continue past a single bad PDF
            print(f"  INGESTION FAILED: {exc}")
            failed.append((title, str(exc)))
            continue

        ocr_pages = sum(1 for page in document.pages if page.text_source == "ocr")
        print(f"  OK — {document.doc_id[:12]}... ({len(document.pages)} pages, {ocr_pages} OCR'd)")
        succeeded += 1

    print(f"\n{succeeded}/{len(entries)} documents ingested.")
    if failed:
        print("Failed:")
        for title, reason in failed:
            print(f"  - {title}: {reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/seed_manifest.json"),
        help="Path to the seed manifest JSON file",
    )
    parser.add_argument(
        "--corpus-dir", type=Path, default=Path("corpus"),
        help="Output corpus directory",
    )
    args = parser.parse_args()
    seed(args.manifest, args.corpus_dir)


if __name__ == "__main__":
    main()
