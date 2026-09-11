from pathlib import Path

from pydantic import ValidationError

from doclens.ingest.models import Document


def load_documents(corpus_dir: Path) -> list[Document]:
    documents: list[Document] = []
    for json_path in sorted(corpus_dir.glob("*.json")):
        if json_path.name == "registry.json":
            continue
        try:
            documents.append(Document.model_validate_json(json_path.read_text(encoding="utf-8")))
        except ValidationError:
            print(f"Skipping non-Document JSON file: {json_path.name}")
    return documents
