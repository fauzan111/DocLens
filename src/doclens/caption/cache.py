import json
from pathlib import Path


class CaptionCache:
    def __init__(self, corpus_dir: Path):
        self.corpus_dir = corpus_dir

    def _cache_path(self, doc_id: str) -> Path:
        return self.corpus_dir / f"{doc_id}.captions.json"

    def _load(self, doc_id: str) -> dict:
        path = self._cache_path(doc_id)
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def has(self, doc_id: str, page_number: int) -> bool:
        return str(page_number) in self._load(doc_id)

    def get(self, doc_id: str, page_number: int) -> str | None:
        return self._load(doc_id).get(str(page_number))

    def set(self, doc_id: str, page_number: int, caption: str | None) -> None:
        path = self._cache_path(doc_id)
        data = self._load(doc_id)
        data[str(page_number)] = caption
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
