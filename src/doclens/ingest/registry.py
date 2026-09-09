import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from doclens.ingest.models import License, SourceRecord


class SourceRegistry:
    def __init__(self, registry_path: Path):
        self.registry_path = registry_path
        self._records: dict[str, SourceRecord] = {}
        if self.registry_path.exists():
            raw = json.loads(self.registry_path.read_text(encoding="utf-8"))
            for entry in raw:
                record = SourceRecord.model_validate(entry)
                self._records[record.sha256] = record

    def register(
        self, source_url: str, license: License, file_bytes: bytes
    ) -> tuple[SourceRecord, bool]:
        sha256 = hashlib.sha256(file_bytes).hexdigest()
        if sha256 in self._records:
            return self._records[sha256], False
        record = SourceRecord(
            source_url=source_url,
            license=license,
            retrieved_at=datetime.now(timezone.utc),
            sha256=sha256,
        )
        self._records[sha256] = record
        self._save()
        return record, True

    def all_records(self) -> list[SourceRecord]:
        return list(self._records.values())

    def _save(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [json.loads(r.model_dump_json()) for r in self._records.values()]
        self.registry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
