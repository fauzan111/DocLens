from datetime import datetime, timezone
from pathlib import Path

from doclens.ingest.models import License
from doclens.ingest.registry import SourceRegistry


def test_register_new_document_returns_is_new_true(tmp_path: Path):
    registry = SourceRegistry(registry_path=tmp_path / "registry.json")
    record, is_new = registry.register(
        source_url="https://example.com/pump-manual.pdf",
        license=License.MANUFACTURER_PUBLIC,
        file_bytes=b"%PDF-1.4 fake pdf bytes",
    )
    assert is_new is True
    assert record.source_url == "https://example.com/pump-manual.pdf"
    assert len(record.sha256) == 64


def test_register_same_bytes_twice_is_deduped(tmp_path: Path):
    registry = SourceRegistry(registry_path=tmp_path / "registry.json")
    file_bytes = b"%PDF-1.4 identical content"
    first, first_is_new = registry.register(
        source_url="https://example.com/a.pdf",
        license=License.PUBLIC_DOMAIN,
        file_bytes=file_bytes,
    )
    second, second_is_new = registry.register(
        source_url="https://example.com/a-mirror.pdf",
        license=License.PUBLIC_DOMAIN,
        file_bytes=file_bytes,
    )
    assert first_is_new is True
    assert second_is_new is False
    assert first.sha256 == second.sha256
    assert len(registry.all_records()) == 1


def test_registry_persists_across_instances(tmp_path: Path):
    registry_path = tmp_path / "registry.json"
    registry_a = SourceRegistry(registry_path=registry_path)
    registry_a.register(
        source_url="https://example.com/a.pdf",
        license=License.CC_BY,
        file_bytes=b"content-a",
    )
    registry_b = SourceRegistry(registry_path=registry_path)
    assert len(registry_b.all_records()) == 1
    assert registry_b.all_records()[0].license == License.CC_BY
