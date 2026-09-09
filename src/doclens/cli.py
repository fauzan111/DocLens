from pathlib import Path

import typer

from doclens.ingest.models import License
from doclens.ingest.pipeline import run_ingestion
from doclens.ingest.registry import SourceRegistry

app = typer.Typer()


@app.command()
def ingest(
    pdf: Path = typer.Option(..., exists=True, help="Path to the source PDF"),
    source_url: str = typer.Option(..., help="Public URL the PDF was retrieved from"),
    license: License = typer.Option(..., help="License classification"),
    title: str = typer.Option(..., help="Human-readable document title"),
    corpus_dir: Path = typer.Option(Path("corpus"), help="Output corpus directory"),
) -> None:
    registry = SourceRegistry(registry_path=corpus_dir / "registry.json")
    document = run_ingestion(
        pdf_bytes=pdf.read_bytes(),
        source_url=source_url,
        license=license,
        title=title,
        corpus_dir=corpus_dir,
        registry=registry,
    )
    typer.echo(f"Ingested '{document.title}' -> {document.doc_id} ({len(document.pages)} pages)")


if __name__ == "__main__":
    app()
