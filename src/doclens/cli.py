from pathlib import Path

import typer

from doclens.ingest.dataset_card import generate_dataset_card
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


@app.command(name="dataset-card")
def dataset_card(
    corpus_dir: Path = typer.Option(Path("corpus"), help="Corpus directory to summarize"),
) -> None:
    registry = SourceRegistry(registry_path=corpus_dir / "registry.json")
    card = generate_dataset_card(corpus_dir=corpus_dir, registry=registry)
    output_path = corpus_dir / "DATASET_CARD.md"
    output_path.write_text(card, encoding="utf-8")
    typer.echo(f"Wrote dataset card to {output_path}")


if __name__ == "__main__":
    app()
