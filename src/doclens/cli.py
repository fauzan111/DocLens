from pathlib import Path

import typer

from doclens.ingest.dataset_card import generate_dataset_card
from doclens.ingest.models import License
from doclens.ingest.pipeline import run_ingestion
from doclens.ingest.registry import SourceRegistry
from doclens.retrieval.pipeline import TextOnlyRetriever, build_index

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


@app.command(name="build-index")
def build_index_command(
    corpus_dir: Path = typer.Option(Path("corpus"), help="Ingested corpus directory"),
    index_dir: Path = typer.Option(Path("index"), help="Output vector index directory"),
) -> None:
    count = build_index(corpus_dir=corpus_dir, index_dir=index_dir)
    typer.echo(f"Indexed {count} chunks from {corpus_dir} into {index_dir}")


@app.command()
def query(
    query_text: str = typer.Argument(..., help="The question to search for"),
    corpus_dir: Path = typer.Option(Path("corpus"), help="Ingested corpus directory"),
    index_dir: Path = typer.Option(Path("index"), help="Vector index directory (from build-index)"),
    top_k: int = typer.Option(5, help="Number of results to return"),
    answer: bool = typer.Option(False, "--answer", help="Also generate a grounded answer via Gemini"),
) -> None:
    retriever = TextOnlyRetriever(corpus_dir=corpus_dir, index_dir=index_dir)
    results = retriever.retrieve(query_text, top_k=top_k)

    for chunk, score in results:
        typer.echo(f"[{score:.4f}] {chunk.doc_title} (page {chunk.page_number}): {chunk.text[:200]}")

    if answer:
        from doclens.generation.generator import GeminiGenerator

        generator = GeminiGenerator()
        result = generator.generate(query_text, [chunk for chunk, _ in results])
        typer.echo(f"\nAnswer: {result.text}")
        for citation in result.citations:
            typer.echo(f"  - {citation.doc_title}, page {citation.page_number}")


if __name__ == "__main__":
    app()
