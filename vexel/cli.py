"""
vexel CLI — entry point for common indexing and search operations.

Usage:
    vexel --help
    vexel index --config vexel.yaml --input entities.jsonl
    vexel search --config vexel.yaml --image query.jpg --top-k 10
    vexel validate-config --config vexel.yaml
"""

from __future__ import annotations

import sys


def _require_typer():
    try:
        import typer

        return typer
    except ImportError:
        print(
            "typer is required for the vexel CLI.  Install with:\n"
            "  pip install typer[all]\n",
            file=sys.stderr,
        )
        sys.exit(1)


def app():
    """CLI entry point registered as ``vexel`` in pyproject.toml."""
    typer = _require_typer()
    cli = typer.Typer(
        name="vexel",
        help="Vexel — visual similarity search for e-commerce.",
        add_completion=False,
    )

    @cli.command("validate-config")
    def validate_config(
        config: str = typer.Option("vexel.yaml", "--config", "-c", help="Path to YAML config file"),
    ):
        """Validate a vexel.yaml config file and print the parsed settings."""
        from vexel.config import VexelConfig

        try:
            cfg = VexelConfig.from_yaml(config)
            typer.echo(f"✓ Config valid: {config}")
            typer.echo(f"  encoder: {cfg.encoder.name} / {cfg.encoder.model}")
            typer.echo(f"  store:   {cfg.store.backend}")
            if cfg.store.backend == "qdrant":
                typer.echo(
                    f"           {cfg.store.qdrant.host}:{cfg.store.qdrant.port}"
                    f"  collection={cfg.store.qdrant.collection}"
                )
        except Exception as exc:
            typer.echo(f"✗ Config invalid: {exc}", err=True)
            raise typer.Exit(1)

    @cli.command("index")
    def index(
        config: str = typer.Option("vexel.yaml", "--config", "-c"),
        input_file: str = typer.Option(..., "--input", "-i", help="JSONL file of entities"),
    ):
        """
        Index entities from a JSONL file.

        Each line must be a JSON object with at least:
            {"id": "<entity_id>", "images": ["<url1>", "<url2>"]}

        Extra fields are forwarded to the vector payload.
        """
        typer.echo("vexel index: not yet implemented in this release.")
        typer.echo(
            "Use the Python API directly:\n"
            "  from vexel import MultiVectorIndexer\n"
            "  indexer = MultiVectorIndexer(config, encoder, store)\n"
            "  await indexer.run(entities, ...)"
        )
        raise typer.Exit(1)

    @cli.command("search")
    def search(
        config: str = typer.Option("vexel.yaml", "--config", "-c"),
        image: str = typer.Option(..., "--image", "-q", help="Path to query image"),
        top_k: int = typer.Option(10, "--top-k", "-k"),
    ):
        """
        Run a one-shot visual search query from a local image file.
        """
        typer.echo("vexel search: not yet implemented in this release.")
        typer.echo(
            "Use the Python API directly:\n"
            "  from vexel import VisualSearchEngine\n"
            "  results = await engine.search(image_bytes, top_k=10)"
        )
        raise typer.Exit(1)

    cli()


if __name__ == "__main__":
    app()