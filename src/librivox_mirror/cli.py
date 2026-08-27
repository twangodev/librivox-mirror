from __future__ import annotations

import typer

from librivox_mirror import __version__

app = typer.Typer(
    name="librivox-mirror",
    help="Build and maintain an ML-ready Hugging Face mirror of LibriVox.",
    no_args_is_help=True,
)


@app.callback()
def root() -> None:
    """Build and maintain an ML-ready Hugging Face mirror of LibriVox."""


@app.command()
def version() -> None:
    """Print the installed package version."""
    typer.echo(__version__)


def main() -> None:
    app()
