from __future__ import annotations

import typer

from dfs_opt.cli.training import app as training_app

app = typer.Typer(add_completion=False, help="DFS optimizer pipelines CLI")
app.add_typer(training_app, name="training")


def _main() -> None:
    app()


if __name__ == "__main__":
    _main()


