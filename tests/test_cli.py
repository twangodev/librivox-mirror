from typer.testing import CliRunner

from librivox_mirror.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "0.1.0"


def test_help_names_the_command() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "librivox-mirror" in result.stdout
