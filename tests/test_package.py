from importlib import resources

from click.testing import CliRunner

from promptphoo import UPSTREAM_VERSION
from promptphoo.cli.main import cli


def test_version():
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert UPSTREAM_VERSION in result.output


def test_vendored_data_present():
    data = resources.files("promptphoo") / "_data"
    journal = data / "drizzle" / "meta" / "_journal.json"
    assert journal.is_file()
    assert (data / "tableOutput.html").is_file()
    assert (data / "python" / "persistent_wrapper.py").is_file()
