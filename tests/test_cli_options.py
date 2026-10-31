import click
import pytest
from click.testing import CliRunner

from promptphoo.cli.options import VariadicOption, collect_key_value


@click.command()
@click.option("-c", "--config", cls=VariadicOption)
@click.option("-o", "--output", cls=VariadicOption)
@click.option("--flag", is_flag=True)
def cmd(config, output, flag):
    click.echo(f"{list(config)}|{list(output)}|{flag}")


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["-c", "a.yaml", "b.yaml"], "['a.yaml', 'b.yaml']|[]|False"),
        (["-c", "a.yaml", "-c", "b.yaml", "--flag"], "['a.yaml', 'b.yaml']|[]|True"),
        (["--config", "a", "b", "-o", "x.json", "y.csv"], "['a', 'b']|['x.json', 'y.csv']|False"),
        (["-c", "a", "--flag", "-o", "x"], "['a']|['x']|True"),
    ],
)
def test_variadic_option(args, expected):
    result = CliRunner().invoke(cmd, args)
    assert result.exit_code == 0, result.output
    assert result.output.strip() == expected


def test_collect_key_value():
    assert collect_key_value("--var", ("a=1", "b=x=y", "a=2")) == {"a": "2", "b": "x=y"}
    assert collect_key_value("--var", ()) is None
    for bad in ("novalue", "=v", " a=1"):
        with pytest.raises(click.BadParameter, match="key=value format"):
            collect_key_value("--var", (bad,))
