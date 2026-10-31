"""Commander-compatible option helpers for click."""

from __future__ import annotations

from typing import Any

import click


class VariadicOption(click.Option):
    """`-c a.yaml b.yaml`: like commander's `<paths...>`, greedily take following non-flag args.

    Repeating the flag also accumulates (`-c a.yaml -c b.yaml`).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("multiple", True)
        super().__init__(*args, **kwargs)

    def add_to_parser(self, parser: Any, ctx: click.Context) -> None:
        super().add_to_parser(parser, ctx)
        for name in self.opts:
            option = parser._long_opt.get(name) or parser._short_opt.get(name)
            if option is None:
                continue
            previous = option.process

            def process(value: Any, state: Any, previous: Any = previous) -> None:
                values = [value]
                while state.rargs and not state.rargs[0].startswith("-"):
                    values.append(state.rargs.pop(0))
                for v in values:
                    previous(v, state)

            option.process = process


def collect_key_value(option_name: str, values: tuple[str, ...]) -> dict[str, str] | None:
    """`--var k=v` (repeatable). Ports collectKeyValueOption."""
    if not values:
        return None
    out: dict[str, str] = {}
    for value in values:
        key, sep, val = value.partition("=")
        if not sep or not key or key.strip() != key:
            raise click.BadParameter(
                f"{option_name} must be specified in key=value format.", param_hint=option_name
            )
        out[key] = val
    return out
