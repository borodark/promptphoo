"""Python 3 port of the promptfoo eval core.

The package root ports src/index.ts and src/node/index.ts for library callers:

    from promptphoo import evaluate, AbortController

    record = evaluate({"prompts": [...], "providers": [...], "tests": [...]})
    print(record.to_evaluate_summary()["stats"])

Names are resolved on first use, so `import promptphoo` stays cheap for callers that only need
the version.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.2.0"

# Version of upstream promptfoo whose behaviour, schema and migrations this port tracks.
UPSTREAM_VERSION = "0.125.1"

# name -> (module, attribute): upstream's root exports, in Python names.
_EXPORTS: dict[str, tuple[str, str]] = {
    "evaluate": ("promptphoo.api", "evaluate"),
    "evaluate_async": ("promptphoo.api", "evaluate_async"),
    "evaluate_in_thread": ("promptphoo.api", "evaluate_in_thread"),
    "evaluate_with_source": ("promptphoo.api", "evaluate_with_source"),
    "load_api_provider": ("promptphoo.providers", "load_api_provider"),
    "load_api_providers": ("promptphoo.providers", "load_api_providers"),
    "AbortController": ("promptphoo.scheduler.cancellation", "AbortController"),
    "AbortSignal": ("promptphoo.scheduler.cancellation", "AbortSignal"),
    "AbortError": ("promptphoo.scheduler.cancellation", "AbortError"),
    "Eval": ("promptphoo.models.eval", "Eval"),
    "ConfigResolutionError": ("promptphoo.config.load", "ConfigResolutionError"),
    "EvalRunError": ("promptphoo.cli.eval", "EvalRunError"),
    "OutputWriteError": ("promptphoo.output.writer", "OutputWriteError"),
    "cache": ("promptphoo.cache", ""),
    "assertions": ("promptphoo.assertions", ""),
}

__all__ = [
    "UPSTREAM_VERSION",
    "__version__",
    "AbortController",
    "AbortError",
    "AbortSignal",
    "ConfigResolutionError",
    "Eval",
    "EvalRunError",
    "OutputWriteError",
    "assertions",
    "cache",
    "evaluate",
    "evaluate_async",
    "evaluate_in_thread",
    "evaluate_with_source",
    "load_api_provider",
    "load_api_providers",
]


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module 'promptphoo' has no attribute {name!r}") from None
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, attribute) if attribute else module


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


if TYPE_CHECKING:  # pragma: no cover - static names for type checkers and IDEs
    from promptphoo import assertions as assertions
    from promptphoo import cache as cache
    from promptphoo.api import evaluate as evaluate
    from promptphoo.api import evaluate_async as evaluate_async
    from promptphoo.api import evaluate_in_thread as evaluate_in_thread
    from promptphoo.api import evaluate_with_source as evaluate_with_source
    from promptphoo.cli.eval import EvalRunError as EvalRunError
    from promptphoo.config.load import ConfigResolutionError as ConfigResolutionError
    from promptphoo.models.eval import Eval as Eval
    from promptphoo.output.writer import OutputWriteError as OutputWriteError
    from promptphoo.providers import load_api_provider as load_api_provider
    from promptphoo.providers import load_api_providers as load_api_providers
    from promptphoo.scheduler.cancellation import AbortController as AbortController
    from promptphoo.scheduler.cancellation import AbortError as AbortError
    from promptphoo.scheduler.cancellation import AbortSignal as AbortSignal
