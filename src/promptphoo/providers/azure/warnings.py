"""Model-graded assertion warning for Azure-only suites. Ports src/providers/azure/warnings.ts."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_AZURE_CLASSES = {"AzureChatCompletionProvider", "AzureCompletionProvider"}
_OPENAI_CLASSES = {
    "OpenAiChatCompletionProvider",
    "OpenAiCompletionProvider",
    "OpenAiAssistantProvider",
}


def maybe_emit_azure_openai_warning(
    test_suite: dict[str, Any], tests: list[dict[str, Any]]
) -> bool:
    """Warn when Azure providers meet model-graded assertions that have no grading provider."""
    from promptphoo.assertions.index import MODEL_GRADED_ASSERTION_TYPES

    providers = test_suite.get("providers") or []
    has_azure = any(type(p).__name__ in _AZURE_CLASSES for p in providers)
    has_openai = any(type(p).__name__ in _OPENAI_CLASSES for p in providers)
    default_test = test_suite.get("defaultTest")
    default_provider = (
        (default_test.get("options") or {}).get("provider")
        if isinstance(default_test, dict)
        else None
    )
    if has_azure and not has_openai and not default_provider:
        model_graded = [
            a
            for t in tests
            for a in (t.get("assert") or [])
            if isinstance(a, dict)
            and a.get("type") != "assert-set"
            and a.get("type") in MODEL_GRADED_ASSERTION_TYPES
            and not a.get("provider")
            and not (t.get("options") or {}).get("provider")
        ]
        if model_graded:
            assert_types = ", ".join(dict.fromkeys(a["type"] for a in model_graded))
            logger.warning(
                "You are using model-graded assertions of types %s while testing an Azure "
                "provider. You may need to override these to use your Azure deployment. To learn "
                "more, see https://promptfoo.dev/docs/providers/azure/#model-graded-tests",
                assert_types,
            )
            return True
    return False
