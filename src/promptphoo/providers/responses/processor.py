"""Shared Responses API output processing for OpenAI and Azure. Ports
src/providers/responses/processor.ts (and the config shapes in types.ts).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from promptphoo.finish_reason import normalize_finish_reason
from promptphoo.jsonutil import js_json_parse
from promptphoo.jsvalues import UNDEFINED, is_truthy, js_string, json_stringify
from promptphoo.providers.base import ProviderResponse
from promptphoo.providers.function_callbacks import FunctionCallbackHandler

logger = logging.getLogger(__name__)

# promptphoo.providers.openai imports this package, so its helpers load lazily.


def _openai_util() -> Any:
    from promptphoo.providers.openai import util

    return util


# (model_name, usage, request_config, response_data) -> cost or None.
CostCalculator = Callable[[str, Any, Any, Any], float | None]


def _extract_metadata(data: Any, processed: dict[str, Any] | None = None) -> dict[str, Any]:
    """User-facing metadata: response id, the model actually used, status and citations."""
    metadata: dict[str, Any] = {}
    if isinstance(data.get("id"), str) and data["id"]:
        metadata["responseId"] = data["id"]
    if isinstance(data.get("model"), str) and data["model"]:
        metadata["model"] = data["model"]
    if isinstance(data.get("status"), str) and data["status"]:
        metadata["responseStatus"] = data["status"]
    details = data.get("incomplete_details")
    reason = details.get("reason") if isinstance(details, dict) else None
    if isinstance(reason, str) and reason:
        metadata["incompleteReason"] = reason
    annotations = (processed or {}).get("annotations")
    if isinstance(annotations, list) and annotations:
        metadata["annotations"] = annotations
    return metadata


def get_responses_token_usage(data: Any, cached: bool) -> dict[str, Any]:
    """Token usage from Chat-style (prompt/completion) or Responses-style (input/output) usage."""
    usage = data.get("usage") if isinstance(data, dict) else None
    if not isinstance(usage, dict):
        return {}
    if cached:
        total = usage.get("total_tokens") or (
            (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0)
        )
        return {"cached": total, "total": total, "numRequests": 1}
    prompt = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    completion = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    total = usage.get("total_tokens") or prompt + completion
    details = _openai_util().get_openai_completion_token_details(usage)
    return {
        "total": total,
        "prompt": prompt,
        "completion": completion,
        "numRequests": 1,
        **({"completionDetails": details} if details else {}),
    }


class ResponsesProcessor:
    """`config` mirrors upstream's ProcessorConfig: `modelName`, `providerType`,
    `functionCallbackHandler` and `costCalculator`."""

    def __init__(
        self,
        model_name: str,
        provider_type: str,
        function_callback_handler: FunctionCallbackHandler,
        cost_calculator: CostCalculator,
    ) -> None:
        self.model_name = model_name
        self.provider_type = provider_type
        self.function_callback_handler = function_callback_handler
        self.cost_calculator = cost_calculator

    async def process_response_output(
        self,
        data: Any,
        request_config: dict[str, Any],
        cached: bool,
        suppress_reasoning_output: bool = False,
    ) -> ProviderResponse:
        logger.debug("Processing %s responses output", self.provider_type)
        if is_truthy(data.get("error")):
            return {"error": _openai_util().format_openai_error(data)}

        # Keep usable partial output while exposing why generation stopped.
        details = data.get("incomplete_details")
        finish_reason = (
            normalize_finish_reason(details.get("reason") if isinstance(details, dict) else None)
            if data.get("status") == "incomplete"
            else None
        )
        context = {
            "config": request_config,
            "cached": cached,
            "data": data,
            "suppressReasoningOutput": suppress_reasoning_output,
        }
        try:
            processed = await self._process_output(data.get("output"), context)
            cost = self.cost_calculator(self.model_name, data.get("usage"), request_config, data)
            if processed["isRefusal"]:
                return {
                    "output": processed["refusal"],
                    "tokenUsage": get_responses_token_usage(data, cached),
                    "isRefusal": True,
                    "cached": cached,
                    **({} if cost is None else {"cost": cost}),
                    "raw": data,
                    "metadata": _extract_metadata(data, processed),
                    **({"finishReason": finish_reason} if finish_reason else {}),
                }

            final_output: Any = processed["result"]
            response_format = request_config.get("response_format")
            if (
                isinstance(response_format, dict)
                and response_format.get("type") == "json_schema"
                and isinstance(final_output, str)
            ):
                try:
                    final_output = js_json_parse(final_output)
                except ValueError as error:
                    logger.error("Failed to parse JSON output: SyntaxError: %s", error)

            result: ProviderResponse = {
                "output": final_output,
                "tokenUsage": get_responses_token_usage(data, cached),
                "cached": cached,
                **({} if cost is None else {"cost": cost}),
                "raw": data,
                "metadata": _extract_metadata(data, processed),
                **({"finishReason": finish_reason} if finish_reason else {}),
            }
            # Backwards compatibility for code reading result.raw.annotations.
            if processed.get("annotations"):
                result["raw"] = {**data, "annotations": processed["annotations"]}
            return result
        except Exception as err:
            response: ProviderResponse = {
                "error": f"Error parsing response: {_openai_util().js_error_string(err)}\n"
                f"Response: {json_stringify(data)}"
            }
            if data.get("status") == "incomplete":
                response.update(
                    {
                        "tokenUsage": get_responses_token_usage(data, cached),
                        "cached": cached,
                        "raw": data,
                        "metadata": _extract_metadata(data),
                        **({"finishReason": finish_reason} if finish_reason else {}),
                    }
                )
            return response

    async def _process_output(self, output: Any, context: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(output, list) or not output:
            raise ValueError("Invalid response format: Missing output array")
        result = ""
        refusal = ""
        is_refusal = False
        annotations: list[Any] = []
        for item in output:
            if not isinstance(item, dict):
                logger.warning("Skipping invalid output item: %s", json_stringify(item))
                continue
            processed = await self._process_output_item(item, context)
            if processed.get("isRefusal"):
                refusal = processed.get("content") or ""
                is_refusal = True
            elif processed.get("content"):
                result = f"{result}\n{processed['content']}" if result else processed["content"]
            if processed.get("annotations"):
                annotations.extend(processed["annotations"])
        return {
            "result": result,
            "refusal": refusal,
            "isRefusal": is_refusal,
            "annotations": annotations or None,
        }

    async def _process_output_item(
        self, item: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        item_type = item.get("type")
        if item_type == "function_call":
            return await self._process_function_call(item, context)
        if item_type == "message":
            return await self._process_message(item, context)
        if item_type == "tool_result":
            return {"content": json_stringify(item)}
        if item_type == "reasoning":
            summary = item.get("summary")
            if context.get("suppressReasoningOutput") or not summary:
                return {}
            texts = "\n".join(_js(s, "text") for s in summary)
            return {"content": f"Reasoning: {texts}"}
        if item_type == "web_search_call":
            action = item.get("action")
            if isinstance(action, dict):
                action_type = action.get("type")
                if action_type == "search":
                    content = f'Web Search: "{_js(action, "query")}"'
                elif action_type == "open_page":
                    content = f"Opening page: {_js(action, 'url')}"
                elif action_type == "find_in_page":
                    content = f'Finding in page: "{_js(action, "query")}"'
                else:
                    content = f"Web action: {_js(action, 'type')}"
            else:
                content = f"Web Search Call (status: {item.get('status') or 'unknown'})"
            if item.get("status") == "failed" and is_truthy(item.get("error")):
                content += f" (Error: {js_string(item['error'])})"
            return {"content": content}
        if item_type == "code_interpreter_call":
            content = f"Code Interpreter: {item.get('code') or 'Running code...'}"
            if item.get("status") == "failed" and is_truthy(item.get("error")):
                content += f" (Error: {js_string(item['error'])})"
            return {"content": content}
        if item_type == "mcp_list_tools":
            tools = json_stringify(item.get("tools"), 2) if "tools" in item else "undefined"
            return {"content": f"MCP Tools from {_js(item, 'server_label')}: {tools}"}
        if item_type == "mcp_call":
            name = _js(item, "name")
            if is_truthy(item.get("error")):
                return {"content": f"MCP Tool Error ({name}): {js_string(item['error'])}"}
            return {"content": f"MCP Tool Result ({name}): {_js(item, 'output')}"}
        if item_type == "mcp_approval_request":
            return {
                "content": f"MCP Approval Required for {_js(item, 'server_label')}."
                f"{_js(item, 'name')}: {_js(item, 'arguments')}"
            }
        logger.debug("Unknown output item type: %s", item_type)
        return {}

    async def _process_function_call(
        self, item: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        # `{}` arguments on a completed call is a status update (often Chat-format tools sent
        # to the Responses API): report the call instead of running the callback.
        if item.get("arguments") == "{}" and item.get("status") == "completed":
            result = json_stringify(
                {
                    "type": "function_call",
                    "name": item.get("name"),
                    "status": "no_arguments_provided",
                    "note": "Function called but no arguments were extracted. Consider using "
                    "the correct Responses API tool format.",
                }
            )
        else:
            result = await self.function_callback_handler.process_calls(
                item, context["config"].get("functionToolCallbacks")
            )
        return {"content": result}

    async def _process_message(
        self, item: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        if item.get("role") != "assistant":
            return {}
        content: Any = ""
        is_refusal = False
        refusal: Any = ""
        annotations: list[Any] = []
        if is_truthy(item.get("content")):
            for content_item in item["content"]:
                if not isinstance(content_item, dict):
                    logger.warning(
                        "Skipping invalid content item: %s", json_stringify(content_item)
                    )
                    continue
                part_type = content_item.get("type")
                if part_type == "output_text":
                    content += _js(content_item, "text")
                    part_annotations = content_item.get("annotations")
                    if isinstance(part_annotations, list) and part_annotations:
                        annotations.extend(part_annotations)
                elif part_type in ("tool_use", "function_call"):
                    content = await self.function_callback_handler.process_calls(
                        content_item, context["config"].get("functionToolCallbacks")
                    )
                elif part_type == "refusal":
                    refusal = content_item.get("refusal")
                    is_refusal = True
        elif is_truthy(item.get("refusal")):
            refusal = item["refusal"]
            is_refusal = True
        return {
            "content": refusal if is_refusal else content,
            "isRefusal": is_refusal,
            "annotations": annotations or None,
        }


def _js(obj: Any, key: str) -> str:
    """`${obj.key}` in a template literal: a missing key prints `undefined`, null `null`."""
    return js_string(obj.get(key, UNDEFINED) if isinstance(obj, dict) else UNDEFINED)
