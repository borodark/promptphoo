"""Built-in local providers: echo, sequence, exec. Ports src/providers/{echo,sequence,
scriptCompletion}.ts."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from typing import Any

from promptphoo import cache, envars
from promptphoo.jsvalues import json_stringify
from promptphoo.prompts.processors import get_file_hashes, parse_script_parts
from promptphoo.providers.base import ApiProvider, ProviderResponse
from promptphoo.templating import get_engine
from promptphoo.tokens import accumulate_response_token_usage, create_empty_token_usage

logger = logging.getLogger(__name__)

_ANSI_ESCAPE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def strip_text(text: str) -> str:
    return _ANSI_ESCAPE.sub("", text)


class EchoProvider(ApiProvider):
    handles_own_delay = True

    def default_id(self) -> str:
        return "echo"

    def __str__(self) -> str:
        return "[Echo Provider]"

    async def call_api(self, prompt, context=None, options=None) -> ProviderResponse:
        if self.delay and self.delay > 0:
            await asyncio.sleep(self.delay / 1000)
        return {
            "output": prompt,
            "raw": prompt,
            "cost": 0,
            "cached": False,
            "tokenUsage": {"total": 0, "prompt": 0, "completion": 0, "numRequests": 1},
            "isRefusal": False,
            "metadata": (context or {}).get("metadata") or {},
        }


class SequenceProvider(ApiProvider):
    def __init__(self, options: dict[str, Any]) -> None:
        super().__init__(options)
        if not isinstance(self.config.get("inputs"), list):
            raise ValueError("Expected sequence provider config to contain an array of inputs")
        self.sequence_inputs: list[str] = self.config["inputs"]
        self.separator: str = self.config.get("separator") or "\n---\n"

    def default_id(self) -> str:
        return "sequence-provider"

    async def call_api(self, prompt, context=None, options=None) -> ProviderResponse:
        context = context or {}
        original = context.get("originalProvider")
        if original is None:
            raise ValueError("Expected originalProvider to be set")
        engine = get_engine()
        responses: list[Any] = []
        usage = create_empty_token_usage()
        for item in self.sequence_inputs:
            rendered = engine.render_string(item, {**(context.get("vars") or {}), "prompt": prompt})
            response = await original.call_api(rendered, context, options)
            if response.get("error"):
                return response
            responses.append(response.get("output"))
            accumulate_response_token_usage(usage, response)
        return {"output": self.separator.join(json_stringify(r) if not isinstance(r, str) else r
                                              for r in responses),
                "tokenUsage": usage}  # fmt: skip


class ScriptCompletionProvider(ApiProvider):
    """`exec:<command>`: runs the command with prompt, options JSON and context JSON as args."""

    def __init__(self, script_path: str, options: dict[str, Any] | None = None) -> None:
        super().__init__(options)
        self.script_path = script_path

    def default_id(self) -> str:
        return f"exec:{self.script_path}"

    async def call_api(self, prompt, context=None, options=None) -> ProviderResponse:
        parts = parse_script_parts(self.script_path)
        hashes = get_file_hashes(parts)
        if not hashes:
            logger.warning("Could not find any valid files in the command: %s", self.script_path)
        key = f"exec:{self.script_path}:{':'.join(hashes)}:{prompt}:{json_stringify(self.options)}"
        if hashes and cache.is_cache_enabled():
            cached = await cache.get_cache().get(key)
            if cached:
                import json

                return {**json.loads(cached), "cached": True}

        if not parts:
            raise ValueError("No command found in script path")
        ctx = {
            k: v
            for k, v in (context or {}).items()
            if k not in ("getCache", "logger", "filters", "originalProvider")
        }
        args = [*parts[1:], prompt, json_stringify(self.options or {}), json_stringify(ctx)]
        proc = await asyncio.create_subprocess_exec(
            parts[0],
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.config.get("basePath") or None,
            env=envars.get_process_env(),
        )
        stdout_b, stderr_b = await proc.communicate()
        if proc.returncode:
            raise RuntimeError(
                f"Command failed: {self.script_path}\n{stderr_b.decode('utf-8', 'replace')}"
            )
        stdout = strip_text(stdout_b.decode("utf-8", "replace").strip())
        stderr = strip_text(stderr_b.decode("utf-8", "replace").strip())
        if stderr:
            logger.debug("Error output from script %s: %s", self.script_path, stderr)
            if not stdout:
                raise RuntimeError(stderr)
        result = {"output": stdout}
        if hashes and cache.is_cache_enabled():
            await cache.get_cache().set(key, json_stringify(result))
        return result


def sha256(text: str | bytes) -> str:
    return hashlib.sha256(text.encode() if isinstance(text, str) else text).hexdigest()
