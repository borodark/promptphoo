"""`python:` / `file://x.py` provider. Ports src/providers/pythonCompletion.ts and
src/python/workerPool.ts.

Scripts run in-process unless `config.pythonExecutable` or PROMPTFOO_PYTHON names another
interpreter; then a pool of persistent_wrapper.py workers is used, speaking the same
`CALL|fn|request|response` / `READY` / `DONE|response` protocol as Node.
Arguments and results are JSON round-tripped either way, so scripts see what they would under
Node and cannot mutate evaluator state.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from promptphoo import cache, cli_state, envars, python_hooks
from promptphoo.files import parse_path_or_glob, process_config_file_references
from promptphoo.jsvalues import json_stringify
from promptphoo.providers.base import ApiProvider, ProviderResponse
from promptphoo.providers.simple import sha256

logger = logging.getLogger(__name__)

_RESULT_FIELDS = {
    "call_api": "output",
    "call_embedding_api": "embedding",
    "call_classification_api": "classification",
}
NON_SERIALIZABLE_CONTEXT_KEYS = ("getCache", "logger", "filters", "originalProvider")


def sanitize_script_context(context: dict[str, Any] | None) -> dict[str, Any] | None:
    if context is None:
        return None
    return {k: v for k, v in context.items() if k not in NON_SERIALIZABLE_CONTEXT_KEYS}


def _json_roundtrip(value: Any) -> Any:
    return json.loads(json_stringify(value))


def has_script_result_error(result: Any) -> bool:
    return isinstance(result, dict) and result.get("error") not in (None, "")


def validate_script_result(api_type: str, function_name: str, result: Any) -> None:
    field = _RESULT_FIELDS[api_type]
    if not isinstance(result, dict) or (field not in result and "error" not in result):
        raise ValueError(
            f"The Python script `{function_name}` function must return a dict with an own "
            f"`{field}` or `error` property (inherited prototype properties are rejected), "
            f"instead got: {json_stringify(result)}"
        )


class PythonWorker:
    """One persistent_wrapper.py subprocess."""

    def __init__(self, script: str, function: str, python: str, timeout_ms: int | None) -> None:
        self.script = script
        self.function = function
        self.python = python
        self.timeout = (timeout_ms or 120_000) / 1000
        self.proc: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        wrapper = Path(__file__).parent.parent / "_data" / "python" / "persistent_wrapper.py"
        self.proc = await asyncio.create_subprocess_exec(
            self.python, str(wrapper), self.script, self.function,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=None,
            cwd=os.path.dirname(self.script) or None,
            env=envars.get_process_env(),
        )  # fmt: skip
        await self._wait_for(lambda line: line == "READY")

    async def _wait_for(self, predicate) -> None:
        assert self.proc and self.proc.stdout
        while True:
            raw = await asyncio.wait_for(self.proc.stdout.readline(), self.timeout)
            if not raw:
                raise RuntimeError(f"Python worker for {self.script} exited unexpectedly")
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if predicate(line):
                return
            if line:
                logger.debug("[python] %s", line)

    async def call(self, function: str, args: list[Any]) -> Any:
        assert self.proc and self.proc.stdin
        with tempfile.TemporaryDirectory(prefix="promptfoo-python-") as tmp:
            request = os.path.join(tmp, "request.json")
            response = os.path.join(tmp, "response.json")
            Path(request).write_text(json_stringify(args), encoding="utf-8")
            self.proc.stdin.write(f"CALL|{function}|{request}|{response}\n".encode())
            await self.proc.stdin.drain()
            await self._wait_for(lambda line: line == f"DONE|{response}")
            payload = json.loads(Path(response).read_text(encoding="utf-8"))
        if payload.get("type") == "error":
            raise RuntimeError(
                f"Error running Python script: {payload.get('error')}\n"
                f"{payload.get('traceback', '')}"
            )
        return payload.get("data")

    async def shutdown(self) -> None:
        if self.proc and self.proc.returncode is None:
            try:
                assert self.proc.stdin
                self.proc.stdin.write(b"SHUTDOWN\n")
                await self.proc.stdin.drain()
                await asyncio.wait_for(self.proc.wait(), 5)
            except (TimeoutError, BrokenPipeError, ConnectionResetError):
                self.proc.kill()


class PythonWorkerPool:
    def __init__(self, script, function, count, python, timeout_ms) -> None:
        self.workers = [PythonWorker(script, function, python, timeout_ms) for _ in range(count)]
        self.idle: asyncio.Queue[PythonWorker] = asyncio.Queue()

    async def initialize(self) -> None:
        await asyncio.gather(*(w.start() for w in self.workers))
        for w in self.workers:
            self.idle.put_nowait(w)

    async def execute(self, function: str, args: list[Any]) -> Any:
        worker = await self.idle.get()
        try:
            return await worker.call(function, args)
        finally:
            self.idle.put_nowait(worker)

    async def shutdown(self) -> None:
        await asyncio.gather(*(w.shutdown() for w in self.workers))


class PythonProvider(ApiProvider):
    def __init__(self, run_path: str, options: dict[str, Any] | None = None) -> None:
        super().__init__(options)
        base_path = self.config.get("basePath") or ""
        parsed = parse_path_or_glob(base_path, run_path)
        self.script_path = os.path.relpath(parsed["file_path"], base_path or os.curdir)
        self.function_name: str | None = parsed.get("function_name")
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._pool: PythonWorkerPool | None = None

    def default_id(self) -> str:
        return f"python:{self.script_path}:{self.function_name or 'default'}"

    @property
    def abs_path(self) -> str:
        return os.path.abspath(os.path.join(self.config.get("basePath") or "", self.script_path))

    def _python_executable(self) -> str | None:
        """An interpreter other than the running one, if configured."""
        configured = (
            self.config.get("pythonExecutable")
            or (self.env or {}).get("PROMPTFOO_PYTHON")
            or envars.get_env_string("PROMPTFOO_PYTHON")
        )
        if not configured:
            return None
        if os.path.realpath(configured) == os.path.realpath(sys.executable):
            return None
        return configured

    def _worker_count(self) -> int:
        if self.config.get("workers") is not None:
            return max(1, int(self.config["workers"]))
        env_workers = (self.env or {}).get("PROMPTFOO_PYTHON_WORKERS")
        n = (
            int(env_workers)
            if env_workers is not None
            else envars.get_env_int("PROMPTFOO_PYTHON_WORKERS")
        )
        if n is not None:
            return max(1, n)
        if cli_state.max_concurrency() is not None:
            return max(1, cli_state.max_concurrency() or 1)
        return 1

    async def initialize(self) -> None:
        async with self._init_lock:
            if self._initialized:
                return
            self.config = process_config_file_references(
                self.config, self.config.get("basePath") or ""
            )
            python = self._python_executable()
            if python:
                pool = PythonWorkerPool(
                    self.abs_path, self.function_name or "call_api", self._worker_count(),
                    python, self.config.get("timeout"),
                )  # fmt: skip
                await pool.initialize()
                self._pool = pool
            self._initialized = True

    async def _execute(self, prompt: str, context: dict[str, Any] | None, api_type: str) -> Any:
        await self.initialize()
        file_hash = sha256(Path(self.abs_path).read_text(encoding="utf-8"))
        key = (
            f"python:{self.script_path}:{self.function_name or 'default'}:{api_type}:{file_hash}:"
            f"{prompt}:{json_stringify(self.options)}:{json_stringify((context or {}).get('vars'))}"
        )
        if cache.is_cache_enabled():
            cached = await cache.get_cache().get(key)
            if cached:
                result = json.loads(cached)
                if api_type == "call_api" and isinstance(result, dict):
                    result["cached"] = True
                    if result.get("tokenUsage"):
                        total = result["tokenUsage"].get("total") or 0
                        result["tokenUsage"] = {
                            "cached": total,
                            "total": total,
                            "numRequests": result["tokenUsage"].get("numRequests") or 1,
                        }
                return result

        options = {**self.options, "config": {**(self.options.get("config") or {}), **self.config}}
        sanitized = sanitize_script_context(context)
        args: list[Any] = (
            [prompt, options, sanitized] if api_type == "call_api" else [prompt, options]
        )
        function = self.function_name or api_type
        if self._pool:
            result = await self._pool.execute(function, _json_roundtrip(args))
        else:
            fn = python_hooks.load_function(self.abs_path, function)
            result = _json_roundtrip(await python_hooks.call_async(fn, *_json_roundtrip(args)))

        validate_script_result(api_type, function, result)
        if cache.is_cache_enabled() and not has_script_result_error(result):
            await cache.get_cache().set(key, json_stringify(result))
        if api_type == "call_api" and isinstance(result, dict):
            result["cached"] = False
            usage = result.get("tokenUsage")
            if usage and not usage.get("numRequests"):
                usage["numRequests"] = 1
        return result

    async def call_api(self, prompt, context=None, options=None) -> ProviderResponse:
        return await self._execute(prompt, context, "call_api")

    async def call_embedding_api(self, prompt: str) -> dict[str, Any]:
        return await self._execute(prompt, None, "call_embedding_api")

    async def call_classification_api(self, prompt: str) -> dict[str, Any]:
        return await self._execute(prompt, None, "call_classification_api")

    async def cleanup(self) -> None:
        if self._pool:
            await self._pool.shutdown()
            self._pool = None
        self._initialized = False
