"""Prompt rendering with var loading. Ports renderPrompt / resolveVariables /
collectFileMetadata from src/evaluatorHelpers.ts."""

from __future__ import annotations

import base64
import logging
import os
import re
from typing import Any

from promptphoo import cli_state, envars, python_hooks
from promptphoo.files import is_audio_file, is_image_file, is_javascript_file, is_video_file
from promptphoo.jsonutil import js_json_parse
from promptphoo.jsvalues import js_string, json_stringify
from promptphoo.templating import extract_variables_from_template, get_engine, render_vars_in_object
from promptphoo.yamlload import load_yaml

logger = logging.getLogger(__name__)

FILE_METADATA_KEY = "_promptfooFileMetadata"

_IMAGE_MIME = {
    "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "gif": "image/gif",
    "bmp": "image/bmp", "webp": "image/webp", "svg": "image/svg+xml", "tif": "image/tiff",
    "tiff": "image/tiff", "ico": "image/x-icon", "avif": "image/avif", "heic": "image/heic",
    "heif": "image/heif",
}  # fmt: skip
_MAGIC = (
    (("/9j/",), "image/jpeg"),
    (("iVBORw0KGgo",), "image/png"),
    (("R0lGODlh", "R0lGODdh"), "image/gif"),
    (("UklGR",), "image/webp"),
    (("Qk0", "Qk1"), "image/bmp"),
    (("SUkq", "TU0A"), "image/tiff"),
    (("AAABAA",), "image/x-icon"),
)
_PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")
_PARTIAL_TAG = re.compile(r"({%[^%]*$|{{[^}]*$|{#[^#]*$)", re.MULTILINE)


def _ext(path: str) -> str:
    return path.split(".")[-1] if "." in path else path


def collect_file_metadata(vars_: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, value in vars_.items():
        if isinstance(value, str) and value.startswith("file://"):
            path = os.path.abspath(
                os.path.join(cli_state.base_path() or "", value[len("file://") :])
            )
            kind = (
                "image" if is_image_file(path)
                else "video" if is_video_file(path)
                else "audio" if is_audio_file(path)
                else None
            )  # fmt: skip
            if kind:
                out[name] = {"path": value, "type": kind, "format": _ext(path)}
    return out


def resolve_variables(
    variables: dict[str, Any],
    skip: list[str] | None = None,
    resolved_from_skipped: set[str] | None = None,
) -> dict[str, Any]:
    """Substitute `{{name}}` references between vars, up to 5 passes."""
    skip = skip or []
    for _ in range(5):
        done = True
        for key in list(variables):
            value = variables[key]
            if key in skip or (resolved_from_skipped and key in resolved_from_skipped):
                continue
            if not isinstance(value, str):
                continue
            m = _PLACEHOLDER.search(value)
            if m and variables.get(m.group(1)) is not None:
                variables[key] = value.replace(m.group(0), js_string(variables[m.group(1)]), 1)
                if resolved_from_skipped is not None and (
                    m.group(1) in skip or m.group(1) in resolved_from_skipped
                ):
                    resolved_from_skipped.add(key)
                done = False
        if done:
            break
    return variables


def _auto_wrap_raw(prompt: str) -> str:
    partial = _PARTIAL_TAG.search(prompt) is not None
    wrapped = re.search(r"{%\s*raw\s*%}", prompt) and re.search(r"{%\s*endraw\s*%}", prompt)
    return f"{{% raw %}}{prompt}{{% endraw %}}" if partial and not wrapped else prompt


def _references_undefined(template: str, vars_: dict[str, Any]) -> bool:
    for name in extract_variables_from_template(template):
        m = re.match(r"^([A-Za-z_]\w*)", name)
        if m and m.group(1) != "env" and m.group(1) not in vars_:
            return True
    return False


async def _load_var_file(name: str, value: str, base_prompt: str, vars_: dict, provider) -> Any:
    path = os.path.abspath(os.path.join(cli_state.base_path() or "", value[len("file://") :]))
    ext = _ext(path)
    if is_javascript_file(path):
        raise ValueError(
            f"JavaScript var loaders are not supported by promptphoo: {value}. "
            "Use a Python file with get_var(var_name, prompt, other_vars) instead."
        )
    if ext == "py":
        out = await python_hooks.run_python(path, "get_var", [name, base_prompt, vars_])
        out = out or {}
        if out.get("error"):
            raise RuntimeError(f"Error running Python script {path}: {out['error']}")
        if out.get("output") is None:
            raise RuntimeError(f"Python script {path} did not return any output")
        if not isinstance(out["output"], str):
            raise TypeError(
                "pythonScriptOutput.output must be a string. "
                f"Received: {type(out['output']).__name__}"
            )
        return out["output"].strip()
    if ext in ("yaml", "yml"):
        with open(path, encoding="utf-8") as f:
            return json_stringify(load_yaml(f.read()))
    if ext == "pdf" and not envars.get_env_bool("PROMPTFOO_DISABLE_PDF_AS_TEXT"):
        return _extract_pdf_text(path)
    if (is_image_file(path) or is_video_file(path) or is_audio_file(path)) and not (
        envars.get_env_bool("PROMPTFOO_DISABLE_MULTIMEDIA_AS_BASE64")
    ):
        kind = "image" if is_image_file(path) else "video" if is_video_file(path) else "audio"
        try:
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode("ascii")
        except OSError as e:
            raise RuntimeError(f"Failed to load {kind} {path}: {e}") from e
        if kind != "image":
            return data
        extension = os.path.splitext(path)[1]
        mime = _IMAGE_MIME.get(extension.lower().lstrip("."), "image/jpeg")
        unknown = not extension or mime == "image/jpeg"
        detected = next((m for prefixes, m in _MAGIC if data.startswith(prefixes)), None)
        if detected:
            mime = detected
        elif unknown:
            logger.warning(
                "Could not detect image format for %s, defaulting to image/jpeg. Supported "
                "formats: JPEG, PNG, GIF, WebP, BMP, TIFF, ICO, AVIF, HEIC, SVG",
                path,
            )
        return f"data:{mime};base64,{data}"
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def _extract_pdf_text(path: str) -> str:
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "pypdf is not installed. Please install it with: pip install promptphoo[pdf]"
        ) from e
    try:
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from PDF {path}: {e}") from e


async def render_prompt(
    prompt: dict[str, Any],
    vars_: dict[str, Any],
    filters: dict[str, Any] | None = None,
    provider: Any = None,
    skip_render_vars: list[str] | None = None,
    literal_vars: list[str] | None = None,
) -> str:
    """Load file:// vars, run prompt functions, and render the template. Mutates vars_.

    `literal_vars` (model outputs kept with storeOutputAs) are inserted as they are: never loaded
    as file:// paths, substituted into or rendered as templates, unlike in Node, where a model's
    output could read files or `{{ env.* }}` secrets into the next prompt (D73). Unlike
    `skip_render_vars` they still lose one trailing newline, as every var does."""
    engine = get_engine(filters)
    trim_skip = skip_render_vars or []
    skip = [*trim_skip, *(literal_vars or [])]
    base_prompt: str = prompt["raw"]

    for name, value in list(vars_.items()):
        if name in skip:
            continue
        if isinstance(value, str) and value.startswith("file://"):
            vars_[name] = await _load_var_file(name, value, base_prompt, vars_, provider)
        elif isinstance(value, str) and value.startswith("package:"):
            raise ValueError("package: var sources are not supported by promptphoo")

    if prompt.get("function"):
        result = await prompt["function"]({"vars": vars_, "provider": provider})
        if isinstance(result, str):
            base_prompt = result
        elif isinstance(result, dict | list):
            if isinstance(result, dict) and "prompt" in result:
                p = result["prompt"]
                base_prompt = p if isinstance(p, str) else json_stringify(p)
                if result.get("config"):
                    prompt["config"] = {**(prompt.get("config") or {}), **result["config"]}
            else:
                base_prompt = json_stringify(result)
        else:
            raise TypeError(
                f"Prompt function must return a string or object, got {type(result).__name__}"
            )

    for key, value in vars_.items():
        if isinstance(value, str) and key not in trim_skip and value.endswith("\n"):
            vars_[key] = value[:-1]
    resolved_from_skipped: set[str] = set()
    resolve_variables(vars_, skip, resolved_from_skipped)

    raw = prompt["raw"]
    if raw.startswith(("portkey://", "langfuse://", "helicone://")):
        raise ValueError(f"Remote prompt sources are not supported by promptphoo: {raw}")

    try:
        if envars.get_env_bool("PROMPTFOO_DISABLE_JSON_AUTOESCAPE"):
            return engine.render_string(_auto_wrap_raw(base_prompt), vars_)
        parsed = js_json_parse(base_prompt)
        return json_stringify(render_vars_in_object(parsed, vars_), 2)
    except Exception:  # Node falls back to plain-template rendering on any failure
        rendered_vars = {
            k: v
            if not isinstance(v, str)
            or k in skip
            or k in resolved_from_skipped
            or _references_undefined(v, vars_)
            else engine.render_string(_auto_wrap_raw(v), vars_)
            for k, v in vars_.items()
        }
        return engine.render_string(_auto_wrap_raw(base_prompt), rendered_vars)
