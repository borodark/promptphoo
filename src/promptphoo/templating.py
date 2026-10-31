"""Nunjucks-compatible template rendering on Jinja2. Ports src/util/templates.ts and render.ts.

Jinja2 and Nunjucks share syntax but differ in semantics that show up in prompts. This module
closes the observable gaps:

* output of non-strings follows JS String(): true/false, "a,b" for arrays, "" for null/undefined;
* `if`/`not`/`and`/`or`/inline-if use JS truthiness (empty list/dict are truthy);
* `{% for k, v in obj %}` iterates an object's entries;
* JS member idioms: `.length`, `.toUpperCase()`, `.includes()`, `.join()`, ...;
* the trailing newline is kept; `truncate` and `dump` follow Nunjucks / JSON.stringify;
* `env` resolves config/env-file overrides, then os.environ, at lookup time.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import jinja2
from jinja2 import meta, nodes
from jinja2.visitor import NodeTransformer

from promptphoo import envars, python_hooks
from promptphoo.jsvalues import js_string, json_stringify

logger = logging.getLogger(__name__)

FilterMap = Mapping[str, Callable[..., Any]]


# --- env global -------------------------------------------------------------------------------


class TemplateEnv(Mapping[str, str]):
    """Live view of suite overrides > env-file overrides > os.environ (unless disabled)."""

    def _layers(self) -> list[Mapping[str, Any]]:
        layers: list[Mapping[str, Any]] = [envars.get_env_overrides() or {}]
        if not envars.is_template_process_env_disabled():
            layers += [envars.get_env_overrides("file") or {}, os.environ]
        return layers

    def __getitem__(self, key: str) -> str:
        for layer in self._layers():
            value = layer.get(key)
            if value is not None:
                return js_string(value)
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        seen: set[str] = set()
        for layer in self._layers():
            for key in layer:
                if key not in seen:
                    seen.add(key)
                    yield key

    def __len__(self) -> int:
        return sum(1 for _ in self)


TEMPLATE_ENV = TemplateEnv()


# --- JS semantics -----------------------------------------------------------------------------


def js_truthy(value: Any) -> bool:
    if isinstance(value, jinja2.Undefined):
        return False
    from promptphoo.jsvalues import is_truthy

    return is_truthy(value)


def _nj_items(value: Any) -> Any:
    if isinstance(value, Mapping):
        return list(value.items())
    return value


def _finalize(value: Any) -> Any:
    if isinstance(value, str | jinja2.Undefined):
        return value
    if value is None:
        return ""
    return js_string(value)


def _js_slice(seq: Any, start: int = 0, end: int | None = None) -> Any:
    return seq[start:end]


def _js_index_of(seq: Any, item: Any) -> int:
    try:
        return seq.index(item)
    except ValueError:
        return -1


_STR_METHODS: dict[str, Callable[[str], Any]] = {
    "toUpperCase": lambda s: lambda: s.upper(),
    "toLowerCase": lambda s: lambda: s.lower(),
    "trim": lambda s: lambda: s.strip(),
    "trimStart": lambda s: lambda: s.lstrip(),
    "trimEnd": lambda s: lambda: s.rstrip(),
    "includes": lambda s: lambda sub: sub in s,
    "startsWith": lambda s: lambda p: s.startswith(p),
    "endsWith": lambda s: lambda p: s.endswith(p),
    "indexOf": lambda s: lambda sub: s.find(sub),
    "split": lambda s: lambda sep=None: list(s) if sep == "" else s.split(sep),
    "slice": lambda s: lambda a=0, b=None: _js_slice(s, a, b),
    "substring": lambda s: lambda a=0, b=None: s[min(a, b or len(s)) : max(a, b or len(s))],
    "replace": lambda s: lambda old, new: s.replace(old, new, 1),
    "replaceAll": lambda s: lambda old, new: s.replace(old, new),
    "charAt": lambda s: lambda i=0: s[i] if 0 <= i < len(s) else "",
    "toString": lambda s: lambda: s,
}
_LIST_METHODS: dict[str, Callable[[list], Any]] = {
    "join": lambda xs: lambda sep=",": sep.join(js_string(x) if x is not None else "" for x in xs),
    "includes": lambda xs: lambda item: item in xs,
    "indexOf": lambda xs: lambda item: _js_index_of(xs, item),
    "slice": lambda xs: lambda a=0, b=None: _js_slice(xs, a, b),
    "toString": lambda xs: lambda: js_string(xs),
}


class _JsSemantics(NodeTransformer):
    """Rewrite boolean contexts to JS truthiness and two-target for-loops to entries."""

    def _truthy(self, node: nodes.Node) -> nodes.Expr:
        return nodes.Call(nodes.Name("__js_truthy", "load"), [node], [], None, None)  # type: ignore[list-item]

    def visit_If(self, node: nodes.If) -> nodes.Node:
        self.generic_visit(node)
        node.test = self._truthy(node.test)
        for elif_ in node.elif_:
            elif_.test = self._truthy(elif_.test)
        return node

    def visit_CondExpr(self, node: nodes.CondExpr) -> nodes.Node:
        self.generic_visit(node)
        node.test = self._truthy(node.test)
        return node

    def visit_Not(self, node: nodes.Not) -> nodes.Node:
        self.generic_visit(node)
        node.node = self._truthy(node.node)
        return node

    def visit_And(self, node: nodes.And) -> nodes.Node:
        # JS `a && b`: b if a is truthy else a.
        self.generic_visit(node)
        return nodes.CondExpr(self._truthy(node.left), node.right, node.left)

    def visit_Or(self, node: nodes.Or) -> nodes.Node:
        self.generic_visit(node)
        return nodes.CondExpr(self._truthy(node.left), node.left, node.right)

    def visit_For(self, node: nodes.For) -> nodes.Node:
        self.generic_visit(node)
        if isinstance(node.target, nodes.Tuple) and len(node.target.items) == 2:
            node.iter = nodes.Call(nodes.Name("__nj_items", "load"), [node.iter], [], None, None)
        if node.test is not None:
            node.test = self._truthy(node.test)
        return node


_STRICT_EQ = re.compile(r"(?<![=!<>])(===|!==)")
_TAG = re.compile(r"(\{\{.*?\}\}|\{%.*?%\})", re.S)


def _preprocess(source: str) -> str:
    """Translate Nunjucks-only operators inside tags (=== / !==)."""
    if "==" not in source:
        return source
    return _TAG.sub(lambda m: _STRICT_EQ.sub(lambda op: op.group(1)[:2], m.group(1)), source)


# --- filters ----------------------------------------------------------------------------------


def _filter_dump(value: Any, indent: int | str | None = None) -> str:
    if isinstance(value, jinja2.Undefined):
        return "undefined"
    return json_stringify(value, indent)


def _filter_load(value: Any) -> Any:
    return json.loads(value)


def _filter_join(value: Any, sep: str = "", attribute: str | None = None) -> str:
    items = list(value)
    if attribute is not None:
        items = [
            x.get(attribute) if isinstance(x, Mapping) else getattr(x, attribute) for x in items
        ]
    return sep.join("" if x is None else js_string(x) for x in items)


def _filter_nl2br(value: Any) -> str:
    return re.sub(r"\r\n|\n", "<br />\n", js_string(value))


def _filter_string(value: Any) -> str:
    return "" if isinstance(value, jinja2.Undefined) else js_string(value)


def _filter_truncate(
    value: Any, length: int = 255, killwords: bool = False, end: str | None = None
) -> str:
    """Nunjucks truncate: `length` excludes `end`, and cuts at the last space before it."""
    text = _filter_string(value)
    length = length or 255
    if len(text) <= length:
        return text
    if killwords:
        text = text[:length]
    else:
        idx = text.rfind(" ", 0, length + 1)
        text = text[: length if idx == -1 else idx]
    return text + ("..." if end is None else end)


_NUNJUCKS_ESCAPES = {"&": "&amp;", '"': "&quot;", "'": "&#39;", "<": "&lt;", ">": "&gt;",
                     "\\": "&#92;"}  # fmt: skip
_NUNJUCKS_ESCAPE_RE = re.compile(r"[&\"'<>\\]")


def _filter_escape(value: Any) -> Any:
    """Nunjucks `escape`/`e`: its own entity table (`&quot;`, `&#92;`), idempotent on safe
    strings. Jinja's markupsafe writes `&#34;` and leaves backslashes alone."""
    import markupsafe

    if isinstance(value, markupsafe.Markup):
        return value
    text = "" if value is None or isinstance(value, jinja2.Undefined) else js_string(value)
    return markupsafe.Markup(_NUNJUCKS_ESCAPE_RE.sub(lambda m: _NUNJUCKS_ESCAPES[m[0]], text))


def _filter_default(value: Any, default: Any = "", boolean: bool = False) -> Any:
    if isinstance(value, jinja2.Undefined) or (boolean and not js_truthy(value)):
        return default
    return value


# --- environment ------------------------------------------------------------------------------


class NunjucksCompatEnvironment(jinja2.Environment):
    def __init__(self, throw_on_undefined: bool = False, **kwargs: Any) -> None:
        super().__init__(
            autoescape=False,
            keep_trailing_newline=True,
            undefined=jinja2.StrictUndefined if throw_on_undefined else jinja2.ChainableUndefined,
            finalize=_finalize,
            **kwargs,
        )
        self.globals.update(env=TEMPLATE_ENV, __js_truthy=js_truthy, __nj_items=_nj_items)
        self.filters.update(
            dump=_filter_dump,
            load=_filter_load,
            join=_filter_join,
            nl2br=_filter_nl2br,
            string=_filter_string,
            truncate=_filter_truncate,
            default=_filter_default,
            d=_filter_default,
            escape=_filter_escape,
            e=_filter_escape,
            safe=lambda v: v,
        )

    def preprocess(self, source: str, name: str | None = None, filename: str | None = None) -> str:
        return super().preprocess(_preprocess(source), name, filename)

    def _parse(self, source: str, name: str | None, filename: str | None) -> nodes.Template:
        tree = super()._parse(source, name, filename)
        return _JsSemantics().visit(tree)

    def getattr(self, obj: Any, attribute: str) -> Any:
        # JS property access: an own key wins over a dict method (`json.items`, `data.keys`).
        if isinstance(obj, Mapping) and attribute in obj:
            return obj[attribute]
        if attribute == "length" and isinstance(obj, str | list | tuple | Mapping):
            if not (isinstance(obj, Mapping) and "length" in obj):
                return len(obj)
        if isinstance(obj, str) and attribute in _STR_METHODS:
            return _STR_METHODS[attribute](obj)
        if isinstance(obj, list | tuple) and attribute in _LIST_METHODS:
            return _LIST_METHODS[attribute](list(obj))
        return super().getattr(obj, attribute)


class Engine:
    """A configured environment plus promptfoo's renderString behaviour."""

    def __init__(self, env: NunjucksCompatEnvironment, is_grader: bool) -> None:
        self.env = env
        self.is_grader = is_grader

    def render_string(self, template: str, context: Mapping[str, Any] | None = None) -> str:
        if not self.is_grader and envars.get_env_bool("PROMPTFOO_DISABLE_TEMPLATING"):
            return template
        return self.env.from_string(template).render({"env": TEMPLATE_ENV, **(context or {})})


_engines: dict[tuple[int, bool, bool], Engine] = {}


def get_engine(
    filters: FilterMap | None = None, throw_on_undefined: bool = False, is_grader: bool = False
) -> Engine:
    """Equivalent of getNunjucksEngine(filters, throwOnUndefined, isGrader)."""
    key = (id(filters) if filters else 0, throw_on_undefined, is_grader)
    engine = _engines.get(key)
    if engine is None:
        env = NunjucksCompatEnvironment(throw_on_undefined)
        if filters:
            env.filters.update(filters)
        engine = _engines[key] = Engine(env, is_grader)
    return engine


def render_string(template: str, context: Mapping[str, Any] | None = None, **kw: Any) -> str:
    return get_engine(**kw).render_string(template, context)


def read_filters(filters: Mapping[str, str], base_path: str = "") -> dict[str, Callable[..., Any]]:
    """Load `nunjucksFilters` from Python files.

    Each value is `path.py` or `path.py:function`. Without a function name, the module must
    define a function named after the filter, or `main`. JS filter files are not supported.
    """
    result: dict[str, Callable[..., Any]] = {}
    for name, ref in filters.items():
        path, fn = python_hooks.split_path_and_function(ref)
        if not path.endswith(".py"):
            raise ValueError(
                f"nunjucksFilters.{name}: only Python filter files are supported, got {ref!r}"
            )
        for file_path in sorted(glob.glob(os.path.join(base_path, path))) or [
            os.path.join(base_path, path)
        ]:
            module = python_hooks.load_module(Path(file_path))
            candidates = [fn] if fn else [name, "main"]
            for candidate in candidates:
                func = getattr(module, candidate, None)
                if callable(func):
                    result[name] = func
                    break
            else:
                raise AttributeError(
                    f"nunjucksFilters.{name}: {file_path} defines none of {', '.join(candidates)}"
                )
    return result


# --- variable analysis ------------------------------------------------------------------------

_VAR_RE = re.compile(
    r"\{\{[\s]*([^{}\s|]+)[\s]*(?:\|[^}]+)?\}\}|\{%[\s]*(?:if|for)[\s]+([^{}\s]+)[\s]*.*?%\}"
)
_COMMENT_RE = re.compile(r"\{#[\s\S]*?#\}")
_FOR_RE = re.compile(r"\{%[\s]*for[\s]+(\w+)[\s]+in[\s]+(\w+)[\s]*%\}")


def extract_variables_from_template(template: str) -> list[str]:
    """Regex-based variable extraction, identical to the Node implementation."""
    template = _COMMENT_RE.sub("", template)
    found: dict[str, None] = {}
    for m in _VAR_RE.finditer(template):
        var = m.group(1) or m.group(2)
        if var:
            found[var] = None
    for m in _FOR_RE.finditer(template):
        found.pop(m.group(1), None)
        found[m.group(2)] = None
    return list(found)


def extract_variables_from_templates(templates: list[str]) -> list[str]:
    found: dict[str, None] = {}
    for t in templates:
        for v in extract_variables_from_template(t):
            found[v] = None
    return list(found)


def template_references_variable(template: str, variable_name: str) -> bool:
    """True when `variable_name` is a free symbol of the template.

    Falls back to a substring check when the template does not parse, which keeps callers that
    gate serial-only behaviour on this result conservative.
    """
    if not variable_name or variable_name not in template:
        return False
    try:
        env = get_engine().env
        return variable_name in meta.find_undeclared_variables(env.parse(template))
    except jinja2.TemplateSyntaxError:
        return True


# --- object rendering (render.ts) -------------------------------------------------------------


def _replace_templates(value: str, replace: Callable[[str], str]) -> str:
    out: list[str] = []
    pos = 0
    while True:
        start = value.find("{{", pos)
        end = -1 if start == -1 else value.find("}}", start + 2)
        if end == -1:
            return value if pos == 0 else "".join(out) + value[pos:]
        out.append(value[pos:start])
        out.append(replace(value[start : end + 2]))
        pos = end + 2


_ENV_REF = re.compile(r"\benv\.|env\[")
_ENV_NAME = re.compile(r"env\.(\w+)|env\[['\"]([^'\"]+)['\"]\]")


def render_env_only_in_object(
    obj: Any, env_overrides: Mapping[str, Any] | None = None, replace_base: bool = False
) -> Any:
    """Render only `{{ env.* }}` templates, leaving runtime templates like `{{ vars.x }}` intact."""
    if envars.get_env_bool("PROMPTFOO_DISABLE_TEMPLATING"):
        return obj

    env_globals: Mapping[str, Any] | None = None

    def globals_() -> Mapping[str, Any]:
        nonlocal env_globals
        if env_globals is None:
            if replace_base:
                env_globals = dict(env_overrides or {})
            elif env_overrides:
                env_globals = {**TEMPLATE_ENV, **env_overrides}
            else:
                env_globals = TEMPLATE_ENV
        return env_globals

    def replace(match: str) -> str:
        if not _ENV_REF.search(match):
            return match
        name_match = _ENV_NAME.search(match)
        name = name_match and (name_match.group(1) or name_match.group(2))
        g = globals_()
        if "|" in match or (name and g.get(name) is not None):
            try:
                return get_engine().render_string(match, {"env": g})
            except jinja2.TemplateError as e:
                logger.debug('Failed to render env template "%s": %s', match, e)
                return match
        return match

    def render(value: Any) -> Any:
        if isinstance(value, str):
            return _replace_templates(value, replace)
        if isinstance(value, list):
            return [render(v) for v in value]
        if isinstance(value, dict):
            # Conversation history is runtime data and may contain untrusted model output.
            return {k: v if k == "_conversation" else render(v) for k, v in value.items()}
        return value

    return render(obj)


def render_vars_in_object(obj: Any, vars: Mapping[str, Any] | None = None) -> Any:
    """Render every string in `obj` as a template with `vars` as context."""
    if not vars or envars.get_env_bool("PROMPTFOO_DISABLE_TEMPLATING"):
        return obj
    if isinstance(obj, str):
        return get_engine().render_string(obj, vars)
    if isinstance(obj, list):
        return [render_vars_in_object(v, vars) for v in obj]
    if isinstance(obj, dict):
        return {k: render_vars_in_object(v, vars) for k, v in obj.items()}
    if callable(obj):
        return render_vars_in_object(obj({"vars": vars}))
    return obj
