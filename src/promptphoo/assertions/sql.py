"""SQL assertions. Ports src/assertions/sql.ts (is-sql, contains-sql).

Node validates with node-sql-parser (`astify`, `whiteListCheck`, `parse().tableList` /
`columnList`). This port uses sqlglot, the optional `sql` extra, and keeps everything else
as upstream has it: the extra heuristics (empty input, unbalanced backticks, the likely
missing comma, generate_series under MySQL), the code-fence extraction for contains-sql,
the `allowedTables`/`allowedColumns` whitelist semantics (each entry is a case-insensitive
JS regex anchored with ^...$, and column entries get a `type::null::column` twin), and
every reason string.

Dialect mapping (node-sql-parser `database`, case-insensitive, to sqlglot `read`):
MySQL->mysql, MariaDB->mysql, PostgreSQL->postgres, Noql->postgres, TransactSQL->tsql,
Sqlite->sqlite, BigQuery->bigquery, Snowflake->snowflake, Hive->hive, Redshift->redshift,
Trino->trino, Athena->athena, DB2 and FlinkSQL->sqlglot's generic dialect. Other names fail
the way an unsupported database fails in node-sql-parser.

Known divergences from node-sql-parser:
- sqlglot accepts far more than node-sql-parser's PEG grammars, so `_astify` tightens it.
  Only statements are accepted (SELECT/set operations, INSERT, UPDATE, DELETE,
  CREATE/DROP/ALTER/TRUNCATE, USE, SET, SHOW, MERGE, EXPLAIN/DESCRIBE, GRANT/REVOKE,
  transactions, and CALL/LOCK/UNLOCK/RENAME/REPLACE fallbacks). Bare expressions like
  `not sql` or `foo` are rejected, and so is a SELECT with no select list (this covers
  `SELECT * FROM select`). DELETE without FROM is rejected outside TransactSQL.
  ILIKE is rejected for MySQL/MariaDB/TransactSQL/Sqlite, and `LOCK IN SHARE MODE`
  outside MySQL/MariaDB. Beyond those, dialect-specific syntax that one parser accepts
  and the other rejects follows sqlglot.
- When the whitelist parse itself fails, the reason reads `SQL validation failed: <msg>.`
  with sqlglot's message (ANSI codes stripped), not node-sql-parser's
  `Expected ... but "x" found.` text.
- tableList/columnList are rebuilt from the sqlglot AST in node-sql-parser's
  `type::db::table` / `type::table::column` format: statement targets are typed
  insert/update/delete/create/drop/alter, every other table reference is select, and
  column references are typed select with their written qualifier (null when unqualified).
  INSERT columns become insert::<table>::<col> (insert::<table>::(.*) without a column
  list), UPDATE SET targets become update::<table>::<col>, DELETE adds
  delete::<table>::(.*), and `*` becomes select::null::(.*). Lists keep first-seen
  order. CTE names are listed like any other table reference.
- A missing sqlglot raises "sqlglot is not installed. Please install it first" where Node
  names node-sql-parser.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from typing import Any

from promptphoo.assertions.base import (
    AssertionParams,
    AssertionValueError,
    GradingResult,
    Handler,
    coerce_string,
)
from promptphoo.jsregex import compile_js_regex
from promptphoo.jsvalues import is_truthy, js_string

_STANDARD_SELECT_MODIFIER_PATTERN = "(?:all|distinct)"
_MYSQL_SELECT_MODIFIER_PATTERN = (
    "(?:all|distinct(?:row)?|high_priority|straight_join|"
    "sql_(?:big_result|buffer_result|cache|calc_found_rows|no_cache|small_result))"
)
_IDENTIFIER_PATTERN = "[A-Za-z_][A-Za-z0-9_]*"
_JS_TRIM = (
    "\t\n\v\f\r \u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008"
    "\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
)
_JS_S = r"[\t\n\v\f\r \u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff]"
# JS \b is ASCII-word based.
_JS_B = r"(?:(?<![A-Za-z0-9_])(?=[A-Za-z0-9_])|(?<=[A-Za-z0-9_])(?![A-Za-z0-9_]))"


def _create_likely_missing_comma_pattern(modifier: str) -> re.Pattern[str]:
    return re.compile(
        rf"{_JS_B}select{_JS_S}+(?:{modifier}{_JS_S}+)*(?!{modifier}{_JS_B})"
        rf"{_IDENTIFIER_PATTERN}{_JS_S}+{_IDENTIFIER_PATTERN}{_JS_S}+from{_JS_B}",
        re.IGNORECASE | re.ASCII,
    )


_STANDARD_LIKELY_MISSING_COMMA_PATTERN = _create_likely_missing_comma_pattern(
    _STANDARD_SELECT_MODIFIER_PATTERN
)
_MYSQL_LIKELY_MISSING_COMMA_PATTERN = _create_likely_missing_comma_pattern(
    _MYSQL_SELECT_MODIFIER_PATTERN
)
_MYSQL_FAMILY_DATABASES = {"MySQL", "MariaDB"}
_BRACKET_IDENTIFIER_DATABASES = {"TransactSQL", "Sqlite"}
_SQL_EXPRESSION_PLACEHOLDER = " ? "
_DOLLAR_QUOTE_DELIMITER_PATTERN = re.compile(r"\$(?:[A-Za-z_][A-Za-z0-9_]*)?\$")
_GENERATE_SERIES_PATTERN = re.compile(rf"{_JS_B}generate_series{_JS_S}*\(", re.IGNORECASE)


def _get_likely_missing_comma_pattern(database_type: str) -> re.Pattern[str]:
    if database_type in _MYSQL_FAMILY_DATABASES:
        return _MYSQL_LIKELY_MISSING_COMMA_PATTERN
    return _STANDARD_LIKELY_MISSING_COMMA_PATTERN


def _read_dollar_quote_delimiter(sql: str, start: int) -> str | None:
    m = _DOLLAR_QUOTE_DELIMITER_PATTERN.match(sql, start)
    return m.group(0) if m else None


def _find_quoted_text_end(sql: str, start: int, quote: str) -> int:
    closing = "]" if quote == "[" else quote
    cursor = start + 1
    n = len(sql)
    while cursor < n:
        if closing != "]" and sql[cursor] == "\\":
            cursor += 2
            continue
        if sql[cursor] == closing:
            if cursor + 1 < n and sql[cursor + 1] == closing:
                cursor += 2
                continue
            return cursor + 1
        cursor += 1
    return n


def _strip_ignored_sql_text(sql: str, database_type: str) -> str:
    chunks: list[str] = []
    brackets = database_type in _BRACKET_IDENTIFIER_DATABASES
    plain_start = 0
    cursor = 0
    n = len(sql)
    while cursor < n:
        ch = sql[cursor]
        ignored_end: int | None = None
        replacement = " "
        if ch in ("'", '"', "`") or (ch == "[" and brackets):
            ignored_end = _find_quoted_text_end(sql, cursor, ch)
            replacement = _SQL_EXPRESSION_PLACEHOLDER
        elif (ch == "-" and cursor + 1 < n and sql[cursor + 1] == "-") or ch == "#":
            ignored_end = cursor + 1
            while ignored_end < n and sql[ignored_end] not in "\r\n":
                ignored_end += 1
        elif ch == "/" and cursor + 1 < n and sql[cursor + 1] == "*":
            comment_end = sql.find("*/", cursor + 2)
            ignored_end = n if comment_end == -1 else comment_end + 2
        elif ch == "$":
            delimiter = _read_dollar_quote_delimiter(sql, cursor)
            if delimiter:
                quote_end = sql.find(delimiter, cursor + len(delimiter))
                ignored_end = n if quote_end == -1 else quote_end + len(delimiter)
                replacement = _SQL_EXPRESSION_PLACEHOLDER
        if ignored_end is None:
            cursor += 1
            continue
        chunks.append(sql[plain_start:cursor])
        chunks.append(replacement)
        cursor = ignored_end
        plain_start = cursor
    chunks.append(sql[plain_start:])
    return "".join(chunks)


# ---------------------------------------------------------------------------------------------
# node-sql-parser stand-in
# ---------------------------------------------------------------------------------------------

_DIALECTS: dict[str, str] = {
    "mysql": "mysql",
    "mariadb": "mysql",
    "postgresql": "postgres",
    "noql": "postgres",
    "transactsql": "tsql",
    "sqlite": "sqlite",
    "bigquery": "bigquery",
    "snowflake": "snowflake",
    "hive": "hive",
    "redshift": "redshift",
    "trino": "trino",
    "athena": "athena",
    "db2": "",
    "flinksql": "",
}
_NO_ILIKE = {"mysql", "mariadb", "transactsql", "sqlite"}
_COMMAND_KEYWORDS = {"CALL", "LOCK", "UNLOCK", "RENAME", "REPLACE"}
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_LOCK_IN_SHARE_MODE = re.compile(
    rf"{_JS_B}lock{_JS_S}+in{_JS_S}+share{_JS_S}+mode{_JS_B}", re.IGNORECASE
)
_DELETE_WITHOUT_FROM = re.compile(
    rf"(?:^|;){_JS_S}*delete{_JS_B}(?![^;]*{_JS_B}from{_JS_B})", re.IGNORECASE
)


class SqlParseError(Exception):
    pass


class _Parser:
    """The node-sql-parser surface sql.ts uses: astify, whiteListCheck, parse()."""

    def __init__(self) -> None:
        try:
            import sqlglot  # noqa: F401
        except ImportError as err:
            raise RuntimeError("sqlglot is not installed. Please install it first") from err

    def astify(self, sql: str, database: str) -> list[Any]:
        import sqlglot
        from sqlglot import exp
        from sqlglot.errors import SqlglotError

        key = database.lower()
        if key not in _DIALECTS:
            raise SqlParseError(f"{database} is not supported currently")
        logger = logging.getLogger("sqlglot")
        previous = logger.level
        logger.setLevel(logging.CRITICAL)
        try:
            statements = [s for s in sqlglot.parse(sql, read=_DIALECTS[key] or None) if s]
        except SqlglotError as err:
            raise SqlParseError(_ANSI_RE.sub("", str(err))) from err
        except (RecursionError, ValueError) as err:
            raise SqlParseError(str(err)) from err
        finally:
            logger.setLevel(previous)
        if not statements:
            if sql.strip(_JS_TRIM):
                raise SqlParseError("Expected a SQL statement")
            return []  # node-sql-parser accepts empty input; is-sql's own check rejects it

        statement_types: tuple[type, ...] = (
            exp.Query,
            exp.Insert,
            exp.Update,
            exp.Delete,
            exp.Create,
            exp.Drop,
            exp.Alter,
            exp.TruncateTable,
            exp.Use,
            exp.Set,
            exp.Show,
            exp.Merge,
            exp.Describe,
            exp.Grant,
            exp.Revoke,
            exp.Transaction,
            exp.Commit,
            exp.Rollback,
        )
        for stmt in statements:
            if isinstance(stmt, exp.Command):
                word = str(stmt.this or "").split()[:1]
                if not word or word[0].upper() not in _COMMAND_KEYWORDS:
                    raise SqlParseError(f"Unsupported statement: {stmt.this}")
                continue
            if not isinstance(stmt, statement_types):
                raise SqlParseError(f"Expected a SQL statement but found {type(stmt).__name__}")
            for select in stmt.find_all(exp.Select):
                if not select.expressions:
                    raise SqlParseError("Expected a select list")
            if key in _NO_ILIKE and stmt.find(exp.ILike) is not None:
                raise SqlParseError("ILIKE is not supported")

        stripped = _strip_ignored_sql_text(sql, "")
        if key not in ("mysql", "mariadb") and _LOCK_IN_SHARE_MODE.search(stripped):
            raise SqlParseError("LOCK IN SHARE MODE is not supported")
        if key != "transactsql" and _DELETE_WITHOUT_FROM.search(stripped):
            raise SqlParseError("DELETE requires FROM")
        return statements

    def parse(self, sql: str, database: str) -> dict[str, list[str]]:
        statements = self.astify(sql, database)
        tables: dict[str, None] = {}
        columns: dict[str, None] = {}
        for stmt in statements:
            _collect_lists(stmt, tables, columns)
        return {"tableList": list(tables), "columnList": list(columns)}

    def white_list_check(
        self, sql: str, white_list: list[str] | None, database: str, kind: str
    ) -> None:
        if not white_list:
            return
        lists = self.parse(sql, database)
        authority_list = lists["tableList" if kind == "table" else "columnList"]
        for authority in authority_list:
            if not any(
                compile_js_regex(f"^{entry}$", "i").search(authority) for entry in white_list
            ):
                raise SqlParseError(
                    f"authority = '{authority}' is required in {kind} whiteList to execute "
                    f"SQL = '{sql}'"
                )


def _table_name(table: Any) -> tuple[str, str]:
    db = table.args.get("db")
    return (db.name if db is not None else "null"), table.name


def _collect_lists(stmt: Any, tables: dict[str, None], columns: dict[str, None]) -> None:
    from sqlglot import exp

    targets: dict[int, str] = {}
    target_name: str | None = None
    skip_columns: set[int] = set()

    def target_table(node: Any) -> Any:
        if isinstance(node, exp.Schema):
            return node.this
        return node

    if isinstance(stmt, exp.Insert):
        tbl = target_table(stmt.this)
        if isinstance(tbl, exp.Table):
            targets[id(tbl)] = "insert"
            name = tbl.name
            cols = stmt.this.expressions if isinstance(stmt.this, exp.Schema) else []
            if cols:
                for col in cols:
                    columns[f"insert::{name}::{col.name}"] = None
            else:
                columns[f"insert::{name}::(.*)"] = None
    elif isinstance(stmt, exp.Update):
        tbl = stmt.this
        if isinstance(tbl, exp.Table):
            targets[id(tbl)] = "update"
            target_name = tbl.name
    elif isinstance(stmt, exp.Delete):
        tbl = stmt.this
        if isinstance(tbl, exp.Table):
            targets[id(tbl)] = "delete"
            columns[f"delete::{tbl.name}::(.*)"] = None
    elif isinstance(stmt, (exp.Create, exp.Drop, exp.Alter, exp.TruncateTable)):
        kind = {
            exp.Create: "create",
            exp.Drop: "drop",
            exp.Alter: "alter",
            exp.TruncateTable: "truncate",
        }[type(stmt)]
        for tbl in [
            target_table(stmt.this),
            *(stmt.args.get("expressions") or []),
            *(stmt.args.get("tables") or []),
        ]:
            if isinstance(tbl, exp.Table):
                targets[id(tbl)] = kind

    if isinstance(stmt, exp.Update):
        for assignment in stmt.expressions:
            lhs = assignment.this if isinstance(assignment, exp.EQ) else None
            if isinstance(lhs, exp.Column):
                skip_columns.add(id(lhs))
                columns[f"update::{target_name}::{lhs.name}"] = None

    for node in stmt.walk(bfs=False):
        if isinstance(node, exp.Table):
            db, name = _table_name(node)
            if name:
                tables[f"{targets.get(id(node), 'select')}::{db}::{name}"] = None
        elif isinstance(node, exp.Column) and id(node) not in skip_columns:
            if isinstance(node.this, exp.Star):
                columns[f"select::{node.table or 'null'}::(.*)"] = None
            else:
                columns[f"select::{node.table or 'null'}::{node.name}"] = None
        elif isinstance(node, exp.Star) and isinstance(node.parent, exp.Select):
            columns["select::null::(.*)"] = None


def _create_sql_parser() -> _Parser:
    return _Parser()


def handle_is_sql(params: AssertionParams) -> GradingResult:
    assertion = params.assertion
    rendered = params.rendered_value
    output_string = params.output_string
    inverse = params.inverse

    passed = False
    database_type = "MySQL"
    allowed_tables: list[str] | None = None
    allowed_columns: list[str] | None = None
    is_object = isinstance(rendered, (dict, list))
    if is_truthy(rendered) and is_object:
        value = rendered if isinstance(rendered, dict) else {}
        db = value.get("databaseType")
        database_type = js_string(db) if is_truthy(db) else "MySQL"
        allowed_tables = value.get("allowedTables")
        allowed_columns = value.get("allowedColumns")

    if is_truthy(rendered) and not is_object:
        raise AssertionValueError("is-sql assertion must have a object value.")

    sql_parser = _create_sql_parser()
    syntax_reason = (
        f"SQL statement does not conform to the provided {database_type} database syntax."
    )
    failure_reasons: list[str] = []

    normalized_sql = output_string.strip(_JS_TRIM)
    if len(normalized_sql) == 0:
        failure_reasons.append(syntax_reason)
    if normalized_sql.count("`") % 2 != 0:
        failure_reasons.append(syntax_reason)
    sql_for_heuristics = _strip_ignored_sql_text(normalized_sql, database_type)
    if _get_likely_missing_comma_pattern(database_type).search(sql_for_heuristics):
        failure_reasons.append(syntax_reason)
    if database_type == "MySQL" and _GENERATE_SERIES_PATTERN.search(normalized_sql):
        failure_reasons.append(syntax_reason)

    try:
        sql_parser.astify(output_string, database_type)
        passed = not inverse
    except SqlParseError:
        passed = inverse
        failure_reasons.append(syntax_reason)

    if failure_reasons:
        passed = inverse

    if allowed_tables:
        try:
            sql_parser.white_list_check(output_string, allowed_tables, database_type, "table")
        except SqlParseError as err:
            passed = inverse
            actual_tables: list[str] = []
            try:
                actual_tables = sql_parser.parse(output_string, database_type)["tableList"]
            except SqlParseError:
                pass
            if actual_tables:
                failure_reasons.append(
                    "SQL references unauthorized table(s). "
                    f"Found: [{', '.join(actual_tables)}]. "
                    f"Allowed: [{', '.join(str(t) for t in allowed_tables)}]."
                )
            else:
                failure_reasons.append(f"SQL validation failed: {err}.")

    if allowed_columns:
        normalized_allowed = list(allowed_columns)
        for item in allowed_columns:
            parts = str(item).split("::")
            if len(parts) == 3 and parts[1] != "null":
                alt = f"{parts[0]}::null::{parts[2]}"
                if alt not in normalized_allowed:
                    normalized_allowed.append(alt)
        try:
            sql_parser.white_list_check(output_string, normalized_allowed, database_type, "column")
        except SqlParseError as err:
            passed = inverse
            actual_columns: list[str] = []
            try:
                actual_columns = sql_parser.parse(output_string, database_type)["columnList"]
            except SqlParseError:
                pass
            if actual_columns:
                failure_reasons.append(
                    "SQL references unauthorized column(s). "
                    f"Found: [{', '.join(actual_columns)}]. "
                    f"Allowed: [{', '.join(str(c) for c in allowed_columns)}]."
                )
            else:
                failure_reasons.append(f"SQL validation failed: {err}.")

    if inverse and passed is False and not failure_reasons:
        failure_reasons.append("The output SQL statement is valid")

    return {
        "pass": passed,
        "score": 1 if passed else 0,
        "reason": "Assertion passed" if passed else " ".join(failure_reasons),
        "assertion": assertion,
    }


def _parse_code_fence(line: str) -> dict[str, Any] | None:
    index = 0
    while index < 3 and index < len(line) and line[index] in " \t":
        index += 1
    character = line[index] if index < len(line) else ""
    if character not in ("`", "~"):
        return None
    fence_start = index
    while index < len(line) and line[index] == character:
        index += 1
    length = index - fence_start
    if length < 3:
        return None
    info = line[index:]
    return {
        "character": character,
        "length": length,
        "info": info,
        "hasOnlyTrailingWhitespace": info.strip(" \t") == "",
    }


def _extract_sql_code_blocks(output: str) -> list[str]:
    sql_blocks: list[str] = []
    active: dict[str, Any] | None = None
    for line in re.split(r"\r\n?|\n", output):
        fence = _parse_code_fence(line)
        if active is None:
            if fence:
                language = fence["info"].strip(_JS_TRIM).lower()
                active = {
                    "body": [],
                    "fenceCharacter": fence["character"],
                    "fenceLength": fence["length"],
                    "isSql": language in ("", "sql"),
                }
            continue
        if (
            fence
            and fence["hasOnlyTrailingWhitespace"]
            and fence["character"] == active["fenceCharacter"]
            and fence["length"] >= active["fenceLength"]
        ):
            if active["isSql"]:
                sql_blocks.append("\n".join(active["body"]).strip(_JS_TRIM))
            active = None
        else:
            active["body"].append(line)
    return sql_blocks


def handle_contains_sql(params: AssertionParams) -> GradingResult:
    output_string = coerce_string(params.output_string)
    sql_blocks = _extract_sql_code_blocks(output_string)
    candidates = sql_blocks if sql_blocks else [output_string]
    failure_reason = "Output does not contain valid SQL"
    contains_valid_sql = False

    for candidate in candidates:
        if len(candidate.strip(_JS_TRIM)) == 0:
            continue
        res = handle_is_sql(dataclasses.replace(params, output_string=candidate, inverse=False))
        if res["pass"]:
            contains_valid_sql = True
            break
        failure_reason = res["reason"]

    passed = not contains_valid_sql if params.inverse else contains_valid_sql
    return {
        "assertion": params.assertion,
        "pass": passed,
        "score": 1 if passed else 0,
        "reason": "Assertion passed"
        if passed
        else ("The output SQL statement is valid" if contains_valid_sql else failure_reason),
    }


HANDLERS: dict[str, Handler] = {
    "is-sql": handle_is_sql,
    "contains-sql": handle_contains_sql,
}
