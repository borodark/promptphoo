"""Ports test/assertions/sql.test.ts."""

from __future__ import annotations

from typing import Any

import pytest

from promptphoo.assertions.base import AssertionParams, AssertionValueError
from promptphoo.assertions.sql import HANDLERS, handle_contains_sql, handle_is_sql

pytest.importorskip("sqlglot")

ASSERTION = {"type": "is-sql"}
MYSQL_REASON = "SQL statement does not conform to the provided MySQL database syntax."


def is_sql(output: str, rendered: Any = None, inverse: bool = False) -> dict[str, Any]:
    return handle_is_sql(
        AssertionParams(
            assertion=ASSERTION,
            base_type="is-sql",
            inverse=inverse,
            output=output,
            output_string=output,
            rendered_value=rendered,
        )
    )


def contains_sql(output: str, inverse: bool = False) -> dict[str, Any]:
    assertion = {"type": "not-contains-sql" if inverse else "contains-sql"}
    return handle_contains_sql(
        AssertionParams(
            assertion=assertion,
            base_type="contains-sql",
            inverse=inverse,
            output=output,
            output_string=output,
        )
    )


def test_handlers_exported() -> None:
    assert set(HANDLERS) == {"is-sql", "contains-sql"}


# ------------------------------------------------ Basic tests ---------------------------------


def test_valid_sql_statement() -> None:
    assert is_sql("SELECT id, name FROM users") == {
        "assertion": ASSERTION,
        "pass": True,
        "reason": "Assertion passed",
        "score": 1,
    }


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT DISTINCT name FROM users",
        "select distinct id from users",
        "SELECT SQL_NO_CACHE name FROM users",
        "SELECT DISTINCT SQL_NO_CACHE name FROM users",
    ],
)
def test_valid_select_modifiers(sql: str) -> None:
    assert is_sql(sql) == {
        "assertion": ASSERTION,
        "pass": True,
        "reason": "Assertion passed",
        "score": 1,
    }


def test_quoted_column_after_distinct() -> None:
    assert is_sql('SELECT DISTINCT "display name" FROM users', {"databaseType": "PostgreSQL"})[
        "pass"
    ]


@pytest.mark.parametrize(
    ("sql", "rendered"),
    [
        ("SELECT 'select distinct first_name last_name from employees' AS sample", None),
        ("SELECT 'it''s select distinct first_name last_name from employees' AS sample", None),
        ("SELECT 'it\\'s select distinct first_name last_name from employees' AS sample", None),
        ("SELECT 1 /* select distinct first_name last_name from employees */", None),
        ("SELECT 1 -- select distinct first_name last_name from employees\n", None),
        ("SELECT 1 # select distinct first_name last_name from employees\n", None),
        ("SELECT `select distinct first_name last_name from employees` AS sample", None),
        (
            "SELECT [select distinct first_name last_name from employees] AS sample",
            {"databaseType": "TransactSQL"},
        ),
        (
            'SELECT "select distinct first_name last_name from employees" AS sample',
            {"databaseType": "PostgreSQL"},
        ),
        (
            "SELECT $$select distinct first_name last_name from employees$$ AS sample",
            {"databaseType": "PostgreSQL"},
        ),
    ],
)
def test_ignores_sql_like_text_in_literals_and_comments(sql: str, rendered: Any) -> None:
    res = is_sql(sql, rendered)
    assert (res["pass"], res["score"]) == (True, 1)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT a b FROM t",
        "SELECT DISTINCT first_name last_name FROM employees",
        "SELECT SQL_NO_CACHE first_name last_name FROM employees",
        "SELECT first_name /* separator */ last_name FROM employees",
        "SELECT DISTINCTIVE name FROM users",
        "",
        "   ",
    ],
)
def test_invalid_sql_reports_mysql_syntax(sql: str) -> None:
    res = is_sql(sql)
    assert (res["pass"], res["score"]) == (False, 0)
    assert res["reason"] == MYSQL_REASON


@pytest.mark.parametrize("db", ["PostgreSQL", "TransactSQL", "BigQuery"])
def test_mysql_only_modifiers_are_not_modifiers_elsewhere(db: str) -> None:
    res = is_sql("SELECT SQL_NO_CACHE name FROM users", {"databaseType": db})
    assert (res["pass"], res["score"]) == (False, 0)


def test_mariadb_keeps_mysql_modifiers() -> None:
    res = is_sql("SELECT SQL_NO_CACHE name FROM users", {"databaseType": "MariaDB"})
    assert (res["pass"], res["score"]) == (True, 1)


@pytest.mark.parametrize("db", ["PostgreSQL", "BigQuery"])
def test_square_bracket_subscripts(db: str) -> None:
    res = is_sql("SELECT arr[1] value FROM t", {"databaseType": db})
    assert (res["pass"], res["score"]) == (True, 1)


@pytest.mark.parametrize(
    ("db", "sql"),
    [
        ("BigQuery", "SELECT r'hello' value FROM t"),
        ("TransactSQL", "SELECT N'hello' value FROM t"),
        ("Sqlite", "SELECT X'53514C697465' value FROM t"),
    ],
)
def test_prefixed_literals(db: str, sql: str) -> None:
    res = is_sql(sql, {"databaseType": db})
    assert (res["pass"], res["score"]) == (True, 1)


def test_many_unmatched_dollar_quote_tags() -> None:
    sql = "SELECT " + " ".join(f"$tag{i}$ value" for i in range(20_000))
    res = is_sql(sql, {"databaseType": "PostgreSQL"})
    assert (res["pass"], res["score"]) == (False, 0)


def test_contains_sql_fenced_response() -> None:
    res = contains_sql("Result:\n```sql\nSELECT DISTINCT name FROM users\n```")
    assert res == {
        "assertion": {"type": "contains-sql"},
        "pass": True,
        "reason": "Assertion passed",
        "score": 1,
    }


def test_contains_sql_missing_comma_in_fence() -> None:
    res = contains_sql(
        "Here you go:\n```sql\nSELECT DISTINCT first_name last_name FROM employees\n```"
    )
    assert (res["pass"], res["score"], res["reason"]) == (False, 0, MYSQL_REASON)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM orders ORDERY BY order_date",
        "SELECT * FROM select WHERE id = 1",
        "DELETE employees WHERE id = 1",
        "SELECT first_name last_name FROM employees",
    ],
)
def test_invalid_statements_exact(sql: str) -> None:
    assert is_sql(sql) == {
        "pass": False,
        "score": 0,
        "reason": MYSQL_REASON,
        "assertion": ASSERTION,
    }


def test_mismatched_backticks() -> None:
    res = is_sql("SELECT * FROM `employees")
    assert (res["pass"], res["score"], res["assertion"]) == (False, 0, ASSERTION)
    assert MYSQL_REASON in res["reason"]


def test_matched_backticks() -> None:
    assert is_sql("SELECT * FROM `employees` WHERE `id` = 1") == {
        "pass": True,
        "score": 1,
        "reason": "Assertion passed",
        "assertion": ASSERTION,
    }


@pytest.mark.parametrize(
    "sql", ["SELECT `id, `name FROM `users", "SELECT col1 col2 col3 FROM table1"]
)
def test_other_invalid(sql: str) -> None:
    res = is_sql(sql)
    assert (res["pass"], res["score"]) == (False, 0)


# --------------------------------------- Database specific syntax -----------------------------


def test_mysql_only_syntax_fails_postgres() -> None:
    res = is_sql(
        "SELECT * FROM employees WHERE id = 1 LOCK IN SHARE MODE", {"databaseType": "PostgreSQL"}
    )
    assert res == {
        "pass": False,
        "score": 0,
        "reason": "SQL statement does not conform to the provided PostgreSQL database syntax.",
        "assertion": ASSERTION,
    }


ILIKE = "SELECT first_name, last_name FROM employees WHERE first_name ILIKE 'john%'"


def test_postgres_only_syntax_fails_mysql() -> None:
    assert is_sql(ILIKE, {"databaseType": "MySQL"}) == {
        "pass": False,
        "score": 0,
        "reason": MYSQL_REASON,
        "assertion": ASSERTION,
    }


def test_postgres_only_syntax_passes_postgres() -> None:
    assert is_sql(ILIKE, {"databaseType": "PostgreSQL"}) == {
        "pass": True,
        "score": 1,
        "reason": "Assertion passed",
        "assertion": ASSERTION,
    }


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT generate_series(1, 10);",
        "SELECT * FROM table_name WHERE id IN (SELECT generate_series(1, 5));",
    ],
)
def test_generate_series_fails_mysql(sql: str) -> None:
    assert is_sql(sql, {"databaseType": "MySQL"}) == {
        "pass": False,
        "score": 0,
        "reason": MYSQL_REASON,
        "assertion": ASSERTION,
    }


# ------------------------------------------- Allowed tables/columns ---------------------------


def test_violates_allowed_tables() -> None:
    allowed = "(select|update|insert|delete)::null::departments"
    res = is_sql("SELECT * FROM employees", {"databaseType": "MySQL", "allowedTables": [allowed]})
    assert res == {
        "pass": False,
        "score": 0,
        "reason": "SQL references unauthorized table(s). Found: [select::null::employees]. "
        "Allowed: [(select|update|insert|delete)::null::departments].",
        "assertion": ASSERTION,
    }


def test_respects_allowed_tables() -> None:
    allowed = "(select|update|insert|delete)::null::departments"
    res = is_sql("SELECT * FROM departments", {"databaseType": "MySQL", "allowedTables": [allowed]})
    assert res == {"pass": True, "score": 1, "reason": "Assertion passed", "assertion": ASSERTION}


def test_violates_allowed_columns() -> None:
    res = is_sql(
        "SELECT id FROM t",
        {"databaseType": "MySQL", "allowedColumns": ["select::null::name", "update::null::id"]},
    )
    assert res == {
        "pass": False,
        "score": 0,
        "reason": "SQL references unauthorized column(s). Found: [select::null::id]. "
        "Allowed: [select::null::name, update::null::id].",
        "assertion": ASSERTION,
    }


@pytest.mark.parametrize(
    ("sql", "allowed"),
    [
        (
            "INSERT INTO department (dept_name, location) VALUES ('Sales', 'New York')",
            ["insert::department::dept_name", "insert::department::location"],
        ),
        ("UPDATE a SET id = 1", ["update::a::id"]),
        (
            "UPDATE employee SET salary = 50000 WHERE id = 1",
            ["update::employee::salary", "select::employee::id"],
        ),
        ("SELECT name FROM employees", ["select::employees::name"]),
    ],
)
def test_respects_allowed_columns(sql: str, allowed: list[str]) -> None:
    res = is_sql(sql, {"databaseType": "MySQL", "allowedColumns": allowed})
    assert res == {"pass": True, "score": 1, "reason": "Assertion passed", "assertion": ASSERTION}


def test_wrong_table_name_issue_1491() -> None:
    res = is_sql(
        "SELECT * FROM data WHERE id = 1",
        {"databaseType": "MySQL", "allowedTables": ["select::null::data_table"]},
    )
    assert res["pass"] is False
    assert "SQL references unauthorized table(s)" in res["reason"]
    assert "select::null::data" in res["reason"]
    assert "select::null::data_table" in res["reason"]


def test_correct_table_name() -> None:
    res = is_sql(
        "SELECT * FROM data_table WHERE id = 1",
        {"databaseType": "MySQL", "allowedTables": ["select::null::data_table"]},
    )
    assert res == {"pass": True, "score": 1, "reason": "Assertion passed", "assertion": ASSERTION}


def test_double_quoted_table_postgres() -> None:
    res = is_sql(
        'SELECT * FROM "data_table" WHERE id = 1',
        {"databaseType": "PostgreSQL", "allowedTables": ["select::null::data_table"]},
    )
    assert res == {"pass": True, "score": 1, "reason": "Assertion passed", "assertion": ASSERTION}


def test_double_quoted_table_differs() -> None:
    res = is_sql(
        'SELECT * FROM "data" WHERE id = 1',
        {"databaseType": "PostgreSQL", "allowedTables": ["select::null::data_table"]},
    )
    assert res["pass"] is False
    assert "SQL references unauthorized table(s)" in res["reason"]
    assert "select::null::data" in res["reason"]


def test_multiple_tables_some_unauthorized() -> None:
    res = is_sql(
        "SELECT u.name, p.title FROM users u, posts p WHERE u.id = p.user_id",
        {"databaseType": "MySQL", "allowedTables": ["select::null::users"]},
    )
    assert res["pass"] is False
    assert "SQL references unauthorized table(s)" in res["reason"]
    assert "select::null::posts" in res["reason"]


# ---------------------------------------- Extra: value handling --------------------------------


def test_non_object_value_raises() -> None:
    with pytest.raises(AssertionValueError, match="is-sql assertion must have a object value."):
        is_sql("SELECT 1", "MySQL")


def test_inverse_valid_sql() -> None:
    res = is_sql("SELECT 1", inverse=True)
    assert (res["pass"], res["reason"]) == (False, "The output SQL statement is valid")


def test_inverse_invalid_sql() -> None:
    res = is_sql("SELECT a b FROM t", inverse=True)
    assert (res["pass"], res["reason"]) == (True, "Assertion passed")


def test_empty_allowed_lists_skip_the_check() -> None:
    res = is_sql("SELECT * FROM t", {"allowedTables": [], "allowedColumns": []})
    assert res["pass"] is True


# ------------------------------------------------ contains-sql --------------------------------

FENCE = "```"


@pytest.mark.parametrize(
    "output",
    [
        f"Here is the query:\n{FENCE}sql\nSELECT `id` FROM `users`;\n{FENCE}",
        f"{FENCE}sql\nSELECT id, name FROM users\n{FENCE}",
        f"{FENCE}sql\nSELECT `a``b` FROM `users`;\n{FENCE}",
        f"{FENCE}sql\nnot sql\n{FENCE}\n{FENCE}sql\nSELECT 1;\n{FENCE}",
        "````sql\nSELECT `id` FROM `users`;\n````",
        "~~~sql\nSELECT 1;\n~~~",
        "```sql\nSELECT 1;\n````",
        "```\nSELECT 1;\n```",
        "``` SQL \nSELECT 1;\n```",
        "   ```sql\nSELECT 1;\n   ```",
        "```\t\tsql\t\nSELECT 1;\n```\t\t",
        "```" + "\t" * 50_000 + "sql\nSELECT 1;\n```" + "\t" * 50_000,
    ],
)
def test_contains_sql_extracts(output: str) -> None:
    res = contains_sql(output)
    assert (res["pass"], res["score"]) == (True, 1)


def test_contains_sql_after_non_sql_block() -> None:
    output = f"{FENCE}python\nprint('hello')\n{FENCE}\n{FENCE}sql\nSELECT 1;\n{FENCE}"
    assert (contains_sql(output)["pass"], contains_sql(output)["score"]) == (True, 1)
    inv = contains_sql(output, inverse=True)
    assert (inv["pass"], inv["score"], inv["reason"]) == (
        False,
        0,
        "The output SQL statement is valid",
    )


@pytest.mark.parametrize(
    "output",
    [
        "````sql\nSELECT 1;\n~~~\n```\n````",
        "",
        " \n\t",
        f"{FENCE}sql\n{FENCE}",
        f"{FENCE}sql\n  \n{FENCE}",
        f"{FENCE}sql\nSELECT 1;",
        f"{FENCE}python\nprint('hello')\n{FENCE}",
        f"{FENCE}sqlSELECT\nSELECT 1;\n{FENCE}",
        f"    {FENCE}sql\nSELECT 1;\n    {FENCE}",
        f"{FENCE}sql\nSELECT 1;\n{FENCE}sql",
    ],
)
def test_contains_sql_rejects(output: str) -> None:
    res = contains_sql(output)
    assert (res["pass"], res["score"]) == (False, 0)


def test_contains_sql_default_failure_reason() -> None:
    assert contains_sql("")["reason"] == "Output does not contain valid SQL"


def test_not_contains_sql_when_every_block_invalid() -> None:
    output = f"{FENCE}sql\nnot sql\n{FENCE}\n{FENCE}\nstill not sql\n{FENCE}"
    res = contains_sql(output, inverse=True)
    assert (res["pass"], res["score"]) == (True, 1)
