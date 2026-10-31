"""Ports test/assertions/xml.test.ts and test/assertions/html.test.ts.

Expected values come from upstream's specs. The extra cases at the end were checked against
real Node (saxes 6.0.0, fast-xml-parser 5.7.2, parse5 7.3.0) running the upstream sources.
"""

from __future__ import annotations

import textwrap
from typing import Any

import pytest

from promptphoo.assertions.base import AssertionParams, AssertionValueError
from promptphoo.assertions.markup import (
    HANDLERS,
    contains_xml,
    handle_contains_html,
    handle_is_html,
    handle_is_xml,
    validate_xml,
)

VALID = {"isValid": True, "reason": "XML is valid and contains all required elements"}


def params(
    output: str, assertion_type: str, base_type: str, inverse: bool = False, rendered: Any = None
) -> AssertionParams:
    return AssertionParams(
        assertion={"type": assertion_type},
        base_type=base_type,
        inverse=inverse,
        output=output,
        output_string=output,
        rendered_value=rendered,
    )


def test_handlers_exported() -> None:
    expected = {
        "is-xml": handle_is_xml,
        "contains-xml": handle_is_xml,
        "is-html": handle_is_html,
        "contains-html": handle_contains_html,
    }
    assert expected == HANDLERS


# ============================================ validateXml ====================================


def test_simple_valid_xml() -> None:
    assert validate_xml("<root><child>Content</child></root>") == VALID


def test_malformed_xml() -> None:
    res = validate_xml("<root><child>Content</child></root")
    assert res["isValid"] is False
    assert "XML parsing failed" in res["reason"]


@pytest.mark.parametrize(
    "xml",
    [
        "<root/>",
        "  <root/>\n",
        "﻿<root/>",
        "\n<!-- before --><root/>",
        "<!-- before --><root/><!-- after -->",
        "<\U00010000>astral name</\U00010000>",
        "<!DOCTYPE root><root/>",
        "<!DOCTYPE \U00010000><\U00010000/>",
    ],
)
def test_well_formed_edge_cases(xml: str) -> None:
    assert validate_xml(xml) == VALID


@pytest.mark.parametrize(
    "xml",
    [
        "<a></b>",
        "<a><b></a></b>",
        "<a>text</a><b>more</b>",
        "<a attr=unquoted>x</a>",
        "<a/><b/>",
        "<a/>trailing",
        "<root><!-- bad -- comment --></root>",
        '<root attr="1" attr="2"/>',
        "<p:root/>",
        '<root xmlns:xml="wrong"/>',
        '<root xmlns:a="urn:x" xmlns:b="urn:x" a:id="1" b:id="2"/>',
        "<root>&undefined;</root>",
        "<!DOCTYPE root><root>&undefined;</root>",
        "<root>\u0001</root>",
        "",
        '"<a/>"',
        "```xml\n<a/>\n```",
    ],
)
def test_non_well_formed_xml(xml: str) -> None:
    res = validate_xml(xml)
    assert res["isValid"] is False
    assert "XML parsing failed" in res["reason"]


@pytest.mark.parametrize(
    "xml",
    [
        '<!DOCTYPE root [<!ENTITY writer "Donald">]><root>&writer;</root>',
        '<!DOCTYPE root [<!ENTITY logo SYSTEM "logo.png" NDATA png>]><root>&logo;</root>',
        '<!DOCTYPE root [<!ENTITY bad "<a>">]><root>&bad;</root>',
        "<!DOCTYPE root [<!ELEMENT root (child>]><root/>",
    ],
)
def test_rejects_dtd_internal_subsets(xml: str) -> None:
    assert validate_xml(xml) == {
        "isValid": False,
        "reason": "DTD internal subsets are not supported by is-xml validation",
    }


@pytest.mark.parametrize(
    "xml",
    [
        '<!DOCTYPE root SYSTEM "weird[name].dtd"><root/>',
        '<!DOCTYPE root PUBLIC "-//x//EN" "weird[bracket].dtd"><root/>',
    ],
)
def test_external_doctype_with_brackets_in_literals(xml: str) -> None:
    assert validate_xml(xml) == VALID


@pytest.mark.parametrize(
    "xml",
    [
        '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" '
        '"http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">'
        '<html xmlns="http://www.w3.org/1999/xhtml"><body>&nbsp;</body></html>',
        '<?xml version="1.0" standalone="no"?><!DOCTYPE root SYSTEM "entities.dtd">'
        "<root>&external;</root>",
    ],
)
def test_external_doctype_entity_references(xml: str) -> None:
    assert validate_xml(xml) == VALID


def test_standalone_rejects_undeclared_external_entities() -> None:
    res = validate_xml(
        '<?xml version="1.0" standalone="yes"?><!DOCTYPE root SYSTEM "entities.dtd">'
        "<root>&external;</root>"
    )
    assert res["isValid"] is False
    assert "XML parsing failed" in res["reason"]


@pytest.mark.parametrize(
    "xml",
    [
        "<!DOCTYPE><root/>",
        "<!DOCTYPEroot><root/>",
        "<!DOCTYPE 123><root/>",
        "<!DOCTYPE root junk><root/>",
        "<!DOCTYPE root SYSTEM><root/>",
        '<!DOCTYPE root SYSTEM "sys.dtd" junk><root/>',
        '<!DOCTYPE root SYSTEM "sys.dtd#fragment"><root/>',
        '<!DOCTYPE root PUBLIC "-//x//EN"><root/>',
        '<!DOCTYPE root PUBLIC "-//x//EN" "sys.dtd#fragment"><root/>',
        '<!DOCTYPE root PUBLIC "-//x//[invalid]//EN" "sys.dtd"><root/>',
        '<!DOCTYPE root PUBLIC "-//x//\tinvalid//EN" "sys.dtd"><root/>',
    ],
)
def test_malformed_external_doctype(xml: str) -> None:
    assert validate_xml(xml) == {
        "isValid": False,
        "reason": "XML parsing failed: Malformed DOCTYPE declaration",
    }


def test_attributes_and_namespaces() -> None:
    assert validate_xml('<root><child id="1">Content</child></root>') == VALID
    assert (
        validate_xml('<root xmlns:ns="http://example.com"><ns:child>Content</ns:child></root>')
        == VALID
    )


@pytest.mark.parametrize(
    ("xml", "required"),
    [
        (
            "<analysis><classification>T-shirt</classification><color>Red</color></analysis>",
            ["analysis.classification", "analysis.color"],
        ),
        (
            "<root><parent><child><grandchild>Content</grandchild></child></parent></root>",
            ["root.parent.child.grandchild"],
        ),
        (
            "<root><emptyChild></emptyChild><nonEmptyChild>Content</nonEmptyChild></root>",
            ["root.emptyChild", "root.nonEmptyChild"],
        ),
        ("<root><child>Content1</child><child>Content2</child></root>", ["root.child"]),
        (
            "<root><item><name>First</name></item><item><name>Second</name></item></root>",
            ["root.item.0.name", "root.item.1.name"],
        ),
        ("<root><child><![CDATA[<p>This is CDATA content</p>]]></child></root>", ["root.child"]),
        (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<?xml-stylesheet type="text/xsl" href="style.xsl"?>'
            "<root><child>Content</child></root>",
            ["root.child"],
        ),
        ("<root><!-- This is a comment --><child>Content</child></root>", ["root.child"]),
    ],
)
def test_required_elements_present(xml: str, required: list[str]) -> None:
    assert validate_xml(xml, required) == VALID


def test_required_element_missing() -> None:
    assert validate_xml(
        "<analysis><classification>T-shirt</classification></analysis>",
        ["analysis.classification", "analysis.color"],
    ) == {"isValid": False, "reason": "XML is missing required elements: analysis.color"}


def test_nested_required_element_missing() -> None:
    assert validate_xml(
        "<root><parent><child></child></parent></root>", ["root.parent.child.grandchild"]
    ) == {
        "isValid": False,
        "reason": "XML is missing required elements: root.parent.child.grandchild",
    }


ANALYSIS_REQUIRED = [
    "analysis.classification",
    "analysis.color",
    "analysis.features",
    "analysis.style",
    "analysis.confidence",
    "analysis.reasoning",
]


def test_example_xml_structure() -> None:
    xml = textwrap.dedent(
        """\
        <analysis>
          <classification>T-shirt/top</classification>
          <color>White with black print</color>
          <features>Large circular graphic design on the front, resembling a smiley face</features>
          <style>Modern, casual streetwear</style>
          <confidence>9</confidence>
          <reasoning>The image clearly shows a short-sleeved garment.</reasoning>
        </analysis>"""
    )
    assert validate_xml(xml, ANALYSIS_REQUIRED) == VALID


def test_example_xml_structure_missing_element() -> None:
    xml = textwrap.dedent(
        """\
        <analysis>
          <classification>T-shirt/top</classification>
          <color>White with black print</color>
          <style>Modern, casual streetwear</style>
          <confidence>9</confidence>
          <reasoning>The image clearly shows a short-sleeved garment.</reasoning>
        </analysis>"""
    )
    assert validate_xml(xml, ANALYSIS_REQUIRED) == {
        "isValid": False,
        "reason": "XML is missing required elements: analysis.features",
    }


# ============================================ containsXml ====================================

NO_MATCH = {"isValid": False, "reason": "No valid XML content found matching the requirements"}
NOT_FOUND = {"isValid": False, "reason": "No XML content found in the output"}


@pytest.mark.parametrize(
    ("text", "required"),
    [
        ("Some text <root><child>Content</child></root> more text", None),
        ("Text <root><child>Content</child></root> more", ["root.child"]),
        (
            "Text <root><item><name>First</name></item>"
            "<item><name>Second</name></item></root> more",
            ["root.item.1.name"],
        ),
        ("<root1>Content</root1> text <root2><child>More</child></root2>", ["root2.child"]),
        ("<xml1>content1</xml1> more text <xml2>content2</xml2>", ["xml1", "xml2"]),
        ("Text <unclosed><root><child>Content</child></root>", ["root.child"]),
    ],
)
def test_contains_xml_valid(text: str, required: list[str] | None) -> None:
    assert contains_xml(text, required)["isValid"] is True


@pytest.mark.parametrize(
    ("text", "required"),
    [
        ("Text <élément>Content</élément> more", ["élément"]),
        ("Text <root><child>Café 東京 مرحبا</child></root> more", ["root.child"]),
        (
            'Text <root><child label="Crème brûlée 東京">Content</child></root> more',
            ["root.child.@_label"],
        ),
        ("<root><!-- </root> --><child>Content</child></root>", ["root.child"]),
        ("<root><![CDATA[</root>]]><child>Content</child></root>", ["root.child"]),
    ],
)
def test_contains_xml_valid_exact(text: str, required: list[str]) -> None:
    assert contains_xml(text, required) == VALID


def test_contains_xml_no_xml() -> None:
    assert contains_xml("This is just plain text") == NOT_FOUND


def test_contains_xml_missing_required() -> None:
    assert contains_xml("Text <root><child>Content</child></root> more", ["root.missing"]) == (
        NO_MATCH
    )


def test_contains_xml_required_root_relative() -> None:
    text = "Text <wrapper><root><child>Content</child></root></wrapper>"
    assert contains_xml(text, ["root.child"]) == NO_MATCH


def test_contains_xml_parses_only_maximal_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    from promptphoo.assertions import markup

    calls: list[str] = []
    original = markup._fxp_parse

    def spy(xml: str) -> Any:
        calls.append(xml)
        return original(xml)

    monkeypatch.setattr(markup, "_fxp_parse", spy)
    text = "<a>" * 1000 + "</a>" * 1000
    assert contains_xml(text, ["a.missing"]) == NO_MATCH
    assert len(calls) == 1


def test_contains_xml_malformed_tag_stream() -> None:
    assert contains_xml("<a" * 5000) == NOT_FOUND


def test_contains_xml_malformed_comment_stream() -> None:
    assert contains_xml("<!--" * 10000) == NOT_FOUND


# ============================ required element names (both validators) ======================

VALIDATORS = [pytest.param(validate_xml, id="is-xml"), pytest.param(contains_xml, id="contains")]


@pytest.mark.parametrize("validate", VALIDATORS)
@pytest.mark.parametrize("required", ["toString", "root.toString", "root.item.0.toString"])
def test_rejects_inherited_property(validate: Any, required: str) -> None:
    xml = "<root><item><name>First</name></item><item>Second</item></root>"
    assert validate(xml, [required])["isValid"] is False


@pytest.mark.parametrize("validate", VALIDATORS)
def test_accepts_empty_false_zero_values(validate: Any) -> None:
    xml = "<root><empty/><enabled>false</enabled><count>0</count></root>"
    assert validate(xml, ["root.empty", "root.enabled", "root.count"])["isValid"] is True


@pytest.mark.parametrize("validate", VALIDATORS)
@pytest.mark.parametrize(
    "name",
    [
        "toString",
        "valueOf",
        "hasOwnProperty",
        "__defineGetter__",
        "__defineSetter__",
        "__lookupGetter__",
        "__lookupSetter__",
    ],
)
def test_matches_actual_dangerous_named_element(validate: Any, name: str) -> None:
    assert validate(f"<root><{name}/></root>", [f"root.{name}"])["isValid"] is True
    assert validate("<root><child/></root>", [f"root.{name}"])["isValid"] is False


@pytest.mark.parametrize("validate", VALIDATORS)
def test_to_string_paths_distinct(validate: Any) -> None:
    assert validate("<root><toString/></root>", ["root.__toString"])["isValid"] is False
    assert validate("<root><__toString/></root>", ["root.toString"])["isValid"] is False
    assert (
        validate("<root><toString/><__toString/></root>", ["root.toString", "root.__toString"])[
            "isValid"
        ]
        is True
    )


@pytest.mark.parametrize("validate", VALIDATORS)
def test_nested_original_names_in_repeated_elements(validate: Any) -> None:
    xml = (
        "<root><toString><valueOf>First</valueOf></toString>"
        "<toString><valueOf>Second</valueOf></toString></root>"
    )
    assert validate(xml, ["root.toString.0.valueOf", "root.toString.1.valueOf"])["isValid"]


@pytest.mark.parametrize("validate", VALIDATORS)
@pytest.mark.parametrize("name", ["__proto__", "constructor", "prototype"])
def test_rejects_critical_element_names(validate: Any, name: str) -> None:
    assert validate(f"<root><{name}>value</{name}></root>")["isValid"] is False


# =========================================== handleIsXml =====================================


def test_handle_is_xml_string_value() -> None:
    p = params("<a><b>1</b></a>", "is-xml", "is-xml", rendered="a.b, a.c")
    assert handle_is_xml(p) == {
        "pass": False,
        "score": 0,
        "reason": "XML is missing required elements: a.c",
        "assertion": {"type": "is-xml"},
    }


def test_handle_is_xml_object_value_and_inverse() -> None:
    p = params(
        "<a><b>1</b></a>", "not-is-xml", "is-xml", inverse=True, rendered={"requiredElements": []}
    )
    assert handle_is_xml(p)["reason"] == "XML is valid and contains all required elements"
    p = params("text <a/>", "contains-xml", "contains-xml", rendered=["a"])
    assert handle_is_xml(p) == {
        "pass": True,
        "score": 1,
        "reason": "Assertion passed",
        "assertion": {"type": "contains-xml"},
    }


def test_handle_is_xml_bad_object_value() -> None:
    with pytest.raises(
        AssertionValueError, match="xml assertion must contain a string, array value, or no value"
    ):
        handle_is_xml(params("<a/>", "is-xml", "is-xml", rendered={"foo": 1}))


# ======================================= extra Node-checked cases ============================


@pytest.mark.parametrize(
    ("xml", "required", "valid"),
    [
        ('<?xml version="1.0"?><root a="1"><child b="x">t</child></root>', ["?xml"], True),
        ('<root a="1"><child b="x">t</child></root>', ["root.@_a", "root.child.#text"], True),
        ("<root><parent><child>t</child></parent></root>", ["root.parent.child.x"], False),
        ("<root>text<child/>more</root>", ["root.#text", "root.child"], True),
        ("<root><a disabled/></root>", ["root.a.@_disabled"], False),  # saxes: no bare attrs
        ("<root><item/><item/></root>", ["root.item.01"], False),
        ("<root><a></a></root>", [""], False),
        ("<a>" * 150 + "</a>" * 150, None, False),  # fast-xml-parser maxNestedTags
    ],
)
def test_extra_validate_xml(xml: str, required: list[str] | None, valid: bool) -> None:
    assert validate_xml(xml, required)["isValid"] is valid


def test_extra_is_xml_reasons() -> None:
    assert validate_xml("<root><a></a></root>", [""]) == {
        "isValid": False,
        "reason": "XML is missing required elements: ",
    }
    assert validate_xml("<root><__proto__>value</__proto__></root>") == {
        "isValid": False,
        "reason": 'XML parsing failed: [SECURITY] Invalid name: "__proto__" is a reserved '
        "JavaScript keyword that could cause prototype pollution",
    }
    assert validate_xml("<a>" * 150 + "</a>" * 150) == {
        "isValid": False,
        "reason": "XML parsing failed: Maximum nested tags exceeded",
    }


@pytest.mark.parametrize(
    ("text", "required", "valid"),
    [
        ("<a><b/><c/></a> and <d/>", ["b"], False),
        ("x <a/> y <b/> z", ["a", "b"], True),
        ("<a><b>1</b><c>2</c></a>", ["b", "c"], False),  # nested inside well-formed <a>
        ("x <a disabled/>", ["a.@_disabled"], True),
        ("<p>Hi</p><p>There</p>", ["p"], True),
        ('<a x="1>2">t</a>', ["a.@_x"], True),
    ],
)
def test_extra_contains_xml(text: str, required: list[str], valid: bool) -> None:
    assert contains_xml(text, required)["isValid"] is valid


# =========================================== contains-html ===================================


def contains_html(output: str, inverse: bool = False) -> dict[str, Any]:
    t = "not-contains-html" if inverse else "contains-html"
    return handle_contains_html(params(output, t, "contains-html", inverse=inverse))


def is_html(output: str, inverse: bool = False) -> dict[str, Any]:
    t = "not-is-html" if inverse else "is-html"
    return handle_is_html(params(output, t, "is-html", inverse=inverse))


def test_contains_html_pass() -> None:
    assert contains_html("<div>Hello World</div>") == {
        "pass": True,
        "score": 1,
        "reason": "Assertion passed",
        "assertion": {"type": "contains-html"},
    }


def test_contains_html_fail() -> None:
    assert contains_html("Just plain text without any HTML") == {
        "pass": False,
        "score": 0,
        "reason": "Expected output to contain HTML content",
        "assertion": {"type": "contains-html"},
    }


@pytest.mark.parametrize(
    "html",
    [
        "<p>paragraph</p>",
        '<a href="link">text</a>',
        '<img src="image.jpg" />',
        "<h1>heading</h1>",
        '<div class="container"><span>nested</span></div>',
        "<!DOCTYPE html><html><head><title>Test</title></head><body>Content</body></html>",
        "Some text with <strong>emphasis</strong> in it",
        '<input type="text" value="test" />',
        "<br />",
        "<table><tr><td>cell</td></tr></table>",
        "<div>Text with &amp; entity</div>",
        "<p>Multiple &nbsp; HTML &lt; entities &gt;</p>",
        '<custom-element attr="value">Web component</custom-element>',
        '<div data-id="123" class="test">Modern HTML</div>',
        """
      <div class="container" id="main">
        <h1 style="color: red;">Title</h1>
        <p data-value="123">Paragraph with <a href="/link">link</a></p>
        <img src="image.jpg" alt="description" width="100" height="100" />
      </div>
    """,
        "<div>&nbsp;</div>",
        "<!-- comment --><span>text</span>",
        "<br/><hr/>",
        "Text with <b>bold</b> and <i>italic</i>",
    ],
)
def test_contains_html_detects(html: str) -> None:
    res = contains_html(html)
    assert (res["pass"], res["score"]) == (True, 1)


@pytest.mark.parametrize(
    "text",
    [
        "Just plain text",
        "2 < 3 and 4 > 1",
        "email@example.com",
        "Math: a < b > c",
        "< div >",
        "Generic <example> text",
        "if (a<b) { return c>d; }",
        "The price is <$50",
        "Email me at <john@example.com>",
        "<custom>",
        "&amp;",
        "<!-- comment -->",
    ],
)
def test_contains_html_no_false_positives(text: str) -> None:
    res = contains_html(text)
    assert (res["pass"], res["score"]) == (False, 0)


def test_contains_html_inverse() -> None:
    assert contains_html("Just plain text", inverse=True) == {
        "pass": True,
        "score": 1,
        "reason": "Assertion passed",
        "assertion": {"type": "not-contains-html"},
    }
    assert contains_html("<div>HTML content</div>", inverse=True) == {
        "pass": False,
        "score": 0,
        "reason": "Expected output to not contain HTML content",
        "assertion": {"type": "not-contains-html"},
    }


# ============================================== is-html ======================================


@pytest.mark.parametrize(
    "html",
    [
        "<div>Hello World</div>",
        "<!DOCTYPE html><html><head><title>Test</title></head><body>Content</body></html>",
        "<p>Simple paragraph</p>",
        "<div><span>Nested</span></div>",
        '<img src="test.jpg" />',
        "<br />",
        '<div class="test" id="main">Content</div>',
        "  <div>With whitespace</div>  ",
        "<div>Fragment without doctype</div>",
        "<h1>Title</h1><p>Paragraph</p>",
        "<ul><li>Item 1</li><li>Item 2</li></ul>",
        '<form><input type="text" /><button>Submit</button></form>',
        "<script>const x = 1;</script>",
        "<style>body { color: red; }</style>",
        "<title>Page</title>",
        "<svg><circle /></svg>",
        "<template><div>x</div></template>",
        '<custom-element data-id="123">Web component</custom-element>',
        "<CUSTOM-ELEMENT>Uppercase custom element</CUSTOM-ELEMENT>",
        "<body>x</body>",
        '<body attr="val">x</body>',
        "<body/>",
        "<notatag>" * 15_000 + "<div>leaf</div>" + "</notatag>" * 15_000,
    ],
)
def test_is_html_valid(html: str) -> None:
    res = is_html(html)
    assert (res["pass"], res["score"]) == (True, 1)


@pytest.mark.parametrize(
    ("output", "reason"),
    [
        ("Just plain text", "Output must be wrapped in HTML tags"),
        ("Some text with <strong>HTML</strong> inside", "Output must be wrapped in HTML tags"),
        ("Text before <div>HTML</div>", "Output must be wrapped in HTML tags"),
        ("<div>HTML</div> Text after", "Output must be wrapped in HTML tags"),
        ('<?xml version="1.0"?><root>XML</root>', "Output appears to be XML, not HTML"),
        ("", "Output is empty"),
        ("   ", "Output is empty"),
        ("<notarealtag>", "Output does not contain recognized HTML elements"),
        ("2 < 3 and 4 > 1", "Output must be wrapped in HTML tags"),
        ("<!DOCTYPE html>", "Output does not contain recognized HTML elements"),
        ("<table>stray text<tr><td>x</td></tr></table>", "Output must be wrapped in HTML tags"),
        ("<headphones>plain text</headphones>", "Output does not contain recognized HTML elements"),
        ("<bodyguard>x</bodyguard>", "Output does not contain recognized HTML elements"),
        ("<htmlworld>x</htmlworld>", "Output does not contain recognized HTML elements"),
        ("<headd>x</headd>", "Output does not contain recognized HTML elements"),
        ("<html", "Output does not contain recognized HTML elements"),
        ("<head", "Output does not contain recognized HTML elements"),
        ("<body", "Output does not contain recognized HTML elements"),
        ("plain <body>", "Output must be wrapped in HTML tags"),
        ("plain <body", "Output must be wrapped in HTML tags"),
    ],
)
def test_is_html_invalid(output: str, reason: str) -> None:
    res = is_html(output)
    assert (res["pass"], res["score"], res["reason"]) == (False, 0, reason)


@pytest.mark.parametrize(
    "content",
    [
        "Here is some HTML: <div>test</div>",
        "<div>HTML</div> and some text",
        "Text <br /> more text",
    ],
)
def test_is_html_rejects_mixed_content(content: str) -> None:
    assert is_html(content)["pass"] is False


def test_is_html_inverse() -> None:
    assert is_html("Just plain text", inverse=True) == {
        "pass": True,
        "score": 1,
        "reason": "Assertion passed",
        "assertion": {"type": "not-is-html"},
    }
    assert is_html("<div>Valid HTML</div>", inverse=True) == {
        "pass": False,
        "score": 0,
        "reason": "Output is valid HTML",
        "assertion": {"type": "not-is-html"},
    }


@pytest.mark.parametrize(
    ("output", "valid"),
    [
        ("<html><body>x</body>y</html>", True),
        ("<div/>text", True),
        ("<p>a<div>b</div>c", False),
        ("<table><div>x</div></table>", True),
        ("<table><tr><td>x</td></tr></table>tail", False),
        ("<head><title>t</title></head>loose", False),
        ("<a><b></a>text</b>", True),
        ("<ul><h1></ul><", False),
        ("<p><table>text<body>", True),
        ("<frameset></frameset>", False),
        ("<span><frameset><b>", False),
        ("<div><math><mi></button><frameset>", False),
        ("<textarea>x</textarea>", True),
        ("<template>text</template>", True),
    ],
)
def test_is_html_extra_tree_cases(output: str, valid: bool) -> None:
    assert is_html(output)["pass"] is valid
