"""Text-similarity assertions: levenshtein, bleu, gleu, rouge-n/l/s and meteor.

Ports src/assertions/levenshtein.ts, bleu.ts, gleu.ts, ngrams.ts, rouge.ts and meteor.ts, plus the
parts of their npm dependencies that decide the score:

- fastest-levenshtein@1.0.16 ``distance``: exact edit distance over UTF-16 code units (like JS
  strings), computed with the bit-parallel Myers/Hyyro algorithm.
- js-rouge@3.2.3 (the version in promptfoo's lockfile): ``treeBankTokenize``, ``nGram``,
  ``fMeasure``, ``lcsIndices`` and the ``l``/``s`` scorers are exact ports. Its sentence segmenter
  (``sentenceSegment``, ~6,300 lines of heuristics for quotations, citations, lists, URLs and
  ellipses) is ported for its main paths in the ``caseNeutral`` mode promptfoo uses; see the
  comment above ``_sentence_segment``. ROUGE-N never segments, so it is exact.
- V8's ``Math.log``/``Math.exp`` (fdlibm) are ported so BLEU matches bit for bit.
- natural@8.1.1: ``PorterStemmer`` is an exact port; ``WordNet.lookup`` is an exact port of its
  index binary search and data-file parsing, reading WordNet dict files from nltk's data
  (``wordnet31`` preferred, matching wordnet-db 3.1, else ``wordnet`` 3.0). Without them METEOR
  returns the "requires" failure that Node returns when ``natural`` is missing.
"""

from __future__ import annotations

import math
import re
import struct
import unicodedata
from collections.abc import Callable
from functools import cache, reduce
from typing import Any

from promptphoo.assertions.base import (
    AssertionParams,
    GradingResult,
    Handler,
    invariant,
)
from promptphoo.jsvalues import js_string, to_fixed

# --------------------------------------------------------------------------------------------
# JS string helpers
# --------------------------------------------------------------------------------------------

# JS \s (WhiteSpace + LineTerminator). Unlike Python's \s it has U+FEFF and lacks \x1c-\x1f, \x85.
_WS = "\t\n\x0b\x0c\r \xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
_WS_CHARS = frozenset(
    "\t\n\x0b\x0c\r \xa0\u1680\u2028\u2029\u202f\u205f\u3000\ufeff"
    + "".join(chr(c) for c in range(0x2000, 0x200B))
)
_WS_RUN = re.compile(f"[{_WS}]+")
_JS_TRIM = re.compile(f"^[{_WS}]+|[{_WS}]+$")


def _is_ws(ch: str) -> bool:
    return ch in _WS_CHARS


def _js_trim(s: str) -> str:
    return _JS_TRIM.sub("", s)


def _js_trim_start(s: str) -> str:
    return re.sub(f"^[{_WS}]+", "", s)


def _js_trim_end(s: str) -> str:
    return re.sub(f"[{_WS}]+$", "", s)


def _js_split_ws(s: str) -> list[str]:
    """s.split(/\\s+/)."""
    return _WS_RUN.split(s)


def _js_lower(s: str) -> str:
    return s.lower()


def _char(s: str, i: int) -> str:
    """JS `s[i] ?? ''` (no negative indexing)."""
    return s[i] if 0 <= i < len(s) else ""


def _js_slice(s: str, start: int, end: int | None = None) -> str:
    n = len(s)
    a = max(n + start, 0) if start < 0 else min(start, n)
    b = n if end is None else max(n + end, 0) if end < 0 else min(end, n)
    return s[a:b] if a < b else ""


def _js_substr(s: str, start: int, length: int | None = None) -> str:
    n = len(s)
    a = max(n + start, 0) if start < 0 else min(start, n)
    b = n if length is None else min(a + max(length, 0), n)
    return s[a:b]


@cache
def _category_ranges() -> dict[str, list[tuple[int, int]]]:
    """Code-point ranges per Unicode general category (one pass over all code points)."""
    ranges: dict[str, list[tuple[int, int]]] = {}
    category = unicodedata.category
    start, current = 0, category("\0")
    for cp in range(1, 0x110000):
        c = category(chr(cp))
        if c != current:
            ranges.setdefault(current, []).append((start, cp - 1))
            start, current = cp, c
    ranges.setdefault(current, []).append((start, 0x10FFFF))
    return ranges


@cache
def _uclass(categories: str) -> str:
    """A regex character-class body for the Unicode general categories in `categories`
    (space-separated prefixes such as "L M" or "Nd"), i.e. JS \\p{...} under the u flag."""
    prefixes = tuple(categories.split())
    spans = sorted(
        span for cat, rs in _category_ranges().items() if cat.startswith(prefixes) for span in rs
    )
    return "".join(
        re.escape(chr(a)) if a == b else f"{re.escape(chr(a))}-{re.escape(chr(b))}"
        for a, b in spans
    )


def _cat(ch: str) -> str:
    return unicodedata.category(ch) if len(ch) == 1 else ""


def _is_cased(ch: str) -> bool:
    """\\p{Cased}: Lu/Ll/Lt plus Other_Lowercase/Other_Uppercase (approximated by case mappings)."""
    if len(ch) != 1:
        return False
    return _cat(ch) in ("Lu", "Ll", "Lt") or ch.lower() != ch or ch.upper() != ch


def _is_id_continue(ch: str) -> bool:
    if not ch:
        return False
    return ("a" + ch).isidentifier() or ch in "\u00b7\u0387\u1369\u136a\u136b\u136c\u136d"


def _starts_with_cased(s: str) -> bool:
    t = _js_trim(s)
    return bool(t) and _is_cased(t[0])


# --------------------------------------------------------------------------------------------
# V8 Math.log / Math.exp (base::ieee754, ports of FreeBSD msun e_log.c / e_exp.c). glibc's libm
# can differ in the last bit, which would make BLEU scores differ from Node's.
# --------------------------------------------------------------------------------------------


def _hi_lo(x: float) -> tuple[int, int]:
    bits = struct.unpack("<Q", struct.pack("<d", x))[0]
    return bits >> 32, bits & 0xFFFFFFFF


def _from_words(hi: int, lo: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", ((hi & 0xFFFFFFFF) << 32) | lo))[0]


def _signed32(v: int) -> int:
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v & 0x80000000 else v


_LN2_HI = 6.93147180369123816490e-01
_LN2_LO = 1.90821492927058770002e-10
_TWO54 = 1.80143985094819840000e16
_LG = (
    6.666666666666735130e-01, 3.999999999940941908e-01, 2.857142874366239149e-01,
    2.222219843214978396e-01, 1.818357216161805012e-01, 1.531383769920937332e-01,
    1.479819860511658591e-01,
)  # fmt: skip


def js_log(x: float) -> float:
    """Math.log as V8 computes it."""
    hx_u, lx = _hi_lo(x)
    hx = _signed32(hx_u)
    k = 0
    if hx < 0x00100000:
        if ((hx & 0x7FFFFFFF) | lx) == 0:
            return -math.inf
        if hx < 0:
            return math.nan
        k -= 54
        x *= _TWO54
        hx = _signed32(_hi_lo(x)[0])
    if hx >= 0x7FF00000:
        return x + x
    k += (hx >> 20) - 1023
    hx &= 0x000FFFFF
    i = (hx + 0x95F64) & 0x100000
    x = _from_words(hx | (i ^ 0x3FF00000), _hi_lo(x)[1])
    k += i >> 20
    f = x - 1.0
    if (0x000FFFFF & (2 + hx)) < 3:
        if f == 0:
            if k == 0:
                return 0.0
            dk = float(k)
            return dk * _LN2_HI + dk * _LN2_LO
        r = f * f * (0.5 - 0.33333333333333333 * f)
        if k == 0:
            return f - r
        dk = float(k)
        return dk * _LN2_HI - ((r - dk * _LN2_LO) - f)
    lg1, lg2, lg3, lg4, lg5, lg6, lg7 = _LG
    s = f / (2.0 + f)
    dk = float(k)
    z = s * s
    i = hx - 0x6147A
    w = z * z
    j = 0x6B851 - hx
    t1 = w * (lg2 + w * (lg4 + w * lg6))
    t2 = z * (lg1 + w * (lg3 + w * (lg5 + w * lg7)))
    i |= j
    r = t2 + t1
    if i > 0:
        hfsq = 0.5 * f * f
        if k == 0:
            return f - (hfsq - s * (hfsq + r))
        return dk * _LN2_HI - ((hfsq - (s * (hfsq + r) + dk * _LN2_LO)) - f)
    if k == 0:
        return f - s * (f - r)
    return dk * _LN2_HI - ((s * (f - r) - dk * _LN2_LO) - f)


_EXP_P = (
    1.66666666666666019037e-01, -2.77777777770155933842e-03, 6.61375632143793436117e-05,
    -1.65339022054652515390e-06, 4.13813679705723846039e-08,
)  # fmt: skip
_INVLN2 = 1.44269504088896338700e00
_TWOM1000 = 9.33263618503218878990e-302


def js_exp(x: float) -> float:
    """Math.exp as V8 computes it."""
    hx, lx = _hi_lo(x)
    xsb = (hx >> 31) & 1
    hx &= 0x7FFFFFFF
    hi = lo = 0.0
    k = 0
    if hx >= 0x40862E42:
        if hx >= 0x7FF00000:
            if ((hx & 0xFFFFF) | lx) != 0:
                return x + x
            return x if xsb == 0 else 0.0
        if x > 7.09782712893383973096e02:
            return math.inf
        if x < -7.45133219101941108420e02:
            return 0.0
    if hx > 0x3FD62E42:
        if hx < 0x3FF0A2B2:
            if x == 1.0:
                return math.e
            hi = x - (_LN2_HI if xsb == 0 else -_LN2_HI)
            lo = _LN2_LO if xsb == 0 else -_LN2_LO
            k = 1 - xsb - xsb
        else:
            k = int(_INVLN2 * x + (0.5 if xsb == 0 else -0.5))
            t = float(k)
            hi = x - t * _LN2_HI
            lo = t * _LN2_LO
        x = hi - lo
    elif hx < 0x3E300000:
        return 1.0 + x
    t = x * x
    if k >= -1021:
        twopk = _from_words(0x3FF00000 + (k << 20), 0)
    else:
        twopk = _from_words(0x3FF00000 + ((k + 1000) << 20), 0)
    p1, p2, p3, p4, p5 = _EXP_P
    c = x - t * (p1 + t * (p2 + t * (p3 + t * (p4 + t * p5))))
    if k == 0:
        return 1.0 - ((x * c) / (c - 2.0) - x)
    y = 1.0 - ((lo - (x * c) / (2.0 - c)) - hi)
    if k >= -1021:
        if k == 1024:
            return y * 2.0 * 8.98846567431158e307
        return y * twopk
    return y * twopk * _TWOM1000


# --------------------------------------------------------------------------------------------
# ngrams.ts
# --------------------------------------------------------------------------------------------


def get_ngrams(words: list[str], n: int) -> list[str]:
    if n <= 0 or n > len(words):
        return []
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def count_ngrams(ngrams: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for gram in ngrams:
        counts[gram] = counts.get(gram, 0) + 1
    return counts


# --------------------------------------------------------------------------------------------
# levenshtein.ts (fastest-levenshtein `distance`)
# --------------------------------------------------------------------------------------------


def _utf16_units(s: str) -> list[int]:
    data = s.encode("utf-16-le", "surrogatepass")
    return [data[i] | (data[i + 1] << 8) for i in range(0, len(data), 2)]


def levenshtein_distance(a: str, b: str) -> int:
    """Edit distance over UTF-16 code units, as fastest-levenshtein computes it on JS strings."""
    ua, ub = _utf16_units(a), _utf16_units(b)
    if len(ua) < len(ub):
        ua, ub = ub, ua
    m = len(ub)
    if m == 0:
        return len(ua)
    peq: dict[int, int] = {}
    for i, c in enumerate(ub):
        peq[c] = peq.get(c, 0) | (1 << i)
    mask = (1 << m) - 1
    last = 1 << (m - 1)
    pv, mv, score = mask, 0, m
    for c in ua:
        eq = peq.get(c, 0)
        xv = eq | mv
        xh = (((eq & pv) + pv) ^ pv) | eq
        ph = (mv | ~(xh | pv)) & mask
        mh = pv & xh
        if ph & last:
            score += 1
        if mh & last:
            score -= 1
        ph = ((ph << 1) | 1) & mask
        mh = (mh << 1) & mask
        pv = (mh | ~(xv | ph)) & mask
        mv = ph & xv
    return score


def handle_levenshtein(params: AssertionParams) -> GradingResult:
    assertion, rendered_value, inverse = params.assertion, params.rendered_value, params.inverse
    invariant(
        isinstance(rendered_value, str), '"levenshtein" assertion type must have a string value'
    )
    lev_distance = levenshtein_distance(params.output_string, rendered_value)
    threshold = assertion.get("threshold")
    if threshold is None:
        threshold = 5
    passed = (lev_distance <= threshold) != inverse
    return {
        "pass": passed,
        "score": 1 if passed else 0,
        "reason": "Assertion passed"
        if passed
        else f"Levenshtein distance {lev_distance} is "
        f"{'less than or equal to' if inverse else 'greater than'} "
        f"threshold {js_string(threshold)}",
        "assertion": assertion,
    }


# --------------------------------------------------------------------------------------------
# bleu.ts
# --------------------------------------------------------------------------------------------


def _is_string_or_string_list(value: Any) -> bool:
    return isinstance(value, str) or (
        isinstance(value, list) and all(isinstance(v, str) for v in value)
    )


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _bleu_tokenize(text: str) -> list[str]:
    return _js_split_ws(_js_trim(_js_lower(text)))


def calculate_bleu_score(
    candidate: str | None,
    references: list[str],
    weights: list[Any] | None = None,
) -> float:
    if weights is None:
        weights = [0.25, 0.25, 0.25, 0.25]
    if candidate is None or len(references) == 0 or len(weights) != 4:
        raise ValueError("Invalid inputs")
    if any(not _is_finite_number(w) for w in weights):
        raise ValueError("Weights must be finite numbers")
    if any(w < 0 for w in weights):
        raise ValueError("Weights must be non-negative")
    if abs(reduce(lambda a, b: a + b, weights) - 1) > 1e-4:
        raise ValueError("Weights must sum to 1")
    if _js_trim(candidate) == "":
        return 0

    candidate_words = _bleu_tokenize(candidate)
    reference_words_list = [_bleu_tokenize(r) for r in references if _js_trim(r) != ""]
    if not reference_words_list:
        return 0

    ref_lengths = [len(words) for words in reference_words_list]
    cand_len = len(candidate_words)

    def closer(prev: int, curr: int) -> int:
        dist_curr = abs(curr - cand_len)
        dist_prev = abs(prev - cand_len)
        return curr if dist_curr < dist_prev or (dist_curr == dist_prev and curr < prev) else prev

    closest_ref_length = reduce(closer, ref_lengths)

    usable_orders: list[tuple[float, float]] = []
    for n in range(1, 5):
        candidate_ngrams = get_ngrams(candidate_words, n)
        total_count = len(candidate_ngrams)
        if total_count == 0:
            continue
        candidate_counts = count_ngrams(candidate_ngrams)
        max_clipped = 0
        for reference_words in reference_words_list:
            reference_counts = count_ngrams(get_ngrams(reference_words, n))
            clipped = 0
            for gram, count in candidate_counts.items():
                clipped += min(count, reference_counts.get(gram, 0))
            max_clipped = max(max_clipped, clipped)
        usable_orders.append((max_clipped / total_count, weights[n - 1]))

    bp = 1.0 if cand_len > closest_ref_length else js_exp(1 - closest_ref_length / cand_len)

    weight_sum = 0
    for _, weight in usable_orders:
        weight_sum += weight
    if weight_sum == 0:
        return 0

    weighted = 0.0
    for precision, weight in usable_orders:
        smoothed = 1e-7 if precision == 0 else precision
        weighted = weighted + (weight / weight_sum) * js_log(smoothed)
    return bp * js_exp(weighted)


def handle_bleu_score(params: AssertionParams) -> GradingResult:
    assertion, rendered_value, inverse = params.assertion, params.rendered_value, params.inverse
    invariant(
        _is_string_or_string_list(rendered_value),
        '"bleu" assertion type must have a string or array of strings value',
    )
    threshold = assertion.get("threshold")
    if threshold is None:
        threshold = 0.5
    references = rendered_value if isinstance(rendered_value, list) else [rendered_value]
    score = calculate_bleu_score(params.output_string, references)
    passed = (score >= threshold) != inverse
    return {
        "pass": passed,
        "score": 1 - score if inverse else score,
        "reason": "Assertion passed"
        if passed
        else f"BLEU score {to_fixed(score, 4)} is {'greater' if inverse else 'less'} "
        f"than threshold {js_string(threshold)}",
        "assertion": assertion,
    }


# --------------------------------------------------------------------------------------------
# gleu.ts
# --------------------------------------------------------------------------------------------


def _tokenize_for_gleu(text: str) -> list[str]:
    words = [w.rstrip(".") for w in _js_split_ws(_js_trim(_js_lower(text)))]
    return words if any(words) else []


def calculate_gleu_score(
    candidate: str | None, references: list[str], min_n: int = 1, max_n: int = 4
) -> float:
    if candidate is None or len(references) == 0:
        raise ValueError("Invalid inputs")
    candidate_words = _tokenize_for_gleu(candidate)
    if not candidate_words:
        return 0

    def reference_score(reference: str) -> float:
        reference_words = _tokenize_for_gleu(reference)
        if candidate_words == reference_words:
            return 1
        match_count = candidate_total = reference_total = 0
        for n in range(min_n, max_n + 1):
            candidate_ngrams = get_ngrams(candidate_words, n)
            reference_ngrams = get_ngrams(reference_words, n)
            reference_counts = count_ngrams(reference_ngrams)
            for gram, candidate_count in count_ngrams(candidate_ngrams).items():
                match_count += min(candidate_count, reference_counts.get(gram, 0))
            candidate_total += len(candidate_ngrams)
            reference_total += len(reference_ngrams)
        if candidate_total == 0 or reference_total == 0:
            return 0
        return min(match_count / candidate_total, match_count / reference_total)

    return max(reference_score(r) for r in references)


def handle_gleu_score(params: AssertionParams) -> GradingResult:
    assertion, rendered_value, inverse = params.assertion, params.rendered_value, params.inverse
    invariant(
        _is_string_or_string_list(rendered_value),
        '"gleu" assertion must have a string or array of strings value',
    )
    threshold = assertion.get("threshold")
    if threshold is None:
        threshold = 0.5
    references = rendered_value if isinstance(rendered_value, list) else [rendered_value]
    score = calculate_gleu_score(params.output_string, references)
    passed = (score >= threshold) != inverse
    return {
        "pass": passed,
        "score": 1 - score if inverse else score,
        "reason": "Assertion passed"
        if passed
        else f"GLEU score {to_fixed(score, 4)} is "
        f"{'greater than or equal to' if inverse else 'less than'} "
        f"threshold {js_string(threshold)}",
        "assertion": assertion,
    }


# --------------------------------------------------------------------------------------------
# js-rouge@3.2.3: treeBankTokenize (tokenize.ts + text.ts)
# --------------------------------------------------------------------------------------------


def _is_punctuation(ch: str) -> bool:
    return _cat(ch).startswith("P")


def _follows_quoted_modifier(s: str, index: int) -> bool:
    window = _js_slice(s, max(0, index - 2), index)
    if not window or _cat(window[-1]) != "Lm":
        return False
    modifier = window[-1]
    if not _is_alphabetic_footnote(s, index - len(modifier) - 1):
        return False
    start = index - len(modifier)
    while start > 0:
        w = _js_slice(s, max(0, start - 2), start)
        if not w or not _cat(w[-1]).startswith("N"):
            break
        start -= 1
    return re.search("[.!?][\"'\u201d\u2019]$", _js_slice(s, max(0, start - 2), start)) is not None


def _is_alphabetic_footnote(s: str, index: int) -> bool:
    following = _char(s, index + 1)
    if following == unicodedata.normalize("NFKC", following):
        return False
    window = _js_slice(s, index + 1, index + 5)
    if not window or _cat(window[0]) != "Lm":
        return False
    return len(window) == 1 or (
        _is_ws(window[1]) or window[1] in ".,;:!?\"'\u201c\u2018\u201d\u2019])}>"
    )


def _quote_opening_context(s: str, index: int) -> bool:
    if index == 0:
        return True
    prev = _char(s, index - 1)
    return (
        _is_ws(prev)
        or _is_punctuation(prev)
        or prev == "<"
        or _js_slice(s, index - 2, index) == "``"
        or _follows_quoted_modifier(s, index)
    )


def _opens_double_quote(s: str, index: int, inside_quotes: bool) -> bool:
    return not inside_quotes and _quote_opening_context(s, index)


_QUOTE_TOKEN = re.compile("``|''|\"")


def _tokenizer_quotes(text: str) -> str:
    """The quote-classifying `text.replace(/``|''|"/g, ...)` step of treeBankTokenize."""
    inside = False
    out: list[str] = []
    last = 0
    for m in _QUOTE_TOKEN.finditer(text):
        quote, index = m.group(0), m.start()
        prev = _char(text, index - 1)
        if quote == "``":
            inside = True
        else:
            spaced = (
                index + len(quote) < len(text)
                or index == 0
                or (prev != "" and (_is_ws(prev) or _cat(prev) == "Ps" or prev == "<"))
            )
            inside = (
                spaced
                and _opens_double_quote(text, index, inside)
                and prev not in (".", "!", "?")
                and (quote == '"' or (index > 0 and _quote_pair_follows(text, index)))
            )
        out.append(text[last:index])
        out.append(" `` " if inside else " '' ")
        last = m.end()
    out.append(text[last:])
    return "".join(out)


def _quote_pair_follows(text: str, index: int) -> bool:
    nxt = _char(text, index + 2)
    if not nxt or not (_cat(nxt)[:1] in ("L", "N") or _cat(nxt) in ("Sc", "Ps")):
        return False
    later = _QUOTE_TOKEN.search(text, index + 2)
    return later is not None and later.group(0) == "''"


_TB_COLON_COMMA = re.compile(r"[:,](?!\d)")
_TB_SYMBOLS = re.compile(r"[;@#$%&]")
_TB_FINAL_PERIOD = re.compile(
    f"([^.])(?<!(?<![A-Za-z0-9_])[A-Za-z]\\.[A-Za-z])(\\.)([\\])}}>'{_WS}]*)\\Z"
)
_TB_QMARK = re.compile(r"[?!]")
_TB_BRACKETS = re.compile(r"[\][(){}<>]")
_TB_APOS = re.compile(r"([^'])' ")
_TB_POSSESSIVE = re.compile(r"'([sSmMdD]) ")
_TB_CLITICS = re.compile(r"('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) ")
_TB_CONTRACTIONS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        "(can)(not)",
        "(d)('ye)",
        "(gim)(me)",
        "(gon)(na)",
        "(got)(ta)",
        "(lem)(me)",
        "(more)('n)",
        "(wan)(na)(?= )",
        "('t)(is)",
        "('t)(was)",
    )
]


def _is_word_char_for_contraction(ch: str) -> bool:
    return ch in ("\u200c", "\u200d") or _is_id_continue(ch)


def _replace_contraction(s: str, pattern: re.Pattern[str]) -> str:
    out: list[str] = []
    last = 0
    pos = 0
    while True:
        m = pattern.search(s, pos)
        if m is None:
            break
        if _is_word_char_for_contraction(_char(s, m.start() - 1)) or _is_word_char_for_contraction(
            _char(s, m.end())
        ):
            pos = m.start() + 1
            continue
        out.append(s[last : m.start()])
        out.append(f" {m.group(1)} {m.group(2)} ")
        last = pos = m.end()
        if m.end() == m.start():
            pos += 1
    out.append(s[last:])
    return "".join(out)


def tree_bank_tokenize(input_text: str) -> list[str]:
    """js-rouge treeBankTokenize (Penn Treebank-style word tokens for one sentence)."""
    text = _js_trim(re.sub(f"[{_WS}\u0085]+", " ", input_text))
    if not text:
        return []
    parse = _tokenizer_quotes(text)
    parse = parse.replace("...", " ... ")
    parse = _TB_COLON_COMMA.sub(lambda m: f" {m.group(0)} ", parse)
    parse = _TB_SYMBOLS.sub(lambda m: f" {m.group(0)} ", parse)
    parse = _TB_FINAL_PERIOD.sub(lambda m: f"{m.group(1)} {m.group(2)}{m.group(3)} ", parse)
    parse = _TB_QMARK.sub(lambda m: f" {m.group(0)} ", parse)
    parse = _TB_BRACKETS.sub(lambda m: f" {m.group(0)} ", parse)
    parse = parse.replace("--", " -- ")
    parse = f" {parse} "
    parse = _TB_APOS.sub(lambda m: f"{m.group(1)} ' ", parse)
    parse = _TB_POSSESSIVE.sub(lambda m: f" '{m.group(1)} ", parse)
    parse = _TB_CLITICS.sub(lambda m: f" {m.group(1)} ", parse)
    for pattern in _TB_CONTRACTIONS:
        parse = _replace_contraction(parse, pattern)
    parse = re.sub(" {2,}", " ", parse)
    parse = re.sub(r"^ | \Z", "", parse)
    return parse.split(" ")


# --------------------------------------------------------------------------------------------
# js-rouge@3.2.3: utils / validation
# --------------------------------------------------------------------------------------------

_MAX_NGRAM_MATERIALIZATION_WORK = 1_000_000


def _validate_ngram_materialization(tokens: list[str], n: int) -> int:
    paded_length = len(tokens)
    gram_count = paded_length - n + 1
    work = gram_count * (2 * n - 1)
    for index in range(paded_length):
        if work > _MAX_NGRAM_MATERIALIZATION_WORK:
            break
        length = len(tokens[index].encode("utf-16-le", "surrogatepass")) // 2
        work += max(length - 1, 0) * min(index + 1, n, gram_count, paded_length - index)
    if work > _MAX_NGRAM_MATERIALIZATION_WORK:
        raise ValueError("N-gram generation exceeds the materialization limit")
    return work


def rouge_ngram(tokens: list[str], n: int = 2) -> list[str]:
    """js-rouge nGram without padding."""
    if n < 1:
        raise ValueError("ngram size must be a positive safe integer")
    if len(tokens) < n:
        raise ValueError("ngram size cannot be larger than the number of tokens available")
    _validate_ngram_materialization(tokens, n)
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def f_measure(p: float, r: float, beta: float = 1.0) -> float:
    if not math.isfinite(p) or p < 0 or p > 1:
        raise ValueError("Precision value p must have bounds 0 ≤ p ≤ 1")
    if not math.isfinite(r) or r < 0 or r > 1:
        raise ValueError("Recall value r must have bounds 0 ≤ r ≤ 1")
    if beta < 0 or (not math.isfinite(beta) and beta != math.inf):
        raise ValueError("beta must be a non-negative number or Infinity")
    if p == r:
        return p
    if beta == math.inf:
        return r
    if p == 0 or r == 0:
        return 0
    if beta == 0:
        return p
    if beta < 1:
        return f_measure(r, p, 1 / beta)
    if p > r:
        inverse_beta = 1 / beta
        weight = inverse_beta * inverse_beta
        return min(p, r * ((1 + weight) / (1 + weight * (r / p))))
    beta_sq = beta * beta
    if math.isfinite(beta_sq):
        return min(r, p * ((1 + beta_sq) / (1 + beta_sq * (p / r))))
    ratio = ((p * beta) / r) * beta
    return r if ratio == math.inf else r * (ratio / (1 + ratio))


# lcs.ts: divide-and-conquer LCS with the legacy backtracking tie preference.


def _prefix_lengths(
    a: list[str], a_start: int, a_end: int, b: list[str], b_start: int, b_end: int
) -> list[int]:
    width = b_end - b_start
    previous = [0] * (width + 1)
    current = [0] * (width + 1)
    for i in range(a_start, a_end):
        ai = a[i]
        for j in range(1, width + 1):
            if ai == b[b_start + j - 1]:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = max(previous[j], current[j - 1])
        previous, current = current, previous
    return previous


def _find_split(
    a: list[str], a_start: int, a_middle: int, a_end: int, b: list[str], b_start: int, b_end: int
) -> int:
    width = b_end - b_start
    prev_len = _prefix_lengths(a, a_start, a_middle, b, b_start, b_end)
    cur_len = [0] * (width + 1)
    prev_org = list(range(width + 1))
    cur_org = [0] * (width + 1)
    for i in range(a_middle, a_end):
        ai = a[i]
        for j in range(1, width + 1):
            if ai == b[b_start + j - 1]:
                cur_len[j] = prev_len[j - 1] + 1
                cur_org[j] = prev_org[j - 1]
            elif prev_len[j] > cur_len[j - 1]:
                cur_len[j] = prev_len[j]
                cur_org[j] = prev_org[j]
            else:
                cur_len[j] = cur_len[j - 1]
                cur_org[j] = cur_org[j - 1]
        prev_len, cur_len = cur_len, prev_len
        prev_org, cur_org = cur_org, prev_org
    return prev_org[width]


def _collect_lcs_indices(
    a: list[str],
    a_start: int,
    a_end: int,
    b: list[str],
    b_start: int,
    b_end: int,
    indices: list[int],
) -> None:
    if a_start == a_end or b_start == b_end:
        return
    if a_end - a_start == 1:
        for j in range(b_end - 1, b_start - 1, -1):
            if a[a_start] == b[j]:
                indices.append(j)
                return
        return
    a_middle = a_start + (a_end - a_start) // 2
    b_middle = b_start + _find_split(a, a_start, a_middle, a_end, b, b_start, b_end)
    _collect_lcs_indices(a, a_start, a_middle, b, b_start, b_middle, indices)
    _collect_lcs_indices(a, a_middle, a_end, b, b_middle, b_end, indices)


def lcs_indices(a: list[str], b: list[str]) -> list[int]:
    indices: list[int] = []
    _collect_lcs_indices(a, 0, len(a), b, 0, len(b), indices)
    return indices


# --------------------------------------------------------------------------------------------
# js-rouge@3.2.3: sentenceSegment (core path, caseNeutral mode)
# --------------------------------------------------------------------------------------------
#
# Ported: terminal scanning with closing delimiters, straight double-quote pairing (with its
# recovery rule) and bracket context; unspaced boundaries ("end.Next", hostnames, dotted
# identifiers, initials, e-mail); the abbreviation / acronym / ellipsis / line-break merge loop
# with the case-neutral continuation heuristics; and list detection (findListCandidate,
# nextListMarker, segmentList) without its quote-flag and angle-bracket refinements.
# Not ported: single, typographic (curly, German, guillemet) and nested quotation pairing,
# citations, inline asides, dialogue attribution, colon-introduced author initials and the
# ellipsis-specific scanner. Against Node, segmentation agrees on 3998/4000 paragraphs of
# promptfoo's docs and ~98.5% of a synthetic corpus dense with abbreviations, initials and
# quotes. A divergence changes ROUGE-L/ROUGE-S scores only (ROUGE-N never segments).

_HONORIFICS = [
    "jr",
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "sen",
    "corp",
    "rep",
    "gov",
    "atty",
    "supt",
    "det",
    "rev",
    "col",
    "gen",
    "lt",
    "cmdr",
    "adm",
    "capt",
    "sgt",
    "cpl",
    "maj",
    "miss",
    "misses",
    "mister",
    "sir",
    "esq",
    "mstr",
    "phd",
    "adj",
    "adv",
    "asst",
    "bldg",
    "brig",
    "comdr",
    "hon",
    "messrs",
    "mlle",
    "mme",
    "op",
    "ord",
    "pvt",
    "reps",
    "res",
    "sens",
    "sfc",
    "surg",
]
_ABBR_COMMON = [
    "arc",
    "al",
    "exp",
    "rd",
    "st",
    "dist",
    "mt",
    "fy",
    "pd",
    "pl",
    "plz",
    "tce",
    "llb",
    "md",
    "bl",
    "ma",
    "ba",
    "lit",
    "ex",
    "e.g",
    "i.e",
    "circa",
    "ca",
    "cca",
    "vs",
    "v.s",
    "etc",
    "esp",
    "ft",
    "b.c",
    "a.d",
]
_ABBR_ORGANIZATIONS = [
    "co",
    "corp",
    "yahoo",
    "joomla",
    "jeopardy",
    "dept",
    "univ",
    "assn",
    "bros",
    "inc",
    "ltd",
]
_ABBR_PLACES = [
    "ala",
    "ariz",
    "ark",
    "cal",
    "calif",
    "col",
    "colo",
    "conn",
    "del",
    "fed",
    "fla",
    "fl",
    "ga",
    "ida",
    "ind",
    "ia",
    "la",
    "kan",
    "kans",
    "ken",
    "ky",
    "la",
    "md",
    "mich",
    "minn",
    "mont",
    "neb",
    "nebr",
    "nev",
    "okla",
    "penna",
    "penn",
    "pa",
    "dak",
    "tenn",
    "tex",
    "ut",
    "vt",
    "va",
    "wash",
    "wis",
    "wisc",
    "wy",
    "wyo",
    "usafa",
    "alta",
    "ont",
    "que",
    "sask",
    "yuk",
    "ave",
    "blvd",
    "cl",
    "ct",
    "cres",
    "hwy",
    "U.S",
    "U.S.A",
    "E.U",
    "N°",
]
_ABBR_TIME = ["a.m", "p.m"]
_ABBR_DATES = ["jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"]
_GATE_EXCEPTIONS = ["ex", "e.g", "i.e", "circa", "ca", "cca", "vs", "v.s", "esp", "ft", "st", "mt"]
_GATE_EXCEPTIONS += _HONORIFICS
_GATE_SUBSTITUTIONS = (
    _ABBR_COMMON + _ABBR_DATES + _ABBR_ORGANIZATIONS + _ABBR_PLACES + _ABBR_TIME + _HONORIFICS
)
_SENTENCE_SUFFIX_LENGTH = max(10, *(len(w) + 2 for w in _GATE_SUBSTITUTIONS))

# JS \b without the u flag is an ASCII word boundary.
_B = r"(?:(?<![A-Za-z0-9_])(?=[A-Za-z0-9_])|(?<=[A-Za-z0-9_])(?![A-Za-z0-9_]))"
_ABBRV_RE = re.compile(
    _B
    + r"(?!v\.?s[!?] ?\Z)("
    + "|".join(re.escape(w) for w in _GATE_SUBSTITUTIONS)
    + r")[.!?] ?\Z",
    re.IGNORECASE | re.ASCII,
)
_EXCEP_RE = re.compile(
    _B + "(" + "|".join(re.escape(w) for w in _GATE_EXCEPTIONS) + r")[.!?] ?\Z",
    re.IGNORECASE | re.ASCII,
)
_VS_RE = re.compile(_B + r"v\.?s\.\Z", re.IGNORECASE | re.ASCII)
_VS_BE_RE = re.compile(
    _B + f"(?:am|is|are|was|were|be|been|being)[{_WS}]+v\\.?s\\.\\Z", re.IGNORECASE | re.ASCII
)
_GEO_ACRONYM_RE = re.compile(_B + r"U\.S(?:\.A)?\.\Z", re.IGNORECASE | re.ASCII)
_GEO_CONTINUATION_RE = re.compile(
    r"(?:government|army|navy|military|congress|senate|commission)" + _B,
    re.IGNORECASE | re.ASCII,
)
_CONTINUATION_RE = re.compile(
    r"(?:and|or|but|nor|for|yet|so|at|in|on|of|to|from|with|by|as|then|because|while|after|"
    r"before|although|though|since|unless|until|when|where|whether|if|once|whereas)" + _B,
    re.IGNORECASE | re.ASCII,
)
_BREAK_RE = re.compile(r"[\r\n]+")
_ELLIPSE_RE = re.compile(r"\.{2,10}\Z")
_CLOSING_DELIMITERS = "])}>\"'\u201d\u00bb\u203a"
_CLOSING_BRACKETS = "])}>"
_OPENING_BRACKETS = "([{<"


@cache
def _independent_re() -> re.Pattern[str]:
    letter = _uclass("L")
    mark = _uclass("M")
    ws = _WS
    word = f"[{letter}{mark}'\u2019-]+"
    return re.compile(
        f"(?:in[{ws}]+(?:fact|time){_B}|[{letter}]+[{ws}]+[^,.!?]{{1,120}},|"
        f"(?:and|but|or|yet|so|then)[{ws}]+(?:(?:i|we|he|she|they|you|it){_B}|"
        f"(?:(?:the|a|an|my|our|their|his|her)[{ws}]+)?(?!(?:more|later|moved){_B})"
        f"{word}[{ws}]+{word}{_B}))",
        re.IGNORECASE,
    )


@cache
def _numeric_start_re() -> re.Pattern[str]:
    sc, num = _uclass("Sc"), _uclass("N")
    return re.compile(f"(?:[+\u2212-]?[{sc}]?|[{sc}][+\u2212-]?)\\.?[{num}]")


@cache
def _numeric_continuation_re() -> re.Pattern[str]:
    s = f"[^{_WS}]"
    ws = f"[{_WS}]"
    return re.compile(
        f"{s}+(?:{ws}*%|{ws}+(?:time|year)s?{_B}|{ws}+(?:month|week|day|hour|minute|second|star|"
        f"point|percent)s?(?={ws}*[.!?](?:{ws}|\\Z)|{ws}*\\Z))",
        re.IGNORECASE,
    )


@cache
def _ellipsis_quantity_re() -> re.Pattern[str]:
    s = f"[^{_WS}]"
    ws = f"[{_WS}]"
    close = "\\])}>\"'\u201d\u2019\u00bb\u201c\u203a"
    return re.compile(
        f"{s}+(?:{ws}*%|{ws}+(?:time|year|month|week|day|hour|minute|second|star|point|percent)"
        f"s?)(?={ws}*[.!?][{close}]*(?:{ws}|\\Z)|{ws}*[{close}]*\\Z)",
        re.IGNORECASE,
    )


@cache
def _page_number_re() -> re.Pattern[str]:
    num = _uclass("N")
    ws = f"[{_WS}]"
    return re.compile(
        f"{ws}*(?:\\({ws}*[{num}]+{ws}*\\)|#{ws}*[{num}]+|[{num}]+)(?={ws}|[.,;:!?)]|\\Z)"
    )


def _matches_neutral_acronym(last_word: str) -> bool:
    """caseNeutralAcronymReg: /(?:^|[ |.])\\p{Cased}\\p{M}*.?$/u."""
    n = len(last_word)
    for q in range(n):
        if q > 0 and last_word[q - 1] not in " |.":
            continue
        if not _is_cased(last_word[q]):
            continue
        r = q + 1
        while r < n and _cat(last_word[r]).startswith("M"):
            r += 1
        if r == n or (r == n - 1 and last_word[r] not in "\n\r\u2028\u2029"):
            return True
    return False


def _is_abbreviation_exception(
    suffix: str, following: str, ends_delimited_sentence: bool = False
) -> bool:
    continuation = _js_trim_start(following)
    return _EXCEP_RE.search(suffix) is not None and not (
        _VS_RE.search(suffix) is not None
        and (
            ends_delimited_sentence
            or _VS_BE_RE.search(suffix) is not None
            or _is_standalone_pronoun(continuation)
        )
    )


def _is_standalone_pronoun(s: str) -> bool:
    m = re.match(r"(?:this|that|these|those|it|we|they|he|she|i)", s, re.IGNORECASE | re.ASCII)
    if m is None:
        return False
    nxt = _char(s, m.end())
    return not (nxt in ("\u200c", "\u200d") or _is_id_continue(nxt) or _cat(nxt) == "Pd")


def _neutral_start_ok(sentence: str) -> bool:
    """caseNeutral sentence-start test used by the abbreviation and line-break rules."""
    return _starts_with_cased(sentence) and (
        _CONTINUATION_RE.match(sentence) is None or _independent_re().match(sentence) is not None
    )


def _opening_delimiter_end(s: str, start: int) -> int:
    nxt = start
    while nxt < len(s):
        if s[nxt] in "\"'\u201c\u2018\u201e\u00ab\u2039([{<" or _is_ws(s[nxt]):
            nxt += 1
        elif s.startswith("``", nxt):
            nxt += 2
        else:
            break
    return nxt


def _is_neutral_sentence_start(s: str, previous_end: int, nxt: int, quoted_start: bool) -> bool:
    if not _is_cased(_char(s, nxt)):
        return False
    continuation = s[nxt:]
    return (
        quoted_start
        or _CONTINUATION_RE.match(continuation) is None
        or _independent_re().match(continuation) is not None
        or '"' in s[previous_end:nxt]
    )


def _is_numeric_sentence_start(s: str, nxt: int, suffix: str) -> bool:
    return (
        _cat(_char(s, nxt)).startswith("N")
        and _ABBRV_RE.search(suffix) is None
        and _numeric_continuation_re().match(s[nxt:]) is None
    )


def _is_quote_opening_content(ch: str) -> bool:
    return bool(ch) and (_cat(ch)[:1] in ("L", "M", "N", "S") or _cat(ch) == "Ps"
                         or ch in "'\u2018\u201c\u201e\u00ab\u2039`")  # fmt: skip


def _straight_quote_roles(s: str) -> dict[int, bool]:
    """QuotationPairing for straight double quotes: index -> True (opener) / False (closer).
    Unpaired quotes get no role."""
    roles: dict[int, bool] = {}
    opener = -1
    cursor = 0
    preceding_content = ""
    preceding_paired = False

    def recovery(index: int) -> str | None:
        nonlocal cursor, preceding_content, preceding_paired
        if not (
            _quote_opening_context(s, index) and _is_quote_opening_content(_char(s, index + 1))
        ):
            return None
        while cursor < index:
            position = cursor
            cursor += 1
            if roles.get(position) is False:
                preceding_paired = True
            elif not (
                _is_ws(s[position]) or s[position] in "\"'\u201d\u2019\u201c\u00bb\u203a])}>"
            ):
                preceding_content = s[position]
                preceding_paired = False
        if preceding_paired:
            return "close"
        return None if preceding_content in (".", "!", "?") else "open"

    for index, ch in enumerate(s):
        if ch != '"':
            continue
        if opener >= 0 and recovery(index) != "open":
            roles[opener] = True
            roles[index] = False
            opener = -1
        elif _quote_opening_context(s, index):
            opener = index
    return roles


_HOSTNAME_LABEL_RE = re.compile(
    "(com|org|net|edu|gov|mil|io|dev|app|co|uk|us|ca|ai|info|biz|me|tv)"
    f"(?=[/.!?{_WS}\"'\u201c\u201d\u2018\u2019()\\[\\]{{}}<>]|\\Z)",
    re.IGNORECASE,
)
_URL_CONTEXT_RE = re.compile(f"[{_WS}]|https?://|www\\.", re.IGNORECASE)


def _inside_url(s: str, end: int) -> bool:
    inside = False
    for m in _URL_CONTEXT_RE.finditer(s):
        if m.end() > end:
            break
        inside = not _is_ws(m.group(0)[0])
    return inside


def _has_following_at(s: str, nxt: int) -> bool:
    end = nxt
    while end < len(s) and not _is_ws(s[end]):
        if s[end] == "@":
            return True
        end += 1
    return False


def _is_identifier_char(ch: str) -> bool:
    return bool(ch) and (_cat(ch)[:1] in ("L", "M", "N") or ch in "_-")


def _preceding_identifier_token(s: str, index: int) -> str:
    start = index
    while start > 0 and _is_identifier_char(s[start - 1]):
        start -= 1
    return s[start:index]


def _has_ascii_identifier_evidence(token: str) -> bool:
    for i, ch in enumerate(token):
        code = ord(ch)
        if 48 <= code <= 57 or code == 95:
            return True
        lower = 107 if code == 0x212A else code | 32
        if lower < 97 or lower > 122:
            continue
        if _cat(_char(token, i + 1)).startswith("M"):
            continue
        return True
    return False


def _is_dotted_identifier_continuation(following: str) -> bool:
    cursor = 0
    for _ in range(2):
        base = _char(following, cursor)
        if not base or not (_is_cased(base) or _cat(base).startswith("N") or base in "_-"):
            return False
        cursor += 1
        nxt = _char(following, cursor)
        if base == "\u0130" or nxt == "\u0307":
            while nxt and _cat(nxt).startswith("M"):
                cursor += 1
                nxt = _char(following, cursor)
        if nxt == "" or _is_ws(nxt) or nxt in "/.":
            return True
    return False


def _has_trailing_initial(s: str, index: int, token_length: int) -> bool:
    start = max(0, index - token_length - 2)
    window = s[start:index]
    k = len(window)
    while k > 0 and _cat(window[k - 1]).startswith("M"):
        k -= 1
    if k == 0 or not _is_cased(window[k - 1]) or _cat(window[k - 1]).startswith("M"):
        return False
    return k - 1 == 0 or not _is_identifier_char(window[k - 2])


def _is_unspaced_delimited_sentence_start(s: str, index: int) -> bool:
    start = index + 1
    nxt = start
    while nxt < len(s):
        if s[nxt] in "\"'\u201c\u2018\u201e\u00ab\u2039([{<":
            nxt += 1
        elif s.startswith("``", nxt):
            nxt += 2
        else:
            break
    if nxt == start:
        return False
    if s.startswith("``", start):
        suffix = s[max(0, start - _SENTENCE_SUFFIX_LENGTH) : start]
        if _ABBRV_RE.search(suffix.lower()):
            return False
    if re.match(r"(?:\[\d+\]|\(\d+\))", s[start:]):
        return False
    ch = _char(s, nxt)
    return bool(ch) and (_cat(ch).startswith("N") or _is_cased(ch))


def _is_unspaced_sentence_boundary(s: str, index: int, nxt: int) -> bool:
    if _is_unspaced_delimited_sentence_start(s, nxt - 1):
        return True
    next_char = _char(s, nxt)
    if not (_is_cased(next_char) or (s[index] != "." and _cat(next_char).startswith("N"))):
        return False
    m = re.search(f"[^{_WS}]+\\Z", s[max(0, index - 320) : nxt])
    preceding = m.group(0) if m else ""
    inside_url = _inside_url(s, nxt)
    if s[index] != ".":
        return not inside_url
    suffix = s[max(0, index + 1 - _SENTENCE_SUFFIX_LENGTH) : index + 1]
    token = _preceding_identifier_token(s, index)
    following = s[nxt:]
    inside_address = "@" in preceding and not re.search(
        f"@[^{_WS}.]+(?:\\.[^{_WS}.]+)+\\.\\Z", preceding
    )
    inside_hostname = inside_url or _HOSTNAME_LABEL_RE.match(following) is not None
    dotted = (
        any(_is_cased(c) for c in token)
        and _has_ascii_identifier_evidence(token)
        and _is_dotted_identifier_continuation(following)
    )
    initial = False
    if following and _is_cased(following[0]):
        j = 1
        while j < len(following) and _cat(following[j]).startswith("M"):
            j += 1
        initial = _char(following, j) == "."
    next_initial = False
    if (
        following
        and _is_cased(following[0])
        and not re.match(f"[ai](?:[{_WS}]|\\Z)", following, re.I)
    ):
        j = 1
        while j < len(following) and _cat(following[j]).startswith("M"):
            j += 1
        next_initial = j == len(following) or _is_ws(following[j])
    gate = suffix.lower()
    continues_abbreviation = _ABBRV_RE.search(gate) is not None and (
        _is_abbreviation_exception(gate, following)
        or (
            _GEO_ACRONYM_RE.search(gate) is not None
            and re.match(r"(?:government|army|navy|military|congress)" + _B, following, re.I | re.A)
            is not None
        )
    )
    return not (
        continues_abbreviation
        or initial
        or dotted
        or (_has_trailing_initial(s, index, len(token)) and next_initial)
        or inside_address
        or inside_hostname
        or _has_following_at(s, nxt)
    )


def _closing_delimiter_end(s: str, index: int, inside_quotes: bool, roles: dict[int, bool]) -> int:
    end = index + 1
    pending = inside_quotes
    while end < len(s):
        ch = s[end]
        if roles.get(end) is True:
            break
        if (
            inside_quotes
            and ch in "\"'\u201d\u2019\u201c\u00bb\u203a"
            and not (ch == '"' and pending)
        ):
            break
        if ch in _CLOSING_DELIMITERS:
            if ch == '"':
                pending = False
            end += 1
            continue
        nxt = end
        while nxt < len(s) and _is_ws(s[nxt]):
            nxt += 1
        if (
            nxt > end
            and nxt < len(s)
            and (s[nxt] in _CLOSING_BRACKETS or (pending and roles.get(nxt) is False))
        ):
            end = nxt
            continue
        break
    return end


def _sentence_end(
    s: str, index: int, inside_quotes: bool, depth: int, standalone: bool, roles: dict[int, bool]
) -> int:
    end = _closing_delimiter_end(s, index, inside_quotes, roles)
    if end < len(s) and not _is_ws(s[end]):
        if inside_quotes and not any(roles.get(i) is False for i in range(index + 1, end)):
            return -1
        return end if _is_unspaced_sentence_boundary(s, index, end) else -1
    if end == index + 1:
        return -1 if inside_quotes and _char(s, end) not in ("\r", "\n") else end
    closed = sum(1 for ch in s[index + 1 : end] if ch in _CLOSING_BRACKETS)
    closes_quotation = inside_quotes and s[end - 1] == '"'
    if closed > 0 and (
        closed < depth or not (standalone or closes_quotation or _js_trim(s[end:]) == "")
    ):
        return -1
    nxt = _opening_delimiter_end(s, end)
    if nxt == len(s):
        return end
    suffix = s[max(0, index + 1 - _SENTENCE_SUFFIX_LENGTH) : index + 1]
    gate = suffix.lower()
    quoted = re.search("[\"'\u201c\u2018]", s[end:nxt]) is not None
    letter = _is_neutral_sentence_start(s, end, nxt, quoted)
    number = _is_numeric_sentence_start(s, nxt, gate)
    if not (letter or number):
        return -1
    if _ELLIPSE_RE.search(suffix) and closed > 0:
        return -1
    if _ABBRV_RE.search(gate) and _is_abbreviation_exception(gate, s[nxt:]):
        return -1
    return end


def _sentence_chunks(s: str) -> tuple[list[str], list[int]]:
    chunks: list[str] = []
    starts: list[int] = []
    roles = _straight_quote_roles(s)
    last_end = 0
    start = -1
    inside_quotes = False
    depth = 0
    standalone = False
    for index, ch in enumerate(s):
        if index in roles:
            inside_quotes = roles[index]
        if ch in _OPENING_BRACKETS:
            if depth == 0:
                standalone = start == -1
            depth += 1
        elif ch in _CLOSING_BRACKETS:
            depth = max(0, depth - 1)
        if index < last_end or ch in ("\r", "\n"):
            start = -1
            continue
        if start == -1:
            if not _is_ws(ch):
                start = index
            continue
        if ch not in ".?!":
            continue
        terminal_ellipsis = re.search(r"\.{3,4}\Z", s[max(0, index - 3) : index + 1]) is not None
        if (
            not terminal_ellipsis
            and not inside_quotes
            and depth == 0
            and _is_unspaced_delimited_sentence_start(s, index)
        ):
            end = index + 1
        else:
            end = _sentence_end(s, index, inside_quotes, depth, standalone, roles)
        if end == -1:
            continue
        starts += [last_end, start]
        chunks += [s[last_end:start], _BREAK_RE.sub(" ", s[start:end])]
        last_end = end
        start = -1
    starts.append(last_end)
    chunks.append(s[last_end:])
    return chunks, starts


def _trim_spaces(s: str) -> str:
    return s.strip(" ")


class _SentenceBuffer:
    """SentenceBuffer in caseNeutral mode (word casing is ignored there)."""

    def __init__(self, text: str) -> None:
        self.parts: list[str] = []
        self.normalized_through = 0
        self.word_count = 0
        self.open_brackets: list[str] = []
        self.in_quote = False
        self.has_line_breaks = False
        self.starts_with_title_case = False
        self.append(_trim_spaces(text))

    @property
    def empty(self) -> bool:
        return self.word_count == 0

    @property
    def has_open_delimiter(self) -> bool:
        return bool(self.open_brackets) or self.in_quote

    @property
    def suffix(self) -> str:
        suffix = ""
        for part in reversed(self.parts):
            if len(suffix) >= _SENTENCE_SUFFIX_LENGTH:
                break
            suffix = _js_slice(part, len(suffix) - _SENTENCE_SUFFIX_LENGTH) + suffix
        return suffix

    def append(self, text: str) -> None:
        if not text:
            return
        for ch in text:
            if ch == '"':
                self.in_quote = not self.in_quote
            elif not self.in_quote and ch in _OPENING_BRACKETS:
                self.open_brackets.append(ch)
            elif not self.in_quote and ch in _CLOSING_BRACKETS:
                opener = _OPENING_BRACKETS[")]}>".index(ch)]
                if self.open_brackets and self.open_brackets[-1] == opener:
                    self.open_brackets.pop()
        last_part = self.parts[-1] if self.parts else ""
        for m in re.finditer(f"[^{_WS}]+", text):
            if m.start() == 0 and self.word_count and not (last_part and _is_ws(last_part[-1])):
                continue
            if self.empty:
                self.starts_with_title_case = _starts_with_cased(m.group(0))
            self.word_count += 1
        self.has_line_breaks = self.has_line_breaks or _BREAK_RE.search(text) is not None
        self.parts.append(text)

    def trim_end(self) -> None:
        while self.parts:
            trimmed = self.parts[-1].rstrip(" ")
            if trimmed:
                self.parts[-1] = trimmed
                break
            self.parts.pop()
        self.normalized_through = min(self.normalized_through, len(self.parts))

    def normalize_whitespace(self) -> None:
        write = self.normalized_through
        for read in range(write, len(self.parts)):
            part = _WS_RUN.sub(" ", self.parts[read])
            if (write == 0 or self.parts[write - 1].endswith(" ")) and part.startswith(" "):
                part = part[1:]
            if part:
                self.parts[write] = part
                write += 1
        del self.parts[write:]
        self.trim_end()
        self.normalized_through = len(self.parts)
        self.has_line_breaks = False

    def text(self) -> str:
        return "".join(self.parts)


def _is_terminal_ellipsis_boundary(following: str, suffix: str, has_open: bool) -> bool:
    sentence_start = following[_opening_delimiter_end(following, 0) :]
    first = _char(sentence_start, 0)
    starts_letter = _is_cased(first) and (
        re.search(r"\.{4}\Z", suffix) is not None
        or (
            re.match(r"(?:what|and[" + _WS + r"]+then)" + _B, sentence_start, re.I | re.A) is None
            and (
                _CONTINUATION_RE.match(sentence_start) is None
                or _independent_re().match(sentence_start) is not None
            )
        )
    )
    starts_number = (
        _numeric_start_re().match(sentence_start) is not None
        and _ellipsis_quantity_re().match(sentence_start) is None
    )
    return (
        re.search(r"\.{3,4}\Z", suffix) is not None
        and (starts_letter or starts_number)
        and (
            re.search(r"\.{4}\Z", suffix) is not None
            or not (
                has_open
                or re.search(
                    _B + r"(?:am|is|are|was|were|be|been|being|i)\.{3}\Z", suffix, re.I | re.A
                )
            )
        )
    )


_MAX_NESTED_LIST_DEPTH = 32
_YEAR_MARKER_RE = re.compile(r"(?:1\d{3}|20\d{2})\.\Z")
_COUNT_IN_PROSE_RE = re.compile(
    _B + r"(?:am|is|are|was|were|be|been|being|has|have|had|reached|numbered|total(?:ed)?|hit|"
    r"equals?|equaled|became|remained|scored|costs?|in|of|at|by|to|from|about|around|roughly|"
    r"approximately)\Z",
    re.IGNORECASE | re.ASCII,
)
_CROSS_REFERENCE_RE = re.compile(
    _B + r"(?:section|chapter|page|figure|table|paragraph|article|clause)s?\Z",
    re.IGNORECASE | re.ASCII,
)
_JOINED_INITIAL_RE = re.compile(f"(?:{_B}(?:and|or)|[,;&])[{_WS}]*\\Z", re.IGNORECASE | re.ASCII)
_ACRONYM_RE = re.compile(r"[ |.][A-Z].?\Z", re.IGNORECASE | re.ASCII)


@cache
def _list_marker_re() -> re.Pattern[str]:
    cased, mark = _uclass("Lu Ll Lt"), _uclass("M")
    nb = f"(?:(?![\\r\\n])[{_WS}])"
    return re.compile(
        f"(?:^|[{_WS}])(?:(?:[\u2022\u2043]{nb}*|[-*+]{nb}+)?\\d+|[{cased}][{mark}]*)"
        f"(?:\\.\\)|[.)])(?=[{_WS}]+[^{_WS}])"
    )


def _is_initial_marker(marker: str) -> bool:
    if len(marker) < 2 or marker[-1] != "." or not _is_cased(marker[0]):
        return False
    return all(_cat(c).startswith("M") for c in marker[1:-1])


def _marker_family(marker: str) -> tuple[bool, str]:
    return (re.search(r"\d", marker) is not None, ")" if marker.endswith(")") else ".")


def _in_family(marker_text: str, family: tuple[bool, str]) -> bool:
    numeric, ending = family
    text = _js_trim_start(marker_text)
    if not text.endswith(ending) or re.search(r"[\r\n]", text):
        return False
    if numeric:
        return (
            re.match(
                r"(?:[\u2022\u2043](?:(?![\r\n])["
                + _WS
                + r"])*|[-*+](?:(?![\r\n])["
                + _WS
                + r"])+)?\d",
                text,
            )
            is not None
        )
    return _is_cased(text[:1])


def _numeric_marker_value(marker: str) -> str | None:
    m = re.match(r"(?:[\u2022\u2043][^\S\r\n]*|[-*+][^\S\r\n]+)?(\d+)", marker)
    return re.sub(r"^0+(?=\d)", "", m.group(1)) if m else None


def _is_distant_numeric_marker(first: str | None, marker: str, at_boundary: bool) -> bool:
    current = _numeric_marker_value(marker)
    if at_boundary or first is None or current is None:
        return False
    if abs(len(first) - len(current)) > 1:
        return True
    difference = 0
    length = max(len(first), len(current))
    for index in range(length):
        fi = index - (length - len(first))
        ci = index - (length - len(current))
        fd = int(first[fi]) if 0 <= fi < len(first) else 0
        cd = int(current[ci]) if 0 <= ci < len(current) else 0
        difference = difference * 10 + fd - cd
        if abs(difference) > 10:
            return True
    return False


class _ListScan:
    """ListScanState: bracket/quote enclosure and cross-reference context, advanced by cursor."""

    def __init__(self, s: str) -> None:
        self.s = s
        self.roles = _straight_quote_roles(s)
        self.cursor = 0
        self.stack: list[int] = []
        self.quote_end = -1
        self.reference_through: int | None = None
        self.reference_boundary: int | None = None
        self.year_list = False

    def copy(self) -> _ListScan:
        other = _ListScan.__new__(_ListScan)
        other.__dict__.update(self.__dict__)
        other.stack = list(self.stack)
        return other

    def advance(self, end: int) -> None:
        s = self.s
        while self.cursor < end:
            index = self.cursor
            self.cursor += 1
            if self.quote_end >= 0:
                if index >= self.quote_end:
                    self.quote_end = -1
                continue
            if self.roles.get(index) is True:
                closer = next(i for i in range(index + 1, len(s)) if self.roles.get(i) is False)
                self.quote_end = closer
                continue
            ch = s[index]
            if ch in "([{":
                self.stack.append("([{".index(ch))
            elif ch in ")]}":
                closing = ")]}".index(ch)
                if self.stack and self.stack[-1] == closing:
                    self.stack.pop()
            if (
                self.reference_through is not None
                and not self.stack
                and _is_list_reference_boundary(s, index)
            ):
                self.reference_boundary = index

    @property
    def enclosed(self) -> bool:
        return bool(self.stack) or self.quote_end >= 0


def _is_list_reference_boundary(s: str, index: int) -> bool:
    terminal = s[index]
    if terminal not in ".!?:;\r\n":
        return False
    if terminal != ".":
        return True
    gate = s[max(0, index + 1 - _SENTENCE_SUFFIX_LENGTH) : index + 1].lower()
    return not (
        (_ABBRV_RE.search(gate) and _EXCEP_RE.search(gate))
        or _ACRONYM_RE.search(gate)
        or (_char(s, index - 1).isdigit() and _char(s, index + 1).isdigit())
    )


def _marker_prefix(s: str, m: re.Match[str]) -> tuple[bool, bool, bool]:
    """listMarkerPrefix -> (empty, boundary, joinedNameInitial)."""
    index = m.start() - 1
    line_break = re.search(r"[\r\n]", m.group(0)) is not None
    while index >= 0 and (_is_ws(s[index]) or s[index] in "\"'\u201d\u2019\u00bb\u203a])}>"):
        line_break = line_break or s[index] in "\r\n"
        index -= 1
    prev = _char(s, index)
    boundary = line_break or (prev != "" and (prev in ".!?:" or _cat(prev) == "Pd"))
    joined = _is_initial_marker(_js_trim(m.group(0))) and (
        _JOINED_INITIAL_RE.search(s[max(0, m.start() - 8) : m.start()]) is not None
    )
    return index < 0, boundary, joined


class _ListMarkers:
    def __init__(self, s: str) -> None:
        self.s = s
        self.matches = list(_list_marker_re().finditer(s))
        self.pos = 0

    def next_marker(
        self,
        state: _ListScan,
        family: tuple[bool, str] | None = None,
        previous: re.Match[str] | None = None,
    ) -> re.Match[str] | None:
        s = self.s
        body_start = previous.end() if previous else 0
        has_body = previous is None
        while self.pos < len(self.matches):
            m = self.matches[self.pos]
            self.pos += 1
            state.advance(m.start())
            in_family = family is None or _in_family(m.group(0), family)
            if in_family:
                has_body = has_body or _js_trim(s[body_start : m.start()]) != ""
                body_start = m.end()
            enclosed = state.enclosed
            empty, boundary, joined = _marker_prefix(s, m)
            marker = _js_trim(m.group(0))
            # isListCrossReference
            if (
                state.reference_through is not None
                and (state.reference_boundary if state.reference_boundary is not None else -1)
                >= state.reference_through
            ):
                state.reference_through = None
            before = _js_trim_end(s[max(0, m.start() - 24) : m.start()])
            if not enclosed and _CROSS_REFERENCE_RE.search(before):
                state.reference_through = m.start()
            cross = False
            if state.reference_through is not None:
                state.reference_through = m.end()
                cross = True
            author_initial = previous is not None and _is_initial_marker(marker) and joined
            year = _YEAR_MARKER_RE.match(marker) is not None and not (
                state.year_list or empty or boundary
            )
            count = re.fullmatch(r"\d+\.", marker) is not None and (
                _COUNT_IN_PROSE_RE.search(before) is not None
            )
            if (
                in_family
                and has_body
                and not (enclosed or year or count or cross or author_initial)
            ):
                return m
        return None


def _find_list_candidate(
    s: str, markers: _ListMarkers, state: _ListScan
) -> tuple[re.Match[str], re.Match[str], tuple[bool, str], str, _ListScan] | None:
    first_by_family: dict[tuple[bool, str], dict[str, Any]] = {}
    deferred: dict[tuple[bool, str], tuple[Any, int, _ListScan]] = {}
    current = markers.next_marker(state)
    prose_initial_end: int | None = None
    while current is not None:
        marker = _js_trim(current.group(0))
        empty, boundary, joined = _marker_prefix(s, current)
        ambiguous = re.fullmatch(r"\d+\.", marker) is not None or _is_initial_marker(marker)
        family = _marker_family(marker)
        first = first_by_family.get(family)
        # markProseInitial
        if (
            first is None
            and _is_initial_marker(marker)
            and (
                not (empty or boundary)
                or (
                    prose_initial_end is not None
                    and _js_trim(s[prose_initial_end : current.start()]) == ""
                )
            )
        ):
            boundary = False
            prose_initial_end = current.end()
        else:
            prose_initial_end = None
        identity = marker.lower()
        if joined:
            first_by_family.pop(family, None)
            deferred.pop(family, None)
            current = markers.next_marker(state)
            continue
        if first is not None:
            first["has_body"] = (
                first["has_body"] or _js_trim(s[first["body_start"] : current.start()]) != ""
            )
            first["body_start"] = current.end()
        distinct = first is None or first["identity"] != identity
        if first is not None and first["has_body"] and (first["empty"] or distinct):
            if first.get("prefix") is None:
                first["prefix"] = _js_trim(s[: first["marker"].start()])
            candidate = (first["marker"], current, family, first["prefix"])
            earlier = any(
                e["marker"].start() < first["marker"].start() for e in first_by_family.values()
            )
            distant = _is_distant_numeric_marker(first["number"], marker, boundary)
            if not (earlier or distant):
                return (*candidate, state)
            if family not in deferred:
                deferred[family] = (candidate, markers.pos, state.copy())
        if first is None and (empty or not ambiguous or boundary):
            state.year_list = state.year_list or _YEAR_MARKER_RE.match(marker) is not None
            first_by_family[family] = {
                "marker": current, "empty": empty, "identity": identity,
                "number": _numeric_marker_value(marker), "body_start": current.end(),
                "has_body": False,
            }  # fmt: skip
        current = markers.next_marker(state)
    if not deferred:
        return None
    candidate, pos, saved = min(deferred.values(), key=lambda d: d[0][0].start())
    markers.pos = pos
    return (*candidate, saved)


def _segment_list(s: str, depth: int) -> list[str] | None:
    if depth >= _MAX_NESTED_LIST_DEPTH:
        return None
    markers = _ListMarkers(s)
    found = _find_list_candidate(s, markers, _ListScan(s))
    if found is None:
        return None
    current, nxt, family, prefix, state = found
    segments = _sentence_segment(prefix, depth + 1) if prefix else []
    cur: re.Match[str] | None = current
    while cur is not None:
        stop = nxt.start() if nxt is not None else len(s)
        body = _js_trim(s[cur.end() : stop])
        sentences = _segment_core(body) if re.search(r"[.!?\r\n]", body) else [body]
        first_end = 1
        while first_end < len(sentences) and _is_initial_marker(sentences[first_end - 1]):
            first_end += 1
        segments.append(f"{_js_trim(cur.group(0))} {' '.join(sentences[:first_end])}")
        segments += sentences[first_end:]
        cur = nxt
        nxt = markers.next_marker(state, family, cur) if cur is not None else None
    return segments


def _sentence_segment(input_text: str, depth: int = 0) -> list[str]:
    """js-rouge sentenceSegment(input, {caseNeutral: true}) — core path only (see above)."""
    if not input_text:
        return []
    normalized = input_text.replace("\u0085", " ")
    listed = _segment_list(normalized, depth)
    if listed is not None:
        return listed
    return _segment_core(normalized, original=input_text)


def _segment_core(normalized: str, original: str | None = None) -> list[str]:
    chunks, _starts = _sentence_chunks(normalized)
    acc: list[str] = []
    pending: _SentenceBuffer | None = None
    idx = 0
    while idx < len(chunks):
        if pending is not None or chunks[idx]:
            chunk = pending if pending is not None else _SentenceBuffer(chunks[idx])
            pending = None
            chunk.trim_end()
            if chunk.empty:
                idx += 1
                continue
            suffix = chunk.suffix
            gate = suffix.lower()
            lw = re.search(f"[^{_WS}]+\\Z", suffix)
            last_word = lw.group(0) if lw else ""
            next_chunk = chunks[idx + 1] if idx + 1 < len(chunks) else ""
            if chunk.has_line_breaks:
                next_sentence = (
                    next_chunk[_opening_delimiter_end(next_chunk, 0) :] if next_chunk else ""
                )
                abbreviation = _js_trim_end(gate)
                lead = re.match(f"[{_WS}]*", next_chunk)
                separator = suffix[len(_js_trim_end(suffix)) :] + (lead.group(0) if lead else "")
                para = re.search(r"\n[^\S\n]*\n", re.sub(r"\r\n?", "\n", separator)) is not None
                continues_abbreviation = not para and (
                    (
                        _VS_RE.search(abbreviation) is not None
                        and _is_abbreviation_exception(abbreviation, next_sentence)
                    )
                    or (
                        _GEO_ACRONYM_RE.search(abbreviation) is not None
                        and _GEO_CONTINUATION_RE.match(_js_trim_start(next_chunk)) is not None
                    )
                )
                if (
                    next_sentence
                    and _ABBRV_RE.search(abbreviation)
                    and not _is_abbreviation_exception(abbreviation, next_sentence)
                    and _neutral_start_ok(next_sentence)
                    and not chunk.has_open_delimiter
                    and not (
                        _GEO_ACRONYM_RE.search(abbreviation)
                        and _GEO_CONTINUATION_RE.match(_js_trim_start(next_chunk))
                        and not re.search(r"[\r\n][^\S\r\n]*[\r\n]", suffix.replace("\r\n", "\n"))
                    )
                ):
                    chunk.normalize_whitespace()
                    acc.append(chunk.text())
                elif next_chunk and (
                    chunk.starts_with_title_case
                    or continues_abbreviation
                    or (
                        _CONTINUATION_RE.match(_js_trim_start(next_chunk)) is not None
                        and re.search(
                            f"[.!?][\"'\u201d\u2019\u201c\u00bb\u203a\\])}}>][{_WS}]*[\\r\\n]",
                            suffix,
                        )
                        is not None
                    )
                ):
                    chunk.append(f" {next_chunk}")
                    chunk.normalize_whitespace()
                    pending = chunk
                else:
                    for line in _BREAK_RE.split(chunk.text()):
                        sentence = _js_trim(line)
                        if sentence:
                            acc.append(sentence)
            elif next_chunk and _ABBRV_RE.search(gate):
                next_sentence = (
                    re.sub("^[" + _WS + "\"'([{<]+", "", next_chunk)
                    if _VS_RE.search(gate)
                    else next_chunk
                )
                paragraph_break = (
                    re.search(r"\n[^\S\n]*\n", re.sub(r"\r\n?", "\n", next_chunk)) is not None
                )
                if (paragraph_break and _VS_RE.search(gate) and not chunk.has_open_delimiter) or (
                    _neutral_start_ok(next_sentence)
                    and not _is_abbreviation_exception(gate, next_sentence)
                    and not (
                        _GEO_ACRONYM_RE.search(gate)
                        and _GEO_CONTINUATION_RE.match(_js_trim_start(next_chunk))
                    )
                ):
                    acc.append(chunk.text())
                else:
                    chunk.append(" " + _trim_spaces(re.sub(r"[^\S\r\n]+", " ", next_chunk)))
                    pending = chunk
            elif next_chunk and _matches_neutral_acronym(last_word):
                next_sentence = chunks[idx + 2] if idx + 2 < len(chunks) else ""
                continuation = next_sentence or next_chunk
                if last_word.lower() == "p." and _page_number_re().match(continuation):
                    chunk.append(_WS_RUN.sub(" ", next_chunk) + next_sentence)
                    pending = chunk
                    if next_sentence:
                        idx += 1
                else:
                    acc.append(chunk.text())
            elif (
                next_chunk
                or (idx + 2 < len(chunks) and chunks[idx + 2] and re.search(r"\.{3,4}\Z", suffix))
            ) and _ELLIPSE_RE.search(suffix):
                following = _js_trim(next_chunk) or (
                    chunks[idx + 2] if idx + 2 < len(chunks) else ""
                )
                if _is_terminal_ellipsis_boundary(following, suffix, chunk.has_open_delimiter):
                    acc.append(chunk.text())
                    chunks[idx + 1] = re.sub(r"^[^\S\r\n]+", "", next_chunk)
                    idx += 1
                    continue
                chunk.append(re.sub(" +", " ", next_chunk))
                if (
                    not (_js_trim(next_chunk) or _BREAK_RE.search(next_chunk))
                    and idx + 2 < len(chunks)
                    and chunks[idx + 2]
                ):
                    chunk.append(re.sub(" +", " ", chunks[idx + 2]))
                    idx += 1
                pending = chunk
            else:
                acc.append(chunk.text())
        idx += 1
    if not acc:
        return [original if original is not None else normalized]
    return acc


# --------------------------------------------------------------------------------------------
# js-rouge@3.2.3: prepare.ts and the l / s scorers (caseSensitive: false)
# --------------------------------------------------------------------------------------------


def _fold_sentences(input_text: str, sentences: list[str]) -> list[str]:
    folded = re.sub(f"[{_WS}\u0085]+", "", input_text.lower())
    offset = 0

    def fold(m: re.Match[str]) -> str:
        nonlocal offset
        length = len(m.group(0).lower())
        value = folded[offset : offset + length]
        offset += length
        return value

    return [re.sub(f"[^{_WS}\u0085]+", fold, sentence) for sentence in sentences]


def _prepare_summary(input_text: str) -> list[list[str]]:
    sentences = _fold_sentences(input_text, _sentence_segment(input_text))
    return [tree_bank_tokenize(sentence) for sentence in sentences]


_MAX_LCS_SENTENCE_PAIRS = 100_000


def rouge_l(cand: str, ref: str) -> float:
    """js-rouge l(cand, ref, {caseSensitive: false})."""
    if _ROUGE_BLANK.match(cand):
        raise ValueError("Candidate cannot be an empty string")
    if _ROUGE_BLANK.match(ref):
        raise ValueError("Reference cannot be an empty string")
    candidate = _prepare_summary(cand)
    reference = _prepare_summary(ref)
    cand_length = sum(len(s) for s in candidate)
    ref_length = sum(len(s) for s in reference)
    remaining: dict[str, int] = {}
    for sentence in candidate:
        for token in sentence:
            remaining[token] = remaining.get(token, 0) + 1
    if cand_length == 0 or ref_length == 0:
        return 0
    if len(candidate) > _MAX_LCS_SENTENCE_PAIRS / len(reference):
        raise ValueError("ROUGE-L sentence comparison exceeds the work limit")
    matches = 0
    for ref_sentence in reference:
        if not remaining:
            break
        union: dict[int, None] = {}
        for cand_sentence in candidate:
            for index in lcs_indices(cand_sentence, ref_sentence):
                union[index] = None
        for index in union:
            token = ref_sentence[index]
            count = remaining.get(token, 0)
            if count > 0:
                matches += 1
                if count == 1:
                    del remaining[token]
                else:
                    remaining[token] = count - 1
    if matches == 0:
        return 0
    return f_measure(matches / cand_length, matches / ref_length, 1.0)


def _token_positions(tokens: list[str]) -> dict[str, list[int]]:
    positions: dict[str, list[int]] = {}
    for index, token in enumerate(tokens):
        positions.setdefault(token, []).append(index)
    return positions


def _count_unbounded_skip_pairs(first: list[int], second: list[int]) -> int:
    first_valid = 0
    pairs = 0
    for first_position in first:
        while first_valid < len(second) and second[first_valid] <= first_position:
            first_valid += 1
        pairs += len(second) - first_valid
    return pairs


def rouge_s(cand: str, ref: str) -> float:
    """js-rouge s(cand, ref, {caseSensitive: false}) with the default unbounded skip window."""
    if _ROUGE_BLANK.match(cand):
        raise ValueError("Candidate cannot be an empty string")
    if _ROUGE_BLANK.match(ref):
        raise ValueError("Reference cannot be an empty string")
    cand_tokens = [t for s in _prepare_summary(cand) for t in s]
    ref_tokens = [t for s in _prepare_summary(ref) for t in s]
    if len(cand_tokens) < 2 or len(ref_tokens) < 2:
        return 0
    if cand_tokens == ref_tokens:
        return 1
    cand_positions = _token_positions(cand_tokens)
    ref_positions = _token_positions(ref_tokens)
    shared = [
        (positions, ref_positions[token])
        for token, positions in cand_positions.items()
        if token in ref_positions
    ]
    skip2 = 0
    for first_cand, first_ref in shared:
        for second_cand, second_ref in shared:
            cand_count = _count_unbounded_skip_pairs(first_cand, second_cand)
            if cand_count > 0:
                skip2 += min(cand_count, _count_unbounded_skip_pairs(first_ref, second_ref))
    if skip2 == 0:
        return 0

    def bigram_count(n: int) -> float:
        distance = n - 1
        return distance * n - (distance * (distance + 1)) / 2

    return f_measure(skip2 / bigram_count(len(cand_tokens)), skip2 / bigram_count(len(ref_tokens)))


# --------------------------------------------------------------------------------------------
# rouge.ts
# --------------------------------------------------------------------------------------------

_ROUGE_BLANK = re.compile(f"[{_WS}\u0085]*\\Z")


def _rouge_n_score(candidate: str, reference: str, n: int = 1, beta: float = 1) -> float:
    candidate_tokens = tree_bank_tokenize(candidate)
    reference_tokens = tree_bank_tokenize(reference)
    if len(candidate_tokens) < n or len(reference_tokens) < n:
        return 0
    candidate_ngrams = rouge_ngram(candidate_tokens, n)
    reference_ngrams = rouge_ngram(reference_tokens, n)
    reference_counts = count_ngrams(reference_ngrams)
    overlap = 0
    for ngram, count in count_ngrams(candidate_ngrams).items():
        overlap += min(count, reference_counts.get(ngram, 0))
    if overlap == 0:
        return 0
    return f_measure(overlap / len(candidate_ngrams), overlap / len(reference_ngrams), beta)


def handle_rouge_score(params: AssertionParams) -> GradingResult:
    assertion, rendered_value, inverse = params.assertion, params.rendered_value, params.inverse
    output_string, base_type = params.output_string, params.base_type
    invariant(isinstance(rendered_value, str), '"rouge" assertion type must be a string value')
    fn_name = base_type[-1]
    score: float = 0
    if not _ROUGE_BLANK.match(output_string) and not _ROUGE_BLANK.match(rendered_value):
        if fn_name == "n":
            score = _rouge_n_score(_js_lower(output_string), _js_lower(rendered_value))
        elif fn_name == "l":
            score = rouge_l(output_string, rendered_value)
        else:
            score = rouge_s(output_string, rendered_value)
    threshold = assertion.get("threshold")
    if threshold is None:
        threshold = 0.75
    passed = (score >= threshold) != inverse
    return {
        "pass": passed,
        "score": 1 - score if inverse else score,
        "reason": f"{base_type.upper()} score {to_fixed(score, 2)} is "
        f"{'greater than or equal to' if score >= threshold else 'less than'} "
        f"threshold {js_string(threshold)}",
        "assertion": assertion,
    }


# --------------------------------------------------------------------------------------------
# natural@8.1.1 PorterStemmer
# --------------------------------------------------------------------------------------------


def _categorize_groups(token: str) -> str:
    out = re.sub(r"[^aeiouy]+y", "CV", token)
    out = re.sub(r"[aeiou]+", "V", out)
    return re.sub(r"[^V]+", "C", out)


def _categorize_chars(token: str) -> str:
    out = re.sub(r"[^aeiouy]y", "CV", token)
    out = re.sub(r"[aeiou]", "V", out)
    return re.sub(r"[^V]", "C", out)


def _measure(token: str | None) -> float:
    if not token:
        return -1
    groups = re.sub(r"V\Z", "", re.sub(r"^C", "", _categorize_groups(token)))
    return len(groups) / 2


def _ends_with_double_cons(token: str) -> bool:
    return re.search(r"([^aeiou])\1\Z", token) is not None


_Pattern = str | re.Pattern[str]


def _attempt_replace(
    token: str,
    pattern: _Pattern,
    replacement: str,
    callback: Callable[[str], str | None] | None = None,
) -> str | None:
    result: str | None = None
    if isinstance(pattern, str):
        if token[len(token) - len(pattern) :] == pattern if len(pattern) <= len(token) else False:
            result = re.sub(re.escape(pattern) + r"\Z", replacement, token, count=1)
    elif pattern.search(token):
        result = pattern.sub(replacement, token, count=1)
    if result and callback:
        return callback(result)
    return result


def _attempt_replace_patterns(
    token: str, replacements: list[tuple[str, str, str]], threshold: float | None
) -> str:
    replacement = token
    for pattern, strip, repl in replacements:
        if threshold is None or _measure(_attempt_replace(token, pattern, strip)) > threshold:
            replacement = _attempt_replace(replacement, pattern, repl) or replacement
    return replacement


def _replace_patterns(
    token: str, replacements: list[tuple[str, str, str]], threshold: float | None
) -> str:
    return _attempt_replace_patterns(token, replacements, threshold) or token


def _replace_regex(
    token: str, regex: re.Pattern[str], include_parts: list[int], minimum: float
) -> str | None:
    result = ""
    m = regex.search(token)
    if m:
        result = "".join(m.group(i) for i in include_parts)
    return result if _measure(result) > minimum else None


def _step1a(token: str) -> str:
    if re.search(r"(ss|i)es\Z", token):
        return re.sub(r"(ss|i)es\Z", r"\1", token, count=1)
    if token[-1:] == "s" and _js_substr(token, -2, 1) != "s" and len(token) > 2:
        return re.sub(r"s?\Z", "", token, count=1)
    return token


def _step1b(token: str) -> str:
    if token[-3:] == "eed":
        if _measure(token[:-3]) > 0:
            return re.sub(r"eed\Z", "ee", token, count=1)
        return token

    def callback(t: str) -> str | None:
        if "V" in _categorize_groups(t):
            result = _attempt_replace_patterns(
                t, [("at", "", "ate"), ("bl", "", "ble"), ("iz", "", "ize")], None
            )
            if result != t:
                return result
            if _ends_with_double_cons(t) and re.search(r"[^lsz]\Z", t):
                return re.sub(r"([^aeiou])\1\Z", r"\1", t, count=1)
            if (
                _measure(t) == 1
                and _categorize_chars(t)[-3:] == "CVC"
                and re.search(r"[^wxy]\Z", t)
            ):
                return t + "e"
            return t
        return None

    result = _attempt_replace(token, re.compile(r"(ed|ing)\Z"), "", callback)
    return result if result else token


def _step1c(token: str) -> str:
    groups = _categorize_groups(token)
    if token[-1:] == "y" and "V" in groups[: len(groups) - 1]:
        return re.sub(r"y\Z", "i", token, count=1)
    return token


_STEP2 = [
    (s, "", r)
    for s, r in [
        ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
        ("izer", "ize"), ("abli", "able"), ("bli", "ble"), ("alli", "al"), ("entli", "ent"),
        ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"), ("ator", "ate"),
        ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"), ("ousness", "ous"),
        ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"), ("logi", "log"),
    ]
]  # fmt: skip
_STEP3 = [
    (s, "", r)
    for s, r in [
        ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"), ("ical", "ic"),
        ("ful", ""), ("ness", ""),
    ]
]  # fmt: skip
_STEP4_A = re.compile(
    r"^(.+?)(al|ance|ence|er|ic|able|ible|ant|ement|ment|ent|ou|ism|ate|iti|ous|ive|ize)\Z", re.S
)
_STEP4_B = re.compile(r"^(.+?)(s|t)(ion)\Z", re.S)


def _step4(token: str) -> str:
    return (
        _replace_regex(token, _STEP4_A, [1], 1)
        or _replace_regex(token, _STEP4_B, [1, 2], 1)
        or token
    )


def _step5a(token: str) -> str:
    m = _measure(re.sub(r"e\Z", "", token, count=1))
    if m > 1 or (
        m == 1
        and not (
            _js_substr(_categorize_chars(token), -4, 3) == "CVC"
            and re.search(r"[^wxy].\Z", token, re.S)
        )
    ):
        token = re.sub(r"e\Z", "", token, count=1)
    return token


def _step5b(token: str) -> str:
    if _measure(token) > 1:
        return re.sub(r"ll\Z", "l", token, count=1)
    return token


def porter_stem(token: str) -> str:
    """natural's PorterStemmer.stem."""
    if len(token.encode("utf-16-le", "surrogatepass")) // 2 < 3:
        return token
    t = _step1a(token.lower())
    t = _step1b(t)
    t = _step1c(t)
    t = _replace_patterns(t, _STEP2, 0)
    t = _replace_patterns(t, _STEP3, 0)
    t = _step4(t)
    return _step5b(_step5a(t))


# --------------------------------------------------------------------------------------------
# natural@8.1.1 WordNet.lookup over WordNet dict files
# --------------------------------------------------------------------------------------------


class MissingNlpDependency(RuntimeError):
    """The METEOR synonym stage has no WordNet data (Node: the `natural` package is missing)."""


def _js_parse_int(s: str | None, base: int = 10) -> int | None:
    if s is None:
        return None
    m = re.match(r"[+-]?[0-9a-fA-F]+" if base == 16 else r"[+-]?\d+", s.strip())
    return int(m.group(0), base) if m else None


class WordNet:
    """Reads WordNet dict files the way natural's IndexFile/DataFile do (byte-offset binary
    search over index.<pos>, then line reads from data.<pos>)."""

    POS = ("noun", "verb", "adj", "adv")

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files

    @staticmethod
    def _read_line(buf: bytes, pos: int) -> str:
        pos = max(pos, 0)
        while pos != 0:
            if buf[pos] == 10:
                pos += 1
                break
            pos -= 1
        end = buf.find(b"\n", pos)
        return buf[pos : len(buf) if end < 0 else end].decode("utf-8", "replace")

    def _find(self, buf: bytes, key: str) -> list[str] | None:
        size = len(buf) - 1
        pos = math.ceil(size / 2)
        last_pos: int | None = None
        adjustment = pos
        last_key: str | None = None
        while True:
            if last_pos == pos or pos >= size:
                return None
            tokens = _js_split_ws(self._read_line(buf, pos))
            k = tokens[0]
            if k == key:
                return tokens
            if adjustment == 1 or k == last_key:
                return None
            adjustment = math.ceil(adjustment * 0.5)
            last_pos, last_key = pos, k
            pos = pos + adjustment if k < key else pos - adjustment

    def _index_offsets(self, pos: str, word: str) -> list[int] | None:
        tokens = self._find(self.files["index." + pos], word)
        if tokens is None:
            return None
        p_cnt = _js_parse_int(tokens[3] if len(tokens) > 3 else None) or 0
        synset_cnt = _js_parse_int(tokens[2] if len(tokens) > 2 else None) or 0
        offsets: list[int] = []
        for i in range(max(synset_cnt, 0)):
            j = max(p_cnt, 0) + 6 + i
            offsets.append(_js_parse_int(tokens[j] if j < len(tokens) else None) or 0)
        return offsets

    def _data_synonyms(self, pos: str, location: int) -> list[str]:
        buf = self.files["data." + pos]
        end = buf.find(b"\n", location)
        line = buf[location : len(buf) if end < 0 else end].decode("utf-8", "replace")
        tokens = _js_split_ws(line.split("| ")[0])
        w_cnt = _js_parse_int(tokens[3] if len(tokens) > 3 else None, 16) or 0
        return [tokens[4 + i * 2] for i in range(w_cnt) if 4 + i * 2 < len(tokens)]

    def lookup(self, word: str) -> list[list[str]]:
        """Synonym lists of every synset for `word` (natural's lookup().map(r => r.synonyms))."""
        word = _WS_RUN.sub("_", word.lower())
        results: list[list[str]] = []
        for pos in reversed(self.POS):
            offsets = self._index_offsets(pos, word)
            if offsets:
                for offset in reversed(offsets):
                    results.append(self._data_synonyms(pos, offset))
        return results


_wordnet_instance: WordNet | None = None


def _load_wordnet() -> WordNet:
    """Locate WordNet dict files through nltk's data path (wordnet31, then wordnet)."""
    global _wordnet_instance
    if _wordnet_instance is not None:
        return _wordnet_instance
    try:
        import nltk.data  # type: ignore[import-not-found]
    except ImportError as err:
        raise MissingNlpDependency("nltk is not installed") from err
    for resource in ("corpora/wordnet31", "corpora/wordnet"):
        try:
            pointer: Any = nltk.data.find(resource)
        except LookupError:
            continue
        files: dict[str, bytes] = {}
        for pos in WordNet.POS:
            for kind in ("index", "data"):
                with pointer.join(f"{kind}.{pos}").open() as fh:
                    files[f"{kind}.{pos}"] = fh.read()
        _wordnet_instance = WordNet(files)
        return _wordnet_instance
    raise MissingNlpDependency("WordNet data is not installed")


# Indirection points so tests can substitute the stemmer and WordNet (upstream mocks `natural`).
_stem: Callable[[str], str] = porter_stem
_get_wordnet: Callable[[], Any] = _load_wordnet


# --------------------------------------------------------------------------------------------
# meteor.ts
# --------------------------------------------------------------------------------------------

_WordPair = tuple[int, str]
_MatchPair = tuple[int, int]


def _generate_enums(
    candidate: list[str], reference: list[str]
) -> tuple[list[_WordPair], list[_WordPair]]:
    return (
        [(i, w.lower()) for i, w in enumerate(candidate)],
        [(i, w.lower()) for i, w in enumerate(reference)],
    )


def _match_exact_enums(
    candidates: list[_WordPair], references: list[_WordPair]
) -> tuple[list[_MatchPair], list[_WordPair], list[_WordPair]]:
    word_match: list[_MatchPair] = []
    cand = list(candidates)
    ref = list(references)
    for i in range(len(cand) - 1, -1, -1):
        for j in range(len(ref) - 1, -1, -1):
            if cand[i][1] == ref[j][1]:
                word_match.append((cand[i][0], ref[j][0]))
                del cand[i]
                del ref[j]
                break
    return word_match, cand, ref


def _match_stem_enums(
    candidates: list[_WordPair], references: list[_WordPair]
) -> tuple[list[_MatchPair], list[_WordPair], list[_WordPair]]:
    _get_wordnet()  # ensureNaturalPackage(): fails here when the dependency is missing
    stem = _stem
    return _match_exact_enums(
        [(i, stem(w)) for i, w in candidates], [(i, stem(w)) for i, w in references]
    )


def _match_synonym_enums(
    candidates: list[_WordPair], references: list[_WordPair]
) -> list[_MatchPair]:
    wordnet = _get_wordnet()
    word_match: list[_MatchPair] = []
    ref = list(references)
    for i in range(len(candidates) - 1, -1, -1):
        candidate_word = candidates[i][1]
        synonym_set = {candidate_word}
        for synonyms in wordnet.lookup(candidate_word):
            synonym_set.update(s for s in synonyms if "_" not in s)
        for j in range(len(ref) - 1, -1, -1):
            if ref[j][1] in synonym_set:
                word_match.append((candidates[i][0], ref[j][0]))
                del ref[j]
                break
    return word_match


def _count_chunks(matches: list[_MatchPair]) -> int:
    if not matches:
        return 0
    chunks = 1
    for a, b in zip(matches, matches[1:], strict=False):
        if b[0] != a[0] + 1 or b[1] != a[1] + 1:
            chunks += 1
    return chunks


def _single_meteor_score(
    reference: list[str], candidate: list[str], alpha: float, beta: float, gamma: float
) -> float:
    enum_candidate, enum_reference = _generate_enums(candidate, reference)
    translation_length = len(enum_candidate)
    reference_length = len(enum_reference)
    exact, rest_cand, rest_ref = _match_exact_enums(enum_candidate, enum_reference)
    stem, rest_cand2, rest_ref2 = _match_stem_enums(rest_cand, rest_ref)
    synonym = _match_synonym_enums(rest_cand2, rest_ref2)
    # Array.prototype.sort is stable; Python's sort is too.
    all_matches = sorted([*exact, *stem, *synonym], key=lambda m: m[0])
    matches_count = len(all_matches)
    if translation_length == 0 or reference_length == 0 or matches_count == 0:
        return 0.0
    precision = matches_count / translation_length
    recall = matches_count / reference_length
    denominator = alpha * precision + (1 - alpha) * recall
    if denominator == 0:
        return 0.0
    fmean = (precision * recall) / denominator
    frag_frac = _count_chunks(all_matches) / matches_count
    penalty = gamma * math.pow(frag_frac, beta)
    return (1 - penalty) * fmean


def _meteor_words(text: str) -> list[str]:
    return [re.sub(r"\.+\Z", "", w) for w in _js_split_ws(text)]


def calculate_meteor_score(
    candidate: str,
    references: list[str],
    alpha: float = 0.9,
    beta: float = 3.0,
    gamma: float = 0.5,
) -> float:
    if not candidate or len(references) == 0:
        raise ValueError("Invalid inputs")
    return max(
        _single_meteor_score(_meteor_words(r), _meteor_words(candidate), alpha, beta, gamma)
        for r in references
    )


METEOR_MISSING_REASON = (
    "METEOR assertion requires the nltk package and WordNet data. Please install them using: "
    "pip install promptphoo[nlp] && python -m nltk.downloader wordnet31"
)


def handle_meteor_assertion(params: AssertionParams) -> GradingResult:
    """handleMeteorAssertion plus the missing-dependency fallback from assertions/index.ts."""
    assertion, rendered_value, inverse = params.assertion, params.rendered_value, params.inverse
    invariant(
        _is_string_or_string_list(rendered_value),
        '"meteor" assertion must have a string or array of strings value',
    )
    references = rendered_value if isinstance(rendered_value, list) else [rendered_value]

    def opt(key: str, default: float) -> Any:
        value = assertion.get(key)
        return default if value is None else value

    alpha, beta, gamma = opt("alpha", 0.9), opt("beta", 3.0), opt("gamma", 0.5)
    threshold = opt("threshold", 0.5)
    try:
        score = calculate_meteor_score(params.output_string, references, alpha, beta, gamma)
    except MissingNlpDependency:
        return {"pass": False, "score": 0, "reason": METEOR_MISSING_REASON, "assertion": assertion}
    passed = score < threshold if inverse else score >= threshold
    return {
        "pass": passed,
        "score": 1 - score if inverse else score,
        "reason": "METEOR assertion passed"
        if passed
        else f"METEOR score {to_fixed(score, 4)} did not meet threshold {js_string(threshold)}",
        "assertion": assertion,
    }


HANDLERS: dict[str, Handler] = {
    "levenshtein": handle_levenshtein,
    "bleu": handle_bleu_score,
    "gleu": handle_gleu_score,
    "rouge-n": handle_rouge_score,
    "rouge-l": handle_rouge_score,
    "rouge-s": handle_rouge_score,
    "meteor": handle_meteor_assertion,
}
