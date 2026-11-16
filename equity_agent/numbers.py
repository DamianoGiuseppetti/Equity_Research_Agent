"""The rounding rule: when does a number written in text match a ledger value?

THE RULE. A number in text matches a ledger value when the ledger value, scaled to the unit shown in the text and rounded
(half up, away from zero) to the number of decimals shown in the text, equals the number in the text.

  * Unit shown "%" or "percent": the value is shown in percent. A fact of unit "fraction" is multiplied by 100 first
    (0.0814 is "8.1%"); a fact of unit "percent" is used as it is.
  * Scale word "thousand", "million", "billion", "trillion" (or bn, mn, K, M, B, T attached to the number): the number is
    the value divided by that scale ("$332.2 billion" matches 332,238,000,000). Facts whose unit already carries a scale
    ("USD billions", "USD millions", "thousand units") are first turned into absolute amounts.
  * A one-letter scale (K, M, B, T) counts only after a dollar sign ("$5B"); "3M" is a name.
  * Suffix "x" or the word "times": the value of a fact of unit "times" or "ratio", unscaled ("1.4x").
  * No unit: the value as it is ("$347.73", "14.2", "-0.5").
  * Decimals shown decide the precision: "8.1%" needs the value to round to 8.1; "8%" needs it to round to 8.
  * Sign: the sign shown must be the sign of the value. One exception: a number written without a sign matches a negative
    value when the same sentence has a word of direction (fell, declined, down, loss, ...), because "revenue fell 8.5%"
    is how a writer states -0.085.
  * The capital expenditure line is stored negative (cash leaving); it matches the magnitude, "capex of $2.1 billion".
  * A fact whose value is text (an extracted summary) matches a number that appears in its text or in the excerpt kept in
    its note, because the filing itself states that number.

NOT CHECKED, on purpose: ledger-id and gap tags in brackets, days of the month ("December 31"), numbered-list markers, "year 5" and "Scope 1", years (2025), fiscal labels (FY2025, Q3), ISO dates, form and item names (10-K, Item 1A), and
time windows ("10-year", "52-week", "12 months"). They name a period, not a measured value.
"""
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_DOWN, ROUND_HALF_UP, Decimal

from .schemas import Fact

SCALES = {"thousand": 1e3, "k": 1e3, "million": 1e6, "mn": 1e6, "m": 1e6, "billion": 1e9, "bn": 1e9, "b": 1e9,
          "trillion": 1e12, "t": 1e12}
OUTFLOW_LINES = ("capital_expenditure",)       # statement lines stored negative although the text names the amount spent
UNIT_MULTIPLIER = {"USD billions": 1e9, "USD millions": 1e6, "thousand units": 1e3}
NEGATIVE_WORDS = re.compile(r"\b(fell|fall|falls|fallen|declin\w*|decreas\w*|drop\w*|down|lower|loss|losses|negative|contract\w*|"
                            r"shrank|shrink\w*|below|deficit|reduc\w*|lost|slump\w*|weaker|weakened)\b", re.IGNORECASE)

_ID = r"[a-z0-9_.:\-]+\.[a-z0-9_.:\-]+"
ID_BRACKET = re.compile(rf"\[({_ID}(?:\s*[,;]\s*{_ID})*)(?:\s*:[^\]]*)?\]")      # [id], [id1, id2] or [id: 1] (text after the colon is ignored)
GAP_TAG = re.compile(r"[\[(]gap:(G\d+)[^\])]*[\])]")               # [gap:G2] or (gap:G2); words after the tag are tolerated


def split_ids(bracket_text: str) -> list[str]:
    """'a.b, c.d' -> ['a.b', 'c.d']"""
    return [i.strip() for i in re.split(r"[,;]", bracket_text) if i.strip()]
_EXEMPT = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),                                  # ISO dates
    re.compile(r"\bFY\s?\d{2,4}\b|\bQ[1-4]\b|\bH[12]\b", re.IGNORECASE),    # fiscal labels
    re.compile(r"\b(10-K|10-Q|8-K|20-F|S-1)\b"),                            # form names
    re.compile(r"(?m)^[ \t]*\d+[.)](?=\s)"),                                  # a numbered-list marker: "1. Commodity prices"
    re.compile(r"\bItems?\s+\d+[A-C]?\b", re.IGNORECASE),                   # Item 1A
    re.compile(r"\b(?:years?|Scope)\s+\d+(?:\s*(?:to|and|&|-)\s*\d+)?\b", re.IGNORECASE),   # year 5, years 1 to 5, Scope 1 and 2
    re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}\b,?"),     # December 31
    re.compile(r"\b\d+(?:\.\d+)?[- ](?:year|day|week|month|quarter)s?\b", re.IGNORECASE),     # time windows: 10-year, 52 weeks
    re.compile(r"(?<![\d$.,])(?:19|20)\d{2}(?![\d%x]|[.,]\d)"),              # a year
]
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)|\n")      # a full stop inside 8.1 is not the end of a sentence
_MENTION = re.compile(
    r"(?:(?<![\w.,])(?P<neg>[-−–]))?\s?(?P<cur>\$)?\s?(?P<num>\d[\d,]*(?:\.\d+)?)"
    r"(?:\s?(?P<pct>%|percent\b)|(?P<x>x\b)|\s(?P<times>times\b)|"
    r"(?:\s?(?P<scale>thousand\b|million\b|billion\b|trillion\b|bn\b|mn\b)|(?P<abbr>[KMBT]\b)))?")


@dataclass(frozen=True)
class Mention:
    raw: str              # the number as written, for messages
    digits: str           # digits only, commas removed: '1234.5'
    value: Decimal        # signed, as shown
    decimals: int
    form: str             # "percent", "multiple" or "amount"
    scale: float          # 1 when no scale word
    sentence_has_negative_word: bool
    start: int = 0        # position in the text that was scanned
    end: int = 0


def find_mentions(text: str) -> list[Mention]:
    """Every number that must be checked: ledger-id brackets and the exempt kinds are blanked out first."""
    cleaned = GAP_TAG.sub(lambda m: " " * len(m.group(0)), ID_BRACKET.sub(lambda m: " " * len(m.group(0)), text))
    for pattern in _EXEMPT:
        cleaned = pattern.sub(lambda m: " " * len(m.group(0)), cleaned)
    boundaries = [b.span() for b in _SENTENCE_END.finditer(cleaned)]
    out = []
    for m in _MENTION.finditer(cleaned):
        if m.group("abbr") and not m.group("cur"):
            continue      # "3M" is a company name, not 3 million; a one-letter scale needs a dollar sign ("$5B")
        num = m.group("num")
        sign = -1 if m.group("neg") else 1
        start = max([0] + [e for _, e in boundaries if e <= m.start()])
        end = min([len(cleaned)] + [b for b, _ in boundaries if b >= m.end()])
        word = (m.group("scale") or m.group("abbr") or "").lower()
        form = "percent" if m.group("pct") else "multiple" if (m.group("x") or m.group("times")) else "amount"
        out.append(Mention(m.group(0).strip(), num.replace(",", ""), Decimal(num.replace(",", "")) * sign, len(num.split(".")[1]) if "." in num else 0,
                           form, SCALES.get(word, 1.0), bool(NEGATIVE_WORDS.search(cleaned[start:end])),
                           m.start() + len(m.group(0)) - len(m.group(0).lstrip()), m.end()))
    return out


def _roundings(x: float, decimals: int) -> set[Decimal]:
    """The shown values the number x may be written as: rounded half up, after removing float noise (0.0815 x 100 is
    8.150000000000002). On an exact tie (13.225 at two decimals) both neighbours are accepted, because writers round ties
    both ways and neither is wrong."""
    exact, q = Decimal(repr(round(x, 9))), Decimal(1).scaleb(-decimals)
    return {exact.quantize(q, rounding=ROUND_HALF_UP), exact.quantize(q, rounding=ROUND_HALF_DOWN)}


def _number_strings(text: str) -> set[str]:
    return {n.replace(",", "") for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}


def matches(mention: Mention, fact: Fact) -> bool:
    """Does this ledger fact support this number? See the module docstring for the rule."""
    if isinstance(fact.value, str):
        return mention.digits in _number_strings(fact.value + " " + fact.note)
    value, unit = float(fact.value) * UNIT_MULTIPLIER.get(fact.unit, 1.0), fact.unit
    if any(k in fact.id for k in OUTFLOW_LINES):
        value = abs(value)      # statements store spending as a negative number; "capex of $2.1 billion" is its magnitude
    if mention.form == "percent":
        if unit == "fraction":
            shown = value * 100
        elif unit == "percent":
            shown = value
        else:
            return False
    elif mention.form == "multiple":
        if unit not in ("times", "ratio"):
            return False
        shown = value
    else:
        if unit == "fraction":
            return False
        shown = value / mention.scale
    candidates = [mention.value]
    if mention.value > 0 and mention.sentence_has_negative_word:
        candidates.append(-mention.value)       # "fell 8.5%" states -8.5
    return any(c in _roundings(shown, mention.decimals) for c in candidates)


def unmatched_numbers(text: str, facts: list[Fact]) -> list[str]:
    """The numbers in `text` that none of `facts` supports. Empty means every number is accounted for."""
    return [m.raw for m in find_mentions(text) if not any(matches(m, f) for f in facts)]


def display(fact: Fact) -> str:
    """The value as the rule would accept it, for prompts: '$332.2 billion', '8.1%', '14.2', '1.4x'. Text values as they are."""
    v = fact.value
    if isinstance(v, str):
        return v
    v = float(v) * UNIT_MULTIPLIER.get(fact.unit, 1.0)
    if fact.unit == "fraction":
        return f"{v * 100:.1f}%"
    if fact.unit == "percent":
        return f"{v:.2f}%"
    if fact.unit in ("times", "ratio") :
        return f"{v:.2f}x" if fact.unit == "times" else f"{v:.2f}"
    if fact.unit in ("USD", "USD billions", "USD millions"):
        for word, scale in (("trillion", 1e12), ("billion", 1e9), ("million", 1e6)):
            if abs(v) >= scale:
                return f"{'-' if v < 0 else ''}${abs(v) / scale:,.1f} {word}"
        return f"{'-' if v < 0 else ''}${abs(v):,.2f}"
    if fact.unit == "USD/share":
        return f"{'-' if v < 0 else ''}${abs(v):,.2f}"
    return f"{v:,.2f}".rstrip("0").rstrip(".") if abs(v) < 1e6 else f"{v:,.0f}"
