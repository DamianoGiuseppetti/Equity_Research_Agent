"""10-K sections (Items 1, 1A, 7) and the newer 10-Q MD&A, isolated with edgartools, plus the guards around them.

edgartools never raises on a wrong span, so two checks run on every section: it must have a minimum length, and
the filing text right after it must be the expected next heading. If Item 7 only points to another part of the
filing (Deere), the captioned section is searched for in the full text.
"""
import re
from dataclasses import dataclass
from typing import Protocol

from .schemas import DataGap, Fact

CHARS_PER_TOKEN = 3.3     # measured 3.0 to 3.7 characters per token with Anthropic's count_tokens on 15 filing sections
NEXT_HEADING_WINDOW = 600  # characters after the section in which the next item heading must appear

SECTIONS = {  # key -> (label, regex of the heading that must follow, what the memo loses if it is flagged)
    "item1": ("Item 1 (Business)", r"item\s*1a\b", "Company Overview and Industry and Competitive Analysis rest on yfinance and XBRL only"),
    "item1a": ("Item 1A (Risk Factors)", r"item\s*1b\b|item\s*2\b", "Risks and Catalysts has no extracted risk factors"),
    "item7": ("Item 7 (MD&A)", r"item\s*7a\b|item\s*8\b", "Historical Financial Analysis and Financial Outlook have no management commentary"),
    "tenq_mdna": ("MD&A of the latest 10-Q", r"item\s*3\b|item\s*4\b|part\s+ii\b", "the memo has no commentary on the quarter after the 10-K"),
}
# lines that end a captioned MD&A section found by the Deere-style fallback
FALLBACK_END = re.compile(r"(?im)^\W*(statements? of consolidated |consolidated (balance sheets?|statements? of )|notes to consolidated financial|"
                          r"report of independent registered public accounting firm|management.s report on internal control|"
                          r"item\s*8\b|item\s*7a\b|quantitative and qualitative)")


# edgartools runs XOM's Item 7 on into the auditor's report; the report heading marks where the MD&A ends
AUDITOR_REPORT = re.compile(r"(?im)^\W*report of independent registered public accounting firm\W*$")


class SectionsProvider(Protocol):
    def fetch(self, ticker: str, cik: int) -> dict:
        """{"tenk": {accession, filing_date, period, url, form, items: {"1","1A","7"}, full_text}, "tenq": None or
        {accession, filing_date, period, url, form, mdna}} with the 10-Q only if its period is newer than the 10-K's."""
        ...


class EdgarToolsSections:
    def __init__(self, user_agent: str) -> None:
        self._user_agent = user_agent

    def fetch(self, ticker: str, cik: int) -> dict:
        from edgar import Company, set_identity

        set_identity(self._user_agent)
        company = Company(cik)
        k = company.get_filings(form="10-K").latest()
        if k is None:
            raise LookupError(f"no 10-K found for CIK {cik}")
        tenk = k.obj()
        meta = lambda f: {"accession": f.accession_no, "filing_date": str(f.filing_date), "period": str(f.period_of_report),
                          "url": f.filing_url, "form": f.form}
        out = {"tenk": {**meta(k), "items": {"1": tenk.business, "1A": tenk.risk_factors, "7": tenk.management_discussion},
                        "full_text": k.text()}, "tenq": None}
        q = company.get_filings(form="10-Q").latest()
        if q is not None and str(q.period_of_report) > str(k.period_of_report):
            out["tenq"] = {**meta(q), "mdna": q.obj()["Item 2"]}
        return out


@dataclass
class SectionOutcome:
    key: str
    text: str | None
    status: str           # ok | short | end_unverified | missing
    detail: str
    via: str = "edgartools"

    @property
    def usable(self) -> bool:
        return self.status == "ok"


def _flat(s: str) -> str:
    """Whitespace collapsed; zero-width characters (common in Deere and Hurco filings) removed."""
    return re.sub(r"\s+", " ", re.sub("[\u200b\u200c\u200d\ufeff]", "", s)).strip()


def _letters_and_map(text: str) -> tuple[str, list[int]]:
    """Lower-case letters and digits only, and for each kept character its position in `text`.

    edgartools and the plain filing text render tables, page numbers and spacing differently; comparing letters and
    digits only finds the same passage in both.
    """
    kept, where = [], []
    for i, ch in enumerate(text):
        if ch.isalnum():
            kept.append(ch.lower())
            where.append(i)
    return "".join(kept), where


def next_heading_follows(full_text: str, section: str, pattern: str) -> tuple[bool, str]:
    """Is the expected next item heading within a short window after the end of the section in the filing text?"""
    flat_full = _flat(full_text)
    letters_full, where = _letters_and_map(flat_full)
    letters_section = _letters_and_map(_flat(section))[0].rstrip("0123456789")   # drop a trailing page number
    for size in (120, 60, 30):  # a page break inside the tail can hide it from the filing text
        at = letters_full.rfind(letters_section[-size:])
        if at >= 0:
            end = where[at + size - 1] + 1
            break
    else:
        return False, "the end of the section could not be located in the filing text"
    window = flat_full[end: end + NEXT_HEADING_WINDOW]
    if re.search(pattern, window, re.IGNORECASE):
        return True, "the expected next heading follows"
    return False, f"the expected next heading does not follow; the text after it begins: {window[:80]!r}"


def captioned_section(full_text: str, stub: str, min_chars: int) -> str | None:
    """Item 7 that only says 'see the caption "X"': find the heading line X later in the filing and cut to the next major heading."""
    quoted = re.findall(r"[“\"]([^”\"]{8,80})[”\"]", stub)
    if not quoted:
        return None
    caption = quoted[0].strip().rstrip(".,;:")
    heading = re.compile(rf"(?im)^\W*{re.escape(caption)}\W*$")
    stub_at = full_text.find(stub[:60])
    starts = [m.start() for m in heading.finditer(full_text) if m.start() > max(stub_at, 0) + len(stub)]
    if not starts:
        return None
    start = starts[0]
    end = next((m.start() for m in FALLBACK_END.finditer(full_text, start + min_chars)), None)
    return full_text[start:end].strip() if end else None


def cut_at_auditor_report(text: str, min_chars: int) -> str | None:
    """The text before the first auditor's report heading, or None if there is no such heading after min_chars.

    The heading is its own end marker, so the next-item check does not apply to the shortened section.
    """
    m = AUDITOR_REPORT.search(text, min_chars)
    return text[: m.start()].strip() if m else None


def analyse(payload: dict, min_chars: int) -> list[SectionOutcome]:
    tenk, out = payload["tenk"], []
    full = tenk.get("full_text", "")
    for key, item in (("item1", "1"), ("item1a", "1A"), ("item7", "7")):
        label, next_pattern, _ = SECTIONS[key]
        text = (tenk["items"].get(item) or "").strip()
        via = "edgartools"
        if key == "item7" and len(text) < min_chars:
            found = captioned_section(full, text, min_chars) if text else None
            if found:
                text, via = found, "caption fallback"
        elif key == "item7":
            cut = cut_at_auditor_report(text, min_chars)
            if cut:
                text, via = cut, "cut at the auditor's report"
        # a fallback section was cut at a statement heading, which is its own end check; no next-item heading follows it
        out.append(_judge(key, label, text, next_pattern, "" if via != "edgartools" else full, min_chars, via))
    if payload.get("tenq"):
        label, next_pattern, _ = SECTIONS["tenq_mdna"]
        out.append(_judge("tenq_mdna", label, (payload["tenq"].get("mdna") or "").strip(), next_pattern, "", min_chars, "edgartools"))
    return out


def _judge(key, label, text, next_pattern, full, min_chars, via) -> SectionOutcome:
    if not text:
        return SectionOutcome(key, None, "missing", f"{label} could not be isolated", via)
    if len(text) < min_chars:
        return SectionOutcome(key, text, "short", f"{len(text):,} characters, under the {min_chars:,} minimum: probably a stub or a wrong span", via)
    if not full:      # no next-item check possible (10-Q path, or a fallback cut at its own end marker): length is the guard
        return SectionOutcome(key, text, "ok", "length check passed; no next-item heading check for this section", via)
    ok, why = next_heading_follows(full, text, next_pattern)
    return SectionOutcome(key, text, "ok" if ok else "end_unverified", why, via)


def to_facts_and_gaps(outcomes: list[SectionOutcome], payload: dict, ticker: str, retrieved_at, paths: dict[str, str]
                      ) -> tuple[list[Fact], list[DataGap]]:
    """Metadata only: accession, dates, URL, size and guard result. The text itself stays in the section files."""
    facts, gaps = [], []

    def add(fid, value, unit, label, kind, source, as_of, note=""):
        facts.append(Fact(id=fid, value=value, unit=unit, label=label, kind=kind, source=source, as_of=as_of,
                          retrieved_at=retrieved_at, note=note))

    for tag, rec in (("tenk", payload["tenk"]), ("tenq", payload.get("tenq"))):
        if not rec:
            continue
        src = f"SEC EDGAR filing {rec['accession']} ({rec['form']})"
        for slug, label in (("accession", "accession number"), ("filing_date", "filing date"), ("url", "URL"), ("period", "period of report")):
            add(f"filings.{tag}.{slug}", rec[slug], "text", f"{ticker} {rec['form']} {label}", "reported", src, rec["period"])
    for o in outcomes:
        label, _, effect = SECTIONS[o.key]
        rec = payload["tenq"] if o.key == "tenq_mdna" else payload["tenk"]
        src = f"SEC EDGAR filing {rec['accession']} ({rec['form']}), isolated with edgartools" + (f" and the {o.via}" if o.via != "edgartools" else "")
        if o.text:
            add(f"filings.section.{o.key}.chars", len(o.text), "characters", f"{ticker} {label}: length", "reported", src, rec["period"],
                f"text file {paths.get(o.key, '')}")
            add(f"filings.section.{o.key}.tokens_est", round(len(o.text) / CHARS_PER_TOKEN), "tokens", f"{ticker} {label}: estimated tokens",
                "derived", "computed in code from the character count", rec["period"], f"characters / {CHARS_PER_TOKEN}; an estimate, not a tokenizer count")
        add(f"filings.section.{o.key}.status", o.status, "text", f"{ticker} {label}: guard result", "derived", "computed in code",
            rec["period"], f"{o.detail}; via {o.via}")
        if not o.usable:
            gaps.append(DataGap(area="filings", missing=label, reason=f"{o.status}: {o.detail}", effect=effect))
    return facts, gaps
