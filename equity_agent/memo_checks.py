"""check_memo rules, plain code. A section's narrative passes when:
  1. it is not empty;
  2. every number is followed, in its sentence, by a ledger id in brackets (it is supported if any id after it in the sentence matches), the id exists, was given to this section, and the number matches
     that fact under the rounding rule (numbers.py);
  3. every cited fact of kind model_estimate or assumption sits in a sentence with estimate wording (word list below);
  4. every data gap relevant to the section is mentioned, shown by its tag [gap:G1]; a tag that does not exist is a problem.
Tables and the Sources and Methodology section are built by code from the ledger and are not checked here.
"""
import re

from .ledger import Ledger
from .numbers import GAP_TAG, ID_BRACKET, Mention, find_mentions, matches, split_ids

# Word list for rule 3, a plain word-list check: a sentence that cites an estimate must say it is one.
ESTIMATE_WORDS = re.compile(r"\b(estimat\w*|assum\w*|project\w*|forecast\w*|model\w*|impl(?:y|ies|ied)|scenario\w*|illustrat\w*)\b", re.IGNORECASE)
_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n")
MIN_CHARS = 80


def citations(text: str) -> list[tuple[Mention, list[str]]]:
    """(number, ids cited for it): the ids of every bracket that follows the number in its sentence. A number is supported if any
    of them matches it, so "15 scored articles with a mean of 0.25 [mean_score][n_scored]" is checked against both facts. An
    empty list means no bracket follows the number in its sentence."""
    found = []
    for m in find_mentions(text):
        ids: list[str] = []
        stop = _BOUNDARY.search(text, m.end)
        end = stop.start() if stop else len(text)
        for hit in ID_BRACKET.finditer(text, m.end, end):
            ids.extend(split_ids(hit.group(1)))
        found.append((m, ids))
    return found


def _sentence_at(text: str, pos: int) -> str:
    start = max([0] + [b.end() for b in _BOUNDARY.finditer(text, 0, pos)])
    nxt = _BOUNDARY.search(text, pos)
    return text[start: nxt.start() if nxt else len(text)]


def check_section(text: str, ledger: Ledger, allowed_ids: set[str], gap_tags: dict[str, str], known_gap_tags: set[str] | None = None) -> list[str]:
    """The problems found in one section's narrative. Empty list = passed. gap_tags maps the tags this section must mention to a short
    description; known_gap_tags (when given) are all tags that exist, so an invented tag is a problem."""
    if len(text.strip()) < MIN_CHARS:
        return ["the section is missing or nearly empty"]
    problems = []
    for mention, ids in citations(text):
        if not ids:
            problems.append(f"the number {mention.raw!r} (in: ...{text[max(0, mention.start - 40): mention.end + 25].strip()!r}...) has no ledger id after it in its sentence")
            continue
        unknown = [i for i in ids if i not in ledger]
        if unknown:
            problems.append(f"the number {mention.raw!r} cites {unknown}, which are not in the ledger")
            continue
        outside = [i for i in ids if i not in allowed_ids]
        if outside:
            problems.append(f"the number {mention.raw!r} cites {outside}, which were not given to this section")
        elif not any(matches(mention, ledger.get(i)) for i in ids):
            shown = ", ".join(f"{i}={ledger.get(i).value}" for i in ids)
            problems.append(f"the number {mention.raw!r} does not match the cited fact(s) ({shown}) under the rounding rule")
    flagged = set()
    for hit in ID_BRACKET.finditer(text):      # ids that follow no number must still exist, and an estimate must be worded as one
        for fid in split_ids(hit.group(1)):
            if fid not in ledger:
                problems.append(f"the id {fid!r} is not in the ledger")
            elif ledger.get(fid).kind in ("model_estimate", "assumption"):
                sentence = _sentence_at(text, hit.start())
                if not ESTIMATE_WORDS.search(sentence) and sentence not in flagged:
                    flagged.add(sentence)
                    problems.append(f"{ledger.get(fid).kind.replace('_', ' ')} {fid} is not worded as an estimate in: {sentence.strip()[:100]!r}")
    present = set(GAP_TAG.findall(text))
    for tag in sorted(present - (known_gap_tags if known_gap_tags is not None else present)):
        problems.append(f"the gap tag [gap:{tag}] does not exist; use only the tags listed under GAPS TO MENTION")
    for tag, what in gap_tags.items():
        if tag not in present:
            problems.append(f"the data gap [gap:{tag}] ({what}) is not mentioned")
    return problems
