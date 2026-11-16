"""Export helpers: numbered references, the HTML for the PDF, and the PDF itself. Pure functions; the node does the writing.

Reference format. In the exported memo.md every ledger id in the text becomes a number in order of first appearance, written
`[[1]](#ref-1)` (a link to row 1 of the reference tables in Sources and Methodology). A bracket with several ids gives several
numbers. A gap tag `[gap:G2]` becomes `[[G2]](#gap-G2)`, a link to that gap in the data gap list.
"""
import re
from pathlib import Path

from .numbers import GAP_TAG, ID_BRACKET, split_ids

CSS = """
body { font-family: Helvetica, Arial, sans-serif; font-size: 10pt; line-height: 1.4; color: #111; }
h1 { font-size: 19pt; border-bottom: 2px solid #333; padding-bottom: 4pt; }
h2 { font-size: 14pt; margin-top: 18pt; color: #1a3a5c; break-after: avoid; }
h3 { font-size: 11.5pt; break-after: avoid; }
blockquote { border-left: 3px solid #999; margin: 8pt 0; padding: 2pt 10pt; color: #333; background: #f4f4f4; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 8pt; table-layout: auto; }
th, td { border: 1px solid #999; padding: 3pt 4pt; word-break: break-word; vertical-align: top; }
th { background: #e8eef5; text-align: left; }
code { font-size: 7.5pt; word-break: break-all; }
a { color: #1a3a5c; text-decoration: none; }
@page { size: A4; margin: 22mm 16mm 20mm 16mm;
  @top-center { content: string(header-line); font-family: Helvetica, Arial, sans-serif; font-size: 8pt; color: #555; }
  @bottom-center { content: "Page " counter(page) " of " counter(pages); font-family: Helvetica, Arial, sans-serif; font-size: 8pt; color: #555; } }
"""


def number_references(texts: list[str], extra_ids: set[str], known: set[str]) -> dict[str, int]:
    """id -> 1, 2, 3... in order of first appearance in `texts` (in section order); ids that only tables use come last, sorted.
    Ids that are not in the ledger (`known`) get no number."""
    order: list[str] = []
    for text in texts:
        for hit in ID_BRACKET.finditer(text):
            for i in split_ids(hit.group(1)):
                if i in known and i not in order:
                    order.append(i)
    order += sorted(i for i in extra_ids if i in known and i not in order)
    return {i: n for n, i in enumerate(order, start=1)}


def numberize(text: str, refs: dict[str, int]) -> str:
    """Ledger ids and gap tags in the text become numbered links. A bracket with an id that has no number is left as written."""
    def ids(m: re.Match) -> str:
        found = split_ids(m.group(1))
        return "".join(f"[[{refs[i]}]](#ref-{refs[i]})" for i in found) if all(i in refs for i in found) else m.group(0)
    return GAP_TAG.sub(lambda m: f"[[{m.group(1)}]](#gap-{m.group(1)})", ID_BRACKET.sub(ids, text))


def html_document(markdown_text: str, header_line: str) -> str:
    import markdown

    body = markdown.markdown(markdown_text, extensions=["tables", "sane_lists"])
    safe = header_line.replace("&", "&amp;").replace("<", "&lt;")
    return (f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS} .hdr {{ string-set: header-line content(); height: 0; margin: 0; font-size: 0; overflow: hidden; }}</style></head>"
            f"<body><p class='hdr'>{safe}</p>{body}</body></html>")


def write_pdf(markdown_text: str, header_line: str, path: Path) -> None:
    """Markdown to HTML to PDF with WeasyPrint. Raises if WeasyPrint or its system libraries are missing."""
    from weasyprint import HTML

    HTML(string=html_document(markdown_text, header_line)).write_pdf(str(path))
