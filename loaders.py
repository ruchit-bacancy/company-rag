"""
loaders.py — Turn source documents into clean, structured "blocks".

This is STEP 1 of ingestion (document loading). Plain text extraction is not
good enough for long reports full of tables, charts and footnotes, so this
module does the heavy lifting that makes the three advanced tasks work:

  * PDFs are converted page by page to Markdown with PyMuPDF4LLM, which keeps
    headings (#), tables (| ... |) and the labels printed inside charts, and
    drops the running page headers and footers.
  * Every block remembers its PDF page AND the page number printed on that
    page (they differ: e.g. printed page 4 of the World Bank report is PDF
    page 26), so answers can cite pages a reader can actually find.
  * Every block knows which section it belongs to, e.g.
    ["Ch. 1 Global Outlook", "Global context", "Commodity markets"], built
    from the PDF bookmarks plus the detected headings. This "section-aware"
    metadata drives chunking and is shown to the model.
  * Table and figure captions ("TABLE 1.1 Real GDP") are attached to their
    tables and figures.
  * Footnote markers (superscript numbers) are linked to the footnote or
    endnote they point to, so a fact hidden in a footnote is retrieved
    together with the sentence that references it.

A block is a plain dict:

    {"type": "text" | "table" | "figure",
     "text": "...",                  # cleaned Markdown text
     "page": 26,                     # 1-based PDF page (None for Markdown files)
     "page_label": "4",              # printed page number ("" if unknown)
     "section": ["Ch. 1 Global Outlook", "Summary"],
     "caption": "TABLE 1.1 Real GDP",  # table/figure (or its Source/Note lines)
     "notes": {"1": "Headline aggregate growth rates are ..."}}
"""

import contextlib
import io
import re
import unicodedata
from collections import Counter

import pymupdf
import pymupdf4llm

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
PICTURE_RE = re.compile(
    r"<!-- Start of picture text -->(.*?)<!-- End of picture text -->", re.S
)
FIGURE_TOKEN = "\x00FIGURE\x00"   # placeholder for chart text while splitting

# "TABLE 1.1 Real GDP", "FIGURE B1.1.2 An enabling environment for AI"
CAPTION_RE = re.compile(r"^(TABLE|FIGURE|Table|Figure)\s+[A-Z]?\d")
# Some captions are bold text rather than headings ("**TABLE 2.12 ...**").
# Only upper-case ones count, so a sentence like "Table 2.1 shows ..." doesn't.
TEXT_CAPTION_RE = re.compile(r"^(TABLE|FIGURE)\s+[A-Z]?\d")
# Boxes are self-contained mini-articles inside a chapter ("BOX 1.1 ...").
BOX_RE = re.compile(r"^BOX\s+\d", re.I)
# Chart panel titles inside a figure: "A. Energy and fertilizer prices"
PANEL_RE = re.compile(r"^[A-H]\.\s+\S")
# Lines that annotate a table/figure: "Source: ...", "Note: ...", "B. Daily prices ..."
ANNOTATION_RE = re.compile(r"^(Sources?|Notes?)\s*:|^[A-H]\.\s|^\(")
# Content of a superscript that is a footnote marker: "1", "12", "3,4"
NOTE_MARKER_RE = re.compile(r"^\d{1,3}(?:\s*,\s*\d{1,3})*$")
# A footnote/endnote definition: "1 Debt refers to ..." or "[^3] Breakdown ..."
NOTE_DEF_RE = re.compile(r"^(?:\[\^(\d{1,3})\]|(\d{1,3})\.?)\s+(?=[A-Z“\"‘(])(.+)$", re.S)
NOTE_REF_RE = re.compile(r"\[\^(\d{1,3})\]")
NOTES_SECTION_RE = re.compile(r"^(end)?notes$|^footnotes$", re.I)
REFERENCES_RE = re.compile(r"^references$|^bibliography$", re.I)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
def _superscript(match):
    """Turn <sup>1</sup> into a [^1] footnote marker; unwrap anything else."""
    inner = re.sub(r"[*_]", "", match.group(1)).strip()
    if NOTE_MARKER_RE.match(inner):
        return "".join(f"[^{n.strip()}]" for n in inner.split(","))
    # Not a footnote marker (PyMuPDF4LLM sometimes marks a whole line as
    # superscript); keep the words as normal text.
    return f" {inner} "


def clean_text(text):
    """Normalise one paragraph of PyMuPDF4LLM Markdown into plain, clean text."""
    # NFKC folds typographic ligatures (ﬁ -> fi, ﬂ -> fl) back into letters.
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"<sup>(.*?)</sup>", _superscript, text, flags=re.S)
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"\*\*|__", "", text)                       # bold
    text = re.sub(r"(?<![\w\\])_([^_\n]+?)_(?!\w)", r"\1", text)  # italics
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^\s*>\s?", "", line)                    # block quotes
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def table_cells(row):
    """Split a Markdown table row into cells, keeping empty first/last cells."""
    row = row.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|"):
        row = row[:-1]
    return row.split("|")


def clean_table(text):
    """Clean every cell of a Markdown table and drop rows that are empty."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [clean_text(cell).replace("\n", " ") for cell in table_cells(line)]
        if all(re.fullmatch(r":?-{3,}:?", c) for c in cells if c):
            rows.append("|" + "|".join("---" for _ in cells) + "|")   # separator row
        elif any(cells):
            rows.append("|" + "|".join(cells) + "|")
    return "\n".join(rows)


def _clean_heading(text):
    text = clean_text(text)
    text = NOTE_REF_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip(" :")


def _repeats_part(heading, part):
    """True if a heading just restates the chapter bookmark ("GLOBAL OUTLOOK" vs "Ch. 1 Global Outlook")."""
    part = re.sub(r"^ch\.\s*\d+\s*", "", part.lower())
    return part.startswith(heading) and len(heading) >= 0.6 * len(part)


def _is_letter_spaced(text):
    """True for decorative headings like 'U . S . G O V E R N M E N T'."""
    tokens = text.split()
    return len(tokens) > 4 and sum(len(t) == 1 for t in tokens) / len(tokens) > 0.6


# ---------------------------------------------------------------------------
# Page numbers and bookmarks
# ---------------------------------------------------------------------------
def printed_page_labels(doc):
    """
    Work out the page number printed on each PDF page.

    We look for a bare number in the top or bottom margin of every page and
    compute "PDF page - printed number". An offset that repeats on at least
    three pages is trusted (this filters out stray numbers such as footnote
    markers). Pages without a number inherit the offset of their neighbours.
    Documents can contain several numbering runs: the GPO report restarts its
    numbering for the financial section.
    """
    page_count = doc.page_count
    candidates = {}
    for index, page in enumerate(doc):
        height = page.rect.height
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                text = "".join(span["text"] for span in line["spans"]).strip()
                top, bottom = line["bbox"][1], line["bbox"][3]
                in_margin = bottom < 0.08 * height or top > 0.90 * height
                if in_margin and re.fullmatch(r"\d{1,3}", text):
                    candidates.setdefault(index + 1, []).append(index + 1 - int(text))

    counts = Counter(off for offs in candidates.values() for off in set(offs))
    trusted = {off for off, count in counts.items() if count >= 3}
    offsets = {}
    for page, offs in candidates.items():
        good = [off for off in offs if off in trusted]
        if good:
            offsets[page] = good[0]

    labels = []
    for page in range(1, page_count + 1):
        offset = offsets.get(page)
        if offset is None:
            before = next((offsets[p] for p in range(page - 1, 0, -1) if p in offsets), None)
            after = next((offsets[p] for p in range(page + 1, page_count + 1) if p in offsets), None)
            if before == after or before is None or after is None:
                offset = before if before is not None else after
        printed = page - offset if offset is not None else 0
        labels.append(str(printed) if printed >= 1 else "")
    return labels


def bookmark_parts(doc):
    """
    Map each PDF page to its top-level part (chapter) from the PDF bookmarks.

    Only top-level bookmarks are used, and "Box ..." bookmarks are skipped
    (boxes are detected from their headings instead, because a bookmark only
    says where a box starts, not where it ends). Bookmarks written as
    " -- East Asia and Pacific" are sub-parts of the chapter before them.
    """
    entries = []
    for level, title, page in doc.get_toc():
        is_sub = "--" in title
        title = " ".join(title.replace("--", " ").split())
        if level == 1 and page >= 1 and title and not title.lower().startswith("box "):
            entries.append((page, title, is_sub))
    entries.sort(key=lambda entry: entry[0])

    parts, chapter, sub = {}, None, None
    entry_index = 0
    for page in range(1, doc.page_count + 1):
        while entry_index < len(entries) and entries[entry_index][0] <= page:
            _, title, is_sub = entries[entry_index]
            if is_sub:
                sub = title
            else:
                chapter, sub = title, None
            entry_index += 1
        parts[page] = [p for p in (chapter, sub) if p]
    return parts


# ---------------------------------------------------------------------------
# Markdown -> blocks
# ---------------------------------------------------------------------------
class BlockParser:
    """
    Walks through a document's Markdown page by page and emits blocks.

    It keeps a stack of the headings seen so far, so every block knows its
    full section path. A table/figure caption stays "open" until the end of
    its page, so the table, the chart labels, the figure's description and
    its "Source:/Note:" lines are all tied to that caption. Ordinary body text
    that happens to share the page is not.
    """

    def __init__(self):
        self.blocks = []
        self.part = []      # chapter path from the PDF bookmarks
        self.stack = []     # [(level, title)] of the headings we are inside
        self.caption = ""   # open table/figure caption on the current page
        self.panels = []    # chart panel titles waiting for their figure
        self.caption_texts = 0  # text paragraphs tied to the open caption
        self.heading_pending = False  # True until the newest heading gets content

    def section(self):
        return self.part + [title for _level, title in self.stack]

    def feed_page(self, markdown, page=None, page_label="", part=None):
        if part is not None and part != self.part:
            self.part, self.stack = part, []        # a new chapter starts
        self.caption, self.panels = "", []

        # Put chart text in its own paragraph so it can be told apart.
        markdown = PICTURE_RE.sub(lambda m: f"\n\n{FIGURE_TOKEN}{m.group(1)}\n\n", markdown)
        for paragraph in re.split(r"\n\s*\n", markdown):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            lines = paragraph.splitlines()
            heading = HEADING_RE.match(lines[0])
            if heading:
                self._heading(len(heading.group(1)), heading.group(2), page, page_label)
                lines = lines[1:]
            # A paragraph can mix table rows and ordinary lines; split them.
            run, run_is_table = [], None
            for line in lines + [None]:
                is_table = line is not None and line.lstrip().startswith("|")
                if run and (line is None or is_table != run_is_table):
                    self._paragraph("\n".join(run).strip(), page, page_label)
                    run = []
                if line is not None:
                    run.append(line)
                    run_is_table = is_table

    def _heading(self, level, raw, page, page_label):
        title = _clean_heading(raw)
        # Big numbers or very long lines styled as headings are really text.
        if not re.search(r"[A-Za-z]{3}", title) or len(title) > 150:
            self._paragraph(raw, page, page_label)
            return
        if _is_letter_spaced(title):
            return
        if CAPTION_RE.match(title):
            self.caption, self.panels, self.caption_texts = title, [], 0
            return
        if self.caption and PANEL_RE.match(title):
            self.panels.append(title)
            return

        title = re.sub(r"\s*\(continued\)$", "", title, flags=re.I)
        lowered = title.lower()
        if any(lowered == t.lower() for _l, t in self.stack):
            return                                   # "(continued)" on a new page
        if self.part and _repeats_part(lowered, self.part[-1]):
            return                                   # repeats the chapter bookmark
        if BOX_RE.match(title) and self.stack:
            level = min(l for l, _t in self.stack)   # a box sits beside top sections
        if self.stack and self.stack[-1][0] == level and self.heading_pending:
            # Two headings in a row at the same level form one title, e.g.
            # "Consolidated Balance Sheets" + "As of September 30, 2018 and 2017".
            title = f"{self.stack.pop()[1]} — {title}"
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        self.stack.append((level, title))
        self.heading_pending = True
        self.caption, self.panels = "", []

    def _paragraph(self, raw, page, page_label):
        if raw.startswith(FIGURE_TOKEN):
            labels = clean_text(raw[len(FIGURE_TOKEN):]).replace("\n", " ")
            if len(labels) < 25:
                return                               # decorative text in a photo
            parts = [self.caption] if self.caption else []
            parts += self.panels
            parts.append(f"Chart labels: {labels}")
            self.panels = []
            self._emit("figure", "\n".join(parts), page, page_label, self.caption)
            return
        if raw.lstrip().startswith("|"):
            table = clean_table(raw)
            if table.count("\n") >= 1:
                self._emit("table", table, page, page_label, self.caption)
            return
        text = clean_text(raw)
        if len(text) < 4 or not re.search(r"[A-Za-z0-9]", text):
            return
        title = _clean_heading(raw)
        if TEXT_CAPTION_RE.match(title) and "\n" not in text and len(title) <= 150:
            self.caption, self.panels, self.caption_texts = title, [], 0
            return
        # Tie a paragraph to the open caption only if it is the figure's
        # one-paragraph description or a Source/Note/panel annotation.
        caption = ""
        if self.caption and (self.caption_texts == 0 or ANNOTATION_RE.match(text)):
            caption = self.caption
            self.caption_texts += 1
        self._emit("text", text, page, page_label, caption)

    def _emit(self, kind, text, page, page_label, caption):
        self.heading_pending = False
        self.blocks.append({
            "type": kind,
            "text": text,
            "page": page,
            "page_label": page_label,
            "section": self.section(),
            "caption": caption,
            "notes": {},
        })


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
def drop_repeated_lines(blocks, min_pages=5):
    """
    Remove leftover running headers/footers.

    PyMuPDF4LLM drops most of them, but a few survive, e.g. "U. S. Government
    Publishing Office | Financial Year In Review". A short line that appears
    on many different pages is page furniture, not content. Source/Note lines
    under figures also repeat, but they belong to their figure, so we keep them.
    """
    def key(block):
        return re.sub(r"\d+", "#", block["text"].lower())

    pages = {}
    for block in blocks:
        if block["type"] == "text" and len(block["text"]) < 120:
            pages.setdefault(key(block), set()).add(block["page"])
    repeated = {k for k, p in pages.items() if len(p) >= min_pages}
    return [
        b for b in blocks
        if not (b["type"] == "text" and key(b) in repeated and not ANNOTATION_RE.match(b["text"]))
    ]


def fix_th_ligature(blocks):
    """
    Repair words whose "Th" ligature lost its "h" ("Te" -> "The", "Tis" -> "This").

    The World Bank report's body font draws "Th" as one glyph that PDF text
    extraction turns into a bare "T". A capitalised word "T<rest>" is repaired
    when "th<rest>" is a more common word in the same document than
    "t<rest>", e.g. "the" vs "te", "three" vs "tree". Documents without the
    problem are left untouched.
    """
    words = Counter(w for b in blocks for w in re.findall(r"[A-Za-z]+", b["text"]))
    if words["Te"] < 5:
        return

    def repair(match):
        rest = match.group(1)
        fixed = words["th" + rest] + words["Th" + rest]
        if fixed > words["t" + rest]:
            return "Th" + rest
        return match.group(0)

    pattern = re.compile(r"\bT([a-z]+)\b")
    for block in blocks:
        block["text"] = pattern.sub(repair, block["text"])
        block["caption"] = pattern.sub(repair, block["caption"])
        block["section"] = [pattern.sub(repair, s) for s in block["section"]]


def link_notes(blocks):
    """
    Attach footnote/endnote text to every block that references it.

    A marker [^n] is resolved to a definition of note n on the SAME page
    (footnotes, table notes) or, failing that, to the next definition of n in
    an "Endnotes"/"Notes" section (endnotes at the back of a report). Note
    definitions are rewritten as "Note n: ..." so they read naturally when
    they are retrieved on their own.
    """
    definitions = []   # [(index, number, page, in_notes_section)]
    last_note = None
    for index, block in enumerate(blocks):
        in_notes = any(NOTES_SECTION_RE.match(s) for s in block["section"])
        match = NOTE_DEF_RE.match(block["text"].lstrip("-• ")) if block["type"] == "text" else None
        if match:
            number = match.group(1) or match.group(2)
            block["text"] = f"Note {number}: {match.group(3).strip()}"
            definitions.append([index, number, block["page"], in_notes])
            last_note = len(definitions) - 1
        elif in_notes and last_note is not None:
            # A follow-on paragraph or table inside an endnote belongs to it.
            definitions[last_note].append(index)
        elif not in_notes:
            last_note = None

    def note_text(definition):
        index = definition[0]
        parts = [blocks[index]["text"].split(":", 1)[1].strip()]
        parts += [blocks[i]["text"] for i in definition[4:]]
        return "\n".join(parts)

    for index, block in enumerate(blocks):
        for number in dict.fromkeys(NOTE_REF_RE.findall(block["text"])):
            same_page = [d for d in definitions
                         if d[1] == number and d[2] == block["page"] and d[0] != index]
            endnotes = [d for d in definitions
                        if d[1] == number and d[3] and (d[2] or 0) >= (block["page"] or 0)]
            chosen = (same_page or endnotes or [None])[0]
            if chosen:
                block["notes"][number] = note_text(chosen)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def pdf_to_markdown_pages(path):
    """Convert a PDF to one Markdown string per page (headers/footers dropped)."""
    # PyMuPDF4LLM prints an OCR notice we don't need (these PDFs have real text).
    with contextlib.redirect_stdout(io.StringIO()):
        pages = pymupdf4llm.to_markdown(
            path, page_chunks=True, header=False, footer=False,
            use_ocr=False, show_progress=False,
        )
    return [page["text"] for page in pages]


def load_pdf(path):
    """Load a PDF into blocks (see the module docstring for the block format)."""
    doc = pymupdf.open(path)
    labels = printed_page_labels(doc)
    parts = bookmark_parts(doc)
    parser = BlockParser()
    for index, markdown in enumerate(pdf_to_markdown_pages(path)):
        page = index + 1
        parser.feed_page(markdown, page, labels[index], parts.get(page))
    doc.close()
    blocks = drop_repeated_lines(parser.blocks)
    fix_th_ligature(blocks)
    link_notes(blocks)
    return blocks


def load_markdown(path):
    """Load a Markdown file into blocks (no pages; sections come from # headings)."""
    with open(path, "r", encoding="utf-8") as file:
        text = file.read()
    parser = BlockParser()
    parser.feed_page(text)
    link_notes(parser.blocks)
    return parser.blocks


def is_reference_list(block):
    """True for blocks inside a bibliography ("References") section."""
    return any(REFERENCES_RE.match(s) for s in block["section"])
