"""
pdf_processing.py
=================
Shared parsing + hierarchical (parent / child) chunking used by both
ingest_pdfs.py and parse_pdf.py, so the two never drift apart.

Pipeline per PDF:
  extract pages  ->  full_text (+ page offset map, + table of contents)
  full_text      ->  PARENT sections (large, ~1000 tokens)   [fed to the LLM]
  each parent    ->  CHILD chunks   (small, ~180 tokens)      [what we search]

Every child knows its parent_id, its section_path (from the PDF's TOC when
available), and its page range - so retrieval can later expand a matched
child into its full parent section (small-to-big) and cite pages.

Sizing note: all-MiniLM-L6-v2 truncates at 256 tokens. Children are kept at
~180 tokens so the *entire* chunk (plus a short contextual header) is actually
embedded - the old 2000-char chunks were being silently cut in half.

The chunking core takes *splitter objects* as arguments, so tests can inject
cheap character-based splitters while production injects token-based ones.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Child:
    index: int          # global child index within the document
    text: str
    start: int          # char offset in full_text
    end: int
    page_start: Optional[int]
    page_end: Optional[int]
    section_path: str


@dataclass
class Parent:
    index: int
    text: str
    start: int
    end: int
    page_start: Optional[int]
    page_end: Optional[int]
    section_path: str
    children: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# PDF -> pages / full text / offsets / TOC
# ---------------------------------------------------------------------------
def extract_pages(pdf_path):
    """Return (pages, toc). pages = [{'page': 1-based int, 'text': str}, ...]."""
    import fitz  # imported here so tests that don't touch PDFs don't need it
    doc = fitz.open(pdf_path)
    pages = [{"page": i + 1, "text": page.get_text()} for i, page in enumerate(doc)]
    try:
        toc = [(lvl, title, pg) for lvl, title, pg in (doc.get_toc() or [])]
    except Exception:
        toc = []
    doc.close()
    return pages, toc


def join_pages(pages):
    """Join page texts with '\\n' (matching the original pipeline) and build a
    char-offset -> page map so any span can be mapped back to page numbers."""
    full_text = "\n".join(p["text"] for p in pages)
    starts, page_nums, cursor = [], [], 0
    for p in pages:
        starts.append(cursor)
        page_nums.append(p["page"])
        cursor += len(p["text"]) + 1  # +1 for the '\n' separator
    return full_text, starts, page_nums


def page_range(starts, page_nums, span_start, span_end):
    """Map a [span_start, span_end) char span to (page_start, page_end)."""
    if not starts:
        return None, None
    i = bisect.bisect_right(starts, span_start) - 1
    j = bisect.bisect_right(starts, max(span_start, span_end - 1)) - 1
    i = max(0, min(i, len(page_nums) - 1))
    j = max(0, min(j, len(page_nums) - 1))
    return page_nums[i], page_nums[j]


def section_path_for_page(toc, page, doc_title, max_level=2):
    """Nearest preceding heading chain for a page, e.g. 'Compass > Activities'."""
    if not page:
        return doc_title
    path = []
    for lvl, title, pg in toc:
        if pg is None:
            continue
        if pg <= page and lvl <= max_level:
            path = path[: lvl - 1] + [title.strip()]
        elif pg > page:
            break  # TOC is in reading order; nothing further applies
    return " > ".join([doc_title] + path) if path else doc_title


# ---------------------------------------------------------------------------
# Locating chunk text back inside its source (to recover char offsets)
# ---------------------------------------------------------------------------
def _locate(haystack, needle, cursor):
    """Find `needle` at/after `cursor`; return (start, next_cursor).
    Falls back gracefully if the exact substring can't be found."""
    if not needle:
        return cursor, cursor
    idx = haystack.find(needle, cursor)
    if idx == -1:
        idx = haystack.find(needle)          # retry from the top
    if idx == -1:
        return cursor, cursor                # give up: keep cursor where it was
    nxt = idx + max(1, int(len(needle) * 0.5))
    return idx, nxt


# ---------------------------------------------------------------------------
# Hierarchical split
# ---------------------------------------------------------------------------
def split_hierarchical(full_text, parent_splitter, child_splitter,
                       starts=None, page_nums=None, toc=None, doc_title=""):
    """
    Split full_text into Parent sections and, within each, Child chunks.
    `*_splitter` are objects exposing `.split_text(str) -> list[str]`
    (e.g. langchain RecursiveCharacterTextSplitter).
    """
    starts = starts or []
    page_nums = page_nums or []
    toc = toc or []

    parents = []
    parent_texts = parent_splitter.split_text(full_text)

    pcursor = 0
    child_counter = 0
    for p_idx, ptext in enumerate(parent_texts):
        p_start, pcursor = _locate(full_text, ptext, pcursor)
        p_end = p_start + len(ptext)
        p_ps, p_pe = page_range(starts, page_nums, p_start, p_end)
        p_section = section_path_for_page(toc, p_ps, doc_title)

        parent = Parent(index=p_idx, text=ptext, start=p_start, end=p_end,
                        page_start=p_ps, page_end=p_pe, section_path=p_section)

        ccursor = 0
        for ctext in child_splitter.split_text(ptext):
            c_local, ccursor = _locate(ptext, ctext, ccursor)
            c_start = p_start + c_local
            c_end = c_start + len(ctext)
            c_ps, c_pe = page_range(starts, page_nums, c_start, c_end)
            parent.children.append(Child(
                index=child_counter, text=ctext, start=c_start, end=c_end,
                page_start=c_ps, page_end=c_pe, section_path=p_section,
            ))
            child_counter += 1

        parents.append(parent)
    return parents


# ---------------------------------------------------------------------------
# Token-based splitters (production) - sized against the embedding tokenizer
# ---------------------------------------------------------------------------
def make_token_splitter(tokenizer, chunk_tokens, overlap_tokens):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=chunk_tokens,
        chunk_overlap=overlap_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------
def parse_tags(raw):
    """Catalog 'tags' cell -> clean list[str]. Prefers comma/semicolon; never
    splits phrases on single spaces (so 'hate speech' stays one tag)."""
    if not raw:
        return []
    s = str(raw).strip()
    for sep in ("\n", ",", ";", "|", "•"):
        if sep in s:
            return [t.strip(" -•\t") for t in s.split(sep) if t.strip(" -•\t")]
    return [s] if s else []


def contextual_header(title, section_path, tags_text):
    """Short header prepended to a child ONLY for embedding (not stored as text).
    Gives an isolated chunk enough context to be retrieved accurately."""
    bits = []
    if title:
        bits.append(str(title))
    if section_path and section_path != title:
        bits.append(f"Section: {section_path}")
    if tags_text:
        bits.append(f"Topics: {tags_text}")
    return " | ".join(bits)
