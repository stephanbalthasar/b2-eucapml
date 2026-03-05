#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a compact JSON index from a DOCX booklet with:

- Chapters: detected by Heading 1 (consecutively numbered)
- Paragraphs: non-empty body paragraphs only (consecutively numbered)

Footnotes and tables are intentionally ignored.

Output schema:
{
  "paragraphs": [
    {
      "para_num": int,                # 1..N over body paragraphs (headings/empties skipped)
      "id": "p_<hash8>",              # stable-ish id from content hash
      "text": "Paragraph text",
      "chapter_num": int | null,      # chapter this paragraph belongs to (if any)
      "chapter_title": str | null,    # chapter title (if any)
      "style": "Word style name",
      "is_numbered": bool             # list-numbering flag from Word numPr
    },
    ...
  ],
  "chapters": [
    {
      "chapter_num": int,             # 1..M in document order
      "title": "Heading text",
      "text": "Concatenated chapter body text",
      "paragraph_ids": ["p_<hash8>", ...]
    },
    ...
  ]
}
"""

from __future__ import annotations

import argparse
import json
import sys
from hashlib import blake2b
from pathlib import Path
from typing import List, Dict, Any, Optional

from docx import Document
from docx.text.paragraph import Paragraph


# ---------- utilities ----------

def _hash_id(text: str) -> str:
    h = blake2b(digest_size=6)
    h.update(text.strip().encode("utf-8"))
    return "p_" + h.hexdigest()


def _clean_ws(s: str) -> str:
    return " ".join((s or "").split()).strip()


def _is_heading1(par: Paragraph) -> bool:
    """
    Detect Heading 1, including localized/custom base styles.
    """
    name = _clean_ws(getattr(par.style, "name", "")).lower()

    # Common localized names for Heading 1
    h1_names = {
        "heading 1", "überschrift 1", "titre 1", "título 1",
        "intestazione 1", "rubrica 1", "rubrique 1",
        "naslov 1", "başlık 1", "заголовок 1", "標題 1", "見出し 1", "제목 1",
    }
    if name in h1_names:
        return True

    # Generic fallback: startswith token + " 1"
    startswith_tokens = (
        "heading", "überschrift", "titre", "título", "rubrica", "rubrique",
        "intestazione", "naslov", "başlık", "заголовок", "標題", "見出し", "제목"
    )
    if any(name.startswith(tok + " ") and name.endswith("1") for tok in startswith_tokens):
        return True

    # Detect inheritance chain (best-effort)
    try:
        base = par.style.base_style
        while base is not None:
            bname = _clean_ws(getattr(base, "name", "")).lower()
            if bname in h1_names:
                return True
            if any(bname.startswith(tok + " ") and bname.endswith("1") for tok in startswith_tokens):
                return True
            base = base.base_style
    except Exception:
        pass

    return False


def _is_numbered_paragraph(par: Paragraph) -> bool:
    """
    Detect list numbering via Word numbering properties (numPr).
    """
    try:
        pPr = par._p.pPr  # access underlying oxml
        return (pPr is not None) and (pPr.numPr is not None)
    except Exception:
        return False


# ---------- main builder (paragraphs + chapters only) ----------

def build_index(docx_path: str, verbose: bool = False) -> Dict[str, Any]:
    doc = Document(docx_path)

    chapters: List[Dict[str, Any]] = []
    paragraphs: List[Dict[str, Any]] = []

    chapter_num: int = 0
    chapter_title: Optional[str] = None
    chapter_buf: List[str] = []
    chapter_para_ids: List[str] = []

    para_counter = 0

    def _flush_chapter():
        nonlocal chapters, chapter_num, chapter_title, chapter_buf, chapter_para_ids
        if chapter_title is not None:
            chapters.append({
                "chapter_num": chapter_num,
                "title": chapter_title,
                "text": "\n".join(chapter_buf).strip(),
                "paragraph_ids": list(chapter_para_ids),
            })

    # Only iterate over PARAGRAPHS; ignore tables and any other content types
    for par in doc.paragraphs:
        text = _clean_ws(par.text)
        style = _clean_ws(getattr(par.style, "name", ""))

        # New chapter?
        if _is_heading1(par):
            _flush_chapter()
            chapter_num += 1  # consecutive numbering for chapters
            chapter_title = text or f"Chapter {chapter_num}"
            chapter_buf = []
            chapter_para_ids = []
            if verbose:
                print(f"[H1] #{chapter_num}: {chapter_title}")
            continue  # heading itself is NOT indexed as a content paragraph

        # Skip empty lines
        if not text:
            continue

        # Only index non-empty body paragraphs
        para_counter += 1  # consecutive numbering for body paragraphs
        pid = _hash_id(f"{chapter_num}|{text}|{style}|p{para_counter}")
        is_num = _is_numbered_paragraph(par)

        record: Dict[str, Any] = {
            "para_num": para_counter,
            "id": pid,
            "text": text,
            "chapter_num": chapter_num or None,
            "chapter_title": chapter_title if chapter_num else None,
            "style": style,
            "is_numbered": bool(is_num),
        }
        paragraphs.append(record)

        # Accumulate into current chapter
        if chapter_num:
            chapter_buf.append(text)
            chapter_para_ids.append(pid)

    # Close the last chapter if any
    _flush_chapter()

    return {"paragraphs": paragraphs, "chapters": chapters}


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Build booklet index with chapters and paragraphs only")
