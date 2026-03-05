#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a compact JSON index from a DOCX booklet:

- Chapters: detected by Heading 1 (consecutively numbered)
- Paragraphs: non-empty body paragraphs only (consecutively numbered)
- No tables, no footnotes, no inlined numbering artifacts

CLI:
  python mentor/booklet/build_booklet_index.py \
    --src private-src/assets/booklet.docx \
    --out private-src/artifacts/booklet_index.json \
    --expect-chapters 7 --expect-paragraphs 190 --verbose
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


# ---------- main builder ----------

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

    # Only body paragraphs; ignore tables/headers/footers
    for par in doc.paragraphs:
        text = _clean_ws(par.text)
        style = _clean_ws(getattr(par.style, "name", ""))

        # New chapter?
        if _is_heading1(par):
            _flush_chapter()
            chapter_num += 1
            chapter_title = text or f"Chapter {chapter_num}"
            chapter_buf = []
            chapter_para_ids = []
            if verbose:
                print(f"[H1] #{chapter_num}: {chapter_title}")
            continue  # don't index heading lines as body paragraphs

        # Skip empty lines
        if not text:
            continue

        # Index non-empty body paragraphs
        para_counter += 1
        pid = _hash_id(f"{chapter_num}|{text}|{style}|p{para_counter}")
        is_num = _is_numbered_paragraph(par)

        paragraphs.append({
            "para_num": para_counter,
            "id": pid,
            "text": text,
            "chapter_num": chapter_num or None,
            "chapter_title": chapter_title if chapter_num else None,
            "style": style,
            "is_numbered": bool(is_num),
        })

        if chapter_num:
            chapter_buf.append(text)
            chapter_para_ids.append(pid)

    _flush_chapter()
    return {"paragraphs": paragraphs, "chapters": chapters}


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Build booklet index (chapters + body paragraphs only)")
    ap.add_argument("--src", required=True, help="Path to DOCX (e.g., private-src/assets/booklet.docx)")
    ap.add_argument("--out", required=True, help="Path to output JSON (e.g., private-src/artifacts/booklet_index.json)")
    ap.add_argument("--expect-chapters", type=int, default=None, help="Fail if chapter count differs")
    ap.add_argument("--expect-paragraphs", type=int, default=None, help="Fail if paragraph count differs")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = ap.parse_args(argv)

    src = Path(args.src)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        print(f"ERROR: source DOCX not found: {src}", file=sys.stderr)
        return 2

    index = build_index(str(src), verbose=args.verbose)
    ch = len(index.get("chapters", []))
    ps = len(index.get("paragraphs", []))

    print(f"Indexed {ch} chapters, {ps} paragraphs from {src}")

    # Expectations guardrail (optional; fail hard if mismatched)
    if args.expect_chapters is not None and ch != args.expect_chapters:
        print(f"ERROR: Expected {args.expect_chapters} chapters, got {ch}", file=sys.stderr)
        return 3
    if args.expect_paragraphs is not None and ps != args.expect_paragraphs:
        print(f"ERROR: Expected {args.expect_paragraphs} paragraphs, got {ps}", file=sys.stderr)
        return 4

    out.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
