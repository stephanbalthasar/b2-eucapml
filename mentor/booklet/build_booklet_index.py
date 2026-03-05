#!/usr/bin/env python3
"""
Build an index (chapters + numbered paragraphs) from a DOCX.

- Chapters: paragraphs styled Heading 1 / Überschrift 1 (robust detection with XML outline level fallback)
- Paragraphs: those styled "Standard with para numbering" (plus common EN/DE variants)
              or any paragraph that has list numbering (XML numPr) as a fallback
- Outputs a JSON file with chapter and paragraph numbering:
    * chapter_number: 1..N (in encounter order)
    * global_para_number: consecutive across the whole booklet
    * chapter_local_para_number: consecutive within each chapter
"""

import argparse
import datetime as _dt
import json
import os
import sys
from typing import List, Dict, Any

from docx import Document  # pip install python-docx


HEADING1_STYLE_NAMES = {
    "Heading 1",
    "Überschrift 1",
    "Überschrift1",
}

BODY_NUMBERED_STYLES = {
    "Standard with para numbering",
    "Standard mit Nummerierung",
    "Normal mit Nummerierung",
    "Normal with numbering",
    "List Paragraph",  # Word built-in seen in some templates
}


def _style_name(p) -> str:
    try:
        if p.style and p.style.name:
            return str(p.style.name)
    except Exception:
        pass
    return ""


def _has_numbering(p) -> bool:
    """
    Fallback: treat any paragraph with w:numPr as numbered.
    python-docx does not expose numbering at high level, so inspect the underlying XML.
    """
    try:
        pPr = p._p.pPr
        return (pPr is not None) and (pPr.numPr is not None)
    except Exception:
        return False


def _is_heading1(p) -> bool:
    name = _style_name(p)
    if name in HEADING1_STYLE_NAMES:
        return True
    # Tolerant match for templates with slightly different naming that end with " 1"
    if name and ("Heading" in name or "Überschrift" in name) and name.rstrip().endswith("1"):
        return True
    # XML outline level (0 == Heading 1)
    try:
        pPr = p._p.pPr
        if pPr is not None and pPr.outlineLvl is not None:
            return pPr.outlineLvl.val == 0
    except Exception:
        pass
    return False


def _is_numbered_body_para(p) -> bool:
    name = _style_name(p)
    if name in BODY_NUMBERED_STYLES:
        return True
    # Robust fallback: any list-numbered para in the DOCX
    return _has_numbering(p)


def build_index(docx_path: str) -> Dict[str, Any]:
    if not os.path.exists(docx_path):
        raise FileNotFoundError(f"Input DOCX not found: {docx_path}")

    doc = Document(docx_path)

    chapters: List[Dict[str, Any]] = []
    current = None
    chapter_count = 0
    global_para_no = 0

    def ensure_current(default_title: str = ""):
        nonlocal current, chapter_count
        if current is None:
            chapter_count += 1
            current = {"chapter_number": chapter_count, "title": default_title, "paragraphs": []}
            chapters.append(current)

    for p in doc.paragraphs:
        text = (p.text or "").strip()
        if not text:
            continue

        if _is_heading1(p):
            chapter_count += 1
            current = {"chapter_number": chapter_count, "title": text, "paragraphs": []}
            chapters.append(current)
            continue

        if _is_numbered_body_para(p):
            ensure_current(default_title="")
            global_para_no += 1
            local_no = len(current["paragraphs"]) + 1
            current["paragraphs"].append({
                "global_para_number": global_para_no,
                "chapter_local_para_number": local_no,
                "text": text,
            })

    payload = {
        "generated_at": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source": docx_path.replace("\\", "/"),
        "stats": {
            "chapters_detected": len(chapters),
            "paragraphs_indexed": sum(len(ch["paragraphs"]) for ch in chapters),
        },
        "chapters": chapters,
    }
    return payload


def main():
    parser = argparse.ArgumentParser(description="Build JSON index from a DOCX booklet.")
    parser.add_argument("--input", required=True, help="Path to assets/booklet.docx")
    parser.add_argument("--output", required=True, help="Path to artifacts/booklet_index.json")
    args = parser.parse_args()

    payload = build_index(args.input)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Wrote {args.output} with {payload['stats']['chapters_detected']} chapters / {payload['stats']['paragraphs_indexed']} paragraphs.")


if __name__ == "__main__":
    sys.exit(main())
