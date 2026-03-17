# app/router.py
# -----------------------------------------------------------------------------
# HEURISTIC ROUTER — CANONICAL-LEVEL MATCHING (EXACT + FUZZY)
#
# Routing rule:
#   Count how many CANONICAL gazetteer entries (concepts/cases/aliases groups)
#   match the query—either EXACT or FUZZY.
#
#   If >= 2 canonical hits → RAG
#   Else                  → Chat
#
# This prevents alias inflation (Spector, Spector Photo, C‑45/08)
# while still allowing fuzzy typo handling (Lafonat → Lafonta).
#
# No LLM calls. Deterministic. Very fast.
# -----------------------------------------------------------------------------

from __future__ import annotations
import os
import unicodedata
import difflib
from typing import Dict, List, Set


# -----------------------------------------------------------------------------
# Normalization utilities (unicode + hyphens + whitespace)
# -----------------------------------------------------------------------------
_HYPHEN_MAP = dict.fromkeys(map(ord, "‑–—−—"), ord("-"))

def _norm(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = t.translate(_HYPHEN_MAP)
    t = t.lower().strip()
    return " ".join(t.split())


# -----------------------------------------------------------------------------
# File reader
# -----------------------------------------------------------------------------
def _read_file(path: str) -> List[str]:
    if not os.path.exists(path):
        raise RuntimeError(f"Gazetteer file missing: {path}")
    out = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                out.append(s)
    return out


# -----------------------------------------------------------------------------
# Parse alias lines of the form:
#   Canonical | Alias1 | Alias2 | ...
# -----------------------------------------------------------------------------
def _parse_aliases(path: str) -> Dict[str, Set[str]]:
    """
    Returns {canonical: {alias1, alias2, ... canonical}}
    """
    rows = _read_file(path)
    mapping: Dict[str, Set[str]] = {}

    for raw in rows:
        parts = [p.strip() for p in raw.split("|") if p.strip()]
        if not parts:
            continue

        canonical = parts[0]
        aliases = parts[1:] if len(parts) > 1 else []

        s = mapping.setdefault(canonical, set())
        s.add(canonical)
        for a in aliases:
            s.add(a)

    return mapping


# -----------------------------------------------------------------------------
# Build canonical gazetteer:
#   {canonical: {all aliases}}
# -----------------------------------------------------------------------------
def _load_canonical_map() -> Dict[str, Set[str]]:
    """
    Loads:
      - canonical concepts (one per line)
      - canonical cases (one per line)
      - canonical alias groups (one line = one concept)
    Produces a single map:
      canonical -> set(all variants)
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Your real paths:
    concepts_path = os.path.join(base_dir, "..", "mentor", "rag", "gazetteer_concepts.txt")
    cases_path    = os.path.join(base_dir, "..", "mentor", "rag", "gazetteer_cases.txt")
    aliases_path  = os.path.join(base_dir, "..", "mentor", "rag", "gazetteer_aliases.txt")

    # Load base lists
    concepts = _read_file(concepts_path)
    cases    = _read_file(cases_path)

    # Load aliases
    alias_map = _parse_aliases(aliases_path)

    canonical_map: Dict[str, Set[str]] = {}

    # Concepts (each line = one canonical)
