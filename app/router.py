# app/router.py
# -----------------------------------------------------------------------------
# FIX #1 — STABLE CANONICAL-LEVEL ROUTER
#
# Routing rule:
#   Count how many CANONICAL gazetteer entries match the query.
#   - Exact match allowed
#   - Fuzzy match allowed ONLY for single‑word aliases, with safe threshold
#   - Each canonical counts AT MOST 1 (dedup)
#   - Stopwords never fuzzy‑matched
#
# Threshold:
#   >= 2 canonical hits → RAG
#   <  2 → Chat
#
# This guarantees:
#   - “ECJ Lafonat” → 2 concepts (ECJ + Lafonta)
#   - “summarize the ECJ decision in Lafonat” → 2 concepts ONLY
#   - Never inflated counts (no 4‑concept bugs)
#   - Typo tolerance where intended
# -----------------------------------------------------------------------------

from __future__ import annotations
import os
import unicodedata
import difflib
from typing import Dict, Set, List


# -----------------------------------------------------------------------------
# Normalization utilities
# -----------------------------------------------------------------------------
_HYPHEN_MAP = dict.fromkeys(map(ord, "‑–—−—"), ord("-"))
_STOPWORDS = {
    "can", "you", "tell", "me", "about", "what", "summarize",
    "summarise", "please", "in", "the", "a", "an", "on",
    "of", "decision", "case", "ec", "cj", "decision"
}

def _norm(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = t.translate(_HYPHEN_MAP)
    t = t.lower().strip()
    return " ".join(t.split())


# -----------------------------------------------------------------------------
# Gazetteer file readers
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


def _parse_aliases(path: str) -> Dict[str, Set[str]]:
    """
    Parse alias entries:
      Canonical | Alias1 | Alias2 | ...
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
# Build canonical gazetteer: canonical -> {all variants}
# -----------------------------------------------------------------------------
def _load_canonical_map() -> Dict[str, Set[str]]:
    base_dir = os.path.dirname(os.path.abspath(__file__))

    concepts_path = os.path.join(base_dir, "..", "mentor", "rag", "gazetteer_concepts.txt")
    cases_path    = os.path.join(base_dir, "..", "mentor", "rag", "gazetteer_cases.txt")
    aliases_path  = os.path.join(base_dir, "..", "mentor", "rag", "gazetteer_aliases.txt")

    concepts = _read_file(concepts_path)
    cases    = _read_file(cases_path)
    alias_map = _parse_aliases(aliases_path)

    canonical_map: Dict[str, Set[str]] = {}

    # Concepts
    for c in concepts:
        canonical_map.setdefault(c, set()).add(c)

    # Cases
    for c in cases:
        canonical_map.setdefault(c, set()).add(c)

    # Alias entries
    for canon, alset in alias_map.items():
        s = canonical_map.setdefault(canon, set())
        for a in alset:
            s.add(a)

    # Normalize keys + variants
    out: Dict[str, Set[str]] = {}
    for canon, variants in canonical_map.items():
        canon_n = _norm(canon)
        out[canon_n] = {_norm(v) for v in variants if v}

    return out


_CANONICAL_MAP: Dict[str, Set[str]] = _load_canonical_map()


# -----------------------------------------------------------------------------
# Matching logic: EXACT + SAFE FUZZY
# -----------------------------------------------------------------------------
def _canonical_matches(variants: Set[str], q: str, q_words: List[str]) -> bool:
    """
    Returns True if ANY variant of the canonical entry matches the query.
    Rules:
      • Exact substring match allowed (multi-word OK)
      • Fuzzy match ONLY for single‑word variants
      • Ignore stopwords + very short tokens
    """
    for v in variants:

        # Exact match (preferred)
        if v and v in q:
            return True

        # Fuzzy match ONLY for single‑word variants
        if " " in v:
            continue

        for w in q_words:
            if len(w) < 5:
                continue
            if w in _STOPWORDS:
                continue

            ratio = difflib.SequenceMatcher(None, w, v).ratio()
            if ratio >= 0.88:
                return True

    return False


# -----------------------------------------------------------------------------
# PUBLIC API
# -----------------------------------------------------------------------------
def route(user_query: str, *, threshold: int = 2) -> Dict[str, int]:
    """
    Count how many CANONICAL gazetteer entries match the query.
    >= threshold → RAG
    < threshold → Chat
    """
    q = _norm(user_query)
    if not q:
        return {"mode": "chat", "count": 0}

    q_words = q.split()
    hits = 0

    for canon, variants in _CANONICAL_MAP.items():
        if _canonical_matches(variants, q, q_words):
            hits += 1

    mode = "rag" if hits >= threshold else "chat"
    return {"mode": mode, "count": hits}
