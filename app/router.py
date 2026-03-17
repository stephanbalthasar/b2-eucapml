# app/router.py
# -----------------------------------------------------------------------------
# HEURISTIC ROUTER (GAZETTEER + FUZZY)
#
# Logic:
#   - Load gazetteer terms: concepts, cases, aliases
#   - Normalize unicode, hyphens, whitespace
#   - For each user query: exact OR fuzzy match against gazetteer terms
#   - Count how many UNIQUE gazetteer terms matched (no clustering needed)
#   - If >= 2 hits → RAG mode
#   - Else → Chat mode
#
# No LLM calls. Deterministic. Very fast.
# -----------------------------------------------------------------------------

from __future__ import annotations
import os
import unicodedata
import difflib
from typing import List, Set, Dict


# -----------------------------------------------------------------------------
# Utility: normalize unicode + hyphens + collapse whitespace
# -----------------------------------------------------------------------------
_HYPHEN_MAP = dict.fromkeys(map(ord, "‑–—−—"), ord("-"))  # NFKC handles most, this is insurance

def _norm(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = t.translate(_HYHEN_MAP) if False else t.translate(_HYPHEN_MAP)  # keep compatibility if needed
    t = t.lower().strip()
    return " ".join(t.split())


# -----------------------------------------------------------------------------
# 1) Load gazetteer files (concepts, cases, aliases)
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


def _load_aliases(path: str) -> Dict[str, Set[str]]:
    """
    NEW PARSER:
    Each line is: Canonical | alias1 | alias2 | ...
    """
    mapping: Dict[str, Set[str]] = {}
    rows = _read_file(path)

    for raw in rows:
        # Split on "|" (pipe separator)
        parts = [p.strip() for p in raw.split("|") if p.strip()]
        if not parts:
            continue

        canonical = parts[0]
        aliases = parts[1:] if len(parts) > 1 else []

        s = mapping.setdefault(canonical, set())
        for a in aliases:
            if a and a != canonical:
                s.add(a)
        # Always include canonical in its own alias set (useful for matching)
        s.add(canonical)

    return mapping


# -----------------------------------------------------------------------------
# BUILD FLAT GAZETTEER TERM LIST
# -----------------------------------------------------------------------------
def _load_gazetteer() -> Set[str]:
    """
    Loads all canonical terms and all aliases into one unified set.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # These are your existing files in mentor/rag/
    concepts_path = os.path.join(base_dir, "..", "mentor", "rag", "gazetteer_concepts.txt")
    cases_path    = os.path.join(base_dir, "..", "mentor", "rag", "gazetteer_cases.txt")
    aliases_path  = os.path.join(base_dir, "..", "mentor", "rag", "gazetteer_aliases.txt")

    # Read base concepts + cases
    concept_list = _read_file(concepts_path)
    case_list    = _read_file(cases_path)

    # Read alias map
    alias_map = _load_aliases(aliases_path)

    # Flatten everything
    terms: Set[str] = set()

    # Add concepts and cases directly
    for term in concept_list + case_list:
        terms.add(term)

    # Add aliases
    for canonical, alias_set in alias_map.items():
        terms.add(canonical)
        for a in alias_set:
            terms.add(a)

    # Normalize
    return {_norm(t) for t in terms if t}


# Cache at module import
_ALL_TERMS: Set[str] = _load_gazetteer()


# -----------------------------------------------------------------------------
# GAZETTEER MATCHING: exact OR fuzzy
# -----------------------------------------------------------------------------
def _match_terms(query: str) -> int:
    """
    Returns the number of gazetteer terms that match the query,
    either exactly or by fuzzy ratio >= 0.75.
    """
    q = _norm(query)
    if not q:
        return 0

    # Tokenize query words for fuzzy match:
    q_words = q.split()

    hits = 0

    for term in _ALL_TERMS:
        # Exact substring match (cheap)
        if term in q:
            hits += 1
            continue

        # Fuzzy: compare each query word to the term
        # Only useful for short term names, not multi-word
        for w in q_words:
            # Heuristic: ignore extremely short words (<=2 chars)
            if len(w) <= 2:
                continue

            ratio = difflib.SequenceMatcher(None, w, term).ratio()
            if ratio >= 0.75:
                hits += 1
                break  # count each term at most once

    return hits


# -----------------------------------------------------------------------------
# PUBLIC API: route()
# -----------------------------------------------------------------------------
def route(user_query: str, *, threshold: int = 2) -> Dict[str, int]:
    """
    Heuristic routing:
      - Count how many gazetteer terms match the query (exact OR fuzzy)
      - If >= threshold → RAG
      - Else           → Chat
    """
    hits = _match_terms(user_query)
    mode = "rag" if hits >= threshold else "chat"

    return {"mode": mode, "count": hits}
