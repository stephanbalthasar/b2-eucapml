# -*- coding: utf-8 -*-
"""
Local-gazetteer + remote-booklet retriever.

Loads:
- gazetteer_concepts.txt      (local, same folder as this file)
- gazetteer_cases.txt         (local)
- gazetteer_aliases.txt       (local)
- booklet_index.jsonl         (REMOTE PATH EXACTLY LIKE BEFORE)

All semantic logic identical to the previous version.
"""

from __future__ import annotations
import json
import os
import re
import difflib
from typing import Dict, List, Optional, Tuple, Set, Iterable


# ============================================================================
# PATHS
# ============================================================================

HERE = os.path.dirname(os.path.abspath(__file__))

# Local gazetteers (same folder)
LOCAL_CONCEPTS = os.path.join(HERE, "gazetteer_concepts.txt")
LOCAL_CASES = os.path.join(HERE, "gazetteer_cases.txt")
LOCAL_ALIASES = os.path.join(HERE, "gazetteer_aliases.txt")

# Booklet path stays AS IS (same env fallback as before)
BOOKLET_PATH = os.getenv("BOOKLET_PATH", "artifacts/booklet_index.jsonl")


# ============================================================================
# SIMPLE LOCAL FILE READ
# ============================================================================

def _read_local(path: str) -> str:
    if not os.path.exists(path):
        raise RuntimeError(f"Local file not found: {path}")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _read_booklet(path: str) -> str:
    if not os.path.exists(path):
        raise RuntimeError(f"Booklet file not found at: {path}")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


# ============================================================================
# PARSING HELPERS
# ============================================================================

_HYPHEN_MAP = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"), ord("-"))

def _norm_ws_hyphen(s: str) -> str:
    s = (s or "").translate(_HYPHEN_MAP)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[;,\s]*$", "", s)
    return s

def _parse_list(txt: str) -> List[str]:
    out = []
    for line in txt.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(_norm_ws_hyphen(s))
    return out

def _parse_aliases(txt: str) -> Dict[str, Set[str]]:
    mapping = {}
    for raw in txt.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [_norm_ws_hyphen(p) for p in raw.split("\n") if p.strip()]
        if not parts:
            continue
        canon, aliases = parts[0], parts[1:]
        s = mapping.setdefault(canon, set())
        for a in aliases:
            if a and a != canon:
                s.add(a)
    return mapping

def _dedup_preserve(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        xl = x.lower()
        if xl not in seen:
            seen.add(xl)
            out.append(x)
    return out


# ============================================================================
# GAZETTEERS
# ============================================================================

class Gazetteers:
    def __init__(self, concepts: List[str], cases: List[str], alias_map: Dict[str, Set[str]]):
        self.concepts = concepts
        self.cases = cases
        self.alias_map = alias_map

        bi = {}
        for canon, alset in alias_map.items():
            cset = bi.setdefault(canon, set())
            for a in alset:
                cset.add(a)
                bi.setdefault(a, set()).add(canon)
        self.alias_bi = bi


def _load_gazetteers_local() -> Gazetteers:
    txt_concepts = _read_local(LOCAL_CONCEPTS)
    txt_cases = _read_local(LOCAL_CASES)
    txt_aliases = _read_local(LOCAL_ALIASES)

    concepts = _dedup_preserve(_parse_list(txt_concepts))
    cases     = _dedup_preserve(_parse_list(txt_cases))
    aliases   = _parse_aliases(txt_aliases)

    return Gazetteers(concepts, cases, aliases)


# ============================================================================
# CORPUS (BOOKLET) LOADER
# ============================================================================

def _load_corpus(path: str) -> List[Dict]:
    txt = _read_booklet(path)
    nodes = []
    for line in txt.splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        if not (obj.get("text") or "").strip():
            continue
        nodes.append(obj)

    if not nodes:
        raise RuntimeError("Booklet JSONL loaded but contained no usable nodes.")
    return nodes


# ============================================================================
# SIGNAL EXTRACTION (unchanged)
# ============================================================================

RE_SECTION = re.compile(r"§\s*\d+[a-z]?(?:\s*(?:Abs\.?|Satz)\s*\d+)*", re.IGNORECASE)
RE_ARTICLE = re.compile(r"(?:Art\.?|Artikel)\s*\d+[a-z]?(?:\(\d+\))*", re.IGNORECASE)
RE_DOCKET = re.compile(r"\b(?:[CE]-\d+/\d{2}|[IVX]+\s+Z[RB]\s+\d+/\d{2})\b", re.IGNORECASE)

def _strip_nonword(s: str) -> str:
    return re.sub(r"\W+", "", s or "")

def _difflib_best(token: str, candidates: List[str]):
    if not token or not candidates:
        return None, 0.0, 0.0
    scored = [(c, difflib.SequenceMatcher(None, token.lower(), c.lower()).ratio()) for c in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    best, s1 = scored[0]
    s2 = scored[1][1] if len(scored) > 1 else 0.0
    return best, s1, (s1 - s2)

def _should_snap(token: str, score: float, margin: float) -> bool:
    cutoff = 0.92 if len(_strip_nonword(token)) <= 6 else 0.85
    return (score >= cutoff) and (margin >= 0.05)

def _wordish_tokens(q: str) -> List[str]:
    q = _norm_ws_hyphen(q)
    return re.findall(r"[A-Za-zÄÖÜäöüß0-9\-/]+", q)

def _expand_aliases(seed: Set[str], alias_bi: Dict[str, Set[str]]) -> Set[str]:
    out = set(seed)
    for s in list(seed):
        for a in alias_bi.get(s, ()):
            out.add(a)
    return out


# ============================================================================
# CORPUS AUTO-ALIASING
# ============================================================================

def _find_case_numbers(t: str) -> Set[str]:
    return set(RE_DOCKET.findall(t))

def _find_case_names(t: str, names: List[str]) -> Set[str]:
    lt = t.lower()
    res = set()
    for n in names:
        if len(_strip_nonword(n)) >= 4 and n.lower() in lt:
            res.add(n)
    return res

def build_corpus_auto_alias(nodes: List[Dict], gaz: Gazetteers) -> Dict[str, Set[str]]:
    auto = {}
    for n in nodes:
        t = _norm_ws_hyphen(n.get("text", "") or "")
        names = _find_case_names(t, gaz.cases)
        dockets = _find_case_numbers(t)
        for name in names:
            for num in dockets:
                auto.setdefault(name, set()).add(num)
                auto.setdefault(num, set()).add(name)
    return auto


# ============================================================================
# SCORING
# ============================================================================

_WORD_REGEX = re.compile(r"\w+")

def _words(txt: str) -> List[str]:
    return _WORD_REGEX.findall(txt.lower())

def _has_exact(text_lc: str, needle: str) -> bool:
    n = needle.lower()
    if re.search(r"\W", n):
        return n in text_lc
    return re.search(rf"\b{re.escape(n)}\b", text_lc) is not None

def _best_fuzzy_against_words(needle: str, words: List[str]) -> float:
    m = len(_strip_nonword(needle))
    if m == 0:
        return 0.0
    n = needle.lower()
    best = 0.0
    for w in words:
        if abs(len(w) - m) > max(2, m // 2):
            continue
        r = difflib.SequenceMatcher(None, n, w).ratio()
        if r > best:
            best = r
    return best

def score_node(text: str, signals: List[Dict]) -> float:
    t_norm = _norm_ws_hyphen(text)
    t_lc = t_norm.lower()
    tokens = _words(t_norm)
    score = 0.0
    saw_name = False
    saw_no = False

    for s in signals:
        if s["type"] in {"section", "article", "case_no"}:
            if _has_exact(t_lc, s["canonical"]):
                score += 3.0
                if s["type"] == "case_no":
                    saw_no = True
            continue

        hit_exact = False
        for cand in s["expanded"]:
            if _has_exact(t_lc, cand):
                score += 2.5
                hit_exact = True
                if s["type"] == "case_name":
                    saw_name = True
                if RE_DOCKET.fullmatch(cand):
                    saw_no = True
        if hit_exact:
            continue

        if s.get("fuzzy_eligible"):
            sim = _best_fuzzy_against_words(s["canonical"], tokens)
            if sim >= 0.82:
                score += (0.6 * sim)

    if saw_name and saw_no:
        score += 0.4

    return score


# ============================================================================
# SIGNAL EXTRACTION
# ============================================================================

def extract_signals(query: str, gaz: Gazetteers, alias_map: Dict[str, Set[str]]) -> List[Dict]:
    q = _norm_ws_hyphen(query or "")
    if not q:
        return []

    signals = []

    for m in RE_SECTION.finditer(q):
        s = m.group(0)
        signals.append(dict(type="section", surface=s, canonical=s,
                            confidence=1.0, expanded={s}, fuzzy_eligible=False))

    for m in RE_ARTICLE.finditer(q):
        s = m.group(0)
        signals.append(dict(type="article", surface=s, canonical=s,
                            confidence=1.0, expanded={s}, fuzzy_eligible=False))

    for m in RE_DOCKET.finditer(q):
        s = m.group(0)
        signals.append(dict(type="case_no", surface=s, canonical=s,
                            confidence=1.0, expanded={s}, fuzzy_eligible=False))

    for tok in sorted(_wordish_tokens(q), key=lambda x: (-len(_strip_nonword(x)), x.lower())):
        if RE_SECTION.fullmatch(tok) or RE_ARTICLE.fullmatch(tok) or RE_DOCKET.fullmatch(tok):
            continue

        canonical = None
        snapped_type = None
        confidence = 0.0

        best, score, margin = _difflib_best(tok, gaz.concepts)
        if best and _should_snap(tok, score, margin):
            canonical, snapped_type, confidence = best, "concept", score
        else:
            best, score, margin = _difflib_best(tok, gaz.cases)
            if best and _should_snap(tok, score, margin):
                canonical, snapped_type, confidence = best, "case_name", score

        if canonical:
            expanded = _expand_aliases({canonical}, gaz.alias_bi)
            expanded = _expand_aliases(expanded, alias_map)
            signals.append(dict(type=snapped_type, surface=tok, canonical=canonical,
                                confidence=confidence, expanded=expanded,
                                fuzzy_eligible=False))
        else:
            if _strip_nonword(tok):
                signals.append(dict(type="other", surface=tok, canonical=tok,
                                    confidence=0.0, expanded={tok},
                                    fuzzy_eligible=True))

    # dedup
    out = []
    seen = set()
    for s in signals:
        key = (s["type"], s["canonical"].lower(),
               tuple(sorted(x.lower() for x in s["expanded"])))
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


# ============================================================================
# MAIN RETRIEVER
# ============================================================================

class ParagraphRetriever:
    def __init__(self, _ignored=None):
        self.gaz = _load_gazetteers_local()
        self.nodes = _load_corpus(BOOKLET_PATH)

        auto_alias = build_corpus_auto_alias(self.nodes, self.gaz)

        self.alias_bi = dict(self.gaz.alias_bi)
        for k, v in auto_alias.items():
            self.alias_bi.setdefault(k, set()).update(v)

        self._texts_lc = [_norm_ws_hyphen(n.get("text", "") or "").lower()
                          for n in self.nodes]
        self._tokens_list = [_words(_norm_ws_hyphen(n.get("text", "") or ""))
                             for n in self.nodes]

    def search(self, query: str, top_k: int = 6, **_k) -> List[Dict]:
        if not (query or "").strip():
            return []
        signals = extract_signals(query, self.gaz, self.alias_bi)
        if not signals:
            return []

        scored = []
        for i, n in enumerate(self.nodes):
            s = score_node(n.get("text", "") or "", signals)
            if s >= 1.0:
                scored.append((i, s))
        if not scored:
            return []

        scored.sort(key=lambda x: (-x[1],
                                   len(self.nodes[x[0]].get("text", "") or ""),
                                   x[0]))

        idx = [i for (i, _) in scored[:max(1, top_k)]]
        sc  = [float(s) for (_, s) in scored[:max(1, top_k)]]

        return self._package(idx, sc)

    def _package(self, indices: List[int], scores: List[float]) -> List[Dict]:
        out = []
        for rank, (i, s) in enumerate(zip(indices, scores), start=1):
            n = self.nodes[i]
            breadcrumb = n.get("breadcrumb") or n.get("breadcrumbs")
            out.append({
                "text": n.get("text", ""),
                "score": s,
                "rank": rank,
                "node_id": n.get("node_id"),
                "doc_id": n.get("doc_id"),
                "type": n.get("type"),
                "anchor": n.get("anchor"),
                "breadcrumb": breadcrumb,
                "lang": n.get("lang"),
                "links": n.get("links", {}),
            })
        return out


if __name__ == "__main__":
    q = "Lafonat § 33 WpHG Ad-hoc-Publizität"
    r = ParagraphRetriever()
    for h in r.search(q):
        print(f"{h['rank']} | {h['score']:.2f} | {h['text'][:120]}")
