# mentor/rag/supporting_sources_selector.py
from __future__ import annotations
from typing import List, Dict
import math
import string
import numpy as np

# --- local tokenizer: keeps acronyms like MAR / WpHG / ESMA / MiCA ---
_PUNCT = str.maketrans("", "", string.punctuation)

def _tok_keep_acronyms(text: str) -> set[str]:
    if not text:
        return set()
    out: set[str] = set()
    for tok in text.translate(_PUNCT).split():
        upper_count = sum(1 for ch in tok if ch.isupper())
        # e.g., MAR, WpHG, ESMA, MiCA
        if 2 <= len(tok) <= 8 and upper_count >= 2:
            out.add(tok.lower())
            continue
        tl = tok.lower()
        if len(tl) > 3:
            out.add(tl)
    return out

def _score_hits_against_answer(answer_text: str,
                               hits: List[Dict],
                               *,
                               booklet_retriever) -> List[tuple[float, Dict]]:
    """
    Returns list of (score, hit_dict), sorted desc.
    Prefers embeddings if available; otherwise acronym-aware lexical cosine.
    """
    if not answer_text or not hits:
        return []

    # Try embeddings first (portable: no special kwargs)
    embedder = getattr(booklet_retriever, "embedder", None)
    if embedder is not None:
        try:
            p_texts = [h.get("text", "") for h in hits]
            P = embedder.encode(p_texts)
            P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
            a = embedder.encode([answer_text])[0]
            a = a / (np.linalg.norm(a) + 1e-12)
            sims = P @ a
            scored = [(float(sims[i]), {**hits[i], "_sim_mode": "embed"}) for i in range(len(hits))]
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored
        except Exception:
            # Fall back to lexical
            pass

    # Lexical fallback: binary-cosine on acronym-aware tokens
    A = _tok_keep_acronyms(answer_text)
    scored: List[tuple[float, Dict]] = []
    for h in hits:
        H = _tok_keep_acronyms(h.get("text", ""))
        overlap = len(A & H)
        denom = math.sqrt(max(len(A), 1) * max(len(H), 1))
        score = overlap / denom if denom else 0.0
        scored.append((score, {**h, "_sim_mode": "lex"}))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored

def select_supporting_paragraphs(answer_text: str,
                                 hits: List[Dict],
                                 *,
                                 booklet_retriever,
                                 max_n: int = 5) -> List[str]:
    """
    Return 0..5 paragraph numbers that are meaningfully related to the *answer*.
    Uses mode-specific minimum similarity thresholds and a relative gap test.

    Parameters
    ----------
    answer_text : str
        The model's answer text (we select paragraphs that support this).
    hits : list[dict]
        The 15 candidate paragraph dicts you already retrieved.
    booklet_retriever : ParagraphRetriever
        The same retriever you use elsewhere (may or may not have an embedder).
    max_n : int
        Cap on number of paragraph numbers to return.

    Returns
    -------
    list[str]
        Paragraph numbers as strings, or [] if nothing clears the gates.
    """
    ranked = _score_hits_against_answer(answer_text, hits, booklet_retriever=booklet_retriever)
    if not ranked:
        return []

    # Determine mode from the top result (embedding vs lexical)
    top_score, top_hit = ranked[0]
    mode = top_hit.get("_sim_mode", "lex")

    # Absolute + relative gates (same as before; tune if needed)
    # - Embedding mode: top >= 0.28 and (top - median) >= 0.08
    # - Lexical mode:   top >= 0.14 and (top - median) >= 0.05
    scores = [s for s, _ in ranked]
    med = float(np.median(scores))
    min_abs = 0.28 if mode == "embed" else 0.14
    min_gap = 0.08 if mode == "embed" else 0.05
    if top_score < min_abs or (top_score - med) < min_gap:
        return []

    # Per-item collection floor to avoid tail picks
    floor = 0.20 if mode == "embed" else 0.10
    out: List[str] = []
    seen = set()
    for s, h in ranked:
        if s < floor:
            continue
        pnum = h.get("para_num")
        if pnum is None or pnum in seen:
            continue
        seen.add(pnum)
        out.append(str(pnum))
        if len(out) == max_n:
            break
    return out
