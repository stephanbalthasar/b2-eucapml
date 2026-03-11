# mentor/rag/booklet_references_selector.py
from __future__ import annotations
from typing import List, Dict, Tuple
import math, string, numpy as np

# --- tokenizer: keeps MAR/WpHG/ESMA/MiCA, drops short fillers ---
_PUNCT = str.maketrans("", "", string.punctuation)

def tok_keep_acronyms(text: str) -> set[str]:
    if not text:
        return set()
    out: set[str] = set()
    for tok in text.translate(_PUNCT).split():
        upp = sum(1 for ch in tok if ch.isupper())
        if 2 <= len(tok) <= 8 and upp >= 2:
            out.add(tok.lower()); continue
        tl = tok.lower()
        if len(tl) > 3:
            out.add(tl)
    return out

# --- compact & sanitize chunks for prompts (optional) ---
def compact_chunks(hits: List[Dict], *, max_chars: int = 700, max_k: int = 15) -> List[str]:
    chunks: List[str] = []
    seen = set()
    for h in hits[:max_k]:
        t = (h.get("text") or "").strip()
        if not t: 
            continue
        t = t[:max_chars] + ("…" if len(t) > max_chars else "")
        if t in seen: 
            continue
        seen.add(t)
        chunks.append(t)
    return chunks

# --- score 15 candidates against the answer text (embeddings → lexical fallback) ---
def rank_paragraphs_by_text(answer_text: str,
                            hits: List[Dict],
                            *,
                            booklet_retriever,
                            ) -> List[Tuple[float, Dict]]:
    if not answer_text or not hits:
        return []
    embedder = getattr(booklet_retriever, "embedder", None)
    if embedder is not None:
        try:
            P = embedder.encode([h.get("text","") for h in hits])
            P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
            a = embedder.encode([answer_text])[0]
            a = a / (np.linalg.norm(a) + 1e-12)
            sims = P @ a
            scored = [(float(sims[i]), {**hits[i], "_sim_mode": "embed"}) for i in range(len(hits))]
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored
        except Exception:
            pass  # lexical fallback

    # lexical cosine on acronym-aware tokens
    A = tok_keep_acronyms(answer_text)
    scored: List[Tuple[float, Dict]] = []
    for h in hits:
        H = tok_keep_acronyms(h.get("text",""))
        overlap = len(A & H)
        denom = math.sqrt(max(len(A),1)*max(len(H),1))
        score = overlap/denom if denom else 0.0
        scored.append((score, {**h, "_sim_mode":"lex"}))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored

# --- conservative gating: return 0..5 para numbers ---
def pick_para_nums(ranked: List[Tuple[float, Dict]], *, max_n: int = 5) -> List[str]:
    if not ranked:
        return []
    scores = [s for s,_ in ranked]
    top = scores[0]
    mode = ranked[0][1].get("_sim_mode","lex")
    import numpy as _np
    med = float(_np.median(scores))
    min_abs = 0.28 if mode == "embed" else 0.14
    min_gap = 0.08 if mode == "embed" else 0.05
    if top < min_abs or (top - med) < min_gap:
        return []
    floor = 0.20 if mode == "embed" else 0.10
    out, seen = [], set()
    for s,h in ranked:
        if s < floor: 
            continue
        p = h.get("para_num")
        if p is None or p in seen:
            continue
        seen.add(p); out.append(str(p))
        if len(out) == max_n: 
            break
    return out
