# mentor/engines/chat_engine.py
from mentor.prompts import build_tutor_messages
from mentor.rag.supporting_sources_selector import select_supporting_paragraphs


class ChatEngine:
    """
    One-purpose engine:
    - Take user query
    - Find relevant booklet parts
    - Optionally find relevant web snippets
    - Ask LLM to answer grounded in those
    """

    def __init__(self, llm, booklet_index, booklet_retriever, web_retriever=None):
        self.llm = llm
        self.booklet_index = booklet_index
        self.booklet_retriever = booklet_retriever  # Prefer ParagraphRetriever
        self.web_retriever = web_retriever  # may be None for now

    def answer(self, user_query, *, model, temperature, max_tokens=800, conversation_preamble=None):
        """
        Generates an answer and then selects 0..5 booklet paragraphs that are
        meaningfully related to the *answer* (not the question).
        Only appends a footer if at least one paragraph clears a minimum similarity.
        """
        import string, math
        import numpy as np
    
        # ---------- local helpers (self-contained) ----------
        def _tok_keep_acronyms(text: str) -> set[str]:
            import string
            if not text:
                return set()
            punct_table = str.maketrans("", "", string.punctuation)
            toks = text.translate(punct_table).split()
            out = set()
            for tok in toks:
                upper_count = sum(1 for ch in tok if ch.isupper())
                if 2 <= len(tok) <= 8 and upper_count >= 2:
                    out.add(tok.lower()); continue
                tl = tok.lower()
                if len(tl) > 3:
                    out.add(tl)
            return out
    
        def _is_meta_answer(text: str) -> bool:
            """Detect greetings/housekeeping answers where we should show no footer."""
            t = (text or "").lower()
            # short & meta-ish phrasing is enough to skip
            short = len(t) < 280
            meta_kw = (
                "hi", "hello", "hallo", "hey",
                "chat", "reden", "sprechen", "talk", "discuss",
                "fragen", "question", "questions",
                "booklet", "broschüre", "heft",
                "let's", "können wir", "kann ich", "kannst du", "gerne", "klar", "sure"
            )
            hits = sum(k in t for k in meta_kw)
            return short and hits >= 2
        
            # 1) Embeddings (moved to sources selector)
            # 2) Lexical fallback (binary-cosine) (moved to sources selector)
                      
        # 1) Simple keyword extraction (existing logic)
        keywords = self._extract_keywords(user_query)
    
        # 2) Retrieve booklet context (prefer ParagraphRetriever)
        hits = []
        try:
            if hasattr(self.booklet_retriever, "retrieve"):
                hits = self.booklet_retriever.retrieve(user_query, top_k=15) or []
            elif hasattr(self.booklet_retriever, "retrieve_best"):
                # Back-compat: ChapterRetriever branch
                chapter = self.booklet_retriever.retrieve_best(user_query)
                if chapter and isinstance(chapter, dict):
                    from mentor.rag.booklet_retriever import ParagraphRetriever  # your current path
                    chapter_num = chapter.get("chapter_num")
                    chapter_paras = [
                        p for p in (self.booklet_index.get("paragraphs") or [])
                        if p.get("chapter_num") == chapter_num
                    ]
                    if chapter_paras:
                        pr = ParagraphRetriever(chapter_paras)
                        hits = pr.retrieve(user_query, top_k=15) or []
                    else:
                        text = chapter.get("text", "")
                        paras = [t.strip() for t in text.split("\n\n") if t.strip()]
                        hits = [{"text": t} for t in paras[:15]]
            else:
                # Generic fallback (avoid passing unsupported kwargs)
                try:
                    hits = self.booklet_retriever.retrieve(user_query, top_k=15) or []
                except TypeError:
                    hits = self.booklet_retriever.retrieve(query=user_query, top_k=15) or []
        except Exception:
            hits = []
    
        # Normalize to list[str] for prompt
        booklet_chunks = [
            (h.get("text") if isinstance(h, dict) else str(h))
            for h in hits if h
        ]
    
        # 3) Optional web retrieval (kept off unless web_retriever is set)
        web_snippets = []
        if self.web_retriever is not None:
            try:
                web_snippets = self.web_retriever.retrieve(
                    query=user_query,
                    keywords=keywords,
                    top_k=4
                ) or []
            except Exception:
                web_snippets = []
    
        # 4) Build messages for LLM
        
        messages = self._build_prompt(
            user_query=user_query,
            booklet_chunks=booklet_chunks,
            web_snippets=web_snippets,
            conversation_preamble=conversation_preamble,
        )
    
        # 5) Ask LLM
        
        result = self.llm.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
        # Coerce to plain string for display
        reply_text = result if isinstance(result, str) else str(result)
        if _is_meta_answer(reply_text):
            return reply_text  # early exit: greetings/housekeeping -> no footer
    
        # 6) NEW: pick 0..5 supporting paragraph numbers based on the *answer*
        selected_para_nums = select_supporting_paragraphs(
            reply_text,
            hits,
            booklet_retriever=self.booklet_retriever,
            max_n=5
        )
    
        # Append footer only if we actually have meaningful support
        if selected_para_nums:
            footer = "\n\n---\n" + "_Key paragraphs: " + ", ".join(selected_para_nums) + "._"
            reply_text += footer
    
        return reply_text

    # -------- helpers (kept) --------------------
    def _extract_keywords(self, text):
        # TODO: later include legal keyword extraction
        # For now: split by spaces and take simple tokens
        return [w.strip() for w in text.split() if len(w) > 3]

    def _build_prompt(self, user_query, booklet_chunks, web_snippets, conversation_preamble=None):
        # Single source of truth for tutor prompts
        return build_tutor_messages(
            user_query=user_query,
            booklet_chunks=booklet_chunks,
            web_snippets=web_snippets,
            conversation_preamble=conversation_preamble,
        )
