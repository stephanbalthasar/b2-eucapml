# mentor/engines/chat_engine.py

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

    def answer(self, user_query, *, model, temperature, max_tokens=800):
        # 1) Simple keyword extraction (kept)
        keywords = self._extract_keywords(user_query)

        # 2) Retrieve booklet context — prefer ParagraphRetriever (top 15)
        hits = []
        try:
            if hasattr(self.booklet_retriever, "retrieve"):
                hits = self.booklet_retriever.retrieve(user_query, top_k=15) or []
            elif hasattr(self.booklet_retriever, "retrieve_best"):
                # Back-compat: ChapterRetriever branch
                chapter = self.booklet_retriever.retrieve_best(user_query)
                if chapter and isinstance(chapter, dict):
                    from mentor.booklet.retriever import ParagraphRetriever
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

        # --- NEW: collect paragraph numbers used -------------------------
        para_nums = []
        for h in hits:
            if isinstance(h, dict) and "para_num" in h and h["para_num"] is not None:
                para_nums.append(str(h["para_num"]))
        # keep order, remove duplicates
        seen = set()
        para_nums = [p for p in para_nums if not (p in seen or seen.add(p))]
        # ------------------------------------------------------------------

        # 3) Optional web retrieval (kept)
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
            web_snippets=web_snippets
        )

        # 5) Ask LLM
        result = self.llm.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # --- NEW: append a deterministic footer with paragraph numbers ----
        # Ensure we have a string to display in Streamlit
        reply_text = result if isinstance(result, str) else str(result)
        if para_nums:
            # Limit to the 15 used; adjust formatting as you prefer
            footer = "\n\n---\n" \
                     "_Also see paragraphs " + ", ".join(para_nums[:15]) + " in the course booklet._"
            reply_text += footer

        return reply_text
        # ------------------------------------------------------------------

    # -------- helpers (kept) --------------------
    def _extract_keywords(self, text):
        # TODO: later include legal keyword extraction
        # For now: split by spaces and take simple tokens
        return [w.strip() for w in text.split() if len(w) > 3]

    def _build_prompt(self, user_query, booklet_chunks, web_snippets):
        system = (
            "You are a helpful EU/German capital markets law tutor. "
            "Use the provided booklet excerpts and optional web snippets. "
            "If unsure, say what is known, and avoid fabricating structural references."
        )
        booklet_block = "\n\n".join(f"- {c}" for c in booklet_chunks[:15]) or "None"
        web_block = "\n\n".join(f"- {s}" for s in web_snippets[:4]) or "None"

        user_content = (
            f"USER QUERY:\n{user_query}\n\n"
            f"RELEVANT BOOKLET EXCERPTS:\n{booklet_block}\n\n"
            f"RELEVANT WEB SNIPPETS:\n{web_block}\n\n"
            "Please answer clearly and concisely."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content}
        ]
