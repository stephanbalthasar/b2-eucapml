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
        # 1) (kept) simple keyword extraction – not critical for paragraph retrieval
        keywords = self._extract_keywords(user_query)

        # 2) Retrieve booklet context — prefer ParagraphRetriever (retrieve top 15 paragraphs)
        hits = []
        try:
            if hasattr(self.booklet_retriever, "retrieve"):
                # ParagraphRetriever path (recommended)
                hits = self.booklet_retriever.retrieve(user_query, top_k=15) or []
            elif hasattr(self.booklet_retriever, "retrieve_best"):
                # Backward-compat: ChapterRetriever was passed; re-rank paragraphs within that chapter
                chapter = self.booklet_retriever.retrieve_best(user_query)
                if chapter and isinstance(chapter, dict):
                    # Build a temporary ParagraphRetriever over the chosen chapter's paragraphs
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
                        # Fallback: split chapter text into crude paragraphs if needed
                        text = chapter.get("text", "")
                        paras = [t.strip() for t in text.split("\n\n") if t.strip()]
                        hits = [{"text": t} for t in paras[:15]]
            else:
                # Last-resort generic retrieve() if present
                hits = self.booklet_retriever.retrieve(
                    query=user_query,
                    keywords=keywords,
                    top_k=15
                ) or []
        except Exception:
            # Defensive fallback: no hard failure on retrieval issues
            hits = []

        # Normalize to list[str] for prompt
        booklet_chunks = [
            (h.get("text") if isinstance(h, dict) else str(h))
            for h in hits if h
        ]

        # 3) (optional) Retrieve web snippets – unchanged
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
        return self.llm.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )

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
        # Use up to 15 paragraph snippets (as requested)
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
