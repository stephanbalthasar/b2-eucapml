# mentor/engines/feedback_engine.py

from mentor.prompts import (
    build_evaluate_messages,
    build_consistency_rewrite_messages,
    build_plan_messages,
    build_followup_messages,
)

from mentor.rag.booklet_retriever import fetch_booklet_chunks_for_prompt

class FeedbackEngine:
    def __init__(self, llm, booklet_retriever=None):
        self.llm = llm
        self.booklet_retriever = booklet_retriever  # ParagraphRetriever or None

    # -------------------------------------------------------
    # (i) PLAN  ---- CHANGED SIGNATURE ----
    # -------------------------------------------------------
    def plan_answer(self, *,
                    case_text: str,
                    question: str,
                    model_answer_slice: str | None,
                    booklet_text: str | None,
                    model: str,
                    temperature: float) -> str:
        messages = build_plan_messages(
            case_text=case_text,
            question_label=question,
            model_answer_slice=model_answer_slice,
            booklet_text=booklet_text
        )
        # For planning, prefer tight settings
        return self.llm.chat(messages=messages, model=model, temperature=min(temperature, 0.2), max_tokens=350)

    # -------------------------------------------------------
    # (ii) Evaluate a submitted answer  — use the prompt builder
    # -------------------------------------------------------
    def evaluate_answer(self, *, student_answer, model_answer, model, temperature, max_words=300):
        # Build the structured, five‑heading prompt with a word ceiling
        messages = build_evaluate_messages(
            student_answer=student_answer,
            model_answer=model_answer,
            max_words=max_words
        )
        # Optional one‑time fingerprint for debugging which path runs:
        # messages.insert(0, {"role": "system", "content": "[EVAL_PATH=v2]"})
        raw = self.llm.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=900  # generous but bounded; headings + content fit comfortably
        )
        return raw if isinstance(raw, str) else str(raw)

    # -------------------------------------------------------
    # (iii) Follow-up questions about the feedback
    # -------------------------------------------------------

    def follow_up_with_history(self, question, context, model, temperature):
        # 0) Base messages (kept)
        messages = []
        messages.append({"role": "system", "content": f"Student exam answer:\n{context['student_answer']}"})
        messages.append({"role": "system", "content": f"Feedback:\n{context['feedback']}"})
    
        # 1) Retrieve booklet paragraphs using BOTH the follow-up question AND the feedback text
        booklet_chunks: list[str] = []
        if getattr(self, "booklet_retriever", None) is not None:
            try:
                # A) query by the current follow-up question (user intent right now)
                _hits_q, chunks_q = fetch_booklet_chunks_for_prompt(
                    self.booklet_retriever,
                    question or "",
                    top_k=15,
                    # optional: uncomment if you want shorter chunks in the prompt
                    # truncate_chars=700,
                )
    
                # B) query by the prior feedback text (captures phrasing/terms the LLM answered with)
                _hits_fb, chunks_fb = fetch_booklet_chunks_for_prompt(
                    self.booklet_retriever,
                    (context.get("feedback") or ""),
                    top_k=15,
                    # truncate_chars=700,
                )
    
                # C) merge + deduplicate while preserving order; cap to ~12 for prompt brevity
                merged = []
                seen = set()
                for t in (chunks_q + chunks_fb):
                    if not t: 
                        continue
                    if t in seen:
                        continue
                    seen.add(t)
                    merged.append(t)
                    if len(merged) == 12:
                        break
    
                booklet_chunks = merged
    
            except Exception:
                booklet_chunks = []
    
        # 1a) Self-check (temporary debug)
        print(f"[FE] booklet_chunks: {len(booklet_chunks)}")
        if booklet_chunks:
            print("[FE] first chunk:", booklet_chunks[0][:120].replace("\n", " "))
    
        # 2) Inject excerpts as a system block (only if we have any)
        if booklet_chunks:
            block = "Relevant booklet excerpts:\n" + "\n\n".join(f"- {c}" for c in booklet_chunks)
            messages.append({"role": "system", "content": block})
    
        # 3) Prior chat turns (kept)
        for role, msg in context["history"]:
            messages.append({
                "role": "user" if role == "student" else "assistant",
                "content": msg
            })
    
        # 4) Current question (kept)
        messages.append({"role": "user", "content": question})
    
        # 5) LLM call (kept)
        return self.llm.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=800,
        )
