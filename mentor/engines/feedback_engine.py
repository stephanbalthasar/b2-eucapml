# mentor/engines/feedback_engine.py
from mentor.prompts import (
    build_evaluate_messages,
    build_consistency_rewrite_messages,
    build_plan_messages,
    build_followup_messages,
)

class FeedbackEngine:
    def __init__(self, llm):
        self.llm = llm

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

    # ---------------------------------------------------------------------
    # ✅ FIXED: Exam follow-up WITH OPTIONAL GROUNDING
    # ---------------------------------------------------------------------
    def follow_up_with_history(
        self,
        question: str,
        context: Dict[str, Any],
        booklet_chunks: Optional[List[str]] = None,
        model: str = None,
        temperature: float = 0.2,
    ) -> str:
        """
        Handle follow-up questions after exam feedback.

        IMPORTANT DESIGN DECISION:
        - This method does NOT decide whether retrieval is necessary.
        - If booklet_chunks are provided, the prompt will enforce strict grounding.
        - If no booklet_chunks are provided, the model may only reason about
          the evaluation and feedback itself.
        """

        messages = build_followup_messages(
            previous_feedback=context.get("feedback", ""),
            followup_question=question,
            booklet_chunks=booklet_chunks,
        )

        return self.llm.chat(
            messages=messages,
            model=model,
            temperature=temperature,
        )
