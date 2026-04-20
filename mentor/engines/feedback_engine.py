# mentor/engines/feedback_engine.py

from typing import Optional, List, Dict, Any

from mentor.prompts import (
    build_evaluate_messages,
    build_consistency_rewrite_messages,
    build_plan_messages,
    build_followup_messages,
)


class FeedbackEngine:
    def __init__(self, llm):
        self.llm = llm

    # ---------------------------------------------------------------------
    # Exam evaluation (unchanged)
    # ---------------------------------------------------------------------
    def evaluate_answer(
        self,
        student_answer: str,
        model_answer: str,
        model: str,
        temperature: float = 0.2,
    ) -> str:
        messages = build_evaluate_messages(
            student_answer=student_answer,
            model_answer=model_answer,
        )
        return self.llm.chat(
            messages=messages,
            model=model,
            temperature=temperature,
        )

    # ---------------------------------------------------------------------
    # Optional consistency rewrite (unchanged)
    # ---------------------------------------------------------------------
    def rewrite_for_consistency(
        self,
        feedback_text: str,
        model_answer: str,
        model: str,
        temperature: float = 0.0,
    ) -> str:
        messages = build_consistency_rewrite_messages(
            feedback_text=feedback_text,
            model_answer=model_answer,
        )
        return self.llm.chat(
            messages=messages,
            model=model,
            temperature=temperature,
        )

    # ---------------------------------------------------------------------
    # Exam answer planning (unchanged)
    # ---------------------------------------------------------------------
    def plan_answer(
        self,
        case_text: str,
        question_label: str,
        model_answer_slice: Optional[str],
        booklet_text: Optional[str],
        model: str,
        temperature: float = 0.2,
    ) -> str:
        messages = build_plan_messages(
            case_text=case_text,
            question_label=question_label,
            model_answer_slice=model_answer_slice,
            booklet_text=booklet_text,
        )
        return self.llm.chat(
            messages=messages,
            model=model,
            temperature=temperature,
        )

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
