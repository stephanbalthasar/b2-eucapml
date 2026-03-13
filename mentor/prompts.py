# mentor/prompts.py
from __future__ import annotations

EVAL_MAX_WORDS = 300
PLAN_MAX_WORDS = 100
FOLLOWUP_MAX_WORDS = 160

def build_evaluate_messages(student_answer: str, model_answer: str,
                            max_words: int = EVAL_MAX_WORDS) -> list[dict]:
    system = (
        "You are an examiner for an EU/German capital markets law exam.\n"
        "Use the MODEL ANSWER as the authoritative benchmark.\n"
        "Do NOT disclose the model answer text. If the student conflicts with it, state the correct "
        "conclusion succinctly and explain briefly why.\n"
        "Do NOT invent legal citations or paragraph numbers. No footnotes. No web sources.\n"
        "Do NOT assign grades, scores, marks etc., provide qualitative feedback only.\n"
        f"Write clearly, ≤ {max_words} words, using the exact five headings below."
    )
    user = (
        "MODEL ANSWER (authoritative, do NOT disclose it):\n"
        f"\"\"\"{(model_answer or '').strip()}\"\"\"\n\n"
        "STUDENT ANSWER:\n"
        f"\"\"\"{(student_answer or '').strip()}\"\"\"\n\n"
        "TASK: Compare the student's reasoning to the MODEL ANSWER and return feedback with the EXACT "
        "headings and format below. Keep it concise and exam‑focused.\n\n"
        "**Student's Core Claims:**\n"
        "• <short bullet per core claim> — [Correct | Incorrect | Unclear]\n"
        "**Mistakes:**\n"
        "• <what is wrong and why> (1–2 lines)\n"
        "**Missing Aspects:**\n"
        "• <material point absent> (why it matters)\n"
        "**Suggestions:**\n"
        "• <how to improve reasoning>\n"
        "**Conclusion:**\n"
        "<one sentence summing up adequacy>\n"
    )
    return [{"role": "system", "content": system},
            {"role": "user",   "content": user}]

def build_consistency_rewrite_messages(feedback_text: str, model_answer: str,
                                       max_words: int = EVAL_MAX_WORDS) -> list[dict]:
    system = (
        "You are an impartial checker. Ensure the FEEDBACK does not contradict the MODEL ANSWER.\n"
        f"If contradictions exist, minimally edit the FEEDBACK to align with the MODEL ANSWER while keeping the same structure and ≤ {max_words} words."
    )
    user = (
        f"MODEL ANSWER (authoritative):\n\"\"\"{(model_answer or '').strip()}\"\"\"\n\n"
        f"FEEDBACK (to check and minimally correct):\n\"\"\"{(feedback_text or '').strip()}\"\"\"\n"
        "Return only the corrected FEEDBACK text (no preface)."
    )
    return [{"role": "system", "content": system},
            {"role": "user",   "content": user}]

def build_plan_messages(case_text: str,
                        question_label: str,
                        model_answer_slice: str | None = None,
                        booklet_text: str | None = None,
                        max_words: int = PLAN_MAX_WORDS) -> list[dict]:
    system = (
        "You are a tutor helping a student plan an exam answer in EU/German capital markets law.\n"
        f"Produce a lean, issue‑first outline (6–9 bullets), ≤ {max_words} words. "
        "No citations. No paragraph numbers. No web sources.\n"
        "Do NOT propose conclusions, only list topics to cover."
        "Do NOT disclose or quote the model answer text. If the correct direction differs from the student's likely path, steer it quietly in the plan."      
    )
    blocks = [
        f"CASE DESCRIPTION:\n\"\"\"{(case_text or '').strip()}\"\"\"",
        f"QUESTION: {question_label}",
    ]
    if model_answer_slice and model_answer_slice.strip():
        blocks.append(
            "AUTHORITATIVE COMPASS (do NOT disclose to student; use only to orient the plan):\n"
            f"\"\"\"{model_answer_slice.strip()}\"\"\""
        )
    if booklet_text and booklet_text.strip():
        blocks.append(
            "RELEVANT BOOKLET CHAPTER (use concepts/terms, but no verbatim quotes):\n"
            f"\"\"\"{booklet_text.strip()}\"\"\""
        )
    task = (
        "TASK: Draft a plan the student can follow under time pressure:\n"
        "• Order issues logically (IRAC‑friendly labels: Issue → Rule/Standard → Application).\n"
        "• Only raise issues for the student to cover, but do not propose a conclusion.\n"
        "• End with a 1‑line Exam Tip."
    )
    user = "\n\n".join(blocks) + "\n\n" + task
    return [{"role": "system", "content": system},
            {"role": "user",   "content": user}]

# --- General tutor chat (grounded with safe fallback and controlled augmentation) ---
SYSTEM_TUTOR = (
    "You are a helpful EU/German capital markets law tutor.\n"
    "\n"
    "GROUNDING & FALLBACK RULES (must follow):\n"
    "• If booklet excerpts and/or web snippets are provided (GROUNDED mode), treat them as the PRIMARY evidence. Rely on them first and avoid claims that are not supported by them.\n"
    "• In GROUNDED mode you MAY add brief, non‑authoritative background to connect the dots (definitions, short context), but you MUST NOT introduce new authorities (no new cases, statutes, paragraph numbers, or pinpoint citations) that are not present in the provided context. Never contradict the excerpts/snippets; if they diverge from common background, defer to them or flag the mismatch succinctly.\n"
    "• If neither booklet excerpts nor web snippets are provided (UNGROUNDED mode), still answer with a concise general background from your own knowledge and CLEARLY prefix the first line with: 'General background (no booklet/web):'. Do NOT invent case numbers, paragraph numbers, or citations. Offer to search official sources (EUR‑Lex/CURIA/ESMA/BaFin) if the user wants grounded references.\n"
    "\n"
    "STYLE:\n"
    "• Be clear and compact by default (≈3–5 sentences or 3–5 bullets).\n"
    "• If no concrete legal question is asked, request one and stop.\n"
    "• Never fabricate authorities; never imply you saw sources you were not given.\n"
)
def build_tutor_messages(
    user_query: str,
    booklet_chunks: list[str],
    web_snippets: list[str],
    conversation_preamble: str | None = None,
    *,
    # Optional: if you later pass a retrieval score gap (e.g., top - median),
    # we will use it to decide STRONG vs WEAK. If None, we fall back to counts.
    similarity_gap_hint: float | None = None,
) -> list[dict]:
    """
    Centralized prompt for the general tutor chat.
    - conversation_preamble: compact transcript (optional)
    - booklet_chunks: up to 15 snippets (already trimmed by the engine)
    - web_snippets: up to 4 snippets
    - similarity_gap_hint: optional float (e.g., top_score - median_score from retriever)
    """

    # --- Conversation preamble (optional) ---
    convo_block = (
        f"Conversation so far (most recent last):\n{conversation_preamble}\n\n"
        if conversation_preamble else ""
    )

    # --- Grounding mode & simple strength heuristic ---
    has_booklet = bool(booklet_chunks)
    has_web = bool(web_snippets)
    grounding_mode = "GROUNDED" if (has_booklet or has_web) else "UNGROUNDED"

    # Effective counts (your builder slices to these limits anyway)
    b_count = min(len(booklet_chunks or []), 15)
    w_count = min(len(web_snippets or []), 4)

    # Tiny rule for CONTEXT STRENGTH:
    # 1) If a similarity gap hint is provided, treat >=0.08 as STRONG, else WEAK.
    # 2) Otherwise, counts-based default:
    #    STRONG if (b_count >= 8) or (w_count >= 2); else WEAK.
    if similarity_gap_hint is not None:
        context_strength = "STRONG" if float(similarity_gap_hint) >= 0.08 else "WEAK"
    else:
        context_strength = "STRONG" if (b_count >= 8 or w_count >= 2) else "WEAK"

    # --- Material blocks ---
    booklet_block = "\n\n".join(f"- {c}" for c in (booklet_chunks or [])[:15]) or "None"
    web_block = "\n\n".join(f"- {s}" for s in (web_snippets or [])[:4]) or "None"

    # --- User content with explicit mode/strength line ---
    user_content = (
        f"{convo_block}"
        f"GROUNDING MODE: {grounding_mode}    CONTEXT STRENGTH: {context_strength}\n\n"
        f"USER QUERY:\n{user_query}\n\n"
        f"RELEVANT BOOKLET EXCERPTS:\n{booklet_block}\n\n"
        f"RELEVANT WEB SNIPPETS:\n{web_block}\n\n"
        "Please answer clearly and concisely. Follow the GROUNDING & FALLBACK RULES from the system message.\n"
    )

    return [
        {"role": "system", "content": SYSTEM_TUTOR},
        {"role": "user", "content": user_content},
    ]

def build_followup_messages(previous_feedback: str,
                            followup_question: str,
                            max_words: int = FOLLOWUP_MAX_WORDS,
                            booklet_chunks: list[str] | None = None) -> list[dict]:
    system = (
        "You answer follow‑up questions about previous feedback. Be precise, "
        f"≤ {max_words} words. If something depends on facts, say what you would check. "
        "Use the provided booklet excerpts if relevant; do not fabricate citations."
    )
    booklet_block = "\n\n".join(f"- {c}" for c in (booklet_chunks or [])[:12]) or "None"
    user = (
        f"PREVIOUS FEEDBACK:\n\"\"\"{(previous_feedback or '').strip()}\"\"\"\n\n"
        f"FOLLOW-UP QUESTION:\n{(followup_question or '').strip()}\n\n"
        f"RELEVANT BOOKLET EXCERPTS:\n{booklet_block}\n\n"
        "Answer clearly. If the student asks for the model answer, politely refuse and re‑explain the principle."
    )
    return [{"role":"system","content":system},
            {"role":"user","content":user}]

# --- Gate: Should we attach booklet references for this answer? ---
def build_sources_gate_messages(user_query: str, answer_text: str) -> list[dict]:
    """
    Returns messages to classify (YES/NO) whether this answer warrants booklet references.
    The assistant must output ONLY 'YES' or 'NO'.
    """
    system = (
        "You are a recall-oriented, permissive classifier for legal answers. "
        "Default to YES unless the answer is purely greeting/meta/logistics with no legal content. "
        "If in doubt, output YES. Output ONLY 'YES' or 'NO'."
    )

    user = (
        "Show sources when ANY of the following is true:\n"
        "• the answer contains legal analysis, definitions, rules/tests, holdings, or conclusions;\n"
        "• it mentions statutes, regulations or cases (e.g., MAR, WpHG, Prospectus rules, ECJ/EuGH, ESMA, BaFin), "
        "  or markers like 'Art.', 'Article', '§', 'para', 'case';\n"
        "• it exceeds ~60 words or uses numbered/structured bullets indicating substantive guidance.\n"
        "Return NO only for greetings, scheduling/clarifications, or meta replies (e.g., 'What is your question?').\n\n"
        f"USER QUERY:\n{(user_query or '').strip()}\n\n"
        f"ASSISTANT ANSWER:\n{(answer_text or '').strip()}\n\n"
        "Answer with ONLY YES or NO."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
