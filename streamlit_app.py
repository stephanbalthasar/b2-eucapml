# streamlit_app.py
# Minimal UI to exercise both engines using the booklet index from the private repo.

import streamlit as st

# --- Load booklet index (server-side; users never see this file) ---
from app.bootstrap_booklet import load_booklet_index
INDEX = load_booklet_index()  # {"paragraphs": [...], "chapters": [...]}

from mentor.booklet.retriever import ParagraphRetriever, ChapterRetriever
from mentor.engines.chat_engine import ChatEngine
from mentor.engines.feedback_engine import FeedbackEngine
from mentor.llm.groq import GroqClient

st.set_page_config(page_title="EUCapML Mentor", page_icon="⚖️", layout="wide")

# Optional tiny banner (remove once you’re confident)
st.caption(f"📖 Booklet loaded — {len(INDEX['chapters'])} chapters, {len(INDEX['paragraphs'])} numbered paragraphs.")

# --- Build retrievers once ---
para_retriever = ParagraphRetriever(INDEX["paragraphs"])
chap_retriever = ChapterRetriever(INDEX["chapters"])

# --- LLM client ---
llm_api_key = st.secrets.get("GROQ_API_KEY")
if not llm_api_key:
    st.error("Missing GROQ_API_KEY in secrets.")
    st.stop()
llm = GroqClient(api_key=llm_api_key)

# --- Engines ---
chat_engine = ChatEngine(
    llm=llm,
    booklet_index=INDEX,
    booklet_retriever=chap_retriever,  # ChatEngine uses chapter-level grounding
    web_retriever=None                 # add later if you want web RAG
)
feedback_engine = FeedbackEngine(llm=llm)

# --- Sidebar controls ---
with st.sidebar:
    model = st.selectbox("Model", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"], index=0)
    temp  = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    if st.button("Reload booklet index (server cache)"):
        st.cache_data.clear()
        st.success("Re-loaded. Re-run the action to use the latest JSON.")

st.title("EUCapML Mentor (Minimal UI)")

mode = st.radio(
    "Pick a mode",
    ["Tutor chat (chapter‑grounded)", "Plan answer", "Submit for feedback", "Follow‑up"],
    horizontal=True
)

if mode == "Tutor chat (chapter‑grounded)":
    q = st.text_area("Your question", height=140, placeholder="e.g., What is 'inside information' under MAR?")
    if st.button("Ask"):
        if not q.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                reply = chat_engine.answer(q, model=model, temperature=temp, max_tokens=800)
            st.markdown(reply)

elif mode == "Plan answer":
    case_text = st.text_area("Case description", height=160)
    question  = st.text_input("Question")
    if st.button("Draft plan"):
        if not case_text.strip() or not question.strip():
            st.warning("Please provide case + question.")
        else:
            with st.spinner("Planning..."):
                plan = feedback_engine.plan_answer(case_text=case_text, question=question, model=model, temperature=temp)
            st.markdown(plan)

elif mode == "Submit for feedback":
    model_answer = st.text_area("Model answer (authoritative slice)", height=140)
    student_ans  = st.text_area("Student answer", height=220)
    if st.button("Evaluate"):
        if not model_answer.strip() or not student_ans.strip():
            st.warning("Please paste both model answer and student answer.")
        else:
            with st.spinner("Evaluating..."):
                fb = feedback_engine.evaluate_answer(student_answer=student_ans, model_answer=model_answer, model=model, temperature=temp)
            st.markdown(fb)

elif mode == "Follow‑up":
    prev = st.text_area("Paste the previous feedback", height=200)
    uq   = st.text_area("Your follow‑up question", height=120)
    if st.button("Ask follow‑up"):
        if not prev.strip() or not uq.strip():
            st.warning("Please paste the feedback and enter a question.")
        else:
            with st.spinner("Answering..."):
                ans = feedback_engine.follow_up(question=uq, previous_feedback=prev, model=model, temperature=temp)
            st.markdown(ans)
