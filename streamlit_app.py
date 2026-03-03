import streamlit as st

from app.bootstrap_booklet import load_booklet_index
from mentor.booklet.retriever import ParagraphRetriever, ChapterRetriever
from mentor.engines.chat_engine import ChatEngine
from mentor.engines.feedback_engine import FeedbackEngine
from mentor.llm.groq import GroqClient

# 1) Load the shared booklet index (server-side; users never see the file)
INDEX = load_booklet_index()
PARAS = INDEX["paragraphs"]
CHAPS = INDEX["chapters"]

# Optional quick sanity banner (remove later once stable)
st.caption(f"📖 Booklet loaded — {len(CHAPS)} chapters, {len(PARAS)} numbered paragraphs.")

# 2) Build retrievers
para_retriever = ParagraphRetriever(PARAS)   # keyword fallback; pass embedder later if you like
chap_retriever = ChapterRetriever(CHAPS)

# 3) Wire engines
llm_api_key = st.secrets.get("GROQ_API_KEY")  # or from env
llm = GroqClient(api_key=llm_api_key)

chat_engine = ChatEngine(
    llm=llm,
    booklet_index=INDEX,
    booklet_retriever=chap_retriever,   # chat: coarse chapter grounding
    web_retriever=None                  # add later
)

feedback_engine = FeedbackEngine(llm=llm)     # feedback: will call para_retriever as needed later


st.set_page_config(page_title="EUCapML Mentor", page_icon="⚖️", layout="wide")
st.title("EUCapML Mentor")
st.caption("Two modes: Feedback (plan/evaluate/explain) and Chat (booklet‑grounded Q&A).")
st.info("UI wiring to engines will be added next. This is a placeholder app.")
