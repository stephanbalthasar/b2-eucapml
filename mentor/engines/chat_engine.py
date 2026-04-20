# mentor/engines/chat_engine.py

from typing import List, Dict, Optional

from mentor.prompts import build_conversational_tutor_messages


class ChatEngine:
    """
    Conversational tutoring engine for EU/German capital markets law.

    Design contract:
    - The FULL conversation transcript is the unit of meaning.
    - Assistant turns are first-class context.
    - Routing only controls whether retrieval is used.
    - Prompt structure NEVER changes based on routing.
    """

    def __init__(
        self,
        *,
        llm,
        booklet_retriever=None,
        web_retriever=None,
    ):
        """
        Parameters:
        - llm: LLM client exposing a .chat(messages, ...) method
        - booklet_retriever / web_retriever:
          kept for compatibility; retrieval is triggered externally
        """
        self.llm = llm
        self.booklet_retriever = booklet_retriever
        self.web_retriever = web_retriever

    # --------------------------------------------------
    # Unified conversational entry point (chat + RAG)
    # --------------------------------------------------
    def answer(
        self,
        *,
        conversation: List[Dict[str, str]],
        retrieved_booklet_chunks: Optional[List[str]] = None,
        retrieved_web_snippets: Optional[List[str]] = None,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> str:
        """
        Produce the next assistant message.

        Required:
        - conversation: full transcript, ordered, with roles

        Optional:
        - retrieved_booklet_chunks
        - retrieved_web_snippets
        """

        if not conversation:
            raise ValueError(
                "ChatEngine.answer() requires a non-empty conversation transcript"
            )

        messages = build_conversational_tutor_messages(
            conversation=conversation,
            retrieved_booklet_chunks=retrieved_booklet_chunks,
            retrieved_web_snippets=retrieved_web_snippets,
        )

        return self.llm.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # --------------------------------------------------
    # Compatibility alias (non-RAG conversational assist)
    # --------------------------------------------------
    def assist(
        self,
        *,
        conversation: List[Dict[str, str]],
        model: str,
        temperature: float = 0.6,
        max_tokens: int = 350,
    ) -> str:
        """
        Convenience wrapper for conversational replies without retrieval.
        Semantically identical to answer().
        """

        return self.answer(
            conversation=conversation,
            retrieved_booklet_chunks=None,
            retrieved_web_snippets=None,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
