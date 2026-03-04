# mentor/llm/llm_provider.py
from typing import List, Dict, Optional, Tuple

# ✅ Flat layout: all these files live in mentor/llm/
from .openrouter import OpenRouterClient
from .groq import GroqClient
from .llm_registry import get_default_model, get_model_registry, ModelInfo


class LLMProvider:
    """
    Provider-agnostic façade over your concrete clients (OpenRouter, Groq, ...).

    Usage:
        llm = LLMProvider()
        default_model, registry = llm.list_models()
        text = llm.complete(messages, provider="openrouter", model="qwen/qwen-3-instruct")
    """

    def __init__(self):
        # Instantiate provider clients (they read keys from st.secrets / env)
        self._openrouter = OpenRouterClient()
        self._groq = GroqClient()
        self._default = get_default_model()  # comes from llm_registry.py

    # ----- Model registry -----
    def list_models(self) -> Tuple[ModelInfo, Dict[str, List[ModelInfo]]]:
        """Return (default_model, registry_by_provider)."""
        return self._default, get_model_registry()

    # ----- Provider plumbing -----
    def _client_for(self, provider: str):
        if provider == "openrouter":
            return self._openrouter
        if provider == "groq":
            return self._groq
        raise ValueError(f"Unknown provider: {provider}")

    def is_available(self, provider: str) -> bool:
        try:
            return bool(self._client_for(provider).is_configured)
        except Exception:
            return False

    # ----- Unified completion entry point -----
    def complete(
        self,
        messages: List[Dict[str, str]],
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1200,
        top_p: float = 0.9,
        allow_fallback: bool = True,
    ) -> str:
        """
        Try the requested provider/model. If that fails and allow_fallback=True,
        fall back to the default (Qwen on OpenRouter).
        """
        chosen_provider = provider or self._default.provider
        chosen_model = model or self._default.model_id

        # --- Primary attempt ---
        primary = self._client_for(chosen_provider)
        try:
            if not getattr(primary, "is_configured", False):
                raise RuntimeError(f"{chosen_provider} not configured.")
            return primary.complete(
                messages,
                model=chosen_model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
        except Exception as e:
            if not allow_fallback:
                raise

            # --- Fallback to default (Qwen via OpenRouter) ---
            fallback = self._client_for(self._default.provider)
            if not getattr(fallback, "is_configured", False):
                # If fallback is also unavailable, re-raise original error
                raise e
            return fallback.complete(
                messages,
                model=self._default.model_id,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
