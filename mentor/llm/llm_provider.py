# mentor/llm/llm_provider.py
from typing import List, Dict, Optional, Tuple
import os

# ✅ Flat layout: all these files live in mentor/llm/
from .openrouter import OpenRouterClient
from .groq import GroqClient
from .llm_registry import get_default_model, get_model_registry, ModelInfo

# Streamlit secrets are available in Cloud runtime; guard the import for tests
try:
    import streamlit as st
    _SECRETS = st.secrets
except Exception:
    _SECRETS = {}

def _get(key: str, default: str = "") -> str:
    # Prefer env (for Docker/CI), then Streamlit secrets
    return os.getenv(key) or _SECRETS.get(key, default)


class _UnavailableClient:
    """Minimal stub used when a provider cannot be initialized."""
    is_configured: bool = False

    def complete(self, *args, **kwargs):
        raise RuntimeError("Selected provider is not configured.")


class LLMProvider:
    """
    Provider-agnostic façade over your concrete clients (OpenRouter, Groq, ...).

    Usage:
        llm = LLMProvider()
        default_model, registry = llm.list_models()
        text = llm.complete(messages, provider="openrouter", model="qwen/qwen-3-instruct")
    """

    def __init__(self):
        # --- OpenRouter (Qwen default) ---
        # Your OpenRouterClient already reads keys/settings from secrets/env.
        self._openrouter = OpenRouterClient()

        # --- Groq (Llama) ---
        # Your groq.py requires an api_key positional arg, so fetch it now.
        groq_key = _get("GROQ_API_KEY", "")
        self._groq_key = groq_key  # keep for is_available fallback logic
        if groq_key:
            try:
                self._groq = GroqClient(api_key=groq_key)
            except TypeError:
                # In case your GroqClient signature changed, try no-arg init (best effort)
                try:
                    self._groq = GroqClient()
                except Exception:
                    self._groq = _UnavailableClient()
        else:
            self._groq = _UnavailableClient()

        # Registry default (should point to OpenRouter/Qwen)
        self._default = get_default_model()

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
            client = self._client_for(provider)
        except ValueError:
            return False

        # Prefer provider client flag if present
        is_cfg = getattr(client, "is_configured", None)
        if isinstance(is_cfg, bool):
            return is_cfg

        # Groq fallback: if client has no flag, use presence of the key
        if provider == "groq":
            return bool(self._groq_key)

        # Default: assume available
        return True

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
            # If a provider is not configured, raise to trigger fallback
            if not self.is_available(chosen_provider):
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
            if not self.is_available(self._default.provider):
                # If fallback is also unavailable, re-raise original error
                raise e
            return fallback.complete(
                messages,
                model=self._default.model_id,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
