# mentor/llm/openrouter.py
import os
import requests

# Streamlit secrets guarded
try:
    import streamlit as st
    _SECRETS = st.secrets
except Exception:
    _SECRETS = {}

def _get(key: str, default: str = "") -> str:
    return os.getenv(key) or _SECRETS.get(key, default)

OPENROUTER_BASE = "https://openrouter.ai/api/v1/chat/completions"

class OpenRouterClient:
    def __init__(self, timeout: int = 60):
        self.api_key = _get("OPENROUTER_API_KEY")
        # Default to a zero-cost router; you can override via secrets
        self.model = _get("OPENROUTER_DEFAULT_MODEL", "openrouter/free")
        self.http_referer = _get("OPENROUTER_HTTP_REFERER", "")
        self.x_title = _get("OPENROUTER_X_TITLE", "EUCapML Case Tutor")
        self.timeout = timeout

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _explain_error(self, resp: requests.Response, model_id: str) -> str:
        status = resp.status_code
        server_text = ""
        try:
            data = resp.json()
            server_text = (
                data.get("error", {}).get("message")
                or data.get("message")
                or data.get("detail")
                or ""
            )
        except Exception:
            server_text = resp.text or ""
        msg = f"OpenRouter error {status} for model '{model_id}'."

        # Targeted hints
        if status in (401, 403):
            return msg + " Check OPENROUTER_API_KEY in Streamlit secrets."
        if status == 404:
            return (
                msg
                + " No eligible endpoints. For free models, enable the privacy toggles for free endpoints in your OpenRouter account; or select another model."
            )
        if status == 400:
            # Most common: invalid slug or params
            return (
                msg
                + " Likely an invalid or unauthorized model slug. Try 'openrouter/free' or a Qwen3 free slug like "
                  "'qwen/qwen3-30b-a3b:free'."
            )
        short = (" " + server_text.strip()) if server_text.strip() else ""
        return msg + short

    def complete(self, messages, model=None, temperature=0.2, max_tokens=1200, top_p=0.9) -> str:
        if not self.is_configured:
            raise RuntimeError("OpenRouter not configured (OPENROUTER_API_KEY missing).")

        model_id = model or self.model
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.x_title:
            headers["X-Title"] = self.x_title

        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "top_p": float(top_p),
        }

        resp = requests.post(OPENROUTER_BASE, headers=headers, json=payload, timeout=self.timeout)

        if not (200 <= resp.status_code < 300):
            raise RuntimeError(self._explain_error(resp, model_id))

        data = resp.json()
        return data["choices"][0]["message"]["content"]
