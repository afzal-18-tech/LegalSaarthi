"""
LegalSaarthi — LLM Client wrapper.
Uses LiteLLM to provide a unified interface for model generation.
"""
import os
# Ensure LiteLLM uses local cost map without attempting GitHub fetch
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

import litellm
from litellm import completion
from typing import Dict, Any, Optional
import re
import time
from dotenv import load_dotenv

from src.utils.logging_setup import get_logger

# Load environment variables (.env) automatically
load_dotenv()

logger = get_logger("generation.llm_client")

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Optional: Disable litellm verbose logging
litellm.suppress_debug_info = True


class LLMClient:
    """Wrapper for LLM generation via litellm with fallback and provenance tracking."""

    def __init__(
        self,
        model_name: str,
        fallback_model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        **kwargs
    ):
        """
        Args:
            model_name: The primary LiteLLM model string (e.g. 'gemini/gemini-1.5-flash').
            fallback_model: Optional fallback model (e.g. 'ollama/qwen2.5:7b').
            temperature: Generation temperature.
            max_tokens: Max output tokens.
            kwargs: Additional litellm kwargs (e.g. api_key, api_base).
        """
        self.model_name = model_name
        self.fallback_model = fallback_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs

        # Provenance tracking
        self.last_model_used: Optional[str] = None
        self.last_latency_s: Optional[float] = None

        # For tracking cost/usage
        self.total_cost = 0.0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    @staticmethod
    def _is_qwen3(model: str) -> bool:
        return "qwen3" in model.lower()

    @staticmethod
    def _prepare_messages(model: str, messages: list) -> list:
        """Model-specific message adjustments.

        Previously appended `/no_think` as a text soft-switch for Qwen3, but
        this was unreliable on longer prompts. Thinking is now disabled via
        the Ollama API-level `think=False` parameter in `_call_model`.
        """
        return messages

    @staticmethod
    def _clean_output(text: Optional[str]) -> Optional[str]:
        if text is None:
            return None
        cleaned = _THINK_RE.sub("", text).strip()
        if not cleaned and text.strip():
            # The entire output was <think>...</think> with nothing left.
            logger.warning(
                "Model output was entirely consumed by <think> tags "
                f"({len(text)} raw chars → 0 clean chars). "
                "Consider increasing max_tokens."
            )
            return None  # triggers retry / fallback
        return cleaned

    def _call_model(self, model: str, messages: list, retries: int = 3) -> Optional[str]:
        """Attempt completion with a specific model with exponential backoff retries."""
        messages = self._prepare_messages(model, messages)

        # Build extra kwargs for this specific call.
        call_kwargs = dict(self.kwargs)
        if self._is_qwen3(model):
            # Disable Qwen3 hidden reasoning at the Ollama API level.
            # This is definitive — unlike the `/no_think` text switch which
            # Qwen3 ignores on longer prompts, the API parameter prevents
            # think tokens from being generated at all.
            call_kwargs["think"] = False

        for attempt in range(1, retries + 1):
            try:
                t0 = time.perf_counter()
                response = completion(
                    model=model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    **call_kwargs
                )
                self.last_latency_s = time.perf_counter() - t0
                
                # Usage tracking (if supported by provider)
                usage = response.get("usage", {})
                self.total_prompt_tokens += usage.get("prompt_tokens", 0)
                self.total_completion_tokens += usage.get("completion_tokens", 0)
                
                try:
                    cost = litellm.completion_cost(completion_response=response)
                    if cost:
                        self.total_cost += cost
                except Exception:
                    pass

                return self._clean_output(response.choices[0].message.content)

            except Exception as e:
                logger.warning(f"Generation with {model} failed (attempt {attempt}/{retries}): {e}")
                if attempt == retries:
                    return None
                time.sleep(2 ** attempt)
        return None

    def generate(self, prompt: str, system_prompt: Optional[str] = None, retries: int = 3) -> str:
        """Generate text from prompt. Tries primary model, falls back to local/secondary model if configured."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # 1. Try primary model
        res = self._call_model(self.model_name, messages, retries=retries)
        if res is not None:
            self.last_model_used = self.model_name
            return res

        # 2. Try fallback model (e.g. local Ollama Qwen) if primary failed
        if self.fallback_model:
            logger.warning(
                f"Primary model '{self.model_name}' failed. Falling back to local/standby model '{self.fallback_model}'..."
            )
            fallback_res = self._call_model(self.fallback_model, messages, retries=retries)
            if fallback_res is not None:
                self.last_model_used = self.fallback_model
                return fallback_res

        self.last_model_used = "failed"
        logger.error(f"Failed to generate response using primary model ({self.model_name})"
                     + (f" and fallback ({self.fallback_model})" if self.fallback_model else ""))
        return f"ERROR: Generation failed for {self.model_name}"

    def get_usage_stats(self) -> Dict[str, Any]:
        """Returns total cost and token usage for this client instance."""
        return {
            "model": self.model_name,
            "cost_usd": self.total_cost,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
        }
