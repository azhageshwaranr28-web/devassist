"""OpenAI-compatible grounded answer generation."""
from __future__ import annotations

from typing import Any

from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import settings
from generation.prompts import SYSTEM_PROMPT, user_prompt


class LLMConfigurationError(RuntimeError):
    pass


class GroundedGenerator:
    def __init__(self) -> None:
        if not settings.llm_api_key or not settings.llm_model or settings.llm_model.startswith("replace-"):
            raise LLMConfigurationError(
                "Set LLM_API_KEY and LLM_MODEL in .env. LLM_BASE_URL is optional for OpenAI-compatible providers."
            )
        kwargs: dict[str, Any] = {"api_key": settings.llm_api_key, "timeout": 45.0, "max_retries": 0}
        if settings.llm_base_url:
            kwargs["base_url"] = settings.llm_base_url
        self.client = OpenAI(**kwargs)

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def generate(self, query: str, chunks: list[dict[str, Any]]) -> str:
        response = self.client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(query, chunks)},
            ],
            temperature=0.1,
            max_tokens=900,
        )
        text = response.choices[0].message.content
        if not text or not text.strip():
            raise RuntimeError("The LLM returned an empty response.")
        return text.strip()
