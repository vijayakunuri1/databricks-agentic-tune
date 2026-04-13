"""Anthropic Claude API wrapper with prompt caching."""
from __future__ import annotations

import json
import logging
from typing import Any

import anthropic

logger = logging.getLogger(__name__)


class ClaudeClient:
    """
    Thin wrapper around the Anthropic SDK.

    - Uses prompt caching (cache_control: ephemeral) on system prompts to
      reduce token cost when agents call Claude in a loop.
    - temperature=0.1 by default (deterministic QC decisions).
    - force_json=True appends a JSON instruction and validates the response.
    """

    def __init__(self, config: Any):
        self.client = anthropic.Anthropic(api_key=config.api_key)
        self.model = config.model
        self.max_tokens = config.max_tokens
        self.temperature = config.temperature
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._call_count = 0

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        force_json: bool = True,
    ) -> str:
        temp = temperature if temperature is not None else self.temperature
        if force_json:
            user_prompt = (
                user_prompt
                + "\n\nRespond with ONLY valid JSON — no markdown fences, no preamble."
            )

        from configs.rate_limiting import check_llm_call, rate_limit_error
        allowed, retry_after = check_llm_call()
        if not allowed:
            raise RuntimeError(
                f"LLM rate limit reached. {rate_limit_error(retry_after, 'llm_call')}"
            )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_prompt}],
            temperature=temp,
        )

        self._call_count += 1
        usage = response.usage
        self._total_input_tokens += usage.input_tokens
        self._total_output_tokens += usage.output_tokens

        text = response.content[0].text.strip()
        if force_json:
            text = self._extract_json(text)
        return text

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        raw = self.complete(system_prompt, user_prompt, force_json=True)
        return json.loads(raw)

    @staticmethod
    def _extract_json(text: str) -> str:
        """Strip markdown fences if the model added them anyway."""
        if text.startswith("```"):
            lines = text.split("\n")
            inner = []
            in_block = False
            for line in lines:
                if line.startswith("```") and not in_block:
                    in_block = True
                    continue
                if line.startswith("```") and in_block:
                    break
                if in_block:
                    inner.append(line)
            return "\n".join(inner)
        return text

    @property
    def stats(self) -> dict[str, int]:
        return {
            "call_count": self._call_count,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
        }
