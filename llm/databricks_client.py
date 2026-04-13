"""LLM client that calls Databricks Model Serving (OpenAI-compatible API)."""
from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class DatabricksLLMClient:
    """
    Calls any Databricks-hosted model via the OpenAI-compatible serving endpoint.

    Authentication uses the workspace token that is automatically injected by
    Databricks into every cluster/notebook job — no secrets required.

    Usage:
        client = DatabricksLLMClient(config)
        text = client.complete(system_prompt, user_prompt)
    """

    def __init__(self, config: Any):
        from openai import OpenAI

        host = config.workspace_host or os.getenv("DATABRICKS_HOST", "")
        token = config.token or os.getenv("DATABRICKS_TOKEN", "")

        # Try loading from the Databricks Secrets scope (user-configured)
        if not host or not token:
            try:
                from configs.secrets import get_secret_provider, DEFAULT_SCOPE
                sp = get_secret_provider()
                host = host or sp.get(DEFAULT_SCOPE, "DATABRICKS_HOST")
                token = token or sp.get(DEFAULT_SCOPE, "DATABRICKS_TOKEN")
            except Exception:
                logger.debug("Secrets-scope lookup for credentials skipped")

        # On Databricks clusters the SDK auto-discovers host + token from the
        # cluster's built-in credential store — no env vars or secrets needed.
        if not host or not token:
            try:
                from databricks.sdk import WorkspaceClient
                _sdk = WorkspaceClient()
                host = host or _sdk.config.host or ""
                token = token or _sdk.config.token or ""
            except Exception:
                # Suppress exception details — they may contain partial credentials
                logger.debug("Databricks SDK auth discovery failed (details suppressed)")

        if not host or not token:
            # Do NOT include host/token values in this message
            missing = []
            if not host:
                missing.append("DATABRICKS_HOST")
            if not token:
                missing.append("DATABRICKS_TOKEN")
            raise ValueError(
                f"Missing Databricks credentials: {', '.join(missing)}. "
                "On a Databricks cluster the SDK auto-discovers these. "
                "Locally, set them in your .env file (see .env.example)."
            )

        host = host.rstrip("/")
        self.client = OpenAI(
            api_key=token,
            base_url=f"{host}/serving-endpoints",
        )
        self.model = config.model
        self.max_tokens = config.max_tokens
        self.temperature = config.temperature
        self._call_count = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0

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

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=self.max_tokens,
            temperature=temp,
        )

        self._call_count += 1
        usage = response.usage
        if usage:
            self._total_input_tokens += usage.prompt_tokens or 0
            self._total_output_tokens += usage.completion_tokens or 0

        text = response.choices[0].message.content.strip()
        if force_json:
            text = self._extract_json(text)
        return text

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        raw = self.complete(system_prompt, user_prompt, force_json=True)
        return json.loads(raw)

    @staticmethod
    def _extract_json(text: str) -> str:
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
