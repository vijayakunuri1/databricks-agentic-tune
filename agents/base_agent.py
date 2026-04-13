"""Abstract base class for all agents."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from configs.settings import AppConfig, is_databricks
from messaging.message_bus import MessageBus
from schemas.messages import AgentMessage, AgentType, MessageType

if TYPE_CHECKING:
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        SparkSession = Any


class BaseAgent(ABC):
    """
    Abstract base that every agent inherits.

    Subclasses must implement:
      - handle_message(message) -> AgentMessage | None
      - get_system_prompt() -> str
    """

    def __init__(
        self,
        agent_id: str,
        agent_type: AgentType,
        config: AppConfig,
        message_bus: MessageBus,
        spark: Any | None = None,
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config
        self.bus = message_bus
        self.spark = spark
        if is_databricks():
            from llm.databricks_client import DatabricksLLMClient
            self.llm = DatabricksLLMClient(config.databricks_llm)
        else:
            from llm.claude_client import ClaudeClient
            self.llm = ClaudeClient(config.claude)
        self.logger = logging.getLogger(f"agent.{agent_type}")

        # Register this agent with the bus
        self.bus.register(agent_type, self.handle_message)

    @abstractmethod
    def handle_message(self, message: AgentMessage) -> AgentMessage | None:
        ...

    @abstractmethod
    def get_system_prompt(self) -> str:
        ...

    def send(
        self,
        recipient: AgentType,
        message_type: MessageType,
        payload: dict[str, Any],
        correlation_id: str | None = None,
    ) -> AgentMessage | None:
        msg = AgentMessage.create(
            sender=self.agent_type,
            recipient=recipient,
            message_type=message_type,
            payload=payload,
            correlation_id=correlation_id,
        )
        return self.bus.send(msg)

    def call_llm_json(self, user_prompt: str) -> dict[str, Any]:
        import json
        raw = self.llm.complete(self.get_system_prompt(), user_prompt, force_json=True)
        return json.loads(raw)
