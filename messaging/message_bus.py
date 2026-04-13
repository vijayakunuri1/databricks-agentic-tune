"""In-process message bus for synchronous agent communication."""
from __future__ import annotations

import logging
from typing import Callable

from schemas.messages import AgentMessage, AgentType

logger = logging.getLogger(__name__)

Handler = Callable[[AgentMessage], AgentMessage | None]


class MessageBus:
    """
    Synchronous in-process message bus.

    Agents register handlers for their message types.
    The Orchestrator drives the conversation by calling send() which
    immediately delivers to the recipient and returns the response.
    """

    def __init__(self, mode: str = "in_process"):
        self.mode = mode
        self._handlers: dict[AgentType, Handler] = {}
        self._message_log: list[AgentMessage] = []

    def register(self, agent_type: AgentType, handler: Handler) -> None:
        self._handlers[agent_type] = handler
        logger.debug("Registered handler for agent: %s", agent_type)

    def send(self, message: AgentMessage) -> AgentMessage | None:
        recipient = message.header.recipient_agent
        handler = self._handlers.get(recipient)
        if handler is None:
            raise RuntimeError(f"No handler registered for agent: {recipient}")

        self._message_log.append(message)
        logger.debug(
            "Delivering %s -> %s [%s]",
            message.header.sender_agent,
            recipient,
            message.header.message_type,
        )
        response = handler(message)
        if response:
            self._message_log.append(response)
        return response

    def get_log(self) -> list[AgentMessage]:
        return list(self._message_log)

    def clear_log(self) -> None:
        self._message_log.clear()
