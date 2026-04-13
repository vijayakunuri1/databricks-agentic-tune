"""Inter-agent message schemas (Pydantic v2)."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


AgentType = Literal["orchestrator", "qc_runner", "data_updater"]

MessageType = Literal[
    "QC_RUN_REQUEST",
    "QC_RUN_COMPLETE",
    "CORRECTION_REQUEST",
    "CORRECTION_COMPLETE",
    "ESCALATE_TO_L2",
    "WORKFLOW_COMPLETE",
    "ERROR",
]


class MessageHeader(BaseModel):
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: str
    sender_agent: AgentType
    recipient_agent: AgentType
    message_type: MessageType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    priority: Literal["high", "normal", "low"] = "normal"


class AgentMessage(BaseModel):
    header: MessageHeader
    payload: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def create(
        cls,
        sender: AgentType,
        recipient: AgentType,
        message_type: MessageType,
        payload: dict[str, Any],
        correlation_id: str | None = None,
        priority: Literal["high", "normal", "low"] = "normal",
        metadata: dict[str, Any] | None = None,
    ) -> "AgentMessage":
        return cls(
            header=MessageHeader(
                correlation_id=correlation_id or str(uuid.uuid4()),
                sender_agent=sender,
                recipient_agent=recipient,
                message_type=message_type,
                priority=priority,
            ),
            payload=payload,
            metadata=metadata or {},
        )

    def reply(
        self,
        message_type: MessageType,
        payload: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> "AgentMessage":
        return AgentMessage.create(
            sender=self.header.recipient_agent,
            recipient=self.header.sender_agent,
            message_type=message_type,
            payload=payload,
            correlation_id=self.header.correlation_id,
            metadata=metadata or {},
        )
