from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from discord_ai_assistant.agent.events import AgentEvent


@dataclass(frozen=True, slots=True)
class AgentContext:
    event: AgentEvent
    recent_messages: tuple[str, ...] = ()
    memories: tuple[str, ...] = ()
    relationship: str | None = None
    state_notes: tuple[str, ...] = ()

    def as_prompt_section(self) -> str:
        sections: list[str] = [f"事件：{self.event.kind.value}"]
        if self.recent_messages:
            sections.append("近期對話：\n" + "\n".join(f"- {item}" for item in self.recent_messages))
        if self.memories:
            sections.append("相關記憶：\n" + "\n".join(f"- {item}" for item in self.memories))
        if self.relationship:
            sections.append(f"互動關係：{self.relationship}")
        if self.state_notes:
            sections.append("目前狀態：\n" + "\n".join(f"- {item}" for item in self.state_notes))
        return "\n\n".join(sections)


class ContextBuilder:
    """Assembles the small, relevant context bundle sent to later agent policies."""

    def build(
        self,
        event: AgentEvent,
        *,
        recent_messages: list[str] | tuple[str, ...] | None = None,
        memories: list[str] | tuple[str, ...] | None = None,
        relationship: str | None = None,
        state_notes: list[str] | tuple[str, ...] | None = None,
    ) -> AgentContext:
        payload = event.payload
        resolved_recent = recent_messages if recent_messages is not None else self._string_list(payload.get("recent_messages"))
        resolved_memories = memories if memories is not None else self._string_list(payload.get("memories"))
        resolved_relationship = relationship if relationship is not None else self._string(payload.get("relationship"))
        resolved_state = state_notes if state_notes is not None else self._string_list(payload.get("state_notes"))
        return AgentContext(
            event=event,
            recent_messages=tuple(resolved_recent[-20:]),
            memories=tuple(resolved_memories[-8:]),
            relationship=resolved_relationship,
            state_notes=tuple(resolved_state[-8:]),
        )

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        if not isinstance(value, (list, tuple)):
            return []
        return [str(item) for item in value if isinstance(item, str) and item.strip()]

    @staticmethod
    def _string(value: Any) -> str | None:
        return value.strip() if isinstance(value, str) and value.strip() else None
