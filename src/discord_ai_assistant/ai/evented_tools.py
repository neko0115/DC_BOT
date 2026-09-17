from __future__ import annotations

from discord_ai_assistant.agent.events import AgentEvent, AgentEventBus, AgentEventKind
from discord_ai_assistant.ai.tools import ToolContext, ToolRouter


class EventedToolRouter(ToolRouter):
    """Existing ToolRouter with passive Agent Core completion events."""

    def __init__(self, *args, event_bus: AgentEventBus | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.event_bus = event_bus

    async def execute(self, name: str, arguments: dict[str, object], context: ToolContext) -> dict[str, object]:
        result = await super().execute(name, arguments, context)
        if self.event_bus is not None:
            message = result.get("message")
            await self.event_bus.publish(
                AgentEvent(
                    AgentEventKind.TOOL_COMPLETED,
                    guild_id=context.guild_id,
                    user_id=context.user_id,
                    payload={
                        "tool_name": name,
                        "message": str(message)[:500] if message is not None else "",
                        "has_effects": bool(result.get("_moxue_effects")),
                    },
                )
            )
        return result
