from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum

from discord_ai_assistant.agent.context import AgentContext, ContextBuilder
from discord_ai_assistant.agent.events import AgentEvent, AgentEventBus


class AgentAction(StrEnum):
    IGNORE = "ignore"
    REPLY_TEXT = "reply_text"
    REPLY_VOICE = "reply_voice"
    CALL_TOOL = "call_tool"
    REMEMBER = "remember"
    START_TOPIC = "start_topic"


@dataclass(frozen=True, slots=True)
class AgentDecision:
    action: AgentAction
    reason: str = ""
    content: str | None = None
    tool_name: str | None = None


AgentPolicy = Callable[[AgentContext], Awaitable[AgentDecision | None]]


class AgentCoordinator:
    """Central decision point for Moxue events.

    v1 intentionally defaults to IGNORE so simply wiring the coordinator cannot make
    墨雪 noisier. Existing SocialParticipant behavior remains authoritative until each
    behavior is migrated behind an explicit policy.
    """

    def __init__(self, bus: AgentEventBus | None = None, context_builder: ContextBuilder | None = None) -> None:
        self.bus = bus or AgentEventBus()
        self.context_builder = context_builder or ContextBuilder()
        self._policies: list[AgentPolicy] = []
        self.last_decision: AgentDecision | None = None
        self._subscribed = False

    async def start(self) -> None:
        if not self._subscribed:
            await self.bus.subscribe(None, self._handle_bus_event)
            self._subscribed = True

    async def stop(self) -> None:
        if self._subscribed:
            await self.bus.unsubscribe(None, self._handle_bus_event)
            self._subscribed = False

    def add_policy(self, policy: AgentPolicy) -> None:
        if policy not in self._policies:
            self._policies.append(policy)

    async def decide(self, context: AgentContext) -> AgentDecision:
        for policy in tuple(self._policies):
            decision = await policy(context)
            if decision is not None:
                self.last_decision = decision
                return decision
        decision = AgentDecision(AgentAction.IGNORE, "no-policy-matched")
        self.last_decision = decision
        return decision

    async def publish(self, event: AgentEvent) -> AgentDecision:
        await self.bus.publish(event)
        return self.last_decision or AgentDecision(AgentAction.IGNORE, "coordinator-not-started")

    async def _handle_bus_event(self, event: AgentEvent) -> None:
        context = self.context_builder.build(event)
        await self.decide(context)
