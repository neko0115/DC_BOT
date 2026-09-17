from __future__ import annotations

import asyncio
import unittest

from discord_ai_assistant.agent import (
    AgentAction,
    AgentCoordinator,
    AgentDecision,
    AgentEvent,
    AgentEventKind,
    ContextBuilder,
)


class AgentCoreTests(unittest.TestCase):
    def test_context_builder_caps_context_sizes(self) -> None:
        builder = ContextBuilder()
        event = AgentEvent(AgentEventKind.MESSAGE_RECEIVED, guild_id=1, channel_id=2, user_id=3)
        context = builder.build(
            event,
            recent_messages=[str(index) for index in range(30)],
            memories=[str(index) for index in range(12)],
            state_notes=[str(index) for index in range(12)],
        )
        self.assertEqual(len(context.recent_messages), 20)
        self.assertEqual(len(context.memories), 8)
        self.assertEqual(len(context.state_notes), 8)
        self.assertEqual(context.recent_messages[0], "10")

    def test_context_builder_hydrates_from_event_payload(self) -> None:
        event = AgentEvent(
            AgentEventKind.MESSAGE_RECEIVED,
            payload={
                "recent_messages": ["A: hello", "B: hi"],
                "memories": ["[興趣] 飛行模擬"],
                "relationship": "認識",
                "state_notes": ["music playing"],
            },
        )
        context = ContextBuilder().build(event)
        self.assertEqual(context.recent_messages, ("A: hello", "B: hi"))
        self.assertEqual(context.memories, ("[興趣] 飛行模擬",))
        self.assertEqual(context.relationship, "認識")
        self.assertEqual(context.state_notes, ("music playing",))

    def test_coordinator_defaults_to_ignore(self) -> None:
        async def run() -> AgentDecision:
            coordinator = AgentCoordinator()
            await coordinator.start()
            try:
                return await coordinator.publish(AgentEvent(AgentEventKind.BOT_READY))
            finally:
                await coordinator.stop()

        decision = asyncio.run(run())
        self.assertEqual(decision.action, AgentAction.IGNORE)
        self.assertEqual(decision.reason, "no-policy-matched")

    def test_first_matching_policy_wins(self) -> None:
        async def ignore_policy(context):
            return None

        async def reply_policy(context):
            return AgentDecision(AgentAction.REPLY_TEXT, "test", "hello")

        async def run() -> AgentDecision:
            coordinator = AgentCoordinator()
            coordinator.add_policy(ignore_policy)
            coordinator.add_policy(reply_policy)
            return await coordinator.decide(ContextBuilder().build(AgentEvent(AgentEventKind.MESSAGE_RECEIVED)))

        decision = asyncio.run(run())
        self.assertEqual(decision.action, AgentAction.REPLY_TEXT)
        self.assertEqual(decision.content, "hello")


if __name__ == "__main__":
    unittest.main()
