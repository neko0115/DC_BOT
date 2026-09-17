from __future__ import annotations

import asyncio
import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from discord_ai_assistant.ai.knowledge_help import KnowledgeHelpState
from discord_ai_assistant.knowledge_help_runtime import KnowledgeHelpRuntime
from discord_ai_assistant.main import AssistantBot


class _FakeSocial:
    def __init__(self) -> None:
        self.knowledge_help = AsyncMock(return_value="5090 有 32 GB GDDR7 顯存。")
        self.allowed_ids: set[int] = set()

    def can_offer_knowledge_help(self, message) -> bool:
        return message.id in self.allowed_ids


def _message(
    message_id: int,
    *,
    author_id: int = 100,
    bot_author: bool = False,
    channel_id: int = 10,
    raw_mentions: list[int] | None = None,
    content: str = "5090多少顯存",
):
    sent = SimpleNamespace(id=9000 + message_id)
    channel = SimpleNamespace(id=channel_id, send=AsyncMock(return_value=sent))
    reply = AsyncMock(return_value=sent)
    reference = SimpleNamespace(message_id=message_id, fail_if_not_exists=False)
    to_reference = Mock(return_value=reference)
    return SimpleNamespace(
        id=message_id,
        guild=SimpleNamespace(id=1),
        channel=channel,
        author=SimpleNamespace(id=author_id, bot=bot_author),
        raw_mentions=list(raw_mentions or []),
        mentions=[SimpleNamespace(id=item) for item in (raw_mentions or [])],
        reference=None,
        clean_content=content,
        reply=reply,
        to_reference=to_reference,
    )


class _FakeBot:
    def __init__(self, core=None) -> None:
        self.user = SimpleNamespace(id=42)
        self._core = core

    def get_cog(self, name: str):
        return self._core if name == "AssistantCommands" else None


class KnowledgeHelpRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def _runtime(self, social: _FakeSocial, *, wait_seconds: float = 0.01, core=None) -> KnowledgeHelpRuntime:
        state = KnowledgeHelpState(
            cooldown_seconds=0,
            acknowledgement_seconds=0.05,
            quiet_seconds=0.05,
        )
        return KnowledgeHelpRuntime(_FakeBot(core), social, wait_seconds=wait_seconds, state=state)

    async def test_candidate_runs_full_delay_to_reference_safe_send_path_when_no_native_runtime_exists(self) -> None:
        social = _FakeSocial()
        social.allowed_ids.add(1)
        runtime = self._runtime(social)
        message = _message(1)

        await runtime.on_message(message)
        await asyncio.sleep(0.04)

        social.knowledge_help.assert_awaited_once_with(message)
        message.to_reference.assert_called_once_with(fail_if_not_exists=False)
        message.channel.send.assert_awaited_once_with(
            "5090 有 32 GB GDDR7 顯存。",
            reference=message.to_reference.return_value,
            mention_author=False,
        )
        message.reply.assert_not_awaited()
        runtime.cog_unload()

    async def test_native_tool_effect_runtime_prevents_duplicate_fallback_request(self) -> None:
        social = _FakeSocial()
        social.allowed_ids.add(1)
        core = SimpleNamespace(_schedule_knowledge_help=lambda message: None)
        runtime = self._runtime(social, core=core)
        message = _message(1)

        await runtime.on_message(message)
        await asyncio.sleep(0.03)

        social.knowledge_help.assert_not_awaited()
        message.channel.send.assert_not_awaited()
        message.reply.assert_not_awaited()
        self.assertFalse(runtime.state.pending_matches((1, 10), 1))
        runtime.cog_unload()

    async def test_other_human_message_cancels_pending_help(self) -> None:
        social = _FakeSocial()
        social.allowed_ids.add(1)
        runtime = self._runtime(social, wait_seconds=0.05)
        question = _message(1, author_id=100)
        other_human = _message(2, author_id=200, content="我來回答")

        await runtime.on_message(question)
        await asyncio.sleep(0.005)
        await runtime.on_message(other_human)
        await asyncio.sleep(0.07)

        social.knowledge_help.assert_not_awaited()
        question.channel.send.assert_not_awaited()
        question.reply.assert_not_awaited()
        runtime.cog_unload()

    async def test_explicit_bot_mention_never_creates_delayed_duplicate(self) -> None:
        social = _FakeSocial()
        social.allowed_ids.add(1)
        runtime = self._runtime(social)
        message = _message(1, raw_mentions=[42])

        await runtime.on_message(message)
        await asyncio.sleep(0.03)

        social.knowledge_help.assert_not_awaited()
        message.channel.send.assert_not_awaited()
        message.reply.assert_not_awaited()
        runtime.cog_unload()

    def test_assistant_bot_setup_registers_fallback_runtime_cog(self) -> None:
        source = inspect.getsource(AssistantBot.setup_hook)
        self.assertIn("KnowledgeHelpRuntime(self, core_commands.social)", source)


if __name__ == "__main__":
    unittest.main()
