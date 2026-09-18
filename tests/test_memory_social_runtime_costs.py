from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from discord_ai_assistant.ai.memory_session import ConversationSessionState
from discord_ai_assistant.memory_v2_runtime import MemoryV2PassiveRuntime
from discord_ai_assistant.storage.agent_database import AgentDatabase


class _Bot:
    async def wait_until_ready(self) -> None:
        return None


class _CountingAI:
    enabled = True

    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    async def social_reply_with_timeout(self, *args, **kwargs) -> str:
        self.calls += 1
        if self.fail:
            raise RuntimeError("extractor unavailable")
        return (
            '{"memories":[{"category":"興趣","content":"最近都在玩原神芙寧娜",'
            '"confidence":0.92,"importance":2,"project":null,"role":null,'
            '"domain":"game","subdomain":"genshin","memory_kind":"current_main",'
            '"entity_type":"character","entity":"芙寧娜","retention":"medium",'
            '"socially_referenceable_candidate":false,"shared_candidate":false,'
            '"shared_group_event":false}]}'
        )


class MemorySocialRuntimeCostTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = AgentDatabase(Path(self.tempdir.name) / "assistant.sqlite3")

    def tearDown(self) -> None:
        self.database.close()
        self.tempdir.cleanup()

    @staticmethod
    def _message(message_id: int = 100):
        return SimpleNamespace(
            author=SimpleNamespace(id=10, bot=False),
            guild=SimpleNamespace(id=1),
            channel=SimpleNamespace(id=50, name="閒聊", category=None),
            clean_content="原神最近都在玩芙寧娜",
            id=message_id,
            reference=None,
            created_at=datetime(2026, 9, 6, 10, 0, tzinfo=timezone.utc),
        )

    async def test_session_observation_alone_costs_zero_gemini_calls(self) -> None:
        ai = _CountingAI()
        runtime = MemoryV2PassiveRuntime(_Bot(), self.database, ai, ConversationSessionState())

        await runtime.on_message(self._message())

        self.assertEqual(ai.calls, 0)
        self.assertEqual(len(runtime._pending[(1, 10)]), 1)
        self.assertEqual(
            int(self.database.connection.execute("SELECT COUNT(*) FROM user_memories").fetchone()[0]),
            0,
        )

    async def test_one_due_durable_batch_costs_one_extraction_call(self) -> None:
        ai = _CountingAI()
        runtime = MemoryV2PassiveRuntime(_Bot(), self.database, ai, ConversationSessionState())
        await runtime.on_message(self._message())
        key = (1, 10)
        runtime._pending[key] = [replace(item, monotonic_at=0.0) for item in runtime._pending[key]]

        with patch("discord_ai_assistant.memory_v2_runtime.time.monotonic", return_value=4000.0):
            await MemoryV2PassiveRuntime._flush.coro(runtime)

        self.assertEqual(ai.calls, 1)
        self.assertEqual(
            int(self.database.connection.execute("SELECT COUNT(*) FROM user_memories").fetchone()[0]),
            1,
        )

    async def test_extraction_failure_writes_nothing_and_runtime_flush_does_not_raise(self) -> None:
        ai = _CountingAI(fail=True)
        runtime = MemoryV2PassiveRuntime(_Bot(), self.database, ai, ConversationSessionState())
        await runtime.on_message(self._message())
        key = (1, 10)
        runtime._pending[key] = [replace(item, monotonic_at=0.0) for item in runtime._pending[key]]

        with patch("discord_ai_assistant.memory_v2_runtime.time.monotonic", return_value=4000.0):
            await MemoryV2PassiveRuntime._flush.coro(runtime)

        self.assertEqual(ai.calls, 1)
        self.assertEqual(
            int(self.database.connection.execute("SELECT COUNT(*) FROM user_memories").fetchone()[0]),
            0,
        )


if __name__ == "__main__":
    unittest.main()
