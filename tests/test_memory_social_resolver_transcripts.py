from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from discord_ai_assistant.ai.memory_domain_registry import DomainMatch
from discord_ai_assistant.ai.memory_session import ConversationSessionState, SessionMessage
from discord_ai_assistant.ai.memory_social_policy import build_social_memory_context
from discord_ai_assistant.storage.agent_database import AgentDatabase


class MemorySocialResolverTranscriptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = AgentDatabase(Path(self.tempdir.name) / "assistant.sqlite3")
        self._remember(1, "原神抽卡保底目前留給芙寧娜", "game", "genshin", "current_state", "原神抽卡狀態")
        self._remember(1, "CS2 最討厭 Mirage 這張圖", "game", "counter_strike", "preference", "CS2 地圖偏好")
        self._remember(2, "平常最常吃豚骨拉麵", "daily", "food", "habit", "飲食習慣")

    def tearDown(self) -> None:
        self.database.close()
        self.tempdir.cleanup()

    def _remember(
        self,
        user_id: int,
        content: str,
        domain: str,
        subdomain: str,
        memory_kind: str,
        category: str,
    ) -> None:
        _, created = self.database._upsert_user_memory(
            1,
            user_id,
            category,
            content,
            source="passive",
            importance=2,
            confidence=0.95,
            memory_kind=memory_kind,
            domain=domain,
            subdomain=subdomain,
            retention="long",
        )
        self.assertTrue(created)

    @staticmethod
    def _explicit(domain: str, subdomain: str, topic: str | None = None) -> DomainMatch:
        return DomainMatch(domain, subdomain, topic, 0.95, "explicit")

    def test_mixed_transcript_keeps_resolver_on_the_resolved_thread(self) -> None:
        state = ConversationSessionState()
        state.observe(
            SessionMessage(50, 1, 100, "原神這池又歪了"),
            explicit=self._explicit("game", "genshin", "gacha"),
            channel_prior=None,
            now=1.0,
        )
        state.observe(
            SessionMessage(50, 2, 101, "晚餐去吃拉麵"),
            explicit=self._explicit("daily", "food"),
            channel_prior=None,
            now=2.0,
        )
        state.observe(
            SessionMessage(50, 1, 102, "CS2 又排到 Mirage"),
            explicit=self._explicit("game", "counter_strike", "map"),
            channel_prior=None,
            now=3.0,
        )

        food_followup = state.observe(
            SessionMessage(50, 2, 103, "那家可以"),
            explicit=None,
            channel_prior=None,
            now=4.0,
        )
        cs_followup = state.observe(
            SessionMessage(50, 1, 104, "又來了"),
            explicit=None,
            channel_prior=None,
            now=5.0,
        )
        genshin_reply = state.observe(
            SessionMessage(50, 1, 105, "又歪了", reply_to_message_id=100),
            explicit=None,
            channel_prior=None,
            now=6.0,
        )

        self.assertEqual((food_followup.domain, food_followup.subdomain), ("daily", "food"))
        self.assertEqual((cs_followup.domain, cs_followup.subdomain), ("game", "counter_strike"))
        self.assertEqual((genshin_reply.domain, genshin_reply.subdomain), ("game", "genshin"))

        food_context = build_social_memory_context(
            self.database,
            guild_id=1,
            speaker_id=2,
            query="那家可以",
            session=food_followup,
            participant_ids=food_followup.participant_ids,
        )
        self.assertIn("豚骨拉麵", food_context)
        self.assertNotIn("Mirage", food_context)
        self.assertNotIn("芙寧娜", food_context)

        cs_context = build_social_memory_context(
            self.database,
            guild_id=1,
            speaker_id=1,
            query="又來了",
            session=cs_followup,
            participant_ids=cs_followup.participant_ids,
        )
        self.assertIn("Mirage", cs_context)
        self.assertNotIn("芙寧娜", cs_context)
        self.assertNotIn("豚骨拉麵", cs_context)

        genshin_context = build_social_memory_context(
            self.database,
            guild_id=1,
            speaker_id=1,
            query="又歪了",
            session=genshin_reply,
            participant_ids=genshin_reply.participant_ids,
        )
        self.assertIn("芙寧娜", genshin_context)
        self.assertNotIn("Mirage", genshin_context)
        self.assertNotIn("豚骨拉麵", genshin_context)


if __name__ == "__main__":
    unittest.main()
