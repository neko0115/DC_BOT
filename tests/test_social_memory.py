from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path

from discord_ai_assistant.ai.memory_phase4 import (
    MemoryObservation,
    ensure_memory_phase4_schema,
    record_memory_provenance,
)
from discord_ai_assistant.ai.memory_session import SessionResolution
from discord_ai_assistant.ai.memory_shared import record_shared_candidate
from discord_ai_assistant.storage.agent_database import AgentDatabase


class SocialMemoryPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = AgentDatabase(Path(self.tempdir.name) / "assistant.sqlite3")
        ensure_memory_phase4_schema(self.database)
        self.policy = importlib.import_module("discord_ai_assistant.ai.memory_social_policy")

    def tearDown(self) -> None:
        self.database.close()
        self.tempdir.cleanup()

    @staticmethod
    def _safe_row(**overrides):
        row = {
            "status": "active",
            "source": "passive",
            "category": "自動偏好",
            "content": "我討厭 Mirage",
            "confidence": 0.91,
            "domain": "game",
            "subdomain": "counter_strike",
            "memory_kind": "preference",
            "project": None,
        }
        row.update(overrides)
        return row

    def _create_memory(
        self,
        *,
        user_id: int,
        content: str = "我討厭 Mirage",
        source: str = "passive",
        domain: str = "game",
        subdomain: str = "counter_strike",
        memory_kind: str = "preference",
        confidence: float = 0.91,
        socially_referenceable: bool = False,
        category: str | None = None,
        expires_at: str | None = None,
    ):
        memory, created = self.database._upsert_user_memory(
            1,
            user_id,
            category or ("自動偏好" if source == "passive" else "偏好"),
            content,
            source=source,
            importance=2,
            confidence=confidence,
            memory_kind=memory_kind,
            domain=domain,
            subdomain=subdomain,
            retention="long",
            expires_at=expires_at,
            socially_referenceable=socially_referenceable,
        )
        self.assertTrue(created)
        return memory

    def _provenance(
        self,
        user_id: int,
        memory_id: int,
        message_id: int,
        session_key: str | None,
        *,
        public_social_context: bool = True,
    ) -> None:
        record_memory_provenance(
            self.database,
            1,
            user_id,
            memory_id,
            [
                MemoryObservation(
                    float(message_id),
                    "public evidence",
                    5,
                    message_id,
                    "2026-09-06T10:00:00+00:00",
                    session_key=session_key,
                    domain="game",
                    subdomain="counter_strike",
                    shared_allowed=public_social_context,
                )
            ],
        )

    def test_cross_user_policy_is_narrow_and_rejects_private_or_sensitive_context(self) -> None:
        can_reference = getattr(self.policy, "can_be_cross_user_referenceable", None)
        self.assertTrue(callable(can_reference), "memory_social_policy must expose can_be_cross_user_referenceable")
        if not callable(can_reference):
            return

        self.assertTrue(can_reference(self._safe_row()))
        rejected = (
            self._safe_row(source="manual"),
            self._safe_row(confidence=0.64),
            self._safe_row(domain="project", subdomain="yuupo", project="YuuPo"),
            self._safe_row(domain="social", subdomain="relationship_context", memory_kind="relationship_context", content="我現在有男朋友"),
            self._safe_row(content="聽說 A 喜歡 B"),
            self._safe_row(content="我今晚住台北市某路 123 號 5 樓"),
            self._safe_row(content="我的政治立場偏某黨"),
            self._safe_row(content="我被診斷出糖尿病"),
            self._safe_row(content="我的銀行帳號是 123456789"),
        )
        for row in rejected:
            with self.subTest(content=row["content"], source=row["source"]):
                self.assertFalse(can_reference(row))

    def test_referenceability_requires_two_distinct_nonnull_public_passive_sessions(self) -> None:
        refresh = getattr(self.policy, "refresh_social_referenceability", None)
        self.assertTrue(callable(refresh), "memory_social_policy must expose refresh_social_referenceability")
        if not callable(refresh):
            return

        safe = self._create_memory(user_id=40)
        self._provenance(40, safe.id, 400, "5:1")
        self.assertFalse(refresh(self.database, safe.id))
        row = self.database.connection.execute(
            "SELECT socially_referenceable FROM user_memories WHERE id = ?", (safe.id,)
        ).fetchone()
        self.assertEqual(int(row["socially_referenceable"]), 0)

        self._provenance(40, safe.id, 401, "5:2")
        self.assertTrue(refresh(self.database, safe.id))
        row = self.database.connection.execute(
            "SELECT socially_referenceable FROM user_memories WHERE id = ?", (safe.id,)
        ).fetchone()
        self.assertEqual(int(row["socially_referenceable"]), 1)

        legacy = self._create_memory(user_id=41, content="我最喜歡 Mirage")
        self._provenance(41, legacy.id, 410, None)
        self._provenance(41, legacy.id, 411, None)
        self.assertFalse(refresh(self.database, legacy.id))

        manual = self._create_memory(user_id=42, content="我最喜歡 Inferno", source="manual")
        self._provenance(42, manual.id, 420, "5:1")
        self._provenance(42, manual.id, 421, "5:2")
        self.assertFalse(refresh(self.database, manual.id))

        relationship = self._create_memory(
            user_id=43,
            content="我現在有男朋友",
            domain="social",
            subdomain="relationship_context",
            memory_kind="relationship_context",
        )
        self._provenance(43, relationship.id, 430, "5:1")
        self._provenance(43, relationship.id, 431, "5:2")
        self.assertFalse(refresh(self.database, relationship.id))

        nonpublic = self._create_memory(user_id=44, content="我最討厭 Dust2")
        self._provenance(44, nonpublic.id, 440, "5:1", public_social_context=False)
        self._provenance(44, nonpublic.id, 441, "5:2", public_social_context=False)
        self.assertFalse(refresh(self.database, nonpublic.id))

    def test_cross_user_search_never_escapes_supplied_participant_ids(self) -> None:
        search = getattr(self.database, "search_socially_referenceable_memories", None)
        self.assertTrue(callable(search), "AgentDatabase needs dedicated cross-user social search")
        if not callable(search):
            return

        allowed = self._create_memory(
            user_id=2,
            content="我討厭 Mirage",
            socially_referenceable=True,
        )
        self._create_memory(
            user_id=3,
            content="我超討厭 Mirage",
            socially_referenceable=True,
        )
        expired = self._create_memory(
            user_id=4,
            content="我以前討厭 Mirage",
            socially_referenceable=True,
            expires_at="2026-01-01T00:00:00+00:00",
        )

        matches = search(1, (2, 4), "Mirage", exclude_user_id=1, limit=5)
        self.assertEqual([match.id for match in matches], [allowed.id])
        self.assertNotIn(expired.id, {match.id for match in matches})

    def test_bounded_resolver_enforces_source_and_total_caps(self) -> None:
        build_context = getattr(self.policy, "build_social_memory_context", None)
        self.assertTrue(callable(build_context), "memory_social_policy must expose build_social_memory_context")
        if not callable(build_context):
            return

        for index in range(6):
            self._create_memory(
                user_id=1,
                content=f"原神 個人偏好 {index}",
                domain="game",
                subdomain="genshin",
                memory_kind="preference",
                category=f"原神偏好{index}",
            )
        for index in range(4):
            record_shared_candidate(
                self.database,
                guild_id=1,
                author_id=10 + index,
                content=f"原神 群體事件 {index}",
                domain="game",
                subdomain="genshin",
                memory_kind="shared_episode",
                confidence=0.95,
                importance=2,
                entity_type="event",
                entity=f"genshin_event_{index}",
                session_key=f"50:{index + 1}",
                channel_id=50,
                message_id=500 + index,
                observed_at="2026-09-06T10:00:00+00:00",
                shared_group_event=True,
                participant_ids=(10 + index, 20 + index),
            )
        for user_id in (2, 3):
            for index in range(3):
                self._create_memory(
                    user_id=user_id,
                    content=f"原神 公開偏好 U{user_id}-{index}",
                    domain="game",
                    subdomain="genshin",
                    memory_kind="preference",
                    socially_referenceable=True,
                    category=f"公開原神偏好{user_id}-{index}",
                )

        context = build_context(
            self.database,
            guild_id=1,
            speaker_id=1,
            query="原神 最近在玩什麼",
            session=SessionResolution("50:99", "game", "genshin", None, 0.9, "explicit", (1, 2, 3)),
            participant_ids=(1, 2, 3),
        )
        self.assertIn("<social_memory_context>", context)
        self.assertIn("Speaker", context)
        self.assertIn("Shared group context", context)
        self.assertIn("Other public social context", context)
        self.assertNotIn("message_id", context)
        self.assertNotIn("session_key", context)

        sections = {"Speaker": 0, "Shared group context": 0, "Other public social context": 0}
        current = None
        total = 0
        for line in context.splitlines():
            stripped = line.strip()
            if stripped.rstrip(":") in sections:
                current = stripped.rstrip(":")
            elif stripped.startswith("- ") and current in sections:
                sections[current] += 1
                total += 1
        self.assertLessEqual(sections["Speaker"], 4)
        self.assertLessEqual(sections["Shared group context"], 3)
        self.assertLessEqual(sections["Other public social context"], 2)
        self.assertLessEqual(total, 8)
        self.assertGreater(sections["Speaker"], 0)
        self.assertGreater(sections["Shared group context"], 0)
        self.assertGreater(sections["Other public social context"], 0)

    def test_resolver_filters_private_other_member_context_and_unrelated_familiarity(self) -> None:
        build_context = getattr(self.policy, "build_social_memory_context", None)
        self.assertTrue(callable(build_context))
        if not callable(build_context):
            return

        self._create_memory(
            user_id=1,
            content="原神最近主玩芙寧娜",
            domain="game",
            subdomain="genshin",
            memory_kind="current_main",
        )
        self._create_memory(
            user_id=2,
            content="我討厭 Mirage",
            socially_referenceable=True,
        )
        self._create_memory(
            user_id=2,
            content="我現在有男朋友",
            domain="social",
            subdomain="relationship_context",
            memory_kind="relationship_context",
            socially_referenceable=True,
            category="關係",
        )
        record_shared_candidate(
            self.database,
            guild_id=1,
            author_id=2,
            content="上次 Minecraft 紅石農場被拆壞變成群內梗",
            domain="game",
            subdomain="minecraft",
            memory_kind="inside_joke",
            confidence=0.95,
            importance=2,
            entity_type="event",
            entity="redstone_joke",
            session_key="60:1",
            channel_id=60,
            message_id=600,
            observed_at="2026-09-06T10:00:00+00:00",
            shared_group_event=True,
            participant_ids=(2, 3),
        )

        cs_session = SessionResolution("50:1", "game", "counter_strike", "map", 0.9, "explicit", (1, 2))
        context = build_context(
            self.database,
            guild_id=1,
            speaker_id=1,
            query="Mirage 真的好玩嗎",
            session=cs_session,
            participant_ids=(1, 2),
        )
        self.assertIn("我討厭 Mirage", context)
        self.assertNotIn("男朋友", context)
        self.assertNotIn("紅石農場", context)
        self.assertNotIn("芙寧娜", context)

        unrelated = build_context(
            self.database,
            guild_id=1,
            speaker_id=1,
            query="晚餐要吃什麼",
            session=SessionResolution("50:2", "daily", "food", None, 0.9, "explicit", (1, 2)),
            participant_ids=(1, 2),
        )
        self.assertNotIn("Mirage", unrelated)
        self.assertNotIn("芙寧娜", unrelated)
        self.assertNotIn("紅石農場", unrelated)

    def test_session_metadata_enriches_short_ambiguous_query(self) -> None:
        build_context = getattr(self.policy, "build_social_memory_context", None)
        self.assertTrue(callable(build_context))
        if not callable(build_context):
            return

        memory = self._create_memory(
            user_id=1,
            content="原神抽卡的保底目前留給芙寧娜",
            domain="game",
            subdomain="genshin",
            memory_kind="current_state",
            category="原神抽卡狀態",
        )
        without_session = build_context(
            self.database,
            guild_id=1,
            speaker_id=1,
            query="又歪了",
            session=None,
            participant_ids=(1,),
        )
        with_session = build_context(
            self.database,
            guild_id=1,
            speaker_id=1,
            query="又歪了",
            session=SessionResolution("50:3", "game", "genshin", "gacha", 0.9, "participant", (1, 2)),
            participant_ids=(1, 2),
        )
        self.assertNotIn(memory.content, without_session)
        self.assertIn(memory.content, with_session)


if __name__ == "__main__":
    unittest.main()
