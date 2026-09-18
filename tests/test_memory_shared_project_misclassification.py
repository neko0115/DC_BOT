from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from discord_ai_assistant.ai.memory_phase2 import PassiveMemoryV2Draft
from discord_ai_assistant.ai.memory_phase4 import MemoryObservation
from discord_ai_assistant.memory_v2_runtime import store_passive_memory_v2_drafts
from discord_ai_assistant.storage.agent_database import AgentDatabase


class SharedGameProjectMisclassificationRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = AgentDatabase(Path(self.tempdir.name) / "assistant.sqlite3")

    def tearDown(self) -> None:
        self.database.close()
        self.tempdir.cleanup()

    def _assert_routes_to_shared_candidate(
        self,
        *,
        draft_content: str,
        observation_content: str,
        message_id: int,
        draft_category: str = "事件",
    ) -> None:
        draft = PassiveMemoryV2Draft(
            category=draft_category,
            content=draft_content,
            confidence=0.90,
            importance=2,
            project="Minecraft",
            role="event",
            domain="project",
            subdomain="Minecraft",
            memory_kind="episodic",
            retention="long",
            shared_candidate=False,
            shared_group_event=False,
        )
        observation = MemoryObservation(
            monotonic_at=1.0,
            content=observation_content,
            channel_id=5,
            message_id=message_id,
            created_at="2026-09-06T15:11:00+00:00",
            session_key=f"5:{message_id}",
            domain="game",
            subdomain="minecraft",
            topic="redstone",
            shared_allowed=True,
            participant_ids=(10,),
        )

        stored = store_passive_memory_v2_drafts(
            self.database,
            1,
            10,
            [draft],
            observations=[observation],
        )

        self.assertEqual(stored, 0)
        personal = self.database.connection.execute(
            "SELECT id FROM user_memories WHERE guild_id = 1 AND user_id = 10"
        ).fetchall()
        self.assertEqual(personal, [])
        shared = self.database.connection.execute(
            "SELECT domain, subdomain, memory_kind, status FROM guild_memories WHERE guild_id = 1 ORDER BY id DESC"
        ).fetchone()
        self.assertIsNotNone(shared)
        self.assertEqual(
            (shared["domain"], shared["subdomain"], shared["memory_kind"], shared["status"]),
            ("game", "minecraft", "shared_episode", "candidate"),
        )

    def test_game_episode_observation_overrides_mistaken_project_metadata(self) -> None:
        self._assert_routes_to_shared_candidate(
            draft_content="專案 Minecraft：上次紅石農場被拆除導致整個農場運作停擺。",
            observation_content="Minecraft 上次紅石農場被拆掉，結果整個農場都停了",
            message_id=100,
        )

    def test_original_episode_marker_survives_even_when_ai_paraphrase_drops_it(self) -> None:
        self._assert_routes_to_shared_candidate(
            draft_content="專案 Minecraft：過去曾發生紅石自動門被拆除，導致成員被困在基地外之事件。",
            observation_content="Minecraft 上次紅石自動門被拆掉，大家最後都卡在基地外面",
            message_id=101,
        )

    def test_project_category_with_event_role_still_routes_public_game_episode_to_shared(self) -> None:
        self._assert_routes_to_shared_candidate(
            draft_category="專案",
            draft_content="專案 Minecraft：上次紅石電梯因被拆除而導致整台電梯系統失效。",
            observation_content="Minecraft 上次紅石電梯被拆掉，結果整台電梯都停了",
            message_id=102,
        )

    def test_yesterday_episode_marker_routes_misclassified_project_to_shared(self) -> None:
        self._assert_routes_to_shared_candidate(
            draft_content=(
                "專案 Minecraft：海底神殿裝備箱曾遭拆除，"
                "導致收集的海綿掉入水中。"
            ),
            observation_content=(
                "昨天 Minecraft 海底神殿那個裝備箱真的被挖走了，"
                "害我們收集的海綿全部掉到水裡"
            ),
            message_id=103,
        )


if __name__ == "__main__":
    unittest.main()
