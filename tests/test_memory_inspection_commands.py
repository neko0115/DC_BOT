from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from discord import app_commands

from discord_ai_assistant.ai.memory_shared import record_shared_candidate
from discord_ai_assistant.main import AssistantBot
from discord_ai_assistant.memory_inspection_commands import (
    install_memory_inspection_commands,
    normalize_memory_channel_mode,
)
from discord_ai_assistant.storage.agent_database import AgentDatabase


EXPECTED_MEMORY_COMMANDS = {
    "search",
    "projects",
    "project",
    "provenance",
    "conflicts",
    "channel",
    "shared_search",
    "shared_forget",
}


class _Response:
    def __init__(self) -> None:
        self.sent: list[tuple[str, bool]] = []

    def is_done(self) -> bool:
        return bool(self.sent)

    async def send_message(self, text: str, *, ephemeral: bool = False) -> None:
        self.sent.append((text, ephemeral))


class _Followup:
    def __init__(self, response: _Response) -> None:
        self.response = response

    async def send(self, text: str, *, ephemeral: bool = False) -> None:
        self.response.sent.append((text, ephemeral))


class _Core:
    def __init__(self, database: AgentDatabase, *, is_dj: bool) -> None:
        self.database = database
        self._is_dj_value = is_dj

    def _is_dj(self, interaction) -> bool:
        return self._is_dj_value


class MemoryInspectionCommandTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = AgentDatabase(Path(self.tempdir.name) / "assistant.sqlite3")

    def tearDown(self) -> None:
        self.database.close()
        self.tempdir.cleanup()

    @staticmethod
    def _group(core) -> SimpleNamespace:
        return SimpleNamespace(
            memory=app_commands.Group(name="memory_test", description="test memory group"),
            core=core,
        )

    @staticmethod
    def _interaction(*, guild_id: int = 1, channel_id: int = 5):
        response = _Response()
        return SimpleNamespace(
            guild=SimpleNamespace(id=guild_id),
            guild_id=guild_id,
            channel_id=channel_id,
            user=SimpleNamespace(id=10),
            response=response,
            followup=_Followup(response),
        )

    async def test_installer_adds_commands_to_existing_memory_group(self) -> None:
        grouped = self._group(SimpleNamespace(database=SimpleNamespace()))

        install_memory_inspection_commands(grouped)

        names = {command.name for command in grouped.memory.commands}
        self.assertEqual(names, EXPECTED_MEMORY_COMMANDS)

    async def test_installer_is_idempotent(self) -> None:
        grouped = self._group(SimpleNamespace(database=SimpleNamespace()))

        install_memory_inspection_commands(grouped)
        install_memory_inspection_commands(grouped)

        names = [command.name for command in grouped.memory.commands]
        for name in EXPECTED_MEMORY_COMMANDS:
            self.assertEqual(names.count(name), 1)

    async def test_channel_mode_validation_is_strict_and_known_game_aware(self) -> None:
        accepted = {
            "auto": "auto",
            "mixed": "mixed",
            "social": "social",
            "game": "game",
            "GAME:Genshin": "game:genshin",
            "game:minecraft": "game:minecraft",
            "project": "project",
            "off": "off",
        }
        for raw, expected in accepted.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_memory_channel_mode(raw), expected)
        for raw in ("", "random", "game:unknown_title", "project:yuupo", "mixed:game"):
            with self.subTest(raw=raw):
                self.assertIsNone(normalize_memory_channel_mode(raw))

    async def test_channel_command_requires_dj_and_updates_current_channel_only(self) -> None:
        denied_group = self._group(_Core(self.database, is_dj=False))
        install_memory_inspection_commands(denied_group)
        interaction = self._interaction(channel_id=50)
        command = denied_group.memory.get_command("channel")
        self.assertIsNotNone(command)
        await command.callback(interaction, "game:genshin")
        self.assertIsNone(self.database.get_state("memory_channel_mode:1:50"))
        self.assertTrue(interaction.response.sent[-1][1])

        allowed_group = self._group(_Core(self.database, is_dj=True))
        install_memory_inspection_commands(allowed_group)
        interaction = self._interaction(channel_id=50)
        command = allowed_group.memory.get_command("channel")
        await command.callback(interaction, "game:genshin")
        self.assertEqual(self.database.get_state("memory_channel_mode:1:50"), "game:genshin")
        self.assertIsNone(self.database.get_state("memory_channel_mode:1:51"))
        self.assertTrue(interaction.response.sent[-1][1])

    async def test_shared_forget_is_dj_only_guild_scoped_and_leaves_personal_memory(self) -> None:
        shared_id = record_shared_candidate(
            self.database,
            guild_id=1,
            author_id=10,
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
            participant_ids=(10, 11),
        )
        self.assertIsNotNone(shared_id)
        personal, _ = self.database._upsert_user_memory(
            1,
            10,
            "偏好",
            "我喜歡 Minecraft",
            source="manual",
            importance=2,
            confidence=1.0,
            domain="game",
            subdomain="minecraft",
            memory_kind="preference",
        )

        denied_group = self._group(_Core(self.database, is_dj=False))
        install_memory_inspection_commands(denied_group)
        denied = self._interaction(guild_id=1)
        await denied_group.memory.get_command("shared_forget").callback(denied, int(shared_id))
        self.assertIsNotNone(
            self.database.connection.execute(
                "SELECT id FROM guild_memories WHERE id = ?", (int(shared_id),)
            ).fetchone()
        )

        wrong_guild_group = self._group(_Core(self.database, is_dj=True))
        install_memory_inspection_commands(wrong_guild_group)
        wrong_guild = self._interaction(guild_id=2)
        await wrong_guild_group.memory.get_command("shared_forget").callback(wrong_guild, int(shared_id))
        self.assertIsNotNone(
            self.database.connection.execute(
                "SELECT id FROM guild_memories WHERE id = ?", (int(shared_id),)
            ).fetchone()
        )

        allowed = self._interaction(guild_id=1)
        await wrong_guild_group.memory.get_command("shared_forget").callback(allowed, int(shared_id))
        self.assertIsNone(
            self.database.connection.execute(
                "SELECT id FROM guild_memories WHERE id = ?", (int(shared_id),)
            ).fetchone()
        )
        self.assertIsNotNone(
            self.database.connection.execute(
                "SELECT id FROM user_memories WHERE id = ?", (personal.id,)
            ).fetchone()
        )

    async def test_shared_search_is_ephemeral_and_active_only(self) -> None:
        record_shared_candidate(
            self.database,
            guild_id=1,
            author_id=10,
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
            participant_ids=(10, 11),
        )
        grouped = self._group(_Core(self.database, is_dj=False))
        install_memory_inspection_commands(grouped)
        interaction = self._interaction(guild_id=1)

        await grouped.memory.get_command("shared_search").callback(interaction, "Minecraft 紅石")

        text, ephemeral = interaction.response.sent[-1]
        self.assertTrue(ephemeral)
        self.assertIn("紅石農場", text)
        self.assertLessEqual(len(text), 2000)

    async def test_production_setup_installs_memory_inspection_before_sync(self) -> None:
        source = inspect.getsource(AssistantBot.setup_hook)
        self.assertIn("install_memory_inspection_commands(grouped_commands)", source)
        self.assertLess(
            source.index("install_memory_inspection_commands(grouped_commands)"),
            source.index("await self.add_cog(grouped_commands)"),
        )


if __name__ == "__main__":
    unittest.main()
