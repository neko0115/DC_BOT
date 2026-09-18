from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from discord import app_commands

from discord_ai_assistant.main import AssistantBot
from discord_ai_assistant.memory_add import (
    MEMORY_ADD_CHOICES,
    install_memory_add_command,
    normalize_memory_add_category,
    store_manual_memory_choice,
)
from discord_ai_assistant.storage.agent_database import AgentDatabase


EXPECTED_CHOICES = (
    "偏好 / 喜好",
    "習慣",
    "興趣",
    "個人資訊 / 事實",
    "提醒",
    "其他",
)


class MemoryAddChoiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = AgentDatabase(Path(self.tempdir.name) / "assistant.sqlite3")

    def tearDown(self) -> None:
        self.database.close()
        self.tempdir.cleanup()

    def test_public_memory_add_exposes_exact_fixed_choices(self) -> None:
        self.assertEqual(tuple(choice.label for choice in MEMORY_ADD_CHOICES), EXPECTED_CHOICES)
        grouped = SimpleNamespace(
            memory=app_commands.Group(name="memory_test", description="test memory group"),
            core=SimpleNamespace(database=self.database),
        )
        install_memory_add_command(grouped)
        command = grouped.memory.get_command("add")
        self.assertIsNotNone(command)
        assert command is not None
        parameter = next(item for item in command.parameters if item.name == "category")
        self.assertEqual(tuple(choice.name for choice in parameter.choices), EXPECTED_CHOICES)
        self.assertEqual(tuple(choice.value for choice in parameter.choices), EXPECTED_CHOICES)

    def test_unsupported_category_is_rejected_server_side(self) -> None:
        for value in ("", "偏好", "project", "1", "任意分類"):
            with self.subTest(value=value):
                self.assertIsNone(normalize_memory_add_category(value))
                with self.assertRaises(ValueError):
                    store_manual_memory_choice(self.database, 1, 2, value, "可以自由輸入的記憶內容")

    def test_choice_maps_to_approved_memory_kind_and_keeps_free_text(self) -> None:
        cases = (
            ("偏好 / 喜好", "偏好", "preference"),
            ("習慣", "習慣", "habit"),
            ("興趣", "興趣", "interest"),
            ("個人資訊 / 事實", "個人資訊", "semantic"),
            ("其他", "其他", "semantic"),
        )
        for index, (choice, stored_category, expected_kind) in enumerate(cases, start=1):
            content = f"自由文字內容 {index}，保留給既有安全與長度檢查"
            memory = store_manual_memory_choice(self.database, 1, index, choice, content)
            row = self.database.connection.execute(
                "SELECT category, content, memory_kind, source FROM user_memories WHERE id = ?",
                (memory.id,),
            ).fetchone()
            self.assertEqual(row["category"], stored_category)
            self.assertEqual(row["content"], content)
            self.assertEqual(row["memory_kind"], expected_kind)
            self.assertEqual(row["source"], "manual")

    def test_reminder_keeps_existing_manual_reminder_compatible_inference(self) -> None:
        memory = store_manual_memory_choice(
            self.database,
            1,
            2,
            "提醒",
            "昨天已經完成過這件事",
        )
        row = self.database.connection.execute(
            "SELECT category, memory_kind FROM user_memories WHERE id = ?", (memory.id,)
        ).fetchone()
        self.assertEqual(row["category"], "提醒")
        self.assertEqual(row["memory_kind"], "episodic")

    def test_production_setup_installs_fixed_memory_add_before_group_sync(self) -> None:
        source = inspect.getsource(AssistantBot.setup_hook)
        self.assertIn("install_memory_add_command(grouped_commands)", source)
        self.assertLess(
            source.index("install_memory_add_command(grouped_commands)"),
            source.index("await self.add_cog(grouped_commands)"),
        )


if __name__ == "__main__":
    unittest.main()
