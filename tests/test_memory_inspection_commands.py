from __future__ import annotations

import inspect
import unittest
from types import SimpleNamespace

from discord import app_commands

from discord_ai_assistant.main import AssistantBot
from discord_ai_assistant.memory_inspection_commands import install_memory_inspection_commands


class MemoryInspectionCommandTests(unittest.TestCase):
    def test_installer_adds_commands_to_existing_memory_group(self) -> None:
        group = app_commands.Group(name="memory_test", description="test memory group")
        grouped = SimpleNamespace(
            memory=group,
            core=SimpleNamespace(database=SimpleNamespace()),
        )

        install_memory_inspection_commands(grouped)

        names = {command.name for command in group.commands}
        self.assertEqual(names, {"search", "projects", "project", "provenance", "conflicts"})

    def test_installer_is_idempotent(self) -> None:
        group = app_commands.Group(name="memory_test2", description="test memory group")
        grouped = SimpleNamespace(
            memory=group,
            core=SimpleNamespace(database=SimpleNamespace()),
        )

        install_memory_inspection_commands(grouped)
        install_memory_inspection_commands(grouped)

        names = [command.name for command in group.commands]
        for name in ("search", "projects", "project", "provenance", "conflicts"):
            self.assertEqual(names.count(name), 1)

    def test_production_setup_installs_memory_inspection_before_sync(self) -> None:
        source = inspect.getsource(AssistantBot.setup_hook)
        self.assertIn("install_memory_inspection_commands(grouped_commands)", source)
        self.assertLess(
            source.index("install_memory_inspection_commands(grouped_commands)"),
            source.index("await self.add_cog(grouped_commands)"),
        )


if __name__ == "__main__":
    unittest.main()
