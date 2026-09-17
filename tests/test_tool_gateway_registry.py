from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from discord_ai_assistant.tool_gateway.registry import ToolRegistry


class ToolRegistryTests(unittest.TestCase):
    def test_loads_catalog_and_invokes_tool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tools_root = Path(directory)
            self._write_tool(tools_root / "sample")
            registry = ToolRegistry(tools_root)
            registry.reload()

            self.assertEqual(registry.errors, {})
            self.assertEqual(len(registry.tools), 1)
            catalog = registry.catalog()
            self.assertEqual(catalog[0]["name"], "sample")
            self.assertEqual(catalog[0]["actions"][0]["function_name"], "x_sample_echo")
            self.assertFalse(catalog[0]["actions"][0]["requires_dj"])
            self.assertEqual(catalog[0]["actions"][0]["trigger_keywords"], ["sample"])

            result = asyncio.run(
                registry.invoke("sample", "echo", {"text": "hello"}, {"guild_id": 123})
            )
            self.assertEqual(result, {"text": "hello", "guild_id": 123})

    def test_broken_tool_is_reported_without_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tools_root = Path(directory)
            tool_root = tools_root / "sample"
            self._write_tool(tool_root)
            (tool_root / "CHANGELOG.md").unlink()
            registry = ToolRegistry(tools_root)
            registry.reload()

            self.assertEqual(registry.tools, ())
            self.assertIn("sample", registry.errors)
            self.assertIn("CHANGELOG.md", registry.errors["sample"])

    def test_dj_only_action_is_enforced_by_registry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tools_root = Path(directory)
            self._write_tool(tools_root / "sample", requires_dj=True)
            registry = ToolRegistry(tools_root)
            registry.reload()

            with self.assertRaises(PermissionError):
                asyncio.run(
                    registry.invoke("sample", "echo", {"text": "hello"}, {"guild_id": 123, "is_dj": False})
                )
            result = asyncio.run(
                registry.invoke("sample", "echo", {"text": "hello"}, {"guild_id": 123, "is_dj": True})
            )
            self.assertEqual(result["text"], "hello")

    @staticmethod
    def _write_tool(tool_root: Path, *, requires_dj: bool = False) -> None:
        tool_root.mkdir(parents=True)
        manifest = {
            "name": "sample",
            "version": "0.1.0",
            "description": "sample test tool",
            "enabled": True,
            "entrypoint": "tool.py",
            "trigger_keywords": ["sample"],
            "always_available": False,
            "actions": {
                "echo": {
                    "description": "echo text",
                    "requires_dj": requires_dj,
                    "parameters": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                }
            },
        }
        (tool_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (tool_root / "TOOL.md").write_text("# Sample\n", encoding="utf-8")
        (tool_root / "CHANGELOG.md").write_text("# Changelog\n", encoding="utf-8")
        (tool_root / "tool.py").write_text(
            "async def invoke(action, arguments, context):\n"
            "    return {'text': arguments.get('text'), 'guild_id': context.get('guild_id')}\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
