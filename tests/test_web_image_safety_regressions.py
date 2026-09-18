from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from discord_ai_assistant.ai.gemini import AssistantReply
from discord_ai_assistant.artifact_delivery import resolve_artifact_effect
from discord_ai_assistant.artifact_tool_effect_commands import ArtifactToolEffectAssistantCommands
from discord_ai_assistant.tool_effect_commands import ToolEffectAssistantCommands

ROOT = Path(__file__).resolve().parents[1]


def load_tool(name: str):
    path = ROOT / "tools" / name / "tool.py"
    spec = importlib.util.spec_from_file_location(f"moxue_{name}_safety_regression", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WebImageSafetyRegressionTests(unittest.IsolatedAsyncioTestCase):
    def test_web_research_results_remain_bounded_and_filter_unsafe_sources(self) -> None:
        tool = load_tool("web_research")
        clipped = tool._clip("x" * 5000, 100)
        self.assertLessEqual(len(clipped), 100)
        self.assertTrue(clipped.endswith("…"))

        sources = tool._search_sources(
            {
                "results": [
                    {
                        "title": "Good",
                        "url": "https://example.com/a",
                        "content": "a" * 5000,
                        "score": 0.9,
                    },
                    {
                        "title": "Local",
                        "url": "http://127.0.0.1/private",
                        "content": "secret",
                        "score": 1.0,
                    },
                ]
            }
        )
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["url"], "https://example.com/a")
        self.assertLessEqual(len(sources[0]["content"]), tool.MAX_RESULT_CHARACTERS)

    def test_web_research_freshness_stays_allowlisted(self) -> None:
        tool = load_tool("web_research")
        self.assertEqual(tool._freshness_value("week"), "week")
        self.assertIsNone(tool._freshness_value("all"))
        self.assertIsNone(tool._freshness_value(123))

    async def test_web_research_missing_key_does_not_expose_authorization_details(self) -> None:
        tool = load_tool("web_research")
        with patch.dict(os.environ, {"TAVILY_API_KEY": ""}, clear=False):
            result = await tool.invoke("web_research", {"query": "x"}, {})
        self.assertTrue(result["error"])
        self.assertNotIn("Bearer", result["message"])

    def test_image_generation_keeps_webp_detection_and_prompt_bound(self) -> None:
        tool = load_tool("image_generation")
        self.assertEqual(
            tool._detect_image_type(b"RIFF\x00\x00\x00\x00WEBPrest"),
            ("image/webp", ".webp"),
        )
        with self.assertRaises(ValueError):
            tool._prompt({"prompt": "x" * (tool.MAX_PROMPT_CHARACTERS + 1)})

    async def test_image_generation_missing_credentials_do_not_expose_authorization_details(self) -> None:
        tool = load_tool("image_generation")
        with patch.dict(
            os.environ,
            {
                "CLOUDFLARE_ACCOUNT_ID": "",
                "CLOUDFLARE_API_TOKEN": "",
                "IMAGE_GEN_USER_COOLDOWN_SECONDS": "0",
            },
            clear=False,
        ):
            result = await tool.invoke("generate_image", {"prompt": "test"}, {"user_id": 456})
        self.assertTrue(result["error"])
        self.assertNotIn("Bearer", result["message"])

    def test_artifact_resolver_rejects_absolute_paths_and_unsupported_mime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            artifact = project_root / "data" / "tool-artifacts" / "image_generation" / "valid.jpg"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"\xff\xd8\xffpayload")

            effect = {
                "type": "attach_artifact",
                "relative_path": "image_generation/valid.jpg",
                "filename": "generated.jpg",
                "mime_type": "image/jpeg",
                "delete_after_send": True,
            }
            absolute = dict(effect, relative_path="/tmp/outside.jpg")
            unsupported = dict(effect, mime_type="text/plain")
            self.assertIsNone(resolve_artifact_effect(project_root, 1024 * 1024, absolute))
            self.assertIsNone(resolve_artifact_effect(project_root, 1024 * 1024, unsupported))

    def test_artifact_runtime_keeps_original_cog_name(self) -> None:
        self.assertEqual(ArtifactToolEffectAssistantCommands.__cog_name__, "AssistantCommands")

    async def test_valid_artifact_is_sent_and_cleaned_after_send_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project_root = Path(directory)
            artifact = project_root / "data" / "tool-artifacts" / "image_generation" / "abc.jpg"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"\xff\xd8\xffpayload")
            channel = SimpleNamespace(guild=SimpleNamespace(id=1), send=AsyncMock())

            core = ArtifactToolEffectAssistantCommands.__new__(ArtifactToolEffectAssistantCommands)
            core.bot = SimpleNamespace(get_channel=lambda channel_id: channel if channel_id == 10 else None)
            core.settings = SimpleNamespace(project_root=project_root, max_upload_bytes=25 * 1024 * 1024)
            reply = AssistantReply(
                "圖片已生成。",
                used_tools=True,
                effects=(
                    {
                        "type": "attach_artifact",
                        "relative_path": "image_generation/abc.jpg",
                        "filename": "generated.jpg",
                        "mime_type": "image/jpeg",
                        "delete_after_send": True,
                    },
                ),
            )

            with patch.object(
                ToolEffectAssistantCommands,
                "apply_tool_effects",
                AsyncMock(return_value=None),
            ):
                result = await core.apply_tool_effects(1, 10, reply)

            self.assertIsNone(result)
            channel.send.assert_awaited_once()
            kwargs = channel.send.await_args.kwargs
            self.assertEqual(len(kwargs["files"]), 1)
            self.assertEqual(kwargs["files"][0].filename, "generated.jpg")
            self.assertFalse(artifact.exists())


if __name__ == "__main__":
    unittest.main()
