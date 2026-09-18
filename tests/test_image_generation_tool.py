from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
TOOL_DIR = ROOT / "tools" / "image_generation"


def load_tool():
    path = TOOL_DIR / "tool.py"
    if not path.is_file():
        raise AssertionError("image_generation tool.py has not been implemented")
    spec = importlib.util.spec_from_file_location("moxue_image_generation_tool", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ImageGenerationToolTests(unittest.IsolatedAsyncioTestCase):
    def test_manifest_requires_explicit_generation_words(self) -> None:
        manifest_path = TOOL_DIR / "manifest.json"
        self.assertTrue(manifest_path.is_file(), "image_generation manifest has not been implemented")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertFalse(manifest["always_available"])
        trigger_text = " ".join(manifest["trigger_keywords"])
        self.assertIn("生圖", trigger_text)
        self.assertIn("幫我畫", trigger_text)

    def test_manifest_exposes_canonical_moxue_appearance_flag(self) -> None:
        manifest = json.loads((TOOL_DIR / "manifest.json").read_text(encoding="utf-8"))
        properties = manifest["actions"]["generate_image"]["parameters"]["properties"]
        self.assertIn("use_moxue_appearance", properties)
        self.assertEqual(properties["use_moxue_appearance"]["type"], "boolean")
        self.assertIn("墨雪", properties["use_moxue_appearance"]["description"])

    def test_prompt_uses_canonical_moxue_profile_when_requested(self) -> None:
        tool = load_tool()
        prompt = tool._prompt({"prompt": "用尾巴比愛心", "use_moxue_appearance": True})
        self.assertIn("銀白", prompt)
        self.assertIn("兩條", prompt)
        self.assertIn("exactly two", prompt.lower())

    def test_prompt_auto_detects_named_moxue_reference(self) -> None:
        tool = load_tool()
        prompt = tool._prompt({"prompt": "墨雪在夜空下揮魔法杖"})
        self.assertIn("兩條", prompt)
        self.assertIn("銀白", prompt)

    def test_detect_image_type_and_artifact_descriptor_are_bounded(self) -> None:
        tool = load_tool()
        self.assertEqual(tool._detect_image_type(b"\xff\xd8\xffrest"), ("image/jpeg", ".jpg"))
        self.assertEqual(tool._detect_image_type(b"\x89PNG\r\n\x1a\nrest"), ("image/png", ".png"))
        with self.assertRaises(tool.ImageGenerationError):
            tool._detect_image_type(b"not-an-image")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = tool._write_artifact(b"\xff\xd8\xffjpeg-data", root=root)
            self.assertTrue(str(artifact["relative_path"]).startswith("image_generation/"))
            self.assertEqual(artifact["mime_type"], "image/jpeg")
            self.assertTrue(artifact["delete_after_send"])
            self.assertNotIn("data", artifact)
            self.assertTrue((root / str(artifact["relative_path"])).is_file())

    async def test_generate_image_keeps_binary_out_of_model_result(self) -> None:
        tool = load_tool()
        fake_jpeg = b"\xff\xd8\xffgenerated"
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {"IMAGE_GEN_USER_COOLDOWN_SECONDS": "0"},
            clear=False,
        ), patch.object(
            tool,
            "_request_image",
            AsyncMock(return_value=(fake_jpeg, tool.DEFAULT_MODEL)),
        ), patch.object(
            tool,
            "_project_root",
            return_value=Path(directory),
        ):
            result = await tool._generate_image({"prompt": "a white cat in a cockpit"}, {"user_id": 42})

        self.assertNotIn("image", result)
        self.assertNotIn("base64", str(result).lower())
        self.assertEqual(result["provider"], "cloudflare_workers_ai")
        artifacts = result["_moxue_artifacts"]
        self.assertEqual(len(artifacts), 1)
        self.assertTrue(str(artifacts[0]["relative_path"]).startswith("image_generation/"))

    async def test_missing_cloudflare_credentials_returns_safe_error(self) -> None:
        tool = load_tool()
        with patch.dict(
            os.environ,
            {
                "CLOUDFLARE_ACCOUNT_ID": "",
                "CLOUDFLARE_API_TOKEN": "",
                "IMAGE_GEN_USER_COOLDOWN_SECONDS": "0",
            },
            clear=False,
        ):
            result = await tool.invoke("generate_image", {"prompt": "test"}, {"user_id": 123})
        self.assertTrue(result["error"])
        self.assertIn("CLOUDFLARE_ACCOUNT_ID", result["message"])


if __name__ == "__main__":
    unittest.main()
