from __future__ import annotations

import importlib.util
import json
import os
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
TOOL_DIR = ROOT / "tools" / "web_research"


def load_tool():
    path = TOOL_DIR / "tool.py"
    if not path.is_file():
        raise AssertionError("web_research tool.py has not been implemented")
    spec = importlib.util.spec_from_file_location("moxue_web_research_tool", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WebResearchToolTests(unittest.IsolatedAsyncioTestCase):
    def test_manifest_is_explicit_only_and_does_not_steal_generic_search(self) -> None:
        manifest_path = TOOL_DIR / "manifest.json"
        self.assertTrue(manifest_path.is_file(), "web_research manifest has not been implemented")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertFalse(manifest["always_available"])
        trigger_text = " ".join(manifest["trigger_keywords"])
        self.assertNotIn("搜尋", trigger_text)
        self.assertNotIn("最新", trigger_text)
        self.assertIn("深入研究", trigger_text)
        self.assertIn("讀網頁", trigger_text)

    def test_public_url_validation_rejects_local_private_and_credentials(self) -> None:
        tool = load_tool()
        self.assertTrue(tool._is_safe_public_url("https://example.com/article"))
        for url in (
            "http://localhost:8000/private",
            "http://127.0.0.1/admin",
            "http://169.254.169.254/latest/meta-data",
            "http://192.168.1.20/status",
            "https://user:pass@example.com/secret",
            "file:///etc/passwd",
        ):
            with self.subTest(url=url):
                self.assertFalse(tool._is_safe_public_url(url))

    async def test_deep_research_performs_search_then_top_source_extract(self) -> None:
        tool = load_tool()
        search_payload = {
            "results": [
                {"title": "A", "url": "https://a.example/x", "content": "search-a", "score": 0.9},
                {"title": "B", "url": "https://b.example/x", "content": "search-b", "score": 0.8},
                {"title": "C", "url": "https://c.example/x", "content": "search-c", "score": 0.7},
                {"title": "D", "url": "https://d.example/x", "content": "search-d", "score": 0.6},
            ],
            "usage": {"credits": 2},
        }
        extract_payload = {
            "results": [
                {"url": "https://a.example/x", "raw_content": "full-a"},
                {"url": "https://b.example/x", "raw_content": "full-b"},
                {"url": "https://c.example/x", "raw_content": "full-c"},
            ],
            "usage": {"credits": 1},
        }
        post = AsyncMock(side_effect=[search_payload, extract_payload])
        with patch.object(tool, "_post_json", post):
            result = await tool._web_research({"query": "test topic", "depth": "deep"})

        self.assertEqual(post.await_count, 2)
        first_endpoint, first_payload = post.await_args_list[0].args
        second_endpoint, second_payload = post.await_args_list[1].args
        self.assertEqual(first_endpoint, "search")
        self.assertEqual(first_payload["search_depth"], "advanced")
        self.assertEqual(second_endpoint, "extract")
        self.assertEqual(second_payload["urls"], [
            "https://a.example/x",
            "https://b.example/x",
            "https://c.example/x",
        ])
        self.assertEqual(result["sources"][0]["content"], "full-a")
        self.assertNotIn("raw_content", result)

    async def test_missing_api_key_returns_safe_error_without_secret_data(self) -> None:
        tool = load_tool()
        with patch.dict(os.environ, {"TAVILY_API_KEY": ""}, clear=False):
            result = await tool.invoke("web_research", {"query": "x"}, {})
        self.assertTrue(result["error"])
        self.assertIn("TAVILY_API_KEY", result["message"])


if __name__ == "__main__":
    unittest.main()
