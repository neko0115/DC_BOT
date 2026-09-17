from __future__ import annotations

import unittest
from types import SimpleNamespace

from discord_ai_assistant.ai.gemini import GeminiAssistant


class GeminiResponseTests(unittest.TestCase):
    def test_casual_chat_does_not_include_tools(self) -> None:
        self.assertEqual(GeminiAssistant._tools_for_request("今天過得如何"), [])

    def test_music_and_fresh_requests_select_relevant_tools(self) -> None:
        tools = GeminiAssistant._tools_for_request("推薦幾首最近的歌並幫我播放")

        self.assertEqual(tools[0], {"type": "google_search"})
        self.assertGreater(len(tools), 1)

    def test_weather_requests_select_google_search(self) -> None:
        tools = GeminiAssistant._tools_for_request("明天會下雨嗎？")

        self.assertEqual(tools, [{"type": "google_search"}])

    def test_long_chat_questions_receive_a_longer_timeout(self) -> None:
        self.assertEqual(GeminiAssistant._chat_timeout_seconds("問題" * 101), 150)
        self.assertEqual(GeminiAssistant._chat_timeout_seconds("短問題"), 120)

    def test_search_citations_are_appended(self) -> None:
        annotation = SimpleNamespace(type="url_citation", title="Example", url="https://example.com")
        block = SimpleNamespace(annotations=[annotation])
        step = SimpleNamespace(content=[block])
        interaction = SimpleNamespace(output_text="推薦清單", steps=[step])

        response = GeminiAssistant._format_response(interaction, "fallback")

        self.assertIn("推薦清單", response)
        self.assertIn("https://example.com", response)
