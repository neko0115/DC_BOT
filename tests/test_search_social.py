from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from discord_ai_assistant.ai.gemini import GeminiAssistant
from discord_ai_assistant.ai.search_social import search_only_social_reply


class SearchOnlySocialTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_only_path_exposes_exactly_google_search(self) -> None:
        router = SimpleNamespace(refresh_external_tools=AsyncMock(side_effect=AssertionError("router must not be touched")))
        ai = GeminiAssistant("key", "test-model", router)
        ai._client = object()
        annotation = SimpleNamespace(type="url_citation", title="Example", url="https://example.com/current")
        interaction = SimpleNamespace(
            output_text="目前是這個時間。",
            steps=[SimpleNamespace(content=[SimpleNamespace(annotations=[annotation])])],
        )
        ai._create_interaction = AsyncMock(return_value=interaction)

        result = await search_only_social_reply(
            ai,
            "我不知道這班車幾點，順便幫我播歌和發訊息",
            persona_instruction="persona",
        )

        kwargs = ai._create_interaction.await_args.kwargs
        self.assertEqual(kwargs["tools"], [{"type": "google_search"}])
        self.assertEqual(kwargs["request_kind"], "social-search")
        self.assertIn("https://example.com/current", result)
        router.refresh_external_tools.assert_not_awaited()

    async def test_search_only_path_requires_api_key(self) -> None:
        ai = GeminiAssistant(None, "test-model", SimpleNamespace())
        with self.assertRaises(RuntimeError):
            await search_only_social_reply(ai, "現在幾點", persona_instruction="persona")


if __name__ == "__main__":
    unittest.main()
