from __future__ import annotations

import unittest
from types import SimpleNamespace

from discord_ai_assistant.ai.control_response import NO_REPLY_CONTROL, normalize_control_response
from discord_ai_assistant.ai.search_social import search_only_social_reply


class ProactiveNoReplyControlTests(unittest.TestCase):
    def test_persona_styled_no_reply_variants_normalize(self) -> None:
        for value in (
            "NO_REPLY",
            "NO_REPLY喵",
            "NO_REPLY 喵～",
            "NO_REPLY。",
            "no_reply😺",
            "NO_REPLY喵\n\n資料來源：\n- Example: https://example.com",
        ):
            with self.subTest(value=value):
                self.assertEqual(normalize_control_response(value), NO_REPLY_CONTROL)

    def test_normal_text_is_not_changed(self) -> None:
        self.assertEqual(normalize_control_response("正常回答喵"), "正常回答喵")
        self.assertEqual(normalize_control_response("NO_REPLY 是控制字串"), "NO_REPLY 是控制字串")


class SearchOnlyNoReplyTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_citations_do_not_leak_persona_styled_no_reply(self) -> None:
        class AI:
            api_key = "fake"
            model = "gemini-3.6-flash"

            def _current_time_text(self) -> str:
                return "2026-09-06 01:00 CST"

            def _get_client(self) -> object:
                return object()

            def _system_instruction(self, persona: str, instruction: str) -> str:
                return f"{persona}\n{instruction}"

            async def _create_interaction(self, *args: object, **kwargs: object) -> object:
                return SimpleNamespace(output_text="NO_REPLY喵")

            def _format_response(self, interaction: object, fallback: str) -> str:
                text = getattr(interaction, "output_text", None) or fallback
                return f"{text}\n\n資料來源：\n- Example: https://example.com"

        result = await search_only_social_reply(
            AI(),
            "查一下最新資訊",
            persona_instruction="persona",
        )

        self.assertEqual(result, NO_REPLY_CONTROL)


if __name__ == "__main__":
    unittest.main()
