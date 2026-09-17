from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from discord_ai_assistant.ai.gemini import GeminiRequestError
from discord_ai_assistant.ai.social import SocialParticipant


class _History:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int, str, int, int, int | None, int | None, int | None]] = []

    def compressed_for(
        self,
        guild_id: int,
        channel_id: int,
        query: str,
        *,
        recent_limit: int,
        relevant_limit: int,
        max_messages: int | None = None,
        max_characters: int | None = None,
        recent_guarantee: int | None = None,
    ) -> str:
        self.calls.append(
            (
                guild_id,
                channel_id,
                query,
                recent_limit,
                relevant_limit,
                max_messages,
                max_characters,
                recent_guarantee,
            )
        )
        return "最近 Discord 對話：\nA: context"


class SocialKnowledgeSearchTests(unittest.IsolatedAsyncioTestCase):
    def _participant(self) -> SocialParticipant:
        participant = SocialParticipant.__new__(SocialParticipant)
        participant.history = _History()
        participant.can_offer_knowledge_help = lambda message: True
        participant._search_request = AsyncMock(return_value="fresh answer")
        participant._request = AsyncMock(return_value="static answer")
        return participant

    @staticmethod
    def _message(content: str) -> SimpleNamespace:
        return SimpleNamespace(
            guild=SimpleNamespace(id=1),
            channel=SimpleNamespace(id=10),
            clean_content=content,
        )

    def test_persona_dnd_does_not_block_explicit_knowledge_question(self) -> None:
        participant = SocialParticipant.__new__(SocialParticipant)
        participant.ai = SimpleNamespace(enabled=True)
        participant.automatic_enabled = lambda guild_id: True
        participant.is_do_not_disturb = lambda guild_id: True
        participant._channel_allows_knowledge_help = lambda message: True
        participant._small_question_context = lambda message: "最近 Discord 對話：\nA: GPU"
        message = self._message("5090多少顯存")

        self.assertTrue(participant.can_offer_knowledge_help(message))

    async def test_fresh_knowledge_gap_routes_to_search_only_request(self) -> None:
        participant = self._participant()

        result = await participant.knowledge_help(self._message("我不知道這班車幾點"))

        self.assertEqual(result, "fresh answer")
        participant._search_request.assert_awaited_once()
        participant._request.assert_not_awaited()

    async def test_simple_nonfresh_question_uses_social_route(self) -> None:
        participant = self._participant()

        result = await participant.knowledge_help(self._message("3090多少顯存"))

        self.assertEqual(result, "static answer")
        participant._request.assert_awaited_once()
        self.assertEqual(participant._request.await_args.kwargs["request_kind"], "social")
        participant._search_request.assert_not_awaited()

    async def test_normal_contextual_question_uses_chat_route(self) -> None:
        participant = self._participant()

        result = await participant.knowledge_help(self._message("為什麼剛剛會 timeout"))

        self.assertEqual(result, "static answer")
        participant._request.assert_awaited_once()
        self.assertEqual(participant._request.await_args.kwargs["request_kind"], "chat")
        participant._search_request.assert_not_awaited()

    async def test_complex_question_uses_chat_request_with_reasoning_marker(self) -> None:
        participant = self._participant()

        result = await participant.knowledge_help(self._message("比較這兩種架構的優缺點和取捨"))

        self.assertEqual(result, "static answer")
        participant._request.assert_awaited_once()
        self.assertEqual(participant._request.await_args.kwargs["request_kind"], "chat")
        self.assertIn("需要分析", participant._request.await_args.args[0])
        participant._search_request.assert_not_awaited()

    async def test_nonfresh_knowledge_gap_keeps_no_tool_social_request(self) -> None:
        participant = self._participant()

        result = await participant.knowledge_help(self._message("我忘了怎麼查記憶體型號了"))

        self.assertEqual(result, "static answer")
        participant._request.assert_awaited_once()
        self.assertEqual(participant._request.await_args.kwargs["request_kind"], "social")
        participant._search_request.assert_not_awaited()

    async def test_search_failure_stays_silent(self) -> None:
        participant = SocialParticipant.__new__(SocialParticipant)
        participant.ai = SimpleNamespace()
        participant.workload = SimpleNamespace(instruction_for=MagicMock(return_value="mood"))

        with patch(
            "discord_ai_assistant.ai.social.search_only_social_reply",
            new=AsyncMock(side_effect=GeminiRequestError("search failed")),
        ):
            result = await participant._search_request("prompt", 1)

        self.assertIsNone(result)

    async def test_search_no_reply_with_sources_stays_silent(self) -> None:
        participant = SocialParticipant.__new__(SocialParticipant)
        participant.ai = SimpleNamespace()
        participant.workload = SimpleNamespace(instruction_for=MagicMock(return_value="mood"))

        with patch(
            "discord_ai_assistant.ai.social.search_only_social_reply",
            new=AsyncMock(return_value="NO_REPLY\n\n資料來源：\n- Example: https://example.com"),
        ):
            result = await participant._search_request("prompt", 1)

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
