from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from discord_ai_assistant.ai.gemini import GOOGLE_SEARCH_TOOL, GeminiRequestError
from discord_ai_assistant.ai.resilient_gemini import ResilientGeminiAssistant


class _GeminiError(RuntimeError):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class _SequenceInteractions:
    def __init__(self, outcomes: list[object | Exception]) -> None:
        self.outcomes = outcomes
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        outcome = self.outcomes[len(self.calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _Router:
    async def refresh_external_tools(self) -> None:
        return None

    def external_declarations_for(self, request_text: str, context: object) -> list[dict[str, object]]:
        return []


class ResilientModelRoutingTests(unittest.IsolatedAsyncioTestCase):
    def _assistant(self, interactions: _SequenceInteractions) -> ResilientGeminiAssistant:
        with patch.dict(os.environ, {}, clear=True):
            assistant = ResilientGeminiAssistant(
                "fake-key",
                "gemini-3.6-flash",
                _Router(),
            )
        assistant._client = SimpleNamespace(interactions=interactions)
        return assistant

    async def test_social_uses_flash_lite_first(self) -> None:
        interactions = _SequenceInteractions([SimpleNamespace(id="s1", output_text="ok")])
        assistant = self._assistant(interactions)

        result = await assistant.social_reply("hello", persona_instruction="persona")

        self.assertEqual(result, "ok")
        self.assertEqual(interactions.calls[0]["model"], "gemini-3.5-flash-lite")

    async def test_search_uses_25_flash_not_default_36(self) -> None:
        interactions = _SequenceInteractions([SimpleNamespace(id="search-1", output_text="fresh")])
        assistant = self._assistant(interactions)

        interaction = await assistant._create_interaction(
            assistant._get_client(),
            model=assistant.model,
            input=[{"type": "text", "text": "current fact"}],
            tools=[dict(GOOGLE_SEARCH_TOOL)],
            timeout_seconds=30,
            request_kind="social-search",
            input_characters=12,
        )

        self.assertEqual(interaction.output_text, "fresh")
        self.assertEqual(interactions.calls[0]["model"], "gemini-2.5-flash")
        self.assertEqual(interactions.calls[0]["tools"], [{"type": "google_search"}])

    async def test_model_429_falls_back_to_next_model(self) -> None:
        interactions = _SequenceInteractions(
            [
                _GeminiError("RESOURCE_EXHAUSTED quota", 429),
                SimpleNamespace(id="chat-2", output_text="backup"),
            ]
        )
        assistant = self._assistant(interactions)

        result = await assistant.social_reply_with_timeout(
            "meeting transcript",
            persona_instruction="meeting",
            timeout_seconds=240,
            request_kind="meeting-report",
        )

        self.assertEqual(result, "backup")
        self.assertEqual(
            [call["model"] for call in interactions.calls],
            ["gemini-3.7-flash", "gemini-3.6-flash"],
        )

    async def test_meeting_keeps_task_specific_timeout(self) -> None:
        interactions = _SequenceInteractions([SimpleNamespace(id="meeting-1", output_text="report")])
        assistant = self._assistant(interactions)

        result = await assistant.social_reply_with_timeout(
            "long meeting transcript",
            persona_instruction="meeting persona",
            timeout_seconds=240,
            request_kind="meeting-report",
        )

        self.assertEqual(result, "report")
        self.assertEqual(interactions.calls[0]["model"], "gemini-3.7-flash")
        # GeminiAssistant subtracts its 5-second asyncio margin before passing the
        # underlying SDK HTTP timeout.
        self.assertEqual(interactions.calls[0]["timeout"], 235)

    async def test_tool_continuation_does_not_replay_on_another_model(self) -> None:
        first = SimpleNamespace(id="tool-1", output_text="", steps=[])
        interactions = _SequenceInteractions(
            [first, _GeminiError("RESOURCE_EXHAUSTED quota", 429)]
        )
        assistant = self._assistant(interactions)

        created = await assistant._create_interaction(
            assistant._get_client(),
            model=assistant.model,
            input=[{"type": "text", "text": "do a tool action"}],
            tools=[{"type": "function", "name": "demo"}],
            timeout_seconds=30,
            request_kind="tool",
            input_characters=16,
        )
        self.assertIs(created, first)

        with self.assertRaises(GeminiRequestError):
            await assistant._create_interaction(
                assistant._get_client(),
                model=assistant.model,
                previous_interaction_id="tool-1",
                input=[{"type": "function_result", "name": "demo", "result": []}],
                tools=[{"type": "function", "name": "demo"}],
                timeout_seconds=30,
                request_kind="tool-result",
                input_characters=5,
            )

        self.assertEqual(
            [call["model"] for call in interactions.calls],
            ["gemini-3.7-flash", "gemini-3.7-flash"],
        )


if __name__ == "__main__":
    unittest.main()
