from __future__ import annotations

import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from discord_ai_assistant.ai.gemini import GOOGLE_SEARCH_TOOL, GeminiAssistant, GeminiRequestError
from discord_ai_assistant.ai.key_pool import FailoverGeminiClient
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

    async def test_social_normalizes_persona_styled_no_reply(self) -> None:
        interactions = _SequenceInteractions([SimpleNamespace(id="s2", output_text="NO_REPLY喵")])
        assistant = self._assistant(interactions)

        result = await assistant.social_reply("hello", persona_instruction="persona")

        self.assertEqual(result, "NO_REPLY")

    async def test_search_uses_current_search_route(self) -> None:
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
        self.assertEqual(interactions.calls[0]["model"], "gemini-3.6-flash")
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

    async def test_timeout_social_path_normalizes_persona_styled_no_reply(self) -> None:
        interactions = _SequenceInteractions([SimpleNamespace(id="long-1", output_text="NO_REPLY 喵～")])
        assistant = self._assistant(interactions)

        result = await assistant.social_reply_with_timeout(
            "passive decision",
            persona_instruction="persona",
            timeout_seconds=30,
            request_kind="social-long",
        )

        self.assertEqual(result, "NO_REPLY")

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

    async def test_once_preserves_explicit_model_and_continuation_affinity(self) -> None:
        first = SimpleNamespace(id='once-tool', output_text='ok')
        interactions = _SequenceInteractions([first, _GeminiError('SYNTHETIC', 503)])
        assistant = self._assistant(interactions)
        assistant._client = FailoverGeminiClient(('synthetic-key',), client_factory=lambda _: SimpleNamespace(interactions=interactions),
                                                  single_attempt_client_factory=lambda _: SimpleNamespace(interactions=interactions))
        options = dict(model='explicit-model', input=[], tools=[{'type': 'function', 'name': 'demo'}],
                       timeout_seconds=15, request_kind='tool', input_characters=0)
        self.assertIs(await assistant._create_interaction_once(assistant._get_client(), **options), first)
        self.assertEqual(assistant.model_router.model_for_interaction('once-tool'), 'explicit-model')
        with self.assertRaises(GeminiRequestError):
            await assistant._create_interaction_once(
                assistant._get_client(), **{**options, 'model': assistant.model}, previous_interaction_id='once-tool')
        self.assertEqual([c['model'] for c in interactions.calls], ['explicit-model', 'explicit-model'])

    async def test_once_unavailable_route_does_not_send_transport(self) -> None:
        interactions = _SequenceInteractions([])
        assistant = self._assistant(interactions)
        assistant._client = FailoverGeminiClient(('synthetic-key',), client_factory=lambda _: SimpleNamespace(interactions=interactions),
                                                  single_attempt_client_factory=lambda _: SimpleNamespace(interactions=interactions))
        for model in assistant.model_router.candidate_models('search'):
            assistant.model_router.mark_failure('search', model, _GeminiError('SYNTHETIC', 404))
        with self.assertRaises(GeminiRequestError):
            await assistant._create_interaction_once(
                assistant._get_client(), model=assistant.model, input=[], tools=[dict(GOOGLE_SEARCH_TOOL)],
                timeout_seconds=15, request_kind='public-term-search', input_characters=0)
        self.assertEqual(interactions.calls, [])


class BaseInteractionOnceTests(unittest.IsolatedAsyncioTestCase):
    async def test_ordinary_failure_keeps_exception_logging(self) -> None:
        error = RuntimeError("SYNTHETIC_ORDINARY_DIAGNOSTIC")
        transport = _SequenceInteractions([error])
        assistant = GeminiAssistant('synthetic-key', 'model', _Router())
        with self.assertLogs('discord_ai_assistant.ai.gemini', level='ERROR') as captured:
            with self.assertRaises(GeminiRequestError):
                await assistant._create_interaction(
                    SimpleNamespace(interactions=transport), timeout_seconds=15,
                    request_kind='chat', input_characters=0, input=[])
        self.assertEqual(len(captured.records), 1)
        self.assertEqual(captured.records[0].getMessage(), 'Gemini interaction request failed')
        self.assertIs(captured.records[0].exc_info[1], error)
        self.assertIn('Traceback (most recent call last)', captured.output[0])
        self.assertIn('SYNTHETIC_ORDINARY_DIAGNOSTIC', captured.output[0])

    async def test_once_bounds_error_type_and_reads_structural_status_without_chain_text(self) -> None:
        error_type = type('LongError' * 20 + '\nInjected', (RuntimeError,), {})
        for link in ('__cause__', '__context__'):
            with self.subTest(link=link):
                error = error_type('PRIVATE_OUTER_BODY https://example.org/?key=FAKE_KEY')
                setattr(error, link, _GeminiError('PRIVATE_INNER_BODY RAW_DISCORD_SENTINEL', 503))
                transport = _SequenceInteractions([error])
                assistant = GeminiAssistant('synthetic-key', 'model', _Router())
                assistant._single_attempt_client = SimpleNamespace(interactions=transport)
                with self.assertLogs('discord_ai_assistant.ai', level='WARNING') as captured:
                    with self.assertRaises(GeminiRequestError):
                        await assistant._create_interaction_once(
                            assistant._single_attempt_client, timeout_seconds=15,
                            request_kind='public-term-search', input_characters=0, input=[])
                self.assertEqual(len(captured.records), 1)
                record = captured.records[0]
                self.assertIsNone(record.exc_info)
                self.assertRegex(record.getMessage(), r'stage=single_attempt error_type=[A-Za-z0-9_]{1,80} status=503 sdk=\S{1,32}$')
                self.assertLess(len(record.getMessage()), 512)
                for forbidden in ('PRIVATE_', 'example.org', 'FAKE_KEY', 'RAW_DISCORD', 'Traceback'):
                    self.assertNotIn(forbidden, captured.output[0])

    async def test_base_once_sdk_construction_failure_has_one_bounded_diagnostic(self) -> None:
        assistant = GeminiAssistant('synthetic-key', 'model', _Router())
        with self.assertLogs('discord_ai_assistant.ai', level='WARNING') as captured, \
                patch('discord_ai_assistant.ai.key_pool._create_single_attempt_sdk_client',
                      side_effect=RuntimeError('PRIVATE_SDK_COMPAT_SENTINEL')):
            with self.assertRaises(RuntimeError):
                await assistant._create_interaction_once(
                    None, timeout_seconds=15, request_kind='public-term-search', input_characters=0)
        self.assertEqual(len(captured.records), 1)
        self.assertIsNone(captured.records[0].exc_info)
        self.assertIn('stage=single_attempt error_type=RuntimeError status=unknown sdk=', captured.records[0].getMessage())
        self.assertLess(len(captured.records[0].getMessage()), 512)
        self.assertNotIn('PRIVATE_SDK_COMPAT_SENTINEL', captured.output[0])

    async def test_once_uses_base_transport_with_existing_http_timeout(self) -> None:
        result = SimpleNamespace(id='once', output_text='ok')
        transport = _SequenceInteractions([result])
        assistant = GeminiAssistant('synthetic-key', 'model', _Router())
        assistant._single_attempt_client = SimpleNamespace(interactions=transport)
        assistant._create_interaction = AsyncMock(side_effect=AssertionError('do not dispatch through a retry override'))
        self.assertIs(await assistant._create_interaction_once(
            SimpleNamespace(interactions=transport), model='model', input=[], tools=[],
            timeout_seconds=15, request_kind='synthetic-once', input_characters=0), result)
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(transport.calls[0]['timeout'], 10)
        assistant._create_interaction.assert_not_awaited()

    async def test_once_preserves_error_mapping_and_original_cause(self) -> None:
        for error in (_GeminiError('SYNTHETIC', 503), asyncio.TimeoutError()):
            with self.subTest(error=type(error).__name__):
                transport = _SequenceInteractions([error])
                assistant = GeminiAssistant('synthetic-key', 'model', _Router())
                assistant._single_attempt_client = SimpleNamespace(interactions=transport)
                with self.assertRaises(GeminiRequestError) as raised:
                    await assistant._create_interaction_once(
                        SimpleNamespace(interactions=transport), model='model', input=[], tools=[],
                        timeout_seconds=15, request_kind='synthetic-once', input_characters=0)
                cause = raised.exception.__cause__
                # The async transport can recreate TimeoutError across the thread boundary.
                self.assertIsInstance(cause, type(error))
                if not isinstance(error, asyncio.TimeoutError):
                    self.assertIs(cause, error)
                self.assertEqual(len(transport.calls), 1)
                self.assertNotIn('SYNTHETIC', str(raised.exception))

    async def test_once_propagates_cancellation_without_replay(self) -> None:
        create = Mock(side_effect=asyncio.CancelledError())
        assistant = GeminiAssistant('synthetic-key', 'model', _Router())
        assistant._single_attempt_client = SimpleNamespace(interactions=SimpleNamespace(create=create))
        with self.assertRaises(asyncio.CancelledError):
            await assistant._create_interaction_once(
                SimpleNamespace(interactions=SimpleNamespace(create=create)), model='model', input=[], tools=[],
                timeout_seconds=15, request_kind='synthetic-once', input_characters=0)
        self.assertEqual(create.call_count, 1)


if __name__ == "__main__":
    unittest.main()
