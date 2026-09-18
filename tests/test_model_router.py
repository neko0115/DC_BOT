from __future__ import annotations

import os
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

from discord_ai_assistant.ai.model_router import (
    GeminiModelRouter,
    WORKLOAD_CHAT,
    WORKLOAD_MEETING,
    WORKLOAD_MEMORY,
    WORKLOAD_REASONING,
    WORKLOAD_SEARCH,
    WORKLOAD_SOCIAL,
    WORKLOAD_TOOLS,
    WORKLOAD_VISION,
    infer_workload,
    load_model_routes,
    seconds_until_pacific_midnight,
)


class _GeminiError(RuntimeError):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class ModelRouterTests(unittest.TestCase):
    def test_default_routes_use_current_search_models(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            routes = load_model_routes("gemini-3.6-flash")
        self.assertEqual(
            routes[WORKLOAD_SEARCH],
            ("gemini-3.6-flash", "gemini-3.5-flash-lite"),
        )
        self.assertEqual(routes[WORKLOAD_SOCIAL][0], "gemini-3.5-flash-lite")
        self.assertEqual(routes[WORKLOAD_MEMORY][0], "gemini-3.1-flash-lite")
        self.assertEqual(routes[WORKLOAD_MEETING][0], "gemini-3.7-flash")

    def test_search_override_accepts_explicit_model_chain(self) -> None:
        with patch.dict(
            os.environ,
            {"GEMINI_MODELS_SEARCH": "gemini-3.7-flash,gemini-3.6-flash"},
            clear=True,
        ):
            routes = load_model_routes("gemini-3.6-flash")
        self.assertEqual(
            routes[WORKLOAD_SEARCH],
            ("gemini-3.7-flash", "gemini-3.6-flash"),
        )

    def test_custom_chat_chain_retains_legacy_model_as_last_resort(self) -> None:
        with patch.dict(
            os.environ,
            {"GEMINI_MODELS_CHAT": "gemini-3.5-flash-lite"},
            clear=True,
        ):
            routes = load_model_routes("legacy-model")
        self.assertEqual(routes[WORKLOAD_CHAT], ("gemini-3.5-flash-lite", "legacy-model"))

    def test_workload_inference(self) -> None:
        self.assertEqual(infer_workload("social", input_characters=20), WORKLOAD_SOCIAL)
        self.assertEqual(infer_workload("memory", input_characters=20), WORKLOAD_MEMORY)
        self.assertEqual(infer_workload("meeting-report", input_characters=9000), WORKLOAD_MEETING)
        self.assertEqual(
            infer_workload(
                "social-search",
                input_characters=20,
                tools=[{"type": "google_search"}],
            ),
            WORKLOAD_SEARCH,
        )
        self.assertEqual(infer_workload("tool", input_characters=20), WORKLOAD_TOOLS)
        self.assertEqual(
            infer_workload(
                "chat",
                input_characters=20,
                input_payload=[{"type": "image", "data": "x"}],
            ),
            WORKLOAD_VISION,
        )
        self.assertEqual(
            infer_workload(
                "chat",
                input_characters=80,
                input_payload=[{"type": "text", "text": "請分析這個架構的 trade-off"}],
            ),
            WORKLOAD_REASONING,
        )
        self.assertEqual(
            infer_workload(
                "chat",
                input_characters=1300,
                input_payload=[{"type": "text", "text": "普通長文"}],
            ),
            WORKLOAD_REASONING,
        )
        self.assertEqual(infer_workload("chat", input_characters=30), WORKLOAD_CHAT)

    def test_quota_failure_temporarily_removes_only_failed_model(self) -> None:
        routes = {workload: ("model-a", "model-b") for workload in (
            WORKLOAD_CHAT,
            WORKLOAD_REASONING,
            WORKLOAD_TOOLS,
            WORKLOAD_VISION,
            WORKLOAD_SOCIAL,
            WORKLOAD_MEMORY,
            WORKLOAD_MEETING,
            WORKLOAD_SEARCH,
        )}
        routes[WORKLOAD_SEARCH] = ("gemini-3.6-flash", "gemini-3.5-flash-lite")
        router = GeminiModelRouter(routes, quota_cooldown_seconds=60)
        router.mark_failure(WORKLOAD_CHAT, "model-a", _GeminiError("RESOURCE_EXHAUSTED", 429))
        self.assertEqual(router.candidate_models(WORKLOAD_CHAT), ("model-b",))

    def test_interaction_continuation_is_pinned_to_originating_model(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            router = GeminiModelRouter(load_model_routes("gemini-3.6-flash"))
        router.remember_interaction(SimpleNamespace(id="interaction-1"), "gemini-3.7-flash")
        self.assertEqual(
            router.candidate_models(WORKLOAD_TOOLS, previous_interaction_id="interaction-1"),
            ("gemini-3.7-flash",),
        )

    def test_shared_search_quota_blocks_current_search_route(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            router = GeminiModelRouter(load_model_routes("gemini-3.6-flash"), quota_cooldown_seconds=60)
        blocked = router.mark_failure(
            WORKLOAD_SEARCH,
            "gemini-3.6-flash",
            _GeminiError("RESOURCE_EXHAUSTED Google Search grounding quota per day", 429),
        )
        self.assertTrue(blocked)
        self.assertEqual(router.candidate_models(WORKLOAD_SEARCH), ())

    def test_daily_reset_uses_next_pacific_midnight(self) -> None:
        pacific = ZoneInfo("America/Los_Angeles")
        now = datetime(2026, 8, 25, 23, 30, tzinfo=pacific)
        self.assertEqual(seconds_until_pacific_midnight(now), 30 * 60)


if __name__ == "__main__":
    unittest.main()
