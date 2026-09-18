from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any, Mapping
from zoneinfo import ZoneInfo

LOGGER = logging.getLogger(__name__)

WORKLOAD_CHAT = "chat"
WORKLOAD_REASONING = "reasoning"
WORKLOAD_TOOLS = "tools"
WORKLOAD_VISION = "vision"
WORKLOAD_SOCIAL = "social"
WORKLOAD_MEMORY = "memory"
WORKLOAD_MEETING = "meeting"
WORKLOAD_SEARCH = "search"

WORKLOADS = (
    WORKLOAD_CHAT,
    WORKLOAD_REASONING,
    WORKLOAD_TOOLS,
    WORKLOAD_VISION,
    WORKLOAD_SOCIAL,
    WORKLOAD_MEMORY,
    WORKLOAD_MEETING,
    WORKLOAD_SEARCH,
)

DEFAULT_MODEL_ROUTES: dict[str, tuple[str, ...]] = {
    WORKLOAD_CHAT: (
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemma-4-31b-it",
    ),
    WORKLOAD_REASONING: (
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-2.5-pro",
        "gemma-4-31b-it",
    ),
    WORKLOAD_TOOLS: (
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-2.5-flash",
    ),
    WORKLOAD_VISION: (
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-2.5-flash",
    ),
    WORKLOAD_SOCIAL: (
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemma-4-26b-a4b-it",
    ),
    WORKLOAD_MEMORY: (
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash-lite",
        "gemma-4-26b-a4b-it",
    ),
    WORKLOAD_MEETING: (
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-2.5-pro",
        "gemma-4-31b-it",
    ),
    # Search uses the currently available Interactions API models. Google may change
    # Search eligibility by project/model over time, so keep this chain configurable
    # instead of hard-locking it to a retired model family.
    WORKLOAD_SEARCH: (
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite",
    ),
}

ENV_BY_WORKLOAD = {
    WORKLOAD_CHAT: "GEMINI_MODELS_CHAT",
    WORKLOAD_REASONING: "GEMINI_MODELS_REASONING",
    WORKLOAD_TOOLS: "GEMINI_MODELS_TOOLS",
    WORKLOAD_VISION: "GEMINI_MODELS_VISION",
    WORKLOAD_SOCIAL: "GEMINI_MODELS_SOCIAL",
    WORKLOAD_MEMORY: "GEMINI_MODELS_MEMORY",
    WORKLOAD_MEETING: "GEMINI_MODELS_MEETING",
    WORKLOAD_SEARCH: "GEMINI_MODELS_SEARCH",
}

MODEL_QUOTA_COOLDOWN_SECONDS = 15 * 60
MODEL_TRANSIENT_COOLDOWN_SECONDS = 30
MAX_INTERACTION_MODEL_PINS = 2048

_QUOTA_MARKERS = (
    "RESOURCE_EXHAUSTED",
    "RATE LIMIT",
    "RATE_LIMIT",
    "QUOTA",
    "TOO MANY REQUESTS",
)
_DAILY_MARKERS = (
    "PER DAY",
    "PERDAY",
    "REQUESTS_PER DAY",
    "REQUESTS_PER_DAY",
    "DAILY",
    " RPD",
)
_MODEL_UNAVAILABLE_MARKERS = (
    "MODEL NOT FOUND",
    "MODEL_NOT_FOUND",
    "UNSUPPORTED MODEL",
    "MODEL IS NOT AVAILABLE",
    "MODEL UNAVAILABLE",
    "OVERLOADED",
)
_SEARCH_MARKERS = (
    "GOOGLE SEARCH",
    "GOOGLE_SEARCH",
    "SEARCH GROUNDING",
    "GROUNDING WITH GOOGLE SEARCH",
    "GROUNDING",
)
_REASONING_MARKERS = (
    "分析",
    "比較",
    "推理",
    "規劃",
    "架構",
    "除錯",
    "debug",
    "analyze",
    "compare",
    "architecture",
    "root cause",
    "trade-off",
    "tradeoff",
)


class ModelRouteUnavailable(RuntimeError):
    """Raised when every model for a workload is locally cooling down/disabled."""


def parse_model_chain(value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = (value or "").strip()
    values = tuple(part.strip() for part in raw.split(",") if part.strip()) if raw else default
    return tuple(dict.fromkeys(values))


def load_model_routes(legacy_model: str | None = None) -> dict[str, tuple[str, ...]]:
    """Load per-workload routes while retaining GEMINI_MODEL as a last-resort fallback.

    Search is intentionally excluded from the legacy fallback so an unrelated
    ``GEMINI_MODEL`` cannot silently change its capability route. Override Search
    explicitly with ``GEMINI_MODELS_SEARCH`` when a project needs a different chain.
    """

    legacy = (legacy_model or "").strip()
    routes: dict[str, tuple[str, ...]] = {}
    for workload, default in DEFAULT_MODEL_ROUTES.items():
        chain = parse_model_chain(os.getenv(ENV_BY_WORKLOAD[workload]), default)
        if workload != WORKLOAD_SEARCH and legacy and legacy not in chain:
            chain = (*chain, legacy)
        routes[workload] = chain
    return routes


def _exception_chain(error: BaseException) -> tuple[BaseException, ...]:
    values: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        values.append(current)
        current = current.__cause__ or current.__context__
    return tuple(values)


def error_text(error: BaseException) -> str:
    return " | ".join(str(item) for item in _exception_chain(error)).upper()


def error_status_code(error: BaseException) -> int | None:
    for item in _exception_chain(error):
        status = getattr(item, "status_code", None)
        if isinstance(status, int):
            return status
    return None


def is_quota_error(error: BaseException) -> bool:
    status = error_status_code(error)
    text = error_text(error)
    return status == 429 or any(marker in text for marker in _QUOTA_MARKERS)


def is_model_fallback_error(error: BaseException) -> bool:
    """Return True when trying another model can plausibly repair this request."""

    status = error_status_code(error)
    text = error_text(error)
    if status == 429 or (isinstance(status, int) and 500 <= status <= 599):
        return True
    if status == 404 or any(marker in text for marker in _MODEL_UNAVAILABLE_MARKERS):
        return True
    return any(isinstance(item, asyncio.TimeoutError) for item in _exception_chain(error))


def is_daily_quota_error(error: BaseException) -> bool:
    text = error_text(error)
    return is_quota_error(error) and any(marker in text for marker in _DAILY_MARKERS)


def is_shared_search_quota_error(error: BaseException) -> bool:
    text = error_text(error)
    return is_quota_error(error) and any(marker in text for marker in _SEARCH_MARKERS)


def retry_delay_seconds(error: BaseException) -> float | None:
    text = error_text(error)
    for pattern in (
        r"RETRY(?:_| )?DELAY[^0-9]{0,24}(\d+(?:\.\d+)?)S",
        r"RETRY AFTER[^0-9]{0,12}(\d+(?:\.\d+)?)",
    ):
        match = re.search(pattern, text)
        if match:
            return max(1.0, min(float(match.group(1)), 24 * 60 * 60))
    return None


def seconds_until_pacific_midnight(now: datetime | None = None) -> float:
    timezone = ZoneInfo("America/Los_Angeles")
    current = now.astimezone(timezone) if now is not None else datetime.now(timezone)
    tomorrow = (current + timedelta(days=1)).date()
    reset = datetime.combine(tomorrow, datetime.min.time(), tzinfo=timezone)
    return max(1.0, (reset - current).total_seconds())


def _input_text(input_payload: object) -> str:
    if isinstance(input_payload, str):
        return input_payload
    if not isinstance(input_payload, list):
        return ""
    pieces: list[str] = []
    for item in input_payload:
        if isinstance(item, dict) and item.get("type") == "text":
            value = item.get("text")
            if isinstance(value, str):
                pieces.append(value)
    return "\n".join(pieces)


def _has_image(input_payload: object) -> bool:
    return bool(
        isinstance(input_payload, list)
        and any(isinstance(item, dict) and item.get("type") == "image" for item in input_payload)
    )


def _has_google_search(tools: object) -> bool:
    return bool(
        isinstance(tools, list)
        and any(isinstance(tool, dict) and tool.get("type") == "google_search" for tool in tools)
    )


def infer_workload(
    request_kind: str,
    *,
    input_characters: int,
    input_payload: object = None,
    tools: object = None,
) -> str:
    kind = request_kind.strip().lower()
    if _has_google_search(tools) or "search" in kind:
        return WORKLOAD_SEARCH
    if kind.startswith("meeting"):
        return WORKLOAD_MEETING
    if kind == "memory" or kind.startswith("memory-"):
        return WORKLOAD_MEMORY
    if kind.startswith("social"):
        return WORKLOAD_SOCIAL
    if kind.startswith("tool"):
        return WORKLOAD_TOOLS
    if _has_image(input_payload):
        return WORKLOAD_VISION
    if kind == "chat":
        text = _input_text(input_payload).lower()
        if input_characters >= 1200 or any(marker in text for marker in _REASONING_MARKERS):
            return WORKLOAD_REASONING
    return WORKLOAD_CHAT


class GeminiModelRouter:
    """Thread-safe workload router with lightweight model cooldown and continuation affinity."""

    def __init__(
        self,
        routes: Mapping[str, tuple[str, ...]],
        *,
        quota_cooldown_seconds: float = MODEL_QUOTA_COOLDOWN_SECONDS,
        transient_cooldown_seconds: float = MODEL_TRANSIENT_COOLDOWN_SECONDS,
    ) -> None:
        normalized = {
            workload: tuple(dict.fromkeys(model.strip() for model in models if model.strip()))
            for workload, models in routes.items()
        }
        for workload in WORKLOADS:
            if not normalized.get(workload):
                raise ValueError(f"Missing Gemini model route for workload: {workload}")
        self.routes = normalized
        self.quota_cooldown_seconds = max(1.0, float(quota_cooldown_seconds))
        self.transient_cooldown_seconds = max(1.0, float(transient_cooldown_seconds))
        self._blocked_until: dict[str, float] = {}
        self._disabled: set[str] = set()
        self._search_blocked_until = 0.0
        self._interaction_models: OrderedDict[str, str] = OrderedDict()
        self._lock = threading.Lock()

    def model_for_interaction(self, interaction_id: str | None) -> str | None:
        if not interaction_id:
            return None
        with self._lock:
            return self._interaction_models.get(interaction_id)

    def candidate_models(self, workload: str, *, previous_interaction_id: str | None = None) -> tuple[str, ...]:
        pinned = self.model_for_interaction(previous_interaction_id)
        if pinned is not None:
            return (pinned,)

        now = time.monotonic()
        with self._lock:
            if workload == WORKLOAD_SEARCH and self._search_blocked_until > now:
                return ()
            return tuple(
                model
                for model in self.routes.get(workload, self.routes[WORKLOAD_CHAT])
                if model not in self._disabled and self._blocked_until.get(model, 0.0) <= now
            )

    def remember_interaction(self, interaction: Any, model: str) -> None:
        interaction_id = getattr(interaction, "id", None)
        if not isinstance(interaction_id, str) or not interaction_id:
            return
        with self._lock:
            self._interaction_models[interaction_id] = model
            self._interaction_models.move_to_end(interaction_id)
            while len(self._interaction_models) > MAX_INTERACTION_MODEL_PINS:
                self._interaction_models.popitem(last=False)

    def mark_success(self, model: str) -> None:
        with self._lock:
            self._blocked_until.pop(model, None)

    def mark_failure(self, workload: str, model: str, error: BaseException) -> bool:
        """Record a model failure and return True when a shared Search pool was blocked."""

        now = time.monotonic()
        text = error_text(error)
        status = error_status_code(error)
        with self._lock:
            if status == 404 or any(marker in text for marker in _MODEL_UNAVAILABLE_MARKERS[:4]):
                self._disabled.add(model)
                LOGGER.warning("Disabled unavailable Gemini model %s for this process", model)
                return False

            if workload == WORKLOAD_SEARCH and is_shared_search_quota_error(error):
                delay = (
                    seconds_until_pacific_midnight()
                    if is_daily_quota_error(error)
                    else retry_delay_seconds(error) or self.quota_cooldown_seconds
                )
                self._search_blocked_until = max(self._search_blocked_until, now + delay)
                LOGGER.warning("Shared Gemini Search pool cooling down for %.0f seconds", delay)
                return True

            if is_quota_error(error):
                delay = (
                    seconds_until_pacific_midnight()
                    if is_daily_quota_error(error)
                    else retry_delay_seconds(error) or self.quota_cooldown_seconds
                )
            else:
                delay = retry_delay_seconds(error) or self.transient_cooldown_seconds
            self._blocked_until[model] = max(self._blocked_until.get(model, 0.0), now + delay)
            LOGGER.warning("Gemini model %s cooling down for %.0f seconds", model, delay)
            return False

    def unavailable_message(self, workload: str) -> str:
        if workload == WORKLOAD_SEARCH:
            return "Gemini 搜尋目前不可用，請稍後再試。"
        return "目前這類工作可用的 Gemini 模型都在冷卻或暫時不可用，請稍後再試。"
