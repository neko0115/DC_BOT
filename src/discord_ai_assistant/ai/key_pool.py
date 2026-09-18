from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

LOGGER = logging.getLogger(__name__)

_KEY_FAILURE_STATUS_CODES = {401, 403, 429}
_KEY_FAILURE_MARKERS = (
    "API_KEY_INVALID",
    "API KEY NOT VALID",
    "INVALID API KEY",
    "RESOURCE_EXHAUSTED",
    "RATE LIMIT",
    "RATE_LIMIT",
    "QUOTA",
    "TOO MANY REQUESTS",
)
DEFAULT_QUOTA_COOLDOWN_SECONDS = 15 * 60


def _create_single_attempt_sdk_client(api_key: str) -> Any:
    """Build an isolated Interactions client; unknown retry APIs fail closed.

    google-genai 2.22.0 still retries once with public attempts=0. Its generated
    Interactions resource explicitly treats retry_config=None as no retries.
    Initialize and validate that seam before returning this dedicated client;
    never change the ordinary client's options or SDK module globals.
    """
    from google import genai
    from google.genai import types

    client = None
    try:
        client = genai.Client(api_key=api_key, http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=0),
        ))
        configuration = client.interactions.sdk_configuration
        retry = configuration.retry_config
        if (getattr(retry, "strategy", None) != "attempt-count-backoff"
                or type(getattr(retry, "max_retries", None)) is not int):
            raise RuntimeError("Unsupported Interactions retry configuration")
        configuration.retry_config = None
        if client.interactions.sdk_configuration.retry_config is not None:
            raise RuntimeError("Interactions retry configuration was not applied")
        return client
    except Exception:
        if client is not None:
            client.close()
        # The explicit one-attempt caller owns the single bounded diagnostic.
        raise RuntimeError("Public Term single-attempt SDK unavailable") from None


def load_gemini_api_keys(primary: str | None = None) -> tuple[str, ...]:
    """Load the primary Gemini key plus optional backup keys without duplicates.

    Backups can be supplied as a comma-separated GEMINI_BACKUP_API_KEYS value and/or
    numbered GEMINI_API_KEY_2 .. GEMINI_API_KEY_10 variables. The primary argument is
    used by callers that already loaded GEMINI_API_KEY through Settings.
    """

    candidates: list[str] = []
    primary_value = (primary or os.getenv("GEMINI_API_KEY", "")).strip()
    if primary_value:
        candidates.append(primary_value)

    backup_value = os.getenv("GEMINI_BACKUP_API_KEYS", "")
    candidates.extend(part.strip() for part in backup_value.split(",") if part.strip())
    for index in range(2, 11):
        value = os.getenv(f"GEMINI_API_KEY_{index}", "").strip()
        if value:
            candidates.append(value)

    unique: list[str] = []
    seen: set[str] = set()
    for key in candidates:
        if key in seen:
            continue
        seen.add(key)
        unique.append(key)
    return tuple(unique)


def is_gemini_key_failure(error: Exception) -> bool:
    """Return True only for failures where another API key is worth trying."""

    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        status_code = getattr(current, "status_code", None)
        if status_code in _KEY_FAILURE_STATUS_CODES:
            return True
        message = str(current).upper()
        if any(marker in message for marker in _KEY_FAILURE_MARKERS):
            return True
        current = current.__cause__ or current.__context__
    return False


def _is_quota_failure(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    message = str(error).upper()
    return status_code == 429 or any(
        marker in message
        for marker in ("RESOURCE_EXHAUSTED", "RATE LIMIT", "RATE_LIMIT", "QUOTA", "TOO MANY REQUESTS")
    )


class GeminiKeyPool:
    """Thread-safe key selector that avoids repeatedly hammering a failed key."""

    def __init__(self, keys: Sequence[str], *, quota_cooldown_seconds: float = DEFAULT_QUOTA_COOLDOWN_SECONDS) -> None:
        normalized = tuple(dict.fromkeys(key.strip() for key in keys if key and key.strip()))
        if not normalized:
            raise ValueError("At least one Gemini API key is required")
        self.keys = normalized
        self.quota_cooldown_seconds = max(1.0, float(quota_cooldown_seconds))
        self._active_index = 0
        self._blocked_until: dict[int, float] = {}
        self._disabled: set[int] = set()
        self._lock = threading.Lock()

    @property
    def active_number(self) -> int:
        with self._lock:
            return self._active_index + 1

    def candidate_indexes(self) -> tuple[int, ...]:
        now = time.monotonic()
        with self._lock:
            count = len(self.keys)
            ordered = tuple((self._active_index + offset) % count for offset in range(count))
            available = tuple(
                index
                for index in ordered
                if index not in self._disabled and self._blocked_until.get(index, 0.0) <= now
            )
            if available:
                return available
            # If every key is cooling down, make one pass in rotation order so callers
            # still receive the real upstream error instead of a local artificial one.
            return tuple(index for index in ordered if index not in self._disabled) or ordered

    def mark_failure(self, index: int, error: Exception) -> None:
        with self._lock:
            if _is_quota_failure(error):
                self._blocked_until[index] = time.monotonic() + self.quota_cooldown_seconds
            else:
                # Invalid/unauthorized keys will not heal during this process lifetime.
                self._disabled.add(index)
            for offset in range(1, len(self.keys) + 1):
                candidate = (index + offset) % len(self.keys)
                if candidate not in self._disabled:
                    self._active_index = candidate
                    break

    def mark_success(self, index: int) -> None:
        with self._lock:
            self._active_index = index
            self._blocked_until.pop(index, None)


class _FailoverInteractions:
    def __init__(
        self,
        pool: GeminiKeyPool,
        client_factory: Callable[[str], Any],
        single_attempt_client_factory: Callable[[str], Any],
    ) -> None:
        self.pool = pool
        self.client_factory = client_factory
        self._clients: dict[int, Any] = {}
        self._single_attempt_client_factory = single_attempt_client_factory
        self._single_attempt_clients: dict[int, Any] = {}
        self._single_attempt_lock = threading.Lock()
        # A previous_interaction_id can be scoped to the project/key that created it.
        # Pin continuations to the originating key so backups from another project are
        # never asked to resume an interaction they do not own.
        self._interaction_keys: dict[str, int] = {}

    def create_once(self, **kwargs: object) -> Any:
        """Try one current key; failure state only affects subsequent requests."""
        previous_id = kwargs.get("previous_interaction_id")
        pinned_index = self._interaction_keys.get(previous_id) if isinstance(previous_id, str) else None
        index = pinned_index if pinned_index is not None else self.pool.candidate_indexes()[0]
        with self._single_attempt_lock:
            client = self._single_attempt_clients.get(index)
            if client is None:
                client = self._single_attempt_client_factory(self.pool.keys[index])
                self._single_attempt_clients[index] = client
        try:
            result = client.interactions.create(**kwargs)
        except Exception as error:
            if is_gemini_key_failure(error):
                self.pool.mark_failure(index, error)
            raise
        interaction_id = getattr(result, "id", None)
        if isinstance(interaction_id, str) and interaction_id:
            self._interaction_keys[interaction_id] = index
        self.pool.mark_success(index)
        return result

    def create(self, **kwargs: object) -> Any:
        previous_id = kwargs.get("previous_interaction_id")
        pinned_index = self._interaction_keys.get(previous_id) if isinstance(previous_id, str) else None
        candidates = (pinned_index,) if pinned_index is not None else self.pool.candidate_indexes()
        last_error: Exception | None = None

        for position, index in enumerate(candidates):
            client = self._clients.get(index)
            if client is None:
                client = self.client_factory(self.pool.keys[index])
                self._clients[index] = client
            try:
                result = client.interactions.create(**kwargs)
            except Exception as error:
                last_error = error
                if not is_gemini_key_failure(error):
                    raise
                self.pool.mark_failure(index, error)
                # A continuation must stay on its originating project. If that key is
                # exhausted, fail this one in-flight continuation safely; the next fresh
                # request will start on the next available backup key.
                if pinned_index is not None or position == len(candidates) - 1:
                    raise
                LOGGER.warning(
                    "Gemini API key #%s hit a key-specific failure; switching to backup key",
                    index + 1,
                )
                continue

            interaction_id = getattr(result, "id", None)
            if isinstance(interaction_id, str) and interaction_id:
                self._interaction_keys[interaction_id] = index
            self.pool.mark_success(index)
            return result

        assert last_error is not None
        raise last_error


class FailoverGeminiClient:
    """Small compatibility wrapper exposing the google-genai interactions interface."""

    def __init__(
        self,
        keys: Sequence[str],
        *,
        client_factory: Callable[[str], Any] | None = None,
        single_attempt_client_factory: Callable[[str], Any] | None = None,
        quota_cooldown_seconds: float = DEFAULT_QUOTA_COOLDOWN_SECONDS,
    ) -> None:
        if client_factory is None:
            from google import genai

            client_factory = lambda key: genai.Client(api_key=key)
        self.key_pool = GeminiKeyPool(keys, quota_cooldown_seconds=quota_cooldown_seconds)
        self.interactions = _FailoverInteractions(
            self.key_pool, client_factory,
            single_attempt_client_factory or _create_single_attempt_sdk_client,
        )
