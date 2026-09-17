from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import aiohttp

from discord_ai_assistant.voice.language_routing import detect_tts_language, normalize_tts_language
from discord_ai_assistant.voice.synthesis import (
    SpeechSynthesizer,
    _NormalizedSpeechSynthesizer,
    _validated_text,
)

LOGGER = logging.getLogger(__name__)
MAX_REMOTE_AUDIO_BYTES = 32 * 1024 * 1024
DEFAULT_BUSY_WAIT_SECONDS = 12.0
DEFAULT_BUSY_POLL_SECONDS = 0.5
DEFAULT_HTTP_QUEUE_GRACE_SECONDS = 20.0


class TTSWorkerBusyError(RuntimeError):
    """A healthy worker could not accept this request before its bounded queue deadline."""


@dataclass(frozen=True, slots=True)
class TTSWorkerSpec:
    name: str
    base_url: str


@dataclass(frozen=True, slots=True)
class WorkerHealth:
    name: str
    status: str
    reason: str = ""
    backend: str = ""
    gpu_utilization: float | None = None
    gpu_memory_free_mb: int | None = None
    active_requests: int | None = None
    queued_requests: int | None = None
    max_queue_size: int | None = None

    @property
    def ready(self) -> bool:
        return self.status == "ready"


def parse_worker_specs(value: str | None) -> tuple[TTSWorkerSpec, ...]:
    """Parse `name=http://host:port;name2=http://host:port` preserving priority order."""
    if not value or not value.strip():
        return ()
    result: list[TTSWorkerSpec] = []
    names: set[str] = set()
    for raw_item in value.split(";"):
        item = raw_item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError("TTS_REMOTE_WORKERS entries must use name=http://host:port")
        raw_name, raw_url = item.split("=", 1)
        name = raw_name.strip().lower()
        url = raw_url.strip().rstrip("/")
        if not name or not url:
            raise ValueError("TTS_REMOTE_WORKERS entries require both name and URL")
        if name == "auto" or name in names:
            raise ValueError(f"Duplicate or reserved TTS worker name: {name}")
        if not (url.startswith("http://") or url.startswith("https://")):
            raise ValueError(f"TTS worker {name} URL must start with http:// or https://")
        names.add(name)
        result.append(TTSWorkerSpec(name, url))
    return tuple(result)


def build_worker_tts_payload(text: str, text_lang: str) -> dict[str, str]:
    """Build the narrow Bot -> worker contract for one remote synthesis request."""
    return {
        "text": _validated_text(text),
        "text_lang": normalize_tts_language(text_lang),
    }


class RemoteTTSWorker(_NormalizedSpeechSynthesizer):
    def __init__(
        self,
        output_directory: Path,
        spec: TTSWorkerSpec,
        *,
        token: str | None,
        health_timeout_seconds: float,
        request_timeout_seconds: float,
        health_cache_seconds: float = 3.0,
        http_queue_grace_seconds: float = DEFAULT_HTTP_QUEUE_GRACE_SECONDS,
    ) -> None:
        super().__init__(output_directory)
        self.spec = spec
        self.token = token
        self.health_timeout_seconds = max(0.5, float(health_timeout_seconds))
        self.request_timeout_seconds = max(5.0, float(request_timeout_seconds))
        self.health_cache_seconds = max(0.0, float(health_cache_seconds))
        self.http_queue_grace_seconds = max(0.0, float(http_queue_grace_seconds))
        self._session: aiohttp.ClientSession | None = None
        self._last_health: WorkerHealth | None = None
        self._last_health_at = 0.0

    async def _client(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _headers(self) -> dict[str, str]:
        if not self.token:
            return {}
        return {"Authorization": f"Bearer {self.token}"}

    async def health(self, *, force: bool = False) -> WorkerHealth:
        now = time.monotonic()
        if (
            not force
            and self._last_health is not None
            and now - self._last_health_at <= self.health_cache_seconds
        ):
            return self._last_health
        try:
            session = await self._client()
            timeout = aiohttp.ClientTimeout(total=self.health_timeout_seconds)
            async with session.get(
                f"{self.spec.base_url}/health",
                headers=self._headers(),
                timeout=timeout,
            ) as response:
                if response.status != 200:
                    health = WorkerHealth(self.spec.name, "unavailable", f"HTTP {response.status}")
                else:
                    payload = await response.json(content_type=None)
                    status = str(payload.get("status", "unavailable")).lower()
                    if status not in {"ready", "busy", "unavailable"}:
                        status = "unavailable"
                    health = WorkerHealth(
                        self.spec.name,
                        status,
                        str(payload.get("reason", ""))[:200],
                        str(payload.get("backend", ""))[:80],
                        _optional_float(payload.get("gpu_utilization")),
                        _optional_int(payload.get("gpu_memory_free_mb")),
                        _optional_int(payload.get("active_requests")),
                        _optional_int(payload.get("queued_requests")),
                        _optional_int(payload.get("max_queue_size")),
                    )
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, TypeError) as error:
            health = WorkerHealth(self.spec.name, "unavailable", str(error)[:200])
        self._last_health = health
        self._last_health_at = now
        return health

    async def synthesize(self, text: str, *, text_lang: str | None = None) -> Path:
        value = _validated_text(text)
        language = normalize_tts_language(text_lang, fallback="zh")
        raw_path, output_path = self._paths(f"worker-{self.spec.name}")
        try:
            session = await self._client()
            timeout = aiohttp.ClientTimeout(
                total=self.request_timeout_seconds + self.http_queue_grace_seconds
            )
            async with session.post(
                f"{self.spec.base_url}/v1/tts",
                json=build_worker_tts_payload(value, language),
                headers=self._headers(),
                timeout=timeout,
            ) as response:
                if response.status != 200:
                    detail = (await response.text())[:500]
                    if response.status in {429, 503} and _looks_like_worker_busy(detail):
                        raise TTSWorkerBusyError(
                            f"TTS worker {self.spec.name} is busy: {detail}"
                        )
                    raise RuntimeError(
                        f"TTS worker {self.spec.name} returned HTTP {response.status}: {detail}"
                    )
                length = response.content_length
                if length is not None and length > MAX_REMOTE_AUDIO_BYTES:
                    raise RuntimeError(
                        f"TTS worker {self.spec.name} returned an oversized audio response"
                    )
                audio = await response.read()
                if not audio or len(audio) > MAX_REMOTE_AUDIO_BYTES:
                    raise RuntimeError(f"TTS worker {self.spec.name} returned invalid audio")
            raw_path.write_bytes(audio)
            LOGGER.info("Remote TTS worker selected: %s (language=%s)", self.spec.name, language)
            return await self._normalize(raw_path, output_path)
        except Exception:
            raw_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
            raise

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()


class WorkerPoolSpeechSynthesizer:
    """Prefer Moxue GPT-SoVITS workers, waiting briefly for busy nodes before provider fallback."""

    def __init__(
        self,
        output_directory: Path,
        specs: Iterable[TTSWorkerSpec],
        *,
        token: str | None,
        fallback: SpeechSynthesizer | None,
        default_mode: str = "auto",
        health_timeout_seconds: float = 2.0,
        request_timeout_seconds: float = 45.0,
        failure_cooldown_seconds: float = 15.0,
        busy_wait_seconds: float = DEFAULT_BUSY_WAIT_SECONDS,
        busy_poll_seconds: float = DEFAULT_BUSY_POLL_SECONDS,
    ) -> None:
        self.workers = tuple(
            RemoteTTSWorker(
                output_directory,
                spec,
                token=token,
                health_timeout_seconds=health_timeout_seconds,
                request_timeout_seconds=request_timeout_seconds,
            )
            for spec in specs
        )
        self.fallback = fallback
        self.failure_cooldown_seconds = max(0.0, float(failure_cooldown_seconds))
        self.busy_wait_seconds = max(0.0, float(busy_wait_seconds))
        self.busy_poll_seconds = max(0.05, float(busy_poll_seconds))
        self._blocked_until: dict[str, float] = {}
        self._mode = "auto"
        self.set_mode(default_mode)

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def available_modes(self) -> tuple[str, ...]:
        return ("auto", *(worker.spec.name for worker in self.workers))

    @property
    def providers(self) -> tuple[str, ...]:
        names = tuple(f"worker:{worker.spec.name}" for worker in self.workers)
        fallback_providers = (
            tuple(getattr(self.fallback, "providers", ())) if self.fallback is not None else ()
        )
        return (*names, *fallback_providers)

    def set_mode(self, mode: str) -> None:
        value = mode.strip().lower() or "auto"
        worker_names = {worker.spec.name for worker in self.workers}
        if value != "auto" and value not in worker_names:
            available = ", ".join(("auto", *sorted(worker_names)))
            raise ValueError(f"Unknown TTS worker mode '{value}'. Available: {available}")
        self._mode = value

    async def status(self) -> tuple[WorkerHealth, ...]:
        if not self.workers:
            return ()
        return tuple(await asyncio.gather(*(worker.health(force=True) for worker in self.workers)))

    def _ordered_workers(self) -> tuple[RemoteTTSWorker, ...]:
        if self._mode == "auto":
            return self.workers
        return tuple(worker for worker in self.workers if worker.spec.name == self._mode)

    async def _health_after_bounded_wait(self, worker: RemoteTTSWorker) -> WorkerHealth:
        health = await worker.health()
        if health.status != "busy" or self.busy_wait_seconds <= 0:
            return health

        deadline = time.monotonic() + self.busy_wait_seconds
        LOGGER.info(
            "TTS worker %s is busy; waiting up to %.1fs before provider fallback",
            worker.spec.name,
            self.busy_wait_seconds,
        )
        while health.status == "busy":
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            await asyncio.sleep(min(self.busy_poll_seconds, remaining))
            health = await worker.health(force=True)
        return health

    async def synthesize(self, text: str) -> Path:
        value = _validated_text(text)
        language = detect_tts_language(value)
        errors: list[str] = []
        for worker in self._ordered_workers():
            now = time.monotonic()
            blocked_until = self._blocked_until.get(worker.spec.name, 0.0)
            if blocked_until > now:
                errors.append(f"{worker.spec.name}: cooldown")
                continue

            health = await self._health_after_bounded_wait(worker)
            if not health.ready:
                errors.append(
                    f"{worker.spec.name}: {health.status}"
                    f"{' (' + health.reason + ')' if health.reason else ''}"
                )
                continue
            try:
                return await worker.synthesize(value, text_lang=language)
            except TTSWorkerBusyError as error:
                errors.append(f"{worker.spec.name}: {error}")
                LOGGER.info("TTS worker %s queue wait expired: %s", worker.spec.name, error)
            except Exception as error:
                self._blocked_until[worker.spec.name] = (
                    time.monotonic() + self.failure_cooldown_seconds
                )
                errors.append(f"{worker.spec.name}: {error}")
                LOGGER.warning(
                    "TTS worker %s failed for language %s; trying next route: %s",
                    worker.spec.name,
                    language,
                    error,
                )

        if self.fallback is not None:
            LOGGER.info(
                "No remote TTS worker selected; using local/provider fallback (%s)",
                "; ".join(errors) or "none configured",
            )
            return await self.fallback.synthesize(value)
        raise RuntimeError(
            "No TTS route available: " + (" | ".join(errors) if errors else "no workers configured")
        )

    async def close(self) -> None:
        await asyncio.gather(*(worker.close() for worker in self.workers), return_exceptions=True)


def _looks_like_worker_busy(detail: str) -> bool:
    value = detail.lower()
    return any(
        marker in value
        for marker in (
            '"status": "busy"',
            '"status":"busy"',
            '"status": "queue_full"',
            '"status":"queue_full"',
            '"status": "queue_timeout"',
            '"status":"queue_timeout"',
            "tts queue is full",
            "queue wait timed out",
        )
    )


def _optional_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _optional_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
