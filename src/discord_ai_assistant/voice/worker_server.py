from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import aiohttp
from aiohttp import web
from dotenv import load_dotenv

from discord_ai_assistant.voice.language_routing import normalize_tts_language

LOGGER = logging.getLogger(__name__)
MAX_TEXT_CHARACTERS = 400
MAX_UPSTREAM_AUDIO_BYTES = 32 * 1024 * 1024
UPSTREAM_HEALTH_TIMEOUT_SECONDS = 2.0
UPSTREAM_HEALTH_CACHE_SECONDS = 2.0


def load_worker_dotenv(project_root: Path | None = None) -> Path:
    """Load worker settings from the repository .env without overriding explicit env vars."""
    root = project_root or Path(__file__).resolve().parents[3]
    env_path = root / ".env"
    load_dotenv(env_path, override=False)
    return env_path


@dataclass(frozen=True, slots=True)
class WorkerServerSettings:
    name: str
    host: str
    port: int
    token: str | None
    upstream_url: str
    ref_audio_path: Path
    prompt_text: str
    prompt_lang: str
    text_lang: str
    speed_factor: float
    upstream_timeout_seconds: float
    max_concurrency: int
    max_queue_size: int
    queue_wait_seconds: float
    gpu_busy_threshold: float
    gpu_min_free_mb: int
    gpu_probe_enabled: bool


def load_worker_settings() -> WorkerServerSettings:
    name = os.getenv("MOXUE_TTS_WORKER_NAME", "worker").strip().lower() or "worker"
    host = os.getenv("MOXUE_TTS_WORKER_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.getenv("MOXUE_TTS_WORKER_PORT", "8891"))
    if not 1 <= port <= 65535:
        raise RuntimeError("MOXUE_TTS_WORKER_PORT must be between 1 and 65535")
    token = os.getenv("MOXUE_TTS_WORKER_TOKEN", "").strip() or None
    if host not in {"127.0.0.1", "localhost", "::1"} and not token:
        raise RuntimeError("MOXUE_TTS_WORKER_TOKEN is required when the worker listens beyond loopback")

    upstream_url = os.getenv("MOXUE_TTS_UPSTREAM_URL", "http://127.0.0.1:9880").strip().rstrip("/")
    if not (upstream_url.startswith("http://") or upstream_url.startswith("https://")):
        raise RuntimeError("MOXUE_TTS_UPSTREAM_URL must start with http:// or https://")

    ref_value = os.getenv("MOXUE_TTS_REF_AUDIO_PATH", "").strip()
    if not ref_value:
        raise RuntimeError("MOXUE_TTS_REF_AUDIO_PATH is required")
    ref_audio_path = Path(ref_value).expanduser().resolve()
    if not ref_audio_path.is_file():
        raise RuntimeError(f"MOXUE_TTS_REF_AUDIO_PATH does not exist: {ref_audio_path}")

    speed_factor = float(os.getenv("MOXUE_TTS_SPEED_FACTOR", "1.0"))
    if not 0.5 <= speed_factor <= 2.0:
        raise RuntimeError("MOXUE_TTS_SPEED_FACTOR must be between 0.5 and 2.0")
    upstream_timeout_seconds = float(os.getenv("MOXUE_TTS_UPSTREAM_TIMEOUT_SECONDS", "45"))
    if not 5 <= upstream_timeout_seconds <= 180:
        raise RuntimeError("MOXUE_TTS_UPSTREAM_TIMEOUT_SECONDS must be between 5 and 180")
    max_concurrency = int(os.getenv("MOXUE_TTS_MAX_CONCURRENCY", "1"))
    if not 1 <= max_concurrency <= 8:
        raise RuntimeError("MOXUE_TTS_MAX_CONCURRENCY must be between 1 and 8")
    max_queue_size = int(os.getenv("MOXUE_TTS_MAX_QUEUE_SIZE", "8"))
    if not 0 <= max_queue_size <= 64:
        raise RuntimeError("MOXUE_TTS_MAX_QUEUE_SIZE must be between 0 and 64")
    queue_wait_seconds = float(os.getenv("MOXUE_TTS_QUEUE_WAIT_SECONDS", "15"))
    if not 1 <= queue_wait_seconds <= 120:
        raise RuntimeError("MOXUE_TTS_QUEUE_WAIT_SECONDS must be between 1 and 120")

    # Kept for backward-compatible configuration and diagnostics. High GPU utilization
    # no longer rejects TTS work because that is the normal state while GPT-SoVITS is
    # synthesizing. Only a genuinely low-memory *idle* worker is held back.
    gpu_busy_threshold = float(os.getenv("MOXUE_TTS_GPU_BUSY_THRESHOLD", "80"))
    if not 1 <= gpu_busy_threshold <= 100:
        raise RuntimeError("MOXUE_TTS_GPU_BUSY_THRESHOLD must be between 1 and 100")
    gpu_min_free_mb = int(os.getenv("MOXUE_TTS_GPU_MIN_FREE_MB", "1800"))
    if gpu_min_free_mb < 0:
        raise RuntimeError("MOXUE_TTS_GPU_MIN_FREE_MB must be non-negative")

    try:
        prompt_lang = normalize_tts_language(
            os.getenv("MOXUE_TTS_PROMPT_LANG", "zh"), fallback="zh"
        )
        text_lang = normalize_tts_language(
            os.getenv("MOXUE_TTS_TEXT_LANG", "zh"), fallback="zh"
        )
    except ValueError as error:
        raise RuntimeError(str(error)) from error

    return WorkerServerSettings(
        name=name,
        host=host,
        port=port,
        token=token,
        upstream_url=upstream_url,
        ref_audio_path=ref_audio_path,
        prompt_text=os.getenv("MOXUE_TTS_PROMPT_TEXT", "").strip(),
        prompt_lang=prompt_lang,
        text_lang=text_lang,
        speed_factor=speed_factor,
        upstream_timeout_seconds=upstream_timeout_seconds,
        max_concurrency=max_concurrency,
        max_queue_size=max_queue_size,
        queue_wait_seconds=queue_wait_seconds,
        gpu_busy_threshold=gpu_busy_threshold,
        gpu_min_free_mb=gpu_min_free_mb,
        gpu_probe_enabled=os.getenv("MOXUE_TTS_GPU_PROBE_ENABLED", "true").strip().lower()
        in {"1", "true", "yes", "on"},
    )


@dataclass(frozen=True, slots=True)
class GPUMetrics:
    utilization: float
    memory_used_mb: int
    memory_total_mb: int

    @property
    def memory_free_mb(self) -> int:
        return max(0, self.memory_total_mb - self.memory_used_mb)


class GPUProbe:
    def __init__(self, enabled: bool, cache_seconds: float = 2.0) -> None:
        self.enabled = enabled
        self.cache_seconds = max(0.0, cache_seconds)
        self._last_at = 0.0
        self._last_value: GPUMetrics | None = None

    async def get(self, *, force: bool = False) -> GPUMetrics | None:
        if not self.enabled or shutil.which("nvidia-smi") is None:
            return None
        now = time.monotonic()
        if not force and self._last_value is not None and now - self._last_at <= self.cache_seconds:
            return self._last_value
        process = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await process.communicate()
        if process.returncode != 0:
            return None
        first_line = stdout.decode("utf-8", errors="replace").splitlines()[0:1]
        if not first_line:
            return None
        try:
            value = self.parse_nvidia_smi(first_line[0])
        except ValueError:
            LOGGER.warning("Could not parse nvidia-smi output: %r", first_line[0])
            return None
        self._last_value = value
        self._last_at = now
        return value

    @staticmethod
    def parse_nvidia_smi(line: str) -> GPUMetrics:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            raise ValueError("expected utilization, memory used, memory total")
        return GPUMetrics(float(parts[0]), int(float(parts[1])), int(float(parts[2])))


class GPTSoVITSUpstreamProbe:
    """Cheap cached reachability probe that never asks GPT-SoVITS to synthesize audio."""

    def __init__(
        self,
        upstream_url: str,
        *,
        timeout_seconds: float = UPSTREAM_HEALTH_TIMEOUT_SECONDS,
        cache_seconds: float = UPSTREAM_HEALTH_CACHE_SECONDS,
    ) -> None:
        self.upstream_url = upstream_url.rstrip("/")
        self.timeout_seconds = max(0.1, float(timeout_seconds))
        self.cache_seconds = max(0.0, cache_seconds)
        self._last_at = 0.0
        self._last_value: tuple[bool, str] | None = None

    async def get(
        self, session: aiohttp.ClientSession, *, force: bool = False
    ) -> tuple[bool, str]:
        now = time.monotonic()
        if not force and self._last_value is not None and now - self._last_at <= self.cache_seconds:
            return self._last_value

        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
            # Never probe bare GET /tts: GPT-SoVITS api_v2 executes the handler and can
            # throw on missing query parameters. OpenAPI is side-effect free.
            async with session.get(
                f"{self.upstream_url}/openapi.json", timeout=timeout
            ) as response:
                if response.status != 200:
                    result = (False, f"GPT-SoVITS OpenAPI HTTP {response.status}")
                else:
                    try:
                        payload = await response.json(content_type=None)
                    except (ValueError, TypeError) as error:
                        result = (
                            False,
                            f"GPT-SoVITS OpenAPI invalid JSON: {str(error)[:120]}",
                        )
                    else:
                        paths = payload.get("paths") if isinstance(payload, dict) else None
                        ready = isinstance(paths, dict) and "/tts" in paths
                        result = (
                            (True, "")
                            if ready
                            else (
                                False,
                                "GPT-SoVITS OpenAPI schema does not expose /tts",
                            )
                        )
        except (aiohttp.ClientError, asyncio.TimeoutError) as error:
            result = (
                False,
                f"GPT-SoVITS unreachable: {str(error)[:200] or error.__class__.__name__}",
            )

        self._last_at = time.monotonic()
        self._last_value = result
        return result


class MoxueTTSWorkerServer:
    """Authenticated LAN proxy with bounded FIFO-style admission before GPT-SoVITS."""

    def __init__(self, settings: WorkerServerSettings) -> None:
        self.settings = settings
        self.gpu = GPUProbe(settings.gpu_probe_enabled)
        self.upstream = GPTSoVITSUpstreamProbe(settings.upstream_url)
        self._active_requests = 0
        self._pending_requests = 0
        self._request_lock = asyncio.Lock()
        self._inference_slots = asyncio.Semaphore(settings.max_concurrency)
        self._session: aiohttp.ClientSession | None = None
        self.app = web.Application(client_max_size=128 * 1024)
        self.app.add_routes(
            [
                web.get("/health", self.handle_health),
                web.post("/v1/tts", self.handle_tts),
            ]
        )
        self.app.on_cleanup.append(self._cleanup)

    @property
    def queued_requests(self) -> int:
        return max(0, self._pending_requests - self._active_requests)

    @property
    def request_capacity(self) -> int:
        return self.settings.max_concurrency + self.settings.max_queue_size

    def _authorized(self, request: web.Request) -> bool:
        if not self.settings.token:
            return True
        supplied = request.headers.get("Authorization", "")
        expected = f"Bearer {self.settings.token}"
        return secrets.compare_digest(supplied, expected)

    async def _client(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _cleanup(self, _: web.Application) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def _availability(self) -> tuple[str, str, GPUMetrics | None]:
        metrics = await self.gpu.get()

        # Do not treat high utilization as a failure: a healthy TTS worker is expected
        # to run near 100% while generating the request ahead of us in the queue.
        if (
            metrics is not None
            and self._active_requests == 0
            and self._pending_requests == 0
            and metrics.memory_free_mb < self.settings.gpu_min_free_mb
        ):
            return "busy", f"GPU free memory {metrics.memory_free_mb} MB", metrics

        upstream_ready, upstream_reason = await self.upstream.get(await self._client())
        if not upstream_ready:
            return "unavailable", upstream_reason, metrics

        if self._pending_requests >= self.request_capacity:
            return "busy", "TTS queue is full", metrics

        return "ready", "", metrics

    async def _reserve_request(self) -> bool:
        async with self._request_lock:
            if self._pending_requests >= self.request_capacity:
                return False
            self._pending_requests += 1
            return True

    async def _mark_active(self) -> None:
        async with self._request_lock:
            self._active_requests += 1

    async def _release_request(self, *, active: bool) -> None:
        if active:
            self._inference_slots.release()
        async with self._request_lock:
            if active:
                self._active_requests = max(0, self._active_requests - 1)
            self._pending_requests = max(0, self._pending_requests - 1)

    async def handle_health(self, request: web.Request) -> web.Response:
        if not self._authorized(request):
            raise web.HTTPUnauthorized()
        status, reason, metrics = await self._availability()
        payload: dict[str, object] = {
            "name": self.settings.name,
            "status": status,
            "reason": reason,
            "backend": "gpt-sovits",
            "active_requests": self._active_requests,
            "queued_requests": self.queued_requests,
            "max_concurrency": self.settings.max_concurrency,
            "max_queue_size": self.settings.max_queue_size,
            "queue_wait_seconds": self.settings.queue_wait_seconds,
            "default_text_lang": self.settings.text_lang,
            "prompt_lang": self.settings.prompt_lang,
        }
        if metrics is not None:
            payload.update(
                {
                    "gpu_utilization": metrics.utilization,
                    "gpu_memory_used_mb": metrics.memory_used_mb,
                    "gpu_memory_total_mb": metrics.memory_total_mb,
                    "gpu_memory_free_mb": metrics.memory_free_mb,
                }
            )
        return web.json_response(payload)

    async def handle_tts(self, request: web.Request) -> web.Response:
        if not self._authorized(request):
            raise web.HTTPUnauthorized()
        try:
            payload = await request.json()
        except (json.JSONDecodeError, ValueError, TypeError):
            raise web.HTTPBadRequest(text="invalid JSON")
        text = str(payload.get("text", "")).strip()
        if not text or len(text) > MAX_TEXT_CHARACTERS:
            raise web.HTTPBadRequest(text=f"text must contain 1-{MAX_TEXT_CHARACTERS} characters")
        try:
            text_lang = normalize_tts_language(
                str(payload.get("text_lang", "")).strip() or None,
                fallback=self.settings.text_lang,
            )
        except ValueError as error:
            raise web.HTTPBadRequest(text=str(error)) from error

        status, reason, _ = await self._availability()
        if status == "unavailable":
            return web.json_response(
                {"status": "unavailable", "reason": reason}, status=503
            )
        if status == "busy" and self._pending_requests >= self.request_capacity:
            return web.json_response(
                {"status": "queue_full", "reason": reason}, status=503
            )
        if status == "busy":
            return web.json_response({"status": "busy", "reason": reason}, status=503)

        if not await self._reserve_request():
            return web.json_response(
                {"status": "queue_full", "reason": "TTS queue is full"}, status=503
            )

        acquired = False
        try:
            try:
                await asyncio.wait_for(
                    self._inference_slots.acquire(),
                    timeout=self.settings.queue_wait_seconds,
                )
            except asyncio.TimeoutError:
                LOGGER.info(
                    "TTS queue wait timed out on worker %s after %.1fs",
                    self.settings.name,
                    self.settings.queue_wait_seconds,
                )
                return web.json_response(
                    {
                        "status": "queue_timeout",
                        "reason": "TTS queue wait timed out",
                    },
                    status=503,
                )

            acquired = True
            await self._mark_active()
            LOGGER.info(
                "TTS request admitted on worker %s (active=%s queued=%s language=%s)",
                self.settings.name,
                self._active_requests,
                self.queued_requests,
                text_lang,
            )

            upstream_payload = {
                "text": text,
                "text_lang": text_lang,
                "ref_audio_path": str(self.settings.ref_audio_path),
                "prompt_text": self.settings.prompt_text,
                "prompt_lang": self.settings.prompt_lang,
                "media_type": "wav",
                "streaming_mode": False,
                "speed_factor": self.settings.speed_factor,
                "text_split_method": "cut5",
                "parallel_infer": True,
            }
            session = await self._client()
            timeout = aiohttp.ClientTimeout(total=self.settings.upstream_timeout_seconds)
            async with session.post(
                f"{self.settings.upstream_url}/tts",
                json=upstream_payload,
                timeout=timeout,
            ) as upstream:
                if upstream.status != 200:
                    detail = (await upstream.text())[:1000]
                    LOGGER.warning(
                        "GPT-SoVITS upstream failed on worker %s for language %s: HTTP %s %s",
                        self.settings.name,
                        text_lang,
                        upstream.status,
                        detail,
                    )
                    return web.json_response(
                        {
                            "status": "upstream_error",
                            "reason": f"HTTP {upstream.status}",
                            "detail": detail,
                        },
                        status=502,
                    )
                length = upstream.content_length
                if length is not None and length > MAX_UPSTREAM_AUDIO_BYTES:
                    return web.json_response(
                        {"status": "upstream_error", "reason": "audio too large"},
                        status=502,
                    )
                audio = await upstream.read()
                if not audio or len(audio) > MAX_UPSTREAM_AUDIO_BYTES:
                    return web.json_response(
                        {"status": "upstream_error", "reason": "invalid audio"},
                        status=502,
                    )
            LOGGER.info(
                "TTS request completed on worker %s (language=%s)",
                self.settings.name,
                text_lang,
            )
            return web.Response(body=audio, content_type="audio/wav")
        except (aiohttp.ClientError, asyncio.TimeoutError) as error:
            LOGGER.warning(
                "TTS upstream unavailable on worker %s: %s",
                self.settings.name,
                error,
            )
            return web.json_response(
                {
                    "status": "upstream_unavailable",
                    "reason": str(error)[:300],
                },
                status=502,
            )
        finally:
            await self._release_request(active=acquired)


def main() -> None:
    load_worker_dotenv()
    settings = load_worker_settings()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    LOGGER.info(
        "Starting Moxue TTS worker %s on %s:%s -> %s "
        "(concurrency=%s queue=%s wait=%.1fs)",
        settings.name,
        settings.host,
        settings.port,
        settings.upstream_url,
        settings.max_concurrency,
        settings.max_queue_size,
        settings.queue_wait_seconds,
    )
    web.run_app(
        MoxueTTSWorkerServer(settings).app,
        host=settings.host,
        port=settings.port,
        print=None,
    )


if __name__ == "__main__":
    main()
