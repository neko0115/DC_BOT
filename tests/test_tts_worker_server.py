from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from discord_ai_assistant.voice.worker_server import (
    GPUMetrics,
    GPUProbe,
    MoxueTTSWorkerServer,
    WorkerServerSettings,
    load_worker_settings,
)


class TTSWorkerServerTests(unittest.TestCase):
    def test_parse_nvidia_smi_metrics(self) -> None:
        metrics = GPUProbe.parse_nvidia_smi("72, 4210, 8192")
        self.assertEqual(metrics.utilization, 72)
        self.assertEqual(metrics.memory_used_mb, 4210)
        self.assertEqual(metrics.memory_total_mb, 8192)
        self.assertEqual(metrics.memory_free_mb, 3982)

    def test_remote_bind_requires_token(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "reference.wav"
            reference.write_bytes(b"RIFF")
            env = {
                "MOXUE_TTS_WORKER_HOST": "0.0.0.0",
                "MOXUE_TTS_REF_AUDIO_PATH": str(reference),
                "MOXUE_TTS_WORKER_TOKEN": "",
            }
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaises(RuntimeError):
                    load_worker_settings()

    def test_worker_settings_accept_authenticated_lan_bind_and_queue(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "reference.wav"
            reference.write_bytes(b"RIFF")
            env = {
                "MOXUE_TTS_WORKER_NAME": "desktop",
                "MOXUE_TTS_WORKER_HOST": "0.0.0.0",
                "MOXUE_TTS_WORKER_PORT": "8891",
                "MOXUE_TTS_WORKER_TOKEN": "secret",
                "MOXUE_TTS_REF_AUDIO_PATH": str(reference),
                "MOXUE_TTS_MAX_QUEUE_SIZE": "8",
                "MOXUE_TTS_QUEUE_WAIT_SECONDS": "15",
                "MOXUE_TTS_GPU_BUSY_THRESHOLD": "80",
                "MOXUE_TTS_GPU_MIN_FREE_MB": "1800",
            }
            with patch.dict(os.environ, env, clear=True):
                settings = load_worker_settings()
        self.assertEqual(settings.name, "desktop")
        self.assertEqual(settings.host, "0.0.0.0")
        self.assertEqual(settings.port, 8891)
        self.assertEqual(settings.token, "secret")
        self.assertEqual(settings.max_queue_size, 8)
        self.assertEqual(settings.queue_wait_seconds, 15)
        self.assertEqual(settings.gpu_busy_threshold, 80)
        self.assertEqual(settings.gpu_min_free_mb, 1800)

    def test_worker_keeps_prompt_language_separate_from_default_target_language(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "reference.wav"
            reference.write_bytes(b"RIFF")
            env = {
                "MOXUE_TTS_REF_AUDIO_PATH": str(reference),
                "MOXUE_TTS_PROMPT_LANG": "zh",
                "MOXUE_TTS_TEXT_LANG": "yue",
            }
            with patch.dict(os.environ, env, clear=True):
                settings = load_worker_settings()
        self.assertEqual(settings.prompt_lang, "zh")
        self.assertEqual(settings.text_lang, "yue")

    def test_worker_rejects_unsupported_configured_language(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "reference.wav"
            reference.write_bytes(b"RIFF")
            env = {
                "MOXUE_TTS_REF_AUDIO_PATH": str(reference),
                "MOXUE_TTS_TEXT_LANG": "fr",
            }
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaises(RuntimeError):
                    load_worker_settings()


class TTSWorkerAvailabilityTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def settings(
        *,
        max_concurrency: int = 1,
        max_queue_size: int = 8,
    ) -> WorkerServerSettings:
        return WorkerServerSettings(
            name="desktop",
            host="127.0.0.1",
            port=8891,
            token=None,
            upstream_url="http://127.0.0.1:9880",
            ref_audio_path=Path("reference.wav"),
            prompt_text="reference",
            prompt_lang="zh",
            text_lang="zh",
            speed_factor=1.0,
            upstream_timeout_seconds=45,
            max_concurrency=max_concurrency,
            max_queue_size=max_queue_size,
            queue_wait_seconds=15,
            gpu_busy_threshold=80,
            gpu_min_free_mb=1800,
            gpu_probe_enabled=False,
        )

    async def test_ready_requires_reachable_upstream(self) -> None:
        server = MoxueTTSWorkerServer(self.settings())
        server.gpu.get = AsyncMock(return_value=None)
        server._client = AsyncMock(return_value=object())
        server.upstream.get = AsyncMock(return_value=(True, ""))

        status, reason, metrics = await server._availability()

        self.assertEqual(status, "ready")
        self.assertEqual(reason, "")
        self.assertIsNone(metrics)
        server.upstream.get.assert_awaited_once()

    async def test_unreachable_upstream_marks_worker_unavailable(self) -> None:
        server = MoxueTTSWorkerServer(self.settings())
        server.gpu.get = AsyncMock(return_value=None)
        server._client = AsyncMock(return_value=object())
        server.upstream.get = AsyncMock(return_value=(False, "GPT-SoVITS unreachable"))

        status, reason, metrics = await server._availability()

        self.assertEqual(status, "unavailable")
        self.assertIn("GPT-SoVITS", reason)
        self.assertIsNone(metrics)

    async def test_high_gpu_utilization_no_longer_rejects_queueable_tts(self) -> None:
        server = MoxueTTSWorkerServer(self.settings())
        server.gpu.get = AsyncMock(return_value=GPUMetrics(95, 4000, 24000))
        server._client = AsyncMock(return_value=object())
        server.upstream.get = AsyncMock(return_value=(True, ""))

        status, reason, metrics = await server._availability()

        self.assertEqual(status, "ready")
        self.assertEqual(reason, "")
        self.assertIsNotNone(metrics)
        server.upstream.get.assert_awaited_once()

    async def test_idle_worker_with_critically_low_free_memory_stays_busy(self) -> None:
        server = MoxueTTSWorkerServer(self.settings())
        server.gpu.get = AsyncMock(return_value=GPUMetrics(10, 23500, 24000))
        server.upstream.get = AsyncMock(return_value=(True, ""))

        status, reason, metrics = await server._availability()

        self.assertEqual(status, "busy")
        self.assertIn("GPU free memory", reason)
        self.assertIsNotNone(metrics)
        server.upstream.get.assert_not_awaited()

    async def test_queue_capacity_accepts_waiters_before_reporting_full(self) -> None:
        server = MoxueTTSWorkerServer(
            self.settings(max_concurrency=1, max_queue_size=2)
        )
        self.assertTrue(await server._reserve_request())
        self.assertTrue(await server._reserve_request())
        self.assertTrue(await server._reserve_request())
        self.assertFalse(await server._reserve_request())
        self.assertEqual(server.queued_requests, 3)

        await server._release_request(active=False)
        await server._release_request(active=False)
        await server._release_request(active=False)
        self.assertEqual(server._pending_requests, 0)

    async def test_health_reports_busy_only_when_bounded_queue_is_full(self) -> None:
        server = MoxueTTSWorkerServer(
            self.settings(max_concurrency=1, max_queue_size=1)
        )
        server.gpu.get = AsyncMock(return_value=None)
        server._client = AsyncMock(return_value=object())
        server.upstream.get = AsyncMock(return_value=(True, ""))
        server._pending_requests = server.request_capacity
        server._active_requests = 1

        status, reason, _ = await server._availability()

        self.assertEqual(status, "busy")
        self.assertEqual(reason, "TTS queue is full")


if __name__ == "__main__":
    unittest.main()
