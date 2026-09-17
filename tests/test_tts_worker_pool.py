from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from discord_ai_assistant.voice.worker_pool import (
    TTSWorkerBusyError,
    TTSWorkerSpec,
    WorkerHealth,
    WorkerPoolSpeechSynthesizer,
    build_worker_tts_payload,
    parse_worker_specs,
)


class _FakeWorker:
    def __init__(
        self,
        name: str,
        status: str,
        output: Path,
        *,
        status_sequence: list[str] | None = None,
    ) -> None:
        self.spec = TTSWorkerSpec(name, f"http://{name}")
        self._status = status
        self._status_sequence = list(status_sequence or [])
        self.output = output
        self.calls = 0
        self.health_calls = 0
        self.languages: list[str | None] = []

    async def health(self, *, force: bool = False) -> WorkerHealth:
        self.health_calls += 1
        if self._status_sequence:
            status = self._status_sequence.pop(0)
        else:
            status = self._status
        return WorkerHealth(self.spec.name, status)

    async def synthesize(self, text: str, *, text_lang: str | None = None) -> Path:
        self.calls += 1
        self.languages.append(text_lang)
        return self.output

    async def close(self) -> None:
        return None


class _FakeFallback:
    providers = ("fallback",)

    def __init__(self, output: Path) -> None:
        self.output = output
        self.calls = 0

    async def synthesize(self, text: str) -> Path:
        self.calls += 1
        return self.output


class TTSWorkerPoolTests(unittest.TestCase):
    def test_parse_worker_specs_preserves_priority(self) -> None:
        specs = parse_worker_specs(
            "desktop=http://192.168.1.20:8891; laptop=http://192.168.1.21:8891;local=http://127.0.0.1:8891"
        )
        self.assertEqual([item.name for item in specs], ["desktop", "laptop", "local"])
        self.assertEqual(specs[0].base_url, "http://192.168.1.20:8891")

    def test_parse_worker_specs_rejects_duplicate_names(self) -> None:
        with self.assertRaises(ValueError):
            parse_worker_specs("desktop=http://one;desktop=http://two")

    def test_worker_request_payload_contains_explicit_language(self) -> None:
        self.assertEqual(
            build_worker_tts_payload("你好", "yue"),
            {"text": "你好", "text_lang": "yue"},
        )
        with self.assertRaises(ValueError):
            build_worker_tts_payload("bonjour", "fr")

    def test_auto_skips_persistently_busy_worker_when_wait_disabled(self) -> None:
        async def run() -> tuple[Path, int, int, list[str | None]]:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                router = WorkerPoolSpeechSynthesizer(
                    root,
                    (),
                    token=None,
                    fallback=None,
                    busy_wait_seconds=0,
                )
                desktop = _FakeWorker("desktop", "busy", root / "desktop.wav")
                laptop = _FakeWorker("laptop", "ready", root / "laptop.wav")
                router.workers = (desktop, laptop)  # type: ignore[assignment]
                result = await router.synthesize("我唔知佢喺邊度")
                return result, desktop.calls, laptop.calls, laptop.languages

        result, desktop_calls, laptop_calls, languages = asyncio.run(run())
        self.assertEqual(result.name, "laptop.wav")
        self.assertEqual(desktop_calls, 0)
        self.assertEqual(laptop_calls, 1)
        self.assertEqual(languages, ["yue"])

    def test_busy_primary_is_waited_for_before_fallback(self) -> None:
        async def run() -> tuple[Path, int, int, int]:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fallback = _FakeFallback(root / "fallback.wav")
                router = WorkerPoolSpeechSynthesizer(
                    root,
                    (),
                    token=None,
                    fallback=fallback,
                    busy_wait_seconds=0.05,
                    busy_poll_seconds=0.01,
                )
                desktop = _FakeWorker(
                    "desktop",
                    "ready",
                    root / "desktop.wav",
                    status_sequence=["busy", "ready"],
                )
                router.workers = (desktop,)  # type: ignore[assignment]
                result = await router.synthesize("測試等待")
                return result, desktop.calls, desktop.health_calls, fallback.calls

        result, worker_calls, health_calls, fallback_calls = asyncio.run(run())
        self.assertEqual(result.name, "desktop.wav")
        self.assertEqual(worker_calls, 1)
        self.assertGreaterEqual(health_calls, 2)
        self.assertEqual(fallback_calls, 0)

    def test_forced_mode_uses_only_selected_worker_then_fallback(self) -> None:
        async def run() -> tuple[Path, int, int, int]:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fallback = _FakeFallback(root / "fallback.wav")
                router = WorkerPoolSpeechSynthesizer(
                    root,
                    (),
                    token=None,
                    fallback=fallback,
                    busy_wait_seconds=0,
                )
                desktop = _FakeWorker("desktop", "ready", root / "desktop.wav")
                laptop = _FakeWorker("laptop", "busy", root / "laptop.wav")
                router.workers = (desktop, laptop)  # type: ignore[assignment]
                router.set_mode("laptop")
                result = await router.synthesize("測試")
                return result, desktop.calls, laptop.calls, fallback.calls

        result, desktop_calls, laptop_calls, fallback_calls = asyncio.run(run())
        self.assertEqual(result.name, "fallback.wav")
        self.assertEqual(desktop_calls, 0)
        self.assertEqual(laptop_calls, 0)
        self.assertEqual(fallback_calls, 1)

    def test_worker_failure_preserves_legacy_fallback(self) -> None:
        class _FailingWorker(_FakeWorker):
            async def synthesize(self, text: str, *, text_lang: str | None = None) -> Path:
                self.calls += 1
                self.languages.append(text_lang)
                raise RuntimeError("worker failed")

        async def run() -> tuple[Path, int, list[str | None]]:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fallback = _FakeFallback(root / "fallback.wav")
                router = WorkerPoolSpeechSynthesizer(root, (), token=None, fallback=fallback)
                worker = _FailingWorker("desktop", "ready", root / "desktop.wav")
                router.workers = (worker,)  # type: ignore[assignment]
                result = await router.synthesize("今日はテストです")
                return result, fallback.calls, worker.languages

        result, fallback_calls, languages = asyncio.run(run())
        self.assertEqual(result.name, "fallback.wav")
        self.assertEqual(fallback_calls, 1)
        self.assertEqual(languages, ["ja"])

    def test_queue_timeout_does_not_put_worker_into_failure_cooldown(self) -> None:
        class _BusyWorker(_FakeWorker):
            async def synthesize(self, text: str, *, text_lang: str | None = None) -> Path:
                self.calls += 1
                raise TTSWorkerBusyError("queue wait timed out")

        async def run() -> tuple[int, int]:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fallback = _FakeFallback(root / "fallback.wav")
                router = WorkerPoolSpeechSynthesizer(
                    root,
                    (),
                    token=None,
                    fallback=fallback,
                    busy_wait_seconds=0,
                )
                worker = _BusyWorker("desktop", "ready", root / "desktop.wav")
                router.workers = (worker,)  # type: ignore[assignment]
                await router.synthesize("第一次")
                await router.synthesize("第二次")
                return worker.calls, fallback.calls

        worker_calls, fallback_calls = asyncio.run(run())
        self.assertEqual(worker_calls, 2)
        self.assertEqual(fallback_calls, 2)


if __name__ == "__main__":
    unittest.main()
