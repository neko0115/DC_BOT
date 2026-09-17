from __future__ import annotations

import asyncio
import unittest

from discord_ai_assistant.voice.worker_server import GPTSoVITSUpstreamProbe


class _FakeResponse:
    def __init__(self, status: int, payload: object) -> None:
        self.status = status
        self._payload = payload

    async def json(self, *, content_type: object = None) -> object:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _ResponseContext:
    def __init__(self, response: _FakeResponse | None = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error

    async def __aenter__(self) -> _FakeResponse:
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


class _FakeSession:
    def __init__(self, context: _ResponseContext) -> None:
        self.context = context
        self.urls: list[str] = []

    def get(self, url: str, *, timeout: object = None) -> _ResponseContext:
        self.urls.append(url)
        return self.context


class GPTSoVITSUpstreamProbeTests(unittest.IsolatedAsyncioTestCase):
    async def test_openapi_schema_with_tts_route_is_ready(self) -> None:
        session = _FakeSession(_ResponseContext(_FakeResponse(200, {"paths": {"/tts": {"post": {}}}})))
        probe = GPTSoVITSUpstreamProbe("http://127.0.0.1:9880", cache_seconds=0)

        ready, reason = await probe.get(session)  # type: ignore[arg-type]

        self.assertTrue(ready)
        self.assertEqual(reason, "")
        self.assertEqual(session.urls, ["http://127.0.0.1:9880/openapi.json"])

    async def test_openapi_schema_without_tts_route_is_unavailable(self) -> None:
        session = _FakeSession(_ResponseContext(_FakeResponse(200, {"paths": {"/control": {"get": {}}}})))
        probe = GPTSoVITSUpstreamProbe("http://127.0.0.1:9880", cache_seconds=0)

        ready, reason = await probe.get(session)  # type: ignore[arg-type]

        self.assertFalse(ready)
        self.assertIn("does not expose /tts", reason)

    async def test_non_200_openapi_response_is_unavailable(self) -> None:
        session = _FakeSession(_ResponseContext(_FakeResponse(503, {})))
        probe = GPTSoVITSUpstreamProbe("http://127.0.0.1:9880", cache_seconds=0)

        ready, reason = await probe.get(session)  # type: ignore[arg-type]

        self.assertFalse(ready)
        self.assertIn("HTTP 503", reason)

    async def test_timeout_is_unavailable_without_calling_tts_handler(self) -> None:
        session = _FakeSession(_ResponseContext(error=asyncio.TimeoutError()))
        probe = GPTSoVITSUpstreamProbe("http://127.0.0.1:9880", cache_seconds=0)

        ready, reason = await probe.get(session)  # type: ignore[arg-type]

        self.assertFalse(ready)
        self.assertIn("unreachable", reason)
        self.assertEqual(session.urls, ["http://127.0.0.1:9880/openapi.json"])


if __name__ == "__main__":
    unittest.main()
