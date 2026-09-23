from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Mapping
from typing import Any

import aiohttp

LOGGER = logging.getLogger(__name__)


class HeartbeatClient:
    def __init__(
        self,
        *,
        url: str,
        token: str,
        service_id: str,
        version: str,
        interval_seconds: float,
        timeout_seconds: float,
        payload_factory: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        self.url = url
        self.token = token
        self.service_id = service_id
        self.version = version
        self.interval_seconds = interval_seconds
        self.timeout_seconds = timeout_seconds
        self.payload_factory = payload_factory or (lambda: {})

        self._task: asyncio.Task[None] | None = None
        self._started_monotonic = time.monotonic()

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        if self.running:
            return
        self._started_monotonic = time.monotonic()
        self._task = asyncio.create_task(self._run(), name=f"heartbeat:{self.service_id}")

    async def close(self) -> None:
        task = self._task
        self._task = None
        if task is None:
            return

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    def build_payload(self) -> dict[str, Any]:
        extra = dict(self.payload_factory())
        ready = bool(extra.pop("ready", True))

        return {
            "service": self.service_id,
            "ready": ready,
            "version": self.version,
            "uptimeSeconds": max(0, int(time.monotonic() - self._started_monotonic)),
        }

    async def send_once(self) -> None:
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                self.url,
                json=self.build_payload(),
                headers=headers,
            ) as response:
                if response.status not in {200, 202, 204}:
                    body = (await response.text()).strip()
                    if len(body) > 240:
                        body = body[:240] + "…"
                    raise RuntimeError(
                        f"Heartbeat endpoint returned HTTP {response.status}"
                        + (f": {body}" if body else "")
                    )

    async def _run(self) -> None:
        while True:
            try:
                await self.send_once()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                LOGGER.warning("Heartbeat delivery failed: %s", error)

            await asyncio.sleep(self.interval_seconds)
