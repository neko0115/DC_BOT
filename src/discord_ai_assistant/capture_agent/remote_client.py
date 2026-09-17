from __future__ import annotations

import asyncio
import json
import logging
import secrets
from typing import Any

import aiohttp

from .capture import virtual_screen_bounds
from .profiles import DATA_ROOT, ProfileStore

LOGGER = logging.getLogger(__name__)
REMOTE_STATE_PATH = DATA_ROOT / "remote.json"


class CaptureAgentRemoteClient:
    def __init__(self, hub_url: str, *, local_url: str = "http://127.0.0.1:8877") -> None:
        self.hub_url = hub_url
        self.local_url = local_url.rstrip("/")
        self.store = ProfileStore()
        self._stop = asyncio.Event()
        self._request_tasks: set[asyncio.Task[None]] = set()
        self._state = self._load_state()
        if not self._state.get("pairing_code"):
            self._state["pairing_code"] = self._new_pairing_code()
            self._save_state()

    @property
    def pairing_code(self) -> str:
        return str(self._state.get("pairing_code") or "")

    @property
    def token(self) -> str | None:
        value = str(self._state.get("token") or "").strip()
        return value or None

    async def run_forever(self) -> None:
        LOGGER.info("Capture Agent remote bridge target: %s", self.hub_url)
        LOGGER.info("Capture Agent pairing code: %s", self.pairing_code)
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=15, sock_read=None)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while not self._stop.is_set():
                try:
                    await self._connect_once(session)
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    LOGGER.warning("Capture Agent remote connection failed: %s", error)
                if not self._stop.is_set():
                    await asyncio.sleep(3)

    async def close(self) -> None:
        self._stop.set()
        for task in list(self._request_tasks):
            task.cancel()
        if self._request_tasks:
            await asyncio.gather(*self._request_tasks, return_exceptions=True)
        self._request_tasks.clear()

    async def _connect_once(self, session: aiohttp.ClientSession) -> None:
        screen = virtual_screen_bounds()
        async with session.ws_connect(self.hub_url, heartbeat=30) as ws:
            await ws.send_json({
                "type": "hello",
                "agent_id": self.store.agent_id,
                "name": self.store.agent_name,
                "token": self.token,
                "pairing_code": self.pairing_code,
                "screen": {"x": screen.x, "y": screen.y, "width": screen.width, "height": screen.height},
            })
            async for message in ws:
                if message.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_message(session, ws, message.data)
                elif message.type in {aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR}:
                    break

    async def _handle_message(
        self,
        session: aiohttp.ClientSession,
        ws: aiohttp.ClientWebSocketResponse,
        raw: str,
    ) -> None:
        try:
            payload = json.loads(raw)
        except ValueError:
            return
        if not isinstance(payload, dict):
            return
        msg_type = payload.get("type")
        if msg_type == "hello_ack":
            if payload.get("paired"):
                LOGGER.info("Capture Agent authenticated with Moxue Hub")
            else:
                code = str(payload.get("pairing_code") or self.pairing_code)
                if code and code != self.pairing_code:
                    self._state["pairing_code"] = code
                    self._save_state()
                LOGGER.info("Capture Agent waiting for pairing; code=%s", code)
        elif msg_type == "paired":
            token = str(payload.get("token") or "").strip()
            if token:
                self._state["token"] = token
                self._state["guild_id"] = payload.get("guild_id")
                self._save_state()
                LOGGER.info("Capture Agent pairing completed; credentials saved")
        elif msg_type == "request":
            task = asyncio.create_task(
                self._process_request(session, ws, payload),
                name=f"capture-agent-rpc-{payload.get('id', 'unknown')}",
            )
            self._request_tasks.add(task)
            task.add_done_callback(self._request_tasks.discard)

    async def _process_request(
        self,
        session: aiohttp.ClientSession,
        ws: aiohttp.ClientWebSocketResponse,
        payload: dict[str, object],
    ) -> None:
        request_id = str(payload.get("id") or "")
        action = str(payload.get("action") or "")
        request_payload = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
        try:
            result = await self._execute_local(session, action, request_payload)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            response: dict[str, object] = {
                "type": "result",
                "id": request_id,
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }
        else:
            response = {"type": "result", "id": request_id, "ok": True, "result": result}
        try:
            await ws.send_json(response)
        except (ConnectionResetError, RuntimeError):
            LOGGER.warning("Could not return RPC result %s because Hub connection closed", request_id)

    async def _execute_local(
        self,
        session: aiohttp.ClientSession,
        action: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        routes = {
            "health": ("GET", "/health"),
            "profiles": ("GET", "/profiles"),
            "select_profile": ("POST", "/profiles/select"),
            "calibrate": ("POST", "/calibrate"),
            "ocr_test": ("POST", "/ocr-test"),
        }
        route = routes.get(action)
        if route is None:
            raise RuntimeError(f"unsupported remote action: {action}")
        method, path = route
        timeout_seconds = 180 if action == "calibrate" else 60
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        async with session.request(method, f"{self.local_url}{path}", json=payload, timeout=timeout) as response:
            text = await response.text()
            try:
                data = json.loads(text)
            except ValueError:
                data = {"message": text or f"Local Capture Agent HTTP {response.status}"}
            if response.status >= 400:
                raise RuntimeError(str(data.get("message") or f"Local Capture Agent HTTP {response.status}"))
            return data if isinstance(data, dict) else {"message": str(data)}

    @staticmethod
    def _new_pairing_code() -> str:
        alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
        return "".join(secrets.choice(alphabet) for _ in range(6))

    @staticmethod
    def _load_state() -> dict[str, Any]:
        try:
            payload = json.loads(REMOTE_STATE_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _save_state(self) -> None:
        REMOTE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        temp = REMOTE_STATE_PATH.with_suffix(".json.tmp")
        temp.write_text(json.dumps(self._state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.replace(REMOTE_STATE_PATH)
