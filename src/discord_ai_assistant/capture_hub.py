from __future__ import annotations

import asyncio
import json
import logging
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aiohttp import WSMsgType, web

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class RemoteAgentConnection:
    agent_id: str
    name: str
    ws: web.WebSocketResponse
    paired: bool
    guild_id: int | None = None
    user_id: int | None = None
    screen: dict[str, object] = field(default_factory=dict)
    pairing_code: str | None = None
    pending_requests: dict[str, asyncio.Future[dict[str, object]]] = field(default_factory=dict)


class CaptureHubServer:
    """Accept outbound Capture Agent WebSocket connections and broker RPC calls."""

    def __init__(self, project_root: Path, host: str = "127.0.0.1", port: int = 8878) -> None:
        self.project_root = project_root
        self.host = host
        self.port = port
        self.data_root = project_root / "data" / "capture-hub"
        self.state_path = self.data_root / "state.json"
        self.data_root.mkdir(parents=True, exist_ok=True)
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._agents: dict[str, RemoteAgentConnection] = {}
        self._pending_codes: dict[str, str] = {}
        self._state = self._load_state()
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        app = web.Application()
        app.add_routes([web.get("/capture/ws", self._websocket), web.get("/capture/health", self._health)])
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()
        LOGGER.info("Moxue Capture Hub listening on ws://%s:%s/capture/ws", self.host, self.port)

    async def close(self) -> None:
        for connection in list(self._agents.values()):
            for future in connection.pending_requests.values():
                if not future.done():
                    future.set_exception(RuntimeError("Capture Hub is shutting down"))
            await connection.ws.close()
        self._agents.clear()
        self._pending_codes.clear()
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
            self._site = None

    async def _health(self, _request: web.Request) -> web.Response:
        return web.json_response({"ok": True, "online_agents": len(self._agents)})

    async def _websocket(self, request: web.Request) -> web.StreamResponse:
        ws = web.WebSocketResponse(heartbeat=30)
        await ws.prepare(request)
        connection: RemoteAgentConnection | None = None
        try:
            first = await ws.receive(timeout=15)
            if first.type != WSMsgType.TEXT:
                await ws.close(code=4000, message=b"hello required")
                return ws
            try:
                hello = json.loads(first.data)
            except ValueError:
                await ws.close(code=4001, message=b"invalid json")
                return ws
            if not isinstance(hello, dict) or hello.get("type") != "hello":
                await ws.close(code=4002, message=b"hello required")
                return ws
            agent_id = str(hello.get("agent_id") or "").strip()
            name = str(hello.get("name") or agent_id or "Moxue-PC").strip()
            token = str(hello.get("token") or "").strip()
            requested_code = str(hello.get("pairing_code") or "").strip().upper()
            screen = hello.get("screen") if isinstance(hello.get("screen"), dict) else {}
            if not agent_id:
                await ws.close(code=4003, message=b"agent_id required")
                return ws

            pairing = self._pairing_for(agent_id)
            paired = bool(pairing and token and secrets.compare_digest(str(pairing.get("token") or ""), token))
            code = None
            if not paired:
                code = requested_code if self._valid_pairing_code(requested_code) else self._new_pairing_code()

            connection = RemoteAgentConnection(
                agent_id=agent_id,
                name=name,
                ws=ws,
                paired=paired,
                guild_id=int(pairing["guild_id"]) if paired and pairing and pairing.get("guild_id") else None,
                user_id=int(pairing["user_id"]) if paired and pairing and pairing.get("user_id") else None,
                screen=dict(screen),
                pairing_code=code,
            )
            previous: RemoteAgentConnection | None
            async with self._lock:
                previous = self._agents.get(agent_id)
                if previous is not None and previous.pairing_code:
                    self._pending_codes.pop(previous.pairing_code, None)
                self._agents[agent_id] = connection
                if code:
                    self._pending_codes[code] = agent_id
            if previous is not None and previous.ws is not ws:
                await previous.ws.close(code=4004, message=b"replaced")

            await ws.send_json({
                "type": "hello_ack",
                "paired": paired,
                "agent_id": agent_id,
                "pairing_code": code,
                "guild_id": connection.guild_id,
            })
            LOGGER.info("Capture Agent connected: %s (%s), paired=%s", name, agent_id, paired)

            async for message in ws:
                if message.type == WSMsgType.TEXT:
                    await self._handle_agent_message(connection, message.data)
                elif message.type in {WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR}:
                    break
        except asyncio.TimeoutError:
            LOGGER.warning("Capture Agent connection timed out before hello")
        finally:
            if connection is not None:
                async with self._lock:
                    current = self._agents.get(connection.agent_id)
                    if current is connection:
                        self._agents.pop(connection.agent_id, None)
                    if connection.pairing_code:
                        self._pending_codes.pop(connection.pairing_code, None)
                for future in connection.pending_requests.values():
                    if not future.done():
                        future.set_exception(RuntimeError("Capture Agent disconnected"))
                LOGGER.info("Capture Agent disconnected: %s (%s)", connection.name, connection.agent_id)
        return ws

    async def _handle_agent_message(self, connection: RemoteAgentConnection, raw: str) -> None:
        try:
            payload = json.loads(raw)
        except ValueError:
            return
        if not isinstance(payload, dict):
            return
        msg_type = payload.get("type")
        if msg_type == "result":
            request_id = str(payload.get("id") or "")
            future = connection.pending_requests.pop(request_id, None)
            if future is not None and not future.done():
                result = payload.get("result")
                if payload.get("ok", True):
                    future.set_result(result if isinstance(result, dict) else {"message": str(result)})
                else:
                    future.set_exception(RuntimeError(str(payload.get("error") or "Capture Agent request failed")))
        elif msg_type == "status":
            if isinstance(payload.get("screen"), dict):
                connection.screen = dict(payload["screen"])

    async def pair(self, code: str, guild_id: int, user_id: int) -> dict[str, object]:
        normalized = code.strip().upper()
        async with self._lock:
            agent_id = self._pending_codes.get(normalized)
            connection = self._agents.get(agent_id or "")
            if connection is None or connection.pairing_code != normalized:
                raise RuntimeError("找不到這個配對碼，請確認筆電 Capture Agent 仍在線。")
            token = secrets.token_urlsafe(32)
            pairings = self._state.setdefault("pairings", {})
            pairings[connection.agent_id] = {
                "token": token,
                "guild_id": int(guild_id),
                "user_id": int(user_id),
                "name": connection.name,
            }
            selected = self._state.setdefault("selected", {})
            selected[str(guild_id)] = connection.agent_id
            self._save_state()
            connection.paired = True
            connection.guild_id = int(guild_id)
            connection.user_id = int(user_id)
            self._pending_codes.pop(normalized, None)
            connection.pairing_code = None
        await connection.ws.send_json({"type": "paired", "token": token, "guild_id": int(guild_id)})
        return {"agent_id": connection.agent_id, "name": connection.name, "selected": True}

    def list_agents(self, guild_id: int) -> dict[str, object]:
        paired: list[dict[str, object]] = []
        pending: list[dict[str, object]] = []
        selected_id = self.selected_agent_id(guild_id)
        for connection in self._agents.values():
            item = {
                "agent_id": connection.agent_id,
                "name": connection.name,
                "online": not connection.ws.closed,
                "screen": connection.screen,
                "selected": connection.agent_id == selected_id,
            }
            if connection.paired and connection.guild_id == guild_id:
                paired.append(item)
            elif not connection.paired:
                pending.append(item)
        return {"agents": paired, "pending": pending, "selected_agent_id": selected_id}

    def selected_agent_id(self, guild_id: int) -> str | None:
        selected = self._state.get("selected") if isinstance(self._state.get("selected"), dict) else {}
        value = selected.get(str(guild_id))
        return str(value) if value else None

    async def select(self, guild_id: int, agent_id: str) -> dict[str, object]:
        connection = self._agents.get(agent_id)
        if connection is None or not connection.paired or connection.guild_id != guild_id:
            raise RuntimeError("找不到已配對且在線的 Capture Agent。")
        self._state.setdefault("selected", {})[str(guild_id)] = agent_id
        self._save_state()
        return {"agent_id": agent_id, "name": connection.name, "selected": True}

    def selected_connection(self, guild_id: int) -> RemoteAgentConnection | None:
        agent_id = self.selected_agent_id(guild_id)
        if not agent_id:
            return None
        connection = self._agents.get(agent_id)
        if connection is None or not connection.paired or connection.guild_id != guild_id or connection.ws.closed:
            return None
        return connection

    async def request(
        self,
        guild_id: int,
        action: str,
        payload: dict[str, object] | None = None,
        *,
        timeout_seconds: float = 120.0,
    ) -> dict[str, object]:
        connection = self.selected_connection(guild_id)
        if connection is None:
            raise RuntimeError("目前沒有已選取且在線的遠端 Capture Agent。")
        request_id = secrets.token_hex(12)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, object]] = loop.create_future()
        connection.pending_requests[request_id] = future
        try:
            await connection.ws.send_json({
                "type": "request",
                "id": request_id,
                "action": action,
                "payload": payload or {},
            })
            return await asyncio.wait_for(future, timeout=timeout_seconds)
        finally:
            connection.pending_requests.pop(request_id, None)

    def _pairing_for(self, agent_id: str) -> dict[str, object] | None:
        pairings = self._state.get("pairings") if isinstance(self._state.get("pairings"), dict) else {}
        value = pairings.get(agent_id)
        return value if isinstance(value, dict) else None

    @staticmethod
    def _new_pairing_code() -> str:
        alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
        return "".join(secrets.choice(alphabet) for _ in range(6))

    @staticmethod
    def _valid_pairing_code(value: str) -> bool:
        return len(value) == 6 and value.isalnum()

    def _load_state(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        payload.setdefault("pairings", {})
        payload.setdefault("selected", {})
        return payload

    def _save_state(self) -> None:
        temp = self.state_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(self._state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.replace(self.state_path)
