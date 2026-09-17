from __future__ import annotations

import asyncio
import hmac
import ipaddress
import logging
from pathlib import Path

from aiohttp import web

from discord_ai_assistant.tool_gateway.registry import ToolRegistry

LOGGER = logging.getLogger(__name__)


class ToolGatewayServer:
    """Expose all repository tools through one local HTTP port."""

    def __init__(
        self,
        tools_root: Path,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        token: str | None = None,
        invocation_timeout_seconds: float = 60.0,
    ) -> None:
        if not token and not _is_loopback_host(host):
            raise ValueError("TOOL_GATEWAY_TOKEN is required when the gateway is not bound to loopback.")
        self.registry = ToolRegistry(tools_root)
        self.host = host
        self.port = port
        self.token = token
        self.invocation_timeout_seconds = invocation_timeout_seconds
        self._runner: web.AppRunner | None = None

    async def start(self) -> None:
        if self._runner is not None:
            return
        self.registry.reload()
        app = web.Application(middlewares=[self._auth_middleware])
        app.add_routes(
            [
                web.get("/health", self._health),
                web.get("/v1/tools", self._list_tools),
                web.get("/v1/tools/{tool_name}", self._get_tool),
                web.post("/v1/tools/{tool_name}/invoke", self._invoke_tool),
                web.post("/v1/reload", self._reload),
            ]
        )
        runner = web.AppRunner(app, access_log=LOGGER)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        try:
            await site.start()
        except Exception:
            await runner.cleanup()
            raise
        self._runner = runner
        LOGGER.info("Tool gateway listening on http://%s:%s", self.host, self.port)

    async def close(self) -> None:
        if self._runner is None:
            return
        await self._runner.cleanup()
        self._runner = None

    @web.middleware
    async def _auth_middleware(self, request: web.Request, handler):  # type: ignore[no-untyped-def]
        if self.token:
            supplied = request.headers.get("Authorization", "")
            expected = f"Bearer {self.token}"
            if not hmac.compare_digest(supplied, expected):
                raise web.HTTPUnauthorized(text="invalid gateway token")
        return await handler(request)

    async def _health(self, request: web.Request) -> web.Response:
        self.registry.reload()
        return web.json_response(
            {
                "ok": True,
                "tool_count": len(self.registry.tools),
                "load_errors": self.registry.errors,
            }
        )

    async def _list_tools(self, request: web.Request) -> web.Response:
        self.registry.reload()
        return web.json_response(
            {
                "tools": self.registry.catalog(),
                "load_errors": self.registry.errors,
            }
        )

    async def _get_tool(self, request: web.Request) -> web.Response:
        self.registry.reload()
        tool = self.registry.get(request.match_info["tool_name"])
        if tool is None:
            raise web.HTTPNotFound(text="unknown tool")
        payload = tool.public_payload()
        payload["tool_documentation"] = tool.documentation
        payload["changelog"] = tool.changelog
        return web.json_response(payload)

    async def _invoke_tool(self, request: web.Request) -> web.Response:
        self.registry.reload()
        tool_name = request.match_info["tool_name"]
        if self.registry.get(tool_name) is None:
            raise web.HTTPNotFound(text="unknown tool")
        try:
            payload = await request.json()
        except Exception as error:
            raise web.HTTPBadRequest(text="request body must be JSON") from error
        if not isinstance(payload, dict):
            raise web.HTTPBadRequest(text="request body must be an object")
        action = payload.get("action")
        arguments = payload.get("arguments", {})
        context = payload.get("context", {})
        if not isinstance(action, str) or not action:
            raise web.HTTPBadRequest(text="action is required")
        if not isinstance(arguments, dict) or not isinstance(context, dict):
            raise web.HTTPBadRequest(text="arguments and context must be objects")
        try:
            result = await asyncio.wait_for(
                self.registry.invoke(tool_name, action, arguments, context),
                timeout=self.invocation_timeout_seconds,
            )
        except KeyError as error:
            raise web.HTTPNotFound(text=str(error)) from error
        except PermissionError as error:
            raise web.HTTPForbidden(text=str(error)) from error
        except asyncio.TimeoutError as error:
            raise web.HTTPGatewayTimeout(text="tool invocation timed out") from error
        except Exception as error:
            LOGGER.exception("External tool %s/%s failed", tool_name, action)
            raise web.HTTPInternalServerError(text=f"tool failed: {type(error).__name__}") from error
        return web.json_response({"ok": True, "tool": tool_name, "action": action, "result": result})

    async def _reload(self, request: web.Request) -> web.Response:
        self.registry.reload()
        return web.json_response(
            {
                "ok": True,
                "tools": self.registry.catalog(),
                "load_errors": self.registry.errors,
            }
        )


def _is_loopback_host(host: str) -> bool:
    normalized = host.strip().lower()
    if normalized in {"localhost", "::1"}:
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False
