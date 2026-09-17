from __future__ import annotations

import logging
import time
import unicodedata
from dataclasses import dataclass
from typing import Any

import aiohttp

LOGGER = logging.getLogger(__name__)
_TRIGGER_VARIANTS = str.maketrans({"録": "錄"})


def _normalize_trigger_text(value: str) -> str:
    """Normalize user/tool text so spacing, punctuation and common CJK glyph variants do not break routing."""
    normalized = unicodedata.normalize("NFKC", value).translate(_TRIGGER_VARIANTS).lower()
    return "".join(character for character in normalized if character.isalnum())


class ToolGatewayUnavailable(RuntimeError):
    """Raised when an enabled external tool gateway cannot be reached."""


@dataclass(frozen=True, slots=True)
class ExternalAction:
    tool_name: str
    action_name: str
    function_name: str
    description: str
    parameters: dict[str, object]
    trigger_keywords: tuple[str, ...]
    always_available: bool
    requires_dj: bool

    def declaration(self) -> dict[str, object]:
        return {
            "type": "function",
            "name": self.function_name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolGatewayClient:
    """Small cached HTTP client used by Gemini's ToolRouter."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        token: str | None = None,
        refresh_seconds: float = 30.0,
        request_timeout_seconds: float = 65.0,
    ) -> None:
        connect_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
        self.base_url = f"http://{connect_host}:{port}"
        self.token = token
        self.refresh_seconds = max(1.0, refresh_seconds)
        self.request_timeout_seconds = max(1.0, request_timeout_seconds)
        self._actions: dict[str, ExternalAction] = {}
        self._last_refresh_attempt = 0.0

    @property
    def actions(self) -> tuple[ExternalAction, ...]:
        return tuple(self._actions[name] for name in sorted(self._actions))

    async def refresh(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and self._last_refresh_attempt and now - self._last_refresh_attempt < self.refresh_seconds:
            return
        self._last_refresh_attempt = now
        timeout = aiohttp.ClientTimeout(total=min(self.request_timeout_seconds, 15.0))
        try:
            async with aiohttp.ClientSession(timeout=timeout, headers=self._headers()) as session:
                async with session.get(f"{self.base_url}/v1/tools") as response:
                    if response.status != 200:
                        detail = await response.text()
                        raise ToolGatewayUnavailable(f"gateway returned HTTP {response.status}: {detail[:200]}")
                    payload = await response.json()
        except (aiohttp.ClientError, TimeoutError) as error:
            raise ToolGatewayUnavailable(str(error)) from error
        self._replace_catalog(payload)

    def declarations_for(self, prompt: str, *, is_dj: bool) -> list[dict[str, object]]:
        text = _normalize_trigger_text(prompt)
        matched_actions: list[ExternalAction] = []
        for action in self.actions:
            if action.requires_dj and not is_dj:
                continue
            if action.always_available or any(keyword and keyword in text for keyword in action.trigger_keywords):
                matched_actions.append(action)
        if matched_actions:
            LOGGER.info(
                "External tool routing selected %s action(s): %s",
                len(matched_actions),
                ", ".join(action.function_name for action in matched_actions),
            )
        else:
            LOGGER.info("External tool routing selected no actions")
        return [action.declaration() for action in matched_actions]

    def handles(self, function_name: str) -> bool:
        return function_name in self._actions

    async def invoke(
        self,
        function_name: str,
        arguments: dict[str, object],
        context: dict[str, object],
    ) -> dict[str, object]:
        action = self._actions.get(function_name)
        if action is None:
            raise ValueError(f"Unknown external function: {function_name}")
        if action.requires_dj and context.get("is_dj") is not True:
            raise PermissionError("此工具操作僅限 DJ 或管理員。")
        timeout = aiohttp.ClientTimeout(total=self.request_timeout_seconds)
        body = {"action": action.action_name, "arguments": arguments, "context": context}
        try:
            async with aiohttp.ClientSession(timeout=timeout, headers=self._headers()) as session:
                async with session.post(f"{self.base_url}/v1/tools/{action.tool_name}/invoke", json=body) as response:
                    payload: Any
                    try:
                        payload = await response.json()
                    except Exception:
                        payload = {"error": (await response.text())[:500]}
                    if response.status == 403:
                        raise PermissionError("此工具操作僅限 DJ 或管理員。")
                    if response.status != 200:
                        raise ToolGatewayUnavailable(
                            f"{action.tool_name}/{action.action_name} returned HTTP {response.status}: {payload}"
                        )
        except (aiohttp.ClientError, TimeoutError) as error:
            raise ToolGatewayUnavailable(str(error)) from error
        result = payload.get("result") if isinstance(payload, dict) else None
        if isinstance(result, dict):
            return result
        return {"result": result}

    def _replace_catalog(self, payload: object) -> None:
        if not isinstance(payload, dict) or not isinstance(payload.get("tools"), list):
            raise ToolGatewayUnavailable("gateway returned an invalid tool catalog")
        actions: dict[str, ExternalAction] = {}
        for tool in payload["tools"]:
            if not isinstance(tool, dict):
                continue
            tool_name = tool.get("name")
            keywords = tool.get("trigger_keywords", [])
            always_available = bool(tool.get("always_available", False))
            action_items = tool.get("actions", [])
            if not isinstance(tool_name, str) or not isinstance(keywords, list) or not isinstance(action_items, list):
                continue
            normalized_tool_keywords = tuple(
                normalized
                for item in keywords
                if isinstance(item, str) and item
                for normalized in [_normalize_trigger_text(item)]
                if normalized
            )
            for item in action_items:
                if not isinstance(item, dict):
                    continue
                action_name = item.get("name")
                function_name = item.get("function_name")
                description = item.get("description")
                parameters = item.get("parameters")
                requires_dj = item.get("requires_dj", False)
                action_keywords = item.get("trigger_keywords")
                if not all(isinstance(value, str) for value in (action_name, function_name, description)):
                    continue
                if not isinstance(parameters, dict) or not isinstance(requires_dj, bool):
                    continue
                if isinstance(action_keywords, list):
                    normalized_action_keywords = tuple(
                        normalized
                        for keyword in action_keywords
                        if isinstance(keyword, str) and keyword
                        for normalized in [_normalize_trigger_text(keyword)]
                        if normalized
                    )
                else:
                    normalized_action_keywords = normalized_tool_keywords
                actions[function_name] = ExternalAction(
                    tool_name=tool_name,
                    action_name=action_name,
                    function_name=function_name,
                    description=description,
                    parameters=parameters,
                    trigger_keywords=normalized_action_keywords,
                    always_available=always_available,
                    requires_dj=requires_dj,
                )
        self._actions = actions
        load_errors = payload.get("load_errors")
        if isinstance(load_errors, dict) and load_errors:
            LOGGER.warning("Tool gateway reported plugin load errors: %s", load_errors)

    def _headers(self) -> dict[str, str]:
        if not self.token:
            return {}
        return {"Authorization": f"Bearer {self.token}"}
