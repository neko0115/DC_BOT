from __future__ import annotations

import importlib.util
import inspect
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Awaitable, Callable

LOGGER = logging.getLogger(__name__)
IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
InvokeCallable = Callable[[str, dict[str, object], dict[str, object]], object | Awaitable[object]]


class ToolLoadError(RuntimeError):
    """Raised when an external tool does not satisfy the gateway contract."""


@dataclass(frozen=True, slots=True)
class ToolAction:
    name: str
    description: str
    parameters: dict[str, object]


@dataclass(frozen=True, slots=True)
class ToolManifest:
    name: str
    version: str
    description: str
    enabled: bool
    entrypoint: str
    trigger_keywords: tuple[str, ...]
    always_available: bool
    actions: tuple[ToolAction, ...]


@dataclass(slots=True)
class RegisteredTool:
    root: Path
    manifest: ToolManifest
    module: ModuleType
    invoke: InvokeCallable
    documentation: str
    changelog: str
    signature: tuple[tuple[str, int, int], ...]

    def public_payload(self) -> dict[str, object]:
        actions: list[dict[str, object]] = []
        for action in self.manifest.actions:
            function_name = external_function_name(self.manifest.name, action.name)
            actions.append(
                {
                    "name": action.name,
                    "function_name": function_name,
                    "description": action.description,
                    "parameters": action.parameters,
                }
            )
        return {
            "name": self.manifest.name,
            "version": self.manifest.version,
            "description": self.manifest.description,
            "trigger_keywords": list(self.manifest.trigger_keywords),
            "always_available": self.manifest.always_available,
            "actions": actions,
        }


def external_function_name(tool_name: str, action_name: str) -> str:
    name = f"x_{tool_name}_{action_name}"
    if len(name) > 64:
        raise ToolLoadError(f"External function name is too long: {name}")
    return name


class ToolRegistry:
    """Discover and execute tools stored under the repository-level tools directory."""

    def __init__(self, tools_root: Path) -> None:
        self.tools_root = tools_root
        self._tools: dict[str, RegisteredTool] = {}
        self.errors: dict[str, str] = {}

    @property
    def tools(self) -> tuple[RegisteredTool, ...]:
        return tuple(self._tools[name] for name in sorted(self._tools))

    def reload(self) -> None:
        self.tools_root.mkdir(parents=True, exist_ok=True)
        discovered: dict[str, RegisteredTool] = {}
        errors: dict[str, str] = {}
        for directory in sorted(self.tools_root.iterdir()):
            if not directory.is_dir() or directory.name.startswith("_"):
                continue
            try:
                signature = self._signature(directory)
                previous = self._tools.get(directory.name)
                if previous and previous.signature == signature:
                    discovered[previous.manifest.name] = previous
                    continue
                registered = self._load_tool(directory, signature)
                if registered.manifest.enabled:
                    if registered.manifest.name in discovered:
                        raise ToolLoadError(f"Duplicate tool name: {registered.manifest.name}")
                    discovered[registered.manifest.name] = registered
            except Exception as error:  # A broken plugin must not take the bot down.
                LOGGER.exception("Failed to load external tool from %s", directory)
                errors[directory.name] = str(error)
        self._tools = discovered
        self.errors = errors

    def catalog(self) -> list[dict[str, object]]:
        return [tool.public_payload() for tool in self.tools]

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    async def invoke(
        self,
        tool_name: str,
        action_name: str,
        arguments: dict[str, object],
        context: dict[str, object],
    ) -> object:
        tool = self._tools.get(tool_name)
        if tool is None:
            raise KeyError(f"Unknown tool: {tool_name}")
        if action_name not in {action.name for action in tool.manifest.actions}:
            raise KeyError(f"Unknown action {action_name!r} for tool {tool_name!r}")
        result = tool.invoke(action_name, arguments, context)
        if inspect.isawaitable(result):
            result = await result
        return result

    def _load_tool(self, directory: Path, signature: tuple[tuple[str, int, int], ...]) -> RegisteredTool:
        manifest_path = directory / "manifest.json"
        docs_path = directory / "TOOL.md"
        changelog_path = directory / "CHANGELOG.md"
        if not manifest_path.is_file():
            raise ToolLoadError("manifest.json is required")
        if not docs_path.is_file():
            raise ToolLoadError("TOOL.md is required")
        if not changelog_path.is_file():
            raise ToolLoadError("CHANGELOG.md is required")

        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = self._parse_manifest(raw, directory.name)
        entrypoint = (directory / manifest.entrypoint).resolve()
        if entrypoint.parent != directory.resolve() or not entrypoint.is_file():
            raise ToolLoadError("entrypoint must be a file directly inside the tool directory")

        module_name = f"moxue_external_tool_{manifest.name}_{abs(hash(signature))}"
        spec = importlib.util.spec_from_file_location(module_name, entrypoint)
        if spec is None or spec.loader is None:
            raise ToolLoadError(f"Unable to import entrypoint: {entrypoint.name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        invoke = getattr(module, "invoke", None)
        if not callable(invoke):
            sys.modules.pop(module_name, None)
            raise ToolLoadError("entrypoint must export invoke(action, arguments, context)")

        return RegisteredTool(
            root=directory,
            manifest=manifest,
            module=module,
            invoke=invoke,
            documentation=docs_path.read_text(encoding="utf-8"),
            changelog=changelog_path.read_text(encoding="utf-8"),
            signature=signature,
        )

    @staticmethod
    def _signature(directory: Path) -> tuple[tuple[str, int, int], ...]:
        tracked = [directory / "manifest.json", directory / "TOOL.md", directory / "CHANGELOG.md"]
        manifest_path = directory / "manifest.json"
        if manifest_path.is_file():
            try:
                raw = json.loads(manifest_path.read_text(encoding="utf-8"))
                entrypoint = raw.get("entrypoint", "tool.py") if isinstance(raw, dict) else "tool.py"
                if isinstance(entrypoint, str):
                    tracked.append(directory / entrypoint)
            except (OSError, json.JSONDecodeError):
                pass
        signature: list[tuple[str, int, int]] = []
        for path in tracked:
            if path.is_file():
                stat = path.stat()
                signature.append((path.name, stat.st_mtime_ns, stat.st_size))
            else:
                signature.append((path.name, -1, -1))
        return tuple(signature)

    @staticmethod
    def _parse_manifest(raw: object, directory_name: str) -> ToolManifest:
        if not isinstance(raw, dict):
            raise ToolLoadError("manifest.json must contain a JSON object")

        name = ToolRegistry._identifier(raw.get("name"), "name")
        if name != directory_name:
            raise ToolLoadError("manifest name must match the tool directory name")
        version = ToolRegistry._text(raw.get("version"), "version")
        description = ToolRegistry._text(raw.get("description"), "description")
        entrypoint = ToolRegistry._text(raw.get("entrypoint", "tool.py"), "entrypoint")
        enabled = raw.get("enabled", True)
        always_available = raw.get("always_available", False)
        if not isinstance(enabled, bool) or not isinstance(always_available, bool):
            raise ToolLoadError("enabled and always_available must be booleans")

        keywords_raw = raw.get("trigger_keywords", [])
        if not isinstance(keywords_raw, list) or not all(isinstance(item, str) for item in keywords_raw):
            raise ToolLoadError("trigger_keywords must be a string array")
        keywords = tuple(item.strip().lower() for item in keywords_raw if item.strip())

        actions_raw = raw.get("actions")
        if not isinstance(actions_raw, dict) or not actions_raw:
            raise ToolLoadError("actions must be a non-empty object")
        actions: list[ToolAction] = []
        for action_name, action_raw in actions_raw.items():
            action_name = ToolRegistry._identifier(action_name, "action name")
            if not isinstance(action_raw, dict):
                raise ToolLoadError(f"Action {action_name} must be an object")
            action_description = ToolRegistry._text(action_raw.get("description"), f"{action_name}.description")
            parameters = action_raw.get("parameters", {"type": "object", "properties": {}})
            if not isinstance(parameters, dict) or parameters.get("type") != "object":
                raise ToolLoadError(f"{action_name}.parameters must be a JSON object schema")
            external_function_name(name, action_name)
            actions.append(ToolAction(action_name, action_description, parameters))

        return ToolManifest(
            name=name,
            version=version,
            description=description,
            enabled=enabled,
            entrypoint=entrypoint,
            trigger_keywords=keywords,
            always_available=always_available,
            actions=tuple(actions),
        )

    @staticmethod
    def _identifier(value: object, field: str) -> str:
        if not isinstance(value, str) or not IDENTIFIER_RE.fullmatch(value):
            raise ToolLoadError(f"{field} must match {IDENTIFIER_RE.pattern}")
        return value

    @staticmethod
    def _text(value: object, field: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ToolLoadError(f"{field} must be a non-empty string")
        return value.strip()
