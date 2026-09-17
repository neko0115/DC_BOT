from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


def normalize_hub_url(value: str) -> str:
    raw = value.strip()
    if not raw:
        return ""
    if "://" not in raw:
        if ":" not in raw:
            raw = f"{raw}:8878"
        raw = f"ws://{raw}"
    parsed = urlsplit(raw)
    scheme = {"http": "ws", "https": "wss"}.get(parsed.scheme.lower(), parsed.scheme.lower())
    if scheme not in {"ws", "wss"}:
        raise ValueError("Hub 位址必須是 IP/hostname、ws:// 或 wss://。")
    if not parsed.netloc:
        raise ValueError("Hub 位址缺少主機名稱。")
    path = parsed.path.rstrip("/")
    if not path:
        path = "/capture/ws"
    return urlunsplit((scheme, parsed.netloc, path, "", ""))


def load_remote_state(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def save_remote_state(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def save_hub_url(path: Path, hub_url: str) -> None:
    state = load_remote_state(path)
    state["hub_url"] = hub_url
    save_remote_state(path, state)
