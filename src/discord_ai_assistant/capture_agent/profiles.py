from __future__ import annotations

import json
import os
import socket
import sys
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _runtime_paths() -> tuple[Path, Path]:
    if getattr(sys, "frozen", False):
        local_app_data = os.getenv("LOCALAPPDATA", "").strip()
        app_root = Path(local_app_data) / "MoxueCapture" if local_app_data else Path.home() / ".moxue-capture"
        return app_root, app_root
    project_root = Path(__file__).resolve().parents[3]
    return project_root / "data" / "capture-agent", project_root


DATA_ROOT, PROJECT_ROOT = _runtime_paths()
IDENTITY_PATH = DATA_ROOT / "agent.json"
PROFILES_PATH = DATA_ROOT / "profiles.json"
if getattr(sys, "frozen", False):
    MEETING_CONFIG_PATH = DATA_ROOT / "meeting-config.json"
    MEETING_DEFAULT_CONFIG_PATH = DATA_ROOT / "meeting-default-config.json"
else:
    MEETING_CONFIG_PATH = PROJECT_ROOT / "data" / "game-meeting-recorder" / "config.json"
    MEETING_DEFAULT_CONFIG_PATH = PROJECT_ROOT / "tools" / "game_meeting_recorder" / "config.json"


@dataclass(frozen=True, slots=True)
class ScreenBounds:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class CaptureProfile:
    id: str
    name: str
    game: str
    screen: ScreenBounds
    roi_x: float
    roi_y: float
    roi_width: float
    roi_height: float

    def pixel_roi(self, screen: ScreenBounds | None = None) -> dict[str, int]:
        target = screen or self.screen
        x = target.x + round(self.roi_x * target.width)
        y = target.y + round(self.roi_y * target.height)
        width = max(20, round(self.roi_width * target.width))
        height = max(20, round(self.roi_height * target.height))
        return {"x": x, "y": y, "width": width, "height": height}

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["screen"] = asdict(self.screen)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CaptureProfile":
        raw_screen = payload.get("screen") if isinstance(payload.get("screen"), dict) else {}
        return cls(
            id=str(payload["id"]),
            name=str(payload.get("name") or payload["id"]),
            game=str(payload.get("game") or "unknown"),
            screen=ScreenBounds(
                int(raw_screen.get("x", 0)),
                int(raw_screen.get("y", 0)),
                int(raw_screen.get("width", 1)),
                int(raw_screen.get("height", 1)),
            ),
            roi_x=float(payload.get("roi_x", 0.0)),
            roi_y=float(payload.get("roi_y", 0.0)),
            roi_width=float(payload.get("roi_width", 1.0)),
            roi_height=float(payload.get("roi_height", 1.0)),
        )


class ProfileStore:
    def __init__(self) -> None:
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        self.agent_id, self.agent_name = self._load_or_create_identity()

    def _load_or_create_identity(self) -> tuple[str, str]:
        if IDENTITY_PATH.is_file():
            try:
                payload = json.loads(IDENTITY_PATH.read_text(encoding="utf-8"))
                agent_id = str(payload.get("agent_id") or "").strip()
                agent_name = str(payload.get("name") or "").strip()
                if agent_id and agent_name:
                    return agent_id, agent_name
            except (OSError, ValueError, TypeError):
                pass
        agent_id = str(uuid.uuid4())
        agent_name = socket.gethostname() or "Moxue-PC"
        IDENTITY_PATH.write_text(
            json.dumps({"agent_id": agent_id, "name": agent_name}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return agent_id, agent_name

    def _load_payload(self) -> dict[str, Any]:
        if not PROFILES_PATH.is_file():
            return {"active_profile_id": None, "profiles": []}
        try:
            payload = json.loads(PROFILES_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {"active_profile_id": None, "profiles": []}
        if not isinstance(payload, dict):
            return {"active_profile_id": None, "profiles": []}
        if not isinstance(payload.get("profiles"), list):
            payload["profiles"] = []
        return payload

    def _save_payload(self, payload: dict[str, Any]) -> None:
        PROFILES_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_profiles(self) -> list[CaptureProfile]:
        profiles: list[CaptureProfile] = []
        for item in self._load_payload().get("profiles", []):
            if not isinstance(item, dict):
                continue
            try:
                profiles.append(CaptureProfile.from_dict(item))
            except (KeyError, TypeError, ValueError):
                continue
        return profiles

    def active_profile(self) -> CaptureProfile | None:
        payload = self._load_payload()
        active_id = payload.get("active_profile_id")
        for profile in self.list_profiles():
            if profile.id == active_id:
                return profile
        return None

    def upsert(self, profile: CaptureProfile, *, activate: bool = True) -> None:
        payload = self._load_payload()
        profiles = [item for item in payload.get("profiles", []) if isinstance(item, dict) and item.get("id") != profile.id]
        profiles.append(profile.to_dict())
        payload["profiles"] = profiles
        if activate:
            payload["active_profile_id"] = profile.id
        self._save_payload(payload)

    def select(self, profile_id: str) -> CaptureProfile | None:
        profile = next((item for item in self.list_profiles() if item.id == profile_id), None)
        if profile is None:
            return None
        payload = self._load_payload()
        payload["active_profile_id"] = profile.id
        self._save_payload(payload)
        return profile

    def sync_meeting_config(self, current_screen: ScreenBounds) -> CaptureProfile | None:
        profile = self.active_profile()
        if profile is None:
            return None
        if MEETING_CONFIG_PATH.is_file():
            try:
                config = json.loads(MEETING_CONFIG_PATH.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                config = {}
        elif MEETING_DEFAULT_CONFIG_PATH.is_file():
            try:
                config = json.loads(MEETING_DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                config = {}
        else:
            config = {}
        chat = config.setdefault("chat", {})
        chat.update(profile.pixel_roi(current_screen))
        chat["enabled"] = True
        chat["profile_id"] = profile.id
        chat["profile_name"] = profile.name
        chat["normalized_roi"] = {
            "x": profile.roi_x,
            "y": profile.roi_y,
            "width": profile.roi_width,
            "height": profile.roi_height,
        }
        MEETING_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        MEETING_CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        return profile
