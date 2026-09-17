from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Track:
    id: int
    title: str
    original_name: str
    stored_name: str
    uploaded_by: int
    stream_url: str | None = None
    audio_path: str | None = None
    delete_after_play: bool = False

    def path(self, library_root: Path) -> Path:
        return library_root / self.stored_name


@dataclass(frozen=True, slots=True)
class QueuedTrack:
    track: Track
    requested_by: int


@dataclass(frozen=True, slots=True)
class UserMemory:
    id: int
    category: str
    content: str
    created_at: str


@dataclass(frozen=True, slots=True)
class UserActivity:
    message_count: int
    ai_request_count: int
    first_seen: str
    last_seen: str
