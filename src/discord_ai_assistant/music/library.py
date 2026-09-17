from __future__ import annotations

import uuid
from pathlib import Path

import discord

from discord_ai_assistant.models import Track
from discord_ai_assistant.storage.database import Database

ALLOWED_EXTENSIONS = {".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav"}


class LibraryService:
    def __init__(self, root: Path, database: Database, max_upload_bytes: int) -> None:
        self.root = root
        self.database = database
        self.max_upload_bytes = max_upload_bytes
        self.root.mkdir(parents=True, exist_ok=True)

    async def add_attachment(self, attachment: discord.Attachment, uploaded_by: int) -> Track:
        original_name = attachment.filename
        extension = Path(original_name).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            supported = ", ".join(sorted(ALLOWED_EXTENSIONS))
            raise ValueError(f"不支援此格式，僅接受：{supported}")
        if attachment.size > self.max_upload_bytes:
            raise ValueError(f"檔案超過上傳上限（{self.max_upload_bytes // 1024 // 1024} MB）。")

        stored_name = f"{uuid.uuid4().hex}{extension}"
        destination = self.root / stored_name
        await attachment.save(destination)
        try:
            return self.database.add_track(Path(original_name).stem, original_name, stored_name, uploaded_by)
        except Exception:
            destination.unlink(missing_ok=True)
            raise

    def search(self, query: str) -> list[Track]:
        return self.database.search_tracks(query)

    def list(self, order: str, page: int, per_page: int = 20) -> tuple[list[Track], int]:
        return self.database.list_tracks(order, page, per_page)

    def random_track(self, exclude_track_id: int | None = None) -> Track | None:
        return self.database.random_track(exclude_track_id)

    def delete(self, track_id: int) -> Track | None:
        track = self.database.delete_track(track_id)
        if not track:
            return None

        path = track.path(self.root).resolve()
        library_root = self.root.resolve()
        if library_root in path.parents:
            path.unlink(missing_ok=True)
        return track
