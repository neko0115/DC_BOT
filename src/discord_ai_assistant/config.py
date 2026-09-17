from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from discord_ai_assistant.time_utils import resolve_timezone


@dataclass(frozen=True, slots=True)
class Settings:
    project_root: Path
    discord_token: str
    gemini_api_key: str | None
    discord_guild_id: int | None
    dj_role_name: str
    dj_role_id: int | None
    gemini_model: str
    max_upload_bytes: int
    persona_channel_name: str
    introduction_channel_name: str
    passive_decision_cooldown_seconds: int
    passive_response_cooldown_seconds: int
    topic_idle_minutes: int
    topic_min_interval_minutes: int
    persona_timezone: str
    voice_empty_disconnect_minutes: int
    voice_test_channel_name: str
    voice_recognition_enabled: bool
    voice_recognition_model: str
    voice_recognition_language: str
    voice_recognition_beam_size: int
    voice_recognition_silence_seconds: float
    voice_recognition_max_segment_seconds: float
    voice_recognition_initial_prompt: str
    youtube_cookies_from_browser: str | None
    youtube_cookies_file: Path | None
    youtube_po_token: str | None

    @property
    def database_path(self) -> Path:
        return self.project_root / "data" / "assistant.sqlite3"

    @property
    def library_path(self) -> Path:
        return self.project_root / "library"


def load_settings(project_root: Path) -> Settings:
    load_dotenv(project_root / ".env")
    token = os.getenv("DISCORD_TOKEN", "").strip()
    if not token:
        raise RuntimeError("DISCORD_TOKEN is required. Copy .env.example to .env first.")

    guild_id = os.getenv("DISCORD_GUILD_ID", "").strip()
    dj_role_id = os.getenv("DJ_ROLE_ID", "").strip()
    max_upload_mb = int(os.getenv("MAX_UPLOAD_MB", "25"))
    if max_upload_mb < 1:
        raise RuntimeError("MAX_UPLOAD_MB must be at least 1.")

    passive_decision_cooldown_seconds = int(os.getenv("PASSIVE_DECISION_COOLDOWN_SECONDS", "90"))
    passive_response_cooldown_seconds = int(os.getenv("PASSIVE_RESPONSE_COOLDOWN_SECONDS", "120"))
    topic_idle_minutes = int(os.getenv("TOPIC_IDLE_MINUTES", "15"))
    topic_min_interval_minutes = int(os.getenv("TOPIC_MIN_INTERVAL_MINUTES", "45"))
    if passive_decision_cooldown_seconds < 15 or passive_response_cooldown_seconds < 15:
        raise RuntimeError("Passive response cooldowns must be at least 15 seconds.")
    if topic_idle_minutes < 1 or topic_min_interval_minutes < 1:
        raise RuntimeError("Topic intervals must be at least 1 minute.")
    voice_empty_disconnect_minutes = int(os.getenv("VOICE_EMPTY_DISCONNECT_MINUTES", "30"))
    if voice_empty_disconnect_minutes < 1:
        raise RuntimeError("VOICE_EMPTY_DISCONNECT_MINUTES must be at least 1.")
    voice_recognition_beam_size = int(os.getenv("VOICE_RECOGNITION_BEAM_SIZE", "5"))
    voice_recognition_silence_seconds = float(os.getenv("VOICE_RECOGNITION_SILENCE_SECONDS", "1.8"))
    voice_recognition_max_segment_seconds = float(os.getenv("VOICE_RECOGNITION_MAX_SEGMENT_SECONDS", "20"))
    if not 1 <= voice_recognition_beam_size <= 10:
        raise RuntimeError("VOICE_RECOGNITION_BEAM_SIZE must be between 1 and 10.")
    if not 0.5 <= voice_recognition_silence_seconds <= 10:
        raise RuntimeError("VOICE_RECOGNITION_SILENCE_SECONDS must be between 0.5 and 10.")
    if not voice_recognition_silence_seconds < voice_recognition_max_segment_seconds <= 60:
        raise RuntimeError("VOICE_RECOGNITION_MAX_SEGMENT_SECONDS must be greater than silence and at most 60.")
    persona_timezone = os.getenv("PERSONA_TIMEZONE", "Asia/Taipei").strip() or "Asia/Taipei"
    try:
        resolve_timezone(persona_timezone)
    except ValueError as error:
        raise RuntimeError(str(error)) from error
    youtube_cookies_from_browser = os.getenv("YOUTUBE_COOKIES_FROM_BROWSER", "").strip() or None
    youtube_cookies_file_value = os.getenv("YOUTUBE_COOKIES_FILE", "").strip()
    if youtube_cookies_from_browser and youtube_cookies_file_value:
        raise RuntimeError("Set only one of YOUTUBE_COOKIES_FROM_BROWSER or YOUTUBE_COOKIES_FILE.")
    youtube_cookies_file = None
    if youtube_cookies_file_value:
        youtube_cookies_file = Path(youtube_cookies_file_value).expanduser()
        if not youtube_cookies_file.is_absolute():
            youtube_cookies_file = project_root / youtube_cookies_file
        if not youtube_cookies_file.is_file():
            raise RuntimeError(f"YOUTUBE_COOKIES_FILE does not exist: {youtube_cookies_file}")

    return Settings(
        project_root=project_root,
        discord_token=token,
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip() or None,
        discord_guild_id=int(guild_id) if guild_id else None,
        dj_role_name=os.getenv("DJ_ROLE_NAME", "DJ").strip() or "DJ",
        dj_role_id=int(dj_role_id) if dj_role_id else None,
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip(),
        max_upload_bytes=max_upload_mb * 1024 * 1024,
        persona_channel_name=os.getenv("PERSONA_CHANNEL_NAME", "墨雪的貓窩").strip(),
        introduction_channel_name=os.getenv("INTRODUCTION_CHANNEL_NAME", "墨雪的自我介紹").strip(),
        passive_decision_cooldown_seconds=passive_decision_cooldown_seconds,
        passive_response_cooldown_seconds=passive_response_cooldown_seconds,
        topic_idle_minutes=topic_idle_minutes,
        topic_min_interval_minutes=topic_min_interval_minutes,
        persona_timezone=persona_timezone,
        voice_empty_disconnect_minutes=voice_empty_disconnect_minutes,
        voice_test_channel_name=os.getenv("VOICE_TEST_CHANNEL_NAME", "測試").strip(),
        voice_recognition_enabled=os.getenv("VOICE_RECOGNITION_ENABLED", "false").strip().lower()
        in {"1", "true", "yes", "on"},
        voice_recognition_model=os.getenv("VOICE_RECOGNITION_MODEL", "small").strip() or "small",
        voice_recognition_language=os.getenv("VOICE_RECOGNITION_LANGUAGE", "zh").strip() or "zh",
        voice_recognition_beam_size=voice_recognition_beam_size,
        voice_recognition_silence_seconds=voice_recognition_silence_seconds,
        voice_recognition_max_segment_seconds=voice_recognition_max_segment_seconds,
        voice_recognition_initial_prompt=(
            os.getenv("VOICE_RECOGNITION_INITIAL_PROMPT", "以下是繁體中文 Discord 語音聊天的逐字稿。").strip()
        ),
        youtube_cookies_from_browser=youtube_cookies_from_browser,
        youtube_cookies_file=youtube_cookies_file,
        youtube_po_token=os.getenv("YOUTUBE_PO_TOKEN", "").strip() or None,
    )
