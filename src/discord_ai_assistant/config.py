from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from discord_ai_assistant.time_utils import resolve_timezone


DEFAULT_TTS_GEMINI_STYLE = (
    "Speak in natural Taiwan Mandarin with a youthful, soft female voice. "
    "Sound warm, relaxed, slightly playful, and conversational. "
    "Use natural pauses and subtle emotional variation. Avoid announcer-like delivery."
)


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
    tts_provider: str
    tts_gemini_model: str
    tts_gemini_voice: str
    tts_gemini_style: str
    tts_gemini_timeout_seconds: float
    tts_kokoro_enabled: bool
    tts_kokoro_voice: str
    tts_kokoro_speed: float
    tts_remote_workers: str
    tts_worker_token: str | None
    tts_worker_mode: str
    tts_worker_health_timeout_seconds: float
    tts_worker_request_timeout_seconds: float
    tts_worker_failure_cooldown_seconds: float
    youtube_cookies_from_browser: str | None
    youtube_cookies_file: Path | None
    youtube_po_token: str | None
    tool_gateway_enabled: bool
    tool_gateway_embedded: bool
    tool_gateway_host: str
    tool_gateway_port: int
    tool_gateway_token: str | None
    tool_gateway_refresh_seconds: float
    tool_gateway_timeout_seconds: float
    heartbeat_url: str | None
    heartbeat_token: str | None
    heartbeat_interval_seconds: float
    heartbeat_timeout_seconds: float

    @property
    def database_path(self) -> Path:
        return self.project_root / "data" / "assistant.sqlite3"

    @property
    def library_path(self) -> Path:
        return self.project_root / "library"

    @property
    def tools_path(self) -> Path:
        return self.project_root / "tools"


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

    tts_provider = os.getenv("TTS_PROVIDER", "auto").strip().lower() or "auto"
    if tts_provider not in {"auto", "gemini", "kokoro", "sapi"}:
        raise RuntimeError("TTS_PROVIDER must be one of: auto, gemini, kokoro, sapi")
    tts_gemini_timeout_seconds = float(os.getenv("TTS_GEMINI_TIMEOUT_SECONDS", "45"))
    if not 5 <= tts_gemini_timeout_seconds <= 120:
        raise RuntimeError("TTS_GEMINI_TIMEOUT_SECONDS must be between 5 and 120.")
    tts_kokoro_speed = float(os.getenv("TTS_KOKORO_SPEED", "1.0"))
    if not 0.5 <= tts_kokoro_speed <= 2.0:
        raise RuntimeError("TTS_KOKORO_SPEED must be between 0.5 and 2.0.")
    tts_worker_health_timeout_seconds = float(os.getenv("TTS_WORKER_HEALTH_TIMEOUT_SECONDS", "2"))
    tts_worker_request_timeout_seconds = float(os.getenv("TTS_WORKER_REQUEST_TIMEOUT_SECONDS", "45"))
    tts_worker_failure_cooldown_seconds = float(os.getenv("TTS_WORKER_FAILURE_COOLDOWN_SECONDS", "15"))
    if not 0.5 <= tts_worker_health_timeout_seconds <= 15:
        raise RuntimeError("TTS_WORKER_HEALTH_TIMEOUT_SECONDS must be between 0.5 and 15.")
    if not 5 <= tts_worker_request_timeout_seconds <= 180:
        raise RuntimeError("TTS_WORKER_REQUEST_TIMEOUT_SECONDS must be between 5 and 180.")
    if not 0 <= tts_worker_failure_cooldown_seconds <= 600:
        raise RuntimeError("TTS_WORKER_FAILURE_COOLDOWN_SECONDS must be between 0 and 600.")

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

    tool_gateway_host = os.getenv("TOOL_GATEWAY_HOST", "127.0.0.1").strip() or "127.0.0.1"
    tool_gateway_port = int(os.getenv("TOOL_GATEWAY_PORT", "8765"))
    tool_gateway_refresh_seconds = float(os.getenv("TOOL_GATEWAY_REFRESH_SECONDS", "30"))
    tool_gateway_timeout_seconds = float(os.getenv("TOOL_GATEWAY_TIMEOUT_SECONDS", "60"))
    if not 1 <= tool_gateway_port <= 65535:
        raise RuntimeError("TOOL_GATEWAY_PORT must be between 1 and 65535.")
    if tool_gateway_refresh_seconds < 1:
        raise RuntimeError("TOOL_GATEWAY_REFRESH_SECONDS must be at least 1.")
    if tool_gateway_timeout_seconds < 1:
        raise RuntimeError("TOOL_GATEWAY_TIMEOUT_SECONDS must be at least 1.")

    heartbeat_url = os.getenv("MOXUE_HEARTBEAT_URL", "").strip() or None
    heartbeat_token = os.getenv("MOXUE_HEARTBEAT_TOKEN", "").strip() or None
    heartbeat_interval_seconds = float(os.getenv("MOXUE_HEARTBEAT_INTERVAL_SECONDS", "600"))
    heartbeat_timeout_seconds = float(os.getenv("MOXUE_HEARTBEAT_TIMEOUT_SECONDS", "8"))
    if heartbeat_url and not heartbeat_token:
        raise RuntimeError("MOXUE_HEARTBEAT_TOKEN is required when MOXUE_HEARTBEAT_URL is set.")
    if heartbeat_url and not (
        heartbeat_url.startswith("https://")
        or heartbeat_url.startswith("http://127.0.0.1")
        or heartbeat_url.startswith("http://localhost")
    ):
        raise RuntimeError("MOXUE_HEARTBEAT_URL must use HTTPS, except for localhost development.")
    if not 15 <= heartbeat_interval_seconds <= 600:
        raise RuntimeError("MOXUE_HEARTBEAT_INTERVAL_SECONDS must be between 15 and 600.")
    if not 1 <= heartbeat_timeout_seconds <= 30:
        raise RuntimeError("MOXUE_HEARTBEAT_TIMEOUT_SECONDS must be between 1 and 30.")

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
        tts_provider=tts_provider,
        tts_gemini_model=(
            os.getenv("TTS_GEMINI_MODEL", "gemini-3.1-flash-tts-preview").strip()
            or "gemini-3.1-flash-tts-preview"
        ),
        tts_gemini_voice=os.getenv("TTS_GEMINI_VOICE", "Leda").strip() or "Leda",
        tts_gemini_style=os.getenv("TTS_GEMINI_STYLE", DEFAULT_TTS_GEMINI_STYLE).strip(),
        tts_gemini_timeout_seconds=tts_gemini_timeout_seconds,
        tts_kokoro_enabled=os.getenv("TTS_KOKORO_ENABLED", "true").strip().lower()
        in {"1", "true", "yes", "on"},
        tts_kokoro_voice=os.getenv("TTS_KOKORO_VOICE", "zf_xiaoni").strip() or "zf_xiaoni",
        tts_kokoro_speed=tts_kokoro_speed,
        tts_remote_workers=os.getenv("TTS_REMOTE_WORKERS", "").strip(),
        tts_worker_token=os.getenv("TTS_WORKER_TOKEN", "").strip() or None,
        tts_worker_mode=os.getenv("TTS_WORKER_MODE", "auto").strip().lower() or "auto",
        tts_worker_health_timeout_seconds=tts_worker_health_timeout_seconds,
        tts_worker_request_timeout_seconds=tts_worker_request_timeout_seconds,
        tts_worker_failure_cooldown_seconds=tts_worker_failure_cooldown_seconds,
        youtube_cookies_from_browser=youtube_cookies_from_browser,
        youtube_cookies_file=youtube_cookies_file,
        youtube_po_token=os.getenv("YOUTUBE_PO_TOKEN", "").strip() or None,
        tool_gateway_enabled=os.getenv("TOOL_GATEWAY_ENABLED", "true").strip().lower()
        in {"1", "true", "yes", "on"},
        tool_gateway_embedded=os.getenv("TOOL_GATEWAY_EMBEDDED", "true").strip().lower()
        in {"1", "true", "yes", "on"},
        tool_gateway_host=tool_gateway_host,
        tool_gateway_port=tool_gateway_port,
        tool_gateway_token=os.getenv("TOOL_GATEWAY_TOKEN", "").strip() or None,
        tool_gateway_refresh_seconds=tool_gateway_refresh_seconds,
        tool_gateway_timeout_seconds=tool_gateway_timeout_seconds,
        heartbeat_url=heartbeat_url,
        heartbeat_token=heartbeat_token,
        heartbeat_interval_seconds=heartbeat_interval_seconds,
        heartbeat_timeout_seconds=heartbeat_timeout_seconds,
    )
