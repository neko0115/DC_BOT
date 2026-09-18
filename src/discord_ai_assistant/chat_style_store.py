from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from discord_ai_assistant.ai.chat_style import ChatStyleSample, prepare_chat_style_content


UTC = timezone.utc
SAMPLE_TTL = timedelta(hours=72)
SAMPLE_LIMIT = 80
INITIAL_SAMPLE_THRESHOLD = 30
INCREMENTAL_SAMPLE_THRESHOLD = 24
PROFILE_UPDATE_COOLDOWN = timedelta(hours=8)
PROFILE_UPDATE_ROLLING_WINDOW = timedelta(hours=24)
PROFILE_UPDATE_ROLLING_LIMIT = 2


@dataclass(frozen=True, slots=True)
class ChatStyleProfileRecord:
    guild_id: int
    user_id: int
    profile: dict[str, Any]
    sample_watermark_id: int
    effective_sample_count: int
    last_successful_update_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class ProfileUpdateEligibility:
    eligible: bool
    reason: str
    effective_sample_count: int
    new_sample_count: int


@dataclass(frozen=True, slots=True)
class PersonalChatTerm:
    guild_id: int
    user_id: int
    term: str
    meaning: str
    confidence: float
    occurrence_count: int


def normalize_term_text(value: str) -> str:
    """Exact consistency only; no semantic or fuzzy reconciliation."""
    return " ".join(value.split()).casefold()


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _iso(value: datetime) -> str:
    return _as_utc(value).isoformat(timespec="microseconds")


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(UTC)


class ChatStyleStore:
    """Guild/user-scoped Chat Style persistence kept outside Memory V2 tables."""

    def __init__(self, database: Any) -> None:
        self.database = database
        self.connection = database.connection
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS chat_style_profiles (
                guild_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                profile_json TEXT NOT NULL,
                sample_watermark_id INTEGER NOT NULL DEFAULT 0,
                effective_sample_count INTEGER NOT NULL DEFAULT 0,
                last_successful_update_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (guild_id, user_id)
            );

            CREATE TABLE IF NOT EXISTS chat_style_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                guild_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                channel_id INTEGER NOT NULL,
                message_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                session_key TEXT,
                is_reply INTEGER NOT NULL DEFAULT 0,
                is_public_evidence INTEGER NOT NULL DEFAULT 0,
                UNIQUE (guild_id, message_id)
            );

            CREATE INDEX IF NOT EXISTS idx_chat_style_samples_owner_time
                ON chat_style_samples(guild_id, user_id, observed_at, id);

            CREATE TABLE IF NOT EXISTS chat_style_profile_updates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                guild_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                sample_watermark_id INTEGER NOT NULL,
                completed_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_chat_style_profile_updates_owner_time
                ON chat_style_profile_updates(guild_id, user_id, completed_at);

            CREATE TABLE IF NOT EXISTS chat_style_learning (
                guild_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                enabled INTEGER NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (guild_id, user_id)
            );

            CREATE TABLE IF NOT EXISTS chat_style_terms (
                guild_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                term TEXT NOT NULL,
                meaning TEXT NOT NULL,
                confidence REAL NOT NULL,
                occurrence_count INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'active' CHECK(status = 'active'),
                updated_at TEXT NOT NULL,
                PRIMARY KEY (guild_id, user_id, term)
            );
            """
        )
        columns = {row[1] for row in self.connection.execute("PRAGMA table_info(chat_style_samples)")}
        if "is_public_evidence" not in columns:
            self.connection.execute(
                "ALTER TABLE chat_style_samples ADD COLUMN is_public_evidence INTEGER NOT NULL DEFAULT 0"
            )
        self.connection.commit()

    def learning_enabled(self, guild_id: int, user_id: int) -> bool:
        row = self.connection.execute(
            "SELECT enabled FROM chat_style_learning WHERE guild_id = ? AND user_id = ?",
            (guild_id, user_id),
        ).fetchone()
        # Unknown/corrupt persisted values must not grant private learning access.
        return True if row is None else row[0] == 1

    def set_learning_enabled(
        self,
        guild_id: int,
        user_id: int,
        enabled: bool,
        *,
        updated_at: datetime,
    ) -> None:
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO chat_style_learning(guild_id, user_id, enabled, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(guild_id, user_id) DO UPDATE SET
                    enabled = excluded.enabled,
                    updated_at = excluded.updated_at
                """,
                (guild_id, user_id, int(enabled), _iso(updated_at)),
            )

    def record_sample(
        self,
        *,
        guild_id: int,
        user_id: int,
        channel_id: int,
        message_id: int,
        content: str,
        observed_at: datetime,
        session_key: str | None,
        is_reply: bool,
        is_bot: bool,
        is_dm: bool,
        has_stickers: bool,
        is_system: bool,
        is_public_evidence: bool = False,
        source_content: str | None = None,
    ) -> int | None:
        if not self.learning_enabled(guild_id, user_id):
            return None
        filtered = prepare_chat_style_content(
            content,
            source_content=source_content,
            is_bot=is_bot,
            is_dm=is_dm,
            has_stickers=has_stickers,
            is_system=is_system,
        )
        if filtered is None:
            return None

        observed_at = _as_utc(observed_at)
        with self.connection:
            cursor = self.connection.execute(
                """
                INSERT OR IGNORE INTO chat_style_samples(
                    guild_id, user_id, channel_id, message_id, content,
                    observed_at, session_key, is_reply, is_public_evidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    guild_id,
                    user_id,
                    channel_id,
                    message_id,
                    filtered,
                    _iso(observed_at),
                    session_key,
                    int(is_reply),
                    int(is_public_evidence is True),
                ),
            )
            if cursor.rowcount == 0:
                return None
            sample_id = int(cursor.lastrowid)
            self._prune_owner(guild_id, user_id, now=observed_at)
            return sample_id

    def _prune_owner(self, guild_id: int, user_id: int, *, now: datetime) -> None:
        cutoff = _iso(_as_utc(now) - SAMPLE_TTL)
        self.connection.execute(
            """
            DELETE FROM chat_style_samples
            WHERE guild_id = ? AND user_id = ? AND observed_at < ?
            """,
            (guild_id, user_id, cutoff),
        )
        self.connection.execute(
            """
            DELETE FROM chat_style_samples
            WHERE guild_id = ? AND user_id = ? AND id NOT IN (
                SELECT id FROM chat_style_samples
                WHERE guild_id = ? AND user_id = ?
                ORDER BY observed_at DESC, id DESC
                LIMIT ?
            )
            """,
            (guild_id, user_id, guild_id, user_id, SAMPLE_LIMIT),
        )

    def list_samples(
        self,
        guild_id: int,
        user_id: int,
        *,
        now: datetime,
    ) -> list[ChatStyleSample]:
        now = _as_utc(now)
        with self.connection:
            self._prune_owner(guild_id, user_id, now=now)
        rows = self.connection.execute(
            """
            SELECT id, guild_id, user_id, channel_id, message_id, content,
                   observed_at, session_key, is_reply, is_public_evidence
            FROM chat_style_samples
            WHERE guild_id = ? AND user_id = ? AND observed_at <= ?
            ORDER BY observed_at ASC, id ASC
            """,
            (guild_id, user_id, _iso(now)),
        ).fetchall()
        return [
            ChatStyleSample(
                id=int(row[0]),
                guild_id=int(row[1]),
                user_id=int(row[2]),
                channel_id=int(row[3]),
                message_id=int(row[4]),
                content=str(row[5]),
                observed_at=_parse_timestamp(str(row[6])),
                session_key=str(row[7]) if row[7] is not None else None,
                is_reply=bool(row[8]),
                is_public_evidence=bool(row[9]),
            )
            for row in rows
        ]

    def public_term_samples(
        self, guild_id: int, user_id: int, term: str,
        samples: list[ChatStyleSample], *, now: datetime,
    ) -> list[ChatStyleSample]:
        """Revalidate source IDs against retained, public, owner-scoped rows."""
        if not self.learning_enabled(guild_id, user_id):
            return []
        term = normalize_term_text(term)
        if not term:
            return []
        source_ids = {sample.id for sample in samples}
        # Latin word boundaries avoid counting e.g. 'hop' inside 'shopping'.
        pattern = re.compile(r"(?<![a-z0-9_])" + re.escape(term) + r"(?![a-z0-9_])")
        return [sample for sample in self.list_samples(guild_id, user_id, now=now)
                if sample.id in source_ids and sample.is_public_evidence
                and pattern.search(normalize_term_text(sample.content))]

    def register_personal_term(
        self, guild_id: int, user_id: int, *, term: str, meaning: str,
        confidence: float, samples: list[ChatStyleSample], now: datetime,
    ) -> PersonalChatTerm | None:
        valid = self.public_term_samples(guild_id, user_id, term, samples, now=now)
        if len(valid) < 3:
            return None
        term, meaning = normalize_term_text(term), normalize_term_text(meaning)
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO chat_style_terms
                    (guild_id, user_id, term, meaning, confidence, occurrence_count, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(guild_id, user_id, term) DO UPDATE SET
                    meaning = excluded.meaning, confidence = excluded.confidence,
                    occurrence_count = excluded.occurrence_count, updated_at = excluded.updated_at
                """,
                (guild_id, user_id, term, meaning, confidence, len(valid), _iso(now)),
            )
        return self.get_personal_term(guild_id, user_id, term)

    def get_personal_term(self, guild_id: int, user_id: int, term: str) -> PersonalChatTerm | None:
        row = self.connection.execute(
            "SELECT guild_id, user_id, term, meaning, confidence, occurrence_count "
            "FROM chat_style_terms WHERE guild_id = ? AND user_id = ? AND term = ?",
            (guild_id, user_id, normalize_term_text(term)),
        ).fetchone()
        return None if row is None else PersonalChatTerm(*row)

    def latest_sample_id(
        self,
        guild_id: int,
        user_id: int,
        *,
        now: datetime,
    ) -> int | None:
        now = _as_utc(now)
        with self.connection:
            self._prune_owner(guild_id, user_id, now=now)
        row = self.connection.execute(
            """
            SELECT id FROM chat_style_samples
            WHERE guild_id = ? AND user_id = ? AND observed_at <= ?
            ORDER BY observed_at DESC, id DESC LIMIT 1
            """,
            (guild_id, user_id, _iso(now)),
        ).fetchone()
        return None if row is None else int(row[0])

    def get_profile(self, guild_id: int, user_id: int) -> ChatStyleProfileRecord | None:
        row = self.connection.execute(
            """
            SELECT guild_id, user_id, profile_json, sample_watermark_id,
                   effective_sample_count, last_successful_update_at, updated_at
            FROM chat_style_profiles
            WHERE guild_id = ? AND user_id = ?
            """,
            (guild_id, user_id),
        ).fetchone()
        if row is None:
            return None
        try:
            profile = json.loads(str(row[2]))
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(profile, dict):
            return None
        return ChatStyleProfileRecord(
            guild_id=int(row[0]),
            user_id=int(row[1]),
            profile=profile,
            sample_watermark_id=int(row[3]),
            effective_sample_count=int(row[4]),
            last_successful_update_at=_parse_timestamp(str(row[5])),
            updated_at=_parse_timestamp(str(row[6])),
        )

    def commit_profile(
        self,
        guild_id: int,
        user_id: int,
        profile: dict[str, Any],
        *,
        sample_watermark_id: int,
        updated_at: datetime,
        sample_cutoff: datetime | None = None,
    ) -> None:
        if not isinstance(profile, dict):
            raise ValueError("chat style profile must be an object")
        updated_at = _as_utc(updated_at)
        sample_cutoff = updated_at if sample_cutoff is None else _as_utc(sample_cutoff)
        payload = json.dumps(profile, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self.connection:
            self._prune_owner(guild_id, user_id, now=sample_cutoff)
            row = self.connection.execute(
                """
                SELECT COUNT(*) FROM chat_style_samples
                WHERE guild_id = ? AND user_id = ? AND observed_at <= ? AND id <= ?
                """,
                (guild_id, user_id, _iso(sample_cutoff), int(sample_watermark_id)),
            ).fetchone()
            effective_count = int(row[0]) if row is not None else 0
            timestamp = _iso(updated_at)
            self.connection.execute(
                """
                INSERT INTO chat_style_profiles(
                    guild_id, user_id, profile_json, sample_watermark_id,
                    effective_sample_count, last_successful_update_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(guild_id, user_id) DO UPDATE SET
                    profile_json = excluded.profile_json,
                    sample_watermark_id = excluded.sample_watermark_id,
                    effective_sample_count = excluded.effective_sample_count,
                    last_successful_update_at = excluded.last_successful_update_at,
                    updated_at = excluded.updated_at
                """,
                (
                    guild_id,
                    user_id,
                    payload,
                    int(sample_watermark_id),
                    effective_count,
                    timestamp,
                    timestamp,
                ),
            )
            self.connection.execute(
                """
                INSERT INTO chat_style_profile_updates(
                    guild_id, user_id, sample_watermark_id, completed_at
                ) VALUES (?, ?, ?, ?)
                """,
                (guild_id, user_id, int(sample_watermark_id), timestamp),
            )

    def profile_update_eligibility(
        self,
        guild_id: int,
        user_id: int,
        *,
        now: datetime,
    ) -> ProfileUpdateEligibility:
        if not self.learning_enabled(guild_id, user_id):
            return ProfileUpdateEligibility(False, "learning_disabled", 0, 0)

        now = _as_utc(now)
        with self.connection:
            self._prune_owner(guild_id, user_id, now=now)
        now_iso = _iso(now)
        count_row = self.connection.execute(
            """
            SELECT COUNT(*) FROM chat_style_samples
            WHERE guild_id = ? AND user_id = ? AND observed_at <= ?
            """,
            (guild_id, user_id, now_iso),
        ).fetchone()
        effective_count = int(count_row[0]) if count_row is not None else 0
        profile = self.get_profile(guild_id, user_id)
        if profile is None:
            if effective_count >= INITIAL_SAMPLE_THRESHOLD:
                return ProfileUpdateEligibility(True, "initial_threshold", effective_count, effective_count)
            return ProfileUpdateEligibility(False, "insufficient_samples", effective_count, effective_count)

        new_row = self.connection.execute(
            """
            SELECT COUNT(*) FROM chat_style_samples
            WHERE guild_id = ? AND user_id = ? AND id > ? AND observed_at <= ?
            """,
            (guild_id, user_id, profile.sample_watermark_id, now_iso),
        ).fetchone()
        new_count = int(new_row[0]) if new_row is not None else 0
        if new_count < INCREMENTAL_SAMPLE_THRESHOLD:
            return ProfileUpdateEligibility(False, "insufficient_new_samples", effective_count, new_count)
        if now - profile.last_successful_update_at < PROFILE_UPDATE_COOLDOWN:
            return ProfileUpdateEligibility(False, "cooldown", effective_count, new_count)

        cutoff = _iso(now - PROFILE_UPDATE_ROLLING_WINDOW)
        update_row = self.connection.execute(
            """
            SELECT COUNT(*) FROM chat_style_profile_updates
            WHERE guild_id = ? AND user_id = ?
              AND completed_at > ? AND completed_at <= ?
            """,
            (guild_id, user_id, cutoff, now_iso),
        ).fetchone()
        update_count = int(update_row[0]) if update_row is not None else 0
        if update_count >= PROFILE_UPDATE_ROLLING_LIMIT:
            return ProfileUpdateEligibility(False, "daily_budget", effective_count, new_count)
        return ProfileUpdateEligibility(True, "incremental_threshold", effective_count, new_count)
