from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


VALID_CANDIDATE_KINDS = {"asr_alias", "domain_term", "report_preference"}
_ARROW_RE = re.compile(
    r"^\s*(?P<source>[^\n→>-]{1,120}?)\s*(?:->|→)\s*(?P<target>[^|｜：:\n]{1,120}?)"
    r"(?:\s*(?:\||｜|：|:)\s*(?P<explanation>.+))?\s*$"
)


@dataclass(frozen=True, slots=True)
class FeedbackCandidate:
    kind: str
    source_text: str
    canonical_term: str
    explanation: str = ""
    aliases: tuple[str, ...] = ()
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class TrainingExample:
    audio_sha256: str
    start_seconds: float
    end_seconds: float
    original_text: str
    corrected_text: str
    source_alias: str
    canonical_term: str


def parse_explicit_corrections(text: str) -> list[FeedbackCandidate]:
    """Parse human-authored `wrong -> correct | explanation` lines conservatively."""
    result: list[FeedbackCandidate] = []
    seen: set[tuple[str, str]] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("-*• ").strip()
        match = _ARROW_RE.match(line)
        if not match:
            continue
        source = " ".join(match.group("source").split()).strip("`'\"")
        target = " ".join(match.group("target").split()).strip("`'\"")
        explanation = " ".join((match.group("explanation") or "").split())[:500]
        if not source or not target or source.casefold() == target.casefold():
            continue
        key = (source.casefold(), target.casefold())
        if key in seen:
            continue
        seen.add(key)
        result.append(
            FeedbackCandidate(
                kind="asr_alias",
                source_text=source,
                canonical_term=target,
                explanation=explanation,
                aliases=(source,),
                confidence=1.0,
            )
        )
        if len(result) >= 12:
            break
    return result


def parse_ai_candidates(text: str) -> list[FeedbackCandidate]:
    """Parse bounded JSON suggestions. These are always pending until a human approves them."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        return []
    try:
        payload = json.loads(cleaned[start : end + 1])
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    raw_candidates = payload.get("candidates", []) if isinstance(payload, dict) else []
    if not isinstance(raw_candidates, list):
        return []

    result: list[FeedbackCandidate] = []
    seen: set[tuple[str, str, str]] = set()
    for raw in raw_candidates[:12]:
        if not isinstance(raw, dict):
            continue
        kind = str(raw.get("kind", "")).strip()
        if kind not in VALID_CANDIDATE_KINDS:
            continue
        source = " ".join(str(raw.get("source_text", "")).split())[:120]
        canonical = " ".join(str(raw.get("canonical_term", "")).split())[:120]
        explanation = " ".join(str(raw.get("explanation", "")).split())[:500]
        aliases_raw = raw.get("aliases", [])
        if isinstance(aliases_raw, str):
            aliases_raw = re.split(r"[,，;；\n]+", aliases_raw)
        aliases: list[str] = []
        if isinstance(aliases_raw, list):
            alias_seen: set[str] = set()
            for item in aliases_raw:
                alias = " ".join(str(item).split())[:120]
                folded = alias.casefold()
                if alias and folded not in alias_seen and folded != canonical.casefold():
                    alias_seen.add(folded)
                    aliases.append(alias)
        if kind == "asr_alias":
            if not source or not canonical or source.casefold() == canonical.casefold():
                continue
            if source.casefold() not in {item.casefold() for item in aliases}:
                aliases.insert(0, source)
        elif kind == "domain_term":
            if not canonical or not explanation:
                continue
        elif kind == "report_preference":
            if not explanation:
                continue
            canonical = ""
            source = ""
            aliases = []
        try:
            confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.5))))
        except (TypeError, ValueError):
            confidence = 0.5
        key = (kind, source.casefold(), canonical.casefold() or explanation.casefold())
        if key in seen:
            continue
        seen.add(key)
        result.append(
            FeedbackCandidate(
                kind=kind,
                source_text=source,
                canonical_term=canonical,
                explanation=explanation,
                aliases=tuple(aliases),
                confidence=confidence,
            )
        )
    return result


def infer_partial_asr_candidates(
    original_report: str,
    revised_report: str,
    transcript: str,
    *,
    limit: int = 12,
) -> list[FeedbackCandidate]:
    """Surface short human replacements as opt-in ASR candidates, even if only one occurrence was fixed."""
    matcher = difflib.SequenceMatcher(None, original_report, revised_report, autojunk=False)
    result: list[FeedbackCandidate] = []
    seen: set[tuple[str, str]] = set()

    def clean(value: str) -> str:
        return value.strip().strip(" \\t\\r\\n`*_~'\"「」『』()（）[]【】<>《》-—")

    def term_like(value: str) -> bool:
        return bool(value) and len(value) <= 8 and "\n" not in value and bool(
            re.fullmatch(r"[0-9A-Za-z\u3400-\u9fff ._+-]+", value)
        )

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace":
            continue
        source = clean(original_report[i1:i2])
        target = clean(revised_report[j1:j2])

        # SequenceMatcher often isolates one changed CJK character (阿霧 -> 阿鳴
        # becomes 霧 -> 鳴). Include one shared neighboring CJK character so the
        # candidate stays specific enough for a human to approve safely.
        if len(source) == 1 and len(target) == 1 and i1 > 0 and j1 > 0:
            left_original = original_report[i1 - 1]
            left_revised = revised_report[j1 - 1]
            if left_original == left_revised and re.fullmatch(r"[\u3400-\u9fff]", left_original):
                source = left_original + source
                target = left_revised + target
        if len(source) == 1 and len(target) == 1 and i2 < len(original_report) and j2 < len(revised_report):
            right_original = original_report[i2]
            right_revised = revised_report[j2]
            if right_original == right_revised and re.fullmatch(r"[\u3400-\u9fff]", right_original):
                source += right_original
                target += right_revised

        source = clean(source)
        target = clean(target)
        if not term_like(source) or not term_like(target):
            continue
        if source.casefold() == target.casefold() or source not in transcript:
            continue
        key = (source.casefold(), target.casefold())
        if key in seen:
            continue
        seen.add(key)
        result.append(
            FeedbackCandidate(
                kind="asr_alias",
                source_text=source,
                canonical_term=target,
                explanation="人工完整修正版至少修正過一次；請人工確認是否為可重用的 ASR 錯詞對應。",
                aliases=(source,),
                confidence=0.75,
            )
        )
        if len(result) >= max(1, min(limit, 12)):
            break
    return result


def merge_candidates(*groups: Iterable[FeedbackCandidate]) -> list[FeedbackCandidate]:
    """Prefer explicit human mappings over AI suggestions for the same source/canonical pair."""
    result: list[FeedbackCandidate] = []
    seen: set[tuple[str, str, str]] = set()
    for group in groups:
        for item in group:
            key = (item.kind, item.source_text.casefold(), item.canonical_term.casefold())
            if item.kind == "report_preference":
                key = (item.kind, "", item.explanation.casefold())
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
            if len(result) >= 12:
                return result
    return result


class MeetingFeedbackStore:
    """Guild-scoped human review data used for safe terminology learning and future ASR tuning."""

    def __init__(self, database: Any, project_root: Path) -> None:
        self.database = database
        self.connection = database.connection
        self.project_root = project_root
        self._migrate()

    def _migrate(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS meeting_feedback_revisions (
                id INTEGER PRIMARY KEY,
                review_id TEXT NOT NULL,
                guild_id INTEGER NOT NULL,
                profile_key TEXT,
                audio_sha256 TEXT NOT NULL,
                title TEXT NOT NULL,
                correction_text TEXT NOT NULL,
                original_report TEXT NOT NULL,
                revised_report TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending',
                created_by INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_meeting_feedback_review
                ON meeting_feedback_revisions(guild_id, review_id, id DESC);

            CREATE TABLE IF NOT EXISTS meeting_feedback_candidates (
                id INTEGER PRIMARY KEY,
                revision_id INTEGER NOT NULL REFERENCES meeting_feedback_revisions(id) ON DELETE CASCADE,
                kind TEXT NOT NULL,
                source_text TEXT NOT NULL DEFAULT '',
                canonical_term TEXT NOT NULL DEFAULT '',
                explanation TEXT NOT NULL DEFAULT '',
                aliases_json TEXT NOT NULL DEFAULT '[]',
                confidence REAL NOT NULL DEFAULT 0.5,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_meeting_feedback_candidates_revision
                ON meeting_feedback_candidates(revision_id, status, id);

            CREATE TABLE IF NOT EXISTS meeting_asr_training_examples (
                id INTEGER PRIMARY KEY,
                revision_id INTEGER NOT NULL REFERENCES meeting_feedback_revisions(id) ON DELETE CASCADE,
                guild_id INTEGER NOT NULL,
                profile_key TEXT,
                audio_sha256 TEXT NOT NULL,
                start_seconds REAL NOT NULL,
                end_seconds REAL NOT NULL,
                original_text TEXT NOT NULL,
                corrected_text TEXT NOT NULL,
                source_alias TEXT NOT NULL,
                canonical_term TEXT NOT NULL,
                created_by INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(guild_id, audio_sha256, start_seconds, end_seconds, source_alias, canonical_term)
            );
            CREATE INDEX IF NOT EXISTS idx_meeting_asr_training_profile
                ON meeting_asr_training_examples(guild_id, profile_key, id);

            CREATE TABLE IF NOT EXISTS meeting_report_preferences (
                id INTEGER PRIMARY KEY,
                guild_id INTEGER NOT NULL,
                profile_key TEXT,
                instruction TEXT NOT NULL,
                source_revision_id INTEGER REFERENCES meeting_feedback_revisions(id) ON DELETE SET NULL,
                created_by INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(guild_id, profile_key, instruction)
            );

            CREATE TABLE IF NOT EXISTS meeting_report_examples (
                id INTEGER PRIMARY KEY,
                guild_id INTEGER NOT NULL,
                profile_key TEXT,
                review_id TEXT NOT NULL,
                title TEXT NOT NULL,
                report_text TEXT NOT NULL,
                approved_by INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(guild_id, review_id)
            );
            CREATE INDEX IF NOT EXISTS idx_meeting_report_examples_profile
                ON meeting_report_examples(guild_id, profile_key, id DESC);
            """
        )
        self.connection.commit()

    def create_revision(
        self,
        *,
        review_id: str,
        guild_id: int,
        profile_key: str | None,
        audio_sha256: str,
        title: str,
        correction_text: str,
        original_report: str,
        created_by: int,
    ) -> int:
        correction = correction_text.strip()
        if not correction:
            raise ValueError("修正內容不可為空。")
        cursor = self.connection.execute(
            """INSERT INTO meeting_feedback_revisions(
                   review_id, guild_id, profile_key, audio_sha256, title,
                   correction_text, original_report, created_by
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                review_id,
                guild_id,
                profile_key,
                audio_sha256,
                " ".join(title.split())[:160],
                correction[:12000],
                original_report[:60000],
                created_by,
            ),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def update_revision(self, revision_id: int, *, status: str | None = None, revised_report: str | None = None) -> None:
        updates: list[str] = ["updated_at = CURRENT_TIMESTAMP"]
        values: list[object] = []
        if status is not None:
            updates.append("status = ?")
            values.append(status[:40])
        if revised_report is not None:
            updates.append("revised_report = ?")
            values.append(revised_report[:60000])
        values.append(revision_id)
        self.connection.execute(
            f"UPDATE meeting_feedback_revisions SET {', '.join(updates)} WHERE id = ?", values
        )
        self.connection.commit()

    def add_candidates(self, revision_id: int, candidates: Iterable[FeedbackCandidate]) -> list[int]:
        ids: list[int] = []
        for item in candidates:
            cursor = self.connection.execute(
                """INSERT INTO meeting_feedback_candidates(
                       revision_id, kind, source_text, canonical_term, explanation,
                       aliases_json, confidence, status
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')""",
                (
                    revision_id,
                    item.kind,
                    item.source_text,
                    item.canonical_term,
                    item.explanation,
                    json.dumps(item.aliases, ensure_ascii=False),
                    item.confidence,
                ),
            )
            ids.append(int(cursor.lastrowid))
        self.connection.commit()
        return ids

    def candidates(self, revision_id: int, status: str | None = None) -> list[tuple[int, FeedbackCandidate]]:
        if status is None:
            rows = self.connection.execute(
                "SELECT * FROM meeting_feedback_candidates WHERE revision_id = ? ORDER BY id",
                (revision_id,),
            ).fetchall()
        else:
            rows = self.connection.execute(
                """SELECT * FROM meeting_feedback_candidates
                   WHERE revision_id = ? AND status = ? ORDER BY id""",
                (revision_id, status),
            ).fetchall()
        result: list[tuple[int, FeedbackCandidate]] = []
        for row in rows:
            try:
                aliases = tuple(str(item) for item in json.loads(str(row["aliases_json"])))
            except (TypeError, ValueError, json.JSONDecodeError):
                aliases = ()
            result.append(
                (
                    int(row["id"]),
                    FeedbackCandidate(
                        kind=str(row["kind"]),
                        source_text=str(row["source_text"]),
                        canonical_term=str(row["canonical_term"]),
                        explanation=str(row["explanation"]),
                        aliases=aliases,
                        confidence=float(row["confidence"]),
                    ),
                )
            )
        return result

    def select_candidates(self, revision_id: int, selected_candidate_ids: Iterable[int]) -> None:
        """Persist the first review decision before learning or regeneration can fail."""
        ids = sorted({int(item) for item in selected_candidate_ids})
        placeholders = ",".join("?" for _ in ids) or "NULL"
        self.connection.execute(
            f"""UPDATE meeting_feedback_candidates
                SET status = CASE WHEN id IN ({placeholders}) THEN 'approved' ELSE 'skipped' END
                WHERE revision_id = ? AND status = 'pending'""",
            (*ids, revision_id),
        )
        self.connection.commit()

    def set_candidate_status(self, candidate_ids: Iterable[int], status: str) -> None:
        ids = [int(item) for item in candidate_ids]
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        self.connection.execute(
            f"UPDATE meeting_feedback_candidates SET status = ? WHERE id IN ({placeholders})",
            (status[:40], *ids),
        )
        self.connection.commit()

    def add_report_preference(
        self,
        *,
        guild_id: int,
        profile_key: str | None,
        instruction: str,
        revision_id: int,
        created_by: int,
    ) -> None:
        normalized = " ".join(instruction.split())[:500]
        if not normalized:
            return
        self.connection.execute(
            """INSERT OR IGNORE INTO meeting_report_preferences(
                   guild_id, profile_key, instruction, source_revision_id, created_by
               ) VALUES (?, ?, ?, ?, ?)""",
            (guild_id, profile_key, normalized, revision_id, created_by),
        )
        self.connection.commit()

    def report_preferences(self, guild_id: int, profile_key: str | None, limit: int = 12) -> list[str]:
        rows = self.connection.execute(
            """SELECT instruction FROM meeting_report_preferences
               WHERE guild_id = ? AND (profile_key = ? OR (profile_key IS NULL AND ? IS NULL))
               ORDER BY id DESC LIMIT ?""",
            (guild_id, profile_key, profile_key, max(1, min(limit, 20))),
        ).fetchall()
        return [str(row["instruction"]) for row in reversed(rows)]

    def record_asr_examples(
        self,
        *,
        revision_id: int,
        guild_id: int,
        profile_key: str | None,
        audio_sha256: str,
        segments: Iterable[dict[str, object]],
        candidates: Iterable[FeedbackCandidate],
        created_by: int,
    ) -> int:
        inserted = 0
        aliases: list[tuple[str, str]] = []
        for candidate in candidates:
            if candidate.kind != "asr_alias":
                continue
            for alias in candidate.aliases or ((candidate.source_text,) if candidate.source_text else ()):
                if alias and candidate.canonical_term and alias != candidate.canonical_term:
                    aliases.append((alias, candidate.canonical_term))
        for segment in segments:
            original = str(segment.get("text", "")).strip()
            if not original:
                continue
            for source, canonical in aliases:
                if source not in original:
                    continue
                corrected = original.replace(source, canonical)
                cursor = self.connection.execute(
                    """INSERT OR IGNORE INTO meeting_asr_training_examples(
                           revision_id, guild_id, profile_key, audio_sha256,
                           start_seconds, end_seconds, original_text, corrected_text,
                           source_alias, canonical_term, created_by
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        revision_id,
                        guild_id,
                        profile_key,
                        audio_sha256,
                        float(segment.get("start", 0.0)),
                        float(segment.get("end", 0.0)),
                        original,
                        corrected,
                        source,
                        canonical,
                        created_by,
                    ),
                )
                if cursor.rowcount > 0:
                    inserted += 1
        self.connection.commit()
        return inserted

    def add_report_example(
        self,
        *,
        guild_id: int,
        profile_key: str | None,
        review_id: str,
        title: str,
        report_text: str,
        approved_by: int,
    ) -> None:
        self.connection.execute(
            """INSERT INTO meeting_report_examples(
                   guild_id, profile_key, review_id, title, report_text, approved_by
               ) VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(guild_id, review_id) DO UPDATE SET
                 profile_key = excluded.profile_key,
                 title = excluded.title,
                 report_text = excluded.report_text,
                 approved_by = excluded.approved_by""",
            (guild_id, profile_key, review_id, title[:160], report_text[:60000], approved_by),
        )
        self.connection.commit()

    def recent_report_examples(self, guild_id: int, profile_key: str | None, limit: int = 3) -> list[str]:
        rows = self.connection.execute(
            """SELECT report_text FROM meeting_report_examples
               WHERE guild_id = ? AND (profile_key = ? OR (profile_key IS NULL AND ? IS NULL))
               ORDER BY id DESC LIMIT ?""",
            (guild_id, profile_key, profile_key, max(1, min(limit, 5))),
        ).fetchall()
        return [str(row["report_text"])[:10000] for row in reversed(rows)]

    def training_stats(self, guild_id: int, profile_key: str | None) -> dict[str, int]:
        row = self.connection.execute(
            """SELECT COUNT(*) AS examples, COUNT(DISTINCT audio_sha256) AS recordings
               FROM meeting_asr_training_examples
               WHERE guild_id = ? AND (profile_key = ? OR (profile_key IS NULL AND ? IS NULL))""",
            (guild_id, profile_key, profile_key),
        ).fetchone()
        return {
            "examples": int(row["examples"]) if row else 0,
            "recordings": int(row["recordings"]) if row else 0,
        }

    def export_training_manifest(self, guild_id: int, profile_key: str | None) -> tuple[Path, int]:
        rows = self.connection.execute(
            """SELECT * FROM meeting_asr_training_examples
               WHERE guild_id = ? AND (profile_key = ? OR (profile_key IS NULL AND ? IS NULL))
               ORDER BY id""",
            (guild_id, profile_key, profile_key),
        ).fetchall()
        safe_profile = re.sub(r"[^0-9A-Za-z._-]+", "-", profile_key or "general").strip("-._") or "general"
        root = self.project_root / "data" / "game-meeting-recorder" / "training" / str(guild_id) / safe_profile
        root.mkdir(parents=True, exist_ok=True)
        path = root / "whisper_feedback_manifest.jsonl"
        lines: list[str] = []
        imports_root = self.project_root / "data" / "game-meeting-recorder" / "imports"
        for row in rows:
            digest = str(row["audio_sha256"])
            audio_candidates = sorted((imports_root / digest).glob("original.*"))
            audio_path = audio_candidates[0] if audio_candidates else None
            payload = {
                "audio_sha256": digest,
                "audio_path": str(audio_path) if audio_path else None,
                "start_seconds": float(row["start_seconds"]),
                "end_seconds": float(row["end_seconds"]),
                "original_text": str(row["original_text"]),
                "corrected_text": str(row["corrected_text"]),
                "source_alias": str(row["source_alias"]),
                "canonical_term": str(row["canonical_term"]),
            }
            lines.append(json.dumps(payload, ensure_ascii=False))
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return path, len(lines)
