from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from hashlib import sha256
from typing import Any, Iterable

from discord_ai_assistant.ai.memory_retention import (
    RETENTION_LONG,
    RETENTION_MEDIUM,
    RETENTION_SHARED,
    expires_at_for_retention,
)
from discord_ai_assistant.ai.memory_social_policy import is_shareable_guild_memory
from discord_ai_assistant.ai.memory_v2 import memory_terms, normalize_memory_text, score_memory_candidate


_EVENT_COMPARE_DELETE = str.maketrans("", "", " \t\r\n，。,:：；;、!?！？（）()[]【】")
_EVENT_NEGATIVE_STATE_MARKERS = (
    "無法",
    "不能",
    "過不了",
    "沒辦法",
    "失效",
    "停擺",
    "停了",
    "壞掉",
    "損壞",
    "被拆",
    "遭拆",
    "拆除",
    "炸掉",
    "炸毀",
    "摧毀",
)
_EVENT_POSITIVE_STATE_MARKERS = (
    "可以正常",
    "可以過",
    "能夠過",
    "能過",
    "恢復通行",
    "正常通行",
    "恢復正常",
    "修復完成",
    "修好",
    "修復好",
    "可以使用",
    "能正常使用",
)
_EVENT_SIGNATURE_PREFIX = "v1:"
_EVENT_SIGNATURE_HASH_LENGTH = 16
_EVENT_SIGNATURE_MAX_TERMS = 240
_EVENT_SIGNATURE_MIN_JACCARD = 0.45
_EVENT_SIGNATURE_MIN_CONTAINMENT = 0.65
_SAME_SESSION_REFERENCE_MIN_SHARED_HASHES = 8
_SAME_SESSION_REFERENCE_MIN_CONTAINMENT = 0.30
_SAME_SESSION_REFERENCE_MIN_SEQUENCE = 0.60


@dataclass(frozen=True, slots=True)
class GuildMemoryMatch:
    id: int
    content: str
    domain: str
    subdomain: str | None
    kind: str
    entity_type: str | None
    entity: str | None
    score: float
    retention: str
    reinforcement_count: int


def ensure_shared_memory_schema(database: Any) -> None:
    """Create the guild-owned shared-memory and evidence tables in place."""

    connection = getattr(database, "connection", None)
    if connection is None:
        return
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS guild_memories (
            id INTEGER PRIMARY KEY,
            guild_id INTEGER NOT NULL,
            domain TEXT NOT NULL,
            subdomain TEXT,
            memory_kind TEXT NOT NULL,
            entity_type TEXT,
            entity TEXT,
            content TEXT NOT NULL,
            normalized_content TEXT NOT NULL,
            importance INTEGER NOT NULL DEFAULT 1,
            confidence REAL NOT NULL DEFAULT 0.0,
            retention TEXT NOT NULL DEFAULT 'shared',
            expires_at TEXT,
            reinforcement_count INTEGER NOT NULL DEFAULT 0,
            last_reinforced TEXT,
            status TEXT NOT NULL DEFAULT 'candidate',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_guild_memories_lookup
            ON guild_memories(guild_id, status, domain, subdomain, id DESC);
        CREATE INDEX IF NOT EXISTS idx_guild_memories_expiry
            ON guild_memories(guild_id, status, expires_at);

        CREATE TABLE IF NOT EXISTS guild_memory_evidence (
            id INTEGER PRIMARY KEY,
            guild_memory_id INTEGER NOT NULL REFERENCES guild_memories(id) ON DELETE CASCADE,
            guild_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            channel_id INTEGER NOT NULL,
            message_id INTEGER NOT NULL,
            observed_at TEXT,
            session_key TEXT,
            evidence_kind TEXT NOT NULL DEFAULT 'passive',
            event_signature TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(guild_memory_id, message_id)
        );
        CREATE INDEX IF NOT EXISTS idx_guild_memory_evidence_owner
            ON guild_memory_evidence(guild_id, guild_memory_id, user_id, id DESC);
        CREATE INDEX IF NOT EXISTS idx_guild_memory_evidence_session
            ON guild_memory_evidence(guild_id, guild_memory_id, session_key);
        """
    )
    evidence_columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(guild_memory_evidence)").fetchall()
    }
    if "event_signature" not in evidence_columns:
        connection.execute(
            "ALTER TABLE guild_memory_evidence ADD COLUMN event_signature TEXT"
        )
    # Development databases created by an earlier Social Rebalance checkpoint may
    # already contain candidates without an expiry. Backfill them conservatively from
    # their creation time so a one-off observation cannot remain promotable forever.
    connection.execute(
        """UPDATE guild_memories
           SET expires_at = strftime('%Y-%m-%dT%H:%M:%S+00:00', created_at, '+30 days')
           WHERE status = 'candidate' AND expires_at IS NULL"""
    )
    connection.commit()


def _utc_datetime(value: str | None) -> datetime:
    if value:
        candidate = value.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            parsed = datetime.now(timezone.utc)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return datetime.now(timezone.utc)


def _normalized_optional(value: str | None) -> str | None:
    normalized = normalize_memory_text(value or "")
    return normalized.casefold() or None


def build_event_signature(content: str) -> str | None:
    """Return a bounded non-raw lexical signature for one original observation.

    Each term is hashed independently so evidence can compare overlap without storing
    the Discord sentence itself. This is a privacy-minimizing fingerprint, not
    encryption; low-entropy term hashes must still be treated as internal metadata.
    """

    terms = sorted(set(memory_terms(content)), key=lambda item: (-len(item), item))
    if not terms:
        return None
    hashes = sorted(
        {
            sha256(term.encode("utf-8")).hexdigest()[:_EVENT_SIGNATURE_HASH_LENGTH]
            for term in terms[:_EVENT_SIGNATURE_MAX_TERMS]
        }
    )
    if not hashes:
        return None
    return _EVENT_SIGNATURE_PREFIX + ",".join(hashes)


def _event_signature_hashes(value: str | None) -> set[str]:
    if not value or not value.startswith(_EVENT_SIGNATURE_PREFIX):
        return set()
    payload = value[len(_EVENT_SIGNATURE_PREFIX) :]
    if not payload:
        return set()
    parts = payload.split(",")
    if len(parts) > _EVENT_SIGNATURE_MAX_TERMS:
        return set()
    valid_hex = set("0123456789abcdef")
    if any(
        len(part) != _EVENT_SIGNATURE_HASH_LENGTH
        or any(character not in valid_hex for character in part)
        for part in parts
    ):
        return set()
    return set(parts)


def _normalize_event_signature(value: str | None) -> str | None:
    hashes = _event_signature_hashes(value)
    if not hashes:
        return None
    return _EVENT_SIGNATURE_PREFIX + ",".join(sorted(hashes))


def _event_signature_overlap(left: str | None, right: str | None) -> tuple[float, float]:
    left_hashes = _event_signature_hashes(left)
    right_hashes = _event_signature_hashes(right)
    if not left_hashes or not right_hashes:
        return 0.0, 0.0
    intersection = len(left_hashes & right_hashes)
    union = len(left_hashes | right_hashes)
    smaller = min(len(left_hashes), len(right_hashes))
    return (
        intersection / union if union else 0.0,
        intersection / smaller if smaller else 0.0,
    )


def _event_signature_reconciliation_score(
    connection: Any,
    memory_id: int,
    incoming_signature: str | None,
) -> float:
    incoming = _normalize_event_signature(incoming_signature)
    if not incoming:
        return 0.0
    best = 0.0
    rows = connection.execute(
        """SELECT event_signature FROM guild_memory_evidence
           WHERE guild_memory_id = ? AND event_signature IS NOT NULL
           ORDER BY id DESC LIMIT 16""",
        (int(memory_id),),
    ).fetchall()
    for row in rows:
        jaccard, containment = _event_signature_overlap(
            str(row["event_signature"] or ""), incoming
        )
        if (
            jaccard >= _EVENT_SIGNATURE_MIN_JACCARD
            and containment >= _EVENT_SIGNATURE_MIN_CONTAINMENT
        ):
            best = max(best, 0.65)
    return best


def _term_overlap(left: str, right: str) -> tuple[float, float]:
    left_terms = set(memory_terms(left))
    right_terms = set(memory_terms(right))
    if not left_terms or not right_terms:
        return 0.0, 0.0
    intersection = len(left_terms & right_terms)
    union = len(left_terms | right_terms)
    smaller = min(len(left_terms), len(right_terms))
    jaccard = intersection / union if union else 0.0
    containment = intersection / smaller if smaller else 0.0
    return jaccard, containment


def _comparison_text(value: str) -> str:
    normalized = normalize_memory_text(value).casefold()
    if normalized.startswith("專案 "):
        normalized = normalized.removeprefix("專案 ")
    elif normalized.startswith("專案"):
        normalized = normalized[2:].lstrip("：: ")
    return normalized.translate(_EVENT_COMPARE_DELETE)


def _has_obvious_state_conflict(left: str, right: str) -> bool:
    left_text = normalize_memory_text(left)
    right_text = normalize_memory_text(right)
    left_negative = any(marker in left_text for marker in _EVENT_NEGATIVE_STATE_MARKERS)
    right_negative = any(marker in right_text for marker in _EVENT_NEGATIVE_STATE_MARKERS)
    left_positive = any(marker in left_text for marker in _EVENT_POSITIVE_STATE_MARKERS)
    right_positive = any(marker in right_text for marker in _EVENT_POSITIVE_STATE_MARKERS)
    return bool(
        (left_negative and right_positive and not right_negative)
        or (right_negative and left_positive and not left_negative)
    )


def _paraphrase_similarity(left: str, right: str, *, containment: float) -> float:
    """Recognize close CJK paraphrases without broadly lowering the lexical threshold."""

    if _has_obvious_state_conflict(left, right):
        return 0.0
    left_compare = _comparison_text(left)
    right_compare = _comparison_text(right)
    if not left_compare or not right_compare:
        return 0.0
    matcher = SequenceMatcher(None, left_compare, right_compare, autojunk=False)
    sequence = matcher.ratio()
    if containment >= 0.55 and sequence >= 0.78:
        return sequence

    # Gemini can preserve the same event anchors while rewriting the connective text
    # enough to make both global overlap scores look weak. Accept that case only when
    # there are several separated, substantial matching fragments rather than by
    # lowering the global threshold for every same-domain episode.
    strong_blocks = [block.size for block in matcher.get_matching_blocks() if block.size >= 3]
    if (
        len(strong_blocks) >= 3
        and max(strong_blocks, default=0) >= 8
        and sum(strong_blocks) >= 18
    ):
        return 0.65
    return 0.0


def _entity_supported_paraphrase_similarity(left: str, right: str, *, containment: float) -> float:
    """Allow a narrow fallback when Gemini chooses parent/child names for one entity."""

    if containment < 0.50 or _has_obvious_state_conflict(left, right):
        return 0.0
    left_compare = _comparison_text(left)
    right_compare = _comparison_text(right)
    if not left_compare or not right_compare:
        return 0.0
    sequence = SequenceMatcher(None, left_compare, right_compare, autojunk=False).ratio()
    return 0.65 if sequence >= 0.75 else 0.0


def _same_session_reference_reconciliation_score(
    connection: Any,
    memory_id: int,
    row_content: str,
    incoming_content: str,
    incoming_signature: str | None,
    session_key: str | None,
    *,
    author_id: int | None = None,
    require_other_author: bool = False,
) -> float:
    """Match one lossy conversational reference only inside an already-known session."""

    if not session_key:
        return 0.0
    incoming_hashes = _event_signature_hashes(incoming_signature)
    if not incoming_hashes:
        return 0.0
    rows = connection.execute(
        """SELECT event_signature, user_id FROM guild_memory_evidence
           WHERE guild_memory_id = ? AND session_key = ? AND event_signature IS NOT NULL
           ORDER BY id DESC LIMIT 16""",
        (int(memory_id), session_key),
    ).fetchall()
    best_shared = 0
    best_containment = 0.0
    for row in rows:
        if (
            require_other_author
            and author_id is not None
            and int(row["user_id"]) == int(author_id)
        ):
            continue
        existing_hashes = _event_signature_hashes(str(row["event_signature"] or ""))
        if not existing_hashes:
            continue
        shared = len(existing_hashes & incoming_hashes)
        smaller = min(len(existing_hashes), len(incoming_hashes))
        containment = shared / smaller if smaller else 0.0
        if (shared, containment) > (best_shared, best_containment):
            best_shared = shared
            best_containment = containment
    if (
        best_shared < _SAME_SESSION_REFERENCE_MIN_SHARED_HASHES
        or best_containment < _SAME_SESSION_REFERENCE_MIN_CONTAINMENT
    ):
        return 0.0
    left_compare = _comparison_text(row_content)
    right_compare = _comparison_text(incoming_content)
    if not left_compare or not right_compare:
        return 0.0
    sequence = SequenceMatcher(None, left_compare, right_compare, autojunk=False).ratio()
    return 0.65 if sequence >= _SAME_SESSION_REFERENCE_MIN_SEQUENCE else 0.0


def _hierarchical_entity_match(
    row_entity_type: str | None,
    row_entity: str | None,
    entity_type: str | None,
    entity: str | None,
) -> bool:
    """Return true only for same-type parent/child entity names, never arbitrary overlap."""

    normalized_row_type = _normalized_optional(row_entity_type)
    normalized_type = _normalized_optional(entity_type)
    normalized_row_entity = _normalized_optional(row_entity)
    normalized_entity = _normalized_optional(entity)
    if (
        not normalized_row_type
        or not normalized_type
        or normalized_row_type != normalized_type
        or not normalized_row_entity
        or not normalized_entity
        or normalized_row_entity == normalized_entity
    ):
        return False
    shorter, longer = sorted((normalized_row_entity, normalized_entity), key=len)
    return len(shorter) >= 4 and shorter in longer

def _has_explicit_entity_conflict(
    row_entity_type: str | None,
    row_entity: str | None,
    entity_type: str | None,
    entity: str | None,
) -> bool:
    row_type = _normalized_optional(row_entity_type)
    incoming_type = _normalized_optional(entity_type)
    row_value = _normalized_optional(row_entity)
    incoming_value = _normalized_optional(entity)

    return bool(
        row_type
        and incoming_type
        and row_type == incoming_type
        and row_value
        and incoming_value
        and row_value != incoming_value
    )

def _expire_due_memories(connection: Any, guild_id: int, *, now: datetime | None = None) -> int:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
    cursor = connection.execute(
        """UPDATE guild_memories
           SET status = 'expired', updated_at = CURRENT_TIMESTAMP
           WHERE guild_id = ? AND status IN ('candidate', 'active')
             AND expires_at IS NOT NULL AND expires_at <= ?""",
        (int(guild_id), current),
    )
    return max(0, int(cursor.rowcount))


def _candidate_rows(
    connection: Any,
    *,
    guild_id: int,
    domain: str,
    subdomain: str | None,
) -> list[Any]:
    return list(
        connection.execute(
            """SELECT * FROM guild_memories
               WHERE guild_id = ? AND domain = ?
                 AND ((subdomain IS NULL AND ? IS NULL) OR subdomain = ?)
                 AND status IN ('candidate', 'active')
               ORDER BY CASE status WHEN 'active' THEN 0 ELSE 1 END, id DESC
               LIMIT 64""",
            (int(guild_id), domain, subdomain, subdomain),
        ).fetchall()
    )


def _reconciliation_score(
    row: Any,
    *,
    normalized_content: str,
    entity_type: str | None,
    entity: str | None,
) -> float:
    row_content = str(row["content"])
    if str(row["normalized_content"]) == normalized_content:
        return 1.0
    lexical, containment = _term_overlap(row_content, normalized_content)
    if _has_obvious_state_conflict(row_content, normalized_content):
        return 0.0

    score = max(
        lexical,
        _paraphrase_similarity(row_content, normalized_content, containment=containment),
    )
    row_entity = _normalized_optional(row["entity"])
    normalized_entity = _normalized_optional(entity)
    if row_entity and normalized_entity and row_entity == normalized_entity:
        # The same place/character/object can be involved in many unrelated group
        # episodes. Entity identity is supporting evidence only; require substantial
        # content overlap before treating two observations as the same episode.
        score = max(score, min(1.0, lexical + 0.25))
    elif _hierarchical_entity_match(
        str(row["entity_type"]) if row["entity_type"] else None,
        str(row["entity"]) if row["entity"] else None,
        entity_type,
        entity,
    ):
        # Gemini may call one observation by a parent facility and another by a child
        # component. Use that structure only to unlock a slightly narrower text gate;
        # it is never sufficient by itself to merge two incidents at the same place.
        score = max(
            score,
            _entity_supported_paraphrase_similarity(
                row_content,
                normalized_content,
                containment=containment,
            ),
        )
    return score


def _find_reconciled_memory(
    connection: Any,
    *,
    guild_id: int,
    domain: str,
    subdomain: str | None,
    normalized_content: str,
    entity_type: str | None,
    entity: str | None,
    event_signature: str | None,
    session_key: str | None,
    author_id: int,
) -> Any | None:
    best: Any | None = None
    best_score = 0.0

    for row in _candidate_rows(
        connection,
        guild_id=guild_id,
        domain=domain,
        subdomain=subdomain,
    ):
        row_content = str(row["content"])
        score = _reconciliation_score(
            row,
            normalized_content=normalized_content,
            entity_type=entity_type,
            entity=entity,
        )
        status = str(row["status"])

        row_entity_type = (
            str(row["entity_type"]) if row["entity_type"] else None
        )
        row_entity = (
            str(row["entity"]) if row["entity"] else None
        )

        candidate_hierarchical_entity_match = bool(
            status == "candidate"
            and str(row["normalized_content"]) != normalized_content
            and _hierarchical_entity_match(
                row_entity_type,
                row_entity,
                entity_type,
                entity,
            )
        )

        candidate_entity_conflict = bool(
            status == "candidate"
            and str(row["normalized_content"]) != normalized_content
            and not candidate_hierarchical_entity_match
            and _has_explicit_entity_conflict(
                row_entity_type,
                row_entity,
                entity_type,
                entity,
            )
        )

        if candidate_entity_conflict:
            # Same-type, explicitly different non-hierarchical entities are
            # negative evidence for one unpromoted shared episode.
            score = 0.0

        elif candidate_hierarchical_entity_match:
            # _reconciliation_score() already contains the narrow parent/child
            # entity path. Do not add broader signature/session fallbacks here.
            pass

        elif not _has_obvious_state_conflict(
            row_content,
            normalized_content,
        ):
            score = max(
                score,
                _event_signature_reconciliation_score(
                    connection,
                    int(row["id"]),
                    event_signature,
                ),
            )

            if status == "active":
                score = max(
                    score,
                    _same_session_reference_reconciliation_score(
                        connection,
                        int(row["id"]),
                        row_content,
                        normalized_content,
                        event_signature,
                        session_key,
                    ),
                )

            elif status == "candidate":
                score = max(
                    score,
                    _same_session_reference_reconciliation_score(
                        connection,
                        int(row["id"]),
                        row_content,
                        normalized_content,
                        event_signature,
                        session_key,
                        author_id=author_id,
                        require_other_author=True,
                    ),
                )

        if score > best_score:
            best, best_score = row, score

    return best if best_score >= 0.65 else None


def _distinct_evidence(connection: Any, memory_id: int) -> tuple[int, int]:
    row = connection.execute(
        """SELECT COUNT(DISTINCT user_id) AS users,
                  COUNT(DISTINCT CASE WHEN session_key IS NOT NULL THEN session_key END) AS sessions
           FROM guild_memory_evidence WHERE guild_memory_id = ?""",
        (int(memory_id),),
    ).fetchone()
    return int(row["users"] or 0), int(row["sessions"] or 0)


def _existing_sessions(connection: Any, memory_id: int) -> set[str]:
    return {
        str(row["session_key"])
        for row in connection.execute(
            """SELECT DISTINCT session_key FROM guild_memory_evidence
               WHERE guild_memory_id = ? AND session_key IS NOT NULL""",
            (int(memory_id),),
        ).fetchall()
    }


def _retention_for_reinforcement(count: int) -> str:
    if count >= 3:
        return RETENTION_LONG
    if count >= 2:
        return RETENTION_MEDIUM
    return RETENTION_SHARED


def _activate_memory(connection: Any, memory_id: int, *, observed_at: datetime) -> None:
    connection.execute(
        """UPDATE guild_memories
           SET status = 'active', retention = ?, reinforcement_count = 0,
               expires_at = ?, updated_at = CURRENT_TIMESTAMP
           WHERE id = ?""",
        (
            RETENTION_SHARED,
            expires_at_for_retention(RETENTION_SHARED, now=observed_at),
            int(memory_id),
        ),
    )


def _reinforce_memory(
    connection: Any,
    memory_id: int,
    *,
    previous_count: int,
    observed_at: datetime,
    observed_at_text: str | None,
) -> None:
    count = previous_count + 1
    retention = _retention_for_reinforcement(count)
    connection.execute(
        """UPDATE guild_memories
           SET reinforcement_count = ?, retention = ?, expires_at = ?,
               last_reinforced = ?, updated_at = CURRENT_TIMESTAMP
           WHERE id = ? AND status = 'active'""",
        (
            count,
            retention,
            expires_at_for_retention(retention, now=observed_at),
            observed_at_text or observed_at.isoformat(),
            int(memory_id),
        ),
    )


def record_shared_candidate(
    database: Any,
    *,
    guild_id: int,
    author_id: int,
    content: str,
    domain: str,
    subdomain: str | None,
    memory_kind: str,
    confidence: float,
    importance: int,
    entity_type: str | None,
    entity: str | None,
    session_key: str | None,
    channel_id: int,
    message_id: int,
    observed_at: str | None,
    event_signature: str | None = None,
    shared_group_event: bool = False,
    participant_ids: Iterable[int] = (),
) -> int | None:
    """Record public evidence and conservatively promote a guild-shared memory.

    Candidate reconciliation, evidence insertion, promotion, reinforcement, and expiry
    refresh happen inside one SQLite transaction. Evidence deliberately contains no
    raw Discord message text.
    """

    normalized_content_text = normalize_memory_text(content)
    normalized_domain = _normalized_optional(domain)
    normalized_subdomain = _normalized_optional(subdomain)
    normalized_kind = _normalized_optional(memory_kind)
    normalized_entity_type = normalize_memory_text(entity_type or "") or None
    normalized_entity = normalize_memory_text(entity or "") or None
    normalized_event_signature = _normalize_event_signature(event_signature)
    if not normalized_domain or not normalized_kind:
        return None
    if not is_shareable_guild_memory(
        normalized_content_text,
        domain=normalized_domain,
        subdomain=normalized_subdomain,
        memory_kind=normalized_kind,
        confidence=float(confidence),
    ):
        return None

    participants = {int(value) for value in participant_ids if int(value) > 0}
    participants.add(int(author_id))
    event_time = _utc_datetime(observed_at)
    ensure_shared_memory_schema(database)
    connection = getattr(database, "connection", None)
    if connection is None:
        return None

    with connection:
        _expire_due_memories(connection, int(guild_id), now=event_time)
        normalized_for_match = normalized_content_text.casefold()
        row = _find_reconciled_memory(
            connection,
            guild_id=int(guild_id),
            domain=normalized_domain,
            subdomain=normalized_subdomain,
            normalized_content=normalized_for_match,
            entity_type=normalized_entity_type,
            entity=normalized_entity,
            event_signature=normalized_event_signature,
            session_key=session_key,
            author_id=int(author_id),
        )
        if row is None:
            cursor = connection.execute(
                """INSERT INTO guild_memories(
                       guild_id, domain, subdomain, memory_kind, entity_type, entity,
                       content, normalized_content, importance, confidence,
                       retention, expires_at, status
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'candidate')""",
                (
                    int(guild_id),
                    normalized_domain,
                    normalized_subdomain,
                    normalized_kind,
                    normalized_entity_type,
                    normalized_entity,
                    normalized_content_text,
                    normalized_for_match,
                    max(1, min(int(importance), 3)),
                    max(0.0, min(float(confidence), 1.0)),
                    RETENTION_SHARED,
                    expires_at_for_retention(RETENTION_SHARED, now=event_time),
                ),
            )
            memory_id = int(cursor.lastrowid)
            row = connection.execute(
                "SELECT * FROM guild_memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
        else:
            memory_id = int(row["id"])
            connection.execute(
                """UPDATE guild_memories
                   SET importance = MAX(importance, ?), confidence = MAX(confidence, ?),
                       updated_at = CURRENT_TIMESTAMP
                   WHERE id = ?""",
                (
                    max(1, min(int(importance), 3)),
                    max(0.0, min(float(confidence), 1.0)),
                    memory_id,
                ),
            )

        previous_status = str(row["status"])
        previous_reinforcement = int(row["reinforcement_count"] or 0)
        previous_sessions = _existing_sessions(connection, memory_id)
        cursor = connection.execute(
            """INSERT OR IGNORE INTO guild_memory_evidence(
                   guild_memory_id, guild_id, user_id, channel_id, message_id,
                   observed_at, session_key, evidence_kind, event_signature
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                memory_id,
                int(guild_id),
                int(author_id),
                int(channel_id),
                int(message_id),
                observed_at,
                session_key,
                "group_event" if shared_group_event else "passive",
                normalized_event_signature,
            ),
        )
        if int(cursor.rowcount) <= 0:
            return memory_id

        if previous_status == "candidate":
            distinct_users, distinct_sessions = _distinct_evidence(connection, memory_id)
            immediate_group_event = bool(shared_group_event and len(participants) >= 2)
            if immediate_group_event or distinct_users >= 2 or distinct_sessions >= 2:
                _activate_memory(connection, memory_id, observed_at=event_time)
            return memory_id

        if previous_status == "active" and session_key and session_key not in previous_sessions:
            _reinforce_memory(
                connection,
                memory_id,
                previous_count=previous_reinforcement,
                observed_at=event_time,
                observed_at_text=observed_at,
            )
        return memory_id


def search_guild_memories(
    database: Any,
    guild_id: int,
    query: str,
    *,
    domain: str | None = None,
    subdomain: str | None = None,
    limit: int = 3,
) -> list[GuildMemoryMatch]:
    """Return only active, unexpired shared memories ranked for the current topic."""

    ensure_shared_memory_schema(database)
    connection = getattr(database, "connection", None)
    if connection is None:
        return []
    normalized_query = normalize_memory_text(query)
    if not normalized_query:
        return []
    normalized_domain = _normalized_optional(domain)
    normalized_subdomain = _normalized_optional(subdomain)
    bounded_limit = max(1, min(int(limit), 20))

    with connection:
        _expire_due_memories(connection, int(guild_id))

    clauses = ["guild_id = ?", "status = 'active'"]
    parameters: list[Any] = [int(guild_id)]
    if normalized_domain:
        clauses.append("domain = ?")
        parameters.append(normalized_domain)
    if normalized_subdomain:
        clauses.append("subdomain = ?")
        parameters.append(normalized_subdomain)
    rows = connection.execute(
        f"SELECT * FROM guild_memories WHERE {' AND '.join(clauses)} ORDER BY id DESC LIMIT 128",
        tuple(parameters),
    ).fetchall()

    query_lower = normalized_query.casefold()
    scored: list[tuple[float, Any]] = []
    for row in rows:
        score = float(score_memory_candidate(normalized_query, row))
        row_domain = str(row["domain"])
        row_subdomain = str(row["subdomain"] or "")
        row_entity = str(row["entity"] or "")
        if normalized_domain and row_domain == normalized_domain:
            score += 1.0
        if normalized_subdomain and row_subdomain == normalized_subdomain:
            score += 1.5
        if row_subdomain and row_subdomain in query_lower:
            score += 1.0
        if row_entity and row_entity.casefold() in query_lower:
            score += 0.75
        if score > 0:
            scored.append((score, row))
    scored.sort(key=lambda item: (item[0], int(item[1]["id"])), reverse=True)

    return [
        GuildMemoryMatch(
            id=int(row["id"]),
            content=str(row["content"]),
            domain=str(row["domain"]),
            subdomain=str(row["subdomain"]) if row["subdomain"] else None,
            kind=str(row["memory_kind"]),
            entity_type=str(row["entity_type"]) if row["entity_type"] else None,
            entity=str(row["entity"]) if row["entity"] else None,
            score=round(float(score), 6),
            retention=str(row["retention"]),
            reinforcement_count=int(row["reinforcement_count"] or 0),
        )
        for score, row in scored[:bounded_limit]
    ]


def delete_guild_memory(database: Any, guild_id: int, memory_id: int) -> bool:
    """Hard-delete one guild-owned memory scoped to the current guild.

    Foreign-key cascade removes only that shared memory's evidence rows. Personal
    `user_memories` are a separate table and are intentionally never touched here.
    """

    ensure_shared_memory_schema(database)
    connection = getattr(database, "connection", None)
    if connection is None:
        return False
    with connection:
        cursor = connection.execute(
            "DELETE FROM guild_memories WHERE id = ? AND guild_id = ?",
            (int(memory_id), int(guild_id)),
        )
    return int(cursor.rowcount) > 0
