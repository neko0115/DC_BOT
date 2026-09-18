from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

RETENTION_SHORT = "short"
RETENTION_MEDIUM = "medium"
RETENTION_LONG = "long"
RETENTION_SHARED = "shared"
VALID_RETENTIONS = {
    RETENTION_SHORT,
    RETENTION_MEDIUM,
    RETENTION_LONG,
    RETENTION_SHARED,
}
RETENTION_DAYS = {
    RETENTION_SHORT: 14,
    RETENTION_MEDIUM: 90,
    RETENTION_SHARED: 30,
}


def _utc_datetime(value: datetime | None = None) -> datetime:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc)


def expires_at_for_retention(retention: str, *, now: datetime | None = None) -> str | None:
    """Return the UTC expiry timestamp for one retention class."""

    normalized = retention.strip().casefold()
    if normalized == RETENTION_LONG:
        return None
    days = RETENTION_DAYS.get(normalized)
    if days is None:
        raise ValueError(f"Unsupported memory retention: {retention}")

    current = _utc_datetime(now)
    return (current + timedelta(days=days)).isoformat()


def reinforce_personal_memory(
    database: Any,
    memory_id: int,
    *,
    session_key: str | None,
    now: datetime | None = None,
) -> bool:
    """Credit one passive personal-memory reconfirmation for a new session.

    The first non-null provenance session establishes the memory and is not a
    reinforcement. Later sessions can be credited at most once by marking the
    corresponding provenance row. Short/medium expiry is refreshed from the latest
    confirmation without changing the memory kind or turning current state into a
    stable preference.
    """

    normalized_session = str(session_key or "").strip()
    if not normalized_session:
        return False
    connection = getattr(database, "connection", None)
    if connection is None:
        return False

    memory = connection.execute(
        """SELECT id, status, source, retention, reinforcement_count, expires_at
           FROM user_memories WHERE id = ?""",
        (int(memory_id),),
    ).fetchone()
    if not memory or str(memory["status"]) != "active" or str(memory["source"]) != "passive":
        return False

    columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(memory_provenance)").fetchall()
    }
    if "session_key" not in columns or "reinforcement_credited" not in columns:
        return False

    sessions = connection.execute(
        """SELECT session_key, MIN(id) AS first_id,
                  MAX(reinforcement_credited) AS credited
           FROM memory_provenance
           WHERE memory_id = ?
             AND session_key IS NOT NULL AND TRIM(session_key) != ''
           GROUP BY session_key
           ORDER BY first_id ASC""",
        (int(memory_id),),
    ).fetchall()
    if len(sessions) < 2:
        return False
    baseline_session = str(sessions[0]["session_key"])
    if normalized_session == baseline_session:
        return False
    target = next((row for row in sessions if str(row["session_key"]) == normalized_session), None)
    if target is None or int(target["credited"] or 0):
        return False

    current = _utc_datetime(now)
    normalized_retention = str(memory["retention"] or RETENTION_LONG).strip().casefold()
    if normalized_retention not in {RETENTION_SHORT, RETENTION_MEDIUM, RETENTION_LONG}:
        return False
    refreshed_expiry = (
        expires_at_for_retention(normalized_retention, now=current)
        if normalized_retention in {RETENTION_SHORT, RETENTION_MEDIUM}
        else None
    )

    with connection:
        cursor = connection.execute(
            """UPDATE memory_provenance
               SET reinforcement_credited = 1
               WHERE id = (
                   SELECT MIN(id) FROM memory_provenance
                   WHERE memory_id = ? AND session_key = ?
                     AND reinforcement_credited = 0
               )""",
            (int(memory_id), normalized_session),
        )
        if int(cursor.rowcount) <= 0:
            return False
        connection.execute(
            """UPDATE user_memories
               SET reinforcement_count = reinforcement_count + 1,
                   last_reinforced = ?,
                   expires_at = ?,
                   updated_at = CURRENT_TIMESTAMP
               WHERE id = ? AND status = 'active'""",
            (
                current.isoformat(),
                refreshed_expiry,
                int(memory_id),
            ),
        )
    return True
