from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping


MEMORY_KIND_SEMANTIC = "semantic"
MEMORY_KIND_PREFERENCE = "preference"
MEMORY_KIND_HABIT = "habit"
MEMORY_KIND_INTEREST = "interest"
MEMORY_KIND_PROJECT = "project"
MEMORY_KIND_EPISODIC = "episodic"

MEMORY_KINDS = {
    MEMORY_KIND_SEMANTIC,
    MEMORY_KIND_PREFERENCE,
    MEMORY_KIND_HABIT,
    MEMORY_KIND_INTEREST,
    MEMORY_KIND_PROJECT,
    MEMORY_KIND_EPISODIC,
}

_GENERIC_CATEGORIES = {
    "偏好",
    "習慣",
    "興趣",
    "專案",
    "提醒",
    "事件",
    "決策",
    "紀錄",
    "里程碑",
}
_LATIN_TOKEN = re.compile(r"[a-z0-9][a-z0-9_.+#-]{1,63}", re.IGNORECASE)
_CJK_RUN = re.compile(r"[\u3400-\u9fff]{2,}")
_SLOT_PATTERNS = (
    re.compile(
        r"^我(?:目前|現在)?的(?P<subject>[^，。,:：]{1,32}?)"
        r"(?:是|為|叫|用的是|使用的是|改成|換成)\s*(?P<value>.+)$"
    ),
    re.compile(
        r"^我的(?P<subject>[^，。,:：]{1,32}?)"
        r"(?:是|為|叫|改成|換成)\s*(?P<value>.+)$"
    ),
)
_PROJECT_PATTERNS = (
    re.compile(
        r"(?:我(?:正在|目前在)做|我的專案(?:是|叫)?|專案(?:是|叫|[:：])?)\s*"
        r"(?P<project>[A-Za-z][A-Za-z0-9_.-]{1,39}|[\u3400-\u9fff]{2,16})"
    ),
    re.compile(r"^\s*(?P<project>[A-Za-z][A-Za-z0-9_.-]{1,39})\b"),
)
_EPISODIC_MARKERS = (
    "上次",
    "之前",
    "曾經",
    "後來",
    "昨天",
    "今天",
    "那次",
    "發生",
    "決定",
    "通過",
    "失敗",
    "修好",
    "修正",
    "完成",
)


@dataclass(frozen=True, slots=True)
class MemoryMatch:
    id: int
    category: str
    content: str
    kind: str
    subject: str | None
    project: str | None
    score: float


def normalize_memory_text(value: str) -> str:
    return " ".join(value.strip().split())


def memory_terms(value: str) -> tuple[str, ...]:
    """Return deterministic lexical terms for mixed CJK / Latin memory search.

    SQLite's default FTS tokenizer is good at Latin identifiers but weak for the short
    Chinese phrases common in Discord.  We therefore index Latin tokens plus CJK
    bigrams/trigrams, and use the same representation for query scoring.
    """

    normalized = normalize_memory_text(value).lower()
    terms: set[str] = set(_LATIN_TOKEN.findall(normalized))
    for run in _CJK_RUN.findall(normalized):
        if len(run) <= 12:
            terms.add(run)
        for size in (2, 3):
            if len(run) < size:
                continue
            terms.update(run[index : index + size] for index in range(len(run) - size + 1))
    return tuple(sorted(terms))


def memory_search_document(*parts: str | None) -> str:
    terms: set[str] = set()
    for part in parts:
        if part:
            terms.update(memory_terms(part))
    return " ".join(sorted(terms))


def build_fts_query(query: str, *, limit: int = 32) -> str:
    terms = list(memory_terms(query))
    # Longer terms are more discriminative; keep deterministic ordering for tests.
    terms.sort(key=lambda item: (-len(item), item))
    return " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms[:limit])


def infer_memory_kind(category: str, content: str) -> str:
    clean_category = normalize_memory_text(category).removeprefix("自動")
    normalized = normalize_memory_text(content)
    if "偏好" in clean_category:
        return MEMORY_KIND_PREFERENCE
    if "習慣" in clean_category:
        return MEMORY_KIND_HABIT
    if "興趣" in clean_category:
        return MEMORY_KIND_INTEREST
    if "專案" in clean_category:
        return MEMORY_KIND_PROJECT
    if any(marker in clean_category for marker in ("事件", "決策", "紀錄", "里程碑", "經歷")):
        return MEMORY_KIND_EPISODIC
    if clean_category == "提醒" and any(marker in normalized for marker in _EPISODIC_MARKERS):
        return MEMORY_KIND_EPISODIC
    return MEMORY_KIND_SEMANTIC


def infer_project_name(category: str, content: str) -> str | None:
    clean_category = normalize_memory_text(category).removeprefix("自動")
    if "專案" not in clean_category:
        return None
    normalized = normalize_memory_text(content)
    for pattern in _PROJECT_PATTERNS:
        match = pattern.search(normalized)
        if match:
            return normalize_memory_text(match.group("project"))[:40]
    return None


def infer_memory_key(category: str, content: str) -> tuple[str | None, str | None]:
    """Infer a conservative mutable slot key and subject.

    Only stable profile-like facts are automatically consolidated.  Free-form project
    notes and episodic memories intentionally do not overwrite one another.
    """

    normalized_category = normalize_memory_text(category).removeprefix("自動")
    normalized_content = normalize_memory_text(content)

    if normalized_category and normalized_category not in _GENERIC_CATEGORIES and len(normalized_category) <= 32:
        subject = normalized_category
        return f"category:{subject.casefold()}", subject

    for pattern in _SLOT_PATTERNS:
        match = pattern.match(normalized_content)
        if not match:
            continue
        subject = normalize_memory_text(match.group("subject")).strip("的 ")
        if not subject or len(subject) > 32:
            continue
        return f"profile:{subject.casefold()}", subject
    return None, None


def _row_value(row: Mapping[str, object], key: str, default: object = None) -> object:
    try:
        value = row[key]
    except (KeyError, IndexError):
        return default
    return default if value is None else value


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        try:
            parsed = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def score_memory_candidate(
    query: str,
    row: Mapping[str, object],
    *,
    fts_position: int | None = None,
    now: datetime | None = None,
) -> float:
    """Rank a memory for one query using relevance first, then stable quality signals."""

    category = str(_row_value(row, "category", ""))
    content = str(_row_value(row, "content", ""))
    subject_value = _row_value(row, "subject")
    project_value = _row_value(row, "project")
    subject = str(subject_value) if subject_value else ""
    project = str(project_value) if project_value else ""

    query_terms = set(memory_terms(query))
    document_terms = set(memory_terms(" ".join(part for part in (category, content, subject, project) if part)))
    if query_terms:
        overlap = query_terms & document_terms
        weighted_overlap = sum(min(2.0, 0.7 + (len(term) * 0.18)) for term in overlap)
        weighted_query = sum(min(2.0, 0.7 + (len(term) * 0.18)) for term in query_terms)
        lexical = min(1.0, weighted_overlap / max(1.0, weighted_query))
    else:
        lexical = 0.0

    normalized_query = normalize_memory_text(query).casefold()
    exact_boost = 0.0
    if normalized_query:
        if project and project.casefold() in normalized_query:
            exact_boost += 0.16
        if subject and subject.casefold() in normalized_query:
            exact_boost += 0.12
        if normalized_query in content.casefold():
            exact_boost += 0.10

    importance_raw = _row_value(row, "importance", 1)
    try:
        importance = max(1.0, min(float(importance_raw), 3.0)) / 3.0
    except (TypeError, ValueError):
        importance = 1.0 / 3.0

    timestamp = (
        _parse_timestamp(_row_value(row, "last_confirmed"))
        or _parse_timestamp(_row_value(row, "updated_at"))
        or _parse_timestamp(_row_value(row, "created_at"))
    )
    current = now.astimezone(timezone.utc) if now is not None else datetime.now(timezone.utc)
    if timestamp is None:
        recency = 0.0
    else:
        age_days = max(0.0, (current - timestamp).total_seconds() / 86400.0)
        recency = 1.0 / (1.0 + age_days / 45.0)

    use_count_raw = _row_value(row, "use_count", 0)
    try:
        use_count = max(0, int(use_count_raw))
    except (TypeError, ValueError):
        use_count = 0
    usage = min(1.0, math.log1p(use_count) / math.log(11.0))

    source = str(_row_value(row, "source", ""))
    source_bonus = 1.0 if source == "manual" else 0.0
    fts_bonus = 0.0 if fts_position is None else 1.0 / max(1.0, float(fts_position + 1))

    score = (
        lexical * 0.58
        + min(0.20, exact_boost)
        + importance * 0.10
        + recency * 0.05
        + usage * 0.025
        + source_bonus * 0.025
        + fts_bonus * 0.05
    )
    return round(min(1.0, score), 6)
