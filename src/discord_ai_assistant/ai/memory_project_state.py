from __future__ import annotations

import re
from typing import Any

from discord_ai_assistant.ai.memory_phase2 import PassiveMemoryV2Draft
from discord_ai_assistant.ai.memory_retention import RETENTION_LONG
from discord_ai_assistant.ai.memory_v2 import MEMORY_KIND_EPISODIC, MEMORY_KIND_PROJECT, normalize_memory_text

PROJECT_MUTABLE_ROLES = {"status", "next_step", "blocker"}
PROJECT_HISTORY_ROLES = {"decision", "milestone", "event"}
PROJECT_SUMMARY_CATEGORY = "專案摘要"
_PROJECT_PREFIX_TEMPLATE = r"^專案\s+{project}\s*[：:]\s*"

_ROLE_CATEGORY = {
    "fact": "專案",
    "status": "專案/狀態",
    "next_step": "專案/下一步",
    "blocker": "專案/阻塞",
    "decision": "決策",
    "milestone": "事件/里程碑",
    "event": "事件",
}
_ROLE_LABEL = {
    "status": "目前狀態",
    "blocker": "目前阻塞",
    "next_step": "下一步",
    "fact": "關鍵資訊",
    "decision": "近期決策",
    "milestone": "近期里程碑",
    "event": "近期事件",
}


def normalize_project_name(value: str) -> str:
    return normalize_memory_text(value).strip("：: -")[:40]


def project_memory_key(project: str, role: str) -> str:
    return f"project:{normalize_project_name(project).casefold()}:{role}"


def project_summary_key(project: str) -> str:
    return project_memory_key(project, "summary")


def category_for_project_role(role: str) -> str:
    return _ROLE_CATEGORY.get(role, "專案")


def memory_kind_for_project_role(role: str) -> str:
    return MEMORY_KIND_EPISODIC if role in PROJECT_HISTORY_ROLES else MEMORY_KIND_PROJECT


def role_from_category(category: str) -> str:
    normalized = category.removeprefix("自動")
    if normalized == PROJECT_SUMMARY_CATEGORY:
        return "summary"
    if "下一步" in normalized:
        return "next_step"
    if "阻塞" in normalized or "blocker" in normalized.casefold():
        return "blocker"
    if "狀態" in normalized or "進度" in normalized:
        return "status"
    if "決策" in normalized:
        return "decision"
    if "里程碑" in normalized:
        return "milestone"
    if "事件" in normalized:
        return "event"
    return "fact"


def _strip_project_prefix(content: str, project: str) -> str:
    pattern = re.compile(_PROJECT_PREFIX_TEMPLATE.format(project=re.escape(project)), re.IGNORECASE)
    stripped = pattern.sub("", normalize_memory_text(content), count=1).strip()
    return stripped or normalize_memory_text(content)


def _publish_created(database: Any, guild_id: int, user_id: int, memory: Any, source: str) -> None:
    publisher = getattr(database, "_publish_memory_created", None)
    if callable(publisher):
        publisher(guild_id, user_id, memory, source=source)


def store_structured_project_memory(
    database: Any,
    guild_id: int,
    user_id: int,
    draft: PassiveMemoryV2Draft,
) -> tuple[Any | None, bool]:
    """Store one project draft using Phase-1 supersession semantics when available."""

    if not draft.project:
        return None, False
    project = normalize_project_name(draft.project)
    if not project:
        return None, False
    role = draft.role or "fact"
    category = f"自動{category_for_project_role(role)}"
    memory_key = project_memory_key(project, role) if role in PROJECT_MUTABLE_ROLES else None
    kind = memory_kind_for_project_role(role)

    upsert = getattr(database, "_upsert_user_memory", None)
    if callable(upsert):
        memory, created = upsert(
            guild_id,
            user_id,
            category,
            draft.content,
            source="passive",
            importance=draft.importance,
            memory_kind=kind,
            project=project,
            confidence=draft.confidence,
            memory_key=memory_key,
            domain="project",
            subdomain=project,
            retention=RETENTION_LONG,
        )
        if created:
            _publish_created(database, guild_id, user_id, memory, "passive")
        return memory, bool(created)

    # Compatibility fallback for alternate Database implementations. It cannot apply
    # project-slot supersession, but it preserves safe storage rather than failing.
    memory = database.add_user_memory_if_new(guild_id, user_id, category, draft.content)
    return memory, memory is not None


def _project_rows(database: Any, guild_id: int, user_id: int, project: str) -> list[Any]:
    connection = getattr(database, "connection", None)
    if connection is None:
        return []
    return list(
        connection.execute(
            """SELECT id, category, content, importance, memory_key, created_at, updated_at, last_confirmed
               FROM user_memories
               WHERE guild_id = ? AND user_id = ? AND status = 'active'
                 AND project = ? AND (memory_key IS NULL OR memory_key != ?)
               ORDER BY COALESCE(last_confirmed, updated_at, created_at) DESC, importance DESC, id DESC""",
            (guild_id, user_id, project, project_summary_key(project)),
        ).fetchall()
    )


def _group_project_rows(rows: list[Any]) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = {
        "status": [],
        "blocker": [],
        "next_step": [],
        "fact": [],
        "decision": [],
        "milestone": [],
        "event": [],
    }
    for row in rows:
        role = role_from_category(str(row["category"]))
        if role in grouped:
            grouped[role].append(row)
    return grouped


def _compact_item(content: str, project: str, limit: int = 90) -> str:
    value = _strip_project_prefix(content, project)
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


def build_project_summary(database: Any, guild_id: int, user_id: int, project: str) -> str | None:
    project = normalize_project_name(project)
    rows = _project_rows(database, guild_id, user_id, project)
    if not rows:
        return None
    grouped = _group_project_rows(rows)
    pieces: list[str] = [f"專案 {project} 摘要"]

    for role in ("status", "blocker", "next_step"):
        if grouped[role]:
            pieces.append(f"{_ROLE_LABEL[role]}：{_compact_item(str(grouped[role][0]['content']), project)}")

    if grouped["fact"]:
        facts = "；".join(_compact_item(str(row["content"]), project, 70) for row in grouped["fact"][:2])
        pieces.append(f"關鍵資訊：{facts}")

    if grouped["decision"]:
        decisions = "；".join(_compact_item(str(row["content"]), project, 65) for row in grouped["decision"][:2])
        pieces.append(f"近期決策：{decisions}")

    history = [*grouped["milestone"], *grouped["event"]]
    history.sort(key=lambda row: int(row["id"]), reverse=True)
    if history:
        events = "；".join(_compact_item(str(row["content"]), project, 65) for row in history[:2])
        pieces.append(f"近期事件：{events}")

    summary = "｜".join(pieces)
    if len(summary) > 300:
        summary = summary[:299].rstrip("｜； ，") + "…"
    return summary


def refresh_project_summary(database: Any, guild_id: int, user_id: int, project: str) -> Any | None:
    """Rebuild one derived project summary memory from active project memories."""

    project = normalize_project_name(project)
    summary = build_project_summary(database, guild_id, user_id, project)
    if not summary:
        return None
    upsert = getattr(database, "_upsert_user_memory", None)
    if not callable(upsert):
        return None
    memory, _ = upsert(
        guild_id,
        user_id,
        PROJECT_SUMMARY_CATEGORY,
        summary,
        source="derived",
        importance=3,
        memory_kind=MEMORY_KIND_PROJECT,
        project=project,
        confidence=1.0,
        memory_key=project_summary_key(project),
        domain="project",
        subdomain=project,
        retention=RETENTION_LONG,
    )
    return memory


def list_user_projects(database: Any, guild_id: int, user_id: int, *, limit: int = 20) -> list[str]:
    connection = getattr(database, "connection", None)
    if connection is None:
        return []
    rows = connection.execute(
        """SELECT project, MAX(COALESCE(last_confirmed, updated_at, created_at)) AS latest
           FROM user_memories
           WHERE guild_id = ? AND user_id = ? AND status = 'active'
             AND project IS NOT NULL AND TRIM(project) != ''
           GROUP BY project
           ORDER BY latest DESC, project COLLATE NOCASE ASC
           LIMIT ?""",
        (guild_id, user_id, max(1, min(limit, 50))),
    ).fetchall()
    return [str(row["project"]) for row in rows]


def project_snapshot_text(database: Any, guild_id: int, user_id: int, project: str) -> str:
    project = normalize_project_name(project)
    rows = _project_rows(database, guild_id, user_id, project)
    if not rows:
        return f"找不到專案 `{project}` 的 active Memory V2 記憶。"
    grouped = _group_project_rows(rows)
    lines = [f"**Memory V2 專案狀態：{project}**"]
    for role in ("status", "blocker", "next_step"):
        if grouped[role]:
            lines.append(f"- {_ROLE_LABEL[role]}：{_strip_project_prefix(str(grouped[role][0]['content']), project)}")
    for role in ("fact", "decision", "milestone", "event"):
        if not grouped[role]:
            continue
        label = _ROLE_LABEL[role]
        for row in grouped[role][:3]:
            lines.append(f"- {label}：{_strip_project_prefix(str(row['content']), project)}")
    return "\n".join(lines)[:2000]
