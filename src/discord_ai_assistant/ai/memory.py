from __future__ import annotations

import json
from dataclasses import dataclass


PASSIVE_MEMORY_KEYWORDS = (
    "我喜歡",
    "我很喜歡",
    "我不喜歡",
    "我討厭",
    "我習慣",
    "我通常",
    "我常常",
    "我的興趣",
    "我的愛好",
    "我在玩",
    "我最近在玩",
    "我正在做",
    "我目前在做",
)
SENSITIVE_MEMORY_MARKERS = ("密碼", "password", "token", "api key", "api_key", "驗證碼", "verification code")
ALLOWED_MEMORY_CATEGORIES = {"偏好", "習慣", "興趣", "專案"}


@dataclass(frozen=True, slots=True)
class MemoryDraft:
    category: str
    content: str


def is_sensitive_memory(content: str) -> bool:
    return any(marker in content.lower() for marker in SENSITIVE_MEMORY_MARKERS)


def is_passive_memory_candidate(content: str) -> bool:
    normalized = " ".join(content.split())
    if not 8 <= len(normalized) <= 300 or "http://" in normalized.lower() or "https://" in normalized.lower():
        return False
    return any(keyword in normalized for keyword in PASSIVE_MEMORY_KEYWORDS)


def parse_memory_drafts(response: str) -> list[MemoryDraft]:
    """Accept only the narrow JSON response schema requested from Gemini."""
    candidate = response.strip()
    if not candidate:
        return []
    if "```" in candidate:
        candidate = candidate.replace("```json", "").replace("```JSON", "").replace("```", "").strip()
    if not candidate.startswith("{"):
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            return []
        candidate = candidate[start : end + 1]
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return []
    raw_memories = payload.get("memories") if isinstance(payload, dict) else None
    if not isinstance(raw_memories, list):
        return []

    drafts: list[MemoryDraft] = []
    seen: set[tuple[str, str]] = set()
    for item in raw_memories[:3]:
        if not isinstance(item, dict):
            continue
        category = item.get("category")
        content = item.get("content")
        if not isinstance(category, str) or not isinstance(content, str):
            continue
        normalized_category = " ".join(category.split())
        normalized_content = " ".join(content.split())
        key = (normalized_category, normalized_content)
        if (
            normalized_category not in ALLOWED_MEMORY_CATEGORIES
            or not 1 <= len(normalized_content) <= 120
            or is_sensitive_memory(normalized_content)
            or key in seen
        ):
            continue
        seen.add(key)
        drafts.append(MemoryDraft(normalized_category, normalized_content))
    return drafts
