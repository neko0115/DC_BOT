from __future__ import annotations

import json
import re
from dataclasses import dataclass

from discord_ai_assistant.ai.persona import is_persona_control_attempt


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
RESERVED_MEMORY_CATEGORY_MARKERS = (
    "系統",
    "提示",
    "規則",
    "人設",
    "角色",
    "指令",
    "開發者",
    "system",
    "developer",
    "instruction",
)

_MEMORY_CONTROL_PATTERNS = (
    re.compile(r"(?:你|墨雪).{0,12}(?:必須|只能|不得|不准|應該|以後要|永遠要|每次都要)"),
    re.compile(
        r"(?:把|將).{0,10}(?:這段|以下|這些).{0,16}"
        r"(?:當成|視為|設為).{0,8}(?:系統|規則|人設|角色|指令|記憶)"
    ),
    re.compile(r"(?:覆蓋|修改|竄改|刪除|清空|重設|重置).{0,12}(?:記憶|人設|規則|指令)"),
)

_EXPLICIT_MEMORY_PREFIX = re.compile(
    r"^\s*(?:(?:請|麻煩|幫我|可以|能不能)\s*)?(?:@?(?:墨雪|你)\s*[，,]?\s*)?"
    r"(?:(?:請|麻煩|幫我|可以|能不能)\s*)?(?:幫我\s*)?(?:記住|記下來|記下|記一下|記起來)\s*"
    r"(?:這件事|這點|以下)?\s*[:：,，]?\s*(.+?)\s*$"
)
_EXPLICIT_MEMORY_SUFFIX = re.compile(
    r"^\s*(?:@?(?:墨雪|你)\s*[，,]?\s*)?(?:(?:請|麻煩|幫我|可以|能不能)\s*)?"
    r"把\s*(.+?)\s*(?:記住|記下來|記起來)\s*$"
)
_EXPLICIT_PROJECT_MARKERS = (
    "我正在做",
    "我目前在做",
    "我的專案",
    "我的工作",
    "這個專案",
    "這項專案",
    "專案目前",
    "專案現在",
)
_EXPLICIT_EPISODIC_MARKERS = (
    "上次",
    "之前",
    "曾經",
    "那次",
    "後來",
    "昨天",
    "今天",
    "發生",
    "我們決定",
    "最後決定",
    "已經修好",
    "已經完成",
    "測試通過",
    "測試失敗",
)


@dataclass(frozen=True, slots=True)
class MemoryDraft:
    category: str
    content: str


def is_sensitive_memory(content: str) -> bool:
    return any(marker in content.lower() for marker in SENSITIVE_MEMORY_MARKERS)


def is_disallowed_memory(category: str, content: str) -> bool:
    """Reject secrets and instruction-like text from every persistent-memory path."""
    combined = f"{category}\n{content}"
    normalized = " ".join(combined.lower().split())
    normalized_category = " ".join(category.lower().split())
    return bool(
        is_sensitive_memory(content)
        or any(marker in normalized_category for marker in RESERVED_MEMORY_CATEGORY_MARKERS)
        or is_persona_control_attempt(combined)
        or any(pattern.search(normalized) for pattern in _MEMORY_CONTROL_PATTERNS)
    )


def is_passive_memory_candidate(content: str) -> bool:
    normalized = " ".join(content.split())
    if not 8 <= len(normalized) <= 300 or "http://" in normalized.lower() or "https://" in normalized.lower():
        return False
    return any(keyword in normalized for keyword in PASSIVE_MEMORY_KEYWORDS)


def parse_explicit_memory_request(content: str) -> MemoryDraft | None:
    """Parse an explicit request to save one personal memory from a normal chat prompt.

    This deliberately recognises only clear imperative phrasing. Ordinary chat remains
    non-persistent and continues through the existing passive-memory opt-in path.
    """
    current_request = content.rsplit("目前請求：", 1)[-1].strip()
    match = _EXPLICIT_MEMORY_PREFIX.match(current_request) or _EXPLICIT_MEMORY_SUFFIX.match(current_request)
    if not match:
        return None
    normalized = " ".join(match.group(1).strip(" \t\r\n「」『』\"'").split())
    if not normalized or len(normalized) > 300:
        return None
    return MemoryDraft(_explicit_memory_category(normalized), normalized)


def _explicit_memory_category(content: str) -> str:
    if content.startswith(("我喜歡", "我很喜歡", "我不喜歡", "我討厭", "我偏好", "偏好")):
        return "偏好"
    if content.startswith(("我習慣", "我通常", "我常常", "我每天", "我週末")):
        return "習慣"
    if content.startswith(("我在玩", "我最近在玩", "我的興趣", "我的愛好")):
        return "興趣"
    if any(marker in content for marker in _EXPLICIT_PROJECT_MARKERS):
        return "專案"
    if any(marker in content for marker in _EXPLICIT_EPISODIC_MARKERS):
        return "事件"
    return "提醒"


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
            or is_disallowed_memory(normalized_category, normalized_content)
            or key in seen
        ):
            continue
        seen.add(key)
        drafts.append(MemoryDraft(normalized_category, normalized_content))
    return drafts
