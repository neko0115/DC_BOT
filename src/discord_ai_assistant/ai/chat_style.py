from __future__ import annotations

import re
import statistics
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Protocol

from discord_ai_assistant.ai.memory import is_disallowed_memory
from discord_ai_assistant.ai.memory_phase2 import _contains_gossip, _is_passive_sensitive


_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_DISCORD_MENTION_RE = re.compile(r"<(?:@!?|@&|#)\d+>")
_DISCORD_CUSTOM_EMOJI_RE = re.compile(r"<a?:[A-Za-z0-9_]{1,32}:[0-9]{13,20}>")
_KEYCAP_EMOJI_RE = re.compile(r"[0-9#*]\ufe0f?\u20e3")
_DISCORD_MASS_MENTION_RE = re.compile(r"(?<![\w@.])@(?:everyone|here)(?![\w.-])")
_LATIN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+#.-]{1,39}")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_QUESTION_RE = re.compile(r"[?？]")


class ChatStyleSampleLike(Protocol):
    content: str
    is_reply: bool


@dataclass(frozen=True, slots=True)
class ChatStyleSample:
    id: int
    guild_id: int
    user_id: int
    channel_id: int
    message_id: int
    content: str
    observed_at: datetime
    session_key: str | None
    is_reply: bool
    is_public_evidence: bool = False


@dataclass(frozen=True, slots=True)
class ChatStyleStats:
    sample_count: int
    average_length: float
    median_length: float
    short_ratio: float
    reply_ratio: float
    emoji_ratio: float
    question_ratio: float
    mixed_script_ratio: float
    punctuation_ratio: float
    recurring_tokens: tuple[str, ...]


def _is_symbol_only(text: str) -> bool:
    """Check the textual remainder without changing the content we persist."""

    candidate = _DISCORD_MENTION_RE.sub("", text)
    candidate = _DISCORD_CUSTOM_EMOJI_RE.sub("", candidate)
    candidate = _KEYCAP_EMOJI_RE.sub("", candidate)
    candidate = _DISCORD_MASS_MENTION_RE.sub("", candidate)
    candidate = _URL_RE.sub("", candidate).strip()
    if not candidate:
        return True
    for character in candidate:
        if character.isspace():
            continue
        category = unicodedata.category(character)
        if category[0] in {"L", "N"}:
            return False
    return True


def _contains_emoji(text: str) -> bool:
    for character in text:
        codepoint = ord(character)
        if (
            0x1F000 <= codepoint <= 0x1FAFF
            or 0x2600 <= codepoint <= 0x27BF
            or 0x2300 <= codepoint <= 0x23FF
        ):
            return True
    return False


def prepare_chat_style_content(
    content: str,
    *,
    source_content: str | None = None,
    is_bot: bool = False,
    is_dm: bool = False,
    has_stickers: bool = False,
    is_system: bool = False,
) -> str | None:
    """Filter one public human message before any raw Chat Style persistence.

    Chat Style intentionally reuses Memory V2's existing secret/private-content
    policy. Raw source preserves artifact provenance for eligibility; only the
    normalized display content is returned for persistence. The result is
    descriptive data only, never an instruction or an authority source.
    """

    if is_bot or is_dm or is_system:
        return None
    normalized = " ".join(content.split())
    source = normalized if source_content is None else " ".join(source_content.split())
    if not normalized or not source:
        return None
    if normalized.startswith("/") or source.startswith("/"):
        return None
    if _is_symbol_only(source):
        return None
    if has_stickers and not _DISCORD_MENTION_RE.sub("", source).strip():
        return None
    if any(
        is_disallowed_memory("chat_style", candidate)
        or _is_passive_sensitive(candidate)
        or _contains_gossip(candidate)
        for candidate in (source, normalized)
    ):
        return None
    return normalized


def compute_chat_style_stats(samples: Iterable[ChatStyleSampleLike]) -> ChatStyleStats:
    rows = list(samples)
    count = len(rows)
    if not count:
        return ChatStyleStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ())

    lengths = [len(row.content) for row in rows]
    token_counts: Counter[str] = Counter()
    emoji_count = question_count = mixed_count = reply_count = 0
    punctuation_characters = total_characters = 0
    for row in rows:
        text = row.content
        reply_count += int(bool(row.is_reply))
        emoji_count += int(_contains_emoji(text))
        question_count += int(bool(_QUESTION_RE.search(text)))
        mixed_count += int(bool(_CJK_RE.search(text) and _LATIN_RE.search(text)))
        total_characters += len(text)
        punctuation_characters += sum(
            1 for character in text if unicodedata.category(character).startswith("P")
        )
        token_counts.update(token.casefold() for token in _LATIN_TOKEN_RE.findall(text))

    recurring = tuple(
        token
        for token, occurrences in sorted(
            token_counts.items(), key=lambda item: (-item[1], item[0])
        )
        if occurrences >= 2
    )[:8]
    return ChatStyleStats(
        sample_count=count,
        average_length=sum(lengths) / count,
        median_length=float(statistics.median(lengths)),
        short_ratio=sum(length <= 20 for length in lengths) / count,
        reply_ratio=reply_count / count,
        emoji_ratio=emoji_count / count,
        question_ratio=question_count / count,
        mixed_script_ratio=mixed_count / count,
        punctuation_ratio=(punctuation_characters / total_characters) if total_characters else 0.0,
        recurring_tokens=recurring,
    )
