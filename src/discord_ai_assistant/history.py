from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass

import discord

from discord_ai_assistant.voice.chat_narration import sanitize_discord_text

RAW_HISTORY_LIMIT = 100
CONTEXT_RECENT_LIMIT = 20
CONTEXT_OLDER_LIMIT = 20
CONTEXT_RECENT_GUARANTEE = 8
CONTEXT_MAX_MESSAGES = 30
CONTEXT_MAX_CHARACTERS = 8000
CONTEXT_MESSAGE_CHAR_LIMIT = 280
LATIN_TERM_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]*")
CJK_RUN_RE = re.compile(r"[\u3400-\u9fff]+")
NOISE_MESSAGES = {
    "哈哈",
    "哈哈哈",
    "呵呵",
    "lol",
    "lmao",
    "www",
    "www.",
    "xd",
    "...",
    "……",
    "…",
}


@dataclass(frozen=True, slots=True)
class HistoryMessage:
    author_name: str
    text: str

    def render(self) -> str:
        text = self.text
        if len(text) > CONTEXT_MESSAGE_CHAR_LIMIT:
            text = text[: CONTEXT_MESSAGE_CHAR_LIMIT - 1].rstrip() + "…"
        return f"{self.author_name}: {text}"


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _terms(value: str) -> set[str]:
    terms = {match.group(0).casefold() for match in LATIN_TERM_RE.finditer(value) if len(match.group(0)) >= 2}
    for run in CJK_RUN_RE.findall(value):
        if len(run) == 1:
            terms.add(run)
            continue
        terms.update(run[index : index + 2] for index in range(len(run) - 1))
    return terms


def _is_noise(message: HistoryMessage) -> bool:
    value = _normalized_text(message.text).strip("!！?？,.，。~～-_ ")
    if not value:
        return True
    return value in NOISE_MESSAGES


def _is_low_information(message: HistoryMessage) -> bool:
    if _is_noise(message):
        return True
    compact = re.sub(r"[^A-Za-z0-9\u3400-\u9fff]+", "", message.text)
    return len(compact) <= 1


def _render_sections(
    older: list[tuple[int, HistoryMessage]],
    recent: list[tuple[int, HistoryMessage]],
) -> str:
    sections: list[str] = []
    if older:
        ordered = [message for _, message in sorted(older, key=lambda item: item[0])]
        sections.append(
            "較早但可能相關的 Discord 對話：\n"
            + "\n".join(message.render() for message in ordered)
        )
    if recent:
        ordered = [message for _, message in sorted(recent, key=lambda item: item[0])]
        sections.append("最近 Discord 對話：\n" + "\n".join(message.render() for message in ordered))
    return "\n\n".join(sections)


class RecentMessageHistory:
    """In-memory, bounded context only. No conversation text is persisted locally."""

    def __init__(
        self,
        per_channel_limit: int = RAW_HISTORY_LIMIT,
        *,
        recent_limit: int = CONTEXT_RECENT_LIMIT,
        older_limit: int = CONTEXT_OLDER_LIMIT,
        max_messages: int = CONTEXT_MAX_MESSAGES,
        max_characters: int = CONTEXT_MAX_CHARACTERS,
        recent_guarantee: int = CONTEXT_RECENT_GUARANTEE,
    ) -> None:
        self.per_channel_limit = max(1, int(per_channel_limit))
        self.recent_limit = max(1, int(recent_limit))
        self.older_limit = max(0, int(older_limit))
        self.max_messages = max(1, int(max_messages))
        self.max_characters = max(200, int(max_characters))
        self.recent_guarantee = max(1, int(recent_guarantee))
        self._messages: dict[tuple[int, int], deque[HistoryMessage]] = defaultdict(
            lambda: deque(maxlen=self.per_channel_limit)
        )

    def record(self, message: discord.Message) -> None:
        if not message.guild or message.author.bot:
            return
        text = sanitize_discord_text(
            message.clean_content,
            has_stickers=bool(message.stickers),
            max_characters=400,
        )
        if text:
            self.add(message.guild.id, message.channel.id, message.author.display_name, text)

    def add(self, guild_id: int, channel_id: int, author_name: str, text: str) -> None:
        """Append already-normalized text. Primarily useful for deterministic tests and bridges."""
        value = " ".join(text.split()).strip()
        if not value:
            return
        self._messages[(guild_id, channel_id)].append(HistoryMessage(author_name.strip() or "Unknown", value[:400]))

    def format_for(self, guild_id: int, channel_id: int, limit: int = 10) -> str:
        messages = list(self._messages[(guild_id, channel_id)])[-max(1, int(limit)) :]
        if not messages:
            return "（沒有可用的近期對話）"
        return "\n".join(message.render() for message in messages)

    def compressed_for(
        self,
        guild_id: int,
        channel_id: int,
        query: str,
        *,
        recent_limit: int | None = None,
        relevant_limit: int | None = None,
        max_messages: int | None = None,
        max_characters: int | None = None,
        recent_guarantee: int | None = None,
    ) -> str:
        """Build deterministic context bounded by both message count and rendered chars.

        Priority is: a guaranteed slice of the newest useful messages, then the most
        query-relevant older messages, then additional recent messages. Packing stops as
        soon as either the message-count or rendered-character budget would be exceeded.
        """
        raw = list(self._messages[(guild_id, channel_id)])
        if not raw:
            return "（沒有可用的近期對話）"

        recent_cap = self.recent_limit if recent_limit is None else max(1, int(recent_limit))
        older_cap = self.older_limit if relevant_limit is None else max(0, int(relevant_limit))
        message_cap = self.max_messages if max_messages is None else max(1, int(max_messages))
        character_cap = self.max_characters if max_characters is None else max(200, int(max_characters))
        guarantee_cap = self.recent_guarantee if recent_guarantee is None else max(1, int(recent_guarantee))
        guarantee_cap = min(guarantee_cap, recent_cap, message_cap)

        deduped_reversed: list[HistoryMessage] = []
        seen: set[tuple[str, str]] = set()
        for message in reversed(raw):
            key = (_normalized_text(message.author_name), _normalized_text(message.text))
            if key in seen:
                continue
            seen.add(key)
            deduped_reversed.append(message)
        messages = list(reversed(deduped_reversed))
        indexed = list(enumerate(messages))

        recent_indexed = [
            item for item in indexed[-recent_cap:] if not _is_noise(item[1])
        ]
        older_pool = indexed[: max(0, len(indexed) - recent_cap)]
        older_indexed = [item for item in older_pool if not _is_low_information(item[1])]

        query_terms = _terms(query)
        scored: list[tuple[int, int, HistoryMessage]] = []
        for index, message in older_indexed:
            overlap = len(query_terms.intersection(_terms(message.text))) if query_terms else 0
            scored.append((overlap, index, message))
        ranked_older = sorted(scored, key=lambda item: (item[0], item[1]), reverse=True)[:older_cap]

        guaranteed_recent = recent_indexed[-guarantee_cap:]
        guaranteed_ids = {index for index, _ in guaranteed_recent}
        additional_recent = [
            item for item in reversed(recent_indexed) if item[0] not in guaranteed_ids
        ]

        # section, index, message in actual packing priority
        priority: list[tuple[str, int, HistoryMessage]] = []
        priority.extend(("recent", index, message) for index, message in reversed(guaranteed_recent))
        priority.extend(("older", index, message) for _, index, message in ranked_older)
        priority.extend(("recent", index, message) for index, message in additional_recent)

        selected_older: list[tuple[int, HistoryMessage]] = []
        selected_recent: list[tuple[int, HistoryMessage]] = []
        selected_ids: set[int] = set()

        for section, index, message in priority:
            if index in selected_ids:
                continue
            if len(selected_ids) >= message_cap:
                break
            target = selected_older if section == "older" else selected_recent
            target.append((index, message))
            candidate = _render_sections(selected_older, selected_recent)
            if len(candidate) > character_cap:
                target.pop()
                break
            selected_ids.add(index)

        rendered = _render_sections(selected_older, selected_recent)
        return rendered or "（沒有可用的近期對話）"
