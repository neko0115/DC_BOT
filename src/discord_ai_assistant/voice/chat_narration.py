from __future__ import annotations

import re
from dataclasses import dataclass

CUSTOM_EMOJI_RE = re.compile(r"<a?:[A-Za-z0-9_~]+:\d+>")
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
REPEATED_CHAR_RE = re.compile(r"(\S)\1{4,}")
WHITESPACE_RE = re.compile(r"\s+")


def _is_emoji_like(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x1F1E6 <= codepoint <= 0x1F1FF
        or 0x1F300 <= codepoint <= 0x1FAFF
        or 0x2600 <= codepoint <= 0x27BF
        or codepoint in {0x200D, 0xFE0E, 0xFE0F}
    )


def _strip_emoji_like(text: str) -> str:
    return "".join(character for character in text if not _is_emoji_like(character))


def sanitize_discord_text(
    content: str,
    *,
    has_stickers: bool = False,
    max_characters: int | None = None,
) -> str | None:
    """Return a compact human-readable form of Discord text for TTS/context.

    Custom emoji markup and Unicode emoji-only noise are removed, URLs are shortened,
    excessive repeated characters are collapsed, and pure sticker/emoji messages are
    intentionally skipped.
    """

    without_markup = CUSTOM_EMOJI_RE.sub(" ", content)
    semantic_without_links = URL_RE.sub(" ", without_markup)
    semantic_without_links = _strip_emoji_like(semantic_without_links)
    if has_stickers and not WHITESPACE_RE.sub(" ", semantic_without_links).strip():
        # A pure sticker can arrive with no content or with only client/CDN metadata.
        return None

    text = URL_RE.sub(" 一個連結 ", without_markup)
    text = _strip_emoji_like(text)
    text = REPEATED_CHAR_RE.sub(lambda match: match.group(1) * 3, text)
    text = WHITESPACE_RE.sub(" ", text).strip()

    if not text:
        return None
    if max_characters is not None and len(text) > max_characters:
        return None
    return text


@dataclass(slots=True)
class _SpeakerState:
    author_id: int
    last_spoken_at: float


class VoiceChatNarrator:
    """Build natural TTS announcements while remembering short speaker turns."""

    def __init__(self, *, continuity_seconds: float = 30.0, max_characters: int = 180) -> None:
        self.continuity_seconds = max(0.0, float(continuity_seconds))
        self.max_characters = max(1, int(max_characters))
        self._speakers: dict[tuple[int, int], _SpeakerState] = {}

    def render(
        self,
        *,
        guild_id: int,
        channel_id: int,
        author_id: int,
        author_name: str,
        content: str,
        now: float,
        has_stickers: bool = False,
    ) -> str | None:
        text = sanitize_discord_text(
            content,
            has_stickers=has_stickers,
            max_characters=self.max_characters,
        )
        if not text:
            return None

        key = (guild_id, channel_id)
        previous = self._speakers.get(key)
        continuing = bool(
            previous
            and previous.author_id == author_id
            and now - previous.last_spoken_at <= self.continuity_seconds
        )
        self._speakers[key] = _SpeakerState(author_id=author_id, last_spoken_at=now)

        if continuing:
            return text
        return f"{author_name}說，{text}"
