from __future__ import annotations

import re

SUPPORTED_TTS_LANGUAGES = ("zh", "yue", "en", "ja", "ko")

_KANA_RE = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff]")
_HANGUL_RE = re.compile(r"[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f]")
_HAN_RE = re.compile(r"[\u3400-\u9fff]")
_LATIN_LETTER_RE = re.compile(r"[A-Za-z]")
_LATIN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")

# Keep Cantonese detection deliberately high-confidence. Ordinary Traditional Chinese
# must default to Mandarin/zh instead of being flipped to yue by generic particles.
_CANTONESE_STRONG_MARKERS = (
    "喺",
    "嘅",
    "唔",
    "冇",
    "佢",
    "佢哋",
    "我哋",
    "你哋",
    "啲",
    "咩",
    "乜",
    "嗰",
    "邊度",
    "點解",
    "點樣",
    "而家",
    "嚟",
    "俾",
    "畀",
    "噉",
    "咁樣",
    "係咪",
    "唔係",
    "唔該",
)
_CANTONESE_WEAK_MARKERS = (
    "喇",
    "啫",
    "囉",
    "㗎",
    "㖭",
    "吖",
    "噃",
    "啱",
)


def normalize_tts_language(value: str | None, *, fallback: str = "zh") -> str:
    """Return a supported GPT-SoVITS target language or raise for invalid explicit input."""

    fallback_value = (fallback or "zh").strip().lower()
    if fallback_value not in SUPPORTED_TTS_LANGUAGES:
        raise ValueError(f"Unsupported fallback TTS language: {fallback_value}")
    if value is None or not value.strip():
        return fallback_value
    normalized = value.strip().lower()
    if normalized not in SUPPORTED_TTS_LANGUAGES:
        allowed = ", ".join(SUPPORTED_TTS_LANGUAGES)
        raise ValueError(f"Unsupported TTS language '{normalized}'. Allowed: {allowed}")
    return normalized


def detect_tts_language(text: str, *, fallback: str = "zh") -> str:
    """Deterministically choose zh/yue/en/ja/ko without another model call.

    This intentionally prefers false negatives for Cantonese over false positives:
    ordinary Traditional Chinese remains `zh` unless clear Cantonese wording appears.
    Japanese and Korean script markers are unambiguous and therefore win first.
    """

    value = text.strip()
    default = normalize_tts_language(None, fallback=fallback)
    if not value:
        return default

    if _KANA_RE.search(value):
        return "ja"
    if _HANGUL_RE.search(value):
        return "ko"

    strong_hits = sum(1 for marker in _CANTONESE_STRONG_MARKERS if marker in value)
    weak_hits = sum(1 for marker in _CANTONESE_WEAK_MARKERS if marker in value)
    if strong_hits >= 1 or weak_hits >= 2:
        return "yue"

    han_count = len(_HAN_RE.findall(value))
    latin_count = len(_LATIN_LETTER_RE.findall(value))
    latin_words = len(_LATIN_WORD_RE.findall(value))

    # A few English model/product names inside a Chinese sentence should not switch
    # the whole utterance to English. Require clear Latin dominance when Han exists.
    if han_count:
        if latin_words >= 3 and latin_count > han_count * 4:
            return "en"
        return "zh"

    if latin_count >= 2:
        return "en"
    return default
