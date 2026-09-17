from __future__ import annotations

from dataclasses import dataclass


MEETING_GLOSSARY_REVISION = "2026-08-25-lifeafter-v1"


@dataclass(frozen=True, slots=True)
class MeetingGlossary:
    name: str
    title_hints: tuple[str, ...]
    transcript_hints: tuple[str, ...]
    terms: tuple[str, ...]
    replacements: tuple[tuple[str, str], ...]


LIFEAFTER_GLOSSARY = MeetingGlossary(
    name="lifeafter",
    title_hints=("明日之後", "LifeAfter", "lifeafter", "高校"),
    transcript_hints=("高校", "爭霸賽", "活力點", "四面楚歌", "無人機"),
    terms=(
        "明日之後",
        "高校",
        "爭霸賽",
        "海域 3",
        "浴血盾牌",
        "肉魚",
        "破盾強攻",
        "屑骨人",
        "晶蝶無人機",
        "黑馬",
        "航海任務",
        "航線任務",
        "迷霧的 boss",
        "三王",
        "活力點",
        "白夜",
        "新興",
        "LINE",
        "DC",
    ),
    replacements=(
        ("卡西 3", "海域 3"),
        ("卡西3", "海域 3"),
        ("預寫", "浴血"),
        ("預血", "浴血"),
        ("肉語", "肉魚"),
        ("謝古人", "屑骨人"),
        ("蝴蝶無人機", "晶蝶無人機"),
        ("黑毛", "黑馬"),
        ("信息", "新興"),
        ("迷糊的 Force", "迷霧的 boss"),
        ("迷糊的 force", "迷霧的 boss"),
        ("三網", "三王"),
    ),
)


GLOSSARIES = (LIFEAFTER_GLOSSARY,)


def glossary_for_title(title: str) -> MeetingGlossary | None:
    normalized = title.strip()
    if not normalized:
        return None
    for glossary in GLOSSARIES:
        if any(hint in normalized for hint in glossary.title_hints):
            return glossary
    return None


def glossary_for_transcript(text: str) -> MeetingGlossary | None:
    for glossary in GLOSSARIES:
        matches = sum(1 for hint in glossary.transcript_hints if hint in text)
        if matches >= 2:
            return glossary
    return None


def enrich_initial_prompt(base_prompt: str | None, glossary: MeetingGlossary | None) -> str | None:
    base = (base_prompt or "").strip()
    if glossary is None:
        return base or None
    glossary_prompt = "可能出現的遊戲專有名詞：" + "、".join(glossary.terms) + "。"
    return f"{base} {glossary_prompt}".strip()


def apply_glossary_corrections(text: str, glossary: MeetingGlossary | None) -> tuple[str, list[dict[str, str]]]:
    if glossary is None:
        return text, []
    corrected = text
    applied: list[dict[str, str]] = []
    for source, target in glossary.replacements:
        count = corrected.count(source)
        if count <= 0:
            continue
        corrected = corrected.replace(source, target)
        applied.append({"from": source, "to": target, "count": str(count)})
    return corrected, applied
