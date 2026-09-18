from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from discord_ai_assistant.ai.memory_v2 import normalize_memory_text


@dataclass(frozen=True, slots=True)
class DomainPack:
    domain: str
    subdomain: str
    aliases: tuple[str, ...]
    topics: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class DomainMatch:
    domain: str
    subdomain: str | None
    topic: str | None
    confidence: float
    source: str
    entity_type: str | None = None
    entity: str | None = None


GAME_PACKS: tuple[DomainPack, ...] = (
    DomainPack(
        "game",
        "lifeafter",
        ("明日之後", "明日之后", "lifeafter", "life after"),
        {
            "dungeon": ("副本", "周本", "活動", "活动"),
            "camp": ("營地", "营地"),
            "cohabitation": ("同居",),
            "profession": ("職業", "职业"),
            "equipment": ("武器", "護甲", "护甲", "裝備", "装备"),
        },
    ),
    DomainPack(
        "game",
        "genshin",
        ("原神", "genshin impact", "genshin"),
        {
            "gacha": ("這池", "这池", "卡池", "抽卡", "祈願", "祈愿", "保底", "歪了", "又歪", "gacha", "banner"),
            "character": ("角色", "命座"),
            "build": ("聖遺物", "圣遗物", "武器", "配隊", "配队"),
            "story": ("劇情", "剧情", "主線", "主线"),
        },
    ),
    DomainPack(
        "game",
        "star_rail",
        ("崩壞：星穹鐵道", "崩坏：星穹铁道", "崩壞:星穹鐵道", "崩坏:星穹铁道", "星穹鐵道", "星穹铁道", "星鐵", "星铁", "honkai: star rail", "honkai star rail", "hsr"),
        {
            "gacha": ("卡池", "抽卡", "保底", "歪了", "又歪", "gacha", "banner"),
            "build": ("遺器", "遗器", "光錐", "光锥"),
            "character": ("角色", "星魂"),
            "team": ("配隊", "配队", "隊伍", "队伍"),
        },
    ),
    DomainPack(
        "game",
        "zzz",
        ("絕區零", "绝区零", "zenless zone zero", "zzz"),
        {
            "gacha": ("調頻", "调频", "抽卡", "卡池", "保底", "歪了", "gacha"),
            "agent": ("代理人", "角色"),
            "build": ("驅動盤", "驱动盘", "音擎"),
        },
    ),
    DomainPack(
        "game",
        "honkai3",
        ("崩壞3", "崩坏3", "崩壞三", "崩坏三", "honkai impact 3rd", "hi3"),
        {
            "gacha": ("補給", "补给", "抽卡", "保底", "gacha"),
            "character": ("女武神", "角色"),
            "build": ("聖痕", "圣痕", "武器"),
        },
    ),
    DomainPack(
        "game",
        "tears_of_themis",
        ("未定事件簿", "tears of themis", "tot"),
        {
            "card": ("卡面", "卡片", "ssr", "mr"),
            "gacha": ("抽卡", "卡池", "保底"),
            "story": ("劇情", "剧情", "路線", "路线"),
        },
    ),
    DomainPack(
        "game",
        "nexus_anima",
        ("honkai: nexus anima", "honkai nexus anima", "nexus anima"),
        {
            "test": ("測試", "测试", "beta"),
            "character": ("角色",),
            "gameplay": ("玩法",),
        },
    ),
    DomainPack(
        "game",
        "petit_planet",
        ("petit planet",),
        {
            "character": ("角色",),
            "gameplay": ("玩法",),
            "progress": ("進度", "进度"),
        },
    ),
    DomainPack(
        "game",
        "counter_strike",
        ("counter-strike", "counter strike", "cs2", "cs"),
        {
            "ranked": ("premier", "faceit", "rating", "排位", "段位"),
            "map": ("mirage", "inferno", "dust2", "nuke", "ancient", "anubis", "vertigo", "overpass"),
            "weapon": ("ak", "ak47", "m4", "awp", "deagle"),
            "utility": ("煙", "烟", "閃", "闪", "molotov", "smoke", "flash"),
        },
    ),
    DomainPack(
        "game",
        "arena_of_valor",
        ("傳說對決", "传说对决", "arena of valor", "aov"),
        {
            "ranked": ("排位", "段位", "排位賽", "排位赛"),
            "hero": ("英雄", "角色"),
            "lane": ("凱撒路", "凯撒路", "魔龍路", "魔龙路", "中路", "打野", "輔助", "辅助"),
            "build": ("出裝", "出装", "奧義", "奥义", "魔紋", "魔纹"),
        },
    ),
    DomainPack(
        "game",
        "minecraft",
        ("minecraft", "mc"),
        {
            "survival": ("生存", "hardcore", "極限", "极限"),
            "server": ("伺服器", "服务器", "server"),
            "redstone": ("紅石", "红石", "redstone"),
            "building": ("建築", "建筑", "蓋房", "盖房"),
            "mod": ("模組", "模组", "modpack", "mod"),
        },
    ),
)

GAME_SUBDOMAINS = frozenset(pack.subdomain for pack in GAME_PACKS)

_FOOD_CUES = (
    "早餐",
    "午餐",
    "晚餐",
    "宵夜",
    "吃什麼",
    "吃什么",
    "燒肉",
    "烧肉",
    "牛肉麵",
    "牛肉面",
    "拉麵",
    "拉面",
    "餐廳",
    "餐厅",
    "飲料",
    "饮料",
    "香菜",
    "火鍋",
    "火锅",
)
_TRAVEL_CUES = ("旅遊", "旅游", "旅行", "自由行", "機票", "机票", "飯店", "酒店")
_REGIONAL_TAIWAN_CUES = ("台灣", "台湾", "臺灣")
_REGIONAL_HONG_KONG_CUES = ("香港", "港式", "粵語", "粤语")
_PROJECT_CUES = ("專案", "project", "pull request", "commit", "merge", "release", "blocker")
_PROJECT_ABBREVIATION_RE = re.compile(r"(?<![a-z0-9])pr(?![a-z0-9])", re.IGNORECASE)
_MIXED_CHANNEL_NAMES = frozenset({"閒聊", "闲聊", "general", "chat", "聊天", "聊天區", "聊天区"})


def _normalized(value: str) -> str:
    return normalize_memory_text(value).casefold()


def _contains_alias(text: str, alias: str) -> bool:
    candidate = _normalized(alias)
    if not candidate:
        return False
    if candidate.isascii() and re.fullmatch(r"[a-z0-9]+", candidate):
        return re.search(rf"(?<![a-z0-9]){re.escape(candidate)}(?![a-z0-9])", text) is not None
    return candidate in text


def _topic_for(pack: DomainPack, text: str) -> str | None:
    for topic, cues in pack.topics.items():
        if any(_contains_alias(text, cue) for cue in cues):
            return topic
    return None


def _game_match(text: str, *, source: str, confidence: float) -> DomainMatch | None:
    normalized = _normalized(text)
    # Longest aliases win so a specific full title beats a shorter overlapping alias.
    candidates: list[tuple[int, DomainPack]] = []
    for pack in GAME_PACKS:
        longest = max((len(alias) for alias in pack.aliases if _contains_alias(normalized, alias)), default=0)
        if longest:
            candidates.append((longest, pack))
    if not candidates:
        return None
    _, pack = max(candidates, key=lambda item: item[0])
    return DomainMatch(
        domain=pack.domain,
        subdomain=pack.subdomain,
        topic=_topic_for(pack, normalized),
        confidence=confidence,
        source=source,
    )


def resolve_explicit_domain(content: str) -> DomainMatch | None:
    """Resolve strong, local message evidence without making an AI request."""

    normalized = _normalized(content)
    if not normalized:
        return None

    game = _game_match(normalized, source="message_alias", confidence=0.95)
    if game is not None:
        return game

    if any(cue in normalized for cue in _FOOD_CUES):
        return DomainMatch("daily", "food", "meal", 0.90, "message_topic")
    if any(cue in normalized for cue in _TRAVEL_CUES):
        return DomainMatch("daily", "travel", "travel", 0.88, "message_topic")
    if any(cue in normalized for cue in _REGIONAL_TAIWAN_CUES):
        return DomainMatch("regional", "taiwan", None, 0.86, "message_topic")
    if any(cue in normalized for cue in _REGIONAL_HONG_KONG_CUES):
        return DomainMatch("regional", "hong_kong", None, 0.86, "message_topic")
    if any(cue in normalized for cue in _PROJECT_CUES) or _PROJECT_ABBREVIATION_RE.search(normalized):
        return DomainMatch("project", None, None, 0.82, "message_topic")
    return None


def infer_channel_prior(channel_name: str, category_name: str | None = None) -> DomainMatch | None:
    """Infer only a weak channel/category prior; message evidence remains authoritative.

    The channel name is more specific than its parent category. A dedicated
    ``#minecraft`` channel inside a generic ``聊天區`` therefore keeps its game prior,
    while an explicitly mixed channel such as ``#閒聊`` stays mixed even inside a
    game-named category.
    """

    channel = _normalized(channel_name)
    category = _normalized(category_name or "")
    if channel in _MIXED_CHANNEL_NAMES:
        return None

    game = _game_match(channel, source="channel_alias", confidence=0.75)
    if game is not None:
        return DomainMatch(game.domain, game.subdomain, None, game.confidence, game.source)

    if category in _MIXED_CHANNEL_NAMES:
        return None

    # A specific game mentioned only in the category may still be a useful weaker prior.
    game = _game_match(category, source="category_alias", confidence=0.68)
    if game is not None:
        return DomainMatch(game.domain, game.subdomain, None, game.confidence, game.source)

    combined = f"{category} {channel}".strip()
    if any(cue in combined for cue in ("開發", "开发", "dev", "project", "專案")):
        return DomainMatch("project", None, None, 0.65, "channel_prior")
    return None
