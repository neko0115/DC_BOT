from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable


DEFAULT_PROFILE_STATE_PREFIX = "app_knowledge_default:"


@dataclass(frozen=True, slots=True)
class AppKnowledgeProfile:
    key: str
    display_name: str
    description: str
    hints: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AppKnowledgeTerm:
    term: str
    explanation: str
    aliases: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AppKnowledgeSnapshot:
    profile: AppKnowledgeProfile
    terms: tuple[AppKnowledgeTerm, ...]

    def whisper_terms(self, limit: int = 80) -> list[str]:
        values: list[str] = []
        for item in self.terms:
            values.append(item.term)
            values.extend(item.aliases)
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = value.strip()
            folded = normalized.casefold()
            if normalized and folded not in seen:
                seen.add(folded)
                result.append(normalized)
            if len(result) >= limit:
                break
        return result

    def explanation_block(self, limit: int = 80) -> str:
        lines = [
            f"領域：{self.profile.display_name} ({self.profile.key})",
            f"領域說明：{self.profile.description or '未提供'}",
            "以下詞庫只用來理解逐字稿中的專有名詞，不代表本次會議實際發生的事件：",
        ]
        for item in self.terms[:limit]:
            alias_text = f"；可能別名/常見誤辨識：{', '.join(item.aliases)}" if item.aliases else ""
            explanation = item.explanation or "未提供解釋"
            lines.append(f"- {item.term}：{explanation}{alias_text}")
        return "\n".join(lines)


class AppKnowledgeStore:
    """Guild-scoped terminology database shared by meeting transcription and other apps."""

    def __init__(self, database: Any) -> None:
        self.database = database
        self.connection = database.connection
        self._migrate()

    def _migrate(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS app_knowledge_profiles (
                id INTEGER PRIMARY KEY,
                guild_id INTEGER NOT NULL,
                profile_key TEXT NOT NULL COLLATE NOCASE,
                display_name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                hints_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(guild_id, profile_key)
            );
            CREATE INDEX IF NOT EXISTS idx_app_knowledge_profiles_guild
                ON app_knowledge_profiles(guild_id, profile_key);
            CREATE TABLE IF NOT EXISTS app_knowledge_terms (
                id INTEGER PRIMARY KEY,
                profile_id INTEGER NOT NULL REFERENCES app_knowledge_profiles(id) ON DELETE CASCADE,
                term TEXT NOT NULL COLLATE NOCASE,
                explanation TEXT NOT NULL DEFAULT '',
                aliases_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(profile_id, term)
            );
            CREATE INDEX IF NOT EXISTS idx_app_knowledge_terms_profile
                ON app_knowledge_terms(profile_id, term);
            """
        )
        self.connection.commit()

    @staticmethod
    def normalize_key(value: str) -> str:
        normalized = re.sub(r"[^0-9A-Za-z._-]+", "-", value.strip().lower()).strip("-._")
        if not normalized or len(normalized) > 48:
            raise ValueError("profile key 需為 1 到 48 字元的英數、.-_ 組合。")
        return normalized

    @staticmethod
    def _clean_text(value: str, *, label: str, maximum: int, allow_empty: bool = False) -> str:
        normalized = " ".join(value.split())
        if not normalized and not allow_empty:
            raise ValueError(f"{label}不可為空。")
        if len(normalized) > maximum:
            raise ValueError(f"{label}不可超過 {maximum} 個字元。")
        return normalized

    @staticmethod
    def parse_list(value: str | Iterable[str] | None, *, maximum_items: int = 24) -> tuple[str, ...]:
        if value is None:
            return ()
        raw_items = re.split(r"[,，\n;；]+", value) if isinstance(value, str) else list(value)
        result: list[str] = []
        seen: set[str] = set()
        for raw in raw_items:
            item = " ".join(str(raw).split())
            folded = item.casefold()
            if not item or folded in seen:
                continue
            seen.add(folded)
            result.append(item[:120])
            if len(result) >= maximum_items:
                break
        return tuple(result)

    def upsert_profile(
        self,
        guild_id: int,
        key: str,
        display_name: str,
        description: str = "",
        hints: str | Iterable[str] | None = None,
    ) -> AppKnowledgeProfile:
        normalized_key = self.normalize_key(key)
        name = self._clean_text(display_name, label="顯示名稱", maximum=80)
        desc = self._clean_text(description, label="領域說明", maximum=500, allow_empty=True)
        existing = self.connection.execute(
            "SELECT hints_json FROM app_knowledge_profiles WHERE guild_id = ? AND profile_key = ?",
            (guild_id, normalized_key),
        ).fetchone()
        if hints is None and existing is not None:
            hints_json = str(existing["hints_json"])
        else:
            hints_json = json.dumps(self.parse_list(hints), ensure_ascii=False)
        self.connection.execute(
            """INSERT INTO app_knowledge_profiles(guild_id, profile_key, display_name, description, hints_json)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(guild_id, profile_key) DO UPDATE SET
                 display_name = excluded.display_name,
                 description = excluded.description,
                 hints_json = excluded.hints_json,
                 updated_at = CURRENT_TIMESTAMP""",
            (guild_id, normalized_key, name, desc, hints_json),
        )
        self.connection.commit()
        profile = self.get_profile(guild_id, normalized_key)
        assert profile is not None
        return profile

    def get_profile(self, guild_id: int, key: str) -> AppKnowledgeProfile | None:
        normalized_key = self.normalize_key(key)
        row = self.connection.execute(
            "SELECT * FROM app_knowledge_profiles WHERE guild_id = ? AND profile_key = ?",
            (guild_id, normalized_key),
        ).fetchone()
        return self._profile(row) if row else None

    def list_profiles(self, guild_id: int) -> list[AppKnowledgeProfile]:
        rows = self.connection.execute(
            "SELECT * FROM app_knowledge_profiles WHERE guild_id = ? ORDER BY display_name COLLATE NOCASE, profile_key",
            (guild_id,),
        ).fetchall()
        return [self._profile(row) for row in rows]

    def set_default_profile(self, guild_id: int, key: str) -> AppKnowledgeProfile:
        profile = self.get_profile(guild_id, key)
        if profile is None:
            raise ValueError("找不到指定的 Knowledge Profile。")
        self.database.set_state(f"{DEFAULT_PROFILE_STATE_PREFIX}{guild_id}", profile.key)
        return profile

    def default_profile(self, guild_id: int) -> AppKnowledgeProfile | None:
        key = self.database.get_state(f"{DEFAULT_PROFILE_STATE_PREFIX}{guild_id}")
        return self.get_profile(guild_id, key) if key else None

    def resolve_profile(self, guild_id: int, text: str) -> AppKnowledgeProfile | None:
        normalized_text = " ".join(text.split()).casefold()
        best: tuple[int, AppKnowledgeProfile] | None = None
        for profile in self.list_profiles(guild_id):
            candidates = (profile.key, profile.display_name, *profile.hints)
            score = sum(1 for candidate in candidates if candidate and candidate.casefold() in normalized_text)
            if score and (best is None or score > best[0]):
                best = (score, profile)
        return best[1] if best else self.default_profile(guild_id)

    def upsert_term(
        self,
        guild_id: int,
        profile_key: str,
        term: str,
        explanation: str,
        aliases: str | Iterable[str] | None = None,
    ) -> AppKnowledgeTerm:
        profile_id = self._profile_id(guild_id, profile_key)
        canonical = self._clean_text(term, label="專有名詞", maximum=120)
        meaning = self._clean_text(explanation, label="簡短解釋", maximum=500)
        existing = self.connection.execute(
            "SELECT aliases_json FROM app_knowledge_terms WHERE profile_id = ? AND term = ?",
            (profile_id, canonical),
        ).fetchone()
        if aliases is None and existing is not None:
            aliases_json = str(existing["aliases_json"])
        else:
            parsed_aliases = tuple(
                alias for alias in self.parse_list(aliases) if alias.casefold() != canonical.casefold()
            )
            aliases_json = json.dumps(parsed_aliases, ensure_ascii=False)
        self.connection.execute(
            """INSERT INTO app_knowledge_terms(profile_id, term, explanation, aliases_json)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(profile_id, term) DO UPDATE SET
                 explanation = excluded.explanation,
                 aliases_json = excluded.aliases_json,
                 updated_at = CURRENT_TIMESTAMP""",
            (profile_id, canonical, meaning, aliases_json),
        )
        self.connection.commit()
        row = self.connection.execute(
            "SELECT * FROM app_knowledge_terms WHERE profile_id = ? AND term = ?",
            (profile_id, canonical),
        ).fetchone()
        assert row is not None
        return self._term(row)

    def get_term(self, guild_id: int, profile_key: str, term: str) -> AppKnowledgeTerm | None:
        profile_id = self._profile_id(guild_id, profile_key)
        row = self.connection.execute(
            "SELECT * FROM app_knowledge_terms WHERE profile_id = ? AND term = ?",
            (profile_id, " ".join(term.split())),
        ).fetchone()
        return self._term(row) if row else None

    def remove_term(self, guild_id: int, profile_key: str, term: str) -> bool:
        profile_id = self._profile_id(guild_id, profile_key)
        cursor = self.connection.execute(
            "DELETE FROM app_knowledge_terms WHERE profile_id = ? AND term = ?",
            (profile_id, " ".join(term.split())),
        )
        self.connection.commit()
        return cursor.rowcount > 0

    def list_terms(self, guild_id: int, profile_key: str) -> list[AppKnowledgeTerm]:
        profile_id = self._profile_id(guild_id, profile_key)
        rows = self.connection.execute(
            "SELECT * FROM app_knowledge_terms WHERE profile_id = ? ORDER BY term COLLATE NOCASE",
            (profile_id,),
        ).fetchall()
        return [self._term(row) for row in rows]

    def snapshot(self, guild_id: int, text: str) -> AppKnowledgeSnapshot | None:
        profile = self.resolve_profile(guild_id, text)
        if profile is None:
            return None
        return AppKnowledgeSnapshot(profile=profile, terms=tuple(self.list_terms(guild_id, profile.key)))

    def ensure_lifeafter_seed(self, guild_id: int) -> None:
        if self.get_profile(guild_id, "lifeafter") is None:
            self.upsert_profile(
                guild_id,
                "lifeafter",
                "明日之後",
                "《明日之後》遊戲相關的團隊討論、活動、裝備、玩家名稱與 Boss 戰術用語。",
                ("明日之後", "LifeAfter", "高校", "爭霸賽"),
            )
        seed_terms = (
            ("高校", "遊戲活動／關卡內容的常用簡稱。", ()),
            ("爭霸賽", "遊戲內競賽活動名稱。", ()),
            ("海域 3", "會議中使用的海域關卡名稱。", ("卡西 3", "卡西3")),
            ("浴血盾牌", "會議中討論的盾牌／裝備稱呼；與玩家「肉魚」相關。", ("預寫", "預血")),
            ("肉魚", "玩家名稱。", ("肉語",)),
            ("破盾強攻", "會議中使用的戰鬥效果／機制名稱。", ()),
            ("屑骨人", "天賦名稱；會議中描述其爆發效果會扣血而不是扣盾。", ("謝古人",)),
            ("晶蝶無人機", "無人機名稱。", ("蝴蝶無人機",)),
            ("黑馬", "此處表示本季表現出乎預期地強，不是角色名稱。", ("黑毛",)),
            ("航海任務", "本號保留在伺服器時所討論的任務名稱。", ()),
            ("航線任務", "小號移動後所討論的任務名稱。", ()),
            ("活力點", "遊戲內資源／行動點數名稱。", ()),
            ("白夜", "玩家名稱。", ()),
            ("新興", "會議中提到的收費打手／玩家名稱。", ("信息",)),
            ("迷霧的 boss", "會議中 Boss 戰相關名稱。", ("迷糊的 Force", "迷糊的 force")),
            ("三王", "第三個 Boss 的口語稱呼。", ("三網",)),
            ("LINE", "用來聯絡成員的通訊平台名稱。", ()),
            ("DC", "Discord 的口語簡稱。", ()),
        )
        for term, explanation, aliases in seed_terms:
            if self.get_term(guild_id, "lifeafter", term) is None:
                self.upsert_term(guild_id, "lifeafter", term, explanation, aliases)
        if self.default_profile(guild_id) is None:
            self.set_default_profile(guild_id, "lifeafter")

    def _profile_id(self, guild_id: int, key: str) -> int:
        normalized_key = self.normalize_key(key)
        row = self.connection.execute(
            "SELECT id FROM app_knowledge_profiles WHERE guild_id = ? AND profile_key = ?",
            (guild_id, normalized_key),
        ).fetchone()
        if row is None:
            raise ValueError("找不到指定的 Knowledge Profile。")
        return int(row["id"])

    @staticmethod
    def _profile(row: Any) -> AppKnowledgeProfile:
        try:
            hints = tuple(str(item) for item in json.loads(str(row["hints_json"])))
        except (TypeError, ValueError, json.JSONDecodeError):
            hints = ()
        return AppKnowledgeProfile(
            key=str(row["profile_key"]),
            display_name=str(row["display_name"]),
            description=str(row["description"]),
            hints=hints,
        )

    @staticmethod
    def _term(row: Any) -> AppKnowledgeTerm:
        try:
            aliases = tuple(str(item) for item in json.loads(str(row["aliases_json"])))
        except (TypeError, ValueError, json.JSONDecodeError):
            aliases = ()
        return AppKnowledgeTerm(
            term=str(row["term"]), explanation=str(row["explanation"]), aliases=aliases
        )
