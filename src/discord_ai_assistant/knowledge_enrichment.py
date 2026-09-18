from __future__ import annotations

import asyncio
import ipaddress
import re
import unicodedata
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable
from urllib.parse import unquote, urlsplit

from idna import IDNAError, encode as idna_encode, uts46_remap

from discord_ai_assistant.ai.chat_style import ChatStyleSample, prepare_chat_style_content
from discord_ai_assistant.app_knowledge import AppKnowledgeStore
from discord_ai_assistant.ai.memory import is_disallowed_memory
from discord_ai_assistant.ai.memory_phase2 import _contains_gossip, _is_passive_sensitive
from discord_ai_assistant.chat_style_store import (
    ChatStyleStore, PersonalChatTerm, normalize_term_text,
)


GUILD_LEXICON_PROFILE_KEY = "guild-lexicon"


@dataclass(frozen=True, slots=True)
class GuildLexiconEntry:
    id: int
    guild_id: int
    term: str
    meaning: str
    status: str
    distinct_user_count: int
    distinct_session_count: int


class KnowledgeEnrichmentStore:
    """Private C terms plus evidence metadata for AppKnowledge-owned guild terms."""

    def __init__(self, database: Any, app_knowledge: AppKnowledgeStore) -> None:
        self.connection = database.connection
        self.chat_style = ChatStyleStore(database)
        self.app_knowledge = app_knowledge
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS guild_lexicon_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                guild_id INTEGER NOT NULL,
                term_key TEXT NOT NULL,
                meaning_key TEXT NOT NULL,
                confidence REAL NOT NULL,
                first_observed_at TEXT NOT NULL,
                last_observed_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK(status IN ('pending', 'candidate', 'active')),
                UNIQUE(guild_id, term_key, meaning_key),
                UNIQUE(guild_id, id)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_guild_lexicon_active_term
                ON guild_lexicon_entries(guild_id, term_key) WHERE status = 'active';
            CREATE TABLE IF NOT EXISTS guild_lexicon_evidence (
                guild_id INTEGER NOT NULL,
                entry_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                source_sample_id INTEGER NOT NULL,
                session_key TEXT,
                observed_at TEXT NOT NULL,
                PRIMARY KEY(guild_id, entry_id, source_sample_id),
                FOREIGN KEY(guild_id, entry_id)
                    REFERENCES guild_lexicon_entries(guild_id, id) ON DELETE CASCADE
            );
            """
        )
        # Evidence intentionally outlives the raw sample; no sample FK or TTL.
        self.connection.commit()

    def register_personal_understanding(
        self, guild_id: int, user_id: int, *, term: str, meaning: str,
        confidence: float, samples: list[ChatStyleSample], now: datetime | None = None,
    ) -> PersonalChatTerm | None:
        if not isinstance(term, str) or not isinstance(meaning, str):
            return None
        term, meaning = normalize_term_text(term), normalize_term_text(meaning)
        if (not term or not meaning or len(term) > 80 or len(meaning) > 240
                or isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                or not 0 <= confidence <= 1
                or prepare_chat_style_content(f"{term} {meaning}") is None):
            return None
        now = now or datetime.now(timezone.utc)
        valid = self.chat_style.public_term_samples(guild_id, user_id, term, samples, now=now)
        if not valid:
            return None
        personal = self.chat_style.register_personal_term(
            guild_id, user_id, term=term, meaning=meaning, confidence=confidence,
            samples=valid, now=now,
        )
        # Guild consensus is independent of the per-user three-usage threshold.
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO guild_lexicon_entries
                    (guild_id, term_key, meaning_key, confidence, first_observed_at, last_observed_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(guild_id, term_key, meaning_key) DO UPDATE SET
                    confidence = MIN(confidence, excluded.confidence),
                    first_observed_at = MIN(first_observed_at, excluded.first_observed_at),
                    last_observed_at = MAX(last_observed_at, excluded.last_observed_at)
                """,
                (guild_id, term, meaning, confidence,
                 min(s.observed_at for s in valid).isoformat(timespec="microseconds"),
                 max(s.observed_at for s in valid).isoformat(timespec="microseconds")),
            )
            entry_id = int(self.connection.execute(
                "SELECT id FROM guild_lexicon_entries WHERE guild_id = ? AND term_key = ? AND meaning_key = ?",
                (guild_id, term, meaning),
            ).fetchone()[0])
            self.connection.executemany(
                """
                INSERT OR IGNORE INTO guild_lexicon_evidence
                    (guild_id, entry_id, user_id, source_sample_id, session_key, observed_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [(guild_id, entry_id, user_id, sample.id,
                  sample.session_key.strip() if sample.session_key and sample.session_key.strip() else None,
                  sample.observed_at.isoformat(timespec="microseconds")) for sample in valid],
            )
        self._promote(guild_id, entry_id, term, meaning)
        return personal

    def get_personal_term(self, guild_id: int, user_id: int, term: str) -> PersonalChatTerm | None:
        return self.chat_style.get_personal_term(guild_id, user_id, term)

    def _counts(self, guild_id: int, entry_id: int) -> tuple[int, int]:
        row = self.connection.execute(
            "SELECT COUNT(DISTINCT user_id), COUNT(DISTINCT session_key) "
            "FROM guild_lexicon_evidence WHERE guild_id = ? AND entry_id = ?",
            (guild_id, entry_id),
        ).fetchone()
        return int(row[0]), int(row[1])

    def _promote(self, guild_id: int, entry_id: int, term: str, meaning: str) -> None:
        users, sessions = self._counts(guild_id, entry_id)
        if users < 2:
            return
        with self.connection:
            self.connection.execute(
                "UPDATE guild_lexicon_entries SET status = 'candidate' "
                "WHERE guild_id = ? AND id = ? AND status = 'pending'",
                (guild_id, entry_id),
            )
        if users < 3 and sessions < 2:
            return
        active = self.connection.execute(
            "SELECT id FROM guild_lexicon_entries WHERE guild_id = ? AND term_key = ? AND status = 'active'",
            (guild_id, term),
        ).fetchone()
        if active is not None and int(active[0]) != entry_id:
            return  # Conflicting later consensus cannot overwrite an active meaning.
        profile = self.app_knowledge.get_profile(guild_id, GUILD_LEXICON_PROFILE_KEY)
        if profile is not None:
            existing = self.app_knowledge.get_term(guild_id, GUILD_LEXICON_PROFILE_KEY, term)
            if existing is not None and normalize_term_text(existing.explanation) != meaning:
                return
        else:
            self.app_knowledge.upsert_profile(guild_id, GUILD_LEXICON_PROFILE_KEY, "Guild Lexicon")
        self.app_knowledge.upsert_term(guild_id, GUILD_LEXICON_PROFILE_KEY, term, meaning)
        # AppKnowledge commits its own writes; mark active only after they succeed.
        with self.connection:
            self.connection.execute(
                "UPDATE guild_lexicon_entries SET status = 'active' WHERE guild_id = ? AND id = ?",
                (guild_id, entry_id),
            )

    def _entry(self, row: Any) -> GuildLexiconEntry | None:
        entry_id, guild_id, term, meaning, status = row
        if status == "active":
            if self.app_knowledge.get_profile(guild_id, GUILD_LEXICON_PROFILE_KEY) is None:
                return None
            canonical = self.app_knowledge.get_term(guild_id, GUILD_LEXICON_PROFILE_KEY, term)
            if canonical is None:
                return None
            term, meaning = canonical.term, canonical.explanation
        users, sessions = self._counts(guild_id, entry_id)
        return GuildLexiconEntry(entry_id, guild_id, term, meaning, status, users, sessions)

    def get_guild_lexicon_entry(
        self, guild_id: int, term: str, *, include_candidate: bool = False,
    ) -> GuildLexiconEntry | None:
        row = self.connection.execute(
            """
            SELECT id, guild_id, term_key, meaning_key, status FROM guild_lexicon_entries
            WHERE guild_id = ? AND term_key = ?
                AND (status = 'active' OR (? AND status = 'candidate'))
            ORDER BY (status = 'active') DESC, id ASC LIMIT 1
            """,
            (guild_id, normalize_term_text(term), include_candidate),
        ).fetchone()
        return None if row is None else self._entry(row)

    def search_guild_lexicon(self, guild_id: int, query: str) -> list[GuildLexiconEntry]:
        rows = self.connection.execute(
            "SELECT id, guild_id, term_key, meaning_key, status FROM guild_lexicon_entries "
            "WHERE guild_id = ? AND status = 'active' AND instr(term_key, ?) > 0 ORDER BY term_key LIMIT 25",
            (guild_id, normalize_term_text(query)),
        ).fetchall()
        return [entry for row in rows if (entry := self._entry(row)) is not None]

    def forget_guild_lexicon_entry(self, guild_id: int, entry_id: int) -> bool:
        row = self.connection.execute(
            "SELECT term_key, status FROM guild_lexicon_entries WHERE guild_id = ? AND id = ?",
            (guild_id, entry_id),
        ).fetchone()
        if row is None:
            return False
        if row[1] == "active" and self.app_knowledge.get_profile(guild_id, GUILD_LEXICON_PROFILE_KEY) is not None:
            self.app_knowledge.remove_term(guild_id, GUILD_LEXICON_PROFILE_KEY, row[0])
        with self.connection:
            self.connection.execute(
                "DELETE FROM guild_lexicon_evidence WHERE guild_id = ? AND entry_id = ?", (guild_id, entry_id),
            )
            self.connection.execute(
                "DELETE FROM guild_lexicon_entries WHERE guild_id = ? AND id = ?", (guild_id, entry_id),
            )
        return True


PUBLIC_MIN_CONFIDENCE = 0.8
PUBLIC_WEB_TIMEOUT_SECONDS = 15.0
PUBLIC_QUERY_LIMIT = 320
_PUBLIC_PRIVATE_MARKERS = re.compile(
    r"\b(?:username|user[_ -]?id|guild[_ -]?(?:id|name)|discord|private|secret|credentials?|"
    r"address|diagnosis|health|politic\w*|salary|bank account|relationship|rumou?r|my|our)\b"
    r"|(?:住在|分手|健康|財務|私密|私人|伺服器名稱|使用者名稱)"
    r"|(?:ignore|override|disregard).{0,32}(?:instructions?|rules?|system)"
    r"|(?:system|developer|persona|tool)[_ -]?(?:prompt|instruction|permission)"
    r"|\d{6,}", re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class PublicKnowledgeKey:
    """Extracted lookup term plus public metadata, never a Discord message or owner.

    The term may be unknown but safe to probe; callers attest web safety separately.
    """
    canonical_term: str
    domain: str = ""
    subdomain: str = ""
    locale: str = ""
    version: str = ""
    context: str = ""
    version_sensitive: bool = False

    def values(self) -> tuple:
        return (self.canonical_term, self.domain, self.subdomain, self.locale,
                self.version, self.context, self.version_sensitive)


def _safe_public_text(value: str) -> bool:
    return not (is_disallowed_memory("public_knowledge", value)
                or _is_passive_sensitive(value) or _contains_gossip(value)
                or _PUBLIC_PRIVATE_MARKERS.search(value))


def _normalized_public_key(key: PublicKnowledgeKey) -> PublicKnowledgeKey | None:
    if not isinstance(key, PublicKnowledgeKey) or type(key.version_sensitive) is not bool:
        return None
    values = []
    for value, limit in zip(key.values()[:-1], (80, 32, 48, 16, 32, 64)):
        if (not isinstance(value, str) or len(value) > limit
                or any(unicodedata.category(char).startswith("C") for char in value)):
            return None
        value = normalize_term_text(unicodedata.normalize("NFKC", value))
        if (len(value) > limit or len(value.split()) > 8
                or (value and (re.fullmatch(r"[\w .+#/-]+", value) is None or value.startswith("/")))
                or not _safe_public_text(value)):
            return None
        values.append(value)
    if not values[0] or not _safe_public_text(" ".join(values)):
        return None
    if key.version_sensitive and not (values[4] or values[5]):
        return None  # Current-state requests need an explicit public discriminator.
    return PublicKnowledgeKey(*values, key.version_sensitive)


def build_public_query(key: PublicKnowledgeKey, *, web_safe: bool = False) -> str | None:
    """Build a bounded query only when the caller attests it is safe for public web.

    The term may be unknown; domain/version/locale/context must be public metadata.
    Sanitization is an independent boundary, not proof of arbitrary names' provenance.
    """
    if web_safe is not True or (key := _normalized_public_key(key)) is None:
        return None
    query = " ".join(value for value in key.values()[:-1] if value) + " terminology"
    return query if len(query) <= PUBLIC_QUERY_LIMIT else None


@dataclass(frozen=True, slots=True)
class PublicSource:
    url: str
    title: str
    meaning: str


@dataclass(frozen=True, slots=True)
class PublicWebResult:
    canonical_term: str
    meaning: str
    confidence: float
    sources: tuple[PublicSource, ...]
    aliases: tuple[str, ...] = ()
    ambiguous: bool = False
    conflicting: bool = False


@dataclass(frozen=True, slots=True)
class PublicKnowledgeEntry:
    id: int
    key: PublicKnowledgeKey
    meaning: str
    aliases: tuple[str, ...]
    confidence: float
    verified_at: datetime
    expires_at: datetime
    sources: tuple[PublicSource, ...]
    source_type: str = field(default="web", init=False)
    trust_level: str = field(default="untrusted_reference", init=False)


# Empty 0x components are also numeric: URL parsers can interpret them as zero.
_NUMERIC_HOST_ALIAS = re.compile(r"(?:0x[0-9a-f]*|[0-9]+)(?:\.(?:0x[0-9a-f]*|[0-9]+))*")


def _canonical_public_hostname(value: str) -> str | None:
    if not isinstance(value, str) or not value or len(value) > 2048:
        return None
    try:
        parsed = urlsplit(value)
        host = uts46_remap(unquote(parsed.hostname or "")).rstrip(".").casefold()
        if parsed.scheme not in {"http", "https"} or not host or parsed.username or parsed.password:
            return None
        parsed.port  # Force urlsplit's deferred invalid/range port validation.
        try:
            address = ipaddress.ip_address(host)
            return address.compressed if address.is_global else None
        except ValueError:
            # Browsers/resolvers may interpret legacy decimal, octal or hex
            # hosts as IPs. They must not fall through as public DNS names.
            # UTS46 removes ignored characters that could otherwise hide a
            # numeric component after percent decoding (e.g. a soft hyphen).
            if _NUMERIC_HOST_ALIAS.fullmatch(host):
                return None
            canonical = idna_encode(host).decode("ascii").rstrip(".").casefold()
            if (canonical == "localhost"
                    or canonical.endswith((".local", ".localhost", ".internal"))):
                return None
            return canonical if "." in canonical else None
    except (ValueError, IDNAError):
        return None


def _public_reference_url(value: str) -> bool:
    return _canonical_public_hostname(value) is not None


def _utc_time(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


class PublicKnowledgeStore:
    """Ownerless cache; sources/aliases are replaced atomically with each verification."""

    def __init__(self, database: Any) -> None:
        self.connection = database.connection
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS public_knowledge_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_term TEXT NOT NULL, domain TEXT NOT NULL, subdomain TEXT NOT NULL,
                locale TEXT NOT NULL, version TEXT NOT NULL, context TEXT NOT NULL,
                version_sensitive INTEGER NOT NULL CHECK(version_sensitive IN (0, 1)),
                meaning TEXT NOT NULL, confidence REAL NOT NULL,
                source_type TEXT NOT NULL DEFAULT 'web' CHECK(source_type = 'web'),
                verified_at TEXT NOT NULL, expires_at TEXT NOT NULL,
                UNIQUE(canonical_term, domain, subdomain, locale, version, context, version_sensitive)
            );
            CREATE TABLE IF NOT EXISTS public_knowledge_sources (
                cache_id INTEGER NOT NULL REFERENCES public_knowledge_cache(id) ON DELETE CASCADE,
                url TEXT NOT NULL, title TEXT NOT NULL, meaning TEXT NOT NULL,
                PRIMARY KEY(cache_id, url)
            );
            CREATE TABLE IF NOT EXISTS public_knowledge_aliases (
                cache_id INTEGER NOT NULL REFERENCES public_knowledge_cache(id) ON DELETE CASCADE,
                alias TEXT NOT NULL, PRIMARY KEY(cache_id, alias)
            );
            CREATE INDEX IF NOT EXISTS idx_public_knowledge_alias ON public_knowledge_aliases(alias);
            """
        )
        self.connection.commit()

    def get(
        self, key: PublicKnowledgeKey, *, now: datetime | None = None, include_expired: bool = False,
    ) -> PublicKnowledgeEntry | None:
        key = _normalized_public_key(key)
        if key is None:
            return None
        rows = self.connection.execute(
            """
            SELECT * FROM public_knowledge_cache AS c
            WHERE domain = ? AND subdomain = ? AND locale = ? AND version = ?
                AND context = ? AND version_sensitive = ?
                AND (canonical_term = ? OR EXISTS (
                    SELECT 1 FROM public_knowledge_aliases a WHERE a.cache_id = c.id AND a.alias = ?))
            ORDER BY (canonical_term = ?) DESC
            """, (*key.values()[1:], key.canonical_term, key.canonical_term, key.canonical_term),
        ).fetchall()
        current = _utc_time(now)
        rows = [row for row in rows if include_expired or
                datetime.fromisoformat(row["verified_at"]) <= current < datetime.fromisoformat(row["expires_at"])]
        if not rows or (len(rows) > 1 and rows[0]["canonical_term"] != key.canonical_term):
            return None  # Ambiguous aliases are not guessed.
        row = rows[0]
        sources = tuple(PublicSource(*source) for source in self.connection.execute(
            "SELECT url, title, meaning FROM public_knowledge_sources WHERE cache_id = ? ORDER BY url", (row["id"],)))
        aliases = tuple(alias[0] for alias in self.connection.execute(
            "SELECT alias FROM public_knowledge_aliases WHERE cache_id = ? ORDER BY alias", (row["id"],)))
        return PublicKnowledgeEntry(
            row["id"], PublicKnowledgeKey(*(row[name] for name in (
                "canonical_term", "domain", "subdomain", "locale", "version", "context")), bool(row["version_sensitive"])),
            row["meaning"], aliases, row["confidence"], datetime.fromisoformat(row["verified_at"]),
            datetime.fromisoformat(row["expires_at"]), sources,
        )

    def store_web_result(
        self, key: PublicKnowledgeKey, result: PublicWebResult, *, now: datetime | None = None,
    ) -> PublicKnowledgeEntry | None:
        key = _normalized_public_key(key)
        if key is None or not isinstance(result, PublicWebResult):
            return None
        if (result.ambiguous is not False or result.conflicting is not False
                or isinstance(result.confidence, bool) or not isinstance(result.confidence, (int, float))
                or not PUBLIC_MIN_CONFIDENCE <= result.confidence <= 1.0
                or not isinstance(result.meaning, str) or not 1 <= len(result.meaning) <= 500
                or not _safe_public_text(result.meaning)
                or not isinstance(result.aliases, (tuple, list)) or len(result.aliases) > 12
                or not isinstance(result.sources, (tuple, list)) or not 1 <= len(result.sources) <= 5):
            return None
        canonical = _normalized_public_key(replace(key, canonical_term=result.canonical_term))
        aliases = [_normalized_public_key(replace(key, canonical_term=alias)) for alias in result.aliases]
        if canonical is None or any(alias is None for alias in aliases):
            return None
        alias_terms = {alias.canonical_term for alias in aliases}
        if key.canonical_term not in alias_terms | {canonical.canonical_term}:
            return None  # Never cache an unrelated provider answer under the requested term.
        meaning = " ".join(result.meaning.split())
        if not meaning:
            return None
        for source in result.sources:
            if (not isinstance(source, PublicSource) or not _public_reference_url(source.url)
                    or not isinstance(source.title, str) or not 1 <= len(source.title) <= 200
                    or not _safe_public_text(source.title) or not isinstance(source.meaning, str)
                    or len(source.meaning) > 500 or not _safe_public_text(source.meaning)
                    or normalize_term_text(source.meaning) != normalize_term_text(meaning)):
                return None  # Conservative exact consistency; no semantic reconciliation.
        if len({source.url for source in result.sources}) != len(result.sources):
            return None
        verified = _utc_time(now)
        expires = verified + timedelta(days=7 if canonical.version_sensitive else 90)
        with self.connection:
            cursor = self.connection.execute(
                """
                INSERT INTO public_knowledge_cache
                    (canonical_term, domain, subdomain, locale, version, context, version_sensitive,
                     meaning, confidence, verified_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(canonical_term, domain, subdomain, locale, version, context, version_sensitive)
                DO UPDATE SET meaning = excluded.meaning, confidence = excluded.confidence,
                    verified_at = excluded.verified_at, expires_at = excluded.expires_at
                WHERE excluded.verified_at >= public_knowledge_cache.verified_at
                """, (*canonical.values(), meaning, result.confidence,
                       verified.isoformat(timespec="microseconds"), expires.isoformat(timespec="microseconds")),
            )
            if cursor.rowcount == 0:
                return None  # A delayed verification must not roll back newer knowledge.
            cache_id = self.connection.execute(
                "SELECT id FROM public_knowledge_cache WHERE canonical_term = ? AND domain = ? AND subdomain = ? "
                "AND locale = ? AND version = ? AND context = ? AND version_sensitive = ?", canonical.values(),
            ).fetchone()[0]
            self.connection.execute("DELETE FROM public_knowledge_sources WHERE cache_id = ?", (cache_id,))
            self.connection.execute("DELETE FROM public_knowledge_aliases WHERE cache_id = ?", (cache_id,))
            self.connection.executemany(
                "INSERT INTO public_knowledge_sources(cache_id, url, title, meaning) VALUES (?, ?, ?, ?)",
                [(cache_id, s.url, s.title, s.meaning) for s in result.sources],
            )
            self.connection.executemany(
                "INSERT INTO public_knowledge_aliases(cache_id, alias) VALUES (?, ?)",
                [(cache_id, alias) for alias in sorted(alias_terms - {canonical.canonical_term})],
            )
        return self.get(canonical, now=verified)


@dataclass(slots=True)
class EnrichmentBudget:
    """Create once per enrichment operation/request and share across its resolve calls."""
    _attempted: set[PublicKnowledgeKey] = field(default_factory=set, init=False)

    @property
    def used(self) -> int:
        return len(self._attempted)

    def try_consume(self, key: PublicKnowledgeKey) -> bool:
        key = _normalized_public_key(key)
        if key is None or key in self._attempted or self.used >= 3:
            return False
        self._attempted.add(key)
        return True


@dataclass(frozen=True, slots=True)
class TermResolution:
    source: str
    meaning: str | None = None
    confidence: float = 0.0
    sources: tuple[PublicSource, ...] = ()
    trust_level: str = field(default="untrusted_reference", init=False)


class KnowledgeResolver:
    """Service only: local meanings first; no runtime, commands, or prompt wiring."""

    def __init__(
        self, local: KnowledgeEnrichmentStore, cache: PublicKnowledgeStore,
        web_lookup: Callable[[str], Awaitable[PublicWebResult | None]],
    ) -> None:
        self.local = local
        self.cache = cache
        self.web_lookup = web_lookup

    def _available(
        self, key: PublicKnowledgeKey, guild_id: int, user_id: int, now: datetime | None,
        *, allow_personal: bool | Callable[[], bool] = True,
    ) -> TermResolution | None:
        if self._personal_allowed(allow_personal):
            personal = self.local.get_personal_term(guild_id, user_id, key.canonical_term)
            if personal is not None:
                return TermResolution("personal", personal.meaning, personal.confidence)
        guild = self.local.get_guild_lexicon_entry(guild_id, key.canonical_term)
        if guild is not None:
            return TermResolution("guild", guild.meaning)
        cached = self.cache.get(key, now=now)
        if cached is not None:
            return TermResolution("cache", cached.meaning, cached.confidence, cached.sources)
        return None

    @staticmethod
    def _personal_allowed(policy: bool | Callable[[], bool]) -> bool:
        if callable(policy):
            try:
                return policy() is True
            except Exception:
                return False
        return policy is True

    async def resolve(
        self, key: PublicKnowledgeKey, *, guild_id: int, user_id: int,
        budget: EnrichmentBudget, web_safe: bool = False, materially_relevant: bool = False,
        allow_personal: bool | Callable[[], bool] = True, now: datetime | None = None,
    ) -> TermResolution:
        unresolved = TermResolution("unresolved")
        if not isinstance(key, PublicKnowledgeKey) or not isinstance(key.canonical_term, str):
            return unresolved
        try:
            available = self._available(key, guild_id, user_id, now, allow_personal=allow_personal)
            if available is not None:
                return available
            query = build_public_query(key, web_safe=web_safe)
            if query is None or materially_relevant is not True or not budget.try_consume(key):
                return unresolved
            response = await asyncio.wait_for(self.web_lookup(query), timeout=PUBLIC_WEB_TIMEOUT_SECONDS)
            # Higher-priority knowledge can arrive while the provider is awaited.
            available = self._available(key, guild_id, user_id, now, allow_personal=allow_personal)
            if available is not None:
                return available
            cached = self.cache.store_web_result(key, response, now=now)
            if cached is not None:
                return TermResolution("web", cached.meaning, cached.confidence, cached.sources)
        except Exception:
            # Do not expose provider errors or fall back after a failed local ownership lookup.
            return unresolved
        return unresolved
