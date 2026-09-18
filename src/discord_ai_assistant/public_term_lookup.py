from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Awaitable, Callable, Literal
from urllib.parse import quote

from discord.utils import escape_markdown

from discord_ai_assistant.ai.gemini import GeminiAssistant
from discord_ai_assistant.ai.memory_domain_registry import GAME_PACKS, resolve_explicit_domain
from discord_ai_assistant.ai.public_knowledge_search import lookup_public_term_with_search
from discord_ai_assistant.chat_style_store import normalize_term_text
from discord_ai_assistant.knowledge_enrichment import (
    EnrichmentBudget, KnowledgeEnrichmentStore, KnowledgeResolver,
    PublicKnowledgeKey, PublicKnowledgeStore, PublicSource, PublicWebResult, TermResolution,
    _normalized_public_key, _public_reference_url, _safe_public_text,
)


@dataclass(frozen=True, slots=True)
class PublicTermLookupIntent:
    term: str
    domain: str = ""
    subdomain: str = ""
    locale: str = "zh-TW"
    version: str = ""
    context: str = ""
    version_sensitive: bool = False


@dataclass(frozen=True, slots=True)
class PublicTermLookupOutcome:
    status: Literal["resolved", "unresolved", "not_applicable"]
    answer: str = ""
    source: str | None = None

    def __post_init__(self) -> None:
        if self.status not in ("resolved", "unresolved", "not_applicable"):
            raise ValueError("Unsupported public-term lookup outcome")


_QUOTES = (("「", "」"), ("“", "”"), ('"', '"'), ("`", "`"))
_QUOTE_CHARACTERS = frozenset(char for pair in _QUOTES for char in pair)
_COURTESY = re.compile(r"^(?:請問|想問|幫我查網路上) *")
_DEFINITIONS = (
    re.compile(r"(?P<body>.+?) *(?:是什麼(?:意思|縮寫)?|什麼意思|是啥|指什麼|代表什麼)"),
    re.compile(r"什麼是 *(?P<body>.+)"),
    re.compile(r"what is +(?P<body>.+)"),
    re.compile(r"what does +(?P<body>.+?) +(?:mean|stand for)"),
)
# Provenance/relative-clause grammar, not a new sensitivity classifier. Check
# the original request so a safe quoted span cannot hide a private preamble.
_PRIVATE_CONTEXT = re.compile(
    r"我的|我們|我们|你們|你们|他們|他们|伺服器|服务器|群裡|群里|群內|群内|"
    r"朋友|同學|同学|內部|内部|私下|剛剛|刚刚|剛才|刚才|昨天|說的|说的|講的|讲的|提到"
)
_UNQUOTED_CLAUSE = re.compile(
    r"\b(?:i|you|we|they|he|she|this|that|these|those|said|says|mentioned|told|"
    r"yesterday|and|or|why|how|where|when|current|latest)\b"
)
# Unicode lexical matching cannot distinguish a joined request from a term.
# Reject non-term constructions only on the bare path, after courtesy/domain
# extraction: intent, assistance predicates, and deictic references. Match
# anchored constructions, not individual words such as 解釋, 這, 我, or 想.
_UNQUOTED_NON_TERM_PREFIX = re.compile(
    r"^(?:"
    r"(?:(?:今天|目前|現在|最近) *(?:我 *)?|我 *)想"
    r"|想 *(?:知道|了解)"
    r"|(?:(?:今天|目前|現在|最近) *)?(?:"
    r"(?:(?:可以|能不能|請|麻煩) *)?(?:告訴我|跟我說|幫我)"
    r"|(?:可以|能不能|請|麻煩) *(?:解釋|說明|介紹)"
    r"|(?:解釋|說明|介紹) *一下)"
    r"|(?:這|那)個"
    r")"
)
# Bare reporting structure is ambiguous regardless of the prefix's identity or
# script. Require content on both sides; quotes explicitly select literal data.
_UNQUOTED_ATTRIBUTION = re.compile(r"^.+(?:告訴我|說|講|叫|稱).+$")
_IDENTITY_SYNTAX = re.compile(r"https?://|www\.|@|<[#@]|\b[\w-]+(?:\.[\w-]+)+/")
_ALIASES = sorted(
    {normalize_term_text(unicodedata.normalize("NFKC", alias)) for pack in GAME_PACKS for alias in pack.aliases},
    key=lambda alias: (-len(alias), alias),
)
_DOMAIN_PREFIX = re.compile(
    r"(?P<alias>" + "|".join(map(re.escape, _ALIASES)) + r")"
    r"(?: +(?P<version>\d{1,4}(?:\.\d{1,4}){1,3})| *(?P<current>現版本|目前版本|最新版本|current version))?"
    r" *的 *(?P<body>.+)"
)


def parse_public_term_lookup(request: str) -> PublicTermLookupIntent | None:
    """Extract one safe-to-probe definition term; no I/O or publicness claim.

    Only the current request supplies metadata. Unrecognized contextual wording
    fails closed; readable Discord channels are not evidence for this parser.
    Shared Public Knowledge validation remains an independent safety boundary.
    """
    if not isinstance(request, str) or not request or len(request) > 512:
        return None
    if any(unicodedata.category(char).startswith("C") for char in request):
        return None  # Do not erase controls before checking their presence.
    original = normalize_term_text(unicodedata.normalize("NFKC", request))
    if (_PRIVATE_CONTEXT.search(original) or _IDENTITY_SYNTAX.search(original)
            or not _safe_public_text(original)):
        return None

    text = _COURTESY.sub("", original, count=1)
    text = re.sub(r"[?。]$", "", text).strip()
    match = next((match for pattern in _DEFINITIONS if (match := pattern.fullmatch(text))), None)
    if match is None:
        return None
    body = match["body"].strip()
    domain = subdomain = version = context = ""
    version_sensitive = False
    prefix = _DOMAIN_PREFIX.fullmatch(body)
    if prefix is not None:
        # Restrict registry input to one complete known alias, never a sentence
        # whose longest matching game might conceal other/private context.
        public_domain = resolve_explicit_domain(prefix["alias"])
        if public_domain is None or public_domain.subdomain is None:
            return None
        domain, subdomain = public_domain.domain, public_domain.subdomain
        version = prefix["version"] or ""
        context = "current" if prefix["current"] else ""
        version_sensitive = bool(version or context)
        body = prefix["body"].strip()

    quote_count = sum(char in _QUOTE_CHARACTERS for char in original)
    if quote_count:
        if quote_count != 2:
            return None
        pair = next((pair for pair in _QUOTES if body.startswith(pair[0]) and body.endswith(pair[1])), None)
        if pair is None:
            return None
        term = normalize_term_text(body[1:-1])
    else:
        # A lexical phrase, not an arbitrary sentence before a question suffix.
        # Possessive/relative context requires a recognized public prefix above.
        if ("的" in body or _UNQUOTED_CLAUSE.search(body) or _UNQUOTED_NON_TERM_PREFIX.match(body)
                or _UNQUOTED_ATTRIBUTION.match(body)
                or re.fullmatch(r"(?a:[a-z0-9_.+#-]+(?: [a-z0-9_.+#-]+)*)|[\w.+#-]+", body) is None):
            return None
        term = body
    if not term or len(term) > 64 or len(term.split()) > 8 or not any(char.isalnum() for char in term):
        return None
    key = _normalized_public_key(PublicKnowledgeKey(
        term, domain=domain, subdomain=subdomain, locale="zh-TW", version=version,
        context=context, version_sensitive=version_sensitive,
    ))
    if key is None:
        return None
    return PublicTermLookupIntent(
        key.canonical_term, domain=key.domain, subdomain=key.subdomain,
        version=key.version, context=key.context, version_sensitive=key.version_sensitive,
    )


_ANSWER_LIMIT = 1900
_SOURCE_LIMIT = 3


class _RenderedPublicTermAnswer(str):
    """Renderer-owned complete records; string-compatible for existing callers."""

    __slots__ = ()

    def __new__(cls, answer: str) -> _RenderedPublicTermAnswer:
        if len(answer) > _ANSWER_LIMIT:
            raise ValueError("Public term answer exceeds renderer budget")
        return super().__new__(cls, answer)


def _display_text(text: str, limit: int) -> str:
    text = " ".join(text.split())
    text = "".join(char for char in text if not unicodedata.category(char).startswith("C"))
    # Channel mentions are not covered by discord.utils.escape_mentions.
    text = text.replace("<", "‹").replace(">", "›").replace("@", "@\u200b")
    text = escape_markdown(text, ignore_links=False)
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _unresolved_definition(term: str) -> PublicTermLookupOutcome:
    answer = (
        f"我目前找不到足夠可靠的「{_display_text(term, 128)}」定義。\n"
        "如果你告訴我是在什麼遊戲或領域看到的，我可以再縮小範圍查。"
    )
    return PublicTermLookupOutcome("unresolved", _RenderedPublicTermAnswer(answer))


def _render_definition(term: str, resolution: TermResolution) -> PublicTermLookupOutcome:
    labels = {
        "personal": "依你先前在這個伺服器建立的用法",
        "guild": "依這個伺服器目前確認的共用用法",
        "cache": "公開資料中的定義",
        "web": "公開資料中的定義",
    }
    if (resolution.source not in labels or not isinstance(resolution.meaning, str)
            or not resolution.meaning.strip()):
        return _unresolved_definition(term)
    meaning = _display_text(resolution.meaning, 600)
    if not meaning:
        return _unresolved_definition(term)
    answer = f"{labels[resolution.source]}，「{_display_text(term, 128)}」：{meaning}"
    if resolution.source in ("cache", "web"):
        references = [source for source in resolution.sources[:5]
                      if isinstance(source, PublicSource) and _public_reference_url(source.url)]
        lines = []
        seen = set()
        for source in sorted(references, key=lambda item: item.url):
            if source.url in seen:
                continue
            seen.add(source.url)
            title = _display_text(source.title, 80) or "參考來源"
            # Preserve complete URLs in storage; encode display-breaking/mention
            # characters only in the displayed link, and never truncate a URL.
            url = quote(source.url, safe=":/?#[]%=&+;,-._~")
            line = f"- {title}: <{url}>"
            candidate = answer + "\n參考來源：\n" + "\n".join([*lines, line])
            if len(candidate) > _ANSWER_LIMIT:
                line = f"- {title}（完整來源網址過長，未顯示）"
            if len(answer + "\n參考來源：\n" + "\n".join([*lines, line])) <= _ANSWER_LIMIT:
                lines.append(line)
            if len(lines) == _SOURCE_LIMIT:
                break
        if lines:
            answer += "\n參考來源：\n" + "\n".join(lines)
    return PublicTermLookupOutcome("resolved", _RenderedPublicTermAnswer(answer), resolution.source)


class PublicTermLookupRuntime:
    """Request-scoped orchestration; all ownership and precedence stays in the resolver."""

    def __init__(
        self, local: KnowledgeEnrichmentStore, cache: PublicKnowledgeStore, ai: GeminiAssistant,
        *, web_provider: Callable[[GeminiAssistant, PublicKnowledgeKey, str], Awaitable[PublicWebResult | None]]
        = lookup_public_term_with_search,
    ) -> None:
        self.local = local
        self.cache = cache
        self.ai = ai
        self.web_provider = web_provider

    async def try_resolve(self, request: str, *, guild_id: int, user_id: int) -> PublicTermLookupOutcome:
        intent = parse_public_term_lookup(request)
        if intent is None:
            return PublicTermLookupOutcome("not_applicable")
        try:
            key = PublicKnowledgeKey(
                intent.term, domain=intent.domain, subdomain=intent.subdomain, locale=intent.locale,
                version=intent.version, context=intent.context, version_sensitive=intent.version_sensitive,
            )
            try:
                initial_allow_personal = self.local.chat_style.learning_enabled(guild_id, user_id) is True
            except Exception:
                initial_allow_personal = False

            def allow_personal_now() -> bool:
                if not initial_allow_personal:
                    return False
                try:
                    return self.local.chat_style.learning_enabled(guild_id, user_id) is True
                except Exception:
                    return False

            budget = EnrichmentBudget()

            async def bound_web_lookup(query: str) -> PublicWebResult | None:
                return await self.web_provider(self.ai, key, query)

            resolver = KnowledgeResolver(self.local, self.cache, bound_web_lookup)
            resolution = await resolver.resolve(
                key, guild_id=guild_id, user_id=user_id, budget=budget,
                web_safe=True, materially_relevant=True, allow_personal=allow_personal_now,
            )
            return _render_definition(intent.term, resolution)
        except Exception:
            # No raw request, term, provider output, or private diagnostics in logs.
            # Cancellation is not an ordinary Exception and remains caller-owned.
            return _unresolved_definition(intent.term)
