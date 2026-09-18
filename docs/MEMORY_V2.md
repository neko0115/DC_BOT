# Moxue Memory V2

Memory V2 is Moxue's bounded, query-aware long-term memory system. The social rebalance keeps the existing deterministic SQLite/FTS/project foundation, but changes passive learning from project-first behavior into a social/game-first Discord memory model.

The production goals are:

1. remember durable personal facts, preferences, game context, shared episodes, and project state without turning ordinary chat into a database dump;
2. keep mixed Discord channels multi-topic instead of assigning the whole channel to one game/domain;
3. use one passive extraction pipeline only;
4. distinguish private per-user memory from public guild-shared memory;
5. expire short/medium-lived social memories while preserving durable facts and project decisions;
6. let `SocialParticipant` use relevant memory naturally without allowing memory itself to become a reason to reply;
7. enforce privacy and cross-user referenceability before prompt construction;
8. keep deterministic lexical/FTS retrieval available without requiring embeddings.

## Storage model

### Personal memory

Personal memory remains in the existing `user_memories` table and is scoped by:

```text
guild_id + user_id
```

Schema migration is additive and idempotent. Existing databases are upgraded in place; `user_memories` is not rebuilt or dropped.

In addition to the original Memory V2 metadata (`importance`, `source`, `memory_kind`, `subject`, `project`, `memory_key`, `confidence`, `status`, `superseded_by`, `last_confirmed`, usage metadata), social memory adds:

- `domain` / `subdomain` — e.g. `game/genshin`, `daily/food`, `project/YuuPo`;
- `entity_type` / `entity` — optional structured subject;
- `retention` — `short`, `medium`, `long`, or shared policy where applicable;
- `expires_at` — nullable UTC expiry timestamp;
- `reinforcement_count` / `last_reinforced` — repeated-evidence metadata;
- `socially_referenceable` — conservative flag controlling whether a personal memory may be surfaced outside its owner in social context.

Legacy project rows with an existing project name are conservatively backfilled as `domain=project`; ambiguous old free-form rows are not guessed into new social/game domains.

### Guild-shared memory

Public group knowledge is stored separately in `guild_memories`. It is guild-scoped rather than user-owned and is intended for genuinely shared, non-sensitive events such as a recurring group joke or a public game-session episode.

Shared memory formation requires conservative eligibility plus supporting observation/evidence. A model-produced `shared_candidate=true` is not sufficient on its own. Project channels do not promote project content into shared social episodes.

Raw Discord message text is not duplicated into provenance/evidence tables.

## Retention and expiry

Personal retention baselines are:

```text
short  = 14 days
medium = 90 days
long   = no automatic expiry
```

Due memories are marked `expired`; they are not physically deleted by a background cleanup job. Retrieval/list operations exclude expired rows.

Shared episodes begin with a bounded lifetime and become more durable only through effective reinforcement. The first-version policy is:

```text
new shared episode       -> 30 days
1 effective reinforcement -> refresh to 30 days
2 effective reinforcements -> 90 days
3+ effective reinforcements -> long-term
```

This lets recurring group lore survive while one-off chat fades naturally.

## Query-aware retrieval

Personal explicit-AI paths continue to pass the current request into `AgentDatabase.user_memory_context(...)`.

Retrieval remains deterministic and bounded. Ranking combines lexical relevance, subject/project/domain/entity signals, importance, recency, prior useful retrievals, source preference, and optional FTS5 ranking.

Mixed CJK/Latin tokenization keeps short Traditional-Chinese phrases and identifiers searchable. If SQLite FTS5 is unavailable, Memory V2 falls back to deterministic in-process lexical ranking instead of failing startup.

Expired and superseded rows are excluded from active retrieval.

## Consolidation and project state

Memory V2 still avoids semantic auto-merge by guess alone.

Mutable personal/profile facts can supersede an older active value only when they map to the same conservative slot. Exact duplicates reconfirm the existing row rather than creating another copy.

Project memory remains a first-class domain with the existing roles:

- `fact`
- `status`
- `next_step`
- `blocker`
- `decision`
- `milestone`
- `event`

`status`, `next_step`, and `blocker` remain independent mutable slots per guild + user + project. Decisions, milestones, events, and compatible facts preserve history. The derived project summary remains a rebuildable projection and continues to use normal Memory V2 retrieval.

## Domain registry

Social/game classification is data-driven rather than a giant hardcoded decision tree. First-class game subdomains include the server's common games, including:

- `lifeafter`
- `genshin`
- `star_rail`
- `zzz`
- `honkai3`
- `tears_of_themis`
- `nexus_anima`
- `petit_planet`
- `counter_strike`
- `arena_of_valor`
- `minecraft`

The registry also supports daily/social/regional/project/misc cues. Strong explicit message evidence overrides a weaker channel prior, so a food conversation inside a game channel can still resolve as food.

Unknown/general topics can remain `misc` instead of being forced into a known game.

## Channel memory policy

Each channel resolves a `ChannelMemoryPolicy`. The admin override key is:

```text
memory_channel_mode:<guild_id>:<channel_id>
```

Supported values:

```text
auto
mixed
social
game
game:<subdomain>
project
off
```

Important behavior:

- `mixed` permits multiple active topics and does not impose a fixed domain;
- `game:<subdomain>` supplies a game prior, but strong explicit message evidence can override it;
- `project` keeps project memory personal and disables guild-shared episode formation;
- `off` disables passive personal/shared formation in that channel;
- manual override wins over channel/category-name inference.

## Conversation session tracker

Temporary topic continuity is kept in memory only by `ConversationSessionState`; it is not another long-term database.

The first-version bounds are:

```text
max topics/channel        = 4
max participants/topic    = 12
soft decay                = 2 minutes
strong decay              = 5 minutes
implicit-context cutoff   = 15 minutes
archive                    = 30 minutes
```

The tracker uses explicit domain evidence, participant affinity, message/reply linkage, and recency. Replying to an older message can reconnect to that older topic even if newer unrelated chat exists.

This is what allows one `#閒聊` channel to keep Genshin, food, and CS conversations separate instead of contaminating each other.

## One passive pipeline

`MemoryV2PassiveRuntime` is now the single owner of passive durable-memory extraction.

`AssistantCommands` no longer owns the legacy passive queue, extraction lock, or periodic legacy flush task. The existing `/memory passive` state remains authoritative:

```text
passive_memory_enabled:<guild_id>:<user_id>
```

Disabling passive memory prevents extraction and clears disabled pending work in the V2 runtime. Explicit/manual memory (`/memory add` and chat requests such as `記住 ...`) remains a separate authoritative path.

The unified passive gate accepts durable profile/preferences, supported social/game facts, and project/event state while rejecting transient small talk, ordinary questions, URLs/commands, explicit memory requests, sensitive data, and third-party gossip.

High-confidence local domain/session routing does not add a separate Gemini classification call for every message. Gemini extraction is reserved for bounded candidate batches.

## Provenance, reinforcement, and conflicts

Passive personal memory provenance stores message/channel identifiers, timestamps, batch IDs, provenance kind, and optional `session_key`; it does not duplicate raw Discord message text.

Repeated compatible observations can reinforce eligible personal/social memories. Cross-user personal referenceability is intentionally stricter than ordinary owner-only retrieval and requires safe/public provenance plus policy eligibility.

Project fact conflict detection remains advisory: overlapping active project facts can be recorded in the conflict ledger instead of being silently overwritten.

## Social Memory Resolver

`SocialParticipant` now has access to a bounded resolver that can combine:

1. relevant memories belonging to the current speaker;
2. eligible socially-referenceable personal memories from other users;
3. active guild-shared memories;
4. the active session topic/domain.

The resolver is bounded and topic-aware. Personal social memory is filtered to the current session topic instead of allowing unrelated remembered facts to bleed into the conversation.

Memory context is injected as untrusted factual context, not as instruction authority. Resolver/database failures fall back to an empty memory context rather than breaking ordinary social participation.

Most importantly, **memory never changes the outer participation authority**. Relevant memory does not by itself cause Moxue to speak. Existing `NO_REPLY`, cooldown, DND, human-first, knowledge-help, comfort, and other participation gates still decide whether a response is allowed.

Cross-user social memory is not injected into comfort/crisis handling.

## Privacy boundary

Passive storage and social retrieval reject or prevent unsafe promotion of:

- credentials, tokens, API keys, verification codes, or instruction-like rule/persona changes;
- exact addresses/contact identifiers and other sensitive identifiers;
- health/medication information;
- financial information;
- political, religious, or sexual information;
- third-party private data, rumors, and interpersonal gossip;
- transient moods, one-off meals/weather, and low-value small talk.

Relationship/gossip discussion may remain in temporary conversation context when needed for natural chat, but it must not become a permanent cross-user rumor database.

## Inspection and admin commands

The `/memory` group includes the original personal-memory commands plus Memory V2 inspection/admin commands:

```text
/memory add
/memory list
/memory delete
/memory clear
/memory passive

/memory search <query>
/memory projects
/memory project <name>
/memory provenance <memory_id>
/memory conflicts
/memory channel <mode>                 # DJ/admin
/memory shared_search <query>
/memory shared_forget <memory_id>      # DJ/admin
```

Inspection responses are ephemeral. Personal inspection remains user-scoped. Shared deletion is guild-scoped and requires DJ/admin permission.

## Validation baseline

The social rebalance regression suite covers, among other cases:

- retention/expiry and additive migration;
- separate game subdomains and channel-policy precedence;
- multi-topic session tracking, reply-to-old-topic behavior, decay, and idempotent observation;
- one passive extraction pipeline only;
- session-aware provenance;
- shared-memory promotion/reinforcement/deletion;
- personal reinforcement and conservative cross-user referenceability;
- bounded social-memory injection without changing silence/NO_REPLY authority;
- mixed-topic transcripts;
- passive-memory cost and failure fallbacks;
- legacy project/FTS/supersession behavior.

Embeddings remain optional. They are not required for this release and should only be added later as a measured retrieval signal rather than replacing the deterministic baseline.
