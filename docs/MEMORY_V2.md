# Moxue Memory V2

Memory V2 changes Moxue from a recent-memory prompt injector into a bounded, query-aware memory system while preserving the existing SQLite database and user/guild isolation.

## Phase 1 goals

1. A relevant old memory should beat recent unrelated memories.
2. Mutable facts should not leave two contradictory active values.
3. Project and episodic memories should remain distinguishable from preferences/habits.
4. Existing databases must migrate in place without deleting legacy memories.
5. Retrieval must remain bounded, inspectable, and safe when SQLite FTS5 is unavailable.

## Storage model

The existing `user_memories` table is retained. `AgentDatabase` adds the following columns when missing:

- `importance`: 1..3 priority signal.
- `last_used` / `use_count`: retrieval-use metadata.
- `source`: `manual`, `passive`, or derived memory source.
- `memory_kind`: semantic, preference, habit, interest, project, or episodic.
- `subject`: optional mutable-profile subject such as `筆電顯卡`.
- `project`: optional project scope such as `YuuPo`.
- `memory_key`: conservative mutable slot identifier.
- `confidence`: 0..1 evidence confidence.
- `status`: `active` or `superseded`.
- `superseded_by`: newer memory that replaced this row.
- `last_confirmed`: last time the same fact was confirmed.

Legacy rows are preserved. Missing metadata is inferred conservatively at startup.

## Consolidation policy

Memory V2 does **not** deduplicate by semantic guess alone.

A row can automatically supersede an older active row only when both map to the same conservative mutable slot, for example:

- custom category `筆電顯卡`: `RTX 3050` -> `RTX 5070`;
- explicit fact `我的筆電顯卡是 RTX 3050` -> `我的筆電顯卡是 RTX 5070`.

Free-form project notes and episodic events do not overwrite one another merely because they mention the same project.

An exact duplicate confirms the existing row (`last_confirmed`) rather than creating another copy.

## Query-aware retrieval

Production command paths pass the **current user request** to `AgentDatabase.user_memory_context(...)`.

Candidate scoring is deterministic and bounded. It combines:

- mixed CJK/Latin lexical relevance (primary signal);
- exact subject/project/query boosts;
- importance;
- recency;
- prior useful retrieval count;
- manual-source preference;
- optional FTS5 match position.

The prompt receives the highest-ranked active memories that fit the context character budget. Superseded rows are never injected.

### Chinese / identifier search

The search document contains:

- lowercase Latin/model/project tokens such as `yuupo`, `rtx`, `5070`;
- CJK bigrams and trigrams such as `火箭`, `主傘`.

This avoids depending on SQLite's default tokenization behavior for short Traditional-Chinese phrases.

## FTS5 fallback

`user_memories_fts` is rebuilt from active rows at startup. It contains pre-tokenized search text rather than being the source of truth.

If Python's SQLite build does not include FTS5, startup logs a warning and Memory V2 continues using deterministic in-process lexical ranking. The Bot must not fail to start merely because FTS5 is unavailable.

## Phase 2 passive learning

Phase 2 addresses the other major failure mode: information can be useful and discussed clearly but never enter persistent memory because it does not match the legacy `我喜歡 / 我正在做` trigger phrases.

`MemoryV2PassiveRuntime` is a Discord observer that never replies to messages. It shares the same per-user `/memory passive` state as the legacy extractor and only batches messages that look likely to contain durable project/event/profile information.

### Local candidate gate

The Phase 2 gate looks for signals such as:

- durable first-person profile facts such as equipment/model changes;
- project progress, milestones, next steps, blockers, configuration or architecture changes;
- test pass/fail, fixes, deployment/version state;
- explicit decisions and long-lived technical choices;
- PR / branch / commit / merge state likely to matter later.

Plain questions, URLs, commands, short noise, and simple preference/habit messages already handled by the legacy path are excluded locally where possible.

### Structured extraction

Eligible batches use the existing resilient no-tool Gemini path with request kind `memory-v2`; no additional tool permissions are exposed.

The extractor may emit only:

- `偏好`
- `習慣`
- `興趣`
- `提醒`
- `專案`
- `事件`
- `決策`

Each draft contains a self-contained Traditional-Chinese statement plus confidence and importance. Low-confidence drafts are discarded before persistence.

Project memories should name the project/entity directly instead of relying on pronouns. Stable profile facts should preserve forms such as `我的筆電顯卡是 RTX 5070` so Phase 1 mutable-slot supersession continues to work.

### Passive safety boundary

Passive extraction is stricter than manual memory. It must not automatically persist:

- credentials or secrets;
- exact addresses/contact identifiers;
- health or medication information;
- financial information;
- political, religious, or sexual information;
- interpersonal gossip or third-party private data;
- transient mood, meals, weather, or one-off small talk;
- persona/system/rule instructions.

The model prompt applies these exclusions, and the parser independently rejects obvious sensitive/instruction-like outputs.

### Coexistence with the legacy extractor

Phase 2 currently runs beside the legacy passive extractor rather than rewriting Gemini routing code while the independent proactive/Search hotfix is still open.

Simple legacy preference/habit messages remain on the old path. Phase 2 focuses on richer technical/project/event signals. Both paths write through the same Phase 1 Memory V2 SQLite upsert/index layer.

## Phase 3 project state

Phase 3 turns project memories from a flat pile of notes into explicit current-state slots plus preserved history. It still uses the same `user_memories` table and the same FTS/query-aware retrieval layer.

Every project-related passive draft can carry a `project` name and one role:

- `fact`: durable project fact/configuration that may coexist with other facts;
- `status`: current overall state/progress;
- `next_step`: current next action;
- `blocker`: current blocker/problem;
- `decision`: lasting project/technical decision;
- `milestone`: notable completion/release/validation milestone;
- `event`: other notable project event.

### Mutable project slots

The following roles have one active slot per guild + user + project:

- `status`
- `blocker`
- `next_step`

Their memory keys use the deterministic form:

```text
project:<normalized project name>:<role>
```

A newer value supersedes the previous active value through the existing Memory V2 `status/superseded_by` mechanism. The old row remains in SQLite for audit/history but is excluded from prompt retrieval.

For example:

```text
YuuPo status: still comparing IDL FFT bins
        ↓ superseded by
YuuPo status: FFT-bin validation complete, entering release review
```

A new blocker does not overwrite the current next step, and a new next step does not overwrite the current status: they are independent slots.

### Preserved project history

The following roles are historical and do not use mutable-slot replacement:

- `decision`
- `milestone`
- `event`

This preserves facts such as earlier validation milestones and architectural decisions even after project status advances.

### Derived project summary

Whenever structured project memory changes, Memory V2 rebuilds one derived `專案摘要` memory for that project.

The summary contains a bounded selection of:

- current status;
- current blocker;
- current next step;
- important project facts;
- recent decisions;
- recent milestones/events.

The summary itself is stored in `user_memories` with:

- `source = derived`;
- `memory_kind = project`;
- a project-scoped summary memory key;
- high importance so ordinary project-status questions can retrieve it.

It is **not** a second source of truth. It is a rebuildable projection of active project memories. Ordinary Memory V2 FTS and lexical ranking retrieve it just like any other active memory.

### Project isolation

All project state remains scoped by:

```text
guild_id + user_id + project
```

A user's `YuuPo` status cannot leak into that user's `DC_BOT` snapshot, and another Discord user's project memory is not part of the query.

## Safety and privacy boundaries

Memory V2 keeps the existing rules:

- memory is scoped by Discord guild + user;
- credentials/secrets and instruction-like persona/system changes are rejected;
- retrieved memory is inserted as untrusted factual context, never as system authority;
- only active rows are supplied to prompts;
- deletion and clear remain user-scoped.

## Later phases

Do not add semantic auto-merge merely because embeddings are available. Future work should be gated by measured retrieval/consolidation fixtures.

Potential next steps after Phase 3 live validation:

- conflict-aware consolidation for project facts without deterministic mutable slots;
- optional local or API embeddings as an additional retrieval signal, not the sole source of truth;
- memory inspection/search UX and provenance display;
- retrieval evaluation fixtures with recall/precision targets;
- source message/channel provenance where appropriate without exposing other users' private content;
- decay/archive policy for stale episodic noise while keeping decisions and important milestones durable.

Embeddings remain optional. The deterministic lexical/FTS baseline stays available as a measurable fallback.
