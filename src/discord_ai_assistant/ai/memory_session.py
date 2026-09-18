from __future__ import annotations

import time
from dataclasses import dataclass, field
from uuid import uuid4

from discord_ai_assistant.ai.memory_domain_registry import DomainMatch

MAX_ACTIVE_TOPICS_PER_CHANNEL = 4
MAX_PARTICIPANTS_PER_TOPIC = 12
MAX_RESOLVED_MESSAGES_PER_CHANNEL = 256
STRONG_TOPIC_CONFIDENCE = 0.75
AMBIGUOUS_THRESHOLD = 0.50
SOFT_DECAY_SECONDS = 2 * 60
STRONG_DECAY_SECONDS = 5 * 60
IMPLICIT_CONTEXT_CUTOFF_SECONDS = 15 * 60
ARCHIVE_SECONDS = 30 * 60


@dataclass(frozen=True, slots=True)
class SessionMessage:
    channel_id: int
    author_id: int
    message_id: int
    content: str
    reply_to_message_id: int | None = None


@dataclass(frozen=True, slots=True)
class SessionResolution:
    session_key: str
    domain: str
    subdomain: str | None
    topic: str | None
    confidence: float
    source: str
    participant_ids: tuple[int, ...]


@dataclass(slots=True)
class _TopicState:
    session_key: str
    domain: str
    subdomain: str | None
    topic: str | None
    confidence: float
    created_at: float
    last_seen: float
    participant_last_seen: dict[int, float] = field(default_factory=dict)
    message_ids: list[int] = field(default_factory=list)


class ConversationSessionState:
    """In-memory, local-first topic routing for Discord conversation fragments."""

    def __init__(self) -> None:
        self._session_namespace = uuid4().hex[:12]
        self._topics: dict[int, list[_TopicState]] = {}
        self._message_to_topic: dict[tuple[int, int], _TopicState] = {}
        self._resolved_messages: dict[tuple[int, int], SessionResolution | None] = {}
        self._recent_message_keys: dict[int, list[tuple[int, int]]] = {}
        self._channel_sequence: dict[int, int] = {}

    @staticmethod
    def _now(value: float | None) -> float:
        return time.monotonic() if value is None else float(value)

    @staticmethod
    def _decayed_confidence(topic: _TopicState, now: float) -> float:
        age = max(0.0, now - topic.last_seen)
        base = max(0.0, min(topic.confidence, 1.0))
        if age <= SOFT_DECAY_SECONDS:
            factor = 1.0 - (0.04 * (age / SOFT_DECAY_SECONDS))
        elif age <= STRONG_DECAY_SECONDS:
            span = STRONG_DECAY_SECONDS - SOFT_DECAY_SECONDS
            factor = 0.96 - (0.16 * ((age - SOFT_DECAY_SECONDS) / span))
        elif age < IMPLICIT_CONTEXT_CUTOFF_SECONDS:
            span = IMPLICIT_CONTEXT_CUTOFF_SECONDS - STRONG_DECAY_SECONDS
            factor = 0.80 - (0.35 * ((age - STRONG_DECAY_SECONDS) / span))
        elif age < ARCHIVE_SECONDS:
            span = ARCHIVE_SECONDS - IMPLICIT_CONTEXT_CUTOFF_SECONDS
            factor = 0.30 - (0.20 * ((age - IMPLICIT_CONTEXT_CUTOFF_SECONDS) / span))
        else:
            factor = 0.0
        return round(max(0.0, min(1.0, base * factor)), 6)

    def _next_session_key(self, channel_id: int) -> str:
        sequence = self._channel_sequence.get(channel_id, 0) + 1
        self._channel_sequence[channel_id] = sequence
        return f"{self._session_namespace}:{channel_id}:{sequence}"

    def _cache_resolution(
        self,
        key: tuple[int, int],
        resolution: SessionResolution | None,
    ) -> SessionResolution | None:
        """Keep listener idempotency/reply lookup bounded per Discord channel."""

        channel_id = int(key[0])
        recent = self._recent_message_keys.setdefault(channel_id, [])
        if key not in self._resolved_messages:
            recent.append(key)
        self._resolved_messages[key] = resolution

        overflow = len(recent) - MAX_RESOLVED_MESSAGES_PER_CHANNEL
        if overflow > 0:
            expired = recent[:overflow]
            del recent[:overflow]
            for old_key in expired:
                self._resolved_messages.pop(old_key, None)
                self._message_to_topic.pop(old_key, None)
        return resolution

    def _prune(self, channel_id: int, now: float) -> list[_TopicState]:
        current = self._topics.get(channel_id, [])
        active = [topic for topic in current if now - topic.last_seen < ARCHIVE_SECONDS]
        if active:
            self._topics[channel_id] = active
        else:
            self._topics.pop(channel_id, None)
        return active

    @staticmethod
    def _same_topic(topic: _TopicState, match: DomainMatch) -> bool:
        if topic.domain != match.domain or topic.subdomain != match.subdomain:
            return False
        if topic.topic == match.topic:
            return True
        # A domain-explicit message without a topic may continue the only active topic
        # for that subdomain.  A strong explicit *different* topic creates a new thread.
        return match.topic is None or topic.topic is None

    def _matching_explicit_topic(
        self,
        topics: list[_TopicState],
        match: DomainMatch,
    ) -> _TopicState | None:
        matches = [topic for topic in topics if self._same_topic(topic, match)]
        if not matches:
            return None
        return max(matches, key=lambda topic: topic.last_seen)

    @staticmethod
    def _participant_ids(topic: _TopicState) -> tuple[int, ...]:
        return tuple(
            author_id
            for author_id, _ in sorted(
                topic.participant_last_seen.items(),
                key=lambda item: (-item[1], item[0]),
            )
        )

    def _resolution(
        self,
        topic: _TopicState,
        *,
        confidence: float,
        source: str,
    ) -> SessionResolution:
        return SessionResolution(
            session_key=topic.session_key,
            domain=topic.domain,
            subdomain=topic.subdomain,
            topic=topic.topic,
            confidence=round(max(0.0, min(confidence, 1.0)), 6),
            source=source,
            participant_ids=self._participant_ids(topic),
        )

    def _add_participant(self, topic: _TopicState, author_id: int, now: float) -> None:
        topic.participant_last_seen[int(author_id)] = now
        if len(topic.participant_last_seen) <= MAX_PARTICIPANTS_PER_TOPIC:
            return
        oldest = min(topic.participant_last_seen.items(), key=lambda item: (item[1], item[0]))[0]
        topic.participant_last_seen.pop(oldest, None)

    def _bind_message(
        self,
        topic: _TopicState,
        message: SessionMessage,
        *,
        now: float,
        source: str,
        observed_confidence: float,
        explicit: DomainMatch | None,
    ) -> SessionResolution:
        self._add_participant(topic, message.author_id, now)
        if message.message_id not in topic.message_ids:
            topic.message_ids.append(message.message_id)
            del topic.message_ids[:-64]
        self._message_to_topic[(message.channel_id, message.message_id)] = topic

        # Explicit evidence restores the strongest locally-observed confidence. An
        # implicit continuation refreshes last_seen but preserves its already-decayed
        # confidence so vague chatter cannot magically become authoritative again.
        if explicit is not None:
            topic.confidence = max(topic.confidence, explicit.confidence)
            if explicit.topic is not None:
                topic.topic = explicit.topic
        elif source == "reply":
            topic.confidence = max(observed_confidence, 0.70)
        else:
            topic.confidence = max(observed_confidence, AMBIGUOUS_THRESHOLD)
        topic.last_seen = now
        return self._resolution(topic, confidence=observed_confidence, source=source)

    def _create_topic(
        self,
        message: SessionMessage,
        match: DomainMatch,
        *,
        now: float,
        source: str,
    ) -> SessionResolution:
        topics = self._prune(message.channel_id, now)
        if len(topics) >= MAX_ACTIVE_TOPICS_PER_CHANNEL:
            evicted = min(
                topics,
                key=lambda topic: (self._decayed_confidence(topic, now), topic.last_seen),
            )
            topics.remove(evicted)
        topic = _TopicState(
            session_key=self._next_session_key(message.channel_id),
            domain=match.domain,
            subdomain=match.subdomain,
            topic=match.topic,
            confidence=max(0.0, min(match.confidence, 1.0)),
            created_at=now,
            last_seen=now,
        )
        topics.append(topic)
        self._topics[message.channel_id] = topics
        return self._bind_message(
            topic,
            message,
            now=now,
            source=source,
            observed_confidence=match.confidence,
            explicit=match if source == "explicit" else None,
        )

    def _implicit_candidate(
        self,
        topics: list[_TopicState],
        *,
        author_id: int,
        now: float,
    ) -> tuple[_TopicState, float, str] | None:
        usable = [
            (topic, self._decayed_confidence(topic, now))
            for topic in topics
            if now - topic.last_seen < IMPLICIT_CONTEXT_CUTOFF_SECONDS
        ]
        usable = [(topic, confidence) for topic, confidence in usable if confidence >= AMBIGUOUS_THRESHOLD]
        if not usable:
            return None

        participated = [
            (topic, confidence)
            for topic, confidence in usable
            if int(author_id) in topic.participant_last_seen
        ]
        if participated:
            topic, confidence = max(
                participated,
                key=lambda item: (
                    item[0].participant_last_seen[int(author_id)],
                    item[1],
                    item[0].last_seen,
                ),
            )
            return topic, confidence, "participant"

        if len(usable) == 1:
            topic, confidence = usable[0]
            return topic, confidence, "active_topic"

        ranked = sorted(usable, key=lambda item: (item[1], item[0].last_seen), reverse=True)
        if ranked[0][1] - ranked[1][1] >= 0.15:
            topic, confidence = ranked[0]
            return topic, confidence, "active_topic"
        return None

    def observe(
        self,
        message: SessionMessage,
        *,
        explicit: DomainMatch | None,
        channel_prior: DomainMatch | None,
        now: float | None = None,
    ) -> SessionResolution | None:
        """Resolve and record one message without calling Gemini.

        The operation is idempotent by `(channel_id, message_id)` within the bounded
        recent-message window so multiple Discord listeners can safely share one session
        tracker regardless of listener ordering without retaining every message forever.
        """

        key = (int(message.channel_id), int(message.message_id))
        if key in self._resolved_messages:
            return self._resolved_messages[key]

        current = self._now(now)
        topics = self._prune(message.channel_id, current)

        if explicit is not None and explicit.confidence >= STRONG_TOPIC_CONFIDENCE:
            topic = self._matching_explicit_topic(topics, explicit)
            if topic is None:
                resolution = self._create_topic(message, explicit, now=current, source="explicit")
            else:
                resolution = self._bind_message(
                    topic,
                    message,
                    now=current,
                    source="explicit",
                    observed_confidence=explicit.confidence,
                    explicit=explicit,
                )
            return self._cache_resolution(key, resolution)

        if message.reply_to_message_id is not None:
            reply_topic = self._message_to_topic.get((message.channel_id, int(message.reply_to_message_id)))
            if reply_topic is not None and current - reply_topic.last_seen < ARCHIVE_SECONDS:
                confidence = max(self._decayed_confidence(reply_topic, current), 0.70)
                resolution = self._bind_message(
                    reply_topic,
                    message,
                    now=current,
                    source="reply",
                    observed_confidence=confidence,
                    explicit=None,
                )
                return self._cache_resolution(key, resolution)

        implicit = self._implicit_candidate(topics, author_id=message.author_id, now=current)
        if implicit is not None:
            topic, confidence, source = implicit
            resolution = self._bind_message(
                topic,
                message,
                now=current,
                source=source,
                observed_confidence=confidence,
                explicit=None,
            )
            return self._cache_resolution(key, resolution)

        if channel_prior is not None:
            topic = self._matching_explicit_topic(topics, channel_prior)
            if topic is None:
                resolution = self._create_topic(
                    message,
                    channel_prior,
                    now=current,
                    source="channel_prior",
                )
            else:
                confidence = self._decayed_confidence(topic, current)
                resolution = self._bind_message(
                    topic,
                    message,
                    now=current,
                    source="channel_prior",
                    observed_confidence=max(confidence, channel_prior.confidence),
                    explicit=None,
                )
            return self._cache_resolution(key, resolution)

        return self._cache_resolution(key, None)

    def context_for_message(self, channel_id: int, message_id: int) -> SessionResolution | None:
        return self._resolved_messages.get((int(channel_id), int(message_id)))

    def active_topics(
        self,
        channel_id: int,
        *,
        now: float | None = None,
    ) -> tuple[SessionResolution, ...]:
        current = self._now(now)
        topics = self._prune(int(channel_id), current)
        ranked = sorted(
            topics,
            key=lambda topic: (self._decayed_confidence(topic, current), topic.last_seen),
            reverse=True,
        )
        return tuple(
            self._resolution(
                topic,
                confidence=self._decayed_confidence(topic, current),
                source="active_topic",
            )
            for topic in ranked
        )
