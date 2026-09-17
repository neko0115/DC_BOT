from __future__ import annotations

from collections import deque
import random

from discord_ai_assistant.models import QueuedTrack


class GuildQueue:
    def __init__(self) -> None:
        self.current: QueuedTrack | None = None
        self._upcoming: deque[QueuedTrack] = deque()
        self._history: deque[QueuedTrack] = deque(maxlen=100)

    def append(self, item: QueuedTrack) -> None:
        self._upcoming.append(item)

    def append_next(self, item: QueuedTrack) -> None:
        self._upcoming.appendleft(item)

    def advance(self, shuffle: bool = False) -> QueuedTrack | None:
        if self.current:
            self._history.append(self.current)
        if not self._upcoming:
            self.current = None
            return None
        if shuffle:
            index = random.randrange(len(self._upcoming))
            self.current = self._upcoming[index]
            del self._upcoming[index]
        else:
            self.current = self._upcoming.popleft()
        return self.current

    def queue_previous(self) -> bool:
        if not self._history:
            return False
        previous = self._history.pop()
        if self.current:
            self._upcoming.appendleft(self.current)
        self._upcoming.appendleft(previous)
        return True

    def clear(self) -> None:
        self.current = None
        self._upcoming.clear()
        self._history.clear()

    def snapshot(self) -> list[QueuedTrack]:
        return list(self._upcoming)
