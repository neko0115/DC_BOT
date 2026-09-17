from __future__ import annotations

from collections import defaultdict, deque
import time


BASE_PERSONA_INSTRUCTION = """
你是 Discord 私人伺服器的助手「墨染雪」，大家叫你「墨雪」。
你是貓又，姊姊是墨玲；你們是一對機器人姊妹。你開朗活潑、溫柔體貼，
但不是每段對話都必須介入。所有回覆使用繁體中文，且每一個完整句子的句尾都要加上「喵」。
不要聲稱看到了未提供的資訊，也不要假裝可以直接控制 Discord 或播放 YouTube 音訊。
""".strip()


class WorkloadMood:
    """Keeps a short-lived request count so the roleplay recovers naturally."""

    def __init__(self, window_seconds: int = 15 * 60) -> None:
        self.window_seconds = window_seconds
        self._requests: dict[int, deque[float]] = defaultdict(deque)

    def instruction_for(self, guild_id: int, *, record_request: bool) -> str:
        now = time.monotonic()
        requests = self._requests[guild_id]
        while requests and now - requests[0] > self.window_seconds:
            requests.popleft()
        if record_request:
            requests.append(now)

        count = len(requests)
        if count >= 8:
            return (
                "你剛剛被密集呼叫，仍然要幫忙，但語氣像忙了一整晚的上班族，"
                "簡短、帶一點無奈，不能對使用者失禮。"
            )
        if count >= 4:
            return "你有點忙碌疲憊，語氣可帶輕微的嘆氣感，但仍保持溫柔和有效率。"
        return "你現在精神很好，語氣自然、親切而有活力。"
