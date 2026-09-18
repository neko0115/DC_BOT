from __future__ import annotations

import re
from dataclasses import dataclass

KNOWLEDGE_HELP_WAIT_SECONDS = 8.0
KNOWLEDGE_HELP_COOLDOWN_SECONDS = 10 * 60.0
KNOWLEDGE_HELP_ACK_SECONDS = 2 * 60.0
KNOWLEDGE_HELP_QUIET_SECONDS = 30 * 60.0

QUESTION_COMPLEXITY_SIMPLE = "simple"
QUESTION_COMPLEXITY_NORMAL = "normal"
QUESTION_COMPLEXITY_COMPLEX = "complex"
QUESTION_COMPLEXITY_SEARCH = "search"

# Proactive help may now recognize ordinary factual/how-to questions as well as explicit
# knowledge gaps. Ambiguous, personal, contentious, sensitive, or high-stakes topics
# still stay silent unless the user explicitly addresses the bot through the normal path.
EXCLUDED_TOPIC_KEYWORDS = (
    "感情",
    "分手",
    "交往",
    "曖昧",
    "喜歡我",
    "愛我",
    "討厭我",
    "怎麼想",
    "怎麼看我",
    "覺得我",
    "吵架",
    "誰對誰錯",
    "八卦",
    "政治",
    "政黨",
    "選舉",
    "自殺",
    "想死",
    "不想活",
    "活不下去",
    "症狀",
    "診斷",
    "藥物",
    "劑量",
    "醫生",
    "法律",
    "律師",
    "訴訟",
    "告他",
    "投資",
    "股票",
    "期貨",
    "槓桿",
    "加密貨幣",
    "密碼",
    "password",
    "token",
    "api key",
    "apikey",
    "驗證碼",
)

FRESHNESS_DEPENDENT_KEYWORDS = (
    "最新",
    "新聞",
    "天氣",
    "班次",
    "時刻表",
    "票價",
    "股價",
    "現在幾點",
    "今天幾點",
    "明天幾點",
    "幾點到",
    "幾點開",
    "幾點關",
    "今天開嗎",
    "現在有開",
    "weather",
    "latest",
    "schedule",
    "timetable",
    "opening hours",
)

FRESHNESS_DEPENDENT_PATTERNS = (
    # Require an event predicate, not an update/release noun in a static definition.
    re.compile(r"(?:今天|目前|現在|最近)[^。！？!?；;]{0,12}(?:更新|發布|發佈|釋出)了"),
    re.compile(r"(?:班車|火車|公車|客運|高鐵|台鐵|捷運|航班|飛機).{0,16}(?:幾點|時間|班次|時刻)"),
    re.compile(r"(?:店|餐廳|商店|場館|景點|銀行|郵局).{0,16}(?:幾點|營業|開門|關門|有開)"),
    re.compile(r"(?:下一個|這個|目前|現在|今天|明天|最近).{0,12}(?:颱風|台風|typhoon)"),
    re.compile(r"(?:颱風|台風|typhoon).{0,28}(?:什麼時候|何時|靠近|最近|路徑|預報|登陸|影響)"),
)

CANCEL_SIGNAL_KEYWORDS = (
    "算了",
    "不用了",
    "沒事了",
    "找到了",
    "找到方法了",
    "會了",
    "解決了",
    "好了",
)

# High-confidence knowledge-gap wording. Question punctuation is intentionally optional.
ZH_GAP_PATTERNS = (
    re.compile(
        r"(?:忘了|忘記|不記得|記不清楚|想不起來|不知道|不曉得|不會|搞不懂).{0,28}"
        r"(?:怎麼|如何|哪裡|哪個|哪一個|什麼|多少|幾點|時間|指令|型號|版本|名稱|叫什麼|原因|差別|設定|用法|班次|時刻表|票價|天氣|營業時間)"
    ),
    re.compile(
        r"(?:怎麼|如何).{0,14}"
        r"(?:查|看|確認|檢查|設定|設置|使用|用|開|關|找|裝|安裝|連接|連|啟動|關閉|輸入|切換|更改|修改|改|弄|做|測試|測|處理|解決)"
    ),
)

# Common spoken Chinese questions. These deliberately do not require ? / ？ because
# Discord users often omit punctuation entirely.
ZH_QUESTION_PATTERNS = (
    re.compile(r"(?:什麼|為什麼|怎麼|如何|哪個|哪一個|哪裡|哪家|多少|幾個|幾點).{0,40}"),
    re.compile(r"(?:是不是|能不能|可不可以|有沒有|會不會).{1,48}"),
    re.compile(r"(?:差在哪|差別是什麼|哪個(?:比較)?好|怎麼選|正常嗎|正常不正常|對嗎|會怎樣|知道嗎|可以嗎|需要嗎)"),
    re.compile(r"(?:有人知道|請問).{2,64}"),
    re.compile(r"(?:天氣|氣溫|溫度).{0,16}(?:怎樣|如何|多少|幾度|會不會|下雨|冷|熱)"),
)

# "誰" is especially ambiguous in group chat: it is often teasing or relationship
# talk rather than a public-knowledge question. Keep only clear factual who-questions.
ZH_PUBLIC_WHO_PATTERNS = (
    re.compile(r"(?:誰是|是誰)(?:[^誰]{2,40})"),
    re.compile(r"誰.{0,18}(?:發明|設計|建立|創立|撰寫|主演|導演|演唱|負責)"),
)

SOCIAL_CHAT_PATTERNS = (
    re.compile(r"^誰.{0,12}(?:誰|調情|調|撩|喜歡|討厭|跟|陪|約|告白|追)"),
    re.compile(
        r"(?:你|妳|他|她|我們|你們|妳們|他們|她們).{0,18}"
        r"(?:在幹嘛|在做什麼|在聊什麼|吃什麼|去哪|跟誰|喜歡誰|是不是喜歡|有沒有喜歡|調情|告白)"
    ),
    re.compile(r"^(?:謝謝|感謝|辛苦了|晚安|早安|笑死|哈哈|好喔|懂了|原來如此).{0,12}[?？]?$"),
)

# Contextual follow-ups frequently omit the subject because it was established in the
# preceding messages. A meaningful context is required for this lower-confidence class.
ZH_CONTEXTUAL_PATTERNS = (
    re.compile(
        r"(?:那|那個|它|這|這個|這樣|那樣|所以).{0,20}"
        r"(?:多少|怎麼|為什麼|是不是|能不能|可不可以|有沒有|會不會|正常嗎|對嗎|差在哪|哪個|會怎樣)"
    ),
    re.compile(r"(?:那|所以)?(?:原因呢|差別呢|結果呢|怎麼辦|怎麼處理|哪個好)"),
)

EN_GAP_PATTERNS = (
    re.compile(r"\b(?:i\s+)?(?:can(?:not|'t)|don't|do not)\s+(?:remember|know)\s+how\b", re.IGNORECASE),
    re.compile(r"\bhow\s+(?:do\s+i|can\s+i|to)\b", re.IGNORECASE),
    re.compile(r"\b(?:what|why|where|which|who|when)\b.{1,80}", re.IGNORECASE),
    re.compile(r"\b(?:can|could|should|is|are|does|do)\s+(?:i|we|this|that|it)\b.{1,80}", re.IGNORECASE),
)

RHETORICAL_LOW_INFORMATION = {
    "蛤",
    "蛤?",
    "蛤？",
    "啥",
    "啥?",
    "啥？",
    "你認真",
    "你認真?",
    "你認真？",
    "認真?",
    "認真？",
    "不是吧",
    "不是吧?",
    "不是吧？",
    "真的假的",
    "真的假的?",
    "真的假的？",
    "哈?",
    "哈？",
    "what?",
    "really?",
}

COMPLEX_QUESTION_MARKERS = (
    "比較",
    "分析",
    "規劃",
    "架構",
    "根因",
    "原因分析",
    "優缺點",
    "取捨",
    "評估",
    "除錯",
    "debug",
    "root cause",
    "trade-off",
    "tradeoff",
)

NORMAL_QUESTION_MARKERS = (
    "為什麼",
    "怎麼會",
    "原因",
    "差在哪",
    "差別",
    "正常嗎",
    "是不是因為",
    "會不會",
    "怎麼處理",
    "怎麼解決",
)

CONTEXT_DEPENDENT_MARKERS = (
    "它",
    "那個",
    "這個",
    "這樣",
    "那樣",
    "剛剛",
    "前面",
    "上面",
    "所以",
)


def _normalized(content: str) -> str:
    return " ".join(content.strip().lower().split())


def _rhetorical_key(content: str) -> str:
    return _normalized(content).strip(" .。!！~～")


def _meaningful_context(context: str | None) -> bool:
    if not context:
        return False
    normalized = _normalized(context)
    if not normalized or "沒有可用的近期對話" in normalized:
        return False
    compact = re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", normalized)
    return len(compact) >= 6


def is_knowledge_gap_candidate(content: str, *, context: str | None = None) -> bool:
    """Detect factual/how-to questions locally without requiring question punctuation.

    The historic function name is retained for compatibility even though the detector now
    covers ordinary questions in addition to explicit "I forgot / I don't know" gaps.
    """

    normalized = _normalized(content)
    if len(normalized) < 3 or len(normalized) > 320:
        return False
    if _rhetorical_key(content) in RHETORICAL_LOW_INFORMATION:
        return False
    if any(keyword in normalized for keyword in EXCLUDED_TOPIC_KEYWORDS):
        return False
    if any(pattern.search(normalized) for pattern in SOCIAL_CHAT_PATTERNS):
        return False
    if any(
        pattern.search(normalized)
        for pattern in (*ZH_GAP_PATTERNS, *ZH_QUESTION_PATTERNS, *ZH_PUBLIC_WHO_PATTERNS, *EN_GAP_PATTERNS)
    ):
        return True
    if _meaningful_context(context) and any(pattern.search(normalized) for pattern in ZH_CONTEXTUAL_PATTERNS):
        return True
    return False


def is_freshness_dependent(content: str) -> bool:
    normalized = _normalized(content)
    return bool(
        any(keyword in normalized for keyword in FRESHNESS_DEPENDENT_KEYWORDS)
        or any(pattern.search(normalized) for pattern in FRESHNESS_DEPENDENT_PATTERNS)
    )


def classify_question_complexity(content: str, *, context: str | None = None) -> str:
    """Choose a cheap routing class for proactive answering.

    This is intentionally deterministic: it does not spend a Gemini call merely to decide
    which Gemini model should answer. Search always wins, then explicit complex markers,
    then contextual/explanatory questions, with short factual/how-to questions using Lite.
    """

    normalized = _normalized(content)
    if is_freshness_dependent(normalized):
        return QUESTION_COMPLEXITY_SEARCH
    if any(marker in normalized for marker in COMPLEX_QUESTION_MARKERS):
        return QUESTION_COMPLEXITY_COMPLEX
    clause_count = sum(normalized.count(marker) for marker in ("而且", "並且", "同時", "以及", "另外", "還有"))
    if len(normalized) >= 180 or clause_count >= 2:
        return QUESTION_COMPLEXITY_COMPLEX
    contextual = _meaningful_context(context) and any(marker in normalized for marker in CONTEXT_DEPENDENT_MARKERS)
    if contextual or any(marker in normalized for marker in NORMAL_QUESTION_MARKERS) or len(normalized) >= 80:
        return QUESTION_COMPLEXITY_NORMAL
    return QUESTION_COMPLEXITY_SIMPLE


def is_cancel_signal(content: str) -> bool:
    normalized = _normalized(content)
    return any(keyword in normalized for keyword in CANCEL_SIGNAL_KEYWORDS)


@dataclass(frozen=True, slots=True)
class PendingKnowledgeHelp:
    author_id: int
    source_message_id: int


@dataclass(frozen=True, slots=True)
class AwaitingKnowledgeAck:
    response_message_id: int
    expires_at: float


class KnowledgeHelpState:
    """Small in-memory anti-annoyance state machine for delayed proactive help."""

    def __init__(
        self,
        *,
        cooldown_seconds: float = KNOWLEDGE_HELP_COOLDOWN_SECONDS,
        acknowledgement_seconds: float = KNOWLEDGE_HELP_ACK_SECONDS,
        quiet_seconds: float = KNOWLEDGE_HELP_QUIET_SECONDS,
    ) -> None:
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self.acknowledgement_seconds = max(0.0, float(acknowledgement_seconds))
        self.quiet_seconds = max(0.0, float(quiet_seconds))
        self._pending: dict[tuple[int, int], PendingKnowledgeHelp] = {}
        self._last_help_at: dict[tuple[int, int], float] = {}
        self._awaiting_ack: dict[tuple[int, int], AwaitingKnowledgeAck] = {}
        self._unacknowledged: dict[tuple[int, int], int] = {}
        self._quiet_until: dict[tuple[int, int], float] = {}

    def can_schedule(self, key: tuple[int, int], now: float) -> bool:
        self._expire_ack(key, now)
        if key in self._pending:
            return False
        if now < self._quiet_until.get(key, 0.0):
            return False
        return now - self._last_help_at.get(key, -self.cooldown_seconds) >= self.cooldown_seconds

    def start_pending(self, key: tuple[int, int], *, author_id: int, source_message_id: int) -> None:
        self._pending[key] = PendingKnowledgeHelp(author_id=author_id, source_message_id=source_message_id)

    def pending_matches(self, key: tuple[int, int], source_message_id: int) -> bool:
        pending = self._pending.get(key)
        return bool(pending and pending.source_message_id == source_message_id)

    def clear_pending(self, key: tuple[int, int], source_message_id: int | None = None) -> bool:
        pending = self._pending.get(key)
        if pending is None:
            return False
        if source_message_id is not None and pending.source_message_id != source_message_id:
            return False
        self._pending.pop(key, None)
        return True

    def observe_human(
        self,
        key: tuple[int, int],
        *,
        author_id: int,
        content: str,
        now: float,
        reply_to_message_id: int | None = None,
        mentions_bot: bool = False,
    ) -> bool:
        """Observe a human message and return True when pending help should be cancelled."""

        self._expire_ack(key, now)
        awaiting = self._awaiting_ack.get(key)
        if awaiting and (mentions_bot or reply_to_message_id == awaiting.response_message_id):
            self._awaiting_ack.pop(key, None)
            self._unacknowledged[key] = 0
            self._quiet_until.pop(key, None)

        pending = self._pending.get(key)
        if pending is None:
            return False
        should_cancel = author_id != pending.author_id or is_cancel_signal(content)
        if should_cancel:
            self._pending.pop(key, None)
        return should_cancel

    def record_sent(self, key: tuple[int, int], *, response_message_id: int, now: float) -> None:
        self._pending.pop(key, None)
        self._last_help_at[key] = now
        self._awaiting_ack[key] = AwaitingKnowledgeAck(
            response_message_id=response_message_id,
            expires_at=now + self.acknowledgement_seconds,
        )

    def unacknowledged_count(self, key: tuple[int, int], now: float) -> int:
        self._expire_ack(key, now)
        return self._unacknowledged.get(key, 0)

    def quiet_until(self, key: tuple[int, int], now: float) -> float:
        self._expire_ack(key, now)
        return self._quiet_until.get(key, 0.0)

    def _expire_ack(self, key: tuple[int, int], now: float) -> None:
        awaiting = self._awaiting_ack.get(key)
        if awaiting is None or now < awaiting.expires_at:
            return
        self._awaiting_ack.pop(key, None)
        count = self._unacknowledged.get(key, 0) + 1
        if count >= 2:
            self._unacknowledged[key] = 0
            self._quiet_until[key] = max(self._quiet_until.get(key, 0.0), now + self.quiet_seconds)
        else:
            self._unacknowledged[key] = count
