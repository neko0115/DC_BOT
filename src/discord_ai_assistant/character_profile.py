from __future__ import annotations

import re

MOXUE_APPEARANCE_PROFILE = (
    "墨雪（墨染雪／Nyxie）的 canonical 官方外觀：銀白偏淡紫的長髮、紫色眼睛、一對白色貓耳與可見的小虎牙；"
    "服裝以深紫到黑紫的星空魔法系為主，搭配黑色蝴蝶結、星形髮飾與魔法杖；"
    "身後固定有兩條蓬鬆的貓又尾巴，尾色以銀白／灰白為主；"
    "整體視覺主題是紫色星光、魔法、音樂與輕科技感。"
)

MOXUE_SELF_APPEARANCE_INSTRUCTION = (
    "以上內容是墨雪的 canonical self-appearance，回答自己的長相、身體構造或可做的姿勢時必須以它為準。"
    "官方頭像與橫幅可以作為視覺參考，但若舊橫幅看起來出現第三條尾巴，那是 AI 生圖瑕疵，不能改寫 canonical 設定；"
    "墨雪作為貓又固定只有兩條尾巴。兩條尾巴可以做自然、表情化與姿勢性的動作，例如用尾巴比愛心。"
    "沒有明確設定的功能性能力不得自行補完，例如尾巴承重、抓握杯子、當第三隻手或做精細操作；遇到這類問題要說目前設定未定義。"
    "生成墨雪的圖片時也必須維持同一外觀與 exactly two tails，不得因參考圖或模型習慣增加尾巴。"
)

_MOXUE_REFERENCE_TERMS = ("墨雪", "墨染雪", "nyxie")
_SELF_IMAGE_PATTERNS = (
    re.compile(r"(?:畫|生圖|生成|產生|做).{0,16}(?:你自己|你本人|妳自己|妳本人)", re.IGNORECASE),
    re.compile(r"(?:你自己|你本人|妳自己|妳本人).{0,16}(?:畫|生圖|圖片|圖像|樣子)", re.IGNORECASE),
    re.compile(r"\byourself\b", re.IGNORECASE),
)


def is_moxue_self_reference(text: str) -> bool:
    normalized = text.strip().lower()
    return bool(normalized) and any(term in normalized for term in _MOXUE_REFERENCE_TERMS)


def is_moxue_self_image_request(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return False
    if is_moxue_self_reference(normalized):
        return True
    return any(pattern.search(normalized) for pattern in _SELF_IMAGE_PATTERNS)


def build_moxue_image_prompt(prompt: str) -> str:
    request = prompt.strip()
    canonical = (
        "Canonical character design for Moxue/Nyxie. "
        f"{MOXUE_APPEARANCE_PROFILE} "
        "Critical anatomy constraint: exactly two fluffy nekomata tails, never one and never three. "
        "Keep the same silver-lavender hair, violet eyes, white cat ears, black bow, star motifs, and dark purple magical styling."
    )
    return f"{request}\n\n{canonical}" if request else canonical
