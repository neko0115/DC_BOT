from __future__ import annotations

import unittest

from discord_ai_assistant.voice.language_routing import (
    detect_tts_language,
    normalize_tts_language,
)


class TTSLanguageRoutingTests(unittest.TestCase):
    def test_traditional_chinese_defaults_to_zh(self) -> None:
        self.assertEqual(detect_tts_language("老大，我是墨雪。今天也請多多指教喔。"), "zh")

    def test_high_confidence_cantonese_routes_to_yue(self) -> None:
        samples = (
            "老大，你今日做咩啊？我喺度等你好耐喇。",
            "我唔知佢哋而家喺邊度。",
            "呢個係咪你想要嘅版本？",
        )
        for sample in samples:
            with self.subTest(sample=sample):
                self.assertEqual(detect_tts_language(sample), "yue")

    def test_english_routes_to_en(self) -> None:
        self.assertEqual(detect_tts_language("Hey boss, I found the answer for you."), "en")

    def test_japanese_routes_to_ja(self) -> None:
        self.assertEqual(detect_tts_language("お兄ちゃん、今日は何してるの？"), "ja")

    def test_korean_routes_to_ko(self) -> None:
        self.assertEqual(detect_tts_language("오늘은 뭐 하고 있어요?"), "ko")

    def test_short_english_model_names_inside_chinese_stay_zh(self) -> None:
        samples = (
            "今天 GPT-SoVITS 壞了，我來看看。",
            "RTX 3090 現在很閒，可以拿來跑 TTS。",
            "這個 Python 版本應該沒問題。",
        )
        for sample in samples:
            with self.subTest(sample=sample):
                self.assertEqual(detect_tts_language(sample), "zh")

    def test_clearly_english_dominant_mixed_sentence_routes_to_en(self) -> None:
        self.assertEqual(detect_tts_language("Hello 老大, how are you doing today?"), "en")

    def test_normalize_uses_fallback_and_rejects_unknown_language(self) -> None:
        self.assertEqual(normalize_tts_language(None, fallback="yue"), "yue")
        self.assertEqual(normalize_tts_language(" JA "), "ja")
        with self.assertRaises(ValueError):
            normalize_tts_language("fr")


if __name__ == "__main__":
    unittest.main()
