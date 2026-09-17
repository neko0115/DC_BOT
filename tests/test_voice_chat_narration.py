from __future__ import annotations

import unittest

from discord_ai_assistant.voice.chat_narration import VoiceChatNarrator, sanitize_discord_text


class DiscordTextSanitizerTests(unittest.TestCase):
    def test_custom_emoji_markup_is_removed(self) -> None:
        self.assertEqual(
            sanitize_discord_text("笑死 <:moxue_cat:123456789012345678> 真的"),
            "笑死 真的",
        )

    def test_pure_custom_or_unicode_emoji_is_skipped(self) -> None:
        self.assertIsNone(sanitize_discord_text("<:moxue_cat:123456789012345678>"))
        self.assertIsNone(sanitize_discord_text("😂😂😂"))

    def test_pure_sticker_message_is_skipped(self) -> None:
        self.assertIsNone(sanitize_discord_text("", has_stickers=True))
        self.assertIsNone(
            sanitize_discord_text(
                "https://media.discordapp.net/stickers/123456789.png",
                has_stickers=True,
            )
        )

    def test_urls_are_shortened_and_repeated_characters_are_collapsed(self) -> None:
        self.assertEqual(
            sanitize_discord_text("哈哈哈哈哈哈 看這個 https://example.com/really/long/path"),
            "哈哈哈 看這個 一個連結",
        )

    def test_excessive_sanitized_text_is_rejected(self) -> None:
        long_non_spam_text = "這是一段正常長文字。" * 25
        self.assertGreater(len(long_non_spam_text), 180)
        self.assertIsNone(sanitize_discord_text(long_non_spam_text, max_characters=180))


class VoiceChatNarratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.narrator = VoiceChatNarrator(continuity_seconds=30, max_characters=180)

    def render(self, author_id: int, author_name: str, content: str, now: float) -> str | None:
        return self.narrator.render(
            guild_id=1,
            channel_id=10,
            author_id=author_id,
            author_name=author_name,
            content=content,
            now=now,
        )

    def test_first_message_announces_speaker(self) -> None:
        self.assertEqual(self.render(11, "布丁", "等一下", 100), "布丁說，等一下")

    def test_same_speaker_continues_without_repeating_name(self) -> None:
        self.render(11, "布丁", "等一下", 100)
        self.assertEqual(self.render(11, "布丁", "我去拿東西", 112), "我去拿東西")

    def test_speaker_change_announces_new_name(self) -> None:
        self.render(11, "布丁", "等一下", 100)
        self.assertEqual(self.render(22, "小明", "好", 105), "小明說，好")

    def test_continuity_timeout_reannounces_name(self) -> None:
        self.render(11, "布丁", "等一下", 100)
        self.assertEqual(self.render(11, "布丁", "回來了", 131), "布丁說，回來了")

    def test_continuity_is_scoped_per_channel(self) -> None:
        self.render(11, "布丁", "等一下", 100)
        result = self.narrator.render(
            guild_id=1,
            channel_id=20,
            author_id=11,
            author_name="布丁",
            content="另一邊",
            now=105,
        )
        self.assertEqual(result, "布丁說，另一邊")


if __name__ == "__main__":
    unittest.main()
