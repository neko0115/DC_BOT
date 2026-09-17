from __future__ import annotations

import unittest

from discord_ai_assistant.history import RecentMessageHistory


class RecentMessageHistoryTests(unittest.TestCase):
    def test_raw_history_is_bounded_in_memory(self) -> None:
        history = RecentMessageHistory(per_channel_limit=5)
        for index in range(8):
            history.add(1, 10, "A", f"訊息 {index}")

        rendered = history.format_for(1, 10, limit=20)

        self.assertNotIn("訊息 0", rendered)
        self.assertNotIn("訊息 2", rendered)
        self.assertIn("訊息 3", rendered)
        self.assertIn("訊息 7", rendered)
        self.assertEqual(len(rendered.splitlines()), 5)

    def test_default_raw_history_keeps_one_hundred_messages(self) -> None:
        history = RecentMessageHistory()
        for index in range(105):
            history.add(1, 10, "A", f"raw-{index}")

        rendered = history.format_for(1, 10, limit=200)
        lines = rendered.splitlines()
        self.assertEqual(len(lines), 100)
        self.assertNotIn("A: raw-4", lines)
        self.assertIn("A: raw-5", lines)
        self.assertIn("A: raw-104", lines)

    def test_compressor_keeps_recent_and_relevant_older_messages(self) -> None:
        history = RecentMessageHistory(per_channel_limit=50, recent_limit=3, older_limit=2)
        history.add(1, 10, "A", "GTX1660Ti 這張卡之後拿來跑常駐 Bot")
        history.add(1, 10, "B", "今天晚餐吃牛肉麵")
        history.add(1, 10, "C", "明天早上可能會下雨")
        history.add(1, 10, "A", "先把服務搬過去測")
        history.add(1, 10, "B", "記得更新套件")
        history.add(1, 10, "A", "那張顯卡的用途你覺得呢")

        context = history.compressed_for(1, 10, query="1660Ti 拿來跑 Bot 適合嗎")

        self.assertIn("GTX1660Ti", context)
        self.assertIn("先把服務搬過去測", context)
        self.assertIn("記得更新套件", context)
        self.assertIn("那張顯卡的用途你覺得呢", context)
        self.assertIn("較早但可能相關", context)
        self.assertIn("最近 Discord 對話", context)

    def test_compressor_deduplicates_and_drops_old_noise(self) -> None:
        history = RecentMessageHistory(per_channel_limit=50, recent_limit=2, older_limit=8)
        history.add(1, 10, "A", "同一個重要資訊")
        history.add(1, 10, "A", "同一個重要資訊")
        history.add(1, 10, "B", "哈哈哈")
        history.add(1, 10, "C", "舊的有效資訊")
        history.add(1, 10, "A", "最近第一句")
        history.add(1, 10, "B", "最近第二句")

        context = history.compressed_for(1, 10, query="重要資訊")

        self.assertEqual(context.count("同一個重要資訊"), 1)
        self.assertNotIn("哈哈哈", context)
        self.assertIn("舊的有效資訊", context)
        self.assertIn("最近第一句", context)
        self.assertIn("最近第二句", context)

    def test_no_query_overlap_falls_back_to_recent_older_context(self) -> None:
        history = RecentMessageHistory(per_channel_limit=50, recent_limit=2, older_limit=2)
        history.add(1, 10, "A", "很早以前的內容")
        history.add(1, 10, "B", "較新的舊內容一")
        history.add(1, 10, "C", "較新的舊內容二")
        history.add(1, 10, "A", "最近一")
        history.add(1, 10, "B", "最近二")

        context = history.compressed_for(1, 10, query="完全不同的主題")

        self.assertNotIn("很早以前的內容", context)
        self.assertIn("較新的舊內容一", context)
        self.assertIn("較新的舊內容二", context)

    def test_compressor_allows_per_call_recent_and_relevant_bounds(self) -> None:
        history = RecentMessageHistory(per_channel_limit=50, recent_limit=6, older_limit=6)
        for index in range(8):
            history.add(1, 10, "A", f"舊內容 {index}")
        history.add(1, 10, "B", "關鍵 GTX1660Ti 資訊")
        history.add(1, 10, "A", "最近甲")
        history.add(1, 10, "B", "最近乙")

        context = history.compressed_for(
            1,
            10,
            query="GTX1660Ti",
            recent_limit=2,
            relevant_limit=1,
        )

        self.assertIn("關鍵 GTX1660Ti 資訊", context)
        self.assertIn("最近甲", context)
        self.assertIn("最近乙", context)
        rendered_messages = [line for line in context.splitlines() if ": " in line]
        self.assertEqual(len(rendered_messages), 3)

    def test_message_count_budget_stops_context_packing(self) -> None:
        history = RecentMessageHistory(per_channel_limit=20, recent_limit=10, older_limit=10)
        for index in range(10):
            history.add(1, 10, "A", f"短訊息 {index}")

        context = history.compressed_for(
            1,
            10,
            query="短訊息",
            max_messages=4,
            max_characters=8000,
            recent_guarantee=2,
        )

        rendered_messages = [line for line in context.splitlines() if ": " in line]
        self.assertEqual(len(rendered_messages), 4)

    def test_character_budget_stops_before_message_budget(self) -> None:
        history = RecentMessageHistory(per_channel_limit=20, recent_limit=10, older_limit=0)
        for index in range(10):
            history.add(1, 10, "A", f"訊息 {index} " + ("內容" * 20))

        context = history.compressed_for(
            1,
            10,
            query="訊息",
            max_messages=10,
            max_characters=220,
            recent_guarantee=2,
        )

        rendered_messages = [line for line in context.splitlines() if ": " in line]
        self.assertLess(len(rendered_messages), 10)
        self.assertLessEqual(len(context), 220)
        self.assertGreaterEqual(len(rendered_messages), 1)

    def test_recent_guarantee_survives_relevance_pressure(self) -> None:
        history = RecentMessageHistory(per_channel_limit=30, recent_limit=6, older_limit=20)
        for index in range(12):
            history.add(1, 10, "Old", f"GTX1660Ti 舊資訊 {index}")
        for index in range(6):
            history.add(1, 10, "Recent", f"眼前對話 {index}")

        context = history.compressed_for(
            1,
            10,
            query="GTX1660Ti",
            max_messages=6,
            max_characters=8000,
            recent_guarantee=3,
        )

        self.assertIn("眼前對話 5", context)
        self.assertIn("眼前對話 4", context)
        self.assertIn("眼前對話 3", context)
        rendered_messages = [line for line in context.splitlines() if ": " in line]
        self.assertEqual(len(rendered_messages), 6)


if __name__ == "__main__":
    unittest.main()
