from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from discord_ai_assistant.ai.chat_style import compute_chat_style_stats, prepare_chat_style_content
from discord_ai_assistant.ai.memory_phase2 import _is_passive_sensitive
from discord_ai_assistant.chat_style_store import ChatStyleStore
from discord_ai_assistant.storage.agent_database import AgentDatabase


UTC = timezone.utc
BASE = datetime(2026, 9, 10, 0, 0, tzinfo=UTC)


def _structured_address_cases() -> tuple[str, ...]:
    hierarchies = (
        ("花蓮縣", "秀林鄉", "富世村"),
        ("高雄市", "桃源區", "梅山里"),
        ("秀林鄉", "富世村"),
        ("桃源區", "梅山里"),
    )
    neighborhoods = ("", "4鄰", "第4鄰", "五鄰", "第五鄰", "十二鄰", "第十二鄰")
    houses = ("123號", "１２３號", "123之1號", "123-1號")
    return tuple(
        separator.join((*hierarchy, *((neighborhood,) if neighborhood else ()), house))
        for hierarchy in hierarchies
        for neighborhood in neighborhoods
        for house in houses
        for separator in ("", " ")
    )


_NON_ADDRESS_CONTROLS = (
    "花蓮縣秀林鄉富世村",
    "高雄市桃源區梅山里",
    "新北市有29區",
    "Minecraft 第4村有100個村民",
    "第12區3號選手",
    "梅山里程碑123",
    "桃源區版本2.0",
    "花蓮縣面積4628平方公里",
    "Python 3.12",
    "port 8002",
    "花蓮縣秀林鄉，富世村第4鄰123號",
)

_LOCALITY_SUFFIX_IN_VILLAGE_CASES = (
    "秀林鄉測鄉試村第4鄰123號",
    "秀林鄉測試鎮村第五鄰123號",
    "桃源區測市試里第十二鄰123-1號",
    "桃源區測試區里第八鄰１２３之４號",
    "示例市區 測試里 第八鄰 123號",
)


def _ambiguous_admin_component_division_cases() -> tuple[str, ...]:
    hierarchies = (
        ("桃源區", "測試區村里"),
        ("秀林鄉", "測試鎮里村"),
    )
    neighborhoods = ("", "第4鄰", "第十二鄰")
    houses = ("123號", "123之1號", "123-1號")
    return tuple(
        separator.join((*hierarchy, *((neighborhood,) if neighborhood else ()), house))
        for hierarchy in hierarchies
        for neighborhood in neighborhoods
        for house in houses
        for separator in ("", " ")
    )


class ChatStyleStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = AgentDatabase(Path(self.tempdir.name) / "assistant.sqlite3")
        self.store = ChatStyleStore(self.database)

    def tearDown(self) -> None:
        self.database.close()
        self.tempdir.cleanup()

    def _record(
        self,
        index: int,
        *,
        guild_id: int = 1,
        user_id: int = 10,
        at: datetime | None = None,
        content: str | None = None,
        source_content: str | None = None,
        is_reply: bool = False,
        is_bot: bool = False,
        is_dm: bool = False,
        has_stickers: bool = False,
        is_system: bool = False,
        session_key: str | None = None,
    ) -> int | None:
        return self.store.record_sample(
            guild_id=guild_id,
            user_id=user_id,
            channel_id=100,
            message_id=1000 + index,
            content=content if content is not None else f"這是第 {index} 則有效聊天樣本 alpha",
            source_content=source_content,
            observed_at=at or BASE + timedelta(minutes=index),
            session_key=session_key,
            is_reply=is_reply,
            is_bot=is_bot,
            is_dm=is_dm,
            has_stickers=has_stickers,
            is_system=is_system,
        )

    def test_raw_artifacts_rejected_before_clean_display_persistence(self) -> None:
        cases = (
            ("<@123456789012345678>", "@SyntheticMember"),
            ("<@&123456789012345678>", "@SyntheticRole"),
            ("<#123456789012345678>", "#synthetic-channel"),
            ("<@123456789012345678> <:cat:222222222222222222>", "@SyntheticMember :cat:"),
            ("<@123456789012345678> https://example.com", "@SyntheticMember https://example.com"),
        )
        for index, (raw, clean) in enumerate(cases):
            with self.subTest(raw=raw):
                self.assertIsNone(prepare_chat_style_content(clean, source_content=raw))
                self.assertIsNone(self._record(index, content=clean, source_content=raw))
                self.assertEqual(self.database.connection.execute(
                    "SELECT COUNT(*) FROM chat_style_samples WHERE message_id=? OR content=?",
                    (1000 + index, clean),
                ).fetchone()[0], 0)

    def test_keycap_and_mass_mention_artifacts_never_persist(self) -> None:
        cases = (
            "1️⃣", "7️⃣ 😂", "1️⃣ 2️⃣ 3️⃣", "#️⃣", "*️⃣", "1\u20e3",
            "@everyone", "@here", "@everyone 1️⃣", "@here 😂",
            "@everyone https://example.com", "@here <:cat:123456789012345678>",
            "<@123456789012345678> @everyone", "<#123456789012345678> 1️⃣",
        )
        for index, content in enumerate(cases):
            with self.subTest(content=content):
                self.assertIsNone(prepare_chat_style_content(content))
                self.assertIsNone(self._record(index, content=content))
                self.assertEqual(self.database.connection.execute(
                    "SELECT COUNT(*) FROM chat_style_samples WHERE message_id=? OR content=?",
                    (1000 + index, content),
                ).fetchone()[0], 0)

    def test_keycap_and_mass_mention_mixed_text_preserves_clean_content(self) -> None:
        cases = (
            "1️⃣ 這波可以", "第 1️⃣ 波可以", "@everyone 開會了", "@here 有人知道嗎",
            "不是 @everyone，是這個設定", "hello 1️⃣",
            "someone@everyone.example", "@everyoneish", "@hereafter",
        )
        for index, content in enumerate(cases):
            with self.subTest(content=content):
                self.assertEqual(prepare_chat_style_content(content), content)
                self.assertIsNotNone(self._record(index, content=content))
                self.assertEqual(self.database.connection.execute(
                    "SELECT content FROM chat_style_samples WHERE message_id=?", (1000 + index,),
                ).fetchone()[0], content)

    def test_meaningful_raw_mentions_persist_only_normalized_clean_text(self) -> None:
        cases = (
            ("<@123456789012345678> 今天真的笑死", "  @SyntheticMember  今天真的笑死  "),
            ("<#123456789012345678> 去這邊看一下紅石比較器", "#general 去這邊看一下紅石比較器"),
        )
        for index, (raw, clean) in enumerate(cases):
            with self.subTest(raw=raw):
                expected = " ".join(clean.split())
                self.assertEqual(prepare_chat_style_content(clean, source_content=raw), expected)
                self.assertIsNotNone(self._record(index, content=clean, source_content=raw))
                persisted = self.database.connection.execute(
                    "SELECT content FROM chat_style_samples WHERE message_id=?", (1000 + index,),
                ).fetchone()[0]
                self.assertEqual(persisted, expected)
                self.assertNotIn("123456789012345678", persisted)

    def test_source_and_display_both_respect_existing_sensitive_and_slash_gates(self) -> None:
        for index, (raw, clean) in enumerate((
            ("我的 API key 是 sk-test-not-real", "一般測試聊天"),
            ("一般測試聊天", "我的 API key 是 sk-test-not-real"),
            ("/memory list", "一般測試聊天"),
            ("一般測試聊天", "/memory list"),
            ("", "@SyntheticMember"),
            ("一般測試聊天", ""),
        )):
            with self.subTest(raw=raw, clean=clean):
                self.assertIsNone(prepare_chat_style_content(clean, source_content=raw))
                self.assertIsNone(self._record(index, content=clean, source_content=raw))
        self.assertEqual(self.database.connection.execute(
            "SELECT COUNT(*) FROM chat_style_samples"
        ).fetchone()[0], 0)

    def test_schema_is_separate_from_user_memories(self) -> None:
        tables = {
            str(row[0])
            for row in self.database.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        self.assertTrue(
            {
                "chat_style_profiles",
                "chat_style_samples",
                "chat_style_profile_updates",
                "chat_style_learning",
            }.issubset(tables)
        )
        memory_columns = {
            str(row[1])
            for row in self.database.connection.execute("PRAGMA table_info(user_memories)").fetchall()
        }
        self.assertFalse(any(name.startswith("chat_style_") for name in memory_columns))

    def test_initial_profile_requires_thirty_effective_samples(self) -> None:
        for index in range(29):
            self._record(index)
        eligibility = self.store.profile_update_eligibility(1, 10, now=BASE + timedelta(hours=1))
        self.assertFalse(eligibility.eligible)
        self.assertEqual(eligibility.effective_sample_count, 29)

        self._record(29)
        eligibility = self.store.profile_update_eligibility(1, 10, now=BASE + timedelta(hours=1))
        self.assertTrue(eligibility.eligible)
        self.assertEqual(eligibility.reason, "initial_threshold")
        self.assertEqual(eligibility.effective_sample_count, 30)

    def test_existing_profile_requires_twenty_four_new_samples_and_eight_hours(self) -> None:
        for index in range(30):
            self._record(index)
        watermark = self.store.latest_sample_id(1, 10, now=BASE + timedelta(hours=1))
        assert watermark is not None
        self.store.commit_profile(
            1,
            10,
            {"style": {"conciseness": "balanced"}, "understanding": []},
            sample_watermark_id=watermark,
            updated_at=BASE + timedelta(hours=1),
        )

        for index in range(30, 53):
            self._record(index, at=BASE + timedelta(hours=2, minutes=index))
        eligibility = self.store.profile_update_eligibility(1, 10, now=BASE + timedelta(hours=8, minutes=58))
        self.assertFalse(eligibility.eligible)
        self.assertEqual(eligibility.new_sample_count, 23)

        self._record(53, at=BASE + timedelta(hours=8, minutes=59))
        eligibility = self.store.profile_update_eligibility(1, 10, now=BASE + timedelta(hours=8, minutes=59))
        self.assertFalse(eligibility.eligible)
        self.assertEqual(eligibility.reason, "cooldown")
        self.assertEqual(eligibility.new_sample_count, 24)

        eligibility = self.store.profile_update_eligibility(1, 10, now=BASE + timedelta(hours=9, minutes=1))
        self.assertTrue(eligibility.eligible)
        self.assertEqual(eligibility.reason, "incremental_threshold")

    def test_existing_profile_requires_twenty_four_new_samples_after_cooldown(self) -> None:
        # Commit one real update at T0; all 54 samples remain within 72h/80 rows.
        for index in range(30):
            self.assertIsNotNone(self._record(index, at=BASE - timedelta(minutes=30 - index)))
        watermark = self.store.latest_sample_id(1, 10, now=BASE)
        self.assertIsNotNone(watermark)
        self.store.commit_profile(
            1, 10, {"style": {"conciseness": "balanced"}, "understanding": []},
            sample_watermark_id=watermark, updated_at=BASE,
        )
        profile = self.store.get_profile(1, 10)
        self.assertIsNotNone(profile)
        self.assertEqual(profile.sample_watermark_id, watermark)
        self.assertEqual(profile.last_successful_update_at, BASE)
        now = BASE + timedelta(hours=9)
        self.assertGreater(now - profile.last_successful_update_at, timedelta(hours=8))
        self.assertEqual(self.database.connection.execute(
            "SELECT COUNT(*) FROM chat_style_profile_updates WHERE guild_id=1 AND user_id=10"
        ).fetchone()[0], 1)

        for index in range(30, 53):
            sample_id = self._record(index, at=BASE + timedelta(hours=8, minutes=index - 30))
            self.assertIsNotNone(sample_id)
            self.assertGreater(sample_id, watermark)
        blocked = self.store.profile_update_eligibility(1, 10, now=now)
        self.assertFalse(blocked.eligible)
        self.assertEqual(blocked.reason, "insufficient_new_samples")
        self.assertEqual(blocked.new_sample_count, 23)
        self.assertEqual(blocked.effective_sample_count, 53)

        self.assertIsNotNone(self._record(53, at=now))
        allowed = self.store.profile_update_eligibility(1, 10, now=now)
        self.assertTrue(allowed.eligible)
        self.assertEqual(allowed.reason, "incremental_threshold")
        self.assertEqual(allowed.new_sample_count, 24)
        self.assertEqual(allowed.effective_sample_count, 54)

    def test_rolling_twenty_four_hour_budget_allows_at_most_two_successful_updates(self) -> None:
        for index in range(30):
            self._record(index)
        first_watermark = self.store.latest_sample_id(1, 10, now=BASE + timedelta(hours=1))
        assert first_watermark is not None
        self.store.commit_profile(1, 10, {"style": {}}, sample_watermark_id=first_watermark, updated_at=BASE + timedelta(hours=1))

        for index in range(30, 54):
            self._record(index, at=BASE + timedelta(hours=9, minutes=index - 30))
        second_watermark = self.store.latest_sample_id(1, 10, now=BASE + timedelta(hours=10))
        assert second_watermark is not None
        self.assertTrue(self.store.profile_update_eligibility(1, 10, now=BASE + timedelta(hours=10)).eligible)
        self.store.commit_profile(1, 10, {"style": {"revision": 2}}, sample_watermark_id=second_watermark, updated_at=BASE + timedelta(hours=10))

        for index in range(54, 78):
            self._record(index, at=BASE + timedelta(hours=18, minutes=index - 54))
        blocked = self.store.profile_update_eligibility(1, 10, now=BASE + timedelta(hours=19))
        self.assertFalse(blocked.eligible)
        self.assertEqual(blocked.reason, "daily_budget")

        allowed = self.store.profile_update_eligibility(1, 10, now=BASE + timedelta(hours=26))
        self.assertTrue(allowed.eligible)

    def test_sample_retention_keeps_newest_eighty_and_expires_after_seventy_two_hours(self) -> None:
        for index in range(81):
            self._record(index, at=BASE + timedelta(minutes=index))
        rows = self.store.list_samples(1, 10, now=BASE + timedelta(hours=2))
        self.assertEqual(len(rows), 80)
        self.assertEqual(rows[0].content, "這是第 1 則有效聊天樣本 alpha")
        self.assertEqual(rows[-1].content, "這是第 80 則有效聊天樣本 alpha")

        rows = self.store.list_samples(1, 10, now=BASE + timedelta(hours=73, minutes=30))
        self.assertEqual(rows, [])
        persisted = self.database.connection.execute(
            "SELECT COUNT(*) FROM chat_style_samples WHERE guild_id = 1 AND user_id = 10"
        ).fetchone()[0]
        self.assertEqual(persisted, 0)

    def test_nonmeaningful_and_sensitive_messages_never_persist_raw_text(self) -> None:
        rejected = (
            dict(content="/memory list"),
            dict(content="https://example.com/a"),
            dict(content="😀😂✨"),
            dict(content="", has_stickers=True),
            dict(content="<@123456789>"),
            dict(content="一般聊天", is_bot=True),
            dict(content="一般聊天", is_dm=True),
            dict(content="一般聊天", is_system=True),
            dict(content="我的 API Key 是 abc123"),
            dict(content="醫生剛診斷我有高血壓"),
            dict(content="聽說小明最近跟某人分手了"),
            dict(content="我的政治立場最近改變了"),
        )
        for index, kwargs in enumerate(rejected):
            with self.subTest(kwargs=kwargs):
                self.assertIsNone(self._record(index, **kwargs))
        self.assertEqual(self.store.list_samples(1, 10, now=BASE + timedelta(hours=1)), [])

        joined = "\n".join(
            str(row[0])
            for row in self.database.connection.execute(
                "SELECT content FROM chat_style_samples"
            ).fetchall()
        )
        for secret in ("abc123", "高血壓", "分手", "政治立場"):
            self.assertNotIn(secret, joined)

    def test_non_text_artifacts_are_rejected_before_sample_persistence(self) -> None:
        cases = (
            "<:cat:123456789012345678>",
            "<a:dance:123456789012345678>",
            "<:cat:111111111111111111> <:dog:222222222222222222>",
            "<:cat:111111111111111111> 😂✨",
            "<@123456789012345678> <:cat:111111111111111111>",
            "https://a.example/x https://b.example/y",
            "https://a.example/x <:cat:111111111111111111>",
            "https://a.example/x 😂",
        )
        for index, content in enumerate(cases):
            with self.subTest(content=content):
                filtered = prepare_chat_style_content(content)
                sample_id = self._record(index, content=content)
                rows = self.database.connection.execute(
                    "SELECT COUNT(*) FROM chat_style_samples WHERE message_id=? OR content=?",
                    (1000 + index, content),
                ).fetchone()[0]
                self.assertEqual(
                    {"filtered": filtered, "sample_id": sample_id, "raw_rows": rows},
                    {"filtered": None, "sample_id": None, "raw_rows": 0},
                )

    def test_meaningful_text_with_artifacts_preserves_normalized_content(self) -> None:
        cases = (
            "  笑死   <:cat:123456789012345678>  ",
            "看這個 https://example.com/x",
            "網址在這 https://a.example https://b.example",
            "hello 😂",
            "紅石比較器",
            "<something>",
            ":smile:",
            "<@123456789012345678> hello <:cat:111111111111111111>",
        )
        for index, content in enumerate(cases):
            with self.subTest(content=content):
                normalized = " ".join(content.split())
                self.assertEqual(prepare_chat_style_content(content), normalized)
                sample_id = self._record(index, content=content)
                self.assertIsNotNone(sample_id)
                row = self.database.connection.execute(
                    "SELECT content FROM chat_style_samples WHERE id=?", (sample_id,),
                ).fetchone()
                self.assertEqual(row[0], normalized)

    def test_sensitive_categories_are_filtered_before_raw_sample_persistence(self) -> None:
        # Entirely synthetic values; a safe control proves collection is enabled.
        self.assertIsNotNone(self._record(999, at=BASE))
        cases = (
            ("address", "我的地址是測試市測試路123號"),
            ("financial", "我的銀行帳號是 TEST-0000-1234"),
            ("credential", "我的 API key 是 sk-test-not-real-123456"),
            ("health", "我的診斷是測試用資料"),
            ("relationship_gossip", "聽說測試甲和測試乙最近在交往"),
        )
        for index, (category, value) in enumerate(cases):
            with self.subTest(category=category):
                filtered = prepare_chat_style_content(
                    value, is_bot=False, is_dm=False, has_stickers=False, is_system=False,
                )
                sample_id = self._record(index, content=value)
                # Query raw storage immediately, without list_samples/pruning.
                message_rows = self.database.connection.execute(
                    "SELECT COUNT(*) FROM chat_style_samples "
                    "WHERE guild_id=? AND user_id=? AND message_id=?",
                    (1, 10, 1000 + index),
                ).fetchone()[0]
                content_rows = self.database.connection.execute(
                    "SELECT COUNT(*) FROM chat_style_samples WHERE content=?", (value,),
                ).fetchone()[0]
                self.assertIsNone(filtered)
                self.assertIsNone(sample_id)
                self.assertEqual(message_rows, 0)
                self.assertEqual(content_rows, 0)

    def test_inherited_identity_religion_and_orientation_categories_never_persist_source_or_display(
        self,
    ) -> None:
        # Literal synthetic fixtures protect every previously uncovered Memory V2 category.
        fixtures = (
            ("identity_traditional", "這是合成身分證識別字 SAMPLE-IDENTITY-A"),
            ("identity_variant", "這是合成身份證識別字 SAMPLE-IDENTITY-B"),
            ("passport", "這是合成護照識別字 SAMPLE-PASSPORT"),
            ("religion", "我的宗教資料是合成測試內容"),
            ("sexual_orientation", "我的性向資料是合成測試內容"),
        )
        safe_content = "這是可持久化的安全對照樣本"
        safe_sample_id = self._record(999, at=BASE, content=safe_content)
        self.assertIsNotNone(safe_sample_id)
        self.assertEqual(
            self.database.connection.execute(
                "SELECT content FROM chat_style_samples WHERE id=?", (safe_sample_id,),
            ).fetchone()[0],
            safe_content,
        )

        index = 0
        for category, value in fixtures:
            for variant, (source, display) in enumerate((
                (value, value),
                (value, "安全顯示文字"),
                ("安全來源文字", value),
            )):
                with self.subTest(category=category, variant=variant):
                    sensitive = _is_passive_sensitive(value)
                    prepared = prepare_chat_style_content(display, source_content=source)
                    sample_id = self._record(
                        index, at=BASE, content=display, source_content=source,
                    )
                    count = self.database.connection.execute(
                        "SELECT COUNT(*) FROM chat_style_samples WHERE message_id=? OR content=?",
                        (1000 + index, display),
                    ).fetchone()[0]
                    self.assertEqual(
                        (sensitive, prepared is None, sample_id is None, count),
                        (True, True, True, 0),
                    )
                index += 1

    def test_unlabelled_phone_and_address_shapes_never_persist_source_or_display(self) -> None:
        values = (
            "0912-345-678", "0912 345 678", "0912345678", "+886 912 345 678",
            "+886 (0)912-345-678", "０９１２－３４５－６７８", "0912.345.678",
            "台北市信義路100號", "臺北市信義路 100 號", "信義路100號",
            "臺北市信義路一百號", "寄到台北市信義區松仁路 100 號 5 樓",
        )
        self.assertIsNotNone(self._record(999, at=BASE, content="安全的樣本控制組"))
        index = 0
        for value in values:
            for raw, display in ((value, value), (value, "安全顯示文字"), ("安全來源文字", value)):
                with self.subTest(value=value, source=raw, display=display):
                    filtered = prepare_chat_style_content(display, source_content=raw)
                    sample_id = self._record(index, at=BASE, content=display, source_content=raw)
                    rows = self.database.connection.execute(
                        "SELECT COUNT(*) FROM chat_style_samples WHERE message_id=? OR content=?",
                        (1000 + index, display),
                    ).fetchone()[0]
                    self.assertEqual((filtered, sample_id, rows), (None, None, 0))
                index += 1

    def test_version_numbers_and_scores_remain_eligible_samples(self) -> None:
        for index, value in enumerate(("Python 3.12.1", "CS2 1.6", "版本 2026.09", "比分 2-0")):
            with self.subTest(value=value):
                self.assertEqual(prepare_chat_style_content(value, source_content=value), value)
                sample_id = self._record(index, content=value)
                self.assertIsNotNone(sample_id)
                self.assertEqual(self.database.connection.execute(
                    "SELECT content FROM chat_style_samples WHERE id=?", (sample_id,),
                ).fetchone()[0], value)

    def test_regrouped_mobile_numbers_never_persist_source_or_display(self) -> None:
        values = {"0912-34-5678", "09-1234-5678", "0912-345-678", "0912 345 678", "0912345678",
                  "09 1234 5678", "09-12-345-678", "+886 912 345 678", "+886-912-34-5678",
                  "(0912) 345-678", "+886 (0)912-34-5678", "0 9 1 2 3 4 5 6 7 8"}
        # Move a separator through every digit boundary, including prefix digits.
        for digits, prefix in (("0912345678", ""), ("886912345678", "+")):
            for split in range(1, len(digits)):
                for separator in (" ", "-"):
                    values.add(prefix + digits[:split] + separator + digits[split:])
            values.add(prefix + "- ".join(digits))
        self.assertIsNotNone(self._record(999, at=BASE, content="安全控制樣本"))
        index = 0
        for value in sorted(values):
            for source, display in ((value, value), (value, "安全顯示文字"), ("安全來源文字", value)):
                with self.subTest(value=value, source=source, display=display):
                    filtered = prepare_chat_style_content(display, source_content=source)
                    sample_id = self._record(index, at=BASE, content=display, source_content=source)
                    count = self.database.connection.execute(
                        "SELECT COUNT(*) FROM chat_style_samples WHERE message_id=?", (1000 + index,),
                    ).fetchone()[0]
                    self.assertEqual((filtered, sample_id, count), (None, None, 0))
                index += 1

    def test_phone_digit_boundaries_preserve_versions_scores_and_overlong_numbers(self) -> None:
        values = ("Python 3.12.1", "CS2 1.6", "版本 2026.09", "比分 2-0", "RFC 9110", "port 8002",
                  "10912345678", "09123456789", "1-0912345678", "0912345678-9",
                  "+8869123456789", "+88691234567", "008869123456789", "091234567")
        for index, value in enumerate(values):
            with self.subTest(value=value):
                self.assertEqual(prepare_chat_style_content(value, source_content=value), value)
                sample_id = self._record(index, at=BASE, content=value)
                self.assertIsNotNone(sample_id)
                self.assertEqual(self.database.connection.execute(
                    "SELECT content FROM chat_style_samples WHERE id=?", (sample_id,),
                ).fetchone()[0], value)

    def test_administrative_addresses_never_persist_source_or_display(self) -> None:
        # Synthetic house-number fixtures, never copied from user messages.
        values = (
            "花蓮縣秀林鄉富世村123號", "嘉義縣阿里山鄉樂野村4鄰123號",
            "南投縣仁愛鄉大同村16鄰25號", "臺東縣蘭嶼鄉紅頭村50號",
            "高雄市桃源區梅山里123號", "臺中市和平區梨山里5鄰88號",
            "花蓮縣 秀林鄉 富世村 123號", "嘉義縣阿里山鄉樂野村 4鄰 123號",
            "花蓮縣秀林鄉富世村１２３號", "花蓮縣秀林鄉富世村123之1號",
            "花蓮縣秀林鄉富世村123-1號", "花蓮縣秀林鄉示例村123號",
            "南投縣仁愛鄉示例村1鄰123號", "秀林鄉富世村123號",
            "秀林 鄉 富世 村 １２３－１ 號", "秀林鄉\u3000富世村\u00a0123號",
            "示例鎮測試里四鄰一百二十三之二號", "示例市測試里123號",
            "秀林鄉富世村123–1號",
        )
        self.assertIsNotNone(self._record(999, at=BASE, content="一般地理話題控制組"))
        index = 0
        for fixture, value in enumerate(values):
            for source, display in ((value, value), (value, "安全顯示文字"), ("安全來源文字", value)):
                with self.subTest(fixture=fixture, variant=index % 3):
                    prepared = prepare_chat_style_content(display, source_content=source)
                    sample_id = self._record(index, at=BASE, content=display, source_content=source)
                    count = self.database.connection.execute(
                        "SELECT COUNT(*) FROM chat_style_samples WHERE message_id=?", (1000 + index,),
                    ).fetchone()[0]
                    self.assertEqual((prepared is None, sample_id is None, count), (True, True, 0))
                index += 1

    def test_geographic_game_and_place_names_without_addresses_remain_eligible(self) -> None:
        values = (
            "Minecraft 村民 100 個", "新北市有 29 區", "第 12 區 3 號選手", "第 5 村任務",
            "梅山里程碑 123", "桃源區版本 2.0", "花蓮縣面積約 4628 平方公里",
            "Python 3.12", "port 8002", "花蓮縣秀林鄉富世村", "高雄市桃源區梅山里",
        )
        for index, value in enumerate(values):
            with self.subTest(fixture=index):
                self.assertEqual(prepare_chat_style_content(value, source_content=value), value)
                sample_id = self._record(index, at=BASE, content=value, source_content=value)
                self.assertIsNotNone(sample_id)
                self.assertEqual(self.database.connection.execute(
                    "SELECT content FROM chat_style_samples WHERE id=?", (sample_id,),
                ).fetchone()[0], value)

    def test_structured_address_matrix_never_persists_source_or_display(self) -> None:
        self.assertIsNotNone(self._record(999, at=BASE, content="安全控制樣本"))
        index = 0
        for fixture, value in enumerate(_structured_address_cases()):
            pairs = (
                (value, value),
                (value, "安全顯示文字"),
                ("安全來源文字", value),
            )
            for variant, (source, display) in enumerate(pairs):
                with self.subTest(fixture=fixture, variant=variant):
                    sensitive = _is_passive_sensitive(value)
                    prepared = prepare_chat_style_content(display, source_content=source)
                    sample_id = self._record(
                        index, at=BASE, content=display, source_content=source,
                    )
                    count = self.database.connection.execute(
                        "SELECT COUNT(*) FROM chat_style_samples WHERE message_id=?",
                        (1000 + index,),
                    ).fetchone()[0]
                    self.assertEqual(
                        (sensitive, prepared is None, sample_id is None, count),
                        (True, True, True, 0),
                    )
                index += 1

    def test_structured_address_non_address_controls_remain_eligible(self) -> None:
        for index, value in enumerate(_NON_ADDRESS_CONTROLS):
            with self.subTest(fixture=index):
                self.assertFalse(_is_passive_sensitive(value))
                self.assertEqual(
                    prepare_chat_style_content(value, source_content=value), value,
                )
                sample_id = self._record(
                    index, at=BASE, content=value, source_content=value,
                )
                self.assertIsNotNone(sample_id)
                stored = self.database.connection.execute(
                    "SELECT content FROM chat_style_samples WHERE id=?", (sample_id,),
                ).fetchone()
                self.assertIsNotNone(stored)
                self.assertEqual(stored[0], value)
        for index, value in enumerate(("縣", "市", "區", "鄉", "村", "里", "鄰", "號", "123")):
            with self.subTest(token=index):
                self.assertFalse(_is_passive_sensitive(value))

    def test_administrative_suffix_only_combinations_remain_eligible_and_persist(self) -> None:
        values = ("縣鄉村里123號", "鄉鎮村里123號", "縣 鄉 村 里 123號")
        for index, value in enumerate(values):
            with self.subTest(fixture=index):
                self.assertFalse(_is_passive_sensitive(value))
                self.assertEqual(
                    prepare_chat_style_content(value, source_content=value), value,
                )
                sample_id = self._record(
                    index, at=BASE, content=value, source_content=value,
                )
                self.assertIsNotNone(sample_id)
                stored = self.database.connection.execute(
                    "SELECT content FROM chat_style_samples WHERE id=?", (sample_id,),
                ).fetchone()
                self.assertIsNotNone(stored)
                self.assertEqual(stored[0], value)

    def test_locality_suffix_inside_village_name_never_persists_source_or_display(self) -> None:
        self.assertIsNotNone(self._record(999, at=BASE, content="安全控制樣本"))
        index = 0
        for fixture, value in enumerate(_LOCALITY_SUFFIX_IN_VILLAGE_CASES):
            for variant, (source, display) in enumerate((
                (value, value),
                (value, "安全顯示文字"),
                ("安全來源文字", value),
            )):
                with self.subTest(fixture=fixture, variant=variant):
                    sensitive = _is_passive_sensitive(value)
                    prepared = prepare_chat_style_content(display, source_content=source)
                    sample_id = self._record(
                        index, at=BASE, content=display, source_content=source,
                    )
                    count = self.database.connection.execute(
                        "SELECT COUNT(*) FROM chat_style_samples WHERE message_id=?",
                        (1000 + index,),
                    ).fetchone()[0]
                    self.assertEqual(
                        (sensitive, prepared is None, sample_id is None, count),
                        (True, True, True, 0),
                    )
                index += 1

    def test_every_bounded_admin_component_division_rejects_before_persistence(self) -> None:
        self.assertIsNotNone(self._record(999, at=BASE, content="安全控制樣本"))
        index = 0
        for fixture, value in enumerate(_ambiguous_admin_component_division_cases()):
            for variant, (source, display) in enumerate((
                (value, value),
                (value, "安全顯示文字"),
                ("安全來源文字", value),
            )):
                with self.subTest(fixture=fixture, variant=variant):
                    sensitive = _is_passive_sensitive(value)
                    prepared = prepare_chat_style_content(display, source_content=source)
                    sample_id = self._record(
                        index, at=BASE, content=display, source_content=source,
                    )
                    count = self.database.connection.execute(
                        "SELECT COUNT(*) FROM chat_style_samples WHERE message_id=?",
                        (1000 + index,),
                    ).fetchone()[0]
                    self.assertEqual(
                        (sensitive, prepared is None, sample_id is None, count),
                        (True, True, True, 0),
                    )
                index += 1

    def test_learning_off_stops_collection_and_update_eligibility_without_deleting_existing_data(self) -> None:
        for index in range(30):
            self._record(index)
        self.assertEqual(len(self.store.list_samples(1, 10, now=BASE + timedelta(hours=1))), 30)
        self.assertTrue(self.store.learning_enabled(1, 10))

        self.store.set_learning_enabled(1, 10, False, updated_at=BASE + timedelta(hours=1))
        self.assertFalse(self.store.learning_enabled(1, 10))
        self.assertIsNone(self._record(31, at=BASE + timedelta(hours=2)))
        self.assertEqual(len(self.store.list_samples(1, 10, now=BASE + timedelta(hours=2))), 30)
        eligibility = self.store.profile_update_eligibility(1, 10, now=BASE + timedelta(hours=2))
        self.assertFalse(eligibility.eligible)
        self.assertEqual(eligibility.reason, "learning_disabled")

    def test_samples_and_learning_state_are_guild_and_user_scoped(self) -> None:
        self._record(1, guild_id=1, user_id=10, content="我在一號伺服器使用 alpha")
        self._record(2, guild_id=2, user_id=10, content="我在二號伺服器使用 beta")
        self._record(3, guild_id=1, user_id=11, content="另一個使用者使用 gamma")
        self.store.set_learning_enabled(1, 10, False, updated_at=BASE)

        self.assertEqual([row.content for row in self.store.list_samples(1, 10, now=BASE + timedelta(hours=1))], ["我在一號伺服器使用 alpha"])
        self.assertEqual([row.content for row in self.store.list_samples(2, 10, now=BASE + timedelta(hours=1))], ["我在二號伺服器使用 beta"])
        self.assertEqual([row.content for row in self.store.list_samples(1, 11, now=BASE + timedelta(hours=1))], ["另一個使用者使用 gamma"])
        self.assertFalse(self.store.learning_enabled(1, 10))
        self.assertTrue(self.store.learning_enabled(2, 10))
        self.assertTrue(self.store.learning_enabled(1, 11))

    def test_local_stats_are_computed_without_a_model_call(self) -> None:
        self._record(1, content="短句😀？", is_reply=True)
        self._record(2, content="alpha 中文 mixed text!", is_reply=False)
        self._record(3, content="alpha 再來一次", is_reply=True)
        rows = self.store.list_samples(1, 10, now=BASE + timedelta(hours=1))
        stats = compute_chat_style_stats(rows)
        self.assertEqual(stats.sample_count, 3)
        self.assertGreater(stats.average_length, 0)
        self.assertGreater(stats.reply_ratio, 0)
        self.assertGreater(stats.emoji_ratio, 0)
        self.assertGreater(stats.question_ratio, 0)
        self.assertGreater(stats.mixed_script_ratio, 0)
        self.assertIn("alpha", stats.recurring_tokens)

    def test_unknown_visibility_defaults_false_and_dm_cannot_be_public(self) -> None:
        self._record(1)
        self.assertFalse(self.store.list_samples(1, 10, now=BASE + timedelta(hours=1))[0].is_public_evidence)
        self.assertIsNone(self.store.record_sample(
            guild_id=1, user_id=10, channel_id=100, message_id=2222,
            content="alpha-hop 使用", observed_at=BASE, is_dm=True, is_public_evidence=True,
            session_key=None, is_reply=False, is_bot=False, has_stickers=False, is_system=False))

    def test_existing_sample_schema_migrates_without_assuming_public_visibility(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            database = AgentDatabase(Path(folder) / "legacy.sqlite3")
            try:
                database.connection.executescript("""
                    CREATE TABLE chat_style_samples (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, guild_id INTEGER NOT NULL,
                        user_id INTEGER NOT NULL, channel_id INTEGER NOT NULL,
                        message_id INTEGER NOT NULL, content TEXT NOT NULL,
                        observed_at TEXT NOT NULL, session_key TEXT,
                        is_reply INTEGER NOT NULL DEFAULT 0, UNIQUE(guild_id, message_id));
                """)
                database.connection.execute(
                    "INSERT INTO chat_style_samples (guild_id,user_id,channel_id,message_id,content,observed_at) "
                    "VALUES (1,10,100,1000,?,?)", ("legacy alpha", BASE.isoformat(timespec="microseconds")))
                database.connection.commit()
                migrated = ChatStyleStore(database)
                ChatStyleStore(database)  # Migration is idempotent.
                samples = migrated.list_samples(1, 10, now=BASE + timedelta(hours=1))
                self.assertEqual(len(samples), 1)
                self.assertEqual(samples[0].content, "legacy alpha")
                self.assertFalse(samples[0].is_public_evidence)
            finally:
                database.close()


if __name__ == "__main__":
    unittest.main()
