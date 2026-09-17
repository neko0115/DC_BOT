from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from discord_ai_assistant.app_knowledge import AppKnowledgeStore
from discord_ai_assistant.storage.database import Database


class AppKnowledgeStoreTests(unittest.TestCase):
    def test_profiles_terms_and_default_are_guild_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database = Database(Path(temporary_directory) / "assistant.sqlite3")
            store = AppKnowledgeStore(database)
            store.upsert_profile(10, "dcs", "DCS", "飛行模擬作戰討論", "DCS, BVR")
            store.upsert_term(10, "dcs", "MAR", "Minimum Abort Range", "minimum abort range")
            store.set_default_profile(10, "dcs")

            self.assertEqual(store.default_profile(10).key, "dcs")
            self.assertIsNone(store.default_profile(11))
            self.assertEqual(store.resolve_profile(10, "今晚 DCS BVR 週會").key, "dcs")
            term = store.list_terms(10, "dcs")[0]
            self.assertEqual(term.term, "MAR")
            self.assertEqual(term.explanation, "Minimum Abort Range")
            self.assertEqual(term.aliases, ("minimum abort range",))
            database.close()

    def test_lifeafter_seed_contains_confirmed_terms_and_explanations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database = Database(Path(temporary_directory) / "assistant.sqlite3")
            store = AppKnowledgeStore(database)
            store.ensure_lifeafter_seed(10)

            profile = store.get_profile(10, "lifeafter")
            self.assertIsNotNone(profile)
            snapshot = store.snapshot(10, "明日之後高校週會")
            self.assertIsNotNone(snapshot)
            assert snapshot is not None
            terms = {item.term: item for item in snapshot.terms}
            self.assertIn("晶蝶無人機", terms)
            self.assertIn("蝴蝶無人機", terms["晶蝶無人機"].aliases)
            self.assertIn("無人機", terms["晶蝶無人機"].explanation)
            self.assertIn("詞庫只用來理解", snapshot.explanation_block())
            database.close()

    def test_user_can_update_explanation_without_changing_aliases_unless_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database = Database(Path(temporary_directory) / "assistant.sqlite3")
            store = AppKnowledgeStore(database)
            store.upsert_profile(10, "game", "Game", "測試遊戲")
            store.upsert_term(10, "game", "晶蝶無人機", "第一版解釋", "蝴蝶無人機")
            store.upsert_term(10, "game", "晶蝶無人機", "新的簡短解釋", "蝴蝶無人機")

            item = store.list_terms(10, "game")[0]
            self.assertEqual(item.explanation, "新的簡短解釋")
            self.assertEqual(item.aliases, ("蝴蝶無人機",))
            database.close()


if __name__ == "__main__":
    unittest.main()
