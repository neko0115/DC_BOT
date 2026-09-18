from __future__ import annotations

import importlib
import importlib.util
import unittest

from discord_ai_assistant.ai.memory_domain_registry import resolve_explicit_domain


class MemorySocialTranscriptSessionTests(unittest.TestCase):
    @staticmethod
    def _module():
        spec = importlib.util.find_spec("discord_ai_assistant.ai.memory_session")
        if spec is None:
            return None
        return importlib.import_module("discord_ai_assistant.ai.memory_session")

    def test_mixed_chat_preserves_genshin_food_and_cs_threads(self) -> None:
        module = self._module()
        self.assertIsNotNone(module, "memory_session module must exist")
        if module is None:
            return
        state = module.ConversationSessionState()

        def observe(author: int, message_id: int, content: str, now: float, reply_to: int | None = None):
            return state.observe(
                module.SessionMessage(50, author, message_id, content, reply_to_message_id=reply_to),
                explicit=resolve_explicit_domain(content),
                channel_prior=None,
                now=now,
            )

        genshin = observe(1, 100, "原神這池又歪了", 0.0)
        b_follow = observe(2, 101, "笑死你又來", 5.0)
        food = observe(3, 102, "晚餐要不要吃燒肉", 10.0)
        a_follow = observe(1, 103, "我真的不想抽了", 15.0)
        cs = observe(4, 104, "等等開 CS2", 20.0)
        reply = observe(2, 105, "你上次不是也這樣", 25.0, reply_to=100)

        self.assertEqual((genshin.subdomain, genshin.topic), ("genshin", "gacha"))
        self.assertEqual(b_follow.session_key, genshin.session_key)
        self.assertEqual(a_follow.session_key, genshin.session_key)
        self.assertEqual(reply.session_key, genshin.session_key)
        self.assertEqual((food.domain, food.subdomain), ("daily", "food"))
        self.assertEqual((cs.domain, cs.subdomain), ("game", "counter_strike"))

        active = state.active_topics(50, now=25.0)
        self.assertEqual(len(active), 3)
        self.assertEqual({item.subdomain for item in active}, {"genshin", "food", "counter_strike"})


if __name__ == "__main__":
    unittest.main()
