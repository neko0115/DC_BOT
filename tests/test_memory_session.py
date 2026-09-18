from __future__ import annotations

import importlib
import importlib.util
import unittest

from discord_ai_assistant.ai.memory_domain_registry import resolve_explicit_domain


class MemorySessionTests(unittest.TestCase):
    @staticmethod
    def _module():
        spec = importlib.util.find_spec("discord_ai_assistant.ai.memory_session")
        if spec is None:
            return None
        return importlib.import_module("discord_ai_assistant.ai.memory_session")

    def test_session_module_exists(self) -> None:
        self.assertIsNotNone(self._module(), "memory_session module must exist")

    def test_ambiguous_fragment_follows_same_participant_topic(self) -> None:
        module = self._module()
        self.assertIsNotNone(module)
        if module is None:
            return
        state = module.ConversationSessionState()
        first = state.observe(
            module.SessionMessage(10, 1, 100, "原神這池又歪了"),
            explicit=resolve_explicit_domain("原神這池又歪了"),
            channel_prior=None,
            now=0.0,
        )
        follow = state.observe(
            module.SessionMessage(10, 1, 101, "我真的不想抽了"),
            explicit=None,
            channel_prior=None,
            now=20.0,
        )
        self.assertIsNotNone(first)
        self.assertIsNotNone(follow)
        self.assertEqual((follow.domain, follow.subdomain, follow.topic), ("game", "genshin", "gacha"))
        self.assertEqual(follow.session_key, first.session_key)
        self.assertIn(1, follow.participant_ids)

    def test_channel_keeps_multiple_topics_and_participant_affinity(self) -> None:
        module = self._module()
        self.assertIsNotNone(module)
        if module is None:
            return
        state = module.ConversationSessionState()
        state.observe(
            module.SessionMessage(10, 1, 100, "原神這池又歪了"),
            explicit=resolve_explicit_domain("原神這池又歪了"),
            channel_prior=None,
            now=0.0,
        )
        state.observe(
            module.SessionMessage(10, 3, 101, "晚餐吃燒肉嗎"),
            explicit=resolve_explicit_domain("晚餐吃燒肉嗎"),
            channel_prior=None,
            now=5.0,
        )
        genshin = state.observe(
            module.SessionMessage(10, 1, 102, "真的不想抽了"),
            explicit=None,
            channel_prior=None,
            now=10.0,
        )
        food = state.observe(
            module.SessionMessage(10, 3, 103, "那家要排隊嗎"),
            explicit=None,
            channel_prior=None,
            now=12.0,
        )
        self.assertEqual((genshin.subdomain, genshin.topic), ("genshin", "gacha"))
        self.assertEqual((food.subdomain, food.topic), ("food", "meal"))
        active = state.active_topics(10, now=12.0)
        self.assertEqual(len(active), 2)
        self.assertEqual({item.subdomain for item in active}, {"genshin", "food"})

    def test_reply_to_older_message_overrides_newer_unrelated_topic(self) -> None:
        module = self._module()
        self.assertIsNotNone(module)
        if module is None:
            return
        state = module.ConversationSessionState()
        genshin = state.observe(
            module.SessionMessage(10, 1, 100, "原神這期深淵好煩"),
            explicit=resolve_explicit_domain("原神這期深淵好煩"),
            channel_prior=None,
            now=0.0,
        )
        state.observe(
            module.SessionMessage(10, 2, 101, "CS2 等等要不要打"),
            explicit=resolve_explicit_domain("CS2 等等要不要打"),
            channel_prior=None,
            now=5.0,
        )
        reply = state.observe(
            module.SessionMessage(10, 3, 102, "最後那隻真的很煩", reply_to_message_id=100),
            explicit=None,
            channel_prior=None,
            now=10.0,
        )
        self.assertIsNotNone(genshin)
        self.assertIsNotNone(reply)
        self.assertEqual(reply.session_key, genshin.session_key)
        self.assertEqual(reply.subdomain, "genshin")
        self.assertEqual(reply.source, "reply")

    def test_topic_decay_cuts_off_implicit_context_and_archives(self) -> None:
        module = self._module()
        self.assertIsNotNone(module)
        if module is None:
            return
        state = module.ConversationSessionState()
        first = state.observe(
            module.SessionMessage(10, 1, 100, "原神這池又歪了"),
            explicit=resolve_explicit_domain("原神這池又歪了"),
            channel_prior=None,
            now=0.0,
        )
        self.assertIsNotNone(first)
        near = state.active_topics(10, now=120.0)
        self.assertEqual(len(near), 1)
        self.assertGreater(near[0].confidence, 0.75)

        mid = state.observe(
            module.SessionMessage(10, 1, 101, "真的不想抽"),
            explicit=None,
            channel_prior=None,
            now=600.0,
        )
        self.assertIsNotNone(mid)
        self.assertLess(mid.confidence, first.confidence)

        # The valid continuation at t=600 refreshes the topic.  Implicit context cuts
        # off 15 minutes after that latest participation, not 15 minutes after creation.
        cutoff = state.observe(
            module.SessionMessage(10, 1, 102, "還是算了"),
            explicit=None,
            channel_prior=None,
            now=1501.0,
        )
        self.assertIsNone(cutoff)
        self.assertEqual(state.active_topics(10, now=2401.0), ())

        revived = state.observe(
            module.SessionMessage(10, 1, 103, "原神我又想抽了"),
            explicit=resolve_explicit_domain("原神我又想抽了"),
            channel_prior=None,
            now=2500.0,
        )
        self.assertIsNotNone(revived)
        self.assertNotEqual(revived.session_key, first.session_key)

    def test_observe_is_idempotent_for_same_message_id(self) -> None:
        module = self._module()
        self.assertIsNotNone(module)
        if module is None:
            return
        state = module.ConversationSessionState()
        message = module.SessionMessage(10, 1, 100, "Minecraft 晚上開生存")
        explicit = resolve_explicit_domain(message.content)
        first = state.observe(message, explicit=explicit, channel_prior=None, now=0.0)
        second = state.observe(message, explicit=explicit, channel_prior=None, now=1.0)
        self.assertEqual(first, second)
        self.assertEqual(len(state.active_topics(10, now=1.0)), 1)

    def test_message_resolution_caches_are_bounded_per_channel(self) -> None:
        module = self._module()
        self.assertIsNotNone(module)
        if module is None:
            return
        cap = module.MAX_RESOLVED_MESSAGES_PER_CHANNEL
        self.assertEqual(cap, 256)
        state = module.ConversationSessionState()

        explicit = resolve_explicit_domain("原神這池又歪了")
        for index in range(cap + 32):
            state.observe(
                module.SessionMessage(10, 1, 1000 + index, "原神這池又歪了"),
                explicit=explicit,
                channel_prior=None,
                now=float(index),
            )
        resolved_keys = [key for key in state._resolved_messages if key[0] == 10]
        topic_keys = [key for key in state._message_to_topic if key[0] == 10]
        self.assertLessEqual(len(resolved_keys), cap)
        self.assertLessEqual(len(topic_keys), cap)

        for index in range(cap + 32):
            state.observe(
                module.SessionMessage(11, 2, 2000 + index, "哈哈"),
                explicit=None,
                channel_prior=None,
                now=float(index),
            )
        unresolved_keys = [key for key in state._resolved_messages if key[0] == 11]
        self.assertLessEqual(len(unresolved_keys), cap)

    def test_session_keys_do_not_collide_across_state_instances(self) -> None:
        module = self._module()
        self.assertIsNotNone(module)
        if module is None:
            return

        first_state = module.ConversationSessionState()
        second_state = module.ConversationSessionState()

        first_content = "Minecraft 昨天村民交易所的工具箱被挖掉了"
        second_content = "Minecraft 昨天地獄交通站的補給箱被挖掉了"

        first = first_state.observe(
            module.SessionMessage(10, 1, 100, first_content),
            explicit=resolve_explicit_domain(first_content),
            channel_prior=None,
            now=0.0,
        )
        second = second_state.observe(
            module.SessionMessage(10, 2, 200, second_content),
            explicit=resolve_explicit_domain(second_content),
            channel_prior=None,
            now=0.0,
        )

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertNotEqual(first.session_key, second.session_key)


if __name__ == "__main__":
    unittest.main()
