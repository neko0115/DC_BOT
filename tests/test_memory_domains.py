from __future__ import annotations

import importlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

from discord_ai_assistant.storage.agent_database import AgentDatabase


class MemoryDomainRegistryTests(unittest.TestCase):
    @staticmethod
    def _registry():
        spec = importlib.util.find_spec("discord_ai_assistant.ai.memory_domain_registry")
        if spec is None:
            return None
        return importlib.import_module("discord_ai_assistant.ai.memory_domain_registry")

    def test_registry_module_exists(self) -> None:
        self.assertIsNotNone(self._registry(), "memory_domain_registry module must exist")

    def test_first_class_game_aliases_resolve_to_separate_subdomains(self) -> None:
        registry = self._registry()
        self.assertIsNotNone(registry)
        if registry is None:
            return
        fixtures = {
            "明日之後今晚打副本": "lifeafter",
            "原神這池又歪了": "genshin",
            "星穹鐵道等等刷遺器": "star_rail",
            "絕區零今天抽代理人": "zzz",
            "崩壞3最近又回坑": "honkai3",
            "未定事件簿這張卡好看": "tears_of_themis",
            "Nexus Anima 最近在測試": "nexus_anima",
            "Petit Planet 今天上線": "petit_planet",
            "CS2 晚點打 Premier": "counter_strike",
            "傳說對決等等排位": "arena_of_valor",
            "Minecraft 晚上開生存": "minecraft",
        }
        for text, subdomain in fixtures.items():
            with self.subTest(text=text):
                match = registry.resolve_explicit_domain(text)
                self.assertIsNotNone(match)
                self.assertEqual(match.domain, "game")
                self.assertEqual(match.subdomain, subdomain)
                self.assertGreaterEqual(match.confidence, 0.75)

        self.assertNotEqual(
            registry.resolve_explicit_domain("原神抽卡").subdomain,
            registry.resolve_explicit_domain("星鐵抽卡").subdomain,
        )

    def test_explicit_genshin_gacha_gets_topic(self) -> None:
        registry = self._registry()
        self.assertIsNotNone(registry)
        if registry is None:
            return
        match = registry.resolve_explicit_domain("原神這池又歪了")
        self.assertIsNotNone(match)
        self.assertEqual((match.domain, match.subdomain, match.topic), ("game", "genshin", "gacha"))
        self.assertGreaterEqual(match.confidence, 0.85)

    def test_pr_abbreviation_routes_to_project(self) -> None:
        registry = self._registry()
        self.assertIsNotNone(registry)
        if registry is None:
            return
        match = registry.resolve_explicit_domain("DC_BOT Memory V2 這個 PR 還在測")
        self.assertIsNotNone(match)
        self.assertEqual(match.domain, "project")
        self.assertGreaterEqual(match.confidence, 0.75)

    def test_strong_message_evidence_can_override_game_channel_prior(self) -> None:
        registry = self._registry()
        self.assertIsNotNone(registry)
        if registry is None:
            return
        prior = registry.infer_channel_prior("minecraft", "遊戲區")
        explicit = registry.resolve_explicit_domain("等等晚餐要不要吃燒肉")
        self.assertIsNotNone(prior)
        self.assertEqual((prior.domain, prior.subdomain), ("game", "minecraft"))
        self.assertIsNotNone(explicit)
        self.assertEqual((explicit.domain, explicit.subdomain), ("daily", "food"))
        self.assertGreater(explicit.confidence, prior.confidence - 0.2)

    def test_specific_channel_prior_beats_mixed_category_but_mixed_channel_stays_mixed(self) -> None:
        registry = self._registry()
        self.assertIsNotNone(registry)
        if registry is None:
            return

        minecraft = registry.infer_channel_prior("minecraft", "聊天區")
        self.assertIsNotNone(minecraft)
        self.assertEqual((minecraft.domain, minecraft.subdomain), ("game", "minecraft"))
        self.assertEqual(minecraft.source, "channel_alias")

        mixed = registry.infer_channel_prior("閒聊", "Minecraft")
        self.assertIsNone(mixed)


class MemoryChannelPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = AgentDatabase(Path(self.tempdir.name) / "assistant.sqlite3")

    def tearDown(self) -> None:
        self.database.close()
        self.tempdir.cleanup()

    @staticmethod
    def _policy_module():
        spec = importlib.util.find_spec("discord_ai_assistant.ai.memory_social_policy")
        if spec is None:
            return None
        return importlib.import_module("discord_ai_assistant.ai.memory_social_policy")

    def test_mixed_main_chat_allows_personal_and_shared_without_fixed_domain(self) -> None:
        policy_module = self._policy_module()
        self.assertIsNotNone(policy_module, "memory_social_policy module must exist")
        if policy_module is None:
            return
        policy = policy_module.resolve_channel_memory_policy(
            self.database,
            guild_id=1,
            channel_id=10,
            channel_name="閒聊",
            category_name="聊天區",
        )
        self.assertEqual(policy.mode, "mixed")
        self.assertIsNone(policy.domain_prior)
        self.assertIsNone(policy.subdomain_prior)
        self.assertTrue(policy.enabled)
        self.assertTrue(policy.allow_personal)
        self.assertTrue(policy.allow_shared)

    def test_specific_channel_policy_beats_mixed_category_but_mixed_channel_stays_mixed(self) -> None:
        policy_module = self._policy_module()
        self.assertIsNotNone(policy_module)
        if policy_module is None:
            return

        minecraft = policy_module.resolve_channel_memory_policy(
            self.database,
            guild_id=1,
            channel_id=11,
            channel_name="minecraft",
            category_name="聊天區",
        )
        self.assertEqual(
            (minecraft.mode, minecraft.domain_prior, minecraft.subdomain_prior),
            ("game", "game", "minecraft"),
        )
        self.assertTrue(minecraft.allow_shared)

        mixed = policy_module.resolve_channel_memory_policy(
            self.database,
            guild_id=1,
            channel_id=12,
            channel_name="閒聊",
            category_name="Minecraft",
        )
        self.assertEqual(mixed.mode, "mixed")
        self.assertIsNone(mixed.domain_prior)
        self.assertIsNone(mixed.subdomain_prior)

    def test_manual_channel_override_wins_over_name_and_category(self) -> None:
        policy_module = self._policy_module()
        self.assertIsNotNone(policy_module)
        if policy_module is None:
            return
        self.database.set_state("memory_channel_mode:1:10", "off")
        policy = policy_module.resolve_channel_memory_policy(
            self.database,
            guild_id=1,
            channel_id=10,
            channel_name="minecraft",
            category_name="遊戲區",
        )
        self.assertEqual(policy.mode, "off")
        self.assertFalse(policy.enabled)
        self.assertFalse(policy.allow_personal)
        self.assertFalse(policy.allow_shared)

    def test_game_override_keeps_subdomain_prior_and_project_disables_shared(self) -> None:
        policy_module = self._policy_module()
        self.assertIsNotNone(policy_module)
        if policy_module is None:
            return
        self.database.set_state("memory_channel_mode:1:20", "game:minecraft")
        game = policy_module.resolve_channel_memory_policy(
            self.database,
            guild_id=1,
            channel_id=20,
            channel_name="whatever",
            category_name=None,
        )
        self.assertEqual((game.mode, game.domain_prior, game.subdomain_prior), ("game", "game", "minecraft"))
        self.assertTrue(game.allow_personal)
        self.assertTrue(game.allow_shared)

        self.database.set_state("memory_channel_mode:1:21", "project")
        project = policy_module.resolve_channel_memory_policy(
            self.database,
            guild_id=1,
            channel_id=21,
            channel_name="dev",
            category_name="開發",
        )
        self.assertEqual(project.mode, "project")
        self.assertTrue(project.allow_personal)
        self.assertFalse(project.allow_shared)


if __name__ == "__main__":
    unittest.main()
