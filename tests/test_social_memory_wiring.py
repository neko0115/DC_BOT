from __future__ import annotations

import inspect
import unittest

from discord_ai_assistant.main import AssistantBot


class SocialMemoryProductionWiringTests(unittest.TestCase):
    def test_setup_hook_shares_memory_session_with_social_participant(self) -> None:
        source = inspect.getsource(AssistantBot.setup_hook)
        self.assertIn(
            "core_commands.social.session_state = core_commands.memory_session",
            source,
        )
        self.assertIn(
            "MemoryV2PassiveRuntime(self, self.database, self.ai, core_commands.memory_session)",
            source,
        )


if __name__ == "__main__":
    unittest.main()
