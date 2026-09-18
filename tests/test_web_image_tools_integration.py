from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from discord_ai_assistant.ai.gemini import GeminiAssistant, GOOGLE_SEARCH_TOOL

ROOT = Path(__file__).resolve().parents[1]


class WebImageToolIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def test_production_bot_uses_artifact_aware_command_cog(self) -> None:
        source = (ROOT / "src" / "discord_ai_assistant" / "main.py").read_text(encoding="utf-8")
        self.assertIn("ArtifactToolEffectAssistantCommands", source)
        self.assertIn("core_commands = ArtifactToolEffectAssistantCommands(", source)

    def test_environment_template_documents_provider_configuration(self) -> None:
        text = (ROOT / ".env.example").read_text(encoding="utf-8")
        for variable in (
            "TAVILY_API_KEY=",
            "WEB_RESEARCH_MAX_RESULTS=5",
            "WEB_RESEARCH_TIMEOUT_SECONDS=20",
            "CLOUDFLARE_ACCOUNT_ID=",
            "CLOUDFLARE_API_TOKEN=",
            "IMAGE_GEN_MODEL=@cf/black-forest-labs/flux-1-schnell",
            "IMAGE_GEN_STEPS=4",
            "IMAGE_GEN_USER_COOLDOWN_SECONDS=60",
            "IMAGE_GEN_MAX_CONCURRENCY=1",
        ):
            with self.subTest(variable=variable):
                self.assertIn(variable, text)

    def test_proactive_search_only_contract_remains_google_only(self) -> None:
        source = (ROOT / "src" / "discord_ai_assistant" / "ai" / "search_social.py").read_text(encoding="utf-8")
        self.assertIn("tools=[dict(GOOGLE_SEARCH_TOOL)]", source)
        self.assertNotIn("refresh_external_tools", source)

    async def test_explicit_deep_research_does_not_also_expose_native_google_search(self) -> None:
        external_tool = {
            "type": "function",
            "name": "x_web_research_web_research",
            "description": "deep research",
            "parameters": {"type": "object", "properties": {}},
        }
        router = SimpleNamespace(
            refresh_external_tools=AsyncMock(),
            external_declarations_for=MagicMock(return_value=[external_tool]),
        )
        ai = GeminiAssistant("key", "model", router)
        ai._client = object()
        ai._create_interaction = AsyncMock(
            return_value=SimpleNamespace(output_text="ok", steps=[])
        )

        await ai.ask(
            "深入研究最近的顯示卡驅動災情，多來源交叉查證",
            SimpleNamespace(),
        )

        tools = ai._create_interaction.await_args.kwargs["tools"]
        self.assertIn(external_tool, tools)
        self.assertNotIn(GOOGLE_SEARCH_TOOL, tools)


if __name__ == "__main__":
    unittest.main()
