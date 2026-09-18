from __future__ import annotations

import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from discord_ai_assistant.ai.gemini import AssistantReply
from discord_ai_assistant.ai.tools import ToolRouter
from discord_ai_assistant.commands import AssistantCommands
from discord_ai_assistant.tool_effect_commands import ToolEffectAssistantCommands, split_discord_message


class ToolEffectTests(unittest.TestCase):
    def test_summary_metadata_becomes_publish_effect(self) -> None:
        result = ToolRouter._attach_external_effects(
            {
                "summary_channel_id": 123456,
                "summary_instruction": "只根據逐字稿整理。",
                "transcript": "hello",
            }
        )
        self.assertEqual(
            result["_moxue_effects"],
            [{"type": "publish_final_reply", "channel_id": 123456, "suppress_origin": True}],
        )

    def test_missing_summary_channel_does_not_create_effect(self) -> None:
        result = ToolRouter._attach_external_effects(
            {"summary_channel_id": None, "summary_instruction": "整理。", "transcript": "hello"}
        )
        self.assertNotIn("_moxue_effects", result)

    def test_artifact_metadata_becomes_core_owned_effect_and_is_hidden_from_model(self) -> None:
        result = ToolRouter._attach_external_effects(
            {
                "message": "done",
                "_moxue_artifacts": [
                    {
                        "relative_path": "image_generation/abc.jpg",
                        "filename": "generated.jpg",
                        "mime_type": "image/jpeg",
                        "delete_after_send": True,
                    }
                ],
            }
        )
        self.assertNotIn("_moxue_artifacts", result)
        self.assertEqual(
            result["_moxue_effects"],
            [
                {
                    "type": "attach_artifact",
                    "relative_path": "image_generation/abc.jpg",
                    "filename": "generated.jpg",
                    "mime_type": "image/jpeg",
                    "delete_after_send": True,
                }
            ],
        )

    def test_assistant_reply_effects_default_to_empty(self) -> None:
        reply = AssistantReply("hello", used_tools=False)
        self.assertEqual(reply.effects, ())

    def test_long_discord_output_is_split_below_safe_limit(self) -> None:
        text = ("一段會議內容 " * 500) + "\n\n" + ("另一段內容 " * 500)
        chunks = split_discord_message(text)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(0 < len(chunk) <= 1900 for chunk in chunks))

    def test_tool_effect_cog_keeps_original_name(self) -> None:
        self.assertEqual(ToolEffectAssistantCommands.__cog_name__, "AssistantCommands")

    def test_runtime_cog_overrides_legacy_five_message_history(self) -> None:
        self.assertIsNot(ToolEffectAssistantCommands._with_history, AssistantCommands._with_history)

    def test_memory_v2_uses_current_request_as_retrieval_query(self) -> None:
        database = SimpleNamespace(
            user_memory_context=MagicMock(return_value="- [事件] 火箭回收段主傘曾經打到尾翼")
        )
        core = SimpleNamespace(
            ai=SimpleNamespace(_request_text=lambda prompt: "之前火箭主傘發生什麼事"),
            database=database,
        )

        enriched = ToolEffectAssistantCommands._with_user_memory(
            core,
            "Discord 對話脈絡：...\n\n目前請求：之前火箭主傘發生什麼事",
            1,
            2,
        )

        database.user_memory_context.assert_called_once_with(
            1,
            2,
            query="之前火箭主傘發生什麼事",
        )
        self.assertIn("火箭回收段主傘", enriched)
        self.assertIn("依目前問題檢索", enriched)

    def test_native_knowledge_help_uses_non_failing_message_reference(self) -> None:
        source = inspect.getsource(ToolEffectAssistantCommands._deliver_knowledge_help)
        self.assertIn("to_reference(fail_if_not_exists=False)", source)
        self.assertIn("message.channel.send(", source)
        self.assertNotIn("message.reply(", source)


if __name__ == "__main__":
    unittest.main()
