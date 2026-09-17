from __future__ import annotations

import asyncio
import unittest

from discord_ai_assistant.tool_gateway.client import ToolGatewayClient


class ToolGatewayClientTests(unittest.TestCase):
    def test_catalog_is_filtered_by_trigger_keyword(self) -> None:
        client = ToolGatewayClient()
        client._replace_catalog(
            {
                "tools": [
                    {
                        "name": "sample",
                        "trigger_keywords": ["工具測試"],
                        "always_available": False,
                        "actions": [
                            {
                                "name": "echo",
                                "function_name": "x_sample_echo",
                                "description": "echo text",
                                "parameters": {"type": "object", "properties": {}},
                                "requires_dj": False,
                            }
                        ],
                    }
                ],
                "load_errors": {},
            }
        )

        self.assertEqual(client.declarations_for("普通聊天", is_dj=False), [])
        declarations = client.declarations_for("幫我做工具測試", is_dj=False)
        self.assertEqual([item["name"] for item in declarations], ["x_sample_echo"])
        self.assertTrue(client.handles("x_sample_echo"))

    def test_trigger_matching_normalizes_spacing_punctuation_and_cjk_variant(self) -> None:
        client = ToolGatewayClient()
        client._replace_catalog(
            {
                "tools": [
                    {
                        "name": "recorder",
                        "trigger_keywords": ["會議紀錄"],
                        "always_available": False,
                        "actions": [
                            {
                                "name": "status",
                                "function_name": "x_recorder_status",
                                "description": "status",
                                "parameters": {"type": "object", "properties": {}},
                                "requires_dj": False,
                            }
                        ],
                    }
                ]
            }
        )

        declarations = client.declarations_for("會議・紀録 工具狀態", is_dj=False)
        self.assertEqual([item["name"] for item in declarations], ["x_recorder_status"])

    def test_action_specific_keywords_expose_only_relevant_action(self) -> None:
        client = ToolGatewayClient()
        client._replace_catalog(
            {
                "tools": [
                    {
                        "name": "recorder",
                        "trigger_keywords": ["會議紀錄"],
                        "always_available": False,
                        "actions": [
                            {
                                "name": "status",
                                "function_name": "x_recorder_status",
                                "description": "status",
                                "parameters": {"type": "object", "properties": {}},
                                "requires_dj": False,
                                "trigger_keywords": ["工具狀態"],
                            },
                            {
                                "name": "start",
                                "function_name": "x_recorder_start",
                                "description": "start",
                                "parameters": {"type": "object", "properties": {}},
                                "requires_dj": True,
                                "trigger_keywords": ["開始會議紀錄"],
                            },
                        ],
                    }
                ]
            }
        )

        status = client.declarations_for("會議紀錄工具狀態", is_dj=True)
        start = client.declarations_for("開始會議紀錄", is_dj=True)
        self.assertEqual([item["name"] for item in status], ["x_recorder_status"])
        self.assertEqual([item["name"] for item in start], ["x_recorder_start"])

    def test_always_available_tool_is_exposed_without_keyword(self) -> None:
        client = ToolGatewayClient()
        client._replace_catalog(
            {
                "tools": [
                    {
                        "name": "sample",
                        "trigger_keywords": [],
                        "always_available": True,
                        "actions": [
                            {
                                "name": "status",
                                "function_name": "x_sample_status",
                                "description": "status",
                                "parameters": {"type": "object", "properties": {}},
                                "requires_dj": False,
                            }
                        ],
                    }
                ]
            }
        )

        self.assertEqual(len(client.declarations_for("hello", is_dj=False)), 1)

    def test_dj_only_action_is_hidden_from_non_dj(self) -> None:
        client = ToolGatewayClient()
        client._replace_catalog(
            {
                "tools": [
                    {
                        "name": "recorder",
                        "trigger_keywords": ["會議紀錄"],
                        "always_available": False,
                        "actions": [
                            {
                                "name": "start",
                                "function_name": "x_recorder_start",
                                "description": "start recording",
                                "parameters": {"type": "object", "properties": {}},
                                "requires_dj": True,
                            },
                            {
                                "name": "status",
                                "function_name": "x_recorder_status",
                                "description": "status",
                                "parameters": {"type": "object", "properties": {}},
                                "requires_dj": False,
                            },
                        ],
                    }
                ]
            }
        )

        non_dj = client.declarations_for("會議紀錄", is_dj=False)
        dj = client.declarations_for("會議紀錄", is_dj=True)
        self.assertEqual([item["name"] for item in non_dj], ["x_recorder_status"])
        self.assertEqual({item["name"] for item in dj}, {"x_recorder_start", "x_recorder_status"})

    def test_dj_only_action_is_blocked_before_http_request(self) -> None:
        client = ToolGatewayClient()
        client._replace_catalog(
            {
                "tools": [
                    {
                        "name": "recorder",
                        "trigger_keywords": ["會議紀錄"],
                        "always_available": False,
                        "actions": [
                            {
                                "name": "start",
                                "function_name": "x_recorder_start",
                                "description": "start recording",
                                "parameters": {"type": "object", "properties": {}},
                                "requires_dj": True,
                            }
                        ],
                    }
                ]
            }
        )

        with self.assertRaises(PermissionError):
            asyncio.run(client.invoke("x_recorder_start", {}, {"is_dj": False}))


if __name__ == "__main__":
    unittest.main()
