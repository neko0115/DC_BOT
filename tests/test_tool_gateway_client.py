from __future__ import annotations

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
                            }
                        ],
                    }
                ],
                "load_errors": {},
            }
        )

        self.assertEqual(client.declarations_for("普通聊天"), [])
        declarations = client.declarations_for("幫我做工具測試")
        self.assertEqual([item["name"] for item in declarations], ["x_sample_echo"])
        self.assertTrue(client.handles("x_sample_echo"))

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
                            }
                        ],
                    }
                ]
            }
        )

        self.assertEqual(len(client.declarations_for("hello")), 1)


if __name__ == "__main__":
    unittest.main()
