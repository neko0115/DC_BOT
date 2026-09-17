from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from discord_ai_assistant.tool_gateway.server import ToolGatewayServer


class ToolGatewayServerTests(unittest.TestCase):
    def test_non_loopback_bind_requires_token(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                ToolGatewayServer(Path(directory), host="0.0.0.0", port=8765, token=None)

    def test_loopback_bind_does_not_require_token(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            server = ToolGatewayServer(Path(directory), host="127.0.0.1", port=8765, token=None)
            self.assertEqual(server.host, "127.0.0.1")


class ToolGatewayServerRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_server_can_start_and_stop_on_loopback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            server = ToolGatewayServer(Path(directory), host="127.0.0.1", port=0, token=None)
            await server.start()
            self.assertIsNotNone(server._runner)
            await server.close()
            self.assertIsNone(server._runner)


if __name__ == "__main__":
    unittest.main()
