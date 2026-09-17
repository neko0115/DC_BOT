from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from discord_ai_assistant.capture_agent.remote_config import normalize_hub_url
from discord_ai_assistant.capture_hub import CaptureHubServer


class CaptureHubTests(unittest.TestCase):
    def test_empty_hub_has_no_selected_or_online_agents(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            hub = CaptureHubServer(Path(folder), port=0)
            self.assertIsNone(hub.selected_agent_id(123))
            self.assertEqual(
                hub.list_agents(123),
                {"agents": [], "pending": [], "selected_agent_id": None},
            )

    def test_pairing_code_validation(self) -> None:
        self.assertTrue(CaptureHubServer._valid_pairing_code("ABC234"))
        self.assertFalse(CaptureHubServer._valid_pairing_code("ABC"))
        self.assertFalse(CaptureHubServer._valid_pairing_code("ABC-23"))

    def test_hub_url_normalization(self) -> None:
        self.assertEqual(
            normalize_hub_url("192.168.1.20"),
            "ws://192.168.1.20:8878/capture/ws",
        )
        self.assertEqual(
            normalize_hub_url("192.168.1.20:9000"),
            "ws://192.168.1.20:9000/capture/ws",
        )
        self.assertEqual(
            normalize_hub_url("https://capture.example.test"),
            "wss://capture.example.test/capture/ws",
        )


if __name__ == "__main__":
    unittest.main()
