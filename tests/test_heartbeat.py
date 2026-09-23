from __future__ import annotations

import unittest
from unittest.mock import patch

from discord_ai_assistant.heartbeat import HeartbeatClient


class HeartbeatClientTests(unittest.TestCase):
    def test_payload_is_minimal_and_reports_ready_state(self) -> None:
        with patch("discord_ai_assistant.heartbeat.time.monotonic", side_effect=[100.0, 145.9]):
            client = HeartbeatClient(
                url="https://example.invalid/api/heartbeat",
                token="secret",
                service_id="discord-bot",
                version="0.7.0",
                interval_seconds=60,
                timeout_seconds=8,
                payload_factory=lambda: {"ready": False, "ignored": "not-forwarded"},
            )
            payload = client.build_payload()

        self.assertEqual(
            payload,
            {
                "service": "discord-bot",
                "ready": False,
                "version": "0.7.0",
                "uptimeSeconds": 45,
            },
        )
        self.assertNotIn("token", payload)
        self.assertNotIn("ignored", payload)

    def test_ready_defaults_true(self) -> None:
        client = HeartbeatClient(
            url="https://example.invalid/api/heartbeat",
            token="secret",
            service_id="discord-bot",
            version="dev",
            interval_seconds=60,
            timeout_seconds=8,
        )

        self.assertTrue(client.build_payload()["ready"])


if __name__ == "__main__":
    unittest.main()
