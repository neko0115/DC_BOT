from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "dev_runner.py"
SPEC = importlib.util.spec_from_file_location("dc_bot_dev_runner", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
DEV_RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DEV_RUNNER
SPEC.loader.exec_module(DEV_RUNNER)


class DevelopmentRunnerTests(unittest.TestCase):
    def test_reuses_existing_moxue_capture_agent(self) -> None:
        health = {"ok": True, "agent_id": "agent-1", "name": "Moxue Capture"}
        with patch.object(DEV_RUNNER, "_existing_capture_agent", return_value=health), patch.object(
            DEV_RUNNER, "_start_module"
        ) as start_module:
            process = DEV_RUNNER._start_capture_agent()

        self.assertIsNone(process)
        start_module.assert_not_called()

    def test_rejects_non_moxue_process_on_capture_port(self) -> None:
        with patch.object(DEV_RUNNER, "_existing_capture_agent", return_value=None), patch.object(
            DEV_RUNNER, "_port_is_open", return_value=True
        ), self.assertRaisesRegex(RuntimeError, "already in use"):
            DEV_RUNNER._start_capture_agent()

    def test_starts_capture_agent_when_port_is_free(self) -> None:
        fake_process = MagicMock()
        with patch.object(DEV_RUNNER, "_existing_capture_agent", return_value=None), patch.object(
            DEV_RUNNER, "_port_is_open", return_value=False
        ), patch.object(DEV_RUNNER, "_start_module", return_value=fake_process) as start_module:
            process = DEV_RUNNER._start_capture_agent()

        self.assertIs(process, fake_process)
        start_module.assert_called_once_with(
            "discord_ai_assistant.capture_agent.cli",
            "--host",
            DEV_RUNNER.CAPTURE_AGENT_HOST,
            "--port",
            str(DEV_RUNNER.CAPTURE_AGENT_PORT),
        )


if __name__ == "__main__":
    unittest.main()
