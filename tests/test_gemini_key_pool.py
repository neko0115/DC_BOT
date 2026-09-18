from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from discord_ai_assistant.ai.key_pool import FailoverGeminiClient, load_gemini_api_keys


class _GeminiError(RuntimeError):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class _Interactions:
    def __init__(self, result: object | None = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls = 0

    def create(self, **kwargs: object) -> object:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


class _SequenceInteractions:
    def __init__(self, outcomes: list[object | Exception]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    def create(self, **kwargs: object) -> object:
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class GeminiKeyPoolTests(unittest.TestCase):
    def test_create_once_defers_key_failover_until_next_request(self) -> None:
        first = _Interactions(error=_GeminiError('SYNTHETIC', 429))
        result = SimpleNamespace(id='backup-once', output_text='ok')
        second = _Interactions(result=result)
        clients = {'primary': SimpleNamespace(interactions=first), 'backup': SimpleNamespace(interactions=second)}
        client = FailoverGeminiClient(('primary', 'backup'), client_factory=lambda key: clients[key],
                                      single_attempt_client_factory=lambda key: clients[key])
        with self.assertRaises(_GeminiError):
            client.interactions.create_once(model='test')
        self.assertEqual((first.calls, second.calls), (1, 0))
        self.assertEqual(client.key_pool.active_number, 2)
        self.assertIs(client.interactions.create_once(model='test'), result)
        self.assertEqual((first.calls, second.calls), (1, 1))

    def test_create_once_keeps_continuations_on_the_successful_key(self) -> None:
        first_result = SimpleNamespace(id='once-pinned', output_text='ok')
        first = _SequenceInteractions([first_result, _GeminiError('SYNTHETIC', 429)])
        backup = _Interactions(result=SimpleNamespace(id='next', output_text='ok'))
        clients = {'primary': SimpleNamespace(interactions=first), 'backup': SimpleNamespace(interactions=backup)}
        client = FailoverGeminiClient(('primary', 'backup'), client_factory=lambda key: clients[key],
                                      single_attempt_client_factory=lambda key: clients[key])
        self.assertIs(client.interactions.create_once(model='test'), first_result)
        with self.assertRaises(_GeminiError):
            client.interactions.create_once(model='test', previous_interaction_id='once-pinned')
        self.assertEqual((first.calls, backup.calls), (2, 0))
        self.assertEqual(client.interactions.create(model='test').id, 'next')
        self.assertEqual((first.calls, backup.calls), (2, 1))

    def test_create_once_does_not_rotate_key_on_server_error(self) -> None:
        first = _Interactions(error=_GeminiError('SYNTHETIC', 503))
        second = _Interactions(result=SimpleNamespace(output_text='unused'))
        clients = {'primary': SimpleNamespace(interactions=first), 'backup': SimpleNamespace(interactions=second)}
        client = FailoverGeminiClient(('primary', 'backup'), client_factory=lambda key: clients[key],
                                      single_attempt_client_factory=lambda key: clients[key])
        with self.assertRaises(_GeminiError):
            client.interactions.create_once(model='test')
        self.assertEqual((first.calls, second.calls), (1, 0))
        self.assertEqual(client.key_pool.active_number, 1)

    def test_loads_primary_csv_and_numbered_keys_without_duplicates(self) -> None:
        environment = {
            "GEMINI_API_KEY": "primary",
            "GEMINI_BACKUP_API_KEYS": "backup-a, backup-b,primary",
            "GEMINI_API_KEY_2": "backup-c",
        }
        with patch.dict(os.environ, environment, clear=True):
            keys = load_gemini_api_keys()
        self.assertEqual(keys, ("primary", "backup-a", "backup-b", "backup-c"))

    def test_quota_error_rotates_to_backup_key(self) -> None:
        first = _Interactions(error=_GeminiError("RESOURCE_EXHAUSTED", 429))
        second_result = SimpleNamespace(id="backup-interaction", output_text="ok")
        second = _Interactions(result=second_result)
        clients = {
            "primary": SimpleNamespace(interactions=first),
            "backup": SimpleNamespace(interactions=second),
        }
        client = FailoverGeminiClient(
            ("primary", "backup"),
            client_factory=lambda key: clients[key],
            quota_cooldown_seconds=60,
        )

        result = client.interactions.create(model="test")

        self.assertIs(result, second_result)
        self.assertEqual(first.calls, 1)
        self.assertEqual(second.calls, 1)
        self.assertEqual(client.key_pool.active_number, 2)

    def test_server_error_does_not_rotate_keys(self) -> None:
        first = _Interactions(error=_GeminiError("upstream unavailable", 500))
        second = _Interactions(result=SimpleNamespace(output_text="should-not-run"))
        clients = {
            "primary": SimpleNamespace(interactions=first),
            "backup": SimpleNamespace(interactions=second),
        }
        client = FailoverGeminiClient(("primary", "backup"), client_factory=lambda key: clients[key])

        with self.assertRaises(_GeminiError):
            client.interactions.create(model="test")

        self.assertEqual(first.calls, 1)
        self.assertEqual(second.calls, 0)
        self.assertEqual(client.key_pool.active_number, 1)

    def test_continuation_stays_on_originating_key(self) -> None:
        first_interaction = SimpleNamespace(id="interaction-1", output_text="tool call")
        first = _SequenceInteractions(
            [first_interaction, _GeminiError("RESOURCE_EXHAUSTED", 429)]
        )
        backup = _Interactions(result=SimpleNamespace(id="wrong-project", output_text="should-not-run"))
        clients = {
            "primary": SimpleNamespace(interactions=first),
            "backup": SimpleNamespace(interactions=backup),
        }
        client = FailoverGeminiClient(
            ("primary", "backup"),
            client_factory=lambda key: clients[key],
            quota_cooldown_seconds=60,
        )

        result = client.interactions.create(model="test")
        self.assertIs(result, first_interaction)
        with self.assertRaises(_GeminiError):
            client.interactions.create(model="test", previous_interaction_id="interaction-1")

        self.assertEqual(first.calls, 2)
        self.assertEqual(backup.calls, 0)
        # The exhausted primary is now cooling down, so the next fresh request starts on backup.
        fresh = client.interactions.create(model="test")
        self.assertEqual(fresh.output_text, "should-not-run")
        self.assertEqual(client.key_pool.active_number, 2)


if __name__ == "__main__":
    unittest.main()
