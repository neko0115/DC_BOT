from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from discord_ai_assistant.meeting_glossary import (
    LIFEAFTER_GLOSSARY,
    apply_glossary_corrections,
    enrich_initial_prompt,
    glossary_for_title,
    glossary_for_transcript,
)
from discord_ai_assistant.meeting_upload_glossary import (
    GlossaryAwareLocalMeetingTranscriber,
    _MEETING_TITLE,
)


class MeetingGlossaryTests(unittest.TestCase):
    def test_lifeafter_glossary_selection_and_prompt(self) -> None:
        self.assertIs(glossary_for_title("明日之後週會"), LIFEAFTER_GLOSSARY)
        self.assertIsNone(glossary_for_title("一般工作週會"))
        self.assertIs(glossary_for_transcript("今天討論高校、爭霸賽和活力點。"), LIFEAFTER_GLOSSARY)
        prompt = enrich_initial_prompt("繁體中文逐字稿。", LIFEAFTER_GLOSSARY)
        self.assertIsNotNone(prompt)
        self.assertIn("浴血盾牌", prompt or "")
        self.assertIn("晶蝶無人機", prompt or "")
        self.assertIn("迷霧的 boss", prompt or "")

    def test_lifeafter_confirmed_corrections_only(self) -> None:
        raw = "卡西 3、預寫、肉語、謝古人、蝴蝶無人機、黑毛、信息、迷糊的 Force、三網、第四網"
        corrected, applied = apply_glossary_corrections(raw, LIFEAFTER_GLOSSARY)
        self.assertIn("海域 3", corrected)
        self.assertIn("浴血", corrected)
        self.assertIn("肉魚", corrected)
        self.assertIn("屑骨人", corrected)
        self.assertIn("晶蝶無人機", corrected)
        self.assertIn("黑馬", corrected)
        self.assertIn("新興", corrected)
        self.assertIn("迷霧的 boss", corrected)
        self.assertIn("三王", corrected)
        self.assertIn("第四網", corrected)  # 尚有歧義，不做自動校正
        self.assertEqual(len(applied), 9)

    def test_glossary_aware_transcriber_biases_prompt_and_records_audit(self) -> None:
        captured: dict[str, object] = {}

        class FakeModel:
            def transcribe(self, path: str, **kwargs):
                captured.update(kwargs)
                return (
                    iter(
                        [
                            SimpleNamespace(
                                start=0.0,
                                end=4.0,
                                text="卡西 3 已經打掉，蝴蝶無人機很好用。",
                                avg_logprob=-0.1,
                                no_speech_prob=0.02,
                            ),
                            SimpleNamespace(
                                start=4.0,
                                end=8.0,
                                text="高校後面都是四面楚歌，三網打完再說。",
                                avg_logprob=-0.2,
                                no_speech_prob=0.03,
                            ),
                        ]
                    ),
                    SimpleNamespace(language="zh"),
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcriber = GlossaryAwareLocalMeetingTranscriber(root)
            config = {
                "whisper_model": "small",
                "whisper_language": "zh",
                "whisper_beam_size": 5,
                "whisper_initial_prompt": "繁體中文遊戲討論。",
            }
            token = _MEETING_TITLE.set("明日之後週會")
            try:
                with mock.patch.object(transcriber, "_get_model", return_value=FakeModel()):
                    result = transcriber.transcribe(root / "meeting.mp3", config)
            finally:
                _MEETING_TITLE.reset(token)

            self.assertIn("海域 3", result.text)
            self.assertIn("晶蝶無人機", result.text)
            self.assertIn("三王", result.text)
            self.assertIn("浴血盾牌", str(captured["initial_prompt"]))
            self.assertIn("迷霧的 boss", str(captured["initial_prompt"]))

            audit = json.loads((root / "glossary.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["profile"], "lifeafter")
            corrections = {(item["from"], item["to"]): item["count"] for item in audit["corrections"]}
            self.assertEqual(corrections[("卡西 3", "海域 3")], 1)
            self.assertEqual(corrections[("蝴蝶無人機", "晶蝶無人機")], 1)
            self.assertEqual(corrections[("三網", "三王")], 1)


if __name__ == "__main__":
    unittest.main()
