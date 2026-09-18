from __future__ import annotations

import unittest

from discord_ai_assistant.ai.persona import BASE_PERSONA_INSTRUCTION
from discord_ai_assistant.character_profile import (
    MOXUE_APPEARANCE_PROFILE,
    MOXUE_SELF_APPEARANCE_INSTRUCTION,
    build_moxue_image_prompt,
    is_moxue_self_reference,
)


class MoxueCharacterProfileTests(unittest.TestCase):
    def test_canonical_profile_has_two_tails_and_visual_traits(self) -> None:
        profile = MOXUE_APPEARANCE_PROFILE
        self.assertIn("兩條", profile)
        self.assertNotIn("三條", profile)
        self.assertIn("銀白", profile)
        self.assertIn("紫色", profile)
        self.assertIn("貓耳", profile)
        self.assertIn("黑色蝴蝶結", profile)
        self.assertIn("星", profile)

    def test_persona_system_instruction_includes_canonical_self_appearance(self) -> None:
        self.assertIn(MOXUE_APPEARANCE_PROFILE, BASE_PERSONA_INSTRUCTION)
        self.assertIn(MOXUE_SELF_APPEARANCE_INSTRUCTION, BASE_PERSONA_INSTRUCTION)
        self.assertIn("兩條", BASE_PERSONA_INSTRUCTION)

    def test_instruction_marks_reference_image_tail_mismatch_as_noncanonical(self) -> None:
        instruction = MOXUE_SELF_APPEARANCE_INSTRUCTION
        self.assertIn("第三條", instruction)
        self.assertIn("生圖瑕疵", instruction)
        self.assertIn("canonical", instruction.lower())
        self.assertIn("尾巴比愛心", instruction)
        self.assertIn("承重", instruction)

    def test_self_reference_detection_is_explicit(self) -> None:
        for text in ("畫墨雪在星空下", "Nyxie tail heart", "墨染雪拿著魔法杖"):
            with self.subTest(text=text):
                self.assertTrue(is_moxue_self_reference(text))
        self.assertFalse(is_moxue_self_reference("畫一隻普通白貓"))

    def test_image_prompt_appends_canonical_profile(self) -> None:
        prompt = build_moxue_image_prompt("墨雪用兩條尾巴比愛心")
        self.assertIn("兩條", prompt)
        self.assertIn("銀白", prompt)
        self.assertIn("紫色", prompt)
        self.assertIn("exactly two", prompt.lower())
        self.assertNotIn("三條尾巴", prompt)


if __name__ == "__main__":
    unittest.main()
