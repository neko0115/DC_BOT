from __future__ import annotations

import asyncio
import base64
import uuid
from pathlib import Path


class WindowsSpeechSynthesizer:
    """Creates a WAV file with the Windows SAPI voice available on the host."""

    def __init__(self, output_directory: Path) -> None:
        self.output_directory = output_directory
        self.output_directory.mkdir(parents=True, exist_ok=True)

    async def synthesize(self, text: str) -> Path:
        if not text.strip() or len(text) > 400:
            raise ValueError("朗讀內容需介於 1 到 400 個字元。")
        output_path = self.output_directory / f"speech-{uuid.uuid4().hex}.wav"
        encoded_text = base64.b64encode(text.strip().encode("utf-8")).decode("ascii")
        escaped_path = str(output_path).replace("'", "''")
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$speaker.SetOutputToWaveFile('{escaped_path}'); "
            f"$text = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{encoded_text}')); "
            "$speaker.Speak($text); $speaker.Dispose()"
        )
        process = await asyncio.create_subprocess_exec(
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            script,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0 or not output_path.is_file():
            output_path.unlink(missing_ok=True)
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Windows 語音合成失敗：{detail or '沒有產生 WAV 檔。'}")
        return output_path
