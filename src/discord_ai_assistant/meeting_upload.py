from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import discord
from discord import app_commands
from discord.ext import commands

from discord_ai_assistant.ai.persona import BASE_PERSONA_INSTRUCTION
from discord_ai_assistant.config import Settings

LOGGER = logging.getLogger(__name__)

SUPPORTED_MEETING_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".webm", ".mp4"}
MAX_MEETING_AUDIO_BYTES = 250 * 1024 * 1024
LOW_CONFIDENCE_THRESHOLD = 0.55


@dataclass(slots=True)
class MeetingTranscription:
    text: str
    segments: list[dict[str, object]]
    language: str | None
    duration_seconds: float
    confidence: float
    model: str
    config_fingerprint: str


def is_supported_meeting_audio(filename: str, size: int) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_MEETING_AUDIO_EXTENSIONS and 0 < size <= MAX_MEETING_AUDIO_BYTES


def format_meeting_timestamp(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_timestamped_transcript(segments: list[dict[str, object]]) -> str:
    lines: list[str] = []
    for item in segments:
        start = format_meeting_timestamp(float(item.get("start", 0.0)))
        end = format_meeting_timestamp(float(item.get("end", 0.0)))
        text = str(item.get("text", "")).strip()
        confidence = float(item.get("confidence", 0.0))
        marker = f" [低信心 {confidence:.2f}]" if confidence < LOW_CONFIDENCE_THRESHOLD else ""
        if text:
            lines.append(f"[{start}-{end}]{marker} {text}")
    return "\n".join(lines)


def build_meeting_report_prompt(title: str, transcript: str) -> str:
    return (
        "請把下面的會議自動逐字稿整理成繁體中文 Markdown 週會報。\n"
        "這是一項忠實整理工作，不是自由補完。必須遵守：\n"
        "1. 只能使用逐字稿中實際出現的資訊，不得引用外部知識、記憶或自行推測。\n"
        "2. 明確區分『已完成／已測試』、『正在做』、『提案／討論』、『決議』與『尚未驗證』。\n"
        "3. 沒有明確說出負責人、日期、數字或結論時，寫『未指定／待確認』，不得自行補。\n"
        "4. 標有『低信心』的句子不可單獨當成正式決議；必要時放進『待確認』。\n"
        "5. 重要成果、決議與待辦盡量附上逐字稿時間碼，方便回查原音檔。\n"
        "6. 不要宣稱轉錄或實測已成功，除非逐字稿明確如此說。\n"
        "7. 報告結構固定為：會議摘要、本週進度、討論事項、問題與風險、會議決議、下週待辦、待確認事項。\n"
        f"\n會議名稱：{title.strip() or '遊戲週會'}\n\n"
        "以下為自動逐字稿：\n---\n"
        f"{transcript}\n---\n"
    )


def _safe_filename(value: str, fallback: str = "meeting") -> str:
    stem = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", value).strip("._-")
    return stem[:80] or fallback


class LocalMeetingTranscriber:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.data_root = project_root / "data" / "game-meeting-recorder"
        self.models_root = self.data_root / "whisper-models"
        self.models_root.mkdir(parents=True, exist_ok=True)
        self._models: dict[str, Any] = {}

    def load_audio_config(self) -> dict[str, Any]:
        runtime = self.data_root / "config.json"
        default = self.project_root / "tools" / "game_meeting_recorder" / "config.json"
        path = runtime if runtime.exists() else default
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"無法讀取會議 Whisper 設定：{error}") from error
        audio = payload.get("audio")
        if not isinstance(audio, dict):
            raise RuntimeError("會議設定缺少 audio 區段。")
        return audio

    @staticmethod
    def config_fingerprint(audio: dict[str, Any]) -> str:
        relevant = {
            key: audio.get(key)
            for key in ("whisper_model", "whisper_language", "whisper_beam_size", "whisper_initial_prompt")
        }
        raw = json.dumps(relevant, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _get_model(self, model_name: str) -> Any:
        model = self._models.get(model_name)
        if model is not None:
            return model
        try:
            from faster_whisper import WhisperModel
        except ImportError as error:
            raise RuntimeError("缺少 faster-whisper；請先執行 `python -m pip install -e .`。") from error
        model = WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8",
            download_root=str(self.models_root),
        )
        self._models[model_name] = model
        return model

    def transcribe(self, audio_path: Path, audio: dict[str, Any] | None = None) -> MeetingTranscription:
        settings = dict(audio or self.load_audio_config())
        model_name = str(settings.get("whisper_model") or "small").strip() or "small"
        language_raw = settings.get("whisper_language")
        language = language_raw.strip() if isinstance(language_raw, str) and language_raw.strip() else None
        beam_size = int(settings.get("whisper_beam_size", 5))
        initial_prompt = str(settings.get("whisper_initial_prompt", "")).strip() or None
        model = self._get_model(model_name)
        raw_segments, info = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=beam_size,
            initial_prompt=initial_prompt,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        segments: list[dict[str, object]] = []
        confidences: list[float] = []
        for segment in raw_segments:
            text = str(getattr(segment, "text", "")).strip()
            if not text:
                continue
            avg_logprob = float(getattr(segment, "avg_logprob", -5.0))
            no_speech = float(getattr(segment, "no_speech_prob", 0.0))
            confidence = max(0.0, min(1.0, math.exp(min(0.0, avg_logprob)) * (1.0 - no_speech)))
            confidences.append(confidence)
            segments.append(
                {
                    "start": round(float(getattr(segment, "start", 0.0)), 3),
                    "end": round(float(getattr(segment, "end", 0.0)), 3),
                    "text": text,
                    "confidence": round(confidence, 4),
                }
            )
        transcript = format_timestamped_transcript(segments)
        duration = max((float(item["end"]) for item in segments), default=0.0)
        return MeetingTranscription(
            text=transcript,
            segments=segments,
            language=getattr(info, "language", language),
            duration_seconds=duration,
            confidence=sum(confidences) / len(confidences) if confidences else 0.0,
            model=model_name,
            config_fingerprint=self.config_fingerprint(settings),
        )


class MeetingReportCommands(commands.Cog):
    def __init__(self, bot: commands.Bot, settings: Settings, ai: Any) -> None:
        self.bot = bot
        self.settings = settings
        self.ai = ai
        self.transcriber = LocalMeetingTranscriber(settings.project_root)
        self.import_root = settings.project_root / "data" / "game-meeting-recorder" / "imports"
        self.import_root.mkdir(parents=True, exist_ok=True)
        self._message_menu = app_commands.ContextMenu(
            name="墨雪：整理錄音成週會報",
            callback=self.meeting_report_context,
        )
        self.bot.tree.add_command(self._message_menu)

    def cog_unload(self) -> None:
        self.bot.tree.remove_command(self._message_menu.name, type=self._message_menu.type)

    @app_commands.command(name="meeting_report", description="上傳 MP3/WAV 等錄音，以本地 Whisper 整理成週會報")
    @app_commands.describe(audio="會議錄音檔", title="週會名稱，例如：明日之後週會")
    async def meeting_report(
        self,
        interaction: discord.Interaction,
        audio: discord.Attachment,
        title: str = "遊戲週會",
    ) -> None:
        await self._run_report(interaction, audio, title)

    async def meeting_report_context(self, interaction: discord.Interaction, message: discord.Message) -> None:
        audio = next(
            (
                item
                for item in message.attachments
                if is_supported_meeting_audio(item.filename, item.size)
            ),
            None,
        )
        if audio is None:
            await interaction.response.send_message(
                "這則訊息沒有可處理的 MP3/WAV/M4A/FLAC/OGG/OPUS/WEBM/MP4 錄音。",
                ephemeral=True,
            )
            return
        title = Path(audio.filename).stem or "遊戲週會"
        await self._run_report(interaction, audio, title)

    async def _run_report(
        self, interaction: discord.Interaction, attachment: discord.Attachment, title: str
    ) -> None:
        if not interaction.guild or not isinstance(interaction.user, discord.Member):
            await interaction.response.send_message("此功能僅限伺服器頻道。", ephemeral=True)
            return
        if not self._is_dj_member(interaction.user):
            await interaction.response.send_message("會議錄音轉錄僅限 DJ 或管理員使用。", ephemeral=True)
            return
        if not is_supported_meeting_audio(attachment.filename, attachment.size):
            await interaction.response.send_message(
                "錄音格式不支援或檔案超過 250 MiB。支援 MP3/WAV/M4A/FLAC/OGG/OPUS/WEBM/MP4。",
                ephemeral=True,
            )
            return

        await interaction.response.defer(thinking=True)
        incoming = self.import_root / ".incoming"
        incoming.mkdir(parents=True, exist_ok=True)
        suffix = Path(attachment.filename).suffix.lower()
        temp_path = incoming / f"{interaction.id}{suffix}"
        try:
            await attachment.save(temp_path)
            digest = await asyncio.to_thread(self._sha256_file, temp_path)
            job_root = self.import_root / digest
            job_root.mkdir(parents=True, exist_ok=True)
            source_path = job_root / f"original{suffix}"
            if source_path.exists():
                temp_path.unlink(missing_ok=True)
            else:
                temp_path.replace(source_path)

            await interaction.edit_original_response(content="錄音已收到，正在使用本地 Whisper 取得逐字稿。")
            transcription, cached = await self._load_or_transcribe(source_path, job_root)
            if not transcription.text.strip():
                await interaction.edit_original_response(content="Whisper 沒有辨識到可用語音內容，因此沒有產生週會報。")
                return

            transcript_txt = job_root / "transcript.txt"
            transcript_json = job_root / "transcript.json"
            await interaction.edit_original_response(content="本地逐字稿已完成，墨雪正在依逐字稿整理週會報。")
            report = await self._generate_report(title, transcription.text)
            stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
            report_path = job_root / f"report_{stamp}.md"
            report_path.write_text(report.rstrip() + "\n", encoding="utf-8")

            duration = format_meeting_timestamp(transcription.duration_seconds)
            cache_text = "（使用既有逐字稿快取）" if cached else ""
            await interaction.edit_original_response(
                content=(
                    f"週會報完成。錄音長度約 `{duration}`，Whisper 模型 `{transcription.model}`，"
                    f"平均辨識信心 `{transcription.confidence:.2f}` {cache_text}"
                ).strip()
            )
            safe_title = _safe_filename(title, "meeting")
            await interaction.followup.send(
                content="附上週會報與可回查時間碼的逐字稿；低信心片段已在逐字稿中標示。",
                files=[
                    discord.File(report_path, filename=f"{safe_title}_週會報.md"),
                    discord.File(transcript_txt, filename=f"{safe_title}_逐字稿.txt"),
                    discord.File(transcript_json, filename=f"{safe_title}_逐字稿.json"),
                ],
            )
        except Exception as error:
            LOGGER.exception("Meeting report generation failed")
            temp_path.unlink(missing_ok=True)
            try:
                await interaction.edit_original_response(content=f"週會錄音處理失敗：{type(error).__name__}: {error}")
            except discord.HTTPException:
                LOGGER.exception("Could not report meeting generation failure")

    async def _load_or_transcribe(
        self, source_path: Path, job_root: Path
    ) -> tuple[MeetingTranscription, bool]:
        audio_config = self.transcriber.load_audio_config()
        fingerprint = self.transcriber.config_fingerprint(audio_config)
        json_path = job_root / "transcript.json"
        txt_path = job_root / "transcript.txt"
        if json_path.exists():
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
                if payload.get("config_fingerprint") == fingerprint:
                    result = MeetingTranscription(
                        text=str(payload.get("text", "")),
                        segments=list(payload.get("segments", [])),
                        language=payload.get("language"),
                        duration_seconds=float(payload.get("duration_seconds", 0.0)),
                        confidence=float(payload.get("confidence", 0.0)),
                        model=str(payload.get("model", audio_config.get("whisper_model", "small"))),
                        config_fingerprint=fingerprint,
                    )
                    if result.text.strip():
                        if not txt_path.exists():
                            txt_path.write_text(result.text.rstrip() + "\n", encoding="utf-8")
                        return result, True
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                LOGGER.warning("Ignoring invalid cached meeting transcript at %s", json_path, exc_info=True)

        result = await asyncio.to_thread(self.transcriber.transcribe, source_path, audio_config)
        payload = {
            "text": result.text,
            "segments": result.segments,
            "language": result.language,
            "duration_seconds": round(result.duration_seconds, 3),
            "confidence": round(result.confidence, 4),
            "model": result.model,
            "config_fingerprint": result.config_fingerprint,
            "source_file": source_path.name,
        }
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        txt_path.write_text(result.text.rstrip() + "\n", encoding="utf-8")
        return result, False

    async def _generate_report(self, title: str, transcript: str) -> str:
        prompt = build_meeting_report_prompt(title, transcript)
        text = await self.ai.social_reply(
            prompt,
            persona_instruction=(
                f"{BASE_PERSONA_INSTRUCTION}\n"
                "你現在執行會議紀錄整理，不可呼叫工具、搜尋外部資訊或使用逐字稿以外的事實。"
            ),
        )
        text = str(text).strip()
        if not text:
            raise RuntimeError("Gemini 沒有回傳週會報內容。")
        return text

    def _is_dj_member(self, member: discord.Member) -> bool:
        if member.guild_permissions.manage_guild:
            return True
        if self.settings.dj_role_id is not None:
            return any(role.id == self.settings.dj_role_id for role in member.roles)
        return any(role.name == self.settings.dj_role_name for role in member.roles)

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
