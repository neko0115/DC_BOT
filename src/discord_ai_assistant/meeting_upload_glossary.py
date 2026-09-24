from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import discord
from discord import app_commands
from discord.ext import commands

from discord_ai_assistant.ai.persona import BASE_PERSONA_INSTRUCTION
from discord_ai_assistant.app_knowledge import AppKnowledgeSnapshot, AppKnowledgeStore
from discord_ai_assistant.meeting_glossary import (
    MEETING_GLOSSARY_REVISION,
    apply_glossary_corrections,
    enrich_initial_prompt,
    glossary_for_title,
    glossary_for_transcript,
)
from discord_ai_assistant.meeting_report_exports import build_meeting_report_exports
from discord_ai_assistant.meeting_upload import (
    LocalMeetingTranscriber,
    MeetingReportCommands as BaseMeetingReportCommands,
    MeetingTranscription,
    _safe_filename,
    build_meeting_report_prompt,
    format_meeting_timestamp,
    format_timestamped_transcript,
    is_supported_meeting_audio,
)


LOGGER = logging.getLogger(__name__)
_MEETING_TITLE: ContextVar[str] = ContextVar("meeting_report_title", default="")
_MEETING_KNOWLEDGE: ContextVar[AppKnowledgeSnapshot | None] = ContextVar(
    "meeting_report_knowledge", default=None
)


def _snapshot_initial_prompt(base_prompt: str | None, snapshot: AppKnowledgeSnapshot | None) -> str | None:
    base = (base_prompt or "").strip()
    if snapshot is None:
        return base or None
    terms = snapshot.whisper_terms(limit=80)
    if not terms:
        return base or None
    domain = (
        f"目前領域是 {snapshot.profile.display_name}。"
        f"可能出現的專有名詞、玩家名與縮寫：{'、'.join(terms)}。"
    )
    return f"{base} {domain}".strip()


def _apply_snapshot_corrections(
    text: str, snapshot: AppKnowledgeSnapshot | None
) -> tuple[str, list[dict[str, object]]]:
    if snapshot is None:
        return text, []
    corrected = text
    applied: list[dict[str, object]] = []
    mappings: list[tuple[str, str]] = []
    for item in snapshot.terms:
        mappings.extend((alias, item.term) for alias in item.aliases if alias and alias != item.term)
    for source, target in sorted(mappings, key=lambda pair: len(pair[0]), reverse=True):
        count = corrected.count(source)
        if count <= 0:
            continue
        corrected = corrected.replace(source, target)
        applied.append({"from": source, "to": target, "count": count})
    return corrected, applied


def _discord_chunks(text: str, limit: int = 1900) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    paragraphs = cleaned.split("\n\n")
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        while len(paragraph) > limit:
            split_at = paragraph.rfind("\n", 0, limit)
            if split_at < limit // 2:
                split_at = limit
            chunks.append(paragraph[:split_at].rstrip())
            paragraph = paragraph[split_at:].lstrip()
        current = paragraph
    if current:
        chunks.append(current)
    return chunks


class GlossaryAwareLocalMeetingTranscriber(LocalMeetingTranscriber):
    @staticmethod
    def _enrich_audio_config(audio: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(audio)
        snapshot = _MEETING_KNOWLEDGE.get()
        if snapshot is not None:
            enriched["whisper_initial_prompt"] = _snapshot_initial_prompt(
                str(enriched.get("whisper_initial_prompt", "")), snapshot
            )
            return enriched
        glossary = glossary_for_title(_MEETING_TITLE.get())
        if glossary is not None:
            enriched["whisper_initial_prompt"] = enrich_initial_prompt(
                str(enriched.get("whisper_initial_prompt", "")), glossary
            )
        return enriched

    def load_audio_config(self) -> dict[str, Any]:
        return self._enrich_audio_config(super().load_audio_config())

    def config_fingerprint(self, audio: dict[str, Any]) -> str:
        base = super().config_fingerprint(audio)
        snapshot = _MEETING_KNOWLEDGE.get()
        if snapshot is not None:
            whisper_payload = {
                "profile": snapshot.profile.key,
                "terms": [
                    {"term": item.term, "aliases": list(item.aliases)} for item in snapshot.terms
                ],
            }
            profile = json.dumps(
                whisper_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
        else:
            glossary = glossary_for_title(_MEETING_TITLE.get())
            profile = glossary.name if glossary is not None else "auto"
        raw = f"{base}|{MEETING_GLOSSARY_REVISION}|{profile}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def transcribe(self, audio_path: Path, audio: dict[str, Any] | None = None) -> MeetingTranscription:
        settings = self._enrich_audio_config(audio) if audio is not None else self.load_audio_config()
        result = super().transcribe(audio_path, settings)
        snapshot = _MEETING_KNOWLEDGE.get()
        corrected_segments: list[dict[str, object]] = []
        totals: list[dict[str, object]] = []
        profile_name: str | None = None

        if snapshot is not None:
            profile_name = snapshot.profile.key
            aggregate: dict[tuple[str, str], int] = {}
            for item in result.segments:
                copied = dict(item)
                corrected, applied = _apply_snapshot_corrections(
                    str(copied.get("text", "")), snapshot
                )
                copied["text"] = corrected
                corrected_segments.append(copied)
                for correction in applied:
                    key = (str(correction["from"]), str(correction["to"]))
                    aggregate[key] = aggregate.get(key, 0) + int(correction["count"])
            totals = [
                {"from": source, "to": target, "count": count}
                for (source, target), count in sorted(aggregate.items())
            ]
        else:
            glossary = glossary_for_title(_MEETING_TITLE.get()) or glossary_for_transcript(result.text)
            if glossary is None:
                return result
            profile_name = glossary.name
            aggregate = {}
            for item in result.segments:
                copied = dict(item)
                corrected, applied = apply_glossary_corrections(
                    str(copied.get("text", "")), glossary
                )
                copied["text"] = corrected
                corrected_segments.append(copied)
                for correction in applied:
                    key = (correction["from"], correction["to"])
                    aggregate[key] = aggregate.get(key, 0) + int(correction["count"])
            totals = [
                {"from": source, "to": target, "count": count}
                for (source, target), count in sorted(aggregate.items())
            ]

        corrected_text = format_timestamped_transcript(corrected_segments)
        audit = {
            "revision": MEETING_GLOSSARY_REVISION,
            "profile": profile_name,
            "source": "app_knowledge_database" if snapshot is not None else "built_in_fallback",
            "corrections": totals,
        }
        try:
            (audio_path.parent / "glossary.json").write_text(
                json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            pass

        return MeetingTranscription(
            text=corrected_text,
            segments=corrected_segments,
            language=result.language,
            duration_seconds=result.duration_seconds,
            confidence=result.confidence,
            model=result.model,
            config_fingerprint=result.config_fingerprint,
        )


class MeetingReportReviewView(discord.ui.View):
    def __init__(
        self,
        cog: "MeetingReportCommands",
        *,
        report_path: Path,
        review_path: Path,
        title: str,
        requester_id: int,
    ) -> None:
        super().__init__(timeout=3600)
        self.cog = cog
        self.report_path = report_path
        self.review_path = review_path
        self.title = title
        self.requester_id = requester_id
        self._lock = asyncio.Lock()

    def _disable(self) -> None:
        for item in self.children:
            if isinstance(item, discord.ui.Button):
                item.disabled = True

    @discord.ui.button(label="核准並發布", style=discord.ButtonStyle.success, emoji="✅")
    async def approve(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        del button
        if not interaction.guild or not isinstance(interaction.user, discord.Member):
            await interaction.response.send_message("只能在原伺服器核准週會報。", ephemeral=True)
            return
        if not self.cog._is_dj_member(interaction.user):
            await interaction.response.send_message("只有 DJ 或管理員可以核准發布。", ephemeral=True)
            return
        async with self._lock:
            await interaction.response.defer(thinking=True, ephemeral=True)
            target = await self.cog._resolve_summary_channel(interaction.guild)
            if target is None:
                await interaction.followup.send(
                    "尚未設定可用的週會報輸出頻道。請先使用 `/meeting output` 設定後再核准。",
                    ephemeral=True,
                )
                return
            try:
                await self.cog._publish_report(target, self.report_path, self.title)
                self.cog._update_review(
                    self.review_path,
                    status="published",
                    approved_by=interaction.user.id,
                    approved_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                    target_channel_id=target.id,
                )
            except Exception as error:
                LOGGER.exception("Reviewed meeting report publication failed")
                await interaction.followup.send(
                    f"核准成功但發布失敗：{type(error).__name__}: {error}", ephemeral=True
                )
                return
            self._disable()
            if interaction.message is not None:
                await interaction.message.edit(
                    content=(
                        f"✅ 週會報已由 {interaction.user.display_name} 核准並發布到 {target.mention}。\n"
                        "草稿與逐字稿仍保留在本機供回查。"
                    ),
                    view=self,
                )
            await interaction.followup.send(f"已發布到 {target.mention}。", ephemeral=True)

    @discord.ui.button(label="退回／不發布", style=discord.ButtonStyle.secondary, emoji="↩️")
    async def reject(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        del button
        if not interaction.guild or not isinstance(interaction.user, discord.Member):
            await interaction.response.send_message("只能在原伺服器操作。", ephemeral=True)
            return
        if not self.cog._is_dj_member(interaction.user):
            await interaction.response.send_message("只有 DJ 或管理員可以退回草稿。", ephemeral=True)
            return
        self.cog._update_review(
            self.review_path,
            status="rejected",
            approved_by=interaction.user.id,
            approved_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        )
        self._disable()
        await interaction.response.edit_message(
            content=f"↩️ 週會報草稿已由 {interaction.user.display_name} 退回，沒有發布到設定頻道。",
            view=self,
        )


class MeetingReportCommands(BaseMeetingReportCommands):
    knowledge = app_commands.Group(
        name="knowledge",
        description="管理墨雪針對不同遊戲、App 或情境使用的專有名詞資料庫",
    )

    def __init__(self, bot: commands.Bot, settings: Any, ai: Any) -> None:
        super().__init__(bot, settings, ai)
        self.transcriber = GlossaryAwareLocalMeetingTranscriber(settings.project_root)
        database = getattr(bot, "database", None)
        self.knowledge_store = AppKnowledgeStore(database) if database is not None else None

    def _local_zone(self):
        try:
            return ZoneInfo(str(self.settings.persona_timezone))
        except (ZoneInfoNotFoundError, AttributeError, TypeError, ValueError):
            return datetime.now().astimezone().tzinfo

    def _knowledge_for(self, guild_id: int, title: str) -> AppKnowledgeSnapshot | None:
        if self.knowledge_store is None:
            return None
        self.knowledge_store.ensure_lifeafter_seed(guild_id)
        if self.knowledge_store.default_profile(guild_id) is None:
            self.knowledge_store.set_default_profile(guild_id, "lifeafter")
        return self.knowledge_store.snapshot(guild_id, title)

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

        snapshot = self._knowledge_for(interaction.guild.id, title)
        title_token = _MEETING_TITLE.set(title)
        knowledge_token = _MEETING_KNOWLEDGE.set(snapshot)
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
            await interaction.edit_original_response(
                content="本地逐字稿已完成，墨雪正在依逐字稿與領域詞庫整理週會報草稿。"
            )
            report_body = await self._generate_report(title, transcription.text)
            generated_at = interaction.created_at.astimezone(self._local_zone())
            duration = format_meeting_timestamp(transcription.duration_seconds)
            profile_name = snapshot.profile.display_name if snapshot is not None else "未指定"
            report = (
                f"> 📅 **週會報日期：** {generated_at:%Y-%m-%d}\n"
                f"> 🕒 **產生時間：** {generated_at:%H:%M} ({generated_at.tzname() or 'local'})\n"
                f"> 🎧 **錄音長度：** {duration}\n"
                f"> 📚 **Knowledge Profile：** {profile_name}\n\n"
                f"{report_body.strip()}\n"
            )
            stamp = generated_at.strftime("%Y%m%d_%H%M%S")
            report_path = job_root / f"report_{stamp}.md"
            report_path.write_text(report, encoding="utf-8")
            review_path = job_root / f"review_{stamp}.json"
            target_id = self._configured_summary_channel_id()
            review = {
                "status": "draft",
                "title": title,
                "audio_sha256": digest,
                "created_at": generated_at.isoformat(timespec="seconds"),
                "guild_id": interaction.guild.id,
                "source_channel_id": interaction.channel_id,
                "requested_by": interaction.user.id,
                "configured_target_channel_id": target_id,
                "knowledge_profile": snapshot.profile.key if snapshot is not None else None,
                "report_file": report_path.name,
            }
            review_path.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")

            cache_text = "（使用既有逐字稿快取）" if cached else ""
            target_text = (
                f"設定發布頻道：<#{target_id}>。"
                if target_id
                else "目前尚未設定發布頻道；請使用 `/meeting output`。"
            )
            await interaction.edit_original_response(
                content=(
                    f"週會報草稿完成。錄音長度 `{duration}`，Whisper `{transcription.model}`，"
                    f"平均辨識信心 `{transcription.confidence:.2f}` {cache_text}\n{target_text}\n"
                    "請先人工審核；只有按下「核准並發布」後才會送到指定頻道。"
                ).strip()
            )
            safe_title = _safe_filename(title, "meeting")
            preview = report[:1300]
            if len(report) > len(preview):
                preview = preview.rstrip() + "\n\n…完整內容請查看週會報附件。"
            view = MeetingReportReviewView(
                self,
                report_path=report_path,
                review_path=review_path,
                title=title,
                requester_id=interaction.user.id,
            )
            await interaction.followup.send(
                content=preview,
                files=[
                    discord.File(report_path, filename=f"{safe_title}_週會報_待審核.md"),
                    discord.File(transcript_txt, filename=f"{safe_title}_逐字稿.txt"),
                    discord.File(transcript_json, filename=f"{safe_title}_逐字稿.json"),
                ],
                view=view,
            )
        except Exception as error:
            LOGGER.exception("Meeting report generation failed")
            temp_path.unlink(missing_ok=True)
            try:
                await interaction.edit_original_response(
                    content=f"週會錄音處理失敗：{type(error).__name__}: {error}"
                )
            except discord.HTTPException:
                LOGGER.exception("Could not report meeting generation failure")
        finally:
            _MEETING_KNOWLEDGE.reset(knowledge_token)
            _MEETING_TITLE.reset(title_token)

    async def _generate_report(self, title: str, transcript: str) -> str:
        prompt = build_meeting_report_prompt(title, transcript)
        snapshot = _MEETING_KNOWLEDGE.get()
        if snapshot is not None:
            prompt += (
                "\n\n---\n領域專有名詞資料庫：\n"
                f"{snapshot.explanation_block()}\n"
                "重要：上述詞庫只可協助理解詞義與修正名詞，不可把詞庫內的解釋當成本次會議發生的事、決議或進度。\n"
            )
        text = await self.ai.social_reply(
            prompt,
            persona_instruction=(
                f"{BASE_PERSONA_INSTRUCTION}\n"
                "你現在執行會議紀錄整理，不可呼叫工具、搜尋外部資訊或使用逐字稿與明示詞庫以外的事實。"
            ),
        )
        text = str(text).strip()
        if not text:
            raise RuntimeError("Gemini 沒有回傳週會報內容。")
        return text

    def _configured_summary_channel_id(self) -> int | None:
        runtime = self.settings.project_root / "data" / "game-meeting-recorder" / "config.json"
        default = self.settings.project_root / "tools" / "game_meeting_recorder" / "config.json"
        path = runtime if runtime.exists() else default
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            value = payload.get("output", {}).get("summary_channel_id")
            return int(value) if value not in (None, "") else None
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    async def _resolve_summary_channel(
        self, guild: discord.Guild
    ) -> discord.TextChannel | discord.Thread | None:
        channel_id = self._configured_summary_channel_id()
        if channel_id is None:
            return None
        channel = guild.get_channel(channel_id)
        if channel is None:
            try:
                channel = await self.bot.fetch_channel(channel_id)
            except (discord.HTTPException, discord.NotFound, discord.Forbidden):
                return None
        channel_guild = getattr(channel, "guild", None)
        if channel_guild is None or getattr(channel_guild, "id", None) != guild.id:
            return None
        return channel if isinstance(channel, (discord.TextChannel, discord.Thread)) else None

    async def _publish_report(
        self, channel: discord.TextChannel | discord.Thread, report_path: Path, title: str
    ) -> None:
        report = report_path.read_text(encoding="utf-8")
        chunks = _discord_chunks(report)
        safe_title = _safe_filename(title, "meeting")
        if not chunks:
            raise RuntimeError("週會報內容為空。")

        exports = await asyncio.to_thread(build_meeting_report_exports, report_path, title=title)
        await channel.send(
            content=chunks[0],
            files=[
                discord.File(exports.markdown, filename=f"{safe_title}_週會報.md"),
                discord.File(exports.docx, filename=f"{safe_title}_週會報.docx"),
                discord.File(exports.pptx, filename=f"{safe_title}_週會報.pptx"),
            ],
            allowed_mentions=discord.AllowedMentions.none(),
        )
        for chunk in chunks[1:]:
            await channel.send(content=chunk, allowed_mentions=discord.AllowedMentions.none())

    @staticmethod
    def _update_review(path: Path, **updates: object) -> None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except (OSError, json.JSONDecodeError):
            payload = {}
        payload.update(updates)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    async def meeting_report_context(
        self, interaction: discord.Interaction, message: discord.Message
    ) -> None:
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
        message_title = " ".join(message.clean_content.split()).strip()
        title = message_title[:120] if message_title else (Path(audio.filename).stem or "遊戲週會")
        await self._run_report(interaction, audio, title)

    async def _knowledge_guard(self, interaction: discord.Interaction) -> bool:
        if not interaction.guild or not isinstance(interaction.user, discord.Member):
            await interaction.response.send_message(
                "Knowledge 指令只能在伺服器內使用。", ephemeral=True
            )
            return False
        if not self._is_dj_member(interaction.user):
            await interaction.response.send_message(
                "只有 DJ 或管理員可以修改專有名詞資料庫。", ephemeral=True
            )
            return False
        if self.knowledge_store is None:
            await interaction.response.send_message(
                "目前無法取得墨雪的 SQLite database。", ephemeral=True
            )
            return False
        self.knowledge_store.ensure_lifeafter_seed(interaction.guild.id)
        return True

    @knowledge.command(
        name="profiles", description="列出這個伺服器可用的遊戲／App／情境 Knowledge Profiles"
    )
    async def knowledge_profiles(self, interaction: discord.Interaction) -> None:
        if not await self._knowledge_guard(interaction):
            return
        assert interaction.guild is not None and self.knowledge_store is not None
        profiles = self.knowledge_store.list_profiles(interaction.guild.id)
        default = self.knowledge_store.default_profile(interaction.guild.id)
        lines = [
            f"- `{item.key}` — **{item.display_name}**"
            f"{'（目前預設）' if default and default.key == item.key else ''}\n"
            f"  {item.description or '未提供說明'}"
            for item in profiles
        ]
        await interaction.response.send_message(
            "\n".join(lines)[:1900] or "目前沒有 Knowledge Profile。", ephemeral=True
        )

    @knowledge.command(
        name="profile_set", description="新增或更新一個遊戲／App／情境 Knowledge Profile"
    )
    @app_commands.describe(
        key="例如 lifeafter、dcs、lab-meeting",
        name="顯示名稱",
        description="這個領域大概在做什麼",
        hints="自動判斷用關鍵字，可用逗號分隔",
    )
    async def knowledge_profile_set(
        self,
        interaction: discord.Interaction,
        key: str,
        name: str,
        description: str,
        hints: str | None = None,
    ) -> None:
        if not await self._knowledge_guard(interaction):
            return
        assert interaction.guild is not None and self.knowledge_store is not None
        try:
            profile = self.knowledge_store.upsert_profile(
                interaction.guild.id, key, name, description, hints
            )
        except ValueError as error:
            await interaction.response.send_message(str(error), ephemeral=True)
            return
        await interaction.response.send_message(
            f"已儲存 Knowledge Profile `{profile.key}`：**{profile.display_name}**。",
            ephemeral=True,
        )

    @knowledge.command(
        name="use", description="設定無法從標題判斷時要使用的預設 Knowledge Profile"
    )
    @app_commands.describe(profile="Profile key，例如 lifeafter")
    async def knowledge_use(self, interaction: discord.Interaction, profile: str) -> None:
        if not await self._knowledge_guard(interaction):
            return
        assert interaction.guild is not None and self.knowledge_store is not None
        try:
            selected = self.knowledge_store.set_default_profile(interaction.guild.id, profile)
        except ValueError as error:
            await interaction.response.send_message(str(error), ephemeral=True)
            return
        await interaction.response.send_message(
            f"預設 Knowledge Profile 已設為 `{selected.key}` — **{selected.display_name}**。",
            ephemeral=True,
        )

    @knowledge.command(
        name="term_set", description="新增或更新專有名詞、簡短解釋與常見別名／誤辨識"
    )
    @app_commands.describe(
        profile="Profile key，例如 lifeafter",
        term="標準專有名詞",
        explanation="簡短解釋，讓墨雪理解它在這個領域大概代表什麼",
        aliases="別名或常見 Whisper 誤辨識，可用逗號分隔；不確定就留空",
    )
    async def knowledge_term_set(
        self,
        interaction: discord.Interaction,
        profile: str,
        term: str,
        explanation: str,
        aliases: str | None = None,
    ) -> None:
        if not await self._knowledge_guard(interaction):
            return
        assert interaction.guild is not None and self.knowledge_store is not None
        try:
            saved = self.knowledge_store.upsert_term(
                interaction.guild.id, profile, term, explanation, aliases
            )
        except ValueError as error:
            await interaction.response.send_message(str(error), ephemeral=True)
            return
        alias_text = f"；別名：{', '.join(saved.aliases)}" if saved.aliases else ""
        await interaction.response.send_message(
            f"已儲存 **{saved.term}**：{saved.explanation}{alias_text}", ephemeral=True
        )

    @knowledge.command(name="terms", description="列出某個 Knowledge Profile 的專有名詞")
    @app_commands.describe(profile="Profile key，例如 lifeafter")
    async def knowledge_terms(self, interaction: discord.Interaction, profile: str) -> None:
        if not await self._knowledge_guard(interaction):
            return
        assert interaction.guild is not None and self.knowledge_store is not None
        try:
            terms = self.knowledge_store.list_terms(interaction.guild.id, profile)
        except ValueError as error:
            await interaction.response.send_message(str(error), ephemeral=True)
            return
        lines: list[str] = []
        for item in terms:
            aliases = f"｜別名：{', '.join(item.aliases)}" if item.aliases else ""
            lines.append(f"- **{item.term}**：{item.explanation}{aliases}")
        await interaction.response.send_message(
            "\n".join(lines)[:1900] or "這個 Profile 還沒有詞條。", ephemeral=True
        )

    @knowledge.command(name="term_remove", description="刪除一個 Knowledge Profile 的專有名詞")
    async def knowledge_term_remove(
        self, interaction: discord.Interaction, profile: str, term: str
    ) -> None:
        if not await self._knowledge_guard(interaction):
            return
        assert interaction.guild is not None and self.knowledge_store is not None
        try:
            removed = self.knowledge_store.remove_term(interaction.guild.id, profile, term)
        except ValueError as error:
            await interaction.response.send_message(str(error), ephemeral=True)
            return
        await interaction.response.send_message(
            f"{'已刪除' if removed else '找不到'} **{term}**。", ephemeral=True
        )
