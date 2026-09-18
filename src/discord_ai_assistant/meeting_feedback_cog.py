from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import discord
from discord import app_commands
from discord.ext import commands

from discord_ai_assistant.ai.persona import BASE_PERSONA_INSTRUCTION
from discord_ai_assistant.app_knowledge import AppKnowledgeSnapshot
from discord_ai_assistant.meeting_feedback import (
    FeedbackCandidate,
    MeetingFeedbackStore,
    merge_candidates,
    parse_ai_candidates,
    parse_explicit_corrections,
)
from discord_ai_assistant.meeting_upload import (
    MeetingTranscription,
    _safe_filename,
    build_meeting_report_prompt,
    format_meeting_timestamp,
    format_timestamped_transcript,
    is_supported_meeting_audio,
)
from discord_ai_assistant.meeting_upload_glossary import (
    MeetingReportCommands as KnowledgeMeetingReportCommands,
    _MEETING_KNOWLEDGE,
    _MEETING_TITLE,
    _apply_snapshot_corrections,
)


LOGGER = logging.getLogger(__name__)


def _candidate_label(candidate: FeedbackCandidate) -> str:
    if candidate.kind == "asr_alias":
        alias = candidate.source_text or (candidate.aliases[0] if candidate.aliases else "?")
        suffix = f" — {candidate.explanation}" if candidate.explanation else ""
        return f"[ASR] `{alias}` → `{candidate.canonical_term}`{suffix}"
    if candidate.kind == "domain_term":
        return f"[詞義] **{candidate.canonical_term}** — {candidate.explanation}"
    return f"[報告偏好] {candidate.explanation}"


class MeetingRevisionModal(discord.ui.Modal):
    correction = discord.ui.TextInput(
        label="修正內容／建議",
        style=discord.TextStyle.paragraph,
        placeholder=(
            "可直接貼修正後的段落；若是明確 ASR 錯詞，建議寫成：\n"
            "卡西 3 -> 海域 3 | 海域關卡名稱\n"
            "蝴蝶無人機 -> 晶蝶無人機 | 遊戲中的無人機名稱"
        ),
        required=True,
        max_length=4000,
    )

    def __init__(
        self,
        cog: "MeetingReportCommands",
        *,
        report_path: Path,
        review_path: Path,
        title: str,
    ) -> None:
        super().__init__(title="退回並提交修正")
        self.cog = cog
        self.report_path = report_path
        self.review_path = review_path
        self.meeting_title = title

    async def on_submit(self, interaction: discord.Interaction) -> None:
        await self.cog.handle_revision_submission(
            interaction,
            report_path=self.report_path,
            review_path=self.review_path,
            title=self.meeting_title,
            correction_text=str(self.correction.value),
        )


class MeetingFeedbackDecisionView(discord.ui.View):
    def __init__(
        self,
        cog: "MeetingReportCommands",
        *,
        revision_id: int,
        review_path: Path,
        report_path: Path,
        title: str,
        correction_text: str,
        candidate_ids: list[int],
    ) -> None:
        super().__init__(timeout=1800)
        self.cog = cog
        self.revision_id = revision_id
        self.review_path = review_path
        self.report_path = report_path
        self.title = title
        self.correction_text = correction_text
        self.candidate_ids = candidate_ids
        self._lock = asyncio.Lock()
        self._handled = False
        if not candidate_ids:
            for item in self.children:
                if isinstance(item, discord.ui.Button) and item.custom_id == "meeting_feedback_apply":
                    item.disabled = True

    def _disable(self) -> None:
        self._handled = True
        for item in self.children:
            if isinstance(item, discord.ui.Button):
                item.disabled = True

    async def _guard(self, interaction: discord.Interaction) -> bool:
        if not interaction.guild or not isinstance(interaction.user, discord.Member):
            await interaction.response.send_message("只能在原伺服器處理修正。", ephemeral=True)
            return False
        if not self.cog._is_dj_member(interaction.user):
            await interaction.response.send_message("只有 DJ 或管理員可以套用週會修正。", ephemeral=True)
            return False
        if self._handled:
            await interaction.response.send_message("這份修正已經處理過了。", ephemeral=True)
            return False
        return True

    @discord.ui.button(
        label="套用學習並重整",
        style=discord.ButtonStyle.success,
        emoji="🧠",
        custom_id="meeting_feedback_apply",
    )
    async def apply_and_regenerate(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        del button
        if not await self._guard(interaction):
            return
        async with self._lock:
            self._disable()
            await interaction.response.defer(thinking=True, ephemeral=True)
            await self.cog.regenerate_from_feedback(
                interaction,
                revision_id=self.revision_id,
                review_path=self.review_path,
                report_path=self.report_path,
                title=self.title,
                correction_text=self.correction_text,
                learn=True,
                selected_candidate_ids=self.candidate_ids,
            )

    @discord.ui.button(label="只重整，不學習", style=discord.ButtonStyle.primary, emoji="🔄")
    async def regenerate_only(
        self, interaction: discord.Interaction, button: discord.ui.Button
    ) -> None:
        del button
        if not await self._guard(interaction):
            return
        async with self._lock:
            self._disable()
            await interaction.response.defer(thinking=True, ephemeral=True)
            await self.cog.regenerate_from_feedback(
                interaction,
                revision_id=self.revision_id,
                review_path=self.review_path,
                report_path=self.report_path,
                title=self.title,
                correction_text=self.correction_text,
                learn=False,
            )

    @discord.ui.button(label="取消", style=discord.ButtonStyle.secondary, emoji="✖️")
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        del button
        if not await self._guard(interaction):
            return
        self._disable()
        if self.cog.feedback_store is not None:
            self.cog.feedback_store.update_revision(self.revision_id, status="cancelled")
            self.cog.feedback_store.set_candidate_status(self.candidate_ids, "cancelled")
        await interaction.response.edit_message(content="已取消這次修正，原草稿保持不變。", view=self)


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

    async def _guard(self, interaction: discord.Interaction) -> bool:
        if not interaction.guild or not isinstance(interaction.user, discord.Member):
            await interaction.response.send_message("只能在原伺服器操作週會報。", ephemeral=True)
            return False
        if not self.cog._is_dj_member(interaction.user):
            await interaction.response.send_message("只有 DJ 或管理員可以審核週會報。", ephemeral=True)
            return False
        return True

    @discord.ui.button(label="核准並發布", style=discord.ButtonStyle.success, emoji="✅")
    async def approve(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        del button
        if not await self._guard(interaction):
            return
        assert interaction.guild is not None and isinstance(interaction.user, discord.Member)
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
                review = self.cog._read_review(self.review_path)
                approved_at = datetime.now().astimezone().isoformat(timespec="seconds")
                self.cog._update_review(
                    self.review_path,
                    status="published",
                    approved_by=interaction.user.id,
                    approved_at=approved_at,
                    target_channel_id=target.id,
                )
                if self.cog.feedback_store is not None:
                    self.cog.feedback_store.add_report_example(
                        guild_id=interaction.guild.id,
                        profile_key=(str(review.get("knowledge_profile")) if review.get("knowledge_profile") else None),
                        review_id=str(review.get("review_id") or self.review_path.stem),
                        title=self.title,
                        report_text=self.report_path.read_text(encoding="utf-8"),
                        approved_by=interaction.user.id,
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
                        "這份核准版本也已保存為同一 Knowledge Profile 的報告範例。"
                    ),
                    view=self,
                )
            await interaction.followup.send(f"已發布到 {target.mention}。", ephemeral=True)

    @discord.ui.button(label="修正並重新整理", style=discord.ButtonStyle.primary, emoji="✏️")
    async def revise(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        del button
        if not await self._guard(interaction):
            return
        await interaction.response.send_modal(
            MeetingRevisionModal(
                self.cog,
                report_path=self.report_path,
                review_path=self.review_path,
                title=self.title,
            )
        )

    @discord.ui.button(label="退回／不發布", style=discord.ButtonStyle.secondary, emoji="↩️")
    async def reject(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        del button
        if not await self._guard(interaction):
            return
        assert isinstance(interaction.user, discord.Member)
        self.cog._update_review(
            self.review_path,
            status="rejected",
            rejected_by=interaction.user.id,
            rejected_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        )
        self._disable()
        await interaction.response.edit_message(
            content=f"↩️ 週會報草稿已由 {interaction.user.display_name} 退回，沒有發布到設定頻道。",
            view=self,
        )


class MeetingReportCommands(KnowledgeMeetingReportCommands):
    def __init__(self, bot: commands.Bot, settings: Any, ai: Any) -> None:
        super().__init__(bot, settings, ai)
        database = getattr(bot, "database", None)
        self.feedback_store = (
            MeetingFeedbackStore(database, settings.project_root) if database is not None else None
        )

    @staticmethod
    def _read_review(path: Path) -> dict[str, object]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return {}

    def _snapshot_for_profile(self, guild_id: int, profile_key: str | None) -> AppKnowledgeSnapshot | None:
        if self.knowledge_store is None or not profile_key:
            return None
        profile = self.knowledge_store.get_profile(guild_id, profile_key)
        if profile is None:
            return None
        return AppKnowledgeSnapshot(
            profile=profile,
            terms=tuple(self.knowledge_store.list_terms(guild_id, profile.key)),
        )

    def _feedback_context(
        self, guild_id: int, profile_key: str | None
    ) -> tuple[list[str], list[str]]:
        if self.feedback_store is None:
            return [], []
        preferences = self.feedback_store.report_preferences(guild_id, profile_key, limit=10)
        examples = self.feedback_store.recent_report_examples(guild_id, profile_key, limit=2)
        return preferences, examples

    async def _generate_report_with_feedback(
        self,
        *,
        title: str,
        transcript: str,
        guild_id: int,
        snapshot: AppKnowledgeSnapshot | None,
        correction_text: str | None = None,
        previous_report: str | None = None,
    ) -> str:
        prompt = build_meeting_report_prompt(title, transcript)
        if snapshot is not None:
            prompt += (
                "\n\n---\n領域專有名詞資料庫：\n"
                f"{snapshot.explanation_block()}\n"
                "重要：上述詞庫只可協助理解詞義與修正名詞，不可把詞庫內的解釋當成本次會議發生的事、決議或進度。\n"
            )
        profile_key = snapshot.profile.key if snapshot is not None else None
        preferences, examples = self._feedback_context(guild_id, profile_key)
        if preferences:
            prompt += (
                "\n\n---\n已由人工核准的穩定報告偏好：\n"
                + "\n".join(f"- {item}" for item in preferences)
                + "\n這些偏好只控制整理方式，不是會議事實。\n"
            )
        if examples:
            prompt += "\n\n---\n以下為近期人工核准報告的格式範例，只可參考結構與寫法，絕對不可沿用其中的事實：\n"
            for index, example in enumerate(examples, start=1):
                prompt += f"\n[範例 {index}]\n{example[:4000]}\n"
        if correction_text:
            prompt += (
                "\n\n---\n本次人工審核提出的修正要求：\n"
                f"{correction_text.strip()}\n"
                "這些修正要求優先於原草稿；但仍不得新增逐字稿與修正要求都沒有支持的事實。\n"
            )
        if previous_report:
            prompt += (
                "\n\n---\n上一版草稿如下，請只把它當成待修正草稿，不可把草稿中沒有逐字稿支持的內容當成事實：\n"
                f"{previous_report[:12000]}\n"
            )
        text = await self.ai.social_reply(
            prompt,
            persona_instruction=(
                f"{BASE_PERSONA_INSTRUCTION}\n"
                "你現在執行會議紀錄整理，不可呼叫工具、搜尋外部資訊或使用逐字稿、人工修正、明示詞庫以外的事實。"
            ),
        )
        result = str(text).strip()
        if not result:
            raise RuntimeError("Gemini 沒有回傳週會報內容。")
        return result

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
                content="本地逐字稿已完成，墨雪正在依逐字稿、領域詞庫與已核准格式偏好整理草稿。"
            )
            report_body = await self._generate_report_with_feedback(
                title=title,
                transcript=transcription.text,
                guild_id=interaction.guild.id,
                snapshot=snapshot,
            )
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
            review_id = f"{digest[:12]}-{stamp}"
            target_id = self._configured_summary_channel_id()
            review = {
                "review_id": review_id,
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
            target_text = f"設定發布頻道：<#{target_id}>。" if target_id else "目前尚未設定發布頻道；請使用 `/meeting output`。"
            await interaction.edit_original_response(
                content=(
                    f"週會報草稿完成。錄音長度 `{duration}`，Whisper `{transcription.model}`，"
                    f"平均辨識信心 `{transcription.confidence:.2f}` {cache_text}\n{target_text}\n"
                    "請人工審核；可直接核准、提交修正後重新整理，或退回不發布。"
                ).strip()
            )
            await self._send_review_draft(
                interaction,
                report_path=report_path,
                review_path=review_path,
                transcript_txt=transcript_txt,
                transcript_json=transcript_json,
                title=title,
                requester_id=interaction.user.id,
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

    async def _send_review_draft(
        self,
        interaction: discord.Interaction,
        *,
        report_path: Path,
        review_path: Path,
        transcript_txt: Path,
        transcript_json: Path,
        title: str,
        requester_id: int,
    ) -> None:
        report = report_path.read_text(encoding="utf-8")
        preview = report[:1300]
        if len(report) > len(preview):
            preview = preview.rstrip() + "\n\n…完整內容請查看週會報附件。"
        safe_title = _safe_filename(title, "meeting")
        view = MeetingReportReviewView(
            self,
            report_path=report_path,
            review_path=review_path,
            title=title,
            requester_id=requester_id,
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

    async def handle_revision_submission(
        self,
        interaction: discord.Interaction,
        *,
        report_path: Path,
        review_path: Path,
        title: str,
        correction_text: str,
    ) -> None:
        if not interaction.guild or not isinstance(interaction.user, discord.Member):
            await interaction.response.send_message("只能在原伺服器提交修正。", ephemeral=True)
            return
        if not self._is_dj_member(interaction.user):
            await interaction.response.send_message("只有 DJ 或管理員可以提交修正。", ephemeral=True)
            return
        if self.feedback_store is None:
            await interaction.response.send_message("目前無法取得 Feedback database。", ephemeral=True)
            return
        correction = correction_text.strip()
        if not correction:
            await interaction.response.send_message("修正內容不可為空。", ephemeral=True)
            return

        review = self._read_review(review_path)
        audio_sha = str(review.get("audio_sha256") or review_path.parent.name)
        profile_key = str(review.get("knowledge_profile")) if review.get("knowledge_profile") else None
        review_id = str(review.get("review_id") or review_path.stem)
        original_report = report_path.read_text(encoding="utf-8")
        await interaction.response.defer(thinking=True, ephemeral=True)
        revision_id = self.feedback_store.create_revision(
            review_id=review_id,
            guild_id=interaction.guild.id,
            profile_key=profile_key,
            audio_sha256=audio_sha,
            title=title,
            correction_text=correction,
            original_report=original_report,
            created_by=interaction.user.id,
        )
        explicit = parse_explicit_corrections(correction)
        ai_candidates = await self._propose_learning_candidates(
            correction_text=correction,
            report=original_report,
            transcript_path=review_path.parent / "transcript.txt",
        )
        candidates = merge_candidates(explicit, ai_candidates)
        candidate_ids = self.feedback_store.add_candidates(revision_id, candidates)
        self._update_review(
            review_path,
            status="revision_pending",
            pending_revision_id=revision_id,
            pending_revision_by=interaction.user.id,
        )

        if candidates:
            lines = [f"{index}. {_candidate_label(item)}" for index, item in enumerate(candidates, start=1)]
            candidate_text = "\n".join(lines)
            message = (
                "我已收到人工修正。下面是**學習候選**，目前都還沒有寫入 Knowledge DB：\n"
                f"{candidate_text[:1400]}\n\n"
                "請確認要把這些候選套用後重整，或只依你的修正重新整理、不學習。"
            )
        else:
            message = (
                "我已收到人工修正，但沒有找到足夠明確、可安全重用的學習候選。\n"
                "仍可選「只重整，不學習」讓墨雪依你的修正重新產生草稿。"
            )
        await interaction.followup.send(
            message,
            ephemeral=True,
            view=MeetingFeedbackDecisionView(
                self,
                revision_id=revision_id,
                review_path=review_path,
                report_path=report_path,
                title=title,
                correction_text=correction,
                candidate_ids=candidate_ids,
            ),
        )

    async def _propose_learning_candidates(
        self, *, correction_text: str, report: str, transcript_path: Path
    ) -> list[FeedbackCandidate]:
        try:
            transcript = transcript_path.read_text(encoding="utf-8")[:14000]
        except OSError:
            transcript = ""
        prompt = (
            "你正在分析人工對會議草稿的修正，目的只是提出『可重用的學習候選』，不是改寫報告。\n"
            "只允許三種 kind：\n"
            "- asr_alias：人工修正明確顯示某個 Whisper 誤辨識應對應到標準專有名詞。\n"
            "- domain_term：人工修正明確提供了一個專有名詞及其可重用的簡短詞義。\n"
            "- report_preference：人工明確提出未來都應遵守的格式／寫作偏好。\n"
            "一次性的會議事實、數字、負責人、時間、戰術結論不得變成可重用學習候選。\n"
            "不確定就不要提出。不得使用外部知識。\n"
            "只回傳 JSON，不要 Markdown：\n"
            '{"candidates":[{"kind":"asr_alias|domain_term|report_preference","source_text":"",'
            '"canonical_term":"","explanation":"","aliases":[],"confidence":0.0}]}\n\n'
            f"人工修正：\n{correction_text[:4000]}\n\n"
            f"原草稿：\n{report[:7000]}\n\n"
            f"逐字稿節錄：\n{transcript}\n"
        )
        try:
            response = await self.ai.social_reply(
                prompt,
                persona_instruction=(
                    "只做資料分類，不可呼叫工具、搜尋、使用記憶或補充外部事實。"
                    "輸出必須是單一 JSON object。"
                ),
            )
        except Exception:
            LOGGER.warning("Could not propose meeting learning candidates", exc_info=True)
            return []
        return parse_ai_candidates(str(response))

    def _apply_learning_candidates(
        self,
        *,
        guild_id: int,
        profile_key: str | None,
        revision_id: int,
        created_by: int,
        candidates: list[tuple[int, FeedbackCandidate]],
    ) -> AppKnowledgeSnapshot | None:
        if self.knowledge_store is None or self.feedback_store is None:
            return self._snapshot_for_profile(guild_id, profile_key)
        applied_ids: list[int] = []
        for candidate_id, candidate in candidates:
            if candidate.kind == "report_preference":
                self.feedback_store.add_report_preference(
                    guild_id=guild_id,
                    profile_key=profile_key,
                    instruction=candidate.explanation,
                    revision_id=revision_id,
                    created_by=created_by,
                )
                applied_ids.append(candidate_id)
                continue
            if not profile_key or not candidate.canonical_term:
                continue
            try:
                existing = self.knowledge_store.get_term(guild_id, profile_key, candidate.canonical_term)
            except ValueError:
                continue
            aliases: list[str] = list(existing.aliases if existing else ())
            for alias in candidate.aliases:
                if alias and alias.casefold() not in {item.casefold() for item in aliases}:
                    aliases.append(alias)
            explanation = candidate.explanation or (existing.explanation if existing else "人工審核確認的專有名詞。")
            try:
                self.knowledge_store.upsert_term(
                    guild_id,
                    profile_key,
                    candidate.canonical_term,
                    explanation,
                    aliases,
                )
            except ValueError:
                LOGGER.warning("Could not apply meeting knowledge candidate %s", candidate, exc_info=True)
                continue
            applied_ids.append(candidate_id)
        self.feedback_store.set_candidate_status(applied_ids, "applied")
        return self._snapshot_for_profile(guild_id, profile_key)

    @staticmethod
    def _reviewed_transcript(
        payload: dict[str, object], snapshot: AppKnowledgeSnapshot | None
    ) -> MeetingTranscription:
        segments: list[dict[str, object]] = []
        for raw in list(payload.get("segments", [])):
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            if snapshot is not None:
                corrected, _ = _apply_snapshot_corrections(str(item.get("text", "")), snapshot)
                item["text"] = corrected
            segments.append(item)
        text = format_timestamped_transcript(segments)
        return MeetingTranscription(
            text=text,
            segments=segments,
            language=(str(payload.get("language")) if payload.get("language") else None),
            duration_seconds=float(payload.get("duration_seconds", 0.0)),
            confidence=float(payload.get("confidence", 0.0)),
            model=str(payload.get("model", "small")),
            config_fingerprint=str(payload.get("config_fingerprint", "")),
        )

    async def regenerate_from_feedback(
        self,
        interaction: discord.Interaction,
        *,
        revision_id: int,
        review_path: Path,
        report_path: Path,
        title: str,
        correction_text: str,
        learn: bool,
        selected_candidate_ids: list[int] | None = None,
    ) -> None:
        assert interaction.guild is not None and isinstance(interaction.user, discord.Member)
        if self.feedback_store is None:
            await interaction.followup.send("Feedback database 無法使用。", ephemeral=True)
            return
        # None is a retry: only previously approved candidates may be learned.
        if selected_candidate_ids is not None or not learn:
            self.feedback_store.select_candidates(
                revision_id, selected_candidate_ids if learn and selected_candidate_ids else [],
            )
        review = self._read_review(review_path)
        profile_key = str(review.get("knowledge_profile")) if review.get("knowledge_profile") else None
        audio_sha = str(review.get("audio_sha256") or review_path.parent.name)
        transcript_json = review_path.parent / "transcript.json"
        try:
            payload = json.loads(transcript_json.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            await interaction.followup.send(f"無法讀取原逐字稿：{error}", ephemeral=True)
            return
        if not isinstance(payload, dict):
            await interaction.followup.send("原逐字稿格式無效。", ephemeral=True)
            return

        approved = self.feedback_store.candidates(revision_id, status="approved")
        snapshot = self._snapshot_for_profile(interaction.guild.id, profile_key)
        learned_examples = 0
        if learn:
            snapshot = self._apply_learning_candidates(
                guild_id=interaction.guild.id,
                profile_key=profile_key,
                revision_id=revision_id,
                created_by=interaction.user.id,
                candidates=approved,
            )
            learned_examples = self.feedback_store.record_asr_examples(
                revision_id=revision_id,
                guild_id=interaction.guild.id,
                profile_key=profile_key,
                audio_sha256=audio_sha,
                segments=[item for item in list(payload.get("segments", [])) if isinstance(item, dict)],
                candidates=[item for _, item in approved],
                created_by=interaction.user.id,
            )
        reviewed = self._reviewed_transcript(payload, snapshot if learn else None)
        title_token = _MEETING_TITLE.set(title)
        knowledge_token = _MEETING_KNOWLEDGE.set(snapshot)
        try:
            report_body = await self._generate_report_with_feedback(
                title=title,
                transcript=reviewed.text or str(payload.get("text", "")),
                guild_id=interaction.guild.id,
                snapshot=snapshot,
                correction_text=correction_text,
                previous_report=report_path.read_text(encoding="utf-8"),
            )
        finally:
            _MEETING_KNOWLEDGE.reset(knowledge_token)
            _MEETING_TITLE.reset(title_token)

        now = datetime.now().astimezone(self._local_zone())
        duration = format_meeting_timestamp(reviewed.duration_seconds)
        profile_name = snapshot.profile.display_name if snapshot is not None else "未指定"
        report = (
            f"> 📅 **週會報日期：** {now:%Y-%m-%d}\n"
            f"> 🕒 **重新整理時間：** {now:%H:%M} ({now.tzname() or 'local'})\n"
            f"> 🎧 **錄音長度：** {duration}\n"
            f"> 📚 **Knowledge Profile：** {profile_name}\n"
            f"> ✏️ **人工修訂版本：** #{revision_id}\n\n"
            f"{report_body.strip()}\n"
        )
        stamp = now.strftime("%Y%m%d_%H%M%S")
        revised_report_path = review_path.parent / f"report_revision_{revision_id}_{stamp}.md"
        revised_report_path.write_text(report, encoding="utf-8")
        reviewed_txt = review_path.parent / f"transcript_reviewed_{revision_id}.txt"
        reviewed_json = review_path.parent / f"transcript_reviewed_{revision_id}.json"
        reviewed_txt.write_text((reviewed.text or str(payload.get("text", ""))).rstrip() + "\n", encoding="utf-8")
        reviewed_payload = dict(payload)
        reviewed_payload.update(
            {
                "text": reviewed.text or str(payload.get("text", "")),
                "segments": reviewed.segments if reviewed.segments else payload.get("segments", []),
                "human_feedback_revision_id": revision_id,
                "knowledge_learning_applied": learn,
            }
        )
        reviewed_json.write_text(json.dumps(reviewed_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.feedback_store.update_revision(
            revision_id,
            status="draft_after_learning" if learn else "draft_without_learning",
            revised_report=report,
        )
        self._update_review(
            review_path,
            status="draft_revision",
            last_revision_id=revision_id,
            report_file=revised_report_path.name,
            reviewed_transcript_file=reviewed_txt.name,
        )

        safe_title = _safe_filename(title, "meeting")
        view = MeetingReportReviewView(
            self,
            report_path=revised_report_path,
            review_path=review_path,
            title=title,
            requester_id=interaction.user.id,
        )
        preview = report[:1300]
        if len(report) > len(preview):
            preview = preview.rstrip() + "\n\n…完整內容請查看附件。"
        if interaction.channel is None:
            await interaction.followup.send("重新整理完成，但找不到原頻道可送出草稿。", ephemeral=True)
            return
        await interaction.channel.send(
            content=preview,
            files=[
                discord.File(revised_report_path, filename=f"{safe_title}_週會報_修訂{revision_id}_待審核.md"),
                discord.File(reviewed_txt, filename=f"{safe_title}_人工校正逐字稿_{revision_id}.txt"),
            ],
            view=view,
            allowed_mentions=discord.AllowedMentions.none(),
        )
        learning_text = (
            f"已套用人工核准的學習候選；另保存 `{learned_examples}` 個可供未來 Whisper fine-tune 的音訊片段標註。"
            if learn
            else "本次只依修正重新整理，沒有寫入 Knowledge DB 或訓練資料。"
        )
        await interaction.followup.send(f"新的待審核草稿已送到目前頻道。{learning_text}", ephemeral=True)

    @app_commands.command(name="meeting_training_stats", description="查看某 Knowledge Profile 累積的 Whisper 人工校正訓練資料")
    @app_commands.describe(profile="Profile key，例如 lifeafter；留空使用目前預設")
    async def meeting_training_stats(
        self, interaction: discord.Interaction, profile: str | None = None
    ) -> None:
        if not interaction.guild or not isinstance(interaction.user, discord.Member):
            await interaction.response.send_message("只能在伺服器內查看訓練資料。", ephemeral=True)
            return
        if not self._is_dj_member(interaction.user) or self.feedback_store is None:
            await interaction.response.send_message("只有 DJ／管理員可以查看訓練資料。", ephemeral=True)
            return
        selected = profile
        if not selected and self.knowledge_store is not None:
            default = self.knowledge_store.default_profile(interaction.guild.id)
            selected = default.key if default else None
        stats = self.feedback_store.training_stats(interaction.guild.id, selected)
        await interaction.response.send_message(
            f"Profile `{selected or 'general'}`：已累積 **{stats['examples']}** 個人工校正語音片段，來自 **{stats['recordings']}** 份錄音。",
            ephemeral=True,
        )

    @app_commands.command(name="meeting_training_export", description="匯出可供未來本地 Whisper fine-tune 使用的人工校正 manifest")
    @app_commands.describe(profile="Profile key，例如 lifeafter；留空使用目前預設")
    async def meeting_training_export(
        self, interaction: discord.Interaction, profile: str | None = None
    ) -> None:
        if not interaction.guild or not isinstance(interaction.user, discord.Member):
            await interaction.response.send_message("只能在伺服器內匯出訓練資料。", ephemeral=True)
            return
        if not self._is_dj_member(interaction.user) or self.feedback_store is None:
            await interaction.response.send_message("只有 DJ／管理員可以匯出訓練資料。", ephemeral=True)
            return
        selected = profile
        if not selected and self.knowledge_store is not None:
            default = self.knowledge_store.default_profile(interaction.guild.id)
            selected = default.key if default else None
        path, count = self.feedback_store.export_training_manifest(interaction.guild.id, selected)
        await interaction.response.send_message(
            content=(
                f"已匯出 `{count}` 筆人工核准 ASR 校正。這只是訓練資料 manifest，"
                "目前**不會**在 Bot 執行期間自動 fine-tune 模型。"
            ),
            file=discord.File(path, filename=path.name),
            ephemeral=True,
        )
