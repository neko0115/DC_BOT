from __future__ import annotations

import asyncio
import importlib.util
import json
import math
import os
import re
import shutil
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

TOOL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TOOL_ROOT.parents[1]
DEFAULT_CONFIG_PATH = TOOL_ROOT / "config.json"
DATA_ROOT = PROJECT_ROOT / "data" / "game-meeting-recorder"
CONFIG_PATH = DATA_ROOT / "config.json"


@dataclass(slots=True)
class AudioPacket:
    samples: Any
    captured_at: str
    duration_seconds: float


@dataclass(slots=True)
class MeetingSession:
    session_id: str
    name: str
    root: Path
    started_at: str
    config: dict[str, Any]
    events: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    recent_chat_lines: deque[str] = field(default_factory=deque)
    stop_event: threading.Event = field(default_factory=threading.Event)
    audio_queue: asyncio.Queue[AudioPacket | None] = field(default_factory=asyncio.Queue)
    tasks: list[asyncio.Task[Any]] = field(default_factory=list)
    stopped_at: str | None = None

    @property
    def events_path(self) -> Path:
        return self.root / "events.jsonl"

    @property
    def metadata_path(self) -> Path:
        return self.root / "session.json"


class GameMeetingRecorder:
    def __init__(self) -> None:
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        self._session: MeetingSession | None = None
        self._last_session: MeetingSession | None = None
        self._whisper_models: dict[str, Any] = {}
        self._ocr_engine: Any | None = None
        self._state_lock = asyncio.Lock()

    async def invoke(self, action: str, arguments: dict[str, object], context: dict[str, object]) -> dict[str, object]:
        handlers = {
            "status": lambda: self.status(),
            "list_audio_devices": lambda: asyncio.to_thread(self.list_audio_devices),
            "configure_audio": lambda: self.configure_audio(arguments),
            "configure_chat": lambda: self.configure_chat(arguments),
            "configure_output": lambda: self.configure_output(arguments),
            "test_chat_capture": lambda: self.test_chat_capture(),
            "start_session": lambda: self.start_session(arguments, context),
            "stop_session": lambda: self.stop_session(),
            "get_transcript": lambda: self.get_transcript(arguments),
        }
        handler = handlers.get(action)
        if handler is None:
            raise ValueError(f"unsupported action: {action}")
        result = handler()
        return await result if asyncio.iscoroutine(result) else result

    def status(self) -> dict[str, object]:
        config = self._load_config()
        session = self._session
        chat = config["chat"]
        return {
            "message": "會議紀錄工具正在錄製。" if session else "會議紀錄工具目前待命。",
            "recording": session is not None,
            "session_id": session.session_id if session else None,
            "session_name": session.name if session else None,
            "event_count": len(session.events) if session else 0,
            "voice_events": self._count(session, "voice"),
            "chat_events": self._count(session, "chat"),
            "errors": list(session.errors[-5:]) if session else [],
            "audio_enabled": bool(config["audio"].get("enabled", True)),
            "chat_enabled": bool(chat.get("enabled", False)),
            "chat_roi": {key: chat.get(key) for key in ("x", "y", "width", "height")},
            "summary_channel_id": config["output"].get("summary_channel_id"),
            "dependencies": self._dependency_status(),
            "config_path": str(CONFIG_PATH),
            "default_config_path": str(DEFAULT_CONFIG_PATH),
            "data_root": str(DATA_ROOT),
        }

    @staticmethod
    def _count(session: MeetingSession | None, source: str) -> int:
        if session is None:
            return 0
        return sum(1 for event in session.events if event.get("source") == source)

    def list_audio_devices(self) -> dict[str, object]:
        self._require_modules(("soundcard", "numpy"), "音訊擷取")
        import soundcard as sc

        default = sc.default_speaker()
        speakers = [
            {"id": str(item.id), "name": item.name, "channels": int(item.channels), "default": str(item.id) == str(default.id)}
            for item in sc.all_speakers()
        ]
        loopbacks = [
            {"id": str(item.id), "name": item.name, "channels": int(item.channels), "is_loopback": True}
            for item in sc.all_microphones(include_loopback=True)
            if bool(getattr(item, "isloopback", False))
        ]
        return {
            "message": "已列出輸出與 loopback 裝置。loopback_device_id 留空時會優先匹配預設輸出。",
            "default_speaker": {"id": str(default.id), "name": default.name},
            "speakers": speakers,
            "loopbacks": loopbacks,
        }

    def configure_audio(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._update_section(
            "audio",
            arguments,
            {
                "loopback_device_id", "sample_rate", "speech_threshold_dbfs", "silence_seconds",
                "pre_roll_seconds", "max_segment_seconds", "whisper_model", "whisper_language",
                "whisper_beam_size", "whisper_initial_prompt",
            },
            "音訊與 Whisper 設定已更新。",
        )

    def configure_chat(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._update_section(
            "chat",
            arguments,
            {"enabled", "x", "y", "width", "height", "poll_seconds", "change_threshold", "ocr_min_confidence", "scale"},
            "聊天室 OCR 設定已更新；建議接著執行 test_chat_capture。",
        )

    def configure_output(self, arguments: dict[str, object]) -> dict[str, object]:
        return self._update_section(
            "output", arguments, {"summary_channel_id", "retain_audio"}, "會議輸出設定已更新。"
        )

    def _update_section(
        self, section: str, arguments: dict[str, object], allowed: set[str], message: str
    ) -> dict[str, object]:
        config = self._load_config()
        values = config[section]
        for key in allowed:
            if key in arguments:
                values[key] = arguments[key]
        self._validate_config(config)
        self._save_config(config)
        return {"message": message, section: values}

    async def test_chat_capture(self) -> dict[str, object]:
        config = self._load_config()
        self._require_modules(("PIL", "rapidocr", "onnxruntime", "numpy"), "聊天室 OCR")
        original, processed = await asyncio.to_thread(self._capture_and_preprocess_chat, config["chat"])
        folder = DATA_ROOT / "calibration"
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        original_path = folder / f"chat_{stamp}.png"
        processed_path = folder / f"chat_{stamp}_processed.png"
        await asyncio.to_thread(original.save, original_path)
        await asyncio.to_thread(processed.save, processed_path)
        lines = await asyncio.to_thread(self._ocr_image, processed, float(config["chat"]["ocr_min_confidence"]))
        return {
            "message": "聊天室 ROI 測試完成；若文字缺漏，調整座標、尺寸或 scale 後再測。",
            "lines": lines,
            "original_image": str(original_path),
            "processed_image": str(processed_path),
        }

    async def start_session(self, arguments: dict[str, object], context: dict[str, object]) -> dict[str, object]:
        async with self._state_lock:
            if self._session is not None:
                return {"message": "目前已經有一場紀錄進行中。", "session_id": self._session.session_id}
            config = self._load_config()
            missing: list[str] = []
            if bool(config["audio"].get("enabled", True)):
                missing += self._missing_modules(("soundcard", "numpy", "faster_whisper"))
            if bool(config["chat"].get("enabled", False)):
                missing += self._missing_modules(("PIL", "rapidocr", "onnxruntime", "numpy"))
            if missing:
                return {
                    "message": "缺少會議紀錄依賴，尚未開始錄製。",
                    "missing_dependencies": sorted(set(missing)),
                    "install_command": 'python -m pip install -e ".[meeting]"',
                }

            now = datetime.now().astimezone()
            session_id = now.strftime("%Y%m%d_%H%M%S")
            name_value = arguments.get("name")
            name = name_value.strip() if isinstance(name_value, str) and name_value.strip() else f"Game session {session_id}"
            root = DATA_ROOT / "sessions" / session_id
            root.mkdir(parents=True, exist_ok=False)
            (root / "audio").mkdir(exist_ok=True)
            session = MeetingSession(session_id, name, root, now.isoformat(timespec="seconds"), config)
            session.recent_chat_lines = deque(maxlen=int(config["chat"].get("dedup_window", 80)))
            self._session = session
            self._write_metadata(session, context)
            loop = asyncio.get_running_loop()
            if bool(config["audio"].get("enabled", True)):
                session.tasks.append(asyncio.create_task(self._transcription_loop(session), name=f"meeting-transcribe-{session_id}"))
                session.tasks.append(
                    asyncio.create_task(asyncio.to_thread(self._audio_capture_worker, session, loop), name=f"meeting-audio-{session_id}")
                )
            if bool(config["chat"].get("enabled", False)):
                session.tasks.append(asyncio.create_task(self._chat_loop(session), name=f"meeting-chat-{session_id}"))
            return {
                "message": "已開始遊戲討論紀錄；語音與聊天室會分開保存，再依時間合併。",
                "session_id": session_id,
                "name": name,
                "audio_enabled": bool(config["audio"].get("enabled", True)),
                "chat_enabled": bool(config["chat"].get("enabled", False)),
                "summary_channel_id": config["output"].get("summary_channel_id"),
            }

    async def stop_session(self) -> dict[str, object]:
        async with self._state_lock:
            session = self._session
            if session is None:
                return {"message": "目前沒有正在進行的會議／遊戲紀錄。"}
            session.stop_event.set()

        timed_out = False
        if session.tasks:
            try:
                await asyncio.wait_for(asyncio.gather(*session.tasks, return_exceptions=True), timeout=50.0)
            except asyncio.TimeoutError:
                timed_out = True
                session.errors.append("停止時等待辨識工作超過 50 秒；逐字稿可能仍有未完成片段。")
                for task in session.tasks:
                    if not task.done():
                        task.cancel()

        async with self._state_lock:
            session.stopped_at = datetime.now().astimezone().isoformat(timespec="seconds")
            self._sort_and_rewrite_events(session)
            self._write_metadata(session)
            self._session = None
            self._last_session = session

        if not bool(session.config["output"].get("retain_audio", True)):
            await asyncio.to_thread(self._delete_audio_files, session)
        payload = self._summary_payload(session, limit_events=350)
        payload["message"] = "已停止紀錄並產生可供墨雪摘要的逐字稿。" if not timed_out else "已停止紀錄，但有辨識工作逾時。"
        payload["incomplete"] = timed_out
        return payload

    def get_transcript(self, arguments: dict[str, object]) -> dict[str, object]:
        raw_limit = arguments.get("limit", 200)
        limit = max(1, min(500, int(raw_limit) if isinstance(raw_limit, (int, float)) else 200))
        session = self._session or self._last_session
        if session is not None:
            return self._summary_payload(session, limit_events=limit)
        latest = self._latest_session_from_disk()
        if latest is None:
            return {"message": "目前還沒有任何會議紀錄。", "events": []}
        events = self._read_events(latest / "events.jsonl")[-limit:]
        metadata = self._read_json(latest / "session.json", {})
        return {
            "message": "已載入最近一次會議紀錄。",
            "session_id": metadata.get("session_id", latest.name),
            "name": metadata.get("name", latest.name),
            "events": events,
            "transcript": self._format_transcript(events),
            "session_path": str(latest),
            "summary_channel_id": metadata.get("summary_channel_id"),
        }

    def _audio_capture_worker(self, session: MeetingSession, loop: asyncio.AbstractEventLoop) -> None:
        audio = session.config["audio"]
        try:
            import numpy as np
            import soundcard as sc

            sample_rate = int(audio["sample_rate"])
            chunk_seconds = float(audio.get("chunk_seconds", 0.25))
            chunk_frames = max(256, int(sample_rate * chunk_seconds))
            threshold_dbfs = float(audio["speech_threshold_dbfs"])
            silence_seconds = float(audio["silence_seconds"])
            pre_roll_chunks = max(1, math.ceil(float(audio["pre_roll_seconds"]) / chunk_seconds))
            max_segment_seconds = float(audio["max_segment_seconds"])
            pre_roll: deque[Any] = deque(maxlen=pre_roll_chunks)
            loopback = self._select_loopback(sc, audio.get("loopback_device_id"))
            active: list[Any] = []
            active_started_at: str | None = None
            silent_for = 0.0
            active_duration = 0.0

            with loopback.recorder(samplerate=sample_rate) as recorder:
                while not session.stop_event.is_set():
                    block = recorder.record(numframes=chunk_frames)
                    if block is None or getattr(block, "size", 0) == 0:
                        continue
                    data = np.asarray(block, dtype=np.float32)
                    mono = data if data.ndim == 1 else np.mean(data, axis=1, dtype=np.float32)
                    rms = float(np.sqrt(np.mean(np.square(mono), dtype=np.float64))) if mono.size else 0.0
                    speech = 20.0 * math.log10(max(rms, 1e-12)) >= threshold_dbfs
                    if not active:
                        if speech:
                            active_started_at = datetime.now().astimezone().isoformat(timespec="milliseconds")
                            active.extend(list(pre_roll))
                            active.append(mono.copy())
                            active_duration = sum(len(item) for item in active) / sample_rate
                        pre_roll.append(mono.copy())
                        continue
                    active.append(mono.copy())
                    block_duration = len(mono) / sample_rate
                    active_duration += block_duration
                    silent_for = 0.0 if speech else silent_for + block_duration
                    if silent_for >= silence_seconds or active_duration >= max_segment_seconds:
                        self._queue_audio(session, loop, np.concatenate(active), active_started_at, sample_rate)
                        active.clear()
                        active_started_at = None
                        silent_for = 0.0
                        active_duration = 0.0
                        pre_roll.clear()
                if active:
                    self._queue_audio(session, loop, np.concatenate(active), active_started_at, sample_rate)
        except Exception as error:
            session.errors.append(f"audio capture: {type(error).__name__}: {error}")
        finally:
            try:
                asyncio.run_coroutine_threadsafe(session.audio_queue.put(None), loop).result(timeout=5)
            except Exception:
                pass

    @staticmethod
    def _queue_audio(session: MeetingSession, loop: asyncio.AbstractEventLoop, samples: Any, captured_at: str | None, sample_rate: int) -> None:
        packet = AudioPacket(
            samples,
            captured_at or datetime.now().astimezone().isoformat(timespec="milliseconds"),
            float(len(samples) / sample_rate),
        )
        asyncio.run_coroutine_threadsafe(session.audio_queue.put(packet), loop).result(timeout=5)

    async def _transcription_loop(self, session: MeetingSession) -> None:
        audio = session.config["audio"]
        sequence = 0
        while True:
            packet = await session.audio_queue.get()
            if packet is None:
                break
            sequence += 1
            wav_path = session.root / "audio" / f"segment_{sequence:04d}.wav"
            try:
                await asyncio.to_thread(self._write_wav, wav_path, packet.samples, int(audio["sample_rate"]))
                result = await asyncio.to_thread(self._transcribe_wav, wav_path, audio)
                text = str(result.get("text", "")).strip()
                if text:
                    self._append_event(session, {
                        "timestamp": packet.captured_at,
                        "source": "voice",
                        "text": text,
                        "confidence": round(float(result.get("confidence", 0.0)), 4),
                        "audio_file": str(wav_path.relative_to(PROJECT_ROOT)),
                        "duration_seconds": round(packet.duration_seconds, 3),
                        "language": result.get("language"),
                    })
            except Exception as error:
                session.errors.append(f"transcription {wav_path.name}: {type(error).__name__}: {error}")

    async def _chat_loop(self, session: MeetingSession) -> None:
        chat = session.config["chat"]
        previous_signature: Any | None = None
        while not session.stop_event.is_set():
            started = time.monotonic()
            try:
                _, processed = await asyncio.to_thread(self._capture_and_preprocess_chat, chat)
                signature = await asyncio.to_thread(self._image_signature, processed)
                changed = previous_signature is None or self._signature_difference(previous_signature, signature) >= float(chat["change_threshold"])
                previous_signature = signature
                if changed:
                    lines = await asyncio.to_thread(self._ocr_image, processed, float(chat["ocr_min_confidence"]))
                    for line in lines:
                        text = str(line["text"]).strip()
                        if text and not self._is_duplicate_chat(session, text):
                            self._append_event(session, {
                                "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                                "source": "chat",
                                "text": text,
                                "confidence": round(float(line["confidence"]), 4),
                            })
            except Exception as error:
                message = f"chat OCR: {type(error).__name__}: {error}"
                if not session.errors or session.errors[-1] != message:
                    session.errors.append(message)
            await asyncio.sleep(max(0.05, float(chat["poll_seconds"]) - (time.monotonic() - started)))

    def _transcribe_wav(self, wav_path: Path, audio: dict[str, Any]) -> dict[str, object]:
        from faster_whisper import WhisperModel

        model_name = str(audio["whisper_model"])
        model = self._whisper_models.get(model_name)
        if model is None:
            model = WhisperModel(model_name, device="cpu", compute_type="int8", download_root=str(DATA_ROOT / "whisper-models"))
            self._whisper_models[model_name] = model
        language_raw = audio.get("whisper_language")
        language = language_raw.strip() if isinstance(language_raw, str) and language_raw.strip() else None
        segments, info = model.transcribe(
            str(wav_path), language=language, beam_size=int(audio["whisper_beam_size"]),
            initial_prompt=str(audio.get("whisper_initial_prompt", "")) or None,
            vad_filter=True, condition_on_previous_text=False,
        )
        realized = list(segments)
        text = " ".join(item.text.strip() for item in realized if item.text and item.text.strip()).strip()
        confidences = []
        for item in realized:
            avg_logprob = float(getattr(item, "avg_logprob", -5.0))
            no_speech = float(getattr(item, "no_speech_prob", 0.0))
            confidences.append(max(0.0, min(1.0, math.exp(min(0.0, avg_logprob)) * (1.0 - no_speech))))
        return {
            "text": text,
            "confidence": sum(confidences) / len(confidences) if confidences else 0.0,
            "language": getattr(info, "language", language),
        }

    @staticmethod
    def _write_wav(path: Path, samples: Any, sample_rate: int) -> None:
        import numpy as np

        pcm = (np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0) * 32767.0).astype("<i2")
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(pcm.tobytes())

    @staticmethod
    def _select_loopback(sc: Any, configured_id: object) -> Any:
        loopbacks = [item for item in sc.all_microphones(include_loopback=True) if bool(getattr(item, "isloopback", False))]
        if not loopbacks:
            raise RuntimeError("找不到任何 loopback 音訊裝置。")
        if isinstance(configured_id, str) and configured_id.strip():
            target = configured_id.strip()
            for item in loopbacks:
                if str(item.id) == target:
                    return item
            return sc.get_microphone(target, include_loopback=True)
        default = sc.default_speaker()
        for item in loopbacks:
            if str(item.id) == str(default.id):
                return item
        default_name = default.name.lower()
        for item in loopbacks:
            if default_name in item.name.lower() or item.name.lower() in default_name:
                return item
        return loopbacks[0]

    @staticmethod
    def _capture_and_preprocess_chat(chat: dict[str, Any]) -> tuple[Any, Any]:
        from PIL import ImageGrab, ImageOps

        x, y = int(chat["x"]), int(chat["y"])
        width, height = int(chat["width"]), int(chat["height"])
        original = ImageGrab.grab(bbox=(x, y, x + width, y + height), all_screens=True)
        processed = original
        scale = float(chat.get("scale", 2.0))
        if scale != 1.0:
            processed = processed.resize((max(1, int(processed.width * scale)), max(1, int(processed.height * scale))))
        processed = ImageOps.autocontrast(processed.convert("RGB"), cutoff=1)
        return original, processed

    def _ocr_image(self, image: Any, min_confidence: float) -> list[dict[str, object]]:
        import numpy as np
        from rapidocr import RapidOCR

        if self._ocr_engine is None:
            self._ocr_engine = RapidOCR()
        result = self._ocr_engine(np.asarray(image))
        texts = getattr(result, "txts", None) or ()
        scores = getattr(result, "scores", None) or ()
        return [
            {"text": str(text).strip(), "confidence": round(float(score), 4)}
            for text, score in zip(texts, scores)
            if str(text).strip() and float(score) >= min_confidence
        ]

    @staticmethod
    def _image_signature(image: Any) -> Any:
        import numpy as np
        return np.asarray(image.convert("L").resize((64, 36)), dtype=np.int16)

    @staticmethod
    def _signature_difference(previous: Any, current: Any) -> float:
        import numpy as np
        return float(np.mean(np.abs(current - previous)) / 255.0)

    def _is_duplicate_chat(self, session: MeetingSession, text: str) -> bool:
        candidate = re.sub(r"[\s\u3000]+", "", text).strip().lower()
        threshold = float(session.config["chat"].get("dedup_similarity", 0.9))
        for existing in session.recent_chat_lines:
            if candidate == existing or SequenceMatcher(None, candidate, existing).ratio() >= threshold:
                return True
        session.recent_chat_lines.append(candidate)
        return False

    def _append_event(self, session: MeetingSession, event: dict[str, Any]) -> None:
        session.events.append(event)
        with session.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _sort_and_rewrite_events(self, session: MeetingSession) -> None:
        session.events.sort(key=lambda item: str(item.get("timestamp", "")))
        with session.events_path.open("w", encoding="utf-8") as handle:
            for event in session.events:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _write_metadata(self, session: MeetingSession, context: dict[str, object] | None = None) -> None:
        metadata: dict[str, object] = {
            "session_id": session.session_id,
            "name": session.name,
            "started_at": session.started_at,
            "stopped_at": session.stopped_at,
            "event_count": len(session.events),
            "voice_events": self._count(session, "voice"),
            "chat_events": self._count(session, "chat"),
            "errors": session.errors,
            "summary_channel_id": session.config["output"].get("summary_channel_id"),
            "retain_audio": session.config["output"].get("retain_audio", True),
            "config_snapshot": session.config,
        }
        if context:
            metadata["started_by"] = {"guild_id": context.get("guild_id"), "user_id": context.get("user_id")}
        session.metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def _summary_payload(self, session: MeetingSession, *, limit_events: int) -> dict[str, object]:
        events = sorted(session.events, key=lambda item: str(item.get("timestamp", "")))
        selected = events[-limit_events:]
        transcript = self._format_transcript(selected)
        truncated = len(events) > len(selected)
        if len(transcript) > 45000:
            transcript = transcript[-45000:]
            truncated = True
        return {
            "session_id": session.session_id,
            "name": session.name,
            "started_at": session.started_at,
            "stopped_at": session.stopped_at,
            "event_count": len(events),
            "voice_events": sum(1 for item in events if item.get("source") == "voice"),
            "chat_events": sum(1 for item in events if item.get("source") == "chat"),
            "errors": list(session.errors),
            "events": selected,
            "transcript": transcript,
            "transcript_truncated": truncated,
            "session_path": str(session.root),
            "events_file": str(session.events_path),
            "summary_channel_id": session.config["output"].get("summary_channel_id"),
            "summary_instruction": (
                "只根據 transcript/events 整理本次進度、重要決策、分工、問題/風險、待辦與未確認資訊。"
                "低信心或語意不完整內容不得自行補成正式決議，必要時標註辨識可能有誤。"
            ),
        }

    @staticmethod
    def _format_transcript(events: list[dict[str, Any]]) -> str:
        lines = []
        for event in events:
            raw_time = str(event.get("timestamp", ""))
            try:
                display_time = datetime.fromisoformat(raw_time).strftime("%H:%M:%S")
            except ValueError:
                display_time = raw_time
            source = str(event.get("source", "unknown")).upper()
            confidence = event.get("confidence")
            conf = f" {float(confidence):.2f}" if isinstance(confidence, (int, float)) else ""
            lines.append(f"[{display_time}] [{source}{conf}] {event.get('text', '')}")
        return "\n".join(lines)

    @staticmethod
    def _delete_audio_files(session: MeetingSession) -> None:
        shutil.rmtree(session.root / "audio", ignore_errors=True)

    @staticmethod
    def _read_events(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        events = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                events.append(value)
        return events

    @staticmethod
    def _read_json(path: Path, fallback: Any) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return fallback

    def _latest_session_from_disk(self) -> Path | None:
        root = DATA_ROOT / "sessions"
        if not root.is_dir():
            return None
        folders = sorted((item for item in root.iterdir() if item.is_dir()), reverse=True)
        return folders[0] if folders else None

    def _load_config(self) -> dict[str, Any]:
        if not CONFIG_PATH.is_file():
            default = self._read_json(DEFAULT_CONFIG_PATH, None)
            if not isinstance(default, dict):
                raise RuntimeError(f"無法讀取預設設定：{DEFAULT_CONFIG_PATH}")
            self._validate_config(default)
            self._save_config(default)
        config = self._read_json(CONFIG_PATH, None)
        if not isinstance(config, dict):
            raise RuntimeError(f"無法讀取設定：{CONFIG_PATH}")
        self._validate_config(config)
        return config

    @staticmethod
    def _save_config(config: dict[str, Any]) -> None:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        temp = CONFIG_PATH.with_suffix(".json.tmp")
        temp.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temp, CONFIG_PATH)

    @staticmethod
    def _validate_config(config: dict[str, Any]) -> None:
        for section in ("audio", "chat", "output"):
            if not isinstance(config.get(section), dict):
                raise ValueError(f"config.{section} must be an object")
        audio, chat, output = config["audio"], config["chat"], config["output"]
        if not 16000 <= int(audio.get("sample_rate", 0)) <= 96000:
            raise ValueError("audio.sample_rate must be 16000..96000")
        if not -70 <= float(audio.get("speech_threshold_dbfs", -100)) <= -5:
            raise ValueError("audio.speech_threshold_dbfs must be -70..-5")
        if not 0.4 <= float(audio.get("silence_seconds", 0)) <= 5:
            raise ValueError("audio.silence_seconds must be 0.4..5")
        if not 0 <= float(audio.get("pre_roll_seconds", -1)) <= 3:
            raise ValueError("audio.pre_roll_seconds must be 0..3")
        if not 5 <= float(audio.get("max_segment_seconds", 0)) <= 120:
            raise ValueError("audio.max_segment_seconds must be 5..120")
        if not 1 <= int(audio.get("whisper_beam_size", 0)) <= 10:
            raise ValueError("audio.whisper_beam_size must be 1..10")
        if int(chat.get("x", -1)) < 0 or int(chat.get("y", -1)) < 0:
            raise ValueError("chat x/y must be >= 0")
        if int(chat.get("width", 0)) < 20 or int(chat.get("height", 0)) < 20:
            raise ValueError("chat width/height must be >= 20")
        if not 0.2 <= float(chat.get("poll_seconds", 0)) <= 5:
            raise ValueError("chat.poll_seconds must be 0.2..5")
        if not 0.001 <= float(chat.get("change_threshold", 0)) <= 0.5:
            raise ValueError("chat.change_threshold must be 0.001..0.5")
        if not 0 <= float(chat.get("ocr_min_confidence", -1)) <= 1:
            raise ValueError("chat.ocr_min_confidence must be 0..1")
        if not 1 <= float(chat.get("scale", 0)) <= 4:
            raise ValueError("chat.scale must be 1..4")
        channel_id = output.get("summary_channel_id")
        if channel_id is not None and (not isinstance(channel_id, int) or channel_id < 1):
            raise ValueError("output.summary_channel_id must be a positive integer or null")
        if not isinstance(output.get("retain_audio", True), bool):
            raise ValueError("output.retain_audio must be boolean")

    @staticmethod
    def _dependency_status() -> dict[str, bool]:
        names = ("soundcard", "numpy", "faster_whisper", "PIL", "rapidocr", "onnxruntime")
        return {name: importlib.util.find_spec(name) is not None for name in names}

    @staticmethod
    def _missing_modules(names: tuple[str, ...]) -> list[str]:
        return [name for name in names if importlib.util.find_spec(name) is None]

    def _require_modules(self, names: tuple[str, ...], feature: str) -> None:
        missing = self._missing_modules(names)
        if missing:
            raise RuntimeError(f"{feature}缺少依賴：{', '.join(missing)}。請執行 python -m pip install -e \".[meeting]\"")


RECORDER = GameMeetingRecorder()


async def invoke(action: str, arguments: dict[str, object], context: dict[str, object]) -> dict[str, object]:
    return await RECORDER.invoke(action, arguments, context)
