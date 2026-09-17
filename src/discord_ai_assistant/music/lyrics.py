from __future__ import annotations

import asyncio
import bisect
import html
import json
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

try:
    import yt_dlp
except ImportError:
    yt_dlp = None

from discord_ai_assistant.models import Track

LRCLIB_GET_URL = "https://lrclib.net/api/get"
LRCLIB_SEARCH_URL = "https://lrclib.net/api/search"
LRCLIB_USER_AGENT = "DC_BOT/0.6.1 (https://github.com/neko0115/DC_BOT)"
REQUEST_INTERVAL_SECONDS = 0.3
REQUEST_TIMEOUT_SECONDS = 12
MAX_STRUCTURED_SEARCHES = 6
MAX_KEYWORD_SEARCHES = 4

_TIMESTAMP_RE = re.compile(r"\[(\d{1,3}):(\d{2}(?:\.\d{1,3})?)\]")
_DECORATION_RE = re.compile(
    r"\s*[\[(](?:official\s*(?:music\s*)?video|official\s*audio|lyrics?|lyric\s*video|mv|m/v|audio|visualizer)[^\])]*[\])]\s*",
    re.IGNORECASE,
)
_TRAILING_DECORATION_RE = re.compile(
    r"(?:\s*[-|｜•·:]?\s*(?:official\s*(?:music\s*)?video|official\s*audio|official\s*mv|music\s*video|lyrics?|lyric\s*video|mv|m/v|audio|visualizer|4k|hd|官方(?:音樂)?(?:錄影帶|影片|mv)))+\s*$",
    re.IGNORECASE,
)
_TITLE_BRACKET_RE = re.compile(r"[【《「『]([^】》」』]{1,160})[】》」』]")
_WHITESPACE_RE = re.compile(r"\s+")
_EAST_ASIAN_RE = re.compile(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]")
_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_JAPANESE_RE = re.compile(r"[\u3040-\u30ff]")
_KOREAN_RE = re.compile(r"[\uac00-\ud7af]")
_LATIN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]")
_VTT_TIME_RE = re.compile(
    r"(?:(\d{1,2}):)?(\d{1,2}):(\d{2}(?:\.\d{1,3})?)\s+-->\s+"
)
_TAG_RE = re.compile(r"<[^>]+>")


@dataclass(frozen=True, slots=True)
class LyricLine:
    start_seconds: float
    text: str


@dataclass(frozen=True, slots=True)
class LyricsResult:
    track_name: str
    artist_name: str
    album_name: str
    duration: float | None
    instrumental: bool
    plain_lyrics: str | None
    synced_lyrics: str | None
    lines: tuple[LyricLine, ...]
    source: str = "LRCLIB"


@dataclass(frozen=True, slots=True)
class TrackMetadata:
    track_name: str | None = None
    artist_name: str | None = None
    album_name: str | None = None
    duration: float | None = None
    language: str | None = None


class LyricsLookupError(RuntimeError):
    pass


class _RateLimited(Exception):
    def __init__(self, retry_after: float) -> None:
        super().__init__(f"retry after {retry_after}")
        self.retry_after = retry_after


def parse_synced_lyrics(value: str | None) -> tuple[LyricLine, ...]:
    if not value:
        return ()
    lines: list[LyricLine] = []
    for raw_line in value.splitlines():
        matches = list(_TIMESTAMP_RE.finditer(raw_line))
        if not matches:
            continue
        text = _TIMESTAMP_RE.sub("", raw_line).strip()
        for match in matches:
            minutes = int(match.group(1))
            seconds = float(match.group(2))
            lines.append(LyricLine(minutes * 60 + seconds, text))
    lines.sort(key=lambda line: line.start_seconds)
    return tuple(lines)


def parse_youtube_json3(value: str) -> tuple[LyricLine, ...]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return ()
    events = payload.get("events") if isinstance(payload, dict) else None
    if not isinstance(events, list):
        return ()
    lines: list[LyricLine] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        start_ms = event.get("tStartMs")
        segments = event.get("segs")
        if not isinstance(start_ms, (int, float)) or not isinstance(segments, list):
            continue
        text = "".join(
            str(segment.get("utf8") or "")
            for segment in segments
            if isinstance(segment, dict)
        )
        text = _clean_caption_text(text)
        _append_caption_line(lines, float(start_ms) / 1000.0, text)
    return tuple(lines)


def parse_youtube_vtt(value: str) -> tuple[LyricLine, ...]:
    lines: list[LyricLine] = []
    rows = value.replace("\r\n", "\n").split("\n")
    index = 0
    while index < len(rows):
        match = _VTT_TIME_RE.search(rows[index])
        if not match:
            index += 1
            continue
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2))
        seconds = float(match.group(3))
        start = hours * 3600 + minutes * 60 + seconds
        index += 1
        text_rows: list[str] = []
        while index < len(rows) and rows[index].strip():
            row = rows[index].strip()
            if not row.startswith("NOTE"):
                text_rows.append(row)
            index += 1
        text = _clean_caption_text(" ".join(text_rows))
        _append_caption_line(lines, start, text)
        index += 1
    return tuple(lines)


def current_line_index(lines: tuple[LyricLine, ...], position_seconds: float) -> int | None:
    if not lines:
        return None
    starts = [line.start_seconds for line in lines]
    index = bisect.bisect_right(starts, max(position_seconds, 0.0)) - 1
    return index if index >= 0 else None


def _clean_caption_text(value: str) -> str:
    value = html.unescape(_TAG_RE.sub("", value))
    return _WHITESPACE_RE.sub(" ", value.replace("\n", " ")).strip()


def _append_caption_line(lines: list[LyricLine], start: float, text: str) -> None:
    if not text:
        return
    if lines and lines[-1].text == text:
        return
    lines.append(LyricLine(max(0.0, start), text))


def _clean_display_title(display_title: str) -> str:
    cleaned = _DECORATION_RE.sub(" ", display_title)
    cleaned = _TRAILING_DECORATION_RE.sub("", cleaned)
    return _WHITESPACE_RE.sub(" ", cleaned).strip(" -|｜•·:")


def infer_title_artist(display_title: str) -> tuple[str, str | None]:
    cleaned = _clean_display_title(display_title)

    bracket = _TITLE_BRACKET_RE.search(cleaned)
    if bracket:
        title = _WHITESPACE_RE.sub(" ", bracket.group(1)).strip(" -|｜•·:")
        artist = cleaned[: bracket.start()].strip(" -|｜•·:")
        if title:
            return title, artist or None

    for separator in (" - ", " – ", " — "):
        if separator in cleaned:
            artist, title = cleaned.split(separator, 1)
            if artist.strip() and title.strip():
                return title.strip(), artist.strip()
    return cleaned or display_title.strip(), None


def _script_kind(token: str) -> str:
    if _EAST_ASIAN_RE.search(token):
        return "east"
    if _LATIN_RE.search(token):
        return "latin"
    return "other"


def _language_variants(value: str | None) -> list[str]:
    if not value:
        return []
    cleaned = _WHITESPACE_RE.sub(" ", value).strip(" -|｜•·:")
    if not cleaned:
        return []

    variants: list[str] = [cleaned]
    for part in re.split(r"\s*[|｜/]\s*", cleaned):
        part = part.strip()
        if part and part not in variants:
            variants.append(part)

    tokens = cleaned.split()
    groups: list[tuple[str, list[str]]] = []
    for token in tokens:
        kind = _script_kind(token)
        if kind == "other" and groups:
            groups[-1][1].append(token)
        elif groups and groups[-1][0] == kind:
            groups[-1][1].append(token)
        else:
            groups.append((kind, [token]))
    if len(groups) > 1:
        for kind, group_tokens in groups:
            if kind == "other":
                continue
            variant = " ".join(group_tokens).strip(" -|｜•·:")
            if variant and variant not in variants:
                variants.append(variant)
    return variants


def search_candidates(display_title: str) -> tuple[tuple[str, str | None], ...]:
    title, artist = infer_title_artist(display_title)
    titles = _language_variants(title) or [title]
    artists = _language_variants(artist)

    candidates: list[tuple[str, str | None]] = []

    def add(candidate_title: str, candidate_artist: str | None) -> None:
        candidate_title = candidate_title.strip()
        candidate_artist = candidate_artist.strip() if candidate_artist else None
        item = (candidate_title, candidate_artist or None)
        if candidate_title and item not in candidates:
            candidates.append(item)

    if artists:
        add(titles[0], artists[0])
        for candidate_title in titles:
            for candidate_artist in artists:
                add(candidate_title, candidate_artist)
            add(candidate_title, None)
    else:
        for candidate_title in titles:
            add(candidate_title, None)

    cleaned = _clean_display_title(display_title)
    if cleaned and cleaned != title:
        add(cleaned, None)
    return tuple(candidates)


def _normalize_for_match(value: str) -> str:
    value = value.casefold()
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return _WHITESPACE_RE.sub(" ", value).strip()


def _similarity(left: str, right: str) -> float:
    left_norm = _normalize_for_match(left)
    right_norm = _normalize_for_match(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def _youtube_url(track: Track) -> str | None:
    for value in (track.original_name, track.stored_name):
        if "youtube.com" in value or "youtu.be" in value:
            return value
    return None


class LyricsProvider:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._last_request_at = 0.0
        self._cache: dict[str, LyricsResult | None] = {}

    async def fetch_for_track(self, track: Track) -> LyricsResult | None:
        cache_key = _normalize_for_match(
            "|".join(
                str(value or "")
                for value in (
                    track.original_name,
                    track.title,
                    track.track_name,
                    track.artist_name,
                    track.album_name,
                    track.duration,
                )
            )
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        async with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]

            metadata = TrackMetadata(
                track_name=track.track_name,
                artist_name=track.artist_name,
                album_name=track.album_name,
                duration=track.duration,
            )
            youtube_info: dict[str, Any] | None = None
            youtube_url = _youtube_url(track)
            if youtube_url and (
                not metadata.track_name
                or not metadata.artist_name
                or metadata.duration is None
            ):
                youtube_info = await self._youtube_info(youtube_url)
                metadata = self._merge_youtube_metadata(metadata, youtube_info)

            if metadata.track_name and metadata.artist_name:
                exact = await self._request_exact(metadata)
                if exact is not None:
                    result = self._record_to_result(exact, source="LRCLIB exact")
                    self._remember(cache_key, result)
                    return result

            candidates = self._track_candidates(track, metadata)
            for track_name, artist_name in candidates[:MAX_STRUCTURED_SEARCHES]:
                records = await self._request_records(
                    track_name=track_name,
                    artist_name=artist_name,
                    album_name=metadata.album_name,
                )
                result = self._select_best(
                    records,
                    track_name,
                    artist_name,
                    album_name=metadata.album_name,
                    duration=metadata.duration,
                )
                if result is not None:
                    self._remember(cache_key, result)
                    return result

            seen_queries: set[str] = set()
            for track_name, artist_name in candidates:
                query = " ".join(part for part in (track_name, artist_name) if part).strip()
                normalized_query = _normalize_for_match(query)
                if not normalized_query or normalized_query in seen_queries:
                    continue
                seen_queries.add(normalized_query)
                records = await self._request_records(query=query)
                result = self._select_best(
                    records,
                    track_name,
                    artist_name,
                    album_name=metadata.album_name,
                    duration=metadata.duration,
                )
                if result is not None:
                    self._remember(cache_key, result)
                    return result
                if len(seen_queries) >= MAX_KEYWORD_SEARCHES:
                    break

            if youtube_url:
                if youtube_info is None:
                    youtube_info = await self._youtube_info(youtube_url)
                caption_result = await self._youtube_caption_result(track, metadata, youtube_info)
                if caption_result is not None:
                    self._remember(cache_key, caption_result)
                    return caption_result

            self._remember(cache_key, None)
            return None

    async def fetch_for_title(self, display_title: str) -> LyricsResult | None:
        track = Track(
            id=-1,
            title=display_title,
            original_name=display_title,
            stored_name=display_title,
            uploaded_by=0,
        )
        return await self.fetch_for_track(track)

    async def fetch(self, track_name: str, artist_name: str | None = None) -> LyricsResult | None:
        display = f"{artist_name} - {track_name}" if artist_name else track_name
        return await self.fetch_for_title(display)

    def _track_candidates(
        self, track: Track, metadata: TrackMetadata
    ) -> tuple[tuple[str, str | None], ...]:
        candidates: list[tuple[str, str | None]] = []

        def add(title: str | None, artist: str | None) -> None:
            if not title or not title.strip():
                return
            item = (title.strip(), artist.strip() if artist and artist.strip() else None)
            if item not in candidates:
                candidates.append(item)

        if metadata.track_name:
            title_variants = _language_variants(metadata.track_name) or [metadata.track_name]
            artist_variants = _language_variants(metadata.artist_name)
            if artist_variants:
                for title in title_variants:
                    for artist in artist_variants:
                        add(title, artist)
                    add(title, None)
            else:
                for title in title_variants:
                    add(title, None)

        for title, artist in search_candidates(track.title):
            add(title, artist)
        return tuple(candidates)

    def _remember(self, key: str, result: LyricsResult | None) -> None:
        self._cache[key] = result
        if len(self._cache) > 128:
            self._cache.pop(next(iter(self._cache)))

    async def _request_exact(self, metadata: TrackMetadata) -> dict[str, object] | None:
        assert metadata.track_name and metadata.artist_name
        params: dict[str, str] = {
            "track_name": metadata.track_name,
            "artist_name": metadata.artist_name,
        }
        if metadata.album_name:
            params["album_name"] = metadata.album_name
        if metadata.duration is not None and 1 <= metadata.duration <= 3600:
            params["duration"] = str(round(metadata.duration, 2))
        await self._throttle()
        try:
            try:
                return await asyncio.to_thread(self._get_exact, params)
            except _RateLimited as error:
                await asyncio.sleep(max(error.retry_after, REQUEST_INTERVAL_SECONDS))
                return await asyncio.to_thread(self._get_exact, params)
        finally:
            self._last_request_at = time.monotonic()

    def _get_exact(self, params: dict[str, str]) -> dict[str, object] | None:
        request = Request(
            f"{LRCLIB_GET_URL}?{urlencode(params)}",
            headers={"User-Agent": LRCLIB_USER_AGENT, "Accept": "application/json"},
        )
        try:
            with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            if error.code == 404:
                return None
            if error.code == 429:
                retry_after = error.headers.get("Retry-After", "1")
                try:
                    delay = float(retry_after)
                except ValueError:
                    delay = 1.0
                raise _RateLimited(delay) from error
            raise LyricsLookupError(f"LRCLIB HTTP {error.code}") from error
        except (URLError, TimeoutError, json.JSONDecodeError) as error:
            raise LyricsLookupError(f"LRCLIB exact lookup failed: {error}") from error
        return payload if isinstance(payload, dict) else None

    async def _request_records(
        self,
        *,
        track_name: str | None = None,
        artist_name: str | None = None,
        album_name: str | None = None,
        query: str | None = None,
    ) -> list[dict[str, object]]:
        await self._throttle()
        try:
            try:
                return await asyncio.to_thread(
                    self._search, track_name, artist_name, album_name, query
                )
            except _RateLimited as error:
                await asyncio.sleep(max(error.retry_after, REQUEST_INTERVAL_SECONDS))
                return await asyncio.to_thread(
                    self._search, track_name, artist_name, album_name, query
                )
        finally:
            self._last_request_at = time.monotonic()

    async def _throttle(self) -> None:
        remaining = REQUEST_INTERVAL_SECONDS - (time.monotonic() - self._last_request_at)
        if remaining > 0:
            await asyncio.sleep(remaining)

    def _search(
        self,
        track_name: str | None,
        artist_name: str | None,
        album_name: str | None,
        query: str | None,
    ) -> list[dict[str, object]]:
        if query:
            params: dict[str, str] = {"q": query}
        elif track_name:
            params = {"track_name": track_name}
            if artist_name:
                params["artist_name"] = artist_name
            if album_name:
                params["album_name"] = album_name
        else:
            return []
        request = Request(
            f"{LRCLIB_SEARCH_URL}?{urlencode(params)}",
            headers={"User-Agent": LRCLIB_USER_AGENT, "Accept": "application/json"},
        )
        try:
            with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            if error.code == 429:
                retry_after = error.headers.get("Retry-After", "1")
                try:
                    delay = float(retry_after)
                except ValueError:
                    delay = 1.0
                raise _RateLimited(delay) from error
            raise LyricsLookupError(f"LRCLIB HTTP {error.code}") from error
        except (URLError, TimeoutError, json.JSONDecodeError) as error:
            raise LyricsLookupError(f"LRCLIB lookup failed: {error}") from error
        if not isinstance(payload, list):
            raise LyricsLookupError("LRCLIB returned an unexpected response")
        return [record for record in payload if isinstance(record, dict)]

    @staticmethod
    def _select_best(
        records: list[dict[str, object]],
        track_name: str,
        artist_name: str | None,
        *,
        album_name: str | None = None,
        duration: float | None = None,
    ) -> LyricsResult | None:
        if not records:
            return None

        def score(record: dict[str, object]) -> float:
            candidate_title = str(record.get("trackName") or record.get("name") or "")
            candidate_artist = str(record.get("artistName") or "")
            candidate_album = str(record.get("albumName") or "")
            value = _similarity(track_name, candidate_title) * 8.0
            if artist_name:
                value += _similarity(artist_name, candidate_artist) * 5.0
            if album_name and candidate_album:
                value += _similarity(album_name, candidate_album) * 2.0
            candidate_duration = record.get("duration")
            if duration is not None and isinstance(candidate_duration, (int, float)):
                difference = abs(float(candidate_duration) - duration)
                if difference <= 2:
                    value += 2.0
                elif difference <= 8:
                    value += 0.75
                elif difference > 30:
                    value -= 1.5
            if record.get("syncedLyrics"):
                value += 1.0
            if record.get("instrumental"):
                value -= 0.25
            return value

        best = max(records, key=score)
        if score(best) < 3.0:
            return None
        return LyricsProvider._record_to_result(best, source="LRCLIB search")

    @staticmethod
    def _record_to_result(record: dict[str, object], *, source: str) -> LyricsResult:
        synced = record.get("syncedLyrics")
        plain = record.get("plainLyrics")
        duration = record.get("duration")
        return LyricsResult(
            track_name=str(record.get("trackName") or record.get("name") or "Unknown Track"),
            artist_name=str(record.get("artistName") or ""),
            album_name=str(record.get("albumName") or ""),
            duration=float(duration) if isinstance(duration, (int, float)) else None,
            instrumental=bool(record.get("instrumental")),
            plain_lyrics=str(plain) if isinstance(plain, str) and plain else None,
            synced_lyrics=str(synced) if isinstance(synced, str) and synced else None,
            lines=parse_synced_lyrics(str(synced) if isinstance(synced, str) else None),
            source=source,
        )

    async def _youtube_info(self, url: str) -> dict[str, Any] | None:
        if yt_dlp is None:
            return None

        def extract() -> dict[str, Any] | None:
            options = {
                "quiet": True,
                "no_warnings": True,
                "skip_download": True,
                "noplaylist": True,
            }
            try:
                with yt_dlp.YoutubeDL(options) as downloader:
                    info = downloader.extract_info(url, download=False)
            except Exception:
                return None
            return info if isinstance(info, dict) else None

        return await asyncio.to_thread(extract)

    @staticmethod
    def _merge_youtube_metadata(
        current: TrackMetadata, info: dict[str, Any] | None
    ) -> TrackMetadata:
        if not info:
            return current

        def text(key: str) -> str | None:
            value = info.get(key)
            return value.strip() if isinstance(value, str) and value.strip() else None

        duration_value = info.get("duration")
        duration = (
            float(duration_value)
            if isinstance(duration_value, (int, float))
            else current.duration
        )
        return TrackMetadata(
            track_name=current.track_name or text("track") or text("alt_title"),
            artist_name=current.artist_name or text("artist") or text("creator"),
            album_name=current.album_name or text("album"),
            duration=duration,
            language=text("language"),
        )

    async def _youtube_caption_result(
        self,
        track: Track,
        metadata: TrackMetadata,
        info: dict[str, Any] | None,
    ) -> LyricsResult | None:
        if not info:
            return None
        selected = self._select_caption(info, track, metadata)
        if selected is None:
            return None
        source_name, language, caption = selected
        body = await asyncio.to_thread(self._download_caption, caption)
        if body is None:
            return None
        extension = str(caption.get("ext") or "").lower()
        if extension == "json3":
            lines = parse_youtube_json3(body)
        elif extension == "vtt":
            lines = parse_youtube_vtt(body)
        else:
            return None
        if len(lines) < 2:
            return None
        inferred_title, inferred_artist = infer_title_artist(track.title)
        return LyricsResult(
            track_name=metadata.track_name or inferred_title,
            artist_name=metadata.artist_name or inferred_artist or "",
            album_name=metadata.album_name or "",
            duration=metadata.duration,
            instrumental=False,
            plain_lyrics=None,
            synced_lyrics=None,
            lines=lines,
            source=f"YouTube {source_name} ({language})",
        )

    def _select_caption(
        self,
        info: dict[str, Any],
        track: Track,
        metadata: TrackMetadata,
    ) -> tuple[str, str, dict[str, Any]] | None:
        priorities = self._caption_language_priorities(track, metadata)
        for source_key, source_name in (
            ("subtitles", "subtitles"),
            ("automatic_captions", "auto captions"),
        ):
            collection = info.get(source_key)
            if not isinstance(collection, dict) or not collection:
                continue
            languages = self._ordered_languages(collection, priorities)
            for language in languages:
                formats = collection.get(language)
                if not isinstance(formats, list):
                    continue
                caption = self._preferred_caption_format(formats)
                if caption is not None:
                    return source_name, language, caption
        return None

    @staticmethod
    def _ordered_languages(
        collection: dict[str, Any], priorities: list[str]
    ) -> list[str]:
        available = [str(key) for key in collection]
        ordered: list[str] = []
        for preferred in priorities:
            for language in available:
                if language == preferred or language.lower() == preferred.lower():
                    if language not in ordered:
                        ordered.append(language)
            base = preferred.split("-", 1)[0].lower()
            for language in available:
                if language.split("-", 1)[0].lower() == base and language not in ordered:
                    ordered.append(language)
        for language in available:
            if language not in ordered and not language.endswith("-translated"):
                ordered.append(language)
        return ordered

    @staticmethod
    def _preferred_caption_format(formats: list[Any]) -> dict[str, Any] | None:
        valid = [item for item in formats if isinstance(item, dict)]
        for extension in ("json3", "vtt"):
            for item in valid:
                if str(item.get("ext") or "").lower() == extension and (
                    isinstance(item.get("data"), str) or isinstance(item.get("url"), str)
                ):
                    return item
        return None

    @staticmethod
    def _download_caption(caption: dict[str, Any]) -> str | None:
        data = caption.get("data")
        if isinstance(data, str):
            return data
        url = caption.get("url")
        if not isinstance(url, str) or not url:
            return None
        headers = {"User-Agent": LRCLIB_USER_AGENT}
        supplied_headers = caption.get("http_headers")
        if isinstance(supplied_headers, dict):
            headers.update(
                {str(key): str(value) for key, value in supplied_headers.items()}
            )
        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                return response.read().decode("utf-8", errors="replace")
        except (HTTPError, URLError, TimeoutError):
            return None

    @staticmethod
    def _caption_language_priorities(
        track: Track, metadata: TrackMetadata
    ) -> list[str]:
        priorities: list[str] = []

        def add(value: str | None) -> None:
            if value and value not in priorities:
                priorities.append(value)

        add(metadata.language)
        combined = " ".join(
            value
            for value in (metadata.track_name, metadata.artist_name, track.title)
            if value
        )
        if _CJK_RE.search(combined):
            for language in ("zh-TW", "zh-Hant", "zh-HK", "zh-Hans", "zh"):
                add(language)
        if _JAPANESE_RE.search(combined):
            add("ja")
        if _KOREAN_RE.search(combined):
            add("ko")
        if _LATIN_RE.search(combined):
            add("en")
        return priorities
