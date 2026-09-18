from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import NamedTuple

MAX_TOOL_ARTIFACT_BYTES = 10 * 1024 * 1024
ALLOWED_ARTIFACT_TYPES: dict[str, frozenset[str]] = {
    "image/jpeg": frozenset({".jpg", ".jpeg"}),
    "image/png": frozenset({".png"}),
    "image/webp": frozenset({".webp"}),
}


class ResolvedArtifact(NamedTuple):
    path: Path
    filename: str
    mime_type: str
    delete_after_send: bool


def _safe_filename(value: object, fallback: str, allowed_suffixes: frozenset[str]) -> str:
    if isinstance(value, str) and value.strip():
        candidate = PurePosixPath(value.replace("\\", "/")).name[:180]
    else:
        candidate = fallback
    if not candidate or Path(candidate).suffix.lower() not in allowed_suffixes:
        return fallback
    return candidate


def _matches_image_magic(path: Path, mime_type: str) -> bool:
    try:
        header = path.read_bytes()[:12]
    except OSError:
        return False
    if mime_type == "image/jpeg":
        return header.startswith(b"\xff\xd8\xff")
    if mime_type == "image/png":
        return header.startswith(b"\x89PNG\r\n\x1a\n")
    if mime_type == "image/webp":
        return len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP"
    return False


def resolve_artifact_effect(
    project_root: Path,
    max_upload_bytes: int,
    effect: dict[str, object],
) -> ResolvedArtifact | None:
    """Resolve one core-owned artifact effect without allowing path or symlink escape."""
    if effect.get("type") != "attach_artifact":
        return None
    relative_path = effect.get("relative_path")
    mime_type = effect.get("mime_type")
    if not isinstance(relative_path, str) or not relative_path or not isinstance(mime_type, str):
        return None
    allowed_suffixes = ALLOWED_ARTIFACT_TYPES.get(mime_type)
    if not allowed_suffixes:
        return None

    normalized = PurePosixPath(relative_path.replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts or not normalized.parts:
        return None

    root = (project_root / "data" / "tool-artifacts").resolve()
    current = root
    try:
        for part in normalized.parts:
            current = current / part
            if current.is_symlink():
                return None
        path = current.resolve(strict=True)
        path.relative_to(root)
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        return None

    if not path.is_file() or path.suffix.lower() not in allowed_suffixes:
        return None
    try:
        size = path.stat().st_size
    except OSError:
        return None
    size_limit = min(max(1, int(max_upload_bytes)), MAX_TOOL_ARTIFACT_BYTES)
    if size <= 0 or size > size_limit or not _matches_image_magic(path, mime_type):
        return None

    filename = _safe_filename(effect.get("filename"), path.name, allowed_suffixes)
    delete_after_send = effect.get("delete_after_send", True)
    if not isinstance(delete_after_send, bool):
        delete_after_send = True
    return ResolvedArtifact(path, filename, mime_type, delete_after_send)
