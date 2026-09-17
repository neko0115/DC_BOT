from __future__ import annotations

from datetime import timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def resolve_timezone(name: str) -> tzinfo:
    """Resolve an IANA zone, including Taiwan on Windows without tzdata installed."""
    if name == "Asia/Taipei":
        return timezone(timedelta(hours=8), name="Asia/Taipei")
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError as error:
        raise ValueError("PERSONA_TIMEZONE must be a valid IANA timezone, such as Asia/Taipei.") from error
