from __future__ import annotations

NO_REPLY_CONTROL = "NO_REPLY"
_NO_REPLY_TRIM_CHARS = " \t\r\n.。!！?？~～…"
_NO_REPLY_PERSONA_SUFFIXES = {
    "",
    "喵",
    "🐱",
    "😺",
    "😸",
    "😽",
    "喵🐱",
    "喵😺",
    "喵😸",
    "喵😽",
}


def normalize_control_response(text: str) -> str:
    """Collapse persona-styled NO_REPLY control values back to the exact sentinel.

    The first output line is authoritative for the control decision so a search response
    such as ``NO_REPLY喵`` followed by auto-appended citations is still suppressed.
    Ordinary text that merely mentions ``NO_REPLY`` is preserved.
    """

    value = text.strip()
    if not value:
        return value
    first_line = value.splitlines()[0].strip()
    upper = first_line.upper()
    if not upper.startswith(NO_REPLY_CONTROL):
        return text
    suffix = first_line[len(NO_REPLY_CONTROL) :].strip(_NO_REPLY_TRIM_CHARS)
    if suffix in _NO_REPLY_PERSONA_SUFFIXES:
        return NO_REPLY_CONTROL
    return text
