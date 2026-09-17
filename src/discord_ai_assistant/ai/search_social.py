from __future__ import annotations

from discord_ai_assistant.ai.gemini import (
    GOOGLE_SEARCH_TOOL,
    TOOL_REQUEST_TIMEOUT_SECONDS,
    GeminiAssistant,
)


async def search_only_social_reply(
    ai: GeminiAssistant,
    prompt: str,
    *,
    persona_instruction: str,
) -> str:
    """Run a proactive social lookup with Google Search as the only capability.

    This path intentionally bypasses GeminiAssistant.ask(), ToolRouter refresh, and
    external tool-gateway declarations.  A prompt therefore cannot escalate a
    proactive current-fact lookup into Discord/music/meeting actions.
    """

    if not ai.api_key:
        raise RuntimeError("尚未設定 GEMINI_API_KEY。")

    instruction = (
        "This is a search-only proactive Discord response. Google Search is the only available tool. "
        "Do not claim to send messages, change Discord state, play music, control meetings/capture, "
        "write memories, or perform any other external action. Use search for current public facts. "
        "If the current fact cannot be verified from useful search results, output only NO_REPLY. "
        "When search citations are available, keep the factual answer concise; source URLs will be appended automatically.\n"
        f"The current server time is {ai._current_time_text()}; use it instead of guessing the date/time."
    )
    interaction = await ai._create_interaction(
        ai._get_client(),
        model=ai.model,
        system_instruction=ai._system_instruction(persona_instruction, instruction),
        input=[{"type": "text", "text": f"<untrusted_discord_input>\n{prompt}\n</untrusted_discord_input>"}],
        tools=[dict(GOOGLE_SEARCH_TOOL)],
        timeout_seconds=TOOL_REQUEST_TIMEOUT_SECONDS,
        request_kind="social-search",
        input_characters=len(prompt),
    )
    return ai._format_response(interaction, "")
