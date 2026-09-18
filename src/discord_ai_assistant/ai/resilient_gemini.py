from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

from discord_ai_assistant.ai.control_response import normalize_control_response
from discord_ai_assistant.ai.gemini import (
    GeminiAssistant, GeminiRequestError, _send_interaction, _single_attempt_diagnostics,
)
from discord_ai_assistant.ai.key_pool import FailoverGeminiClient, load_gemini_api_keys
from discord_ai_assistant.ai.model_router import (
    GeminiModelRouter,
    infer_workload,
    is_model_fallback_error,
    load_model_routes,
)
from discord_ai_assistant.ai.tools import ToolRouter

LOGGER = logging.getLogger(__name__)


class ResilientGeminiAssistant(GeminiAssistant):
    """Gemini assistant with API-key failover plus workload/model failover."""

    def __init__(
        self,
        api_key: str | None,
        model: str,
        router: ToolRouter,
        timezone_name: str = "Asia/Taipei",
    ) -> None:
        super().__init__(api_key, model, router, timezone_name)
        self.api_keys = load_gemini_api_keys(api_key)
        self.api_key = self.api_keys[0] if self.api_keys else None
        self.model_router = GeminiModelRouter(load_model_routes(model))

    @property
    def enabled(self) -> bool:
        return bool(self.api_keys)

    def _get_client(self) -> Any:
        if not self.api_keys:
            raise RuntimeError("尚未設定 GEMINI_API_KEY。")
        if self._client is None:
            self._client = FailoverGeminiClient(self.api_keys)
        return self._client

    async def _create_interaction_once(
        self,
        client: Any,
        *,
        timeout_seconds: int,
        request_kind: str,
        input_characters: int,
        **kwargs: object,
    ) -> Any:
        """Select one available model/key without retrying this interaction.

        Keep normal routing and continuation affinity. Failures update the shared
        route/key state for the next request, never continue to another candidate.
        """
        previous_id = kwargs.get("previous_interaction_id")
        previous_interaction_id = previous_id if isinstance(previous_id, str) else None
        pinned_model = self.model_router.model_for_interaction(previous_interaction_id)
        requested_model = kwargs.get("model")
        explicit_model = requested_model if isinstance(requested_model, str) and requested_model else None
        workload = infer_workload(
            request_kind, input_characters=input_characters,
            input_payload=kwargs.get("input"), tools=kwargs.get("tools"),
        )
        if pinned_model is not None:
            candidates = (pinned_model,)
        elif explicit_model and explicit_model != self.model and previous_interaction_id is None:
            candidates = (explicit_model,)
        else:
            candidates = self.model_router.candidate_models(workload)
        if not candidates:
            raise GeminiRequestError(self.model_router.unavailable_message(workload))
        candidate_model = candidates[0]
        request_options = {**kwargs, "model": candidate_model}
        # Adapt only this invocation. The shared client's ordinary create still
        # retries keys; the base timeout machinery sees a single-key operation.
        once_client = SimpleNamespace(interactions=SimpleNamespace(create=client.interactions.create_once))
        try:
            with _single_attempt_diagnostics():
                interaction = await _send_interaction(once_client, timeout_seconds=timeout_seconds, **request_options)
        except GeminiRequestError as error:
            if is_model_fallback_error(error):
                self.model_router.mark_failure(workload, candidate_model, error)
            raise
        self.model_router.mark_success(candidate_model)
        self.model_router.remember_interaction(interaction, candidate_model)
        return interaction

    async def _create_interaction(
        self,
        client: Any,
        *,
        timeout_seconds: int,
        request_kind: str,
        input_characters: int,
        **kwargs: object,
    ) -> Any:
        """Try the workload's model chain without replaying stateful continuations across models."""

        previous_id = kwargs.get("previous_interaction_id")
        previous_interaction_id = previous_id if isinstance(previous_id, str) else None
        pinned_model = self.model_router.model_for_interaction(previous_interaction_id)
        requested_model = kwargs.get("model")
        explicit_model = requested_model if isinstance(requested_model, str) and requested_model else None
        workload = infer_workload(
            request_kind,
            input_characters=input_characters,
            input_payload=kwargs.get("input"),
            tools=kwargs.get("tools"),
        )

        # A caller that deliberately supplies a model different from the configured
        # legacy/default model keeps that explicit choice. Normal production callers
        # pass self.model, which is replaced by the workload route below.
        explicit_locked = bool(
            explicit_model
            and explicit_model != self.model
            and previous_interaction_id is None
        )
        if pinned_model is not None:
            candidates = (pinned_model,)
        elif explicit_locked:
            candidates = (explicit_model,)  # type: ignore[arg-type]
        else:
            candidates = self.model_router.candidate_models(workload)

        if not candidates:
            raise GeminiRequestError(self.model_router.unavailable_message(workload))

        for position, candidate_model in enumerate(candidates):
            request_options = dict(kwargs)
            request_options["model"] = candidate_model
            LOGGER.info(
                "Gemini route workload=%s model=%s attempt=%s/%s",
                workload,
                candidate_model,
                position + 1,
                len(candidates),
            )
            try:
                interaction = await super()._create_interaction(
                    client,
                    timeout_seconds=timeout_seconds,
                    request_kind=request_kind,
                    input_characters=input_characters,
                    **request_options,
                )
            except GeminiRequestError as error:
                if not is_model_fallback_error(error):
                    raise
                shared_search_blocked = self.model_router.mark_failure(workload, candidate_model, error)
                if (
                    pinned_model is not None
                    or explicit_locked
                    or shared_search_blocked
                    or position == len(candidates) - 1
                ):
                    raise
                LOGGER.warning(
                    "Gemini model %s failed for workload %s; trying %s",
                    candidate_model,
                    workload,
                    candidates[position + 1],
                )
                continue

            self.model_router.mark_success(candidate_model)
            self.model_router.remember_interaction(interaction, candidate_model)
            return interaction

        raise GeminiRequestError(self.model_router.unavailable_message(workload))

    async def social_reply(self, prompt: str, *, persona_instruction: str) -> str:
        """Normalize internal passive-control values after the normal social request."""
        reply = await super().social_reply(prompt, persona_instruction=persona_instruction)
        return normalize_control_response(reply)

    async def social_reply_with_timeout(
        self,
        prompt: str,
        *,
        persona_instruction: str,
        timeout_seconds: int,
        request_kind: str = "social-long",
    ) -> str:
        """Run a no-tool Gemini request with an explicit task-specific timeout.

        Normal Discord conversation keeps the shorter timeout from ``social_reply``.
        Long-running workflows such as meeting-report regeneration can opt into a
        larger timeout without making ordinary chat wait for several minutes. The
        quota-aware router recognizes ``meeting-*`` request kinds as the quality-first
        meeting workload while preserving this caller-provided timeout.
        """
        if not self.api_key:
            raise RuntimeError("尚未設定 GEMINI_API_KEY。")
        resolved_timeout = max(1, int(timeout_seconds))
        interaction = await self._create_interaction(
            self._get_client(),
            model=self.model,
            system_instruction=self._system_instruction(
                persona_instruction,
                "This is a no-tool response. Do not perform or claim external actions or memory changes.",
            ),
            input=[{"type": "text", "text": f"<untrusted_discord_input>\n{prompt}\n</untrusted_discord_input>"}],
            timeout_seconds=resolved_timeout,
            request_kind=request_kind,
            input_characters=len(prompt),
        )
        return normalize_control_response(interaction.output_text or "")
