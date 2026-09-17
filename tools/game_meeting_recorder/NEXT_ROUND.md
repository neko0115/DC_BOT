# Meeting Review V2 — Next Round Plan

> Status: **PLANNING ONLY / NOT IMPLEMENTED**
>
> Frozen baseline: `agent/game-meeting-recorder` at `fd63204b2a20cfb2a7af24b52b713e7a6435993f`.
> All V2 work must branch from this verified baseline and must not retroactively change the completed V1 status.

## Goal

Improve the human-review and learning loop without weakening the current safety rule: reusable knowledge is learned only from human-reviewed material, one-off meeting facts must not become long-term knowledge, and every new capability must distinguish **implemented**, **automated-tested**, and **live E2E validated** states.

## P0 — Next implementation targets

### 1. Per-candidate learning selection

Current behavior is all-or-nothing: `套用學習並重整` applies every proposed candidate.

Planned UX:
- Show up to the existing bounded candidate count in a Discord multi-select control.
- Reviewer explicitly selects which `asr_alias`, `domain_term`, and `report_preference` candidates may be learned.
- Actions:
  - `套用已選項目並重整`
  - `只重整，不學習`
  - `取消`
- Persist a per-candidate result such as approved/rejected/skipped instead of treating one revision as a single learning decision.
- Never learn unselected candidates.

Acceptance criteria:
- Candidate selection survives regeneration/retry safely.
- A failed Gemini regeneration must not lose the reviewer selection or duplicate learning writes.
- Tests prove selected candidates are applied and unselected candidates are not.

### 2. Natural-language audio attachment entrypoint

Target UX:

`@墨雪 這份幫我整理成週會報` + supported audio attachment

Design constraint:
- Integrate routing into the existing message-handling path before generic AI handling.
- Do **not** add an independent competing `on_message` listener that can cause duplicate replies.
- Require explicit meeting/report intent plus a supported audio attachment.
- Do not hijack ordinary music/audio uploads.
- Preserve DJ/admin permission checks.
- Reuse the same imported-audio pipeline as `/meeting_report`; do not fork transcription/report logic.

Acceptance criteria:
- One user request produces one meeting workflow and no duplicate general-AI reply.
- Unsupported or ambiguous audio remains ordinary chat/audio behavior.

### 3. Explicit meeting date and start time

Current report records generation time and audio duration, but generation time is not necessarily the real meeting time.

Planned model:
- Persist optional `meeting_date`, `meeting_start_time`, and timezone separately from report-generation time.
- `/meeting_report` can accept optional meeting date/time.
- Context-menu/natural-language entry may leave meeting time as `未指定` unless the user explicitly supplies it.
- Never ask Gemini to guess a meeting timestamp from unrelated context.
- Keep `generated_at` as a separate audit field.

Acceptance criteria:
- Old recordings can be correctly labeled with their actual meeting date.
- Missing date/time is rendered as unspecified rather than fabricated.

## P1 — Follow-up enhancements

### 4. Knowledge Profile management UX

Improve App/Game terminology maintenance beyond one-term-at-a-time commands:
- profile/term autocomplete where practical;
- bulk JSON/CSV import/export for canonical term + explanation + aliases;
- preview/validation before import;
- duplicate/conflict reporting;
- preserve guild isolation and user-edited values.

### 5. Speaker attribution / diarization evaluation

Do not claim diarization support until measured on real meeting audio.

Investigation order:
1. Preserve known speaker/source labels when audio is captured as separate Discord/source tracks.
2. Evaluate an optional local diarization backend for mixed MP3 recordings.
3. Keep diarization optional because GPU/model/token requirements may be substantially heavier than faster-whisper alone.
4. Report `unknown speaker` when confidence is insufficient instead of inventing identities.

Required validation:
- real mixed-speaker fixture;
- speaker-attribution error measurement;
- performance/runtime measurement on the intended local machine.

### 6. Local Whisper fine-tuning pipeline

V1 already stores reviewed ASR examples and exports a manifest. V2 may build a separate offline training pipeline; the running Discord Bot must not train itself automatically.

Planned stages:
1. Materialize reviewed timestamp ranges into deterministic audio clips.
2. Build train/validation/test splits grouped by recording to prevent leakage.
3. Record baseline Whisper CER/WER on a held-out set.
4. Fine-tune through the standard PyTorch/Transformers training path.
5. Re-evaluate on the same held-out set.
6. Convert an accepted checkpoint to CTranslate2 for faster-whisper inference.
7. Register models by Knowledge Profile only after explicit human approval.

Deployment gate:
- no automatic replacement of the production ASR model;
- a candidate model must beat the predefined baseline/tolerance on held-out data;
- report sample count, CER/WER, regression cases, model/checkpoint provenance, and runtime cost;
- if evidence is insufficient, status remains `NOT VALIDATED` rather than PASS.

## Suggested implementation order

1. Per-candidate selection.
2. Natural-language MP3 routing.
3. Explicit meeting date/time metadata.
4. Knowledge DB bulk-management UX.
5. Diarization feasibility experiment.
6. Offline Whisper fine-tuning and evaluation pipeline after enough reviewed data exists.

## V1 closure baseline

The following are treated as completed V1 capabilities and should be regression-tested rather than redesigned casually:
- local faster-whisper transcription;
- App/Game Knowledge Profiles with explanations and aliases;
- transcript-only factual grounding for meeting reports;
- full-article human correction submission;
- learning-candidate proposal with human confirmation;
- reviewed feedback/training-data persistence;
- draft review and configured-channel publishing;
- meeting-specific Gemini timeout and recoverable retry path;
- original transcript/report preservation and audit artifacts.
