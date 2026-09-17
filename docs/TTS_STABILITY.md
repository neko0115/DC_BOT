# TTS stability and fallback behavior

This branch keeps the trained/local Moxue GPT-SoVITS worker as the preferred voice and makes fallback less eager.

## Request flow

1. `WorkerPoolSpeechSynthesizer` checks the preferred worker.
2. A worker that reports `busy` is polled for a bounded wait before the Bot gives up on it.
3. Updated `moxue-tts-worker` accepts a bounded queue instead of rejecting every request that arrives while inference is active.
4. Only after the local worker queue is unavailable/full/timed out, or the worker genuinely fails, does the existing provider fallback run.
5. Queue saturation does **not** place a healthy worker into the normal failure cooldown.

Worker defaults:

```ini
MOXUE_TTS_MAX_CONCURRENCY=1
MOXUE_TTS_MAX_QUEUE_SIZE=8
MOXUE_TTS_QUEUE_WAIT_SECONDS=15
```

`MOXUE_TTS_GPU_BUSY_THRESHOLD` is retained for backward-compatible configuration, but high GPU utilization is now diagnostic only. GPT-SoVITS is expected to drive the GPU hard while synthesizing. An idle worker with critically low free VRAM can still report `busy`.

The Bot adds HTTP timeout grace for worker-side queueing so an accepted local request is not abandoned just because it waited briefly behind another utterance.

## Playback watchdog

`VoicePlaybackWatchdog` runs every five seconds and checks connected guild voice sessions.

It recovers these cases:

- Discord reports playback while the current track or PCM source is missing.
- FFmpeg has already exited but Discord still reports `is_playing()`.
- A TTS WAV remains in the playing state beyond its measured duration plus a safety grace period.
- The voice client is idle while upcoming audio is still queued, which can happen if a playback callback is lost.

A first stall stops the stale playback and lets the normal callback advance the queue. A second stall within one minute also arms the existing voice reconnect path before the next queued item starts.

## Diagnostics

`/tts_compute_status` now shows worker `active` and `queued/max` counts when the worker exposes them, alongside GPU utilization and free VRAM.

Windows SAPI remains the final emergency fallback. Under normal operation it should be rare; logs should explain which local/provider route failed before SAPI is selected.
