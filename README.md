# Moxue Discord AI Assistant

Moxue is a self-hosted Discord assistant that combines conversational AI,
Memory V2, chat-style adaptation, knowledge enrichment, meeting workflows,
voice/TTS features, music playback, and an extensible tool system.

This repository is the source-only public release. Runtime data, credentials,
model weights, checkpoints, private voice references, and deployment-specific
configuration are not included.

## Highlights

- Discord text and image conversations backed by configurable Gemini model routes.
- Memory V2 with scoped personal, shared, session, and project-oriented memory flows.
- Chat Style learning with user controls to inspect, disable, or reset learned style data.
- Knowledge enrichment with personal aids, guild lexicon support, and bounded public-term lookup.
- Meeting recording, capture, feedback, glossary, and article-oriented workflows.
- Local music playback, playlists, YouTube-assisted lookup, voice recognition, and TTS.
- Distributed TTS workers and a GPT-SoVITS integration boundary for self-hosted speech.
- Tool Gateway integrations, web research, image generation, and artifact delivery.

## Privacy model

The assistant is designed around explicit data boundaries rather than treating all
Discord content as globally shareable context.

Recent conversational context sent to AI providers is bounded. Local Memory and Chat
Style systems apply scope and safety gates before storing or reusing information.
Sensitive, instruction-like, identity-bearing, or otherwise unsafe inputs are rejected
from relevant learning and public-lookup paths.

Public-web enrichment is isolated from private Discord identity and conversation data:
only data that passes the public-query boundary is eligible for public lookup.

Meeting/Capture and distributed worker deployments are intended for the local machine,
a trusted LAN, or a VPN unless the operator adds an appropriate authenticated TLS/WSS
boundary.

See [`docs/privacy-and-safety.md`](docs/privacy-and-safety.md) for the detailed model.

## Requirements

- 64-bit Python **3.12**.
- FFmpeg available on `PATH` for Discord audio and media workflows.
- A Discord bot application and token.
- A Gemini API key for AI-backed conversation features.
- Node.js is recommended for YouTube workflows that require JavaScript challenge handling.
- Optional providers or features may require their own credentials and dependencies.

## Quick start

PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e .
Copy-Item .env.example .env
```

Edit `.env` and set at least:

```env
DISCORD_TOKEN=
GEMINI_API_KEY=
```

Keep real credentials only in `.env`. Then start the bot:

```powershell
discord-ai-assistant
```

Leave `DISCORD_GUILD_ID` empty for global slash-command registration, or set a
development/test guild ID when intentionally using guild-scoped command sync.

## Optional features

Meeting and Capture dependencies:

```powershell
pip install -e ".[meeting]"
```

Local Kokoro/TTS dependencies:

```powershell
pip install -e ".[tts-local]"
```

Capture Agent packaging dependencies:

```powershell
pip install -e ".[capture-build]"
```

Web research and image generation require the corresponding provider credentials in
`.env.example`. Distributed TTS workers require an authenticated trusted-network
deployment.

Optional Moxue Hub status heartbeat:

```env
MOXUE_HEARTBEAT_URL=https://moxueneko.com/api/heartbeat
MOXUE_HEARTBEAT_TOKEN=<shared secret configured in Cloudflare>
MOXUE_HEARTBEAT_INTERVAL_SECONDS=60
```

The heartbeat is outbound-only and reports only service readiness, package version,
and process uptime. The Bot does not open a public health port.

GPT-SoVITS itself is external to this repository. GPT-SoVITS weights, checkpoints,
reference audio, private voice assets, and model files are **not included**.

## Documentation

- [Traditional Chinese usage guide](docs/USAGE.zh-TW.md)
- [Privacy and safety](docs/privacy-and-safety.md)
- [Security policy](SECURITY.md)
- [Public roadmap](ROADMAP.md)
- [Engineering history](docs/engineering-history.md)
- [Memory V2 notes](docs/MEMORY_V2.md)
- [Capture Agent](docs/CAPTURE_AGENT.md)
- [TTS stability notes](docs/TTS_STABILITY.md)

## Security

Do not commit credentials, tokens, cookies, private runtime data, model assets, or
deployment-specific secrets. Loopback is the default for local service endpoints.

See [`SECURITY.md`](SECURITY.md) for supported versions, private vulnerability
reporting, network-boundary guidance, and the third-party security boundary.

## License

The project source is available under the [MIT License](LICENSE).

Dependencies and optional runtimes retain their own licenses. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) before redistributing dependencies,
native binaries, models, or packaged environments.
