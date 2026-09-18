# Security Policy

## Supported versions

Security fixes are provided for the current `0.7.x` public release line.

## Reporting a vulnerability

Report security vulnerabilities privately using GitHub private vulnerability
reporting / Security Advisories when available. If that mechanism is unavailable,
contact the maintainer through a private channel.

Do not post credentials, tokens, cookies, personal information, private server data,
or exploit details in public issues.

## Network deployment

`TOOL_GATEWAY_HOST` and `CAPTURE_HUB_HOST` default to `127.0.0.1`.

Keep local HTTP/WebSocket services on loopback unless remote access is explicitly
required. Capture Hub and distributed TTS workers should remain on a trusted LAN or
VPN unless the operator adds an authenticated TLS/WSS reverse-proxy boundary.

Do not expose plain local worker or Capture Hub transports directly to the public
Internet.

## Secrets and runtime data

Keep Discord/provider credentials, browser cookies, worker tokens, local databases,
runtime logs, private voice references, model weights, and checkpoints outside Git.

Rotate any credential that may have been exposed.

## Third-party boundary

Discord, Gemini, external providers, Python packages, native libraries, models, and
externally operated workers are maintained outside this project.

Review third-party advisories and perform a new security/license review before
redistributing a binary or portable environment that bundles those dependencies.

See `THIRD_PARTY_NOTICES.md` for the source-release dependency boundary.
