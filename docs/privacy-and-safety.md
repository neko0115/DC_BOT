# Privacy and Safety

Moxue is self-hosted, but some optional features intentionally call external providers.
This document describes the main boundaries operators and contributors should preserve.

## Bounded Gemini context

AI requests use bounded, task-relevant context rather than treating an entire Discord
server as model input.

Depending on the feature, a request may include the current question, bounded recent
conversation context, a replied-to message, an allowed image attachment, or authorized
memory.

Content intentionally sent to Gemini or another configured provider leaves the local
machine.

## Local runtime storage

Runtime state such as SQLite application data, saved Memory records, Chat Style state,
music metadata, caches, and tool artifacts remains local unless an external feature is
explicitly invoked.

Runtime databases, logs, caches, private media, and generated sensitive artifacts must
not be committed to the source repository.

## Memory V2 privacy gates

Memory is scoped rather than globally shared.

Personal, shared, session, project, and domain-aware memory paths apply separate
eligibility and reconciliation rules. Sensitive, secret-like, unsafe instruction,
low-confidence, and otherwise disallowed material is rejected by the applicable gates.

Memory inspection, deletion, and administrator controls remain scoped to their intended
authority.

## Chat Style privacy gates

Chat Style does not treat every Discord message as a learning sample.

Mention artifacts, sensitive personal data, phone/address-shaped data, secret-like
content, unsafe instructions, command material, and other excluded categories are
rejected by the applicable collection gates.

Users can inspect their derived profile, disable future Chat Style learning, or reset
their stored Chat Style data.

## Knowledge and public-web isolation

Private Discord content is not evidence that data is safe to send to a public provider.

Public-term lookup creates a bounded query only after passing a dedicated safety
boundary. Discord identities, private provenance, credentials, sensitive information,
local/private-network locations, and unsafe instruction content must not cross that
boundary.

Explicit web-research and image-generation features may contact their configured
providers.

## Meeting and Capture boundary

Meeting workflows and Capture Agent are intended for controlled self-hosted use.

Capture Hub should remain on loopback, a trusted LAN, or a VPN. Do not expose the plain
WebSocket transport directly to the public Internet. A public remote deployment should
add an authenticated TLS/WSS boundary and suitable firewall controls.

Operators are responsible for consent, retention, local access control, and applicable
recording/transcription requirements.

## TTS and GPT-SoVITS boundary

TTS may use an external provider, optional local runtime, or authenticated distributed
worker.

Remote synthesis sends the text required for speech generation to the selected worker.
Worker authentication tokens must remain private.

GPT-SoVITS is external to this repository. Weights, checkpoints, reference audio,
private voice assets, and model files are not included. Operators are responsible for
the provenance, rights, security, and consent associated with configured voice assets.

## Administrator responsibilities

Administrators are responsible for:

- protecting and rotating credentials;
- limiting filesystem and network access to runtime data;
- deciding which external providers are enabled;
- configuring Discord permissions and privileged roles correctly;
- keeping Capture/TTS remote services inside an appropriate trust boundary;
- obtaining appropriate consent for recording, transcription, and voice processing;
- reviewing third-party security and licensing before deployment or redistribution;
- deleting retained data when it is no longer required.
