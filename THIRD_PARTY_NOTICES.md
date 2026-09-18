# Third-Party Notices

Moxue Discord AI Assistant v0.7.0 is published as a source-only repository.
Installed Python dependencies, native libraries, model files, runtime environments,
GPT-SoVITS assets, checkpoints, and private voice assets are not vendored here.

The project source is licensed under MIT. Third-party software remains subject to
its own license terms.

## Declared dependency inventory

Observed versions are from the reviewed dependency environment; dependency ranges
in `pyproject.toml` remain authoritative for installation.

| Group | Package | Reviewed version | Reported license | Upstream |
| --- | --- | --- | --- | --- |
| `capture-build` | `pyinstaller` | `6.22.3` | GNU General Public License v2 (GPLv2) | [link](https://pyinstaller.org) |
| `meeting` | `dxcam` | `0.3.0` | MIT | [link](https://github.com/ra1nty/DXcam) |
| `meeting` | `onnxruntime` | `1.30.0` | MIT License | [link](https://onnxruntime.ai) |
| `meeting` | `Pillow` | `12.3.0` | MIT-CMU | [link](https://python-pillow.github.io) |
| `meeting` | `rapidocr` | `3.9.2` | Apache-2.0 | [link](https://github.com/RapidAI/RapidOCR/releases) |
| `meeting` | `SoundCard` | `0.4.6` | BSD License | [link](https://github.com/bastibe/SoundCard) |
| `runtime` | `aiohttp` | `3.14.3` | Apache-2.0 AND MIT | [link](https://github.com/aio-libs/aiohttp) |
| `runtime` | `discord-ext-voice-recv` | `0.5.3a185` | MIT License | [link](https://github.com/rdphillips7/discord-ext-voice-recv) |
| `runtime` | `discord.py` | `2.7.1` | MIT License | [link](https://discordpy.readthedocs.io/en/latest/) |
| `runtime` | `faster-whisper` | `1.2.1` | MIT License | [link](https://github.com/SYSTRAN/faster-whisper) |
| `runtime` | `google-genai` | `2.24.0` | Apache-2.0 | [link](https://github.com/googleapis/python-genai) |
| `runtime` | `python-dotenv` | `1.2.3` | BSD-3-Clause | [link](https://github.com/theskumar/python-dotenv) |
| `runtime` | `tzdata` | `2026.4` | Apache-2.0 | [link](https://github.com/python/tzdata) |
| `tts-local` | `kokoro` | `0.9.4` | Apache Software License | [link](https://github.com/hexgrad/kokoro) |
| `tts-local` | `misaki` | `0.9.4` | Apache Software License | [link](UNKNOWN) |
| `tts-local` | `soundfile` | `0.14.0` | BSD License | [link](https://github.com/bastibe/python-soundfile) |

## Pinned Git dependency

`discord-ext-voice-recv` is installed from the reviewed pinned commit
`ddd28601fe556f585b869e215f29c8236b95f88f`. The exact pinned source was reviewed as MIT.
The dependency itself is not copied into this Git history.

## Optional local-TTS boundary

The optional `tts-local` dependency closure can include `phonemizer-fork`
(GPLv3-or-later) and `espeakng-loader`. The reviewed `espeakng-loader`
environment included eSpeak NG runtime material; eSpeak NG is GPLv3-or-later.

Those packages, native binaries, language data, model files, and voice assets
are not redistributed by this source-only release.

A future executable, portable environment, container, model bundle, or other
binary distribution requires a new redistribution/license review for the exact
artifacts being shipped.

## Native and transitive notices

Externally installed dependencies can carry additional third-party notices.
The reviewed environment included, among others, ONNX Runtime and OpenCV
third-party notice/license material. Those installed artifacts remain outside
this repository and retain their own notice obligations.
