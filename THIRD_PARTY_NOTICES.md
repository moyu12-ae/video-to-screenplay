# Third-Party Notices

This project incorporates code adapted from the following projects. The full
license texts are preserved in `licenses/`.

## QwenLM/Qwen-MM-Plugins — Apache License 2.0

- Source: https://github.com/QwenLM/Qwen-MM-Plugins
- Used in: `scripts/omni_client.py` (module header carries the same attribution)
- **What was adapted** (from `src/shared/api_omni.py`, `src/shared/omni_media.py`
  and `src/capabilities/api/.../omni/omni_multi_speaker_asr.py`):
  - the DashScope compatible-mode request shape (`stream=True` +
    `stream_options.include_usage` + `modalities=["text"]`, `max_tokens`,
    `temperature=0.3` for JSON calls);
  - the bare `data:;base64,<payload>` audio part form (no mime prefix, format
    carried by the `format` field);
  - the 16 kHz mono audio-fitting strategy (send as-is / WAV / MP3 ladder
    64→16 kbps against the inline budget);
  - the transient-error classification (timeout / connection / 429 / 5xx /
    empty completion) with exponential-backoff retry and fail-fast on auth;
  - the JSON extraction (code-fence stripping, outermost-brace slicing,
    trailing-comma cleanup);
  - the `omni_multi_speaker_asr` diarization prompt, reproduced **verbatim** —
    the diarization behaviour IS that prompt, and the merge stage was validated
    against it.
- **Changes from the original**: stdlib-only transport (urllib + a hand-rolled
  SSE accumulator) instead of the openai SDK; a `V2S_OMNI_*` environment
  namespace beside `DASHSCOPE_*`; an SSRF gate that resolves the endpoint host
  and refuses any non-globally-routable address, plus refusal of HTTP redirects;
  workspace-contained temp files; no OSS upload ladder (audio parts are
  pre-split to fit the inline budget); no MCP plumbing.

- **Additional v0.5 patterns referenced from the same project's skills** (no
  code copied, Apache-2.0 attribution extends to these):
  - the `omni-memory` skill's SW_PROMPT evidence-schema shape (dual visual/audio
    JSON with anti-hallucination rules and a time-basis declaration), its
    anonymous-entity idea (describe people by visible epithets, defer naming)
    and its 5 s inter-window overlap;
  - the `omni-chatcut` movie-commentary skill's evidence_refs validation idea
    (→ our per-scene coverage check) and relative→absolute timestamp mapping;
  - the `omni_av_caption` tool's Visible-Text section shape and the
    1 fps / 448² / ~9 min-per-10 MB inline capacity figures.

## NarratoAI — MIT License (© 2024 linyq)

- Source: https://github.com/linyqh/NarratoAI
- **Engineering patterns referenced, no code copied**: transient-only retries
  with immediate failure on auth errors; failure isolation across batch items
  (one failed part never aborts the rest); per-item response caching for
  resume-after-interruption. Acknowledged in EVOLUTION.md.

## narrator-ai-cli-skill — MIT License

- Source: https://github.com/NarratorAI-Studio/narrator-ai-cli-skill
- **Posture referenced, no code copied**: cost preview + explicit user
  confirmation before launching a multi-call perception pass (our stage 3.7
  gate). Acknowledged in EVOLUTION.md.
