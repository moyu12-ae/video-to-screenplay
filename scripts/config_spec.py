#!/usr/bin/env python3
"""
scripts/config_spec.py - The one source of truth for this plugin's configuration
and its data-egress surface (v0.5.2).

Everything a user or an agent must know before a byte leaves the machine lives in
these tables. SECURITY.md and both READMEs restate them for humans, and
tests/test_packaging.py fails when code and documents drift apart - the same
anti-drift idea that pins the version across the plugin manifests.

Nothing here is executed at import time and no value is ever read from disk:
DASHSCOPE_API_KEY exists only in the process environment, is never written to any
artifact, and is never interpolated into a log line, an error message or an
evidence file. omni_client.resolve_key() is the only reader (test-enforced).
"""

from typing import Dict, List

DEFAULT_BASE_URL_HOST = "dashscope.aliyuncs.com"

ENV_VARS: List[Dict[str, str]] = [
    {
        "name": "DASHSCOPE_API_KEY",
        "required": "yes for the direct-API paths (acoustic diarization, AV understanding)",
        "default": "(unset — the pipeline degrades and says so)",
        "unlocks": "speaker_diarize.py run and av_understand.py run; without it they exit 8",
        "egress": "sent as a Bearer header to DASHSCOPE_BASE_URL only",
    },
    {
        "name": "DASHSCOPE_BASE_URL",
        "required": "no",
        "default": f"https://{DEFAULT_BASE_URL_HOST}/compatible-mode/v1",
        "unlocks": "which host receives the key and the media",
        "egress": "anything you point this at; validate_endpoint() refuses non-https, "
                  "loopback, private, link-local and reserved targets",
    },
    {
        "name": "V2S_OMNI_MODEL",
        "required": "no",
        "default": "qwen3.8-omni-flash",
        "unlocks": "which omnimodal model interprets your footage",
        "egress": "the media parts below are tokenized by this model",
    },
    {
        "name": "V2S_OMNI_ATTEMPTS",
        "required": "no",
        "default": "3",
        "unlocks": "retry budget per request (transient errors only)",
        "egress": "raises how often a payload can be re-sent; never above the billed call",
    },
    {
        "name": "V2S_OMNI_TIMEOUT_SEC",
        "required": "no",
        "default": "1800",
        "unlocks": "wall-clock ceiling per streaming request",
        "egress": "none",
    },
]

# What actually leaves the machine, per stage. Ordered by how much of the source
# an operator is consenting to upload.
DATA_EGRESS: List[str] = [
    "speaker_diarize.py prepare/run: 16 kHz mono audio of the episode, cut into timed parts",
    "av_understand.py run: per-scene video segments (<=90 s windows, 480p/CRF ladder, "
    "mono 16 kHz audio track included) as base64 inline parts",
    "dialogue text is never uploaded for transcription - subtitles are read locally and "
    "spliced verbatim; the AV prompt explicitly forbids transcribing speech",
]

# Key-shaped strings that must never appear in tracked files. Tests assemble fake
# keys from parts so the fixtures stay clean too.
SECRET_PATTERNS: List[str] = [
    r"sk-[A-Za-z0-9]{20,}",              # DashScope / OpenAI-style keys
    r"LTAI[A-Za-z0-9]{12,}",             # Aliyun access-key ids
    r"Bearer\s+[A-Za-z0-9._\-]{16,}",    # hand-written auth headers
]


def names() -> List[str]:
    return [var["name"] for var in ENV_VARS]
