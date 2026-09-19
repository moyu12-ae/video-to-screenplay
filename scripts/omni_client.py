#!/usr/bin/env python3
"""
scripts/omni_client.py - Direct DashScope (OpenAI-compatible) client for Omni tasks.

Adapted from QwenLM/Qwen-MM-Plugins (Apache License 2.0) - specifically
src/shared/api_omni.py, src/shared/omni_media.py and the omni_multi_speaker_asr
prompt. Differences from the original: a stdlib-only transport (urllib plus a
hand-rolled SSE accumulator instead of the openai SDK), our own V2S_OMNI_* env
namespace beside DASHSCOPE_*, workspace-contained temp files, no OSS upload
ladder (parts are pre-split to fit the inline budget), and an SSRF gate that
resolves the endpoint host and refuses any non-global address. Full attribution
lives in THIRD_PARTY_NOTICES.md.

Why this exists: the zcode MCP client aborts tool calls at ~30 s, while these
streaming A/V completions legitimately run for minutes. Speaking HTTP directly
from a subprocess puts retries, backoff and resume under code control.

Hard rules honoured here:
- The API key is read ONLY from the environment (DASHSCOPE_API_KEY); it is never
  logged, never written to disk and never placed in a URL.
- Before any request the endpoint must be https, resolve (via DNS) exclusively
  to globally routable addresses, and never redirect (redirects would silently
  switch hosts after validation).
- Logs and exception messages carry exception kind + HTTP status + host only -
  never request payloads or provider response bodies.
"""

import base64
import http.client
import ipaddress
import json
import os
import random
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.8-omni-flash"
DEFAULT_TIMEOUT_SEC = 1800.0   # streaming A/V completions can run long
DEFAULT_ATTEMPTS = 3
BACKOFF_BASE_SEC = 1.0
BACKOFF_CAP_SEC = 60.0
MAX_TOKENS = 65536
TEMPERATURE = 0.3              # matches upstream call_omni_json

# The endpoint caps ONE inline media item at 10 MB of base64; keep the raw
# payload at 3/4 of that (base64 expansion) minus 3% slack - upstream sizing.
INLINE_RAW_BUDGET_BYTES = int(10_000_000 * 3 / 4 * 0.97)   # ~7.27 MB
WAV_BYTES_PER_SEC = 16_000 * 2                             # 16 kHz mono s16le
MP3_KBPS_LADDER = (64, 48, 40, 32, 24, 16)                 # descending
FFMPEG_TIMEOUT_SEC = 600.0

_RNG = random.SystemRandom()   # jitter source: OS entropy, not the Mersenne twister

# Reproduced verbatim from Qwen-MM-Plugins (Apache-2.0); the diarization
# behaviour IS this prompt, so changing it changes the contract the merge
# stage was validated against.
DIARIZE_PROMPT = (
    "Transcribe ALL speech in this media and perform speaker diarization: assign each segment to a "
    'consistent speaker label (e.g. "Speaker 1", "Speaker 2") and give an accurate start and end '
    "time in seconds.{speakers}{lang} "
    "Output STRICTLY this JSON and nothing else: "
    '{{"segments": [{{"speaker": "<label>", "start": <sec>, "end": <sec>, "text": "<text>"}}]}}'
)


class OmniError(Exception):
    """API failure with a retry classification; messages never embed payloads."""

    def __init__(self, kind: str, status: Optional[int] = None, detail: str = "") -> None:
        self.kind = kind  # auth|bad_request|rate_limited|server|timeout|connection|empty|http
        self.status = status
        self.detail = detail
        super().__init__(
            kind + (f" ({status})" if status else "") + (f": {detail}" if detail else ""))

    @property
    def transient(self) -> bool:
        return self.kind in ("rate_limited", "server", "timeout", "connection", "empty")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        return None  # any redirect surfaces as an HTTPError - no silent host switch


_OPENER = urllib.request.build_opener(_NoRedirect)


def _log(stream: Any, message: str) -> None:
    try:
        (stream if stream is not None else sys.stderr).write(message + "\n")
    except Exception:  # noqa: BLE001 - logging must never break the pipeline
        pass


def _env_int(name: str, default: int) -> int:
    try:
        raw = os.environ.get(name)
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        raw = os.environ.get(name)
        return float(raw) if raw else default
    except ValueError:
        return default


def resolve_key(explicit: Optional[str] = None) -> Optional[str]:
    key = explicit or os.environ.get("DASHSCOPE_API_KEY")
    return key.strip() if key and key.strip() else None


def resolve_base_url(explicit: Optional[str] = None) -> str:
    return (explicit or os.environ.get("DASHSCOPE_BASE_URL") or DEFAULT_BASE_URL).strip()


def resolve_model(explicit: Optional[str] = None) -> str:
    return (explicit or os.environ.get("V2S_OMNI_MODEL") or DEFAULT_MODEL).strip()


def resolve_public_host(host: str) -> List[str]:
    """Resolve `host` and return the addresses only when EVERY one is globally
    routable; refuses loopback / private / link-local / reserved targets, which
    also rules out cloud metadata endpoints (169.254.168.254 and friends)."""
    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        raise OmniError("connection", detail=f"endpoint host {host} did not resolve")
    addrs: List[str] = []
    for info in infos:
        addr = str(info[4][0]).split("%")[0]  # strip an IPv6 scope id if present
        ip = ipaddress.ip_address(addr)
        if not ip.is_global:
            raise OmniError("bad_request",
                            detail=f"endpoint host {host} resolves to non-public {addr}")
        if addr not in addrs:
            addrs.append(addr)
    if not addrs:
        raise OmniError("connection", detail=f"endpoint host {host} did not resolve")
    return addrs


def validate_endpoint(base_url: str,
                      resolver: Optional[Callable[[str], List[str]]] = None) -> str:
    """Return the chat-completions URL after refusing non-https or non-public
    endpoints: scheme must be https, obvious non-public hostnames are refused
    syntactically, and the host must resolve exclusively to global addresses."""
    url = (base_url or "").strip().rstrip("/")
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https":
        raise OmniError("bad_request", detail=f"endpoint must be https, got {parts.scheme!r}")
    host = parts.hostname or ""
    lowered = host.lower()
    if (not host or lowered == "localhost"
            or lowered.endswith((".localhost", ".local", ".internal"))):
        raise OmniError("bad_request", detail=f"refusing non-public endpoint host {host!r}")
    try:
        addr = ipaddress.ip_address(lowered)
    except ValueError:
        addr = None
    if addr is not None and not addr.is_global:
        raise OmniError("bad_request", detail=f"refusing non-public endpoint address {host}")
    for resolved in (resolver or resolve_public_host)(host):
        ip = ipaddress.ip_address(str(resolved).split("%")[0])
        if not ip.is_global:  # re-verify whatever the resolver returned
            raise OmniError("bad_request",
                            detail=f"endpoint host {host} resolves to non-public {resolved}")
    return url + "/chat/completions"


def select_audio_encoding(duration_sec: float, budget: int = INLINE_RAW_BUDGET_BYTES,
                          raw_size: Optional[int] = None) -> Tuple[str, Optional[int]]:
    """Decide how to fit one audio part under the inline budget. Pure;
    deterministic. Returns ("passthrough", None) | ("wav", None) | ("mp3", kbps).
    Raises ValueError when even 16 kbps MP3 cannot fit."""
    if raw_size is not None and raw_size <= budget:
        return "passthrough", None
    if duration_sec * WAV_BYTES_PER_SEC <= budget:
        return "wav", None
    for kbps in MP3_KBPS_LADDER:  # descending; first affordable tier wins
        if kbps * 1000 / 8 * duration_sec <= budget:
            return "mp3", kbps
    raise ValueError(
        f"audio too long to fit the inline budget even at 16 kbps ({duration_sec:.0f} s); "
        "split it into shorter parts")


def _encode_audio(src: Path, kind: str, kbps: Optional[int]) -> Path:
    out = src.parent / f".fit_{src.stem}.{kind}"
    args = ["-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
            "-af", "aresample=async=1:first_pts=0",
            "-c:a", "pcm_s16le" if kind == "wav" else "libmp3lame"]
    if kbps:
        args += ["-b:a", f"{kbps}k"]
    args += ["-y", str(out)]
    try:
        proc = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error"] + args,
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                              timeout=FFMPEG_TIMEOUT_SEC, check=False)
    except subprocess.TimeoutExpired:
        raise RuntimeError("ffmpeg audio fit timed out")
    if proc.returncode != 0 or not out.is_file():
        lines = (proc.stderr or b"").decode("utf-8", "replace").strip().splitlines()
        tail = lines[-1] if lines else "unknown ffmpeg error"
        raise RuntimeError(f"ffmpeg audio fit failed: {tail}")
    return out


def fit_audio(audio_path: str, duration_sec: float,
              budget: int = INLINE_RAW_BUDGET_BYTES) -> Tuple[Path, str]:
    """Materialize the upload payload for one audio part.

    Tiers: send the file as-is when it fits the budget; else 16 kHz mono WAV;
    else MP3 down the ladder (highest affordable tier first, verifying the real
    output size). Temp outputs land beside the source - inside the workspace
    containment, validated by the caller - and are removed by the caller.
    Returns (payload_path, audio_format)."""
    src = Path(audio_path)
    select_audio_encoding(duration_sec, budget)  # fail fast before any ffmpeg work
    if src.stat().st_size <= budget:
        return src, src.suffix.lstrip(".").lower() or "m4a"
    if duration_sec * WAV_BYTES_PER_SEC <= budget:
        return _encode_audio(src, "wav", None), "wav"
    for kbps in MP3_KBPS_LADDER:
        if kbps * 1000 / 8 * duration_sec > budget:
            continue
        out = _encode_audio(src, "mp3", kbps)
        if out.stat().st_size <= budget:
            return out, "mp3"
    raise ValueError("audio payload exceeds the inline budget at every MP3 tier; "
                     "split it into shorter parts")


def accumulate_sse(lines: Iterable[Union[bytes, str]]) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Fold an SSE line stream into (text, usage). Tolerates comment / blank /
    malformed lines (skipped), skips choice-less chunks (the trailing usage
    chunk), accumulates every delta.content, and stops at 'data: [DONE]'."""
    parts: List[str] = []
    usage: Optional[Dict[str, Any]] = None
    for raw in lines:
        line = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
        line = line.strip()
        if not line or line.startswith(":") or not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except ValueError:
            continue
        if not isinstance(chunk, dict):
            continue
        if isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = (choice or {}).get("delta") or {}
            piece = delta.get("content")
            if piece:
                parts.append(str(piece))
    return "".join(parts), usage


def extract_json_payload(text: str) -> Any:
    """Pull the JSON value out of a model reply: strip ``` fences, slice to the
    outermost braces, retry once without trailing commas. Raises ValueError."""
    s = str(text or "").strip()
    if not s:
        raise ValueError("empty completion")
    fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL)
    if fence:
        s = fence.group(1).strip()
    opens = [i for i in (s.find("{"), s.find("[")) if i >= 0]
    closes = [i for i in (s.rfind("}"), s.rfind("]")) if i >= 0]
    if not opens or not closes:
        raise ValueError("no JSON object/array found in completion")
    s = s[min(opens):max(closes) + 1]
    try:
        return json.loads(s)
    except ValueError:
        pass
    s = re.sub(r",(\s*[}\]])", r"\1", s)
    try:
        return json.loads(s)
    except ValueError as e:
        raise ValueError(f"completion is not valid JSON: {e}")


def _map_http_error(e: urllib.error.HTTPError) -> OmniError:
    code = getattr(e, "code", None) or 0
    kind = ("timeout" if code == 408 else
            "rate_limited" if code == 429 else
            "server" if code >= 500 else
            "auth" if code in (401, 403) else
            "bad_request" if code == 400 else "http")
    return OmniError(kind, status=code, detail="upstream rejected the request")


def _post_stream(url: str, api_key: str, body: Dict[str, Any],
                 timeout_sec: float) -> Tuple[str, Optional[Dict[str, Any]]]:
    host = urllib.parse.urlsplit(url).hostname or ""
    req = urllib.request.Request(
        url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"), method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {api_key}",  # header only; never in a URL
        })
    try:
        resp = _OPENER.open(req, timeout=timeout_sec)
    except urllib.error.HTTPError as e:
        raise _map_http_error(e)
    except (TimeoutError, socket.timeout):
        raise OmniError("timeout", detail=f"host {host}")
    except (urllib.error.URLError, http.client.HTTPException, OSError):
        raise OmniError("connection", detail=f"host {host}")

    text: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    try:
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if "text/event-stream" in ctype:
            text, usage = accumulate_sse(resp)
        else:
            data = json.loads(resp.read().decode("utf-8", "replace"))
            choices = data.get("choices") if isinstance(data, dict) else None
            message = (choices[0] or {}).get("message") or {} if choices else {}
            text = str(message.get("content") or "")
            usage = data.get("usage") if isinstance(data, dict) else None
    except (TimeoutError, socket.timeout):
        raise OmniError("timeout", detail=f"host {host}")
    except (http.client.HTTPException, OSError, ValueError):
        raise OmniError("connection", detail=f"host {host}")
    finally:
        resp.close()
    if not (text or "").strip():
        raise OmniError("empty", detail=f"host {host}")
    return text, usage


def call_omni_chat(messages: List[Dict[str, Any]], *, api_key: Optional[str] = None,
                   base_url: Optional[str] = None, model: Optional[str] = None,
                   max_tokens: int = MAX_TOKENS, temperature: float = TEMPERATURE,
                   timeout_sec: Optional[float] = None, attempts: Optional[int] = None,
                   log: Any = None) -> Tuple[str, Optional[Dict[str, Any]]]:
    """POST one streaming chat completion and return (text, usage).

    Transient failures - timeout, connection, 408/429/5xx, empty completion -
    back off exponentially with jitter; auth and malformed-request errors fail
    fast. Attempts/timeout default from V2S_OMNI_ATTEMPTS / V2S_OMNI_TIMEOUT_SEC."""
    key = api_key or resolve_key()
    if not key:
        raise OmniError("auth", detail="DASHSCOPE_API_KEY is not set in the environment")
    url = validate_endpoint(base_url or resolve_base_url())
    body = {
        "model": model or resolve_model(),
        "messages": messages,
        "stream": True,  # the Omni endpoint requires streaming + text modality
        "stream_options": {"include_usage": True},
        "modalities": ["text"],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    to = timeout_sec if timeout_sec is not None else _env_float("V2S_OMNI_TIMEOUT_SEC", DEFAULT_TIMEOUT_SEC)
    n = attempts if attempts is not None else _env_int("V2S_OMNI_ATTEMPTS", DEFAULT_ATTEMPTS)
    last = ""
    for attempt in range(max(1, n)):
        if attempt:
            delay = min(BACKOFF_CAP_SEC, BACKOFF_BASE_SEC * (2 ** (attempt - 1))) * _RNG.uniform(0.5, 1.0)
            _log(log, f"[RETRY] attempt {attempt + 1}/{n} in {delay:.1f}s (previous: {last})")
            time.sleep(delay)
        try:
            return _post_stream(url, key, body, to)
        except OmniError as e:
            if not e.transient:
                raise
            last = str(e)
            if attempt == n - 1:
                raise
    raise OmniError("empty", detail="retry loop exhausted")  # pragma: no cover


def _probe_duration_sec(path: str) -> Optional[float]:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
            stderr=subprocess.DEVNULL, timeout=120.0)
        return float(out.decode("utf-8", "replace").strip())
    except Exception:  # noqa: BLE001 - duration is an optimization, not a requirement
        return None


def _coerce_segment(seg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(seg)
    for k in ("start", "end"):
        try:
            out[k] = float(out[k])
        except (TypeError, ValueError):
            pass
    return out


def diarize_audio_file(audio_path: str, *, api_key: Optional[str] = None,
                       base_url: Optional[str] = None, model: Optional[str] = None,
                       num_speakers: Optional[int] = None, language: Optional[str] = None,
                       duration_sec: Optional[float] = None, timeout_sec: Optional[float] = None,
                       attempts: Optional[int] = None, log: Any = None) -> Dict[str, Any]:
    """One audio part in, one MCP-tool-shaped dict out:
    {"speakers": [...], "segments": [{"speaker","start","end","text"}], "meta": {...}}.
    Times are float seconds - the same schema the merge stage already consumes;
    the extra meta key is ignored there and kept as provenance."""
    src = Path(audio_path)
    if not src.is_file():
        raise OmniError("bad_request", detail=f"audio part not found: {src.name}")
    dur = duration_sec if duration_sec and duration_sec > 0 else _probe_duration_sec(str(src))
    payload, audio_format = fit_audio(str(src), dur or 0.0)
    try:
        b64 = base64.b64encode(payload.read_bytes()).decode("ascii")
        audio_part = {"type": "input_audio",
                      "input_audio": {"data": f"data:;base64,{b64}", "format": audio_format}}
        speakers_hint = f" There are {num_speakers} distinct speakers." if num_speakers else ""
        lang_hint = f" The spoken language is {language}." if language else ""
        prompt = DIARIZE_PROMPT.format(speakers=speakers_hint, lang=lang_hint)
        messages = [{"role": "user", "content": [audio_part, {"type": "text", "text": prompt}]}]
        text, usage = call_omni_chat(messages, api_key=api_key, base_url=base_url, model=model,
                                     timeout_sec=timeout_sec, attempts=attempts, log=log)
        data = extract_json_payload(text)
        if isinstance(data, dict):
            raw = data.get("segments") or data.get("results") or []
        elif isinstance(data, list):
            raw = data
        else:
            raw = []
        segments = [_coerce_segment(s) for s in raw if isinstance(s, dict)]
        speakers = sorted({str(s["speaker"]) for s in segments if s.get("speaker")})
        return {
            "speakers": speakers,
            "segments": segments,
            "meta": {
                "backend": "direct_api",
                "model": model or resolve_model(),
                "audio_encoding": audio_format,
                "endpoint_host": urllib.parse.urlsplit(resolve_base_url(base_url)).hostname,
                "usage": usage,
            },
        }
    finally:
        if payload != src:
            try:
                payload.unlink()
            except OSError:
                pass
