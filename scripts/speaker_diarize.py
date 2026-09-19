#!/usr/bin/env python3
"""
scripts/speaker_diarize.py - Acoustic Speaker Diarization Bridge (Qwen3.8-Omni)

Three actions around one workorder (mirroring narrative_outline.py's pattern
"deterministic prep here, perception task delegated to the model"):

  prepare : ffmpeg-extract 16 kHz mono audio (auto-chunked into <=45 min parts
            when the video runs past 50 min), write diarize_workorder.json with
            the expected save paths, then EXIT 6 awaiting perception.
  run     : (recommended) dial DashScope DIRECTLY per part - omni_client.py
            streams the same qwen3.8-omni-flash request the MCP tool would,
            with code-owned retries/backoff/resume and no client tool window.
  merge   : validate the saved Omni outputs (from `run` or from manual MCP
            calls - same schema), restore part offsets, bind every subtitle
            line to the max-overlap voice cluster, name clusters by majority
            vote over subtitle metadata names, print speakers.json.

Separation of concerns (hard rules):
- WHO speaks (attribution) is 100% acoustic, decided by the Omni model's timbre
  clustering. Music / BGM robustness comes from the model, not from this script.
- Character NAMING uses only the metadata names already present in
  extracted.json (ASS Actor/Name fields, 【角色】/角色： prefixes) as a majority
  vote (>= 0.6 share, >= 2 votes) - metadata never decides attribution.
- Dialogue TEXT still comes exclusively from subtitles downstream ([[SUB:n]]
  splicing); acoustic transcripts are stored as evidence only, never re-typed.

Degradation (decided contract): without the MCP tool / DashScope key, run
`merge --empty-fallback` to emit a valid speakers.json with all-null speakers
plus a WARN - the pipeline keeps running, equivalent to the legacy
"unattributed" state. The old pure-text attribution (dash A/B alternation,
text-syntax per-line assignment) is intentionally removed.

Exit codes: 0 success | 1 bad input | 3 ffmpeg/ffprobe missing
            4 audio extraction failed | 6 workorder written, awaiting perception
            7 Omni outputs missing/invalid/parts incomplete at merge stage
            8 DASHSCOPE_API_KEY missing (run only).
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import omni_client
import op_ed

SPEAKER_LABEL_RE = re.compile(r"^SPEAKER_[A-Z0-9]+$")

AUDIO_EXTRACT_TIMEOUT_SEC = 900.0
FFPROBE_TIMEOUT_SEC = 120.0
# Inline cap of the MCP endpoint (~55 min after its own MP3 re-fit); stay under it.
CHUNK_THRESHOLD_SEC = 3000.0   # 50 min: beyond this, split
CHUNK_TARGET_SEC = 2700.0      # 45 min per part
CHUNK_MERGE_TAIL_SEC = 600.0   # tails shorter than this fold into the previous part
CHUNK_OVERLAP_SEC = 3.0        # adjacent parts share ±3s so merge can link identities
SILENCE_SNAP_TOLERANCE_SEC = 5.0
SILENCE_DETECT_NOISE = "-30dB"
SILENCE_DETECT_MIN_D = "0.3"
COOCCUR_MIN_RATIO = 0.5        # cross-part turns overlapping ≥50% of the shorter = same voice
SUBTITLE_OFFSET_SEARCH_SEC = 5.0   # global subtitle-vs-audio shift search range (±5s)
SUBTITLE_OFFSET_STEP_MS = 250      # shift search granularity
MIN_SUBTITLE_LINES_FOR_OFFSET = 6  # below this, a global shift estimate is noise

EXIT_OK = 0
EXIT_BAD_INPUT = 1
EXIT_MISSING_FFMPEG = 3
EXIT_EXTRACT_FAILED = 4
EXIT_AWAITING_OMNI = 6
EXIT_OMNI_OUTPUT_INVALID = 7
EXIT_MISSING_KEY = 8
RUN_PART_WARN_SEC = 1500.0     # 25 min: beyond this, direct-call wall clock nears the ceiling

ATTRIBUTION_METHOD = "qwen3_8_omni_acoustic_diarization"
MCP_TOOL_NAME = "omni_multi_speaker_asr"

# Confidence tiers for a line's primary acoustic overlap ratio.
CONF_HIGH_RATIO = 0.80
CONF_MID_RATIO = 0.50
CONF_HIGH = 0.90
CONF_MID = 0.75
CONF_LOW = 0.55
SECONDARY_MIN_RATIO = 0.40   # overlap fraction of the LINE for a secondary speaker
NAME_VOTE_MIN_SHARE = 0.60   # metadata-name share required to name a cluster
NAME_VOTE_MIN_COUNT = 2      # minimum metadata votes to name a cluster
TEXT_AGREEMENT_RATIO = 0.60  # difflib ratio between subtitle text and Omni transcript

_PUNCT_RE = re.compile(r"[\s，。：:；;！!？?、,.\[\]【】()（）「」『』…—·\-–]")


def is_provisional_label(speaker: Any) -> bool:
    """True for synthetic labels (SPEAKER_UNKNOWN / SPEAKER_A1 / SPEAKER_00A ...)
    that must never be rendered as a character name in the final screenplay."""
    if speaker is None:
        return True
    s = str(speaker).strip()
    return (not s) or s in {"UNKNOWN", "SPEAKER_UNKNOWN"} or bool(SPEAKER_LABEL_RE.match(s))


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write JSON through a sibling temp file then os.replace it into place.

    A part killed mid-write used to leave a truncated omni_diarized.partNNN.json
    behind, and the next `run` could resume over it instead of re-doing that part."""
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# prepare phase
# ---------------------------------------------------------------------------

def require_binaries() -> None:
    missing = [b for b in ("ffmpeg", "ffprobe") if shutil.which(b) is None]
    if missing:
        sys.stderr.write(f"[FATAL] Missing required executable(s): {', '.join(missing)}; `brew install ffmpeg`\n")
        sys.exit(EXIT_MISSING_FFMPEG)


def _ffprobe(args: List[str]) -> str:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error"] + args,
            stderr=subprocess.DEVNULL,
            timeout=FFPROBE_TIMEOUT_SEC,
        )
        return out.decode("utf-8", "replace").strip()
    except Exception:
        return ""


def probe_duration_ms(path: str) -> Optional[int]:
    out = _ffprobe(["-show_entries", "format=duration", "-of", "csv=p=0", path])
    try:
        return int(float(out) * 1000)
    except (TypeError, ValueError):
        return None


def probe_has_audio(path: str) -> bool:
    return bool(_ffprobe(["-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", path]))


def plan_boundaries(duration_ms: int, chunk_target_sec: Optional[float] = None) -> List[float]:
    """Interior cut points (seconds) at fixed target intervals; a short tail folds
    back. Empty list = single part. Pure function; deterministic."""
    target = float(chunk_target_sec) if chunk_target_sec else CHUNK_TARGET_SEC
    duration_s = duration_ms / 1000.0
    threshold = CHUNK_THRESHOLD_SEC if chunk_target_sec is None else target
    if duration_s <= max(threshold, target):
        return []
    boundaries: List[float] = []
    b = target
    while b < duration_s - 1e-6:
        boundaries.append(b)
        b += target
    if boundaries:
        fold = min(CHUNK_MERGE_TAIL_SEC, target / 2.0)
        if duration_s - boundaries[-1] < fold:
            boundaries.pop()
    return boundaries


def snap_boundaries(boundaries: List[float], silence_points: List[float],
                    tolerance: float = SILENCE_SNAP_TOLERANCE_SEC) -> List[float]:
    """Snap each boundary to the nearest unused silence midpoint within tolerance
    so parts break between speaker turns, never mid-utterance."""
    snapped: List[float] = []
    used: set = set()
    for b in boundaries:
        best_i, best_d = None, tolerance
        for i, s in enumerate(silence_points):
            d = abs(s - b)
            if d <= best_d and i not in used:
                best_i, best_d = i, d
        if best_i is not None:
            used.add(best_i)
            snapped.append(round(float(silence_points[best_i]), 3))
        else:
            snapped.append(round(float(b), 3))
    return sorted(snapped)


def parts_from_boundaries(duration_ms: int, boundaries: List[float],
                          overlap_sec: float = CHUNK_OVERLAP_SEC) -> List[Tuple[int, int]]:
    """Parts between cut points; adjacent parts share ±overlap_sec so merge can
    link identities by utterance co-occurrence. Pure; deterministic."""
    duration_ms = int(duration_ms)
    ov = int(overlap_sec * 1000)
    cuts = [0] + [max(0, min(int(b * 1000), duration_ms)) for b in sorted(boundaries)] + [duration_ms]
    cuts = [c for i, c in enumerate(cuts) if i == 0 or c > cuts[i - 1]]
    parts: List[Tuple[int, int]] = []
    for i in range(len(cuts) - 1):
        start = 0 if i == 0 else max(0, cuts[i] - ov)
        end = duration_ms if i == len(cuts) - 2 else min(duration_ms, cuts[i + 1] + ov)
        if end - start > 0:
            parts.append((start, end))
    return parts


def plan_parts(duration_ms: int, chunk_target_sec: Optional[float] = None,
               silence_points: Optional[List[float]] = None) -> List[Tuple[int, int]]:
    """Full plan: boundaries -> (optional) silence snapping -> overlap-aware parts.
    Single part when no boundaries survive (<=50 min by default, <= target with an
    explicit --chunk-seconds, or after tail folding)."""
    boundaries = plan_boundaries(duration_ms, chunk_target_sec)
    if boundaries and silence_points:
        boundaries = snap_boundaries(boundaries, silence_points)
    return parts_from_boundaries(duration_ms, boundaries)


def parse_silences(log_text: str) -> List[float]:
    """Parse ffmpeg silencedetect stderr into silence midpoints (seconds)."""
    midpoints: List[float] = []
    start = None
    for line in log_text.splitlines():
        m = re.search(r"silence_start:\s*([0-9.]+)", line)
        if m:
            start = float(m.group(1))
            continue
        m = re.search(r"silence_end:\s*([0-9.]+)", line)
        if m and start is not None:
            end = float(m.group(1))
            if end > start:
                midpoints.append((start + end) / 2.0)
            start = None
    return midpoints


def detect_silence_midpoints(audio_path: str) -> List[float]:
    """Silence midpoints of the extracted source audio; part boundaries prefer
    these so a split never lands mid-utterance."""
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-i", audio_path,
           "-af", f"silencedetect=noise={SILENCE_DETECT_NOISE}:d={SILENCE_DETECT_MIN_D}",
           "-f", "null", "-"]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                              timeout=AUDIO_EXTRACT_TIMEOUT_SEC, check=False)
    except subprocess.TimeoutExpired:
        sys.stderr.write("[WARN] silencedetect timed out; boundaries stay unsnapped\n")
        return []
    return parse_silences((proc.stderr or b"").decode("utf-8", "replace"))


def extract_audio_part(video: str, out_path: str, start_ms: int, end_ms: int) -> None:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", video,
        "-ss", f"{start_ms / 1000.0:.3f}", "-t", f"{(end_ms - start_ms) / 1000.0:.3f}",
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "aac", "-b:a", "64k",
        "-y", out_path,
    ]
    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            timeout=AUDIO_EXTRACT_TIMEOUT_SEC, check=False,
        )
    except subprocess.TimeoutExpired:
        sys.stderr.write(f"[FATAL] ffmpeg audio extraction timed out for {out_path}\n")
        sys.exit(EXIT_EXTRACT_FAILED)
    if proc.returncode != 0 or not os.path.isfile(out_path):
        err = (proc.stderr or b"").decode("utf-8", "replace").strip().splitlines()
        tail = err[-1] if err else "unknown ffmpeg error"
        sys.stderr.write(f"[FATAL] ffmpeg failed to extract {out_path}: {tail}\n")
        sys.exit(EXIT_EXTRACT_FAILED)


def slice_audio_part(full_audio: str, out_path: str, start_ms: int, end_ms: int) -> None:
    """Cut a part out of the extracted source audio (stream copy; AAC frame
    precision ~20ms is far below the ±3s overlap budget)."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start_ms / 1000.0:.3f}", "-i", full_audio,
        "-t", f"{(end_ms - start_ms) / 1000.0:.3f}",
        "-c", "copy", "-y", out_path,
    ]
    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            timeout=AUDIO_EXTRACT_TIMEOUT_SEC, check=False,
        )
    except subprocess.TimeoutExpired:
        sys.stderr.write(f"[FATAL] ffmpeg audio slice timed out for {out_path}\n")
        sys.exit(EXIT_EXTRACT_FAILED)
    if proc.returncode != 0 or not os.path.isfile(out_path):
        err = (proc.stderr or b"").decode("utf-8", "replace").strip().splitlines()
        tail = err[-1] if err else "unknown ffmpeg error"
        sys.stderr.write(f"[FATAL] ffmpeg failed to slice {out_path}: {tail}\n")
        sys.exit(EXIT_EXTRACT_FAILED)


def locate_video(ws: Optional[str], override: Optional[str]) -> Optional[str]:
    if override:
        return override if os.path.isfile(override) else None
    if ws:
        mat = os.path.join(ws, "materials")
        if os.path.isdir(mat):
            for f in sorted(os.listdir(mat)):
                if f.lower().endswith((".mp4", ".mkv", ".mov", ".avi", ".webm", ".ts")):
                    return os.path.join(mat, f)
    return None


def make_workorder(video_path: str, duration_ms: Optional[int], parts: List[Tuple[int, int]],
                   num_speakers: Optional[int], language: Optional[str],
                   no_audio: bool, chunk_target_sec: Optional[float] = None,
                   boundaries: Optional[List[float]] = None) -> Dict[str, Any]:
    audio_dir = "audio"
    part_entries: List[Dict[str, Any]] = []
    if not no_audio:
        single = len(parts) == 1
        for i, (s, e) in enumerate(parts):
            suffix = "" if single else f".part{i:03d}"
            part_entries.append({
                "file": f"{audio_dir}/source_audio{suffix}.m4a",
                "start_ms": s,
                "end_ms": e,
                "output": f"{audio_dir}/omni_diarized{suffix}.json",
            })
    work: Dict[str, Any] = {
        "status": "no_audio_stream" if no_audio else "awaiting_omni_diarization",
        "video_path": video_path,
        "duration_ms": duration_ms,
        "mcp_tool": MCP_TOOL_NAME,
        "num_speakers_hint": num_speakers,
        "language_hint": language,
        "chunking": {
            "requested_chunk_sec": chunk_target_sec,
            "boundaries_sec": [round(b, 3) for b in (boundaries or [])],
            "overlap_ms": int(CHUNK_OVERLAP_SEC * 1000),
            "boundary_snap": "silence midpoint ±5s via ffmpeg silencedetect",
        },
        "expected_segment_schema": {"speaker": "<label>", "start": 0.0, "end": 0.0, "text": "<text>"},
        "parts": part_entries,
        "note": (
            "Path A (recommended): `speaker_diarize.py --workspace <ws> run` dials DashScope "
            "directly (needs DASHSCOPE_API_KEY; no client tool-timeout window; finished parts "
            "are kept - rerun `run` to retry only the missing ones). Path B (fallback): call "
            f"the MCP tool {MCP_TOOL_NAME} once per part with format='json' (pass num_speakers "
            "only when the bible states the cast size; omit language to auto-detect). Either "
            "way every result JSON lands verbatim at the matching 'output' path before `merge`."
            if not no_audio else
            "The video has no audio stream: skip the MCP call entirely and run "
            "`merge --empty-fallback` to emit an all-null speakers.json."
        ),
    }
    return work


def cmd_prepare(ws: Optional[str], video_override: Optional[str], num_speakers: Optional[int],
                language: Optional[str], chunk_target_sec: Optional[float] = None) -> None:
    require_binaries()
    video = locate_video(ws, video_override)
    if not video:
        sys.stderr.write("[FATAL] No video file found in materials/ (or --video path invalid)\n")
        sys.exit(EXIT_BAD_INPUT)

    duration_ms = probe_duration_ms(video)
    if duration_ms is None or duration_ms <= 0:
        sys.stderr.write(f"[FATAL] ffprobe could not read duration for {video}\n")
        sys.exit(EXIT_BAD_INPUT)
    has_audio = probe_has_audio(video)

    audio_dir = Path(ws, ".cache", "audio") if ws else Path(".cache", "audio")
    os.makedirs(audio_dir, exist_ok=True)
    full_audio = (audio_dir / "source_audio.m4a").resolve()
    if ws:
        try:
            full_audio.relative_to(Path(ws).resolve())
        except ValueError:
            sys.stderr.write(f"[FATAL] Audio path escaped the workspace containment: {full_audio}\n")
            sys.exit(2)

    boundaries: List[float] = []
    if has_audio:
        extract_audio_part(video, str(full_audio), 0, duration_ms)
        boundaries = plan_boundaries(duration_ms, chunk_target_sec)
        if boundaries:
            boundaries = snap_boundaries(boundaries, detect_silence_midpoints(str(full_audio)))
    parts = parts_from_boundaries(duration_ms, boundaries)
    work = make_workorder(video, duration_ms, parts, num_speakers, language,
                          no_audio=not has_audio, chunk_target_sec=chunk_target_sec,
                          boundaries=boundaries)

    if has_audio and len(parts) > 1:
        for i, (s, e) in enumerate(parts):
            out_path = (audio_dir / f"source_audio.part{i:03d}.m4a").resolve()
            if ws:
                try:
                    out_path.relative_to(Path(ws).resolve())
                except ValueError:
                    sys.stderr.write(f"[FATAL] Audio part path escaped the workspace containment: {out_path}\n")
                    sys.exit(2)
            slice_audio_part(str(full_audio), str(out_path), s, e)

    workorder_path = (audio_dir / "diarize_workorder.json").resolve()
    if ws:
        try:
            workorder_path.relative_to(Path(ws).resolve())
        except ValueError:
            sys.stderr.write(f"[FATAL] Workorder path escaped the workspace containment: {workorder_path}\n")
            sys.exit(2)
    workorder_path.write_text(json.dumps(work, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    sys.stdout.write(json.dumps(work, ensure_ascii=False, indent=2) + "\n")
    if not has_audio:
        sys.stderr.write("[WARN] No audio stream: diarization impossible; run `merge --empty-fallback` next\n")
    else:
        sys.stderr.write(
            f"[PENDING] Workorder written: {workorder_path}\n"
            f"[PENDING] Path A (recommended): speaker_diarize.py --workspace <ws> run   (direct API)\n"
            f"[PENDING] Path B (fallback): agent calls MCP {MCP_TOOL_NAME} per part (format='json'), saves\n"
            f"[PENDING] each JSON block verbatim to the listed 'output' paths; either way, then run `merge`.\n"
        )
    sys.exit(EXIT_AWAITING_OMNI)


def cmd_run(ws: Optional[str], force: bool = False) -> None:
    """Direct-API execution of a prepared workorder (path A): call DashScope per
    part via omni_client, save each result verbatim to the workorder's output
    path, keep finished parts (resume), and never let one part's failure stop
    the rest. Exits 0 when every output is present, 7 otherwise, 8 without a
    key while work is pending."""
    audio_dir = Path(ws, ".cache", "audio") if ws else Path(".cache", "audio")
    workorder_path = (audio_dir / "diarize_workorder.json").resolve()
    if ws:
        try:
            workorder_path.relative_to(Path(ws).resolve())
        except ValueError:
            sys.stderr.write(f"[FATAL] Workorder path escaped the workspace containment: {workorder_path}\n")
            sys.exit(2)
    if not workorder_path.is_file():
        sys.stderr.write(f"[FATAL] diarize_workorder.json not found: {workorder_path} (run `prepare` first)\n")
        sys.exit(EXIT_BAD_INPUT)
    try:
        work = json.loads(workorder_path.read_text(encoding="utf-8"))
    except Exception as e:
        sys.stderr.write(f"[FATAL] Cannot parse workorder {workorder_path}: {e}\n")
        sys.exit(EXIT_OMNI_OUTPUT_INVALID)
    if work.get("status") == "no_audio_stream":
        sys.stderr.write("[FATAL] Video has no audio stream; run `merge --empty-fallback` instead\n")
        sys.exit(EXIT_BAD_INPUT)
    parts = [p for p in work.get("parts", []) if isinstance(p, dict)]
    if not parts:
        sys.stderr.write("[FATAL] Workorder lists no parts\n")
        sys.exit(EXIT_OMNI_OUTPUT_INVALID)

    def contained(pth: Path, what: str) -> Path:
        if ws:
            try:
                pth.relative_to(Path(ws).resolve())
            except ValueError:
                sys.stderr.write(f"[FATAL] {what} escaped the workspace containment: {pth}\n")
                sys.exit(2)
        return pth

    def output_path(p: Dict[str, Any]) -> Path:
        return contained((audio_dir / os.path.basename(p["output"])).resolve(), "Omni output path")

    def part_audio_path(p: Dict[str, Any]) -> Path:
        return contained((audio_dir.parent / p["file"]).resolve(), "Audio part path")

    def is_valid_output(pth: Path) -> bool:
        try:
            data = json.loads(pth.read_text(encoding="utf-8"))
        except Exception:
            return False
        # A silent part is a completed part: re-dialling it would pay again for the
        # same answer on every `run`, and merge no longer rejects it either.
        return classify_omni_output(data) in ("ok", "silent")

    # Long parts stream for roughly 0.5-1x their audio length; warn before any spend.
    for p in parts:
        dur = (int(p.get("end_ms", 0)) - int(p.get("start_ms", 0))) / 1000.0
        if dur > RUN_PART_WARN_SEC:
            sys.stderr.write(
                f"[WARN] Part {os.path.basename(p['file'])} runs {dur / 60.0:.0f} min; a direct call may "
                f"stream for {dur / 120.0:.0f}-{dur / 60.0:.0f} min and can die at the timeout ceiling. If it "
                "does, re-prepare with --chunk-seconds 1200 or raise V2S_OMNI_TIMEOUT_SEC.\n")

    pending = [p for p in parts if force or not (output_path(p).is_file() and is_valid_output(output_path(p)))]
    key = omni_client.resolve_key()
    if pending and not key:
        sys.stderr.write(
            "[FATAL] DASHSCOPE_API_KEY is not set in the environment; direct calls are impossible.\n"
            "[FATAL] Options: (a) export DASHSCOPE_API_KEY=... and rerun `run`;\n"
            f"        (b) call MCP {MCP_TOOL_NAME} per the workorder (fallback path B);\n"
            "        (c) `merge --empty-fallback` to continue with all speakers null.\n")
        sys.exit(EXIT_MISSING_KEY)

    failed: List[str] = []
    for i, p in enumerate(parts):
        out_p = output_path(p)
        if not force and out_p.is_file() and is_valid_output(out_p):
            try:
                prior_state = classify_omni_output(json.loads(out_p.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001 - is_valid_output already parsed it
                prior_state = "ok"
            sys.stderr.write(
                f"[SKIP] part{i:03d}: {out_p.name} already present and valid"
                + (" (reported no speech)" if prior_state == "silent" else "")
                + " (--force to redo)\n")
            continue
        in_p = part_audio_path(p)
        if not in_p.is_file():
            sys.stderr.write(f"[ERROR] part{i:03d}: audio file missing: {in_p.name}\n")
            failed.append(f"part{i:03d} audio missing")
            continue
        dur = (int(p.get("end_ms", 0)) - int(p.get("start_ms", 0))) / 1000.0
        sys.stderr.write(f"[RUN ] part{i:03d}: {in_p.name} ({dur / 60.0:.1f} min) -> {out_p.name}\n")
        try:
            result = omni_client.diarize_audio_file(
                str(in_p), api_key=key, num_speakers=work.get("num_speakers_hint"),
                language=work.get("language_hint"), duration_sec=dur or None)
        except omni_client.OmniError as e:
            sys.stderr.write(f"[ERROR] part{i:03d}: {e}\n")
            failed.append(f"part{i:03d} {e.kind}" + (f" {e.status}" if e.status else ""))
            continue
        except Exception as e:  # noqa: BLE001 - one part must not stop the others
            sys.stderr.write(f"[ERROR] part{i:03d}: {type(e).__name__}: {str(e)[:200]}\n")
            failed.append(f"part{i:03d} {type(e).__name__}")
            continue
        write_json_atomic(out_p, result)
        state = classify_omni_output(result)
        sys.stderr.write(
            f"[OK  ] part{i:03d}: {len(result.get('segments') or [])} segments -> {out_p.name}\n")
        if state == "silent":
            sys.stderr.write(
                f"[WARN] part{i:03d}: Omni reported no speech for this part; it counts as done "
                "(rerun with --force only if the audio really had dialogue)\n")

    missing = [os.path.basename(p["output"]) for p in parts if not output_path(p).is_file()]
    if missing or failed:
        sys.stderr.write(
            f"[FATAL] run incomplete: {len(failed)} failed call(s); missing outputs: "
            f"{', '.join(missing) if missing else '(none saved)'}\n"
            "[FATAL] Fix the cause and rerun `run` - finished parts are kept (resume) - or fall\n"
            f"        back to MCP {MCP_TOOL_NAME} per the workorder / `merge --empty-fallback`.\n")
        sys.exit(EXIT_OMNI_OUTPUT_INVALID)
    sys.stderr.write(f"[DONE ] all {len(parts)} part output(s) present; next: `merge`\n")
    sys.exit(EXIT_OK)


# ---------------------------------------------------------------------------
# merge phase
# ---------------------------------------------------------------------------

def load_subtitle_items(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        sys.stderr.write(f"[FATAL] extracted.json not found: {path} (run subtitle_extractor.py first)\n")
        sys.exit(EXIT_BAD_INPUT)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        sys.stderr.write(f"[FATAL] Cannot parse {path}: {e}\n")
        sys.exit(EXIT_BAD_INPUT)
    items = data.get("items", []) if isinstance(data, dict) else (data or [])
    return [it for it in items if isinstance(it, dict)]


def normalize_omni_segments(data: Dict[str, Any], offset_ms: int) -> List[Dict[str, Any]]:
    """Extract speech turns from one saved Omni JSON block. Times arrive in
    SECONDS (verified against the tool source); converted to ms + part offset.
    Malformed segments are skipped and counted."""
    raw = []
    if isinstance(data, dict):
        raw = data.get("segments") or data.get("results") or []
    elif isinstance(data, list):
        raw = data
    turns: List[Dict[str, Any]] = []
    for seg in raw:
        if not isinstance(seg, dict):
            continue
        speaker = seg.get("speaker")
        try:
            start = float(seg.get("start"))
            end = float(seg.get("end"))
        except (TypeError, ValueError):
            continue
        if not speaker or end <= start or start < 0:
            continue
        turns.append({
            "raw_label": str(speaker),
            "start_ms": int(round(start * 1000)) + offset_ms,
            "end_ms": int(round(end * 1000)) + offset_ms,
            "text": str(seg.get("text") or ""),
        })
    turns.sort(key=lambda t: (t["start_ms"], t["end_ms"], t["raw_label"]))
    return turns


def classify_omni_output(data: Any) -> str:
    """Sort a saved Omni block into 'ok' | 'silent' | 'invalid'.

    'silent' means the model was given speech-free audio and said so - an explicit
    empty segments list. That distinction matters: treating it as invalid made `run`
    re-dial a music-only part on every invocation (paying again for the same answer)
    and made `merge` exit 7, so a chunked episode containing one silent stretch could
    never reach `merge` at all. Segments that are present but all unparseable stay
    'invalid', because that is a bad response rather than a quiet one."""
    if isinstance(data, dict):
        if "segments" in data:
            raw = data.get("segments")
        elif "results" in data:
            raw = data.get("results")
        else:
            return "invalid"
    elif isinstance(data, list):
        raw = data
    else:
        return "invalid"
    if not isinstance(raw, list):
        return "invalid"
    if not raw:
        return "silent"
    return "ok" if normalize_omni_segments(data, 0) else "invalid"


def build_cluster_map(turns: List[Dict[str, Any]]) -> Dict[Tuple[int, str], str]:
    """Link (part, raw_label) nodes into acoustic identities.

    Diarization labels are PART-LOCAL namespaces: "Speaker 2" in part 0 and in
    part 1 may be different people. The only trusted merge evidence is a shared
    utterance — turns from DIFFERENT parts whose absolute-time intervals overlap
    >= COOCCUR_MIN_RATIO of the shorter segment (both parts diarized the same
    audio inside the +-overlap window). Nodes without evidence stay separate:
    over-splitting is safe (the scene pass names clusters from context), wrong
    merging is not. Deterministic; union-find with transitive closure."""
    nodes = sorted({(t["part"], t["raw_label"]) for t in turns})
    parent: Dict[Tuple[int, str], Tuple[int, str]] = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_node: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
    for t in turns:
        by_node.setdefault((t["part"], t["raw_label"]), []).append(t)

    node_list = sorted(by_node)
    for i, na in enumerate(node_list):
        for nb in node_list[i + 1:]:
            if nb[0] == na[0]:
                continue  # same part: one node per raw label already
            linked = False
            for x in by_node[na]:
                for y in by_node[nb]:
                    ov = _overlap_ms(x["start_ms"], x["end_ms"], y["start_ms"], y["end_ms"])
                    shorter = max(1, min(x["end_ms"] - x["start_ms"], y["end_ms"] - y["start_ms"]))
                    if ov / shorter >= COOCCUR_MIN_RATIO:
                        ra, rb = find(na), find(nb)
                        if ra != rb:
                            parent[rb] = ra
                        linked = True
                        break
                if linked:
                    break

    comp_label: Dict[Tuple[int, str], str] = {}
    for t in sorted(turns, key=lambda t: (t["start_ms"], t["end_ms"], t["raw_label"])):
        root = find((t["part"], t["raw_label"]))
        if root not in comp_label:
            comp_label[root] = f"SPEAKER_A{len(comp_label) + 1}"
    return {n: comp_label[find(n)] for n in nodes}


def _norm_text(text: str) -> str:
    return _PUNCT_RE.sub("", str(text)).lower()


def _text_agreement(sub_text: str, turn_text: str) -> Optional[bool]:
    a, b = _norm_text(sub_text), _norm_text(turn_text)
    if not a or not b:
        return None
    return SequenceMatcher(None, a, b).ratio() >= TEXT_AGREEMENT_RATIO


def _overlap_ms(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def estimate_subtitle_offset_ms(items: List[Dict[str, Any]],
                                turns: List[Dict[str, Any]]) -> int:
    """Subtitles are only a ROUGH frame of the dialogue timeline, and fan-sourced
    subs are often systematically shifted against the audio (typically leading by
    1-3 s). The acoustic turns are the accurate timeline, so estimate ONE global
    shift — applied to subtitle windows — that maximizes total best-overlap
    against them: a lag sweep over the two timelines. Deterministic;
    returns 0 unless enough lines exist for a meaningful consensus."""
    if not items or not turns or len(items) < MIN_SUBTITLE_LINES_FOR_OFFSET:
        return 0
    span = int(SUBTITLE_OFFSET_SEARCH_SEC * 1000)
    step = SUBTITLE_OFFSET_STEP_MS
    turn_spans = [(t["start_ms"], t["end_ms"]) for t in turns]
    scored: List[Tuple[float, int, int]] = []  # (total, -abs(delta), delta)
    for delta in range(-span, span + 1, step):
        total = 0.0
        for it in items:
            s0 = int(it.get("start_ms", 0))
            # Read the length from the UNSHIFTED fields: defaulting end_ms to the
            # already-shifted start made the window length a function of the trial
            # delta, so a line without end_ms scored better the further it drifted.
            e0 = int(it.get("end_ms", s0))
            dur = max(1, e0 - s0)
            s = s0 + delta
            best = 0.0
            for ts, te in turn_spans:
                ov = _overlap_ms(s, s + dur, ts, te)
                if ov and ov / dur > best:
                    best = ov / dur
            total += best
        scored.append((round(total, 6), -abs(delta), delta))
    return max(scored)[2]


def bind_lines(items: List[Dict[str, Any]],
               turns: List[Dict[str, Any]],
               offset_ms: int = 0) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """Bind each subtitle line to the max-overlap voice cluster (turns carry
    their cluster_id from build_cluster_map). Line windows are shifted by the
    GLOBAL subtitle-vs-audio offset before matching, so systematic fan-sub drift
    cannot steal a line onto the previous speaker's trailing turn; reported
    start_ms/end_ms stay the original subtitle timecodes. Returns (segments rows,
    cluster_id -> contributing line dicts)."""
    rows: List[Dict[str, Any]] = []
    cluster_lines: Dict[str, List[Dict[str, Any]]] = {}
    for i, item in enumerate(items):
        start = int(item.get("start_ms", 0))
        end = int(item.get("end_ms", start))
        dur = max(1, end - start)
        text = str(item.get("text", ""))
        # corrected windows: subtitle frame shifted onto the acoustic timeline
        c_start, c_end = start + offset_ms, end + offset_ms

        # Overlap is accumulated PER CLUSTER, not per turn. Tracking the two biggest
        # individual turns lost a genuine second voice: two short turns of cluster A
        # together out-covering one long turn of cluster B still reported A alone, so
        # overlap dialogue silently dropped its co-speaker.
        per_cluster: Dict[str, Dict[str, Any]] = {}
        for t in turns:
            ov = _overlap_ms(c_start, c_end, t["start_ms"], t["end_ms"])
            if ov <= 0:
                continue
            acc = per_cluster.get(t["cluster_id"])
            if acc is None:
                per_cluster[t["cluster_id"]] = {"overlap": ov, "turn": t, "turn_overlap": ov}
            else:
                acc["overlap"] += ov
                if ov > acc["turn_overlap"]:
                    acc["turn"] = t
                    acc["turn_overlap"] = ov

        if not per_cluster:
            rows.append({
                "segment_id": i + 1, "start_ms": start, "end_ms": end,
                "speaker": None, "confidence": 0.0,
                "method": "no_speech_overlap", "cluster_id": None,
                "secondary_speaker": None, "text_agreement": None,
            })
            continue

        ranked = sorted(per_cluster.items(), key=lambda kv: (-kv[1]["overlap"], kv[0]))
        cluster_id, primary = ranked[0]
        ratio = primary["overlap"] / dur
        confidence = CONF_HIGH if ratio >= CONF_HIGH_RATIO else (
            CONF_MID if ratio >= CONF_MID_RATIO else CONF_LOW)

        secondary = None
        if len(ranked) > 1 and ranked[1][1]["overlap"] / dur >= SECONDARY_MIN_RATIO:
            secondary = ranked[1][0]

        rows.append({
            "segment_id": i + 1, "start_ms": start, "end_ms": end,
            "speaker": cluster_id,  # provisional for now; named below
            "confidence": confidence,
            "method": "acoustic_diarization", "cluster_id": cluster_id,
            "secondary_speaker": secondary,
            "text_agreement": _text_agreement(text, primary["turn"]["text"]),
        })
        cluster_lines.setdefault(cluster_id, []).append(item)

    return rows, cluster_lines


def name_clusters(cluster_lines: Dict[str, List[Dict[str, Any]]]) -> Tuple[Dict[str, Optional[str]], Dict[str, int]]:
    """Majority-vote naming: a cluster earns the metadata name that dominates its
    lines (share >= 0.6, votes >= 2). Metadata NEVER changes attribution - only
    the display name of the cluster the acoustics already formed."""
    names: Dict[str, Optional[str]] = {}
    manifest: Dict[str, int] = {}
    for cluster_id, lines in cluster_lines.items():
        votes: Counter = Counter()
        for item in lines:
            meta = item.get("speaker")
            if meta and not is_provisional_label(meta):
                votes[str(meta).strip()] += 1
        chosen: Optional[str] = None
        if votes:
            top, count = votes.most_common(1)[0]
            if count >= NAME_VOTE_MIN_COUNT and count / max(1, sum(votes.values())) >= NAME_VOTE_MIN_SHARE:
                chosen = top
        names[cluster_id] = chosen
        if chosen:
            manifest[chosen] = manifest.get(chosen, 0) + len(lines)
    return names, manifest


def build_output(video_name: str, items: List[Dict[str, Any]], rows: List[Dict[str, Any]],
                 cluster_lines: Dict[str, List[Dict[str, Any]]], turns: List[Dict[str, Any]],
                 names: Dict[str, Optional[str]],
                 manifest: Dict[str, int], parts_used: List[str],
                 empty_reason: Optional[str] = None,
                 subtitle_alignment: Optional[Dict[str, Any]] = None,
                 backend: Optional[str] = None,
                 part_metas: Optional[List[Optional[Dict[str, Any]]]] = None,
                 acoustic_enabled: bool = True) -> Dict[str, Any]:
    named_rows = []
    for r in rows:
        row = dict(r)
        if row["cluster_id"]:
            name = names.get(row["cluster_id"])
            row["speaker"] = name if name else row["cluster_id"]
        named_rows.append(row)

    cluster_parts: Dict[str, set] = {}
    for t in turns:
        cluster_parts.setdefault(t["cluster_id"], set()).add(t.get("part_file"))
    clusters = []
    for cluster_id in sorted(cluster_lines.keys(), key=lambda c: int(c.rsplit("A", 1)[1])):
        lines = cluster_lines[cluster_id]
        clusters.append({
            "cluster_id": cluster_id,
            "name": names.get(cluster_id),
            "line_count": len(lines),
            "speech_ms": sum(int(l.get("end_ms", 0)) - int(l.get("start_ms", 0)) for l in lines),
            "parts": sorted(cluster_parts.get(cluster_id, set())),
        })

    unattributed = sum(1 for r in named_rows if r["speaker"] is None)
    metas = [m for m in (part_metas or []) if isinstance(m, dict)]
    models = sorted({str(m["model"]) for m in metas if m.get("model")})
    return {
        "video_path": video_name,
        "total_segments": len(named_rows),
        # Distinct attributed identities, not line count and not the naming manifest:
        # len(manifest) reported 0 whenever acoustics ran but metadata could not name
        # a single cluster, which read as "no speakers found".
        "distinct_speakers_detected": len({r["speaker"] for r in named_rows if r["speaker"]}),
        "clusters_formed": len(cluster_lines),
        "unattributed_segments": unattributed,
        # Downstream reads this to decide whether a blank speaker column means
        # "acoustics found nobody" or "acoustics never ran" - it must not claim
        # clustering happened on a degraded artifact.
        "acoustic_clustering_enabled": acoustic_enabled,
        "attribution_method": ATTRIBUTION_METHOD,
        "degradation": empty_reason,
        "diarization_source": {
            "tool": MCP_TOOL_NAME if acoustic_enabled else "none",
            "model": ("; ".join(models) if models
                      else ("qwen3.8-omni-flash (MCP default)" if acoustic_enabled else "none")),
            "parts": parts_used,
            "backend": backend or ("mcp_tool" if acoustic_enabled else "none"),
        },
        "subtitle_alignment": subtitle_alignment,
        "characters_manifest": manifest,
        "clusters": clusters,
        "speech_turns": [
            {
                "cluster_id": t["cluster_id"],
                "raw_label": t["raw_label"],
                "part": t.get("part_file"),
                "start_ms": t["start_ms"], "end_ms": t["end_ms"],
                "omni_transcript": t["text"],
            }
            for t in turns
        ],
        "segments": named_rows,
    }


def _empty_output(video_name: str, items: List[Dict[str, Any]], reason: str) -> Dict[str, Any]:
    rows = [{
        "segment_id": i + 1,
        "start_ms": int(it.get("start_ms", 0)),
        "end_ms": int(it.get("end_ms", int(it.get("start_ms", 0)))),
        "speaker": None, "confidence": 0.0,
        "method": "unattributed_diarization_unavailable",
        "cluster_id": None, "secondary_speaker": None, "text_agreement": None,
    } for i, it in enumerate(items)]
    return build_output(video_name, items, rows, {}, [], {}, {}, [], empty_reason=reason,
                        acoustic_enabled=False)


def load_omni_parts(ws: Optional[str]) -> Tuple[List[str], List[Dict[str, Any]], List[Optional[Dict[str, Any]]]]:
    """Collect the saved Omni outputs per the workorder (or a legacy single
    omni_diarized.json), tag turns with their part, and link cross-part
    identities via build_cluster_map. Returns (used files, turns, per-part meta
    blocks - the direct-API path stamps provenance the MCP path lacks). Exit 7
    naming missing/invalid files."""
    audio_dir = Path(ws, ".cache", "audio") if ws else Path(".cache", "audio")
    workorder_path = (audio_dir / "diarize_workorder.json").resolve()
    if ws:
        try:
            workorder_path.relative_to(Path(ws).resolve())
        except ValueError:
            sys.stderr.write(f"[FATAL] Workorder path escaped the workspace containment: {workorder_path}\n")
            sys.exit(2)

    expectations: List[Tuple[str, int]] = []  # (filename, offset_ms)
    if os.path.isfile(workorder_path):
        try:
            with open(workorder_path, "r", encoding="utf-8") as f:
                work = json.load(f)
        except Exception as e:
            sys.stderr.write(f"[FATAL] Cannot parse workorder {workorder_path}: {e}\n")
            sys.exit(EXIT_OMNI_OUTPUT_INVALID)
        if work.get("status") == "no_audio_stream":
            sys.stderr.write("[FATAL] Workorder says the video has no audio stream; "
                             "run `merge --empty-fallback` instead\n")
            sys.exit(EXIT_BAD_INPUT)
        for p in work.get("parts", []):
            expectations.append((os.path.basename(p["output"]), int(p.get("start_ms", 0))))
    else:
        expectations.append(("omni_diarized.json", 0))

    used: List[str] = []
    turns: List[Dict[str, Any]] = []
    missing: List[str] = []
    metas: List[Optional[Dict[str, Any]]] = []
    for idx, (fname, offset) in enumerate(expectations):
        path = (audio_dir / fname).resolve()
        if ws:
            try:
                path.relative_to(Path(ws).resolve())
            except ValueError:
                sys.stderr.write(f"[FATAL] Omni output path escaped the workspace containment: {path}\n")
                sys.exit(2)
        if not path.is_file():
            missing.append(fname)
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            sys.stderr.write(f"[FATAL] Omni output {path} is not valid JSON: {e}\n")
            sys.exit(EXIT_OMNI_OUTPUT_INVALID)
        state = classify_omni_output(data)
        if state == "invalid":
            sys.stderr.write(f"[FATAL] Omni output {path} contains no usable segments "
                             f"(expected {{'segments': [{{speaker,start,end,text}}]}})\n")
            sys.exit(EXIT_OMNI_OUTPUT_INVALID)
        if state == "silent":
            sys.stderr.write(f"[WARN] Omni output {fname} reported no speech; its span stays "
                             "unattributed instead of failing the merge\n")
        part_turns = normalize_omni_segments(data, offset)
        for t in part_turns:
            t["part"] = idx
            t["part_file"] = fname
        meta = data.get("meta") if isinstance(data, dict) and isinstance(data.get("meta"), dict) else None
        metas.append(meta)
        used.append(fname)
        turns.extend(part_turns)
    if missing:
        sys.stderr.write("[FATAL] Missing Omni output(s) for: "
                         f"{', '.join(missing)}\n"
                         "[FATAL] Rerun `run` (direct API, keeps finished parts) or call MCP "
                         f"{MCP_TOOL_NAME} per the workorder, then rerun `merge`.\n")
        sys.exit(EXIT_OMNI_OUTPUT_INVALID)

    turns.sort(key=lambda t: (t["start_ms"], t["end_ms"], t["raw_label"]))
    cluster_map = build_cluster_map(turns) if turns else {}
    for t in turns:
        t["cluster_id"] = cluster_map[(t["part"], t["raw_label"])]
    return used, turns, metas


def cmd_merge(ws: Optional[str], subtitles_override: Optional[str], video_name: str,
              empty_fallback: bool) -> None:
    if subtitles_override:
        sub_path = Path(os.path.abspath(subtitles_override))
    else:
        sub_path = (Path(ws, ".cache", "subtitles", "extracted.json") if ws
                    else Path("extracted_subtitles.json")).resolve()
        if ws:
            try:
                sub_path.relative_to(Path(ws).resolve())
            except ValueError:
                sys.stderr.write(f"[FATAL] Subtitles path escaped the workspace containment: {sub_path}\n")
                sys.exit(2)
    items = load_subtitle_items(str(sub_path))

    if empty_fallback:
        sys.stderr.write("[WARN] --empty-fallback: emitting speakers.json with ALL speakers null "
                         "(MCP unavailable or no audio); speaker columns stay blank downstream\n")
        sys.stdout.write(json.dumps(
            _empty_output(video_name, items, "diarization_unavailable_empty_fallback"),
            ensure_ascii=False, indent=2) + "\n")
        return

    parts_used, turns, metas = load_omni_parts(ws)
    backend = ("direct_api" if any(isinstance(m, dict) and m.get("backend") == "direct_api"
                                   for m in metas) else "mcp_tool")

    # OP/ED windows (v0.5.1): songs are not characters. Turns sitting inside a
    # configured window are dropped BEFORE clustering, so the singer never
    # reaches characters_manifest or the writer's cast list.
    op_ed_windows = op_ed.load_windows(ws)
    if op_ed_windows and turns:
        before = len(turns)
        turns = [t for t in turns
                 if op_ed.matching_label(t["start_ms"], t["end_ms"], op_ed_windows) is None]
        if len(turns) != before:
            sys.stderr.write(f"[INFO] OP/ED filter: dropped {before - len(turns)} speech turn(s) "
                             "inside configured windows (songs are not characters)\n")

    if not turns:
        rows, cluster_lines = bind_lines(items, [])
        sys.stderr.write("[WARN] Diarization produced zero speech turns; all speakers null "
                         "(check the audio track / subtitle tier)\n")
        sys.stdout.write(json.dumps(
            build_output(video_name, items, rows, {}, [], {}, {}, parts_used,
                         empty_reason="omni_returned_no_speech",
                         backend=backend, part_metas=metas),
            ensure_ascii=False, indent=2) + "\n")
        return

    offset_ms = estimate_subtitle_offset_ms(items, turns)
    if offset_ms:
        sys.stderr.write(f"[INFO] Global subtitle-vs-audio offset: {offset_ms:+d} ms "
                         "(subtitle windows shifted before binding)\n")
    if abs(offset_ms) >= int(SUBTITLE_OFFSET_SEARCH_SEC * 1000):
        sys.stderr.write(
            f"[WARN] The best offset sits on the edge of the ±{SUBTITLE_OFFSET_SEARCH_SEC:.0f}s search "
            "window, so the real drift may be larger and binding is only partly corrected. Check the "
            "subtitle file's timebase against the video.\n")
    rows, cluster_lines = bind_lines(items, turns, offset_ms)
    alignment = {
        "offset_ms": offset_ms,
        "search_range_sec": SUBTITLE_OFFSET_SEARCH_SEC,
        "step_ms": SUBTITLE_OFFSET_STEP_MS,
        "method": "global_max_overlap_grid_search",
    }
    names, manifest = name_clusters(cluster_lines)
    sys.stdout.write(json.dumps(
        build_output(video_name, items, rows, cluster_lines, turns,
                     names, manifest, parts_used, subtitle_alignment=alignment,
                     backend=backend, part_metas=metas),
        ensure_ascii=False, indent=2) + "\n")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Acoustic speaker diarization bridge (Qwen3.8-Omni): direct DashScope `run` "
                    "or MCP omni_multi_speaker_asr fallback. Attribution=acoustic; "
                    "naming=metadata majority vote.")
    parser.add_argument("action", nargs="?", default=None, choices=["prepare", "run", "merge"],
                        help="prepare: extract audio + write workorder (exit 6). "
                             "run: call DashScope directly per part (needs DASHSCOPE_API_KEY). "
                             "merge: bind saved Omni outputs to subtitle lines -> speakers.json")
    parser.add_argument("--workspace", "-w", default=None, help="Workspace root directory")
    parser.add_argument("--video", "-v", default=None, help="Override video path (prepare)")
    parser.add_argument("--subtitles", "-s", default=None, help="Override extracted.json path (merge)")
    parser.add_argument("--num-speakers", type=int, default=None,
                        help="Optional diarization hint, only when the cast size is known")
    parser.add_argument("--language", default=None, help="Optional spoken-language hint (zh/en/ja/...)")
    parser.add_argument("--chunk-seconds", type=float, default=None,
                        help="prepare: target part length in seconds (silence-aligned, ±3s "
                             "overlap); use ~30-40 when single-part MCP calls time out")
    parser.add_argument("--empty-fallback", action="store_true",
                        help="merge only: emit all-null speakers.json (MCP unavailable / no audio)")
    parser.add_argument("--force", action="store_true",
                        help="run only: re-diagnose parts whose output files already exist")
    args = parser.parse_args()

    ws = os.path.abspath(args.workspace) if args.workspace else None
    video = locate_video(ws, args.video)
    video_name = os.path.basename(video) if video else "video_input"

    if args.action == "prepare":
        cmd_prepare(ws, args.video, args.num_speakers, args.language, args.chunk_seconds)
    elif args.action == "run":
        cmd_run(ws, force=args.force)
    elif args.action == "merge":
        cmd_merge(ws, args.subtitles, video_name, args.empty_fallback)
    else:
        parser.error("action required: prepare | run | merge (see --help)")


if __name__ == "__main__":
    main()
