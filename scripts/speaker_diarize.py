#!/usr/bin/env python3
"""
scripts/speaker_diarize.py - Acoustic Speaker Diarization Bridge (Qwen3.8-Omni)

Two-phase workflow around the `omni_multi_speaker_asr` MCP tool (Qwen-MM-Plugins,
default model qwen3.8-omni-flash), mirroring narrative_outline.py's workorder
pattern ("deterministic prep here, perception task delegated to the agent"):

  prepare : ffmpeg-extract 16 kHz mono audio (auto-chunked into <=45 min parts
            when the video runs past 50 min), write diarize_workorder.json with
            the exact MCP invocation + expected save paths, then EXIT 6 so the
            agent performs the MCP calls.
  merge   : validate the saved Omni outputs, restore part offsets, bind every
            subtitle line to the max-overlap voice cluster, name clusters by
            majority vote over subtitle metadata names, print speakers.json.

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
            4 audio extraction failed | 6 workorder written, awaiting MCP call
            7 Omni outputs missing/invalid/parts incomplete at merge stage.
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

SPEAKER_LABEL_RE = re.compile(r"^SPEAKER_[A-Z0-9]+$")

AUDIO_EXTRACT_TIMEOUT_SEC = 900.0
FFPROBE_TIMEOUT_SEC = 120.0
# Inline cap of the MCP endpoint (~55 min after its own MP3 re-fit); stay under it.
CHUNK_THRESHOLD_SEC = 3000.0   # 50 min: beyond this, split
CHUNK_TARGET_SEC = 2700.0      # 45 min per part
CHUNK_MERGE_TAIL_SEC = 600.0   # tails shorter than this fold into the previous part

EXIT_OK = 0
EXIT_BAD_INPUT = 1
EXIT_MISSING_FFMPEG = 3
EXIT_EXTRACT_FAILED = 4
EXIT_AWAITING_OMNI = 6
EXIT_OMNI_OUTPUT_INVALID = 7

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


def plan_parts(duration_ms: int) -> List[Tuple[int, int]]:
    """Split [0, duration_ms] into parts of <= CHUNK_TARGET_SEC, folding a short
    tail into the previous part. Pure function; deterministic."""
    duration_s = duration_ms / 1000.0
    if duration_s <= CHUNK_THRESHOLD_SEC:
        return [(0, int(duration_ms))]
    raw: List[Tuple[int, int]] = []
    start = 0.0
    while start < duration_s - 1e-6:
        end = min(start + CHUNK_TARGET_SEC, duration_s)
        raw.append((int(start * 1000), int(end * 1000)))
        start = end
    if len(raw) >= 2:
        last_start, last_end = raw[-1]
        if (last_end - last_start) / 1000.0 < CHUNK_MERGE_TAIL_SEC:
            prev_start = raw[-2][0]
            raw = raw[:-2] + [(prev_start, last_end)]
    return raw


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


def locate_video(ws: Optional[str], override: Optional[str]) -> Optional[str]:
    if override:
        return override if os.path.isfile(override) else None
    if ws:
        mat = os.path.join(ws, "materials")
        if os.path.isdir(mat):
            for f in sorted(os.listdir(mat)):
                if f.lower().endswith((".mp4", ".mkv", ".mov", ".avi", ".webm")):
                    return os.path.join(mat, f)
    return None


def make_workorder(video_path: str, duration_ms: Optional[int], parts: List[Tuple[int, int]],
                   num_speakers: Optional[int], language: Optional[str],
                   no_audio: bool) -> Dict[str, Any]:
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
        "expected_segment_schema": {"speaker": "<label>", "start": 0.0, "end": 0.0, "text": "<text>"},
        "parts": part_entries,
        "note": (
            "Call the MCP tool once per part with format='json' (pass num_speakers only when "
            "the bible states the cast size; omit language to auto-detect), then save each "
            "returned JSON block verbatim to the matching 'output' path."
            if not no_audio else
            "The video has no audio stream: skip the MCP call entirely and run "
            "`merge --empty-fallback` to emit an all-null speakers.json."
        ),
    }
    return work


def cmd_prepare(ws: Optional[str], video_override: Optional[str], num_speakers: Optional[int],
                language: Optional[str]) -> None:
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

    parts = plan_parts(duration_ms)
    work = make_workorder(video, duration_ms, parts, num_speakers, language, no_audio=not has_audio)

    if has_audio:
        for i, (s, e) in enumerate(parts):
            suffix = "" if len(parts) == 1 else f".part{i:03d}"
            out_path = (audio_dir / f"source_audio{suffix}.m4a").resolve()
            if ws:
                try:
                    out_path.relative_to(Path(ws).resolve())
                except ValueError:
                    sys.stderr.write(f"[FATAL] Audio part path escaped the workspace containment: {out_path}\n")
                    sys.exit(2)
            extract_audio_part(video, str(out_path), s, e)

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
            f"[PENDING] Next: agent calls MCP {MCP_TOOL_NAME} per part (format='json') and saves each\n"
            f"[PENDING] JSON block verbatim to the listed 'output' paths, then reruns `merge`.\n"
        )
    sys.exit(EXIT_AWAITING_OMNI)


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


def assign_cluster_labels(turns: List[Dict[str, Any]]) -> Dict[str, str]:
    """Map raw Omni labels to stable provisional labels in first-appearance
    (temporal) order: SPEAKER_A1, SPEAKER_A2, ... (matches ^SPEAKER_[A-Z0-9]+$)."""
    label_map: Dict[str, str] = {}
    for t in turns:
        raw = t["raw_label"]
        if raw not in label_map:
            label_map[raw] = f"SPEAKER_A{len(label_map) + 1}"
    return label_map


def _norm_text(text: str) -> str:
    return _PUNCT_RE.sub("", str(text)).lower()


def _text_agreement(sub_text: str, turn_text: str) -> Optional[bool]:
    a, b = _norm_text(sub_text), _norm_text(turn_text)
    if not a or not b:
        return None
    return SequenceMatcher(None, a, b).ratio() >= TEXT_AGREEMENT_RATIO


def _overlap_ms(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def bind_lines(items: List[Dict[str, Any]], turns: List[Dict[str, Any]],
               label_map: Dict[str, str]) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """Bind each subtitle line to the max-overlap voice cluster. Returns
    (segments rows, cluster_id -> contributing line dicts)."""
    rows: List[Dict[str, Any]] = []
    cluster_lines: Dict[str, List[Dict[str, Any]]] = {}
    for i, item in enumerate(items):
        start = int(item.get("start_ms", 0))
        end = int(item.get("end_ms", start))
        dur = max(1, end - start)
        text = str(item.get("text", ""))

        best: Optional[Dict[str, Any]] = None
        second: Optional[Dict[str, Any]] = None
        for t in turns:
            ov = _overlap_ms(start, end, t["start_ms"], t["end_ms"])
            if ov <= 0:
                continue
            cand = {"turn": t, "overlap": ov, "ratio": ov / dur}
            if best is None or ov > best["overlap"]:
                second = best
                best = cand
            elif second is None or ov > second["overlap"]:
                second = cand

        if best is None:
            rows.append({
                "segment_id": i + 1, "start_ms": start, "end_ms": end,
                "speaker": None, "confidence": 0.0,
                "method": "no_speech_overlap", "cluster_id": None,
                "secondary_speaker": None, "text_agreement": None,
            })
            continue

        turn = best["turn"]
        cluster_id = label_map[turn["raw_label"]]
        ratio = best["ratio"]
        confidence = CONF_HIGH if ratio >= CONF_HIGH_RATIO else (
            CONF_MID if ratio >= CONF_MID_RATIO else CONF_LOW)

        secondary = None
        if second is not None and second["turn"]["raw_label"] != turn["raw_label"] \
                and second["ratio"] >= SECONDARY_MIN_RATIO:
            secondary = label_map[second["turn"]["raw_label"]]

        rows.append({
            "segment_id": i + 1, "start_ms": start, "end_ms": end,
            "speaker": cluster_id,  # provisional for now; named below
            "confidence": confidence,
            "method": "acoustic_diarization", "cluster_id": cluster_id,
            "secondary_speaker": secondary,
            "text_agreement": _text_agreement(text, turn["text"]),
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
                 label_map: Dict[str, str], names: Dict[str, Optional[str]],
                 manifest: Dict[str, int], parts_used: List[str],
                 empty_reason: Optional[str] = None) -> Dict[str, Any]:
    named_rows = []
    for r in rows:
        row = dict(r)
        if row["cluster_id"]:
            name = names.get(row["cluster_id"])
            row["speaker"] = name if name else row["cluster_id"]
        named_rows.append(row)

    clusters = []
    for cluster_id in sorted(cluster_lines.keys(), key=lambda c: int(c.rsplit("A", 1)[1])):
        lines = cluster_lines[cluster_id]
        clusters.append({
            "cluster_id": cluster_id,
            "name": names.get(cluster_id),
            "line_count": len(lines),
            "speech_ms": sum(int(l.get("end_ms", 0)) - int(l.get("start_ms", 0)) for l in lines),
        })

    unattributed = sum(1 for r in named_rows if r["speaker"] is None)
    return {
        "video_path": video_name,
        "total_segments": len(named_rows),
        "distinct_speakers_detected": len(manifest),
        "unattributed_segments": unattributed,
        "acoustic_clustering_enabled": True,
        "attribution_method": ATTRIBUTION_METHOD,
        "degradation": empty_reason,
        "diarization_source": {
            "tool": MCP_TOOL_NAME,
            "model": "qwen3.8-omni-flash (MCP default)",
            "parts": parts_used,
        },
        "characters_manifest": manifest,
        "clusters": clusters,
        "speech_turns": [
            {
                "cluster_id": label_map[t["raw_label"]],
                "raw_label": t["raw_label"],
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
    return build_output(video_name, items, rows, {}, [], {}, {}, {}, [], empty_reason=reason)


def load_omni_parts(ws: Optional[str]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Collect the saved Omni outputs per the workorder (or a legacy single
    omni_diarized.json). Exit 7 naming the missing/invalid files."""
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
    blocks: List[Dict[str, Any]] = []
    missing: List[str] = []
    for fname, offset in expectations:
        path = (audio_dir / fname).resolve()
        if ws:
            try:
                path.relative_to(Path(ws).resolve())
            except ValueError:
                sys.stderr.write(f"[FATAL] Omni output path escaped the workspace containment: {path}\n")
                sys.exit(2)
        if not os.path.isfile(path):
            missing.append(fname)
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            sys.stderr.write(f"[FATAL] Omni output {path} is not valid JSON: {e}\n")
            sys.exit(EXIT_OMNI_OUTPUT_INVALID)
        turns = normalize_omni_segments(data, offset)
        if not turns:
            sys.stderr.write(f"[FATAL] Omni output {path} contains no usable segments "
                             f"(expected {{'segments': [{{speaker,start,end,text}}]}})\n")
            sys.exit(EXIT_OMNI_OUTPUT_INVALID)
        used.append(fname)
        blocks.append({"turns": turns})
    if missing:
        sys.stderr.write("[FATAL] Missing Omni output(s) for: "
                         f"{', '.join(missing)}\n"
                         f"[FATAL] Call MCP {MCP_TOOL_NAME} per the workorder and save each JSON "
                         "block verbatim, then rerun `merge`.\n")
        sys.exit(EXIT_OMNI_OUTPUT_INVALID)

    turns = [t for b in blocks for t in b["turns"]]
    turns.sort(key=lambda t: (t["start_ms"], t["end_ms"], t["raw_label"]))
    return used, turns


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

    parts_used, turns = load_omni_parts(ws)
    label_map = assign_cluster_labels(turns)
    rows, cluster_lines = bind_lines(items, turns, label_map)

    if not turns:
        sys.stderr.write("[WARN] Diarization produced zero speech turns; all speakers null "
                         "(check the audio track / subtitle tier)\n")
        sys.stdout.write(json.dumps(
            build_output(video_name, items, rows, {}, [], {}, {}, {}, parts_used,
                         empty_reason="omni_returned_no_speech"),
            ensure_ascii=False, indent=2) + "\n")
        return

    names, manifest = name_clusters(cluster_lines)
    sys.stdout.write(json.dumps(
        build_output(video_name, items, rows, cluster_lines, turns, label_map,
                     names, manifest, parts_used),
        ensure_ascii=False, indent=2) + "\n")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Acoustic speaker diarization bridge around MCP omni_multi_speaker_asr "
                    "(Qwen3.8-Omni). Attribution=acoustic; naming=metadata majority vote.")
    parser.add_argument("action", nargs="?", default=None, choices=["prepare", "merge"],
                        help="prepare: extract audio + write workorder (exit 6). "
                             "merge: bind saved Omni outputs to subtitle lines -> speakers.json")
    parser.add_argument("--workspace", "-w", default=None, help="Workspace root directory")
    parser.add_argument("--video", "-v", default=None, help="Override video path (prepare)")
    parser.add_argument("--subtitles", "-s", default=None, help="Override extracted.json path (merge)")
    parser.add_argument("--num-speakers", type=int, default=None,
                        help="Optional diarization hint, only when the cast size is known")
    parser.add_argument("--language", default=None, help="Optional spoken-language hint (zh/en/ja/...)")
    parser.add_argument("--empty-fallback", action="store_true",
                        help="merge only: emit all-null speakers.json (MCP unavailable / no audio)")
    args = parser.parse_args()

    ws = os.path.abspath(args.workspace) if args.workspace else None
    video = locate_video(ws, args.video)
    video_name = os.path.basename(video) if video else "video_input"

    if args.action == "prepare":
        cmd_prepare(ws, args.video, args.num_speakers, args.language)
    elif args.action == "merge":
        cmd_merge(ws, args.subtitles, video_name, args.empty_fallback)
    else:
        parser.error("action required: prepare | merge (see --help)")


if __name__ == "__main__":
    main()
