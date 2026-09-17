#!/usr/bin/env python3
"""
scripts/scene_detect.py - Fast Headless Scene Cut Detection & Keyframe Extraction

Uses FFmpeg's scene detection filter (select='gt(scene,threshold)')
to discover shot boundaries and extract high-quality keyframe thumbnails
without CPU/GPU re-encoding overhead.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from typing import List, Dict, Any

SCENE_FILTER_TIMEOUT_SEC = 3600.0
DEFAULT_TIMEOUT_SEC = 120.0


def require_binaries() -> None:
    """Fail fast with an actionable message when FFmpeg tooling is absent from PATH."""
    missing = [b for b in ("ffmpeg", "ffprobe") if shutil.which(b) is None]
    if missing:
        sys.stderr.write(
            f"[FATAL] Missing required executable(s): {', '.join(missing)}. "
            "Please install FFmpeg and ensure it is in your system PATH "
            "(macOS: 'brew install ffmpeg', Windows: 'winget install Gyan.FFmpeg' or download from ffmpeg.org).\n"
        )
        sys.exit(3)


def get_video_duration(video_path: str) -> float:
    """Extract duration in seconds using ffprobe. Returns 0.0 when the probe fails."""
    clean_path = os.path.abspath(video_path)
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                clean_path,
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            shell=False,
            timeout=DEFAULT_TIMEOUT_SEC,
        ).strip()
        return float(out)
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError, ValueError):
        return 0.0


def detect_scene_cuts(video_path: str, threshold: float = 0.35) -> List[float]:
    """
    Run FFmpeg scene filter in null sink mode.
    Parses 'pts_time' from showinfo log lines to locate hard cut timestamps.
    Returns [0.0] when detection fails or times out, so the caller never blocks forever.
    """
    clean_video_path = os.path.abspath(video_path)
    clean_thresh = f"{float(threshold):.2f}"

    try:
        proc = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostats",
                "-i", clean_video_path,
                "-filter:v", f"select='gt(scene,{clean_thresh})',showinfo",
                "-f", "null",
                "-",
            ],
            stderr=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            text=True,
            shell=False,
        )
    except OSError as e:
        sys.stderr.write(f"[WARN] Could not launch ffmpeg for scene detection: {e}\n")
        return [0.0]

    try:
        _, stderr_text = proc.communicate(timeout=SCENE_FILTER_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        sys.stderr.write(
            f"[WARN] Scene detection exceeded {SCENE_FILTER_TIMEOUT_SEC:.0f}s and was aborted; "
            "falling back to a single whole-video scene.\n"
        )
        return [0.0]

    cut_timestamps: List[float] = [0.0]
    time_regex = re.compile(r"pts_time:(-?[0-9]+\.?[0-9]*)")

    for line in (stderr_text or "").splitlines():
        if "showinfo" in line and "pts_time:" in line:
            m = time_regex.search(line)
            if m:
                ts = float(m.group(1))
                if ts < 0:
                    continue
                if not cut_timestamps or ts - cut_timestamps[-1] >= 0.5:
                    cut_timestamps.append(ts)

    return cut_timestamps


def extract_keyframe(video_path: str, timestamp: float, output_path: str) -> bool:
    """Extract a single frame image at timestamp. Returns False instead of raising on failure."""
    clean_path = os.path.abspath(output_path)
    try:
        os.makedirs(os.path.dirname(clean_path), exist_ok=True)
    except OSError:
        return False
    ts_val = f"{float(timestamp):.3f}"
    try:
        ret = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostats",
                "-ss", ts_val,
                "-i", os.path.abspath(video_path),
                "-vframes", "1",
                "-q:v", "2",
                "-y",
                clean_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False,
            timeout=DEFAULT_TIMEOUT_SEC,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    if ret.returncode != 0:
        return False
    return os.path.isfile(clean_path)


def format_timecode(seconds: float) -> str:
    total_ms = int(seconds * 1000)
    h = total_ms // 3600000
    m = (total_ms % 3600000) // 60000
    s = (total_ms % 60000) // 1000
    ms = total_ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def main():
    parser = argparse.ArgumentParser(description="FFmpeg Headless Scene Detection")
    parser.add_argument("video", nargs="?", default=None, help="Path to video file (or auto-detected from workspace)")
    parser.add_argument("--threshold", "-t", type=float, default=0.35, help="Scene change sensitivity [0.2-0.5]")
    parser.add_argument("--keyframes-dir", "-k", default=None, help="Directory for keyframes (defaults to .cache/visual/keyframes under workspace, or ./keyframes)")
    parser.add_argument("--workspace", "-w", default=None, help="Optional workspace directory for sandbox path resolution")
    args = parser.parse_args()

    # Workspace-aware path resolution
    ws = os.path.abspath(args.workspace) if args.workspace else None
    video_input = args.video

    if not video_input and ws:
        # Auto-discover video in materials/
        mat_dir = os.path.join(ws, "materials")
        if os.path.isdir(mat_dir):
            valid_exts = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".ts"}
            candidates = sorted(
                f for f in os.listdir(mat_dir) if os.path.splitext(f)[1].lower() in valid_exts
            )
            if len(candidates) > 1:
                sys.stderr.write(
                    f"[WARN] materials/ holds {len(candidates)} videos; auto-discovery picked "
                    f"'{candidates[0]}'. Pass the video explicitly to process another one.\n"
                )
            for f in candidates:
                video_input = os.path.join(mat_dir, f)
                break

    if not video_input:
        sys.stderr.write("Error: Video file path must be specified directly or discoverable in workspace materials/\n")
        sys.exit(1)

    video_path = os.path.abspath(video_input)
    if not os.path.isfile(video_path):
        sys.stderr.write(f"Error: Video file not found: {video_path}\n")
        sys.exit(1)

    # Determine keyframes directory
    if args.keyframes_dir:
        target_kf_dir = args.keyframes_dir
    elif ws:
        target_kf_dir = os.path.join(ws, ".cache", "visual", "keyframes")
    else:
        target_kf_dir = "keyframes"

    # Result JSON always goes to stdout; the documented SKILL flow redirects it into
    # .cache/visual/shots.json (keeps this script a pure UNIX filter, no file writes).

    require_binaries()

    duration = get_video_duration(video_path)
    if duration <= 0.0:
        sys.stderr.write(
            f"[WARN] Could not read a valid duration for {os.path.basename(video_path)}; "
            "scene boundaries may be incomplete.\n"
        )

    cut_timestamps = detect_scene_cuts(video_path, args.threshold)
    if duration > 0.0 and cut_timestamps[-1] < duration:
        cut_timestamps.append(duration)

    scenes = []
    keyframes_dir = os.path.abspath(target_kf_dir)
    try:
        os.makedirs(keyframes_dir, exist_ok=True)
    except OSError as e:
        sys.stderr.write(f"[FATAL] Cannot create keyframes directory {keyframes_dir}: {e}\n")
        sys.exit(4)
    if not os.access(keyframes_dir, os.W_OK):
        sys.stderr.write(f"[FATAL] Keyframes directory is not writable: {keyframes_dir}\n")
        sys.exit(4)
    failed_keyframes: List[int] = []

    for i in range(len(cut_timestamps) - 1):
        scene_id = i + 1
        start_sec = cut_timestamps[i]
        end_sec = cut_timestamps[i + 1]
        start_ms = int(start_sec * 1000)
        end_ms = int(end_sec * 1000)

        mid_sec = start_sec + min(0.3, (end_sec - start_sec) / 2.0)
        fname = f"shot_{scene_id:04d}_{start_ms:06d}ms.jpg"
        target_frame = os.path.join(keyframes_dir, fname)

        if not extract_keyframe(video_path, mid_sec, target_frame):
            failed_keyframes.append(scene_id)

        scenes.append({
            "scene_id": scene_id,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "start_timecode": format_timecode(start_sec),
            "end_timecode": format_timecode(end_sec),
            "duration_ms": end_ms - start_ms,
            "keyframe": fname
        })

    if failed_keyframes:
        shown = failed_keyframes[:10]
        more = "" if len(failed_keyframes) == len(shown) else f" (+{len(failed_keyframes) - len(shown)} more)"
        sys.stderr.write(
            f"[WARN] {len(failed_keyframes)}/{len(scenes)} keyframe(s) failed to extract "
            f"for scene_id {shown}{more}. Downstream phases must not claim visual ground truth "
            "for those scenes.\n"
        )

    result = {
        "video_path": os.path.basename(video_path),
        "total_duration_sec": duration,
        "total_scenes": len(scenes),
        "threshold": args.threshold,
        "keyframes_dir": keyframes_dir,
        "failed_keyframes": failed_keyframes,
        "scenes": scenes
    }

    # Output pure JSON to stdout (standard UNIX filter pattern)
    sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
