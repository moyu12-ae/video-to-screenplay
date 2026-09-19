#!/usr/bin/env python3
"""
scripts/workspace.py - Workspace Manager, Asset Prober, and Sandbox Controller

Implements the single-project workspace architecture:
  <workspace>/
  ├── materials/       (read-only video and external subtitle assets)
  ├── output/          (final clean screenplay deliverables)
  └── .cache/          (hidden sandbox for intermediate json, keyframes, and debug artifacts)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.parse
from pathlib import Path
from typing import Dict, Any, List

import omni_client


def safe_workspace_path(path: str) -> str:
    """Normalize and validate workspace directory path."""
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path


def init_workspace(workspace_root: str) -> Dict[str, str]:
    """Create the standard project workspace layout."""
    ws = safe_workspace_path(workspace_root)
    dirs = {
        "root": ws,
        "materials": os.path.join(ws, "materials"),
        "output": os.path.join(ws, "output"),
        "cache": os.path.join(ws, ".cache"),
        "cache_subtitles": os.path.join(ws, ".cache", "subtitles"),
        "cache_visual": os.path.join(ws, ".cache", "visual"),
        "cache_keyframes": os.path.join(ws, ".cache", "visual", "keyframes"),
        "cache_audio": os.path.join(ws, ".cache", "audio"),
        "cache_alignment": os.path.join(ws, ".cache", "alignment"),
        "cache_debug": os.path.join(ws, ".cache", "debug"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def probe_materials(workspace_root: str) -> Dict[str, Any]:
    """
    Inspect the materials/ directory for video files and subtitle options.
    Returns structured discovery results to drive interactive user confirmation.
    """
    ws = safe_workspace_path(workspace_root)
    materials_dir = os.path.join(ws, "materials")
    if not os.path.isdir(materials_dir):
        return {
            "status": "missing_materials_dir",
            "videos": [],
            "subtitles": [],
            "error": "materials/ directory does not exist"
        }

    valid_video_exts = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".ts"}
    valid_sub_exts = {".srt", ".ass", ".ssa", ".vtt"}

    videos: List[str] = []
    subtitles: List[str] = []

    for entry in sorted(os.listdir(materials_dir)):
        full_path = os.path.join(materials_dir, entry)
        if not os.path.isfile(full_path):
            continue
        ext = os.path.splitext(entry)[1].lower()
        if ext in valid_video_exts:
            videos.append(os.path.abspath(full_path))
        elif ext in valid_sub_exts:
            subtitles.append(os.path.abspath(full_path))

    ffprobe_available = shutil.which("ffprobe") is not None
    if not ffprobe_available:
        sys.stderr.write(
            "[WARN] ffprobe not found on PATH - embedded subtitle streams cannot be "
            "detected (macOS: 'brew install ffmpeg'). Only external subtitle files "
            "will be reported.\n"
        )

    probe_result: Dict[str, Any] = {
        "workspace": ws,
        "materials_dir": materials_dir,
        "ffprobe_available": ffprobe_available,
        "video_count": len(videos),
        "videos": [os.path.basename(v) for v in videos],
        "external_subtitles": [os.path.basename(s) for s in subtitles],
        "video_probes": []
    }

    if not videos:
        return probe_result

    for v_path in videos:
        v_name = os.path.basename(v_path)
        stem = os.path.splitext(v_name)[0]
        v_info: Dict[str, Any] = {
            "video_file": v_name,
            "has_matching_external_sub": False,
            "matching_external_subs": [],
            "embedded_subtitle_streams": []
        }

        # Check matching external subs
        for s_path in subtitles:
            s_name = os.path.basename(s_path)
            if s_name.startswith(stem) or stem in s_name:
                v_info["has_matching_external_sub"] = True
                v_info["matching_external_subs"].append(s_name)

        # Probe container for embedded subtitle streams using literal binary name
        try:
            res = subprocess.run(
                [
                    "ffprobe",
                    "-v", "quiet",
                    "-print_format", "json",
                    "-show_streams",
                    v_path
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                timeout=15
            )
            if res.returncode == 0:
                data = json.loads(res.stdout.decode("utf-8", errors="ignore"))
                for stream in data.get("streams", []):
                    if stream.get("codec_type") == "subtitle":
                        tags = stream.get("tags", {})
                        v_info["embedded_subtitle_streams"].append({
                            "index": stream.get("index"),
                            "codec": stream.get("codec_name"),
                            "language": tags.get("language", "unknown"),
                            "title": tags.get("title", "")
                        })
        except Exception as e:
            v_info["probe_error"] = f"ffprobe failed: {e}"

        probe_result["video_probes"].append(v_info)

    return probe_result


def run_doctor_check(workspace: Path | str | None = None) -> None:
    """Inspect and report readiness of runtime tools, python audio-visual packages,
    and the acoustic-diarization API key (presence only - the key value NEVER
    enters the report, logs or any file)."""
    key = omni_client.resolve_key()
    base_host = urllib.parse.urlsplit(omni_client.resolve_base_url()).hostname or ""
    report = {
        "ffmpeg": {"ready": False, "path": shutil.which("ffmpeg"), "version": None},
        "ffprobe": {"ready": False, "path": shutil.which("ffprobe"), "version": None},
        "numpy": {"ready": False, "version": None},
        "diarization": {
            "ready": bool(key),
            "dashscope_api_key": "set" if key else "missing",
            "dashscope_base_url_host": base_host,
            "model": omni_client.resolve_model(),
            "note": "acoustic speaker diarization (Qwen3.8-Omni) needs DASHSCOPE_API_KEY; "
                    "without it, continue only after an explicit user choice (speakers stay blank)",
        },
        "platform": sys.platform
    }

    if report["ffmpeg"]["path"]:
        report["ffmpeg"]["ready"] = True
        try:
            res = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5, shell=False)
            first_line = res.stdout.splitlines()[0] if res.stdout else ""
            report["ffmpeg"]["version"] = first_line
        except Exception:
            pass

    if report["ffprobe"]["path"]:
        report["ffprobe"]["ready"] = True

    try:
        import numpy as np
        report["numpy"]["ready"] = True
        report["numpy"]["version"] = np.__version__
    except ImportError:
        report["numpy"]["ready"] = False

    sys.stdout.write(json.dumps({"status": "doctor_report", "report": report}, indent=2) + "\n")
    if not (report["ffmpeg"]["ready"] and report["ffprobe"]["ready"]):
        sys.stderr.write("[FATAL] FFmpeg and FFprobe are required on PATH.\n")
        sys.exit(3)


def enforce_subtitles_guard(mode: str | None = None, workspace: Path | None = None) -> None:
    """
    Fail fast if user or detector indicates pure visual video with no subtitles,
    or if subtitles are missing from the workspace.
    Dialogue is an absolute hard dependency for screenplay generation.
    """
    if mode in ("none", "no_subtitles", "pure_visual") or (mode is None and workspace is None):
        sys.stderr.write(
            "\n[FATAL BLOCKER] Screenplay generation strictly requires dialogue subtitles!\n"
            "Screenplay reverse-engineering cannot proceed on pure visual video without dialogue.\n"
            "Action required: Please add external subtitle files (.srt/.ass) to materials/\n"
            "or provide a video containing embedded subtitle tracks, then restart.\n\n"
        )
        sys.exit(5)

    if workspace:
        workspace = Path(workspace)
        # Check if workspace has subtitles in materials or .cache
        materials_dir = workspace / "materials"
        cache_dir = workspace / ".cache"
        found_subs = []
        for d in [materials_dir, cache_dir, workspace]:
            if d.is_dir():
                for ext in [".srt", ".ass", ".vtt", ".sub"]:
                    found_subs.extend(list(d.glob(f"*{ext}")))
        if not found_subs and mode not in ("embedded", "ocr"):
            sys.stderr.write(
                f"\n[FATAL BLOCKER] Subtitles are missing in workspace {workspace}!\n"
                "Screenplay generation strictly requires dialogue subtitles.\n"
                "Action required: Please add external subtitle files (.srt/.ass) to materials/\n"
                "or specify an embedded/ocr subtitle mode, then restart.\n\n"
            )
            sys.exit(5)


def main():
    parser = argparse.ArgumentParser(description="Workspace Manager for Video-to-Screenplay")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Init
    init_parser = subparsers.add_parser("init", help="Initialize project directory layout")
    init_parser.add_argument("--workspace", "-w", required=True, help="Path to workspace root")

    # Probe
    probe_parser = subparsers.add_parser("probe", help="Probe materials for video and subtitle tracks")
    probe_parser.add_argument("--workspace", "-w", required=True, help="Path to workspace root")

    # Guard check
    check_parser = subparsers.add_parser("check-subtitles", help="Enforce dialogue dependency guard")
    check_parser.add_argument("--mode", "-m", required=False, default=None,
                              choices=["external", "embedded", "ocr", "none"],
                              help="Selected subtitle mode")
    check_parser.add_argument("--workspace", "-w", default=None,
                              help="Path to workspace root to check for materials/subtitles")

    # Doctor preflight
    doctor_parser = subparsers.add_parser("doctor", help="Inspect runtime environment, audio-visual tools and model assets")
    doctor_parser.add_argument("--workspace", "-w", default=None, help="Optional workspace to inspect")

    args = parser.parse_args()

    if args.command == "doctor":
        run_doctor_check(args.workspace)
    elif args.command == "init":
        dirs = init_workspace(args.workspace)
        sys.stdout.write(json.dumps({"status": "initialized", "directories": dirs}, indent=2) + "\n")
    elif args.command == "probe":
        probe_res = probe_materials(args.workspace)
        sys.stdout.write(json.dumps(probe_res, ensure_ascii=False, indent=2) + "\n")
    elif args.command == "check-subtitles":
        enforce_subtitles_guard(args.mode, args.workspace)
        sys.stdout.write(json.dumps({"status": "approved", "mode": args.mode, "workspace": str(args.workspace) if args.workspace else None}) + "\n")


if __name__ == "__main__":
    main()
