#!/usr/bin/env python3
"""
scripts/speaker_diarize.py - Lightweight Dialogue & Speaker Labelling Engine

Pure text & metadata-driven speaker attribution:
1. Native Metadata: ASS Actor/Name/Style fields (carried through extracted.json "speaker").
2. Text Syntax: shared prefix parser (see subtitle_extractor.extract_speaker_from_text)
   for 【角色】/角色：/(角色) style prefixes.
3. Dash Alternation: contiguous runs of "-"/"—"/"–" prefixed lines are the classic
   two-party subtitle convention; they toggle between a STABLE pair of provisional
   labels (SPEAKER_00A / SPEAKER_00B) for the whole run, so A/B/A/B ordering survives.
4. Everything else stays unattributed (speaker=null) - silence gaps do NOT create
   phantom speakers; character identity is resolved downstream by LLM context
   attribution against characters_manifest.

Zero acoustic dependencies: no librosa, no pyannote, no torch, no audio extraction.
"""

import argparse
import json
import os
import re
import sys
from typing import List, Dict, Any, Tuple

try:
    from subtitle_extractor import extract_speaker_from_text
except ImportError:  # direct script execution from scripts/
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from subtitle_extractor import extract_speaker_from_text


SPEAKER_LABEL_RE = re.compile(r"^SPEAKER_[A-Z0-9]+$")


def is_provisional_label(speaker: Any) -> bool:
    """True for synthetic labels (SPEAKER_UNKNOWN / SPEAKER_00A ...) that must never
    be rendered as a character name in the final screenplay."""
    if speaker is None:
        return True
    s = str(speaker).strip()
    return (not s) or s in {"UNKNOWN", "SPEAKER_UNKNOWN"} or bool(SPEAKER_LABEL_RE.match(s))


def extract_speaker_tags(
    subtitles: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int], int]:
    """
    Attribution priority per line:
      1. subtitle metadata speaker (ASS Actor/Name)      -> confidence 0.95
      2. text syntax prefix (【角色】/角色：/(角色))      -> confidence 0.85
      3. dash-prefix alternation within a contiguous run -> confidence 0.55
      4. unattributed (speaker=null)                     -> confidence 0.30
    Returns (segments, characters_manifest, unattributed_count) where the
    manifest only counts genuinely attributed names (never provisional
    SPEAKER_* labels).
    """
    speakers: List[Dict[str, Any]] = []
    character_freq: Dict[str, int] = {}

    dash_run_id = -1
    dash_toggle = 0
    in_dash_run = False
    unattributed_count = 0

    for i, item in enumerate(subtitles):
        text = str(item.get("text", "")).strip()
        start_ms = int(item.get("start_ms", 0))
        end_ms = int(item.get("end_ms", start_ms + 1000))

        meta_speaker = item.get("speaker")
        if meta_speaker and str(meta_speaker).strip() and not is_provisional_label(meta_speaker):
            # A metadata/named line ends any open dash run: A/B alternation is
            # only meaningful across a CONTIGUOUS run of dash-prefixed lines.
            in_dash_run = False
            assigned = str(meta_speaker).strip()
            method = "subtitle_metadata"
            confidence = 0.95
        else:
            # Strip a dash prefix before prefix parsing so "-罗温：你好" still resolves.
            probe_text = text.lstrip("-—– ").strip()
            candidate, _clean = extract_speaker_from_text(probe_text)
            if candidate:
                assigned = candidate
                method = "text_syntax"
                confidence = 0.85
            elif text.startswith(("-", "—", "–")):
                # Two-party alternation: a fresh stable A/B pair per contiguous run.
                # The dash convention itself only supports "two voices alternate in
                # this block" - labels stay provisional (masked downstream); three
                # voices in one dash run are corrected by LLM context attribution.
                if not in_dash_run:
                    in_dash_run = True
                    dash_run_id += 1
                    dash_toggle = 0
                assigned = f"SPEAKER_{dash_run_id:02d}{'A' if dash_toggle == 0 else 'B'}"
                dash_toggle ^= 1
                method = "dash_alternation"
                confidence = 0.55
            else:
                in_dash_run = False
                assigned = None
                method = "unattributed"
                confidence = 0.30
                unattributed_count += 1

        if assigned is not None and not is_provisional_label(assigned):
            character_freq[assigned] = character_freq.get(assigned, 0) + 1

        speakers.append({
            "segment_id": i + 1,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "speaker": assigned,
            "confidence": confidence,
            "method": method
        })

    return speakers, character_freq, unattributed_count


def main():
    parser = argparse.ArgumentParser(description="Lightweight Dialogue & Speaker Labelling Engine")
    parser.add_argument("video", nargs="?", default=None, help="Path to video file (optional)")
    parser.add_argument("--subtitles", "-s", default=None, help="Path to extracted.json")
    parser.add_argument("--temp-wav", default=None, help="Deprecated/Ignored: no audio extraction needed")
    parser.add_argument("--workspace", "-w", default=None, help="Optional workspace root directory")

    args = parser.parse_args()
    ws = os.path.abspath(args.workspace) if args.workspace else None

    video_input = args.video
    if not video_input and ws:
        mat_dir = os.path.join(ws, "materials")
        if os.path.isdir(mat_dir):
            for f in sorted(os.listdir(mat_dir)):
                if f.lower().endswith((".mp4", ".mkv", ".mov", ".avi", ".webm")):
                    video_input = os.path.join(mat_dir, f)
                    break

    video_name = os.path.basename(video_input) if video_input else "video_input"
    default_sub = os.path.join(ws, ".cache", "subtitles", "extracted.json") if ws else "extracted_subtitles.json"
    subtitles_path = os.path.abspath(args.subtitles or default_sub)

    subtitles: List[Dict[str, Any]] = []
    if os.path.isfile(subtitles_path):
        try:
            with open(subtitles_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                subtitles = data.get("items", []) if isinstance(data, dict) else data
        except Exception as e:
            sys.stderr.write(f"[WARN] Failed to load subtitles from {subtitles_path}: {e}\n")
    else:
        sys.stderr.write(f"[WARN] Subtitles file not found: {subtitles_path}\n")

    segments, char_freq, unattributed_count = extract_speaker_tags(subtitles)

    result = {
        "video_path": video_name,
        "total_segments": len(segments),
        "distinct_speakers_detected": len(char_freq),
        "unattributed_segments": unattributed_count,
        "acoustic_clustering_enabled": False,
        "attribution_method": "pure_metadata_and_dialogue_syntax",
        "characters_manifest": char_freq,
        "segments": segments
    }

    sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
