#!/usr/bin/env python3
"""
scripts/align_timeline.py - Millisecond Timecode Alignment & Audio-Visual Language Mapper

Merges extracted subtitle timestamps, FFmpeg scene cut boundaries, and acoustic speaker clusters
to construct a clean, unified event timeline. Accurately maps the three essential modes of audiovisual language:
1. DIALOGUE_EXCHANGE: Dialogue turns across shots (anchors conversations without breaking cuts).
2. PURE_VISUAL_SILENCE: Non-dialogue beats (>1.5s silence) prioritizing environmental & character staging.
3. OFF_SCREEN / AUDIO-VISUAL COUNTERPOINT: Flags lines where the speaker is outside frame ((O.S.) / (V.O.)).
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Any, Tuple


def interval_overlap_ms(start1: int, end1: int, start2: int, end2: int) -> int:
    """Calculate overlap duration in milliseconds between two time intervals."""
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    return max(0, overlap_end - overlap_start)


def interval_gap_ms(start1: int, end1: int, start2: int, end2: int) -> int:
    """Milliseconds separating two time intervals; 0 when they touch or overlap."""
    return max(0, start1 - end2, start2 - end1)


def format_timecode_ms(ms: int) -> str:
    h = ms // 3600000
    m = (ms % 3600000) // 60000
    s = (ms % 60000) // 1000
    rem = ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d}.{rem:03d}"


def infer_av_relationship(sub_item: Dict[str, Any], scene: Dict[str, Any]) -> str:
    """Infer the audio-visual dramatic relationship between dialogue line and camera shot."""
    text = sub_item.get("text", "")
    sub_start = sub_item.get("start_ms", 0)
    sub_end = sub_item.get("end_ms", 0)
    sub_dur = sub_end - sub_start

    scene_start = scene.get("start_ms", 0)
    scene_end = scene.get("end_ms", 0)

    # Internal thought / monologue heuristics
    if (text.startswith("(") and text.endswith(")")) or (text.startswith("（") and text.endswith("）")):
        return "INTERNAL_MONOLOGUE"

    # Non-diegetic narration / voice-over heuristics
    if any(k in text for k in ("下一集", "次回", "前情提要", "解说")):
        return "VOICE_OVER"

    # Cross-cut / reaction / L-cut: dialogue starts before shot or extends beyond shot
    overlap = interval_overlap_ms(sub_start, sub_end, scene_start, scene_end)
    if sub_dur > 0 and (overlap / sub_dur < 0.6):
        return "OFF_SCREEN"

    return "ON_SCREEN"


def scene_visual_verified(scene: Dict[str, Any], failed_keyframes: List[Any]) -> bool:
    """A macro scene keeps its visual ground truth iff at least one of its child
    shots has a keyframe that extracted successfully. failed_keyframes holds
    shot-level scene ids propagated from shots.json via scenes.json; a scene
    whose EVERY child failed (or which itself is listed, when child ids are
    absent) stays visually unverified and must not be described as fact."""
    if not failed_keyframes:
        return True
    failed = set(failed_keyframes)
    child_ids = scene.get("child_shot_ids") or []
    if child_ids:
        return not failed.issuperset(child_ids)
    sc_id = scene.get("scene_id") or scene.get("shot_id")
    return sc_id is not None and sc_id not in failed


MIN_OVERLAP_MS = 100


def assign_subtitles_to_shots(
    subtitles_data: List[Dict[str, Any]], scenes_data: List[Dict[str, Any]]
) -> Tuple[Dict[int, int], Dict[int, int], List[int]]:
    """Bind every dialogue cue to exactly ONE shot.

    Primary rule is maximal temporal overlap (> MIN_OVERLAP_MS), so a cue straddling
    a cut no longer duplicates into both neighbours. Cues that clear that bar nowhere
    — a sub-second sliver inside a shot, or a line past the final cut when duration
    probing failed — are bound to the temporally nearest shot rather than dropped:
    splice treats an unclaimed subtitle as fatal, and the writing pass cannot place
    a cue that never reached the manifest, so dropping one deadlocked the pipeline.

    Returns (shot index per cue, overlap per cue, cues bound by nearest-shot fallback).
    """
    best_overlap: Dict[int, int] = {}
    assignment: Dict[int, int] = {}
    for pos, scene in enumerate(scenes_data):
        sc_start = scene.get("start_ms", 0)
        sc_end = scene.get("end_ms", 0)
        for sub_idx, sub in enumerate(subtitles_data):
            ov = interval_overlap_ms(sc_start, sc_end, sub.get("start_ms", 0), sub.get("end_ms", 0))
            if ov > MIN_OVERLAP_MS and ov > best_overlap.get(sub_idx, 0):
                best_overlap[sub_idx] = ov
                assignment[sub_idx] = pos

    fallback: List[int] = []
    if scenes_data:
        for sub_idx in range(len(subtitles_data)):
            if sub_idx in assignment:
                continue
            sub = subtitles_data[sub_idx]
            s_start = sub.get("start_ms", 0)
            s_end = sub.get("end_ms", 0)
            nearest_pos, nearest_gap = None, None
            for pos, scene in enumerate(scenes_data):
                gap = interval_gap_ms(s_start, s_end, scene.get("start_ms", 0), scene.get("end_ms", 0))
                if nearest_gap is None or gap < nearest_gap:
                    nearest_gap, nearest_pos = gap, pos
            if nearest_pos is not None:
                assignment[sub_idx] = nearest_pos
                best_overlap.setdefault(sub_idx, 0)
                fallback.append(sub_idx)

    return assignment, best_overlap, fallback


def main():
    parser = argparse.ArgumentParser(
        description="Audio-Visual Language Timecode Alignment Engine",
        epilog="Both invocation styles work: positional (subtitles scenes) and flags (--subtitles/--scenes).",
    )
    parser.add_argument("subtitles_pos", nargs="?", default=None,
                        help="Positional: path to extracted_subtitles.json (same as --subtitles)")
    parser.add_argument("scenes_pos", nargs="?", default=None,
                        help="Positional: path to scenes.json (same as --scenes)")
    parser.add_argument("--scenes", "-c", default=None, help="Path to scenes.json")
    parser.add_argument("--subtitles", "-s", default=None, help="Path to extracted_subtitles.json")
    parser.add_argument("--speakers", "-p", default=None, help="Optional path to speakers.json")
    parser.add_argument("--workspace", "-w", default=None, help="Optional workspace root directory")

    args = parser.parse_args()

    if args.subtitles_pos and args.subtitles:
        parser.error("pass the subtitles path either positionally or via --subtitles, not both")
    if args.scenes_pos and args.scenes:
        parser.error("pass the scenes path either positionally or via --scenes, not both")

    ws = os.path.abspath(args.workspace) if args.workspace else None

    # Resolve default paths with workspace awareness
    default_sub = os.path.join(ws, ".cache", "subtitles", "extracted.json") if ws else "extracted_subtitles.json"
    default_scenes = os.path.join(ws, ".cache", "visual", "scenes.json") if ws else "scenes.json"
    default_speakers = os.path.join(ws, ".cache", "audio", "speakers.json") if ws else None

    subtitles_path = os.path.abspath(args.subtitles_pos or args.subtitles or default_sub)
    scenes_path = os.path.abspath(args.scenes_pos or args.scenes or default_scenes)
    speakers_path = os.path.abspath(args.speakers or default_speakers) if (args.speakers or default_speakers) else None

    scenes_data: List[Dict[str, Any]] = []
    failed_keyframes: List[Any] = []
    if os.path.isfile(scenes_path):
        try:
            with open(scenes_path, "r", encoding="utf-8") as f:
                d = json.load(f)
                if isinstance(d, dict):
                    scenes_data = d.get("scenes", [])
                    failed_keyframes = d.get("failed_keyframes", []) or []
                else:
                    scenes_data = d
        except Exception as e:
            sys.stderr.write(f"[WARN] Failed to load scenes from {scenes_path}: {e}\n")

    subtitles_data: List[Dict[str, Any]] = []
    if os.path.isfile(subtitles_path):
        try:
            with open(subtitles_path, "r", encoding="utf-8") as f:
                d = json.load(f)
                subtitles_data = d.get("items", []) if isinstance(d, dict) else d
        except Exception as e:
            sys.stderr.write(f"[WARN] Failed to load subtitles from {subtitles_path}: {e}\n")

    speakers_data: List[Dict[str, Any]] = []
    if speakers_path and os.path.isfile(speakers_path):
        try:
            with open(speakers_path, "r", encoding="utf-8") as f:
                d = json.load(f)
                speakers_data = d.get("segments", []) if isinstance(d, dict) else d
        except Exception as e:
            sys.stderr.write(f"[WARN] Failed to load speakers from {speakers_path}: {e}\n")

    # Assign each subtitle to exactly ONE shot: the one with maximal temporal overlap.
    # A cue straddling a cut (L-cut) previously duplicated into both adjacent shots.
    if subtitles_data and not scenes_data:
        sys.stderr.write(
            f"[FATAL] {len(subtitles_data)} dialogue cue(s) but no shots in {scenes_path}. "
            "Nothing can be aligned; run scene_detect.py and semantic_scene_grouper.py first.\n"
        )
        sys.exit(1)

    sub_shot_assignment, sub_best_overlap, fallback_cues = assign_subtitles_to_shots(
        subtitles_data, scenes_data)

    per_shot_sub_indices: Dict[int, List[int]] = {pos: [] for pos in range(len(scenes_data))}
    for sub_idx, pos in sub_shot_assignment.items():
        per_shot_sub_indices[pos].append(sub_idx)

    orphan_cues = len(subtitles_data) - len(sub_shot_assignment)
    if orphan_cues > 0:
        sys.stderr.write(
            f"[FATAL] {orphan_cues} dialogue cue(s) could not be bound to any shot. "
            "splice_screenplay.py treats an unclaimed subtitle as a fatal error, so the "
            "screenplay cannot be assembled from this timeline.\n"
        )
        sys.exit(1)
    if fallback_cues:
        sys.stderr.write(
            f"[WARN] {len(fallback_cues)} cue(s) had no shot with >{MIN_OVERLAP_MS}ms overlap "
            f"and were bound to the nearest shot instead (sub_index: "
            f"{[subtitles_data[i].get('index') for i in fallback_cues[:12]]}).\n"
        )

    aligned_shots = []

    for pos, scene in enumerate(scenes_data):
        sc_id = scene.get("scene_id") or scene.get("shot_id") or "SHOT_UNKNOWN"
        sc_start = scene.get("start_ms", 0)
        sc_end = scene.get("end_ms", 0)
        keyframe = scene.get("keyframe", "")

        matched_dialogues = []
        total_dialogue_dur = 0
        for sub_idx in per_shot_sub_indices[pos]:
            sub = subtitles_data[sub_idx]
            s_start = sub.get("start_ms", 0)
            s_end = sub.get("end_ms", 0)
            total_dialogue_dur += sub_best_overlap[sub_idx]

            matched_speaker = sub.get("speaker") or "SPEAKER_UNKNOWN"
            if matched_speaker == "SPEAKER_UNKNOWN":
                # Max-overlap beats first-match: bleeding subtitle lines would
                # otherwise inherit a temporal neighbour's acoustic speaker.
                best_overlap, best_speaker = 100, "SPEAKER_UNKNOWN"
                for spk in speakers_data:
                    cand = spk.get("speaker")
                    if not cand:
                        continue
                    ov = interval_overlap_ms(s_start, s_end, spk.get("start_ms", 0), spk.get("end_ms", 0))
                    if ov > best_overlap:
                        best_overlap, best_speaker = ov, cand
                matched_speaker = best_speaker

            av_rel = infer_av_relationship(sub, scene)

            matched_dialogues.append({
                "sub_index": sub.get("index"),
                "speaker": matched_speaker,
                "av_relationship": av_rel,
                "start_ms": s_start,
                "end_ms": s_end,
                "start_timecode": sub.get("start_timecode", format_timecode_ms(s_start)),
                "end_timecode": sub.get("end_timecode", format_timecode_ms(s_end)),
                "text": sub.get("text", "").strip()
            })

        shot_duration = max(1, sc_end - sc_start)
        shot_type = "DIALOGUE_SHOT" if matched_dialogues else "SILENT_ACTION"
        visual_verified = scene_visual_verified(scene, failed_keyframes)

        # Audiovisual language state representation
        distinct_spks = sorted(list(set(d["speaker"] for d in matched_dialogues if d.get("speaker"))))
        if matched_dialogues:
            av_mode = "DIALOGUE_EXCHANGE" if len(distinct_spks) > 1 else "MONOLOGUE"
        else:
            av_mode = "PURE_VISUAL_SILENCE"

        has_offscreen = any(d["av_relationship"] in ("OFF_SCREEN", "VOICE_OVER") for d in matched_dialogues)

        audiovisual_state = {
            "mode": av_mode,
            "silent_duration_ms": max(0, shot_duration - total_dialogue_dur),
            "has_offscreen_speech": has_offscreen,
            "distinct_speakers": distinct_spks
        }

        aligned_shots.append({
            "shot_id": sc_id,
            "shot_type": shot_type,
            "visual_verified": visual_verified,
            "audiovisual_state": audiovisual_state,
            "start_ms": sc_start,
            "end_ms": sc_end,
            "start_timecode": scene.get("start_timecode", format_timecode_ms(sc_start)),
            "end_timecode": scene.get("end_timecode", format_timecode_ms(sc_end)),
            "duration_ms": shot_duration,
            "slugline": scene.get("slugline", ""),
            "child_shot_ids": scene.get("child_shot_ids", []),
            "keyframe": keyframe,
            "dialogue_count": len(matched_dialogues),
            "dialogues": matched_dialogues
        })

    unverified = [s["shot_id"] for s in aligned_shots if not s["visual_verified"]]
    if unverified:
        sys.stderr.write(
            f"[WARN] {len(unverified)} shot(s) in the aligned timeline have "
            f"no verified keyframe: {unverified[:5]}. Their action descriptions must be marked unverified, never invented.\n"
        )

    summary = {
        "total_shots": len(aligned_shots),
        "dialogue_shots": sum(1 for s in aligned_shots if s["shot_type"] == "DIALOGUE_SHOT"),
        "silent_action_shots": sum(1 for s in aligned_shots if s["shot_type"] == "SILENT_ACTION"),
        "total_dialogue_cues": len(subtitles_data),
        "cues_assigned": len(sub_shot_assignment),
        "cues_nearest_shot_fallback": len(fallback_cues),
        "unverified_visual_shots": unverified,
        "shots": aligned_shots
    }

    # Output pure JSON to stdout
    sys.stdout.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
