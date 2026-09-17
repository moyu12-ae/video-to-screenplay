#!/usr/bin/env python3
"""
scripts/semantic_scene_grouper.py - Semantic Scene Clustering & Dynamic Programming Solver

Implements the multi-modal movie scene segmentation framework inspired by LGSS (CVPR 2020),
adapted for lightweight, cross-platform (Mac/Win) execution without heavy local neural models.

Key Pipeline:
1. Ingests atomic physical shots (from FFmpeg scene cut detection) and dialogue subtitles.
2. Applies dialogue-continuity soft constraints (L-cut / J-cut protection: a large
   finite penalty, never a hard ban - a hard INF deadlocks the DP into a silent
   single-scene fallback on dialogue-dense episodes).
3. Computes multi-modal boundary transition affinity: silence gap, shot duration,
   punctuation closure, and an LGSS-'place' proxy (keyframe HSV histogram change,
   optional numpy/opencv, degrades gracefully).
4. Solves global optimal scene partition via a 1D Dynamic Programming (DP) algorithm.
5. Emits macro-level screenplay scenes (SCENE_01, SCENE_02...) grouping atomic child shots.

Framework inspired by LGSS (A Local-to-Global Approach to Multi-modal Movie Scene
Segmentation, CVPR 2020), re-implemented as a lightweight DP adaptation that runs
on pure wheels (no place/cast/act/audio model dependencies): the "place" modality
is approximated by classical HSV histogram statistics, and the "cast/audio"
modalities by dialogue turn continuity and silence gaps.
"""

import argparse
import json
import math
import os
import sys
from typing import List, Dict, Any, Set, Tuple, Optional


def load_json_file(file_path: str) -> Any:
    """Load JSON from an absolute or verified file path."""
    clean_path = os.path.abspath(file_path)
    if not os.path.isfile(clean_path):
        raise FileNotFoundError(f"Required data file not found: {clean_path}")
    with open(clean_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_timecode(ms: int) -> str:
    """Format milliseconds into HH:MM:SS.mmm timecode."""
    total_sec = ms // 1000
    milli = ms % 1000
    s = total_sec % 60
    m = (total_sec // 60) % 60
    h = total_sec // 3600
    return f"{h:02d}:{m:02d}:{s:02d}.{milli:03d}"


class SemanticSceneGrouper:
    # Large finite cost replacing the former hard INF ban on dialogue-protected
    # boundaries. Must exceed the maximum affinity reward (8.0) plus typical
    # segment-cost differentials so protected cuts remain strictly discouraged.
    PROHIBITED_PENALTY = 12.0
    # A narrative sequence wall (from narrative_structure.json) snaps to the
    # nearest physical shot cut within this window; the wall is conceptual
    # (it lives in the dialogue stream), the AV cut is where scenes actually break.
    SEQ_WALL_SNAP_MS = 15000

    def __init__(
        self,
        shots_data: List[Dict[str, Any]],
        subtitles_data: List[Dict[str, Any]],
        bible_data: Optional[Dict[str, Any]] = None,
        min_scenes: int = 8,
        max_scenes: int = 35,
        target_scenes: Optional[int] = None,
        keyframes_dir: Optional[str] = None,
        outline_data: Optional[Dict[str, Any]] = None,
        failed_keyframes: Optional[List[Any]] = None,
    ):
        self.shots = sorted(shots_data, key=lambda x: x.get("start_ms", 0))
        self.subtitles = sorted(subtitles_data, key=lambda x: x.get("start_ms", 0))
        self.bible = bible_data or {}
        self.min_scenes = min_scenes
        self.max_scenes = max_scenes
        self.target_scenes = target_scenes
        self.keyframes_dir = keyframes_dir
        self.outline = outline_data or {}
        # Shot-level keyframe extraction failures (from shots.json) propagated so
        # downstream stages keep the "visually unverified" honesty flag alive.
        self.failed_keyframes = list(failed_keyframes or [])

    def compute_dialogue_hard_constraints(self) -> Set[int]:
        """
        Identify shot boundaries that MUST NOT be cut because active speech or tight dialogue
        crosses the boundary (L-cut / J-cut preservation).
        Returns a set of 0-based boundary indices (where boundary i is between shot i and shot i+1).
        """
        prohibited_boundaries: Set[int] = set()
        n_shots = len(self.shots)
        if n_shots <= 1:
            return prohibited_boundaries

        # For each boundary between shot i and shot i+1 (time = cut_time_ms)
        for i in range(n_shots - 1):
            cut_time_ms = self.shots[i].get("end_ms", 0)
            if cut_time_ms <= 0:
                cut_time_ms = self.shots[i + 1].get("start_ms", 0)

            # Check 1: Does any single dialogue utterance span across the cut point?
            for sub in self.subtitles:
                sub_start = sub.get("start_ms", 0)
                sub_end = sub.get("end_ms", 0)
                # Overlap with boundary (speaker is talking when the visual cut happens)
                if sub_start < cut_time_ms - 200 and sub_end > cut_time_ms + 200:
                    prohibited_boundaries.add(i)
                    break

            if i in prohibited_boundaries:
                continue

            # Check 2: Very tight conversational exchange (< 600ms gap) between adjacent shots
            pre_subs = [s for s in self.subtitles if s.get("end_ms", 0) <= cut_time_ms and s.get("end_ms", 0) > cut_time_ms - 800]
            post_subs = [s for s in self.subtitles if s.get("start_ms", 0) >= cut_time_ms and s.get("start_ms", 0) < cut_time_ms + 800]
            if pre_subs and post_subs:
                # If conversational gap is very small, protect boundary
                gap = post_subs[0].get("start_ms", 0) - pre_subs[-1].get("end_ms", 0)
                if gap < 500:
                    prohibited_boundaries.add(i)

        return prohibited_boundaries

    def compute_visual_dissimilarities(self) -> Optional[List[float]]:
        """
        LGSS 'place' modality, lightweight edition: HSV-histogram intersection
        distance between the keyframes on both sides of each boundary
        (0.0 = identical palette, 1.0 = maximal palette change). Uses only
        numpy + opencv-python-headless (pre-compiled wheels, zero models).
        Returns None when deps or keyframes are unavailable - the solver then
        degrades to dialogue/silence heuristics only.
        """
        try:
            import cv2
            import numpy as np
        except ImportError:
            sys.stderr.write("[INFO] numpy/opencv not installed - visual place affinity disabled.\n")
            return None
        if not self.keyframes_dir or not os.path.isdir(self.keyframes_dir):
            return None

        hists: List[Optional[Any]] = []
        for shot in self.shots:
            fname = str(shot.get("keyframe") or "")
            hist = None
            path = os.path.join(self.keyframes_dir, fname)
            if fname and os.path.isfile(path):
                img = cv2.imread(path)
                if img is not None:
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    # Hue carries the "place changed" signal in film footage; global
                    # saturation/value distributions stay similar across shots, so
                    # hue is weighted 3x. Weighted channels are L1-normalized to 1.
                    weights = (0.6, 0.2, 0.2)
                    parts = []
                    for (ch, rng), w in zip(((0, [0, 180]), (1, [0, 256]), (2, [0, 256])), weights):
                        h = cv2.calcHist([hsv], [ch], None, [8], rng).flatten()
                        total = float(h.sum())
                        parts.append((h / total * w) if total > 0 else h * 0.0)
                    hist = np.concatenate(parts)
                    if hist.sum() <= 0:
                        hist = None
            hists.append(hist)

        dissim: List[float] = [0.0] * max(0, len(self.shots) - 1)
        for i in range(len(self.shots) - 1):
            a, b = hists[i], hists[i + 1]
            if a is None or b is None:
                continue  # missing keyframe -> neutral, never fabricate affinity
            intersection = float(np.sum(np.minimum(a, b)))  # in [0, 1]
            dissim[i] = max(0.0, min(1.0, 1.0 - intersection))
        return dissim

    def compute_boundary_affinity_scores(self, visual_dissim: Optional[List[float]] = None) -> List[float]:
        """
        Calculate semantic scene boundary transition affinity for each boundary i
        (between shot i and shot i+1). Higher score = stronger evidence of a macro
        dramatic scene change (place shift, time jump, silence). Range: [0.0, 1.0].
        Dialogue-protected boundaries are not marked here; the DP solver applies a
        large finite soft penalty so it can never deadlock into one giant scene.
        """
        n_shots = len(self.shots)
        if n_shots <= 1:
            return []

        scores: List[float] = [0.0] * (n_shots - 1)

        for i in range(n_shots - 1):
            shot_left = self.shots[i]
            shot_right = self.shots[i + 1]
            cut_time_ms = shot_left.get("end_ms", shot_right.get("start_ms", 0))

            score = 0.20  # Base transition affinity

            # Factor 1: Audio / Dialogue Silence Gap
            # Look for silence duration around the boundary
            pre_subs = [s for s in self.subtitles if s.get("end_ms", 0) <= cut_time_ms]
            post_subs = [s for s in self.subtitles if s.get("start_ms", 0) >= cut_time_ms]
            last_pre_end = pre_subs[-1].get("end_ms", 0) if pre_subs else 0
            first_post_start = post_subs[0].get("start_ms", 0) if post_subs else cut_time_ms + 10000
            silence_ms = max(0, first_post_start - last_pre_end)

            if silence_ms > 4000:
                score += 0.35  # Extended silence strongly indicates scene transition / beat pause
            elif silence_ms > 2000:
                score += 0.20

            # Factor 2: Shot Duration Characteristics
            # Establishing shots or lingering concluding shots before cuts
            dur_left = shot_left.get("duration_ms", 0)
            dur_right = shot_right.get("duration_ms", 0)
            if dur_left > 5000 or dur_right > 5000:
                score += 0.15

            # Factor 3: Punctuation and Subtitle Content cues
            if pre_subs and post_subs:
                pre_text = pre_subs[-1].get("text", "")
                # Ending with period, ellipsis, or question mark suggests closure
                if any(pre_text.endswith(p) for p in ("。", "...", "！", "？", ".", "!", "?")):
                    score += 0.10

            # Factor 4: LGSS 'place' proxy - keyframe palette change across boundary
            if visual_dissim is not None and i < len(visual_dissim):
                score += 0.35 * visual_dissim[i]

            scores[i] = min(1.0, max(0.0, score))

        return scores

    def compute_sequence_partitions(self) -> Optional[List[Dict[str, Any]]]:
        """
        Convert the narrative outline (McKee sequence layer) into snapped shot-index
        partitions, one per sequence. Returns None when no usable outline is present
        (the caller then flat-solves exactly as before). Each conceptual wall sits at
        the midpoint between a sequence's last cue end and the next sequence's first
        cue start; it snaps to the nearest physical shot cut (boundary snapping: the
        turning point picks the closest AV cut, distance is reported).
        """
        seqs = (self.outline or {}).get("sequences") or []
        if not seqs:
            return None
        by_index = {int(s.get("index", s.get("id", 0))): s for s in self.subtitles}
        n = len(self.shots)
        if n == 0 or not by_index:
            return None
        first = by_index.get(int(seqs[0].get("start_sub", -1)))
        last = by_index.get(int(seqs[-1].get("end_sub", -1)))
        if first is None or last is None:
            sys.stderr.write("[WARN] narrative outline: sub indices missing in extracted.json - flat solving.\n")
            return None

        walls: List[float] = []
        for k in range(len(seqs) - 1):
            prev = by_index.get(int(seqs[k].get("end_sub", -1)))
            nxt = by_index.get(int(seqs[k + 1].get("start_sub", -1)))
            if prev is None or nxt is None:
                sys.stderr.write("[WARN] narrative outline: sub indices missing in extracted.json - flat solving.\n")
                return None
            walls.append((float(prev.get("end_ms", 0)) + float(nxt.get("start_ms", 0))) / 2.0)

        cut_times = [float(self.shots[i].get("end_ms", 0) or 0.0) for i in range(n - 1)]
        snapped: List[int] = []
        snap_report: List[Dict[str, Any]] = []
        last = -1
        for w in walls:
            if not cut_times:
                break
            best_i = min(range(len(cut_times)), key=lambda i: abs(cut_times[i] - w))
            dist = abs(cut_times[best_i] - w)
            if best_i <= last:
                best_i = last + 1
                if best_i > n - 2:
                    sys.stderr.write("[WARN] narrative outline: walls collapse onto the final boundary - flat solving.\n")
                    return None
                dist = abs(cut_times[best_i] - w)
            snapped.append(best_i)
            snap_report.append({"wall_ms": int(w), "snapped_boundary": best_i, "snap_distance_ms": int(dist)})
            last = best_i

        parts: List[Dict[str, Any]] = []
        prev = 0
        for k, seq in enumerate(seqs):
            end = snapped[k] if k < len(snapped) else n - 1
            parts.append({
                "start": prev,
                "end": end,
                "sequence_index": k + 1,
                "sequence_title": str(seq.get("title", "")).strip(),
                "sequence_value": f"{seq.get('value_from', '')} -> {seq.get('value_to', '')}",
            })
            prev = end + 1
        self.seq_snap_report = snap_report
        return parts

    def solve_optimal_grouping(
        self, boundary_scores: List[float], prohibited: Set[int]
    ) -> Tuple[List[Tuple[int, int]], List[int]]:
        """
        Solve the flat global optimal scene partitioning (no narrative outline).
        Returns (partitions, forced_cuts): partitions as inclusive
        (start_shot_idx, end_shot_idx) per macro scene; forced_cuts lists the
        dialogue-protected boundaries the optimum had to place a cut on.
        """
        n_shots = len(self.shots)
        if n_shots == 0:
            return [], []
        k_min = max(1, min(self.min_scenes, n_shots))
        k_max = min(self.max_scenes, n_shots)
        if self.target_scenes:
            k_min = max(1, min(self.target_scenes, n_shots))
            k_max = k_min
        return self._solve_range(0, n_shots - 1, k_min, k_max, boundary_scores, prohibited)

    def _solve_range(
        self,
        start_idx: int,
        end_idx: int,
        k_min: int,
        k_max: int,
        boundary_scores: List[float],
        prohibited: Set[int],
    ) -> Tuple[List[Tuple[int, int]], List[int]]:
        """
        1D Dynamic Programming over the shot slice [start_idx, end_idx] (inclusive,
        absolute indices). Boundary evidence is indexed absolutely so a slice can be
        solved independently - this is what makes the narrative hierarchy possible:
        each sequence is solved on its own, with its own granularity.
        Returns partitions in absolute shot indices plus the forced dialogue cuts.
        """
        n_shots = end_idx - start_idx + 1
        if n_shots <= 0:
            return [], []
        if n_shots == 1:
            return [(start_idx, end_idx)], []

        k_min = max(1, min(k_min, n_shots))
        k_max = max(k_min, min(k_max, n_shots))

        # Precompute duration prefix sums for O(1) segment cost queries
        prefix_durations = [0.0] * (n_shots + 1)
        for k in range(n_shots):
            prefix_durations[k + 1] = prefix_durations[k] + (self.shots[start_idx + k].get("duration_ms", 0) / 1000.0)

        def segment_cost(a_idx: int, b_idx: int) -> float:
            seg_sec = prefix_durations[b_idx + 1] - prefix_durations[a_idx]
            cost = 0.0
            if seg_sec < 10.0:
                cost += (10.0 - seg_sec) * 3.0  # Penalize overly fragmented sub-scenes
            elif seg_sec > 300.0:
                cost += (seg_sec - 300.0) * 0.8  # Penalize overly prolonged monolithic scenes
            return cost

        best_partitions = None
        best_overall_score = float("inf")

        # Diagonal search window: a typical dramatic scene spans 1 to 40 shots.
        # Shot-dense footage (fight montages, fast-cut films) can exceed
        # 40 * k_max shots; the window then grows to ceil(n_shots / k_max) so a
        # full-budget partition always exists - otherwise every K is infeasible
        # and the solver silently collapses into one giant scene.
        MAX_SHOTS_PER_SCENE = 40
        max_span = max(MAX_SHOTS_PER_SCENE, -(-n_shots // max(1, k_max)))

        for K in range(k_min, k_max + 1):
            dp: List[List[float]] = [[float("inf")] * n_shots for _ in range(K)]
            parent: List[List[int]] = [[-1] * n_shots for _ in range(K)]

            for j in range(min(n_shots, max_span)):
                dp[0][j] = segment_cost(0, j)

            for t in range(1, K):
                for j in range(t, n_shots):
                    min_p = max(t - 1, j - max_span)
                    for p in range(min_p, j):
                        b_abs = start_idx + p
                        b_score = boundary_scores[b_abs] if b_abs < len(boundary_scores) else 0.0

                        # Soft penalty instead of a hard ban: dialogue-heavy episodes
                        # can prohibit long runs of boundaries; a hard INF deadlocks the
                        # DP into the single-scene fallback. The finite penalty keeps a
                        # solution reachable while still strictly discouraging mid-speech
                        # cuts; run() reports every forced cut.
                        penalty = self.PROHIBITED_PENALTY if b_abs in prohibited else 0.0

                        transition_cost = segment_cost(p + 1, j) - (b_score * 8.0) + penalty
                        total = dp[t - 1][p] + transition_cost
                        if total < dp[t][j]:
                            dp[t][j] = total
                            parent[t][j] = p

            if dp[K - 1][n_shots - 1] < best_overall_score:
                best_overall_score = dp[K - 1][n_shots - 1]
                curr = n_shots - 1
                curr_k = K - 1
                cuts: List[int] = []
                while curr_k > 0 and curr >= 0:
                    p = parent[curr_k][curr]
                    if p == -1:
                        break
                    cuts.append(p)
                    curr = p
                    curr_k -= 1
                cuts.reverse()

                segs: List[Tuple[int, int]] = []
                prev = 0
                for c in cuts:
                    segs.append((start_idx + prev, start_idx + c))
                    prev = c + 1
                segs.append((start_idx + prev, end_idx))
                best_partitions = segs

        if not best_partitions:
            sys.stderr.write(
                f"[WARN] DP found no feasible {k_min}..{k_max}-scene partition over "
                f"{n_shots} shots; collapsing the slice into a single scene. "
                "Check --max-scenes headroom.\n"
            )
            best_partitions = [(start_idx, end_idx)]

        forced_cuts = [seg[1] for seg in best_partitions[:-1] if seg[1] in prohibited]
        return best_partitions, forced_cuts

    def assemble_macro_scenes(
        self, partitions: List[Tuple[int, int]], sequence_labels: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """Format the resulting segments into macro screenplay scenes with metadata.
        sequence_labels (optional) must be parallel to partitions and carries the
        narrative-outline context of each scene."""
        macro_scenes: List[Dict[str, Any]] = []

        locations_pool = self.bible.get("locations", [])
        loc_names = [loc.get("name", "") for loc in locations_pool if loc.get("name")]

        # Max-overlap cue assignment: a subtitle straddling a scene boundary (L-cut)
        # counts exactly once, for the scene it overlaps most. The previous strict
        # containment filter could drop such cues from BOTH adjacent scenes.
        scene_ranges: List[Tuple[int, int]] = [
            (
                self.shots[start_idx].get("start_ms", 0),
                self.shots[end_idx].get("end_ms", 0),
            )
            for start_idx, end_idx in partitions
        ]
        scene_cue_texts: List[List[str]] = [[] for _ in scene_ranges]
        for s in self.subtitles:
            s_start = s.get("start_ms", 0)
            s_end = s.get("end_ms", 0)
            best_r: Optional[int] = None
            best_ov = 0
            for r, (r_start, r_end) in enumerate(scene_ranges):
                ov = min(r_end, s_end) - max(r_start, s_start)
                if ov > best_ov:
                    best_ov = ov
                    best_r = r
            if best_r is not None:
                scene_cue_texts[best_r].append(str(s.get("text", "")))

        for idx, (start_idx, end_idx) in enumerate(partitions, start=1):
            seg_shots = self.shots[start_idx : end_idx + 1]
            start_ms = seg_shots[0].get("start_ms", 0)
            end_ms = seg_shots[-1].get("end_ms", 0)
            duration_ms = max(0, end_ms - start_ms)

            # Choose best representative keyframe (middle shot or first shot)
            rep_shot_idx = start_idx + (len(seg_shots) // 2)
            rep_shot = self.shots[rep_shot_idx]
            rep_keyframe = rep_shot.get("keyframe", seg_shots[0].get("keyframe", ""))

            # Collect child shot ids
            child_shot_ids = [s.get("scene_id", i + 1) for i, s in enumerate(seg_shots, start=start_idx + 1)]

            scene_sub_texts = scene_cue_texts[idx - 1]

            # Suggest slugline (weak heuristic: bible location names mentioned in
            # dialogue; the scene-understanding draft overrides this downstream)
            slugline = f"SCENE {idx:02d}"
            if loc_names and scene_sub_texts:
                combined_text = " ".join(scene_sub_texts)
                for loc_name in loc_names:
                    if loc_name in combined_text:
                        slugline = f"INT/EXT. {loc_name} - SCENE {idx:02d}"
                        break

            macro_scenes.append({
                "scene_id": f"SCENE_{idx:02d}",
                "macro_index": idx,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_ms": duration_ms,
                "start_timecode": format_timecode(start_ms),
                "end_timecode": format_timecode(end_ms),
                "slugline": slugline,
                "keyframe": rep_keyframe,
                "representative_keyframe": rep_keyframe,
                "child_shot_count": len(seg_shots),
                "child_shot_ids": child_shot_ids,
                "dialogue_cue_count": len(scene_sub_texts)
            })
            if sequence_labels is not None and idx - 1 < len(sequence_labels):
                macro_scenes[-1].update(sequence_labels[idx - 1])

        return macro_scenes

    def run(self) -> Dict[str, Any]:
        """Execute full semantic scene clustering pipeline."""
        prohibited = self.compute_dialogue_hard_constraints()
        visual_dissim = self.compute_visual_dissimilarities()
        boundary_scores = self.compute_boundary_affinity_scores(visual_dissim)

        seq_parts = self.compute_sequence_partitions()
        sequence_labels: Optional[List[Dict[str, Any]]] = None
        seq_summary: List[Dict[str, Any]] = []
        if seq_parts is not None:
            # Hierarchical mode: sequences are narrative walls; the scene DP solves
            # each sequence independently (its own granularity), and every scene
            # inherits its sequence's title + value arc. The global max_scenes
            # budget is split across sequences proportionally to their duration -
            # per-sequence caps would otherwise multiply the budget by the sequence
            # count (observed: 8 sequences -> 102 scenes on a 24-min episode).
            total_dur_ms = sum(
                max(0, int(s.get("end_ms", 0)) - int(s.get("start_ms", 0))) for s in self.shots
            ) or 1
            partitions = []
            forced_cuts: List[int] = []
            for part in seq_parts:
                n_part = part["end"] - part["start"] + 1
                part_dur_ms = max(
                    1,
                    int(self.shots[part["end"]].get("end_ms", 0)) - int(self.shots[part["start"]].get("start_ms", 0)),
                )
                k_cap = max(1, round(self.max_scenes * part_dur_ms / total_dur_ms))
                k_max_part = max(1, min(n_part, k_cap))
                segs, forced = self._solve_range(
                    part["start"], part["end"], 1, k_max_part,
                    boundary_scores, prohibited,
                )
                partitions.extend(segs)
                forced_cuts.extend(forced)
                label = {k: part[k] for k in ("sequence_index", "sequence_title", "sequence_value")}
                sequence_labels = (sequence_labels or []) + [dict(label) for _ in segs]
                seq_summary.append({**label, "scene_count": len(segs), "shot_range": [part["start"], part["end"]]})
            sys.stderr.write(
                f"[INFO] narrative hierarchy enabled: {len(seq_parts)} sequence(s) -> "
                f"{len(partitions)} macro scene(s) (global budget {self.max_scenes}).\n"
            )
        else:
            partitions, forced_cuts = self.solve_optimal_grouping(boundary_scores, prohibited)

        if forced_cuts:
            sys.stderr.write(
                f"[WARN] {len(forced_cuts)} dialogue-protected boundar(y/ies) had to be cut "
                f"(soft-penalty optimum): boundaries {forced_cuts[:10]}. Mid-speech scene cuts "
                "may appear; consider --max-scenes headroom or checking subtitle alignment.\n"
            )
        scenes = self.assemble_macro_scenes(partitions, sequence_labels)

        return {
            "total_macro_scenes": len(scenes),
            "total_physical_shots": len(self.shots),
            "total_dialogue_cues": len(self.subtitles),
            "dialogue_protected_boundaries": len(prohibited),
            "forced_dialogue_cuts": forced_cuts,
            "failed_keyframes": self.failed_keyframes,
            "visual_affinity_enabled": visual_dissim is not None,
            "narrative_outline": {
                "enabled": seq_parts is not None,
                "sequences": seq_summary,
                "wall_snaps": getattr(self, "seq_snap_report", []),
            },
            "scenes": scenes
        }


def main():
    parser = argparse.ArgumentParser(
        description="Semantic Scene Clustering & Dynamic Programming Solver (Pure Python, LGSS-inspired)"
    )
    parser.add_argument("--shots", "-s", default=None, help="Path to raw physical shots.json or scenes.json")
    parser.add_argument("--subtitles", "-u", default=None, help="Path to extracted_subtitles.json")
    parser.add_argument("--bible", "-b", default=None, help="Optional path to materials/bible.json")
    parser.add_argument("--workspace", "-w", default=None, help="Workspace root directory")
    parser.add_argument("--min-scenes", type=int, default=8, help="Minimum number of macro scenes (default: 8)")
    parser.add_argument("--max-scenes", type=int, default=35, help="Maximum number of macro scenes (default: 35)")
    parser.add_argument("--target-scenes", type=int, default=None, help="Explicit target scene count")
    parser.add_argument("--keyframes-dir", default=None,
                        help="Keyframes directory for visual place affinity (default: <shots dir>/keyframes)")
    parser.add_argument("--outline", default=None,
                        help="Optional narrative_structure.json (Phase 3.5); default: <ws>/.cache/alignment/narrative_structure.json")

    args = parser.parse_args()

    # Resolve paths via workspace or explicit flags
    shots_path = args.shots
    subs_path = args.subtitles
    bible_path = args.bible
    outline_path = args.outline

    if args.workspace:
        abs_ws = os.path.abspath(args.workspace)
        if not shots_path:
            # Check .cache/visual/shots.json or fallback to .cache/visual/scenes.json
            cand_shots = os.path.join(abs_ws, ".cache", "visual", "shots.json")
            cand_scenes = os.path.join(abs_ws, ".cache", "visual", "scenes.json")
            shots_path = cand_shots if os.path.isfile(cand_shots) else cand_scenes

        if not subs_path:
            cand_subs = os.path.join(abs_ws, ".cache", "subtitles", "extracted.json")
            if os.path.isfile(cand_subs):
                subs_path = cand_subs

        if not bible_path:
            cand_bible = os.path.join(abs_ws, "materials", "bible.json")
            if os.path.isfile(cand_bible):
                bible_path = cand_bible

        if not outline_path:
            cand_outline = os.path.join(abs_ws, ".cache", "alignment", "narrative_structure.json")
            if os.path.isfile(cand_outline):
                outline_path = cand_outline

    if not shots_path or not os.path.isfile(shots_path):
        sys.stderr.write(f"[FATAL] Physical shots file not found: {shots_path}\n")
        sys.exit(1)

    if not subs_path or not os.path.isfile(subs_path):
        sys.stderr.write(f"[FATAL] Subtitles file not found: {subs_path}\n")
        sys.exit(1)

    raw_shots_doc = load_json_file(shots_path)
    # Support both list format or dict with "scenes" key
    if isinstance(raw_shots_doc, list):
        shots_list = raw_shots_doc
        failed_kf: List[Any] = []
    else:
        shots_list = raw_shots_doc.get("scenes", [])
        failed_kf = raw_shots_doc.get("failed_keyframes", []) or []

    raw_subs_doc = load_json_file(subs_path)
    subs_list = raw_subs_doc if isinstance(raw_subs_doc, list) else raw_subs_doc.get("items", [])

    bible_doc = load_json_file(bible_path) if (bible_path and os.path.isfile(bible_path)) else {}

    outline_doc: Dict[str, Any] = {}
    if outline_path and os.path.isfile(outline_path):
        loaded = load_json_file(outline_path)
        if isinstance(loaded, dict) and loaded.get("sequences"):
            outline_doc = loaded
        else:
            sys.stderr.write(f"[WARN] Outline file ignored (no sequences): {outline_path}\n")
    else:
        sys.stderr.write("[INFO] narrative_structure.json absent - flat scene solving (Phase 3.5 optional).\n")

    keyframes_dir = args.keyframes_dir
    if not keyframes_dir and shots_path:
        keyframes_dir = os.path.join(os.path.dirname(os.path.abspath(shots_path)), "keyframes")

    grouper = SemanticSceneGrouper(
        shots_data=shots_list,
        subtitles_data=subs_list,
        bible_data=bible_doc,
        min_scenes=args.min_scenes,
        max_scenes=args.max_scenes,
        target_scenes=args.target_scenes,
        keyframes_dir=keyframes_dir,
        outline_data=outline_doc,
        failed_keyframes=failed_kf,
    )

    result = grouper.run()
    sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
