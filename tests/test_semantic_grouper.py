#!/usr/bin/env python3
"""
tests/test_semantic_grouper.py - Unit and regression tests for SemanticSceneGrouper
"""

import os
import sys
import unittest

# Add scripts directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from semantic_scene_grouper import SemanticSceneGrouper


class TestSemanticSceneGrouper(unittest.TestCase):

    def setUp(self):
        # 12 synthetic physical shots simulating shot cuts during dialogue and action
        self.shots = [
            {"scene_id": 1, "start_ms": 0, "end_ms": 3000, "duration_ms": 3000, "keyframe": "shot_01.jpg"},
            {"scene_id": 2, "start_ms": 3000, "end_ms": 6000, "duration_ms": 3000, "keyframe": "shot_02.jpg"},
            {"scene_id": 3, "start_ms": 6000, "end_ms": 9000, "duration_ms": 3000, "keyframe": "shot_03.jpg"},
            {"scene_id": 4, "start_ms": 9000, "end_ms": 12000, "duration_ms": 3000, "keyframe": "shot_04.jpg"},
            # Big scene boundary silence between 12000 and 18000
            {"scene_id": 5, "start_ms": 18000, "end_ms": 23000, "duration_ms": 5000, "keyframe": "shot_05.jpg"},
            {"scene_id": 6, "start_ms": 23000, "end_ms": 28000, "duration_ms": 5000, "keyframe": "shot_06.jpg"},
            {"scene_id": 7, "start_ms": 28000, "end_ms": 33000, "duration_ms": 5000, "keyframe": "shot_07.jpg"},
            # Another scene boundary after 33000
            {"scene_id": 8, "start_ms": 40000, "end_ms": 45000, "duration_ms": 5000, "keyframe": "shot_08.jpg"},
            {"scene_id": 9, "start_ms": 45000, "end_ms": 50000, "duration_ms": 5000, "keyframe": "shot_09.jpg"},
            {"scene_id": 10, "start_ms": 50000, "end_ms": 55000, "duration_ms": 5000, "keyframe": "shot_10.jpg"},
            {"scene_id": 11, "start_ms": 55000, "end_ms": 60000, "duration_ms": 5000, "keyframe": "shot_11.jpg"},
            {"scene_id": 12, "start_ms": 60000, "end_ms": 65000, "duration_ms": 5000, "keyframe": "shot_12.jpg"}
        ]

        # Dialogue crossing boundary 1 (at 3000ms: 2500ms - 3500ms)
        self.subtitles = [
            {"id": 1, "start_ms": 2500, "end_ms": 3500, "text": "菈菈，快看海边！"},
            {"id": 2, "start_ms": 4000, "end_ms": 5500, "text": "爸爸，有烟雾！"},
            {"id": 3, "start_ms": 7000, "end_ms": 8500, "text": "我们得赶快回家。"},
            # Scene 2 dialogue
            {"id": 4, "start_ms": 19000, "end_ms": 22000, "text": "罗温家木屋里很安静。"},
            {"id": 5, "start_ms": 24000, "end_ms": 27000, "text": "喝点热水吧。"},
            # Scene 3 dialogue
            {"id": 6, "start_ms": 42000, "end_ms": 44000, "text": "明天去防波堤集市。"}
        ]

        self.bible = {
            "locations": [
                {"name": "海边", "type": "EXT"},
                {"name": "罗温家", "type": "INT"},
                {"name": "防波堤", "type": "EXT"}
            ]
        }

    def test_dialogue_hard_constraints(self):
        grouper = SemanticSceneGrouper(self.shots, self.subtitles, self.bible)
        prohibited = grouper.compute_dialogue_hard_constraints()
        # Boundary 0 (between shot 1 and shot 2 at 3000ms) MUST be prohibited because subtitle 1 spans 2500-3500ms
        self.assertIn(0, prohibited, "Boundary 0 must be protected from scene cut due to active speech")

    def test_optimal_scene_grouping(self):
        grouper = SemanticSceneGrouper(
            self.shots, self.subtitles, self.bible, min_scenes=2, max_scenes=5
        )
        res = grouper.run()
        self.assertGreaterEqual(res["total_macro_scenes"], 2)
        self.assertLessEqual(res["total_macro_scenes"], 5)
        self.assertEqual(res["total_physical_shots"], 12)

        # Check that child shots partition all 12 shots without gaps
        covered_shots = []
        for s in res["scenes"]:
            covered_shots.extend(s["child_shot_ids"])
        self.assertEqual(covered_shots, list(range(1, 13)))

    def test_slugline_location_inference(self):
        grouper = SemanticSceneGrouper(
            self.shots, self.subtitles, self.bible, target_scenes=3
        )
        res = grouper.run()
        scenes = res["scenes"]
        self.assertEqual(len(scenes), 3)
        # Check that scene sluglines contain matched locations
        sluglines = [s["slugline"] for s in scenes]
        self.assertTrue(any("海边" in sl or "罗温家" in sl or "防波堤" in sl for sl in sluglines))

    def test_soft_penalty_avoids_single_scene_deadlock(self):
        """Regression: when dialogue spans EVERY boundary, the former hard-INF ban
        deadlocked the DP into a silent single-scene fallback for the whole episode.
        With the soft penalty the solver must still return multiple scenes and
        report the forced cuts."""
        shots = [
            {"scene_id": i + 1, "start_ms": i * 3000, "end_ms": (i + 1) * 3000,
             "duration_ms": 3000, "keyframe": f"shot_{i+1:02d}.jpg"}
            for i in range(10)
        ]
        # One utterance straddling every single shot boundary
        subs = [
            {"id": i + 1, "start_ms": i * 3000 - 500, "end_ms": i * 3000 + 500, "text": "别停下！"}
            for i in range(1, 10)
        ]
        grouper = SemanticSceneGrouper(shots, subs, min_scenes=2, max_scenes=4)
        res = grouper.run()
        self.assertGreaterEqual(res["total_macro_scenes"], 2,
                                "Soft penalty must keep multi-scene solutions reachable")
        self.assertGreater(len(res["forced_dialogue_cuts"]), 0,
                           "Forced dialogue cuts must be reported, not silently swallowed")

    def test_straddling_cue_counted_exactly_once(self):
        """Regression: a subtitle straddling a scene boundary must be attributed to
        exactly one macro scene (max overlap), not vanish from both."""
        straddler = {"id": 99, "start_ms": 22500, "end_ms": 24500, "text": "我们走吧。"}
        grouper = SemanticSceneGrouper(
            self.shots, self.subtitles + [straddler], self.bible, target_scenes=3
        )
        res = grouper.run()
        total_cues = sum(s["dialogue_cue_count"] for s in res["scenes"])
        self.assertEqual(total_cues, len(self.subtitles) + 1)

    def test_visual_affinity_reported_and_optional(self):
        grouper = SemanticSceneGrouper(self.shots, self.subtitles, self.bible, min_scenes=2, max_scenes=5)
        res = grouper.run()
        self.assertIn("visual_affinity_enabled", res)
        self.assertIsInstance(res["visual_affinity_enabled"], bool)

    def test_dp_chosen_cut_set_is_pinned(self):
        """Golden pin for the cost function. Every other solver test asserts a
        property (count >= 2, no collapse, budget respected), all of which still held
        when PROHIBITED_PENALTY and the affinity weights were wildly wrong. Re-tuning
        a constant must either keep these exact cuts or deliberately change this line."""
        grouper = SemanticSceneGrouper(self.shots, self.subtitles, self.bible,
                                       min_scenes=2, max_scenes=5)
        res = grouper.run()
        cuts = [(s["child_shot_ids"][0], s["child_shot_ids"][-1]) for s in res["scenes"]]
        self.assertEqual(cuts, [(1, 4), (5, 6), (7, 8), (9, 10), (11, 12)])
        self.assertEqual(res["forced_dialogue_cuts"], [])

        pinned = SemanticSceneGrouper(self.shots, self.subtitles, self.bible, target_scenes=3)
        cuts3 = [(s["child_shot_ids"][0], s["child_shot_ids"][-1]) for s in pinned.run()["scenes"]]
        self.assertEqual(cuts3, [(1, 4), (5, 6), (7, 12)])

    def test_visual_affinity_flag_is_false_when_no_keyframe_readable(self):
        """Regression: the flag reported the DEPENDENCIES being present, so a run with
        numpy+opencv installed and every keyfile missing claimed visual affinity was
        in play while it contributed nothing."""
        grouper = SemanticSceneGrouper(self.shots, self.subtitles, self.bible, min_scenes=2,
                                        max_scenes=5, keyframes_dir="/nonexistent-keyframes-dir")
        res = grouper.run()
        self.assertFalse(res["visual_affinity_enabled"])
        self.assertEqual(res["visual_boundary_coverage"][0], 0)


class TestNarrativeHierarchy(unittest.TestCase):
    """Phase 3.5: the McKee sequence layer - sequence walls from
    narrative_structure.json partition the shot stream and each sequence is
    solved independently; without an outline the grouper flat-solves as before."""

    def setUp(self):
        self.shots = [
            {"scene_id": i + 1, "start_ms": i * 3000, "end_ms": (i + 1) * 3000,
             "duration_ms": 3000, "keyframe": f"shot_{i+1:02d}.jpg"}
            for i in range(12)
        ]
        self.subs = [
            {"index": 1, "start_ms": 1000, "end_ms": 2000, "text": "第一句"},
            {"index": 2, "start_ms": 13000, "end_ms": 14000, "text": "第二句"},
            {"index": 3, "start_ms": 40000, "end_ms": 41000, "text": "转折之后的第一句"},
            {"index": 4, "start_ms": 60000, "end_ms": 62000, "text": "收尾"},
        ]
        # Outline: sequence 1 = subs 1-2, sequence 2 = subs 3-4.
        # Wall = midpoint(14000, 40000) = 27000 -> nearest shot cut 27000 (boundary 8, exact).
        self.outline = {
            "sequences": [
                {"seq_index": 1, "title": "铺垫序列", "value_from": "安稳", "value_to": "起疑",
                 "start_sub": 1, "end_sub": 2, "evidence": "setup"},
                {"seq_index": 2, "title": "转折序列", "value_from": "起疑", "value_to": "决意",
                 "start_sub": 3, "end_sub": 4, "evidence": "turn"},
            ]
        }

    def test_wall_snaps_to_nearest_shot_cut(self):
        g = SemanticSceneGrouper(self.shots, self.subs, outline_data=self.outline)
        parts = g.compute_sequence_partitions()
        self.assertIsNotNone(parts)
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0]["sequence_title"], "铺垫序列")
        self.assertEqual((parts[0]["start"], parts[0]["end"]), (0, 8))
        self.assertEqual((parts[1]["start"], parts[1]["end"]), (9, 11))
        self.assertEqual(g.seq_snap_report[0]["snapped_boundary"], 8)
        # The wall here is exactly on a cut, so an equality pin is meaningful -
        # "<= SEQ_WALL_SNAP_MS" on this fixture held even when the solver ignored
        # the window entirely.
        self.assertEqual(g.seq_snap_report[0]["snap_distance_ms"], 0)
        self.assertTrue(g.seq_snap_report[0]["within_snap_window"])

    def test_wall_beyond_snap_window_lands_on_preceding_cut(self):
        """Regression: SEQ_WALL_SNAP_MS was declared but never consulted, so a wall
        with no edit anywhere near it teleported to the closest cut however far."""
        shots = [{"scene_id": i + 1, "start_ms": i * 40000, "end_ms": (i + 1) * 40000,
                  "duration_ms": 40000, "keyframe": None} for i in range(6)]
        # Cuts sit at 40s/80s/120s/... Put the wall at 60s: 20s from either neighbour,
        # outside the 15s window.
        subs = [{"index": 1, "start_ms": 55000, "end_ms": 58000, "text": "a"},
                {"index": 2, "start_ms": 62000, "end_ms": 65000, "text": "b"}]
        outline = {"sequences": [
            {"seq_index": 1, "title": "甲", "value_from": "a", "value_to": "b",
             "start_sub": 1, "end_sub": 1},
            {"seq_index": 2, "title": "乙", "value_from": "b", "value_to": "c",
             "start_sub": 2, "end_sub": 2},
        ]}
        g = SemanticSceneGrouper(shots, subs, min_scenes=1, max_scenes=6, outline_data=outline)
        parts = g.compute_sequence_partitions()
        self.assertIsNotNone(parts)
        report = g.seq_snap_report[0]
        self.assertFalse(report["within_snap_window"])
        self.assertGreater(report["snap_distance_ms"], SemanticSceneGrouper.SEQ_WALL_SNAP_MS)
        # Nearest PRECEDING cut (boundary 0 ends at 40s), never the further one.
        self.assertEqual(report["snapped_boundary"], 0)

    def test_single_shot_with_outline_solves_flat_instead_of_crashing(self):
        """Regression: one physical shot plus a multi-sequence outline produced a
        start>end partition and an empty scene list that exited 0."""
        shots = [{"scene_id": 1, "start_ms": 0, "end_ms": 30000, "duration_ms": 30000,
                  "keyframe": None}]
        subs = [{"index": 1, "start_ms": 1000, "end_ms": 2000, "text": "甲"},
                {"index": 2, "start_ms": 20000, "end_ms": 21000, "text": "乙"}]
        outline = {"sequences": [
            {"seq_index": 1, "title": "甲段", "value_from": "a", "value_to": "b",
             "start_sub": 1, "end_sub": 1},
            {"seq_index": 2, "title": "乙段", "value_from": "b", "value_to": "c",
             "start_sub": 2, "end_sub": 2},
        ]}
        g = SemanticSceneGrouper(shots, subs, outline_data=outline)
        self.assertIsNone(g.compute_sequence_partitions())
        res = g.run()
        self.assertFalse(res["narrative_outline"]["enabled"])
        self.assertEqual(res["total_macro_scenes"], 1)

    def test_hierarchical_run_never_spans_a_sequence(self):
        g = SemanticSceneGrouper(self.shots, self.subs, min_scenes=2, max_scenes=6,
                                 outline_data=self.outline)
        res = g.run()
        self.assertTrue(res["narrative_outline"]["enabled"])
        scenes = res["scenes"]
        self.assertGreaterEqual(len(scenes), 2)
        for scene in scenes:
            self.assertEqual(scene["sequence_title"] in ("铺垫序列", "转折序列"), True)
            ids = scene["child_shot_ids"]
            # No scene may straddle the sequence wall (between shot 9 and shot 10, 1-based)
            self.assertFalse(
                (min(ids) <= 9 and max(ids) >= 10),
                f"scene {scene['scene_id']} spans the sequence wall: {ids}",
            )
        # Both sequences are represented in the summary
        titles = {s["sequence_title"] for s in res["narrative_outline"]["sequences"]}
        self.assertEqual(titles, {"铺垫序列", "转折序列"})

    def test_flat_mode_without_outline_is_unchanged(self):
        g = SemanticSceneGrouper(self.shots, self.subs, min_scenes=2, max_scenes=5)
        res = g.run()
        self.assertFalse(res["narrative_outline"]["enabled"])
        self.assertEqual(res["narrative_outline"]["sequences"], [])
        for scene in res["scenes"]:
            self.assertNotIn("sequence_title", scene)

    def test_bad_outline_degrades_to_flat(self):
        bad = {"sequences": [{"seq_index": 1, "title": "断链", "value_from": "a", "value_to": "b",
                              "start_sub": 1, "end_sub": 99}]}
        g = SemanticSceneGrouper(self.shots, self.subs, outline_data=bad)
        self.assertIsNone(g.compute_sequence_partitions())

    def test_hierarchical_mode_respects_global_scene_budget(self):
        """Regression: per-sequence caps multiplied the global budget by the sequence
        count (8 sequences produced 102 scenes on a 24-min episode). The budget must
        be split across sequences proportionally to their duration."""
        shots = [
            {"scene_id": i + 1, "start_ms": i * 5000, "end_ms": (i + 1) * 5000,
             "duration_ms": 5000, "keyframe": f"shot_{i+1:02d}.jpg"}
            for i in range(40)
        ]
        subs = [
            {"index": i + 1, "start_ms": i * 10000 + 1000, "end_ms": i * 10000 + 3000,
             "text": f"台词{i+1}"}
            for i in range(4)
        ]
        outline = {"sequences": [
            {"seq_index": 1, "title": "前段", "value_from": "a", "value_to": "b",
             "start_sub": 1, "end_sub": 2},
            {"seq_index": 2, "title": "后段", "value_from": "b", "value_to": "c",
             "start_sub": 3, "end_sub": 4},
        ]}
        g = SemanticSceneGrouper(shots, subs, min_scenes=2, max_scenes=6, outline_data=outline)
        res = g.run()
        self.assertLessEqual(res["total_macro_scenes"], 8,
                             "hierarchical solving must stay near the global budget, not multiply it")
        self.assertGreaterEqual(res["total_macro_scenes"], 2)


class TestPlaceAffinity(unittest.TestCase):
    """Functional coverage for the LGSS-'place' visual affinity (needs numpy+cv2)."""

    def setUp(self):
        try:
            import cv2  # noqa: F401
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy/opencv not installed - place affinity runs degraded")

    def _grouper_with_images(self, tmpdir, colors):
        import cv2
        import numpy as np

        kf_dir = os.path.join(tmpdir, "keyframes")
        os.makedirs(kf_dir)
        shots = []
        for i, bgr in enumerate(colors):
            name = f"shot_{i+1:04d}.jpg"
            cv2.imwrite(os.path.join(kf_dir, name), np.full((48, 64, 3), bgr, dtype=np.uint8))
            shots.append({"scene_id": i + 1, "start_ms": i * 3000, "end_ms": (i + 1) * 3000,
                          "duration_ms": 3000, "keyframe": name})
        return SemanticSceneGrouper(shots, [], min_scenes=1, max_scenes=2, keyframes_dir=kf_dir)

    def test_hue_weighted_affinity_detects_palette_change(self):
        """Regression: hue carries the place signal - a blue|red boundary must score
        far above a same-palette boundary despite identical saturation/value."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            g = self._grouper_with_images(td, [(255, 140, 0), (250, 150, 10), (0, 0, 255), (10, 10, 250)])
            d = g.compute_visual_dissimilarities()
            self.assertIsNotNone(d)
            self.assertLess(d[0], 0.1, "same-palette boundary must be ~0")
            self.assertGreater(d[1], 0.5, "hue shift (water blue -> interior red) must be strong")
            self.assertLess(d[2], 0.1)

    def test_missing_keyframes_stay_neutral(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            g = self._grouper_with_images(td, [(255, 140, 0), (0, 0, 255)])
            os.remove(os.path.join(g.keyframes_dir, "shot_0002.jpg"))
            self.assertEqual(g.compute_visual_dissimilarities(), [0.0],
                             "missing keyframe -> neutral, never fabricated")


class TestDenseFootageNoCollapse(unittest.TestCase):
    """Regression (P2): shot-dense footage once exceeded the 40-shot-per-scene
    search window times max_scenes, made EVERY K infeasible, and silently
    collapsed into a single macro scene with zero diagnostics."""

    def test_shots_beyond_window_budget_do_not_collapse(self):
        # 200 shots vs a 4-scene budget: old ceiling 40*4=160 < 200 -> old code
        # returned exactly 1 scene; the widened window (ceil(200/4)=50) must
        # keep the partition feasible.
        shots = [
            {"scene_id": i + 1, "start_ms": i * 100, "end_ms": (i + 1) * 100,
             "duration_ms": 100, "keyframe": ""}
            for i in range(200)
        ]
        grouper = SemanticSceneGrouper(shots, [], min_scenes=2, max_scenes=4)
        result = grouper.run()
        self.assertGreaterEqual(result["total_macro_scenes"], 2,
                                "dense footage must not collapse into one scene")
        self.assertLessEqual(result["total_macro_scenes"], 4)


class TestFailedKeyframePropagation(unittest.TestCase):
    """Regression (P1): shots.json's failed_keyframes were dropped by the
    grouper, so scenes.json never carried them and align_timeline's
    visual_verified flag was always True."""

    def test_failed_keyframes_roundtrip_into_output(self):
        shots = [
            {"scene_id": 1, "start_ms": 0, "end_ms": 5000, "duration_ms": 5000, "keyframe": "a.jpg"},
            {"scene_id": 2, "start_ms": 8000, "end_ms": 13000, "duration_ms": 5000, "keyframe": "b.jpg"},
        ]
        grouper = SemanticSceneGrouper(shots, [], failed_keyframes=[2])
        result = grouper.run()
        self.assertEqual(result["failed_keyframes"], [2])

    def test_default_is_empty_list(self):
        shots = [
            {"scene_id": 1, "start_ms": 0, "end_ms": 5000, "duration_ms": 5000, "keyframe": "a.jpg"},
        ]
        result = SemanticSceneGrouper(shots, []).run()
        self.assertEqual(result["failed_keyframes"], [])


if __name__ == "__main__":
    unittest.main()
