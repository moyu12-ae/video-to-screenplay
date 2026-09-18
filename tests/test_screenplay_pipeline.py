#!/usr/bin/env python3
"""
tests/test_screenplay_pipeline.py - Pipeline regression tests against the CURRENT API.

Covers:
- unified speaker prefix parsing (subtitle_extractor is the single source of truth)
- acoustic speaker diarization bridge (prepare/merge around MCP omni_multi_speaker_asr):
  chunk planning, Omni output normalization, max-overlap binding, majority-vote naming
- provisional speaker masking (SPEAKER_* never rendered as a character name)
- O.S./V.O. enum mapping (align_timeline emits OFF_SCREEN/VOICE_OVER/...)
- build_scene_manifest contract (manifest written inside the workspace, resume status)
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = (Path(__file__).parent.parent / "scripts").resolve()
sys.path.insert(0, str(SCRIPTS_DIR))

from subtitle_extractor import extract_speaker_from_text
from speaker_diarize import (
    bind_lines,
    build_cluster_map,
    is_provisional_label,
    name_clusters,
    normalize_omni_segments,
    parse_silences,
    plan_parts,
    snap_boundaries,
)
from align_timeline import infer_av_relationship


class TestSpeakerPrefixParsing(unittest.TestCase):

    def test_bracket_prefix(self):
        spk, clean = extract_speaker_from_text("【罗温】我们该出发了。")
        self.assertEqual(spk, "罗温")
        self.assertEqual(clean, "我们该出发了。")

    def test_single_char_bracket_name(self):
        """Regression: 1-char Chinese names must resolve via the shared parser."""
        spk, clean = extract_speaker_from_text("【兰】走吧。")
        self.assertEqual(spk, "兰")
        self.assertEqual(clean, "走吧。")

    def test_colon_prefix(self):
        spk, clean = extract_speaker_from_text("菈菈：等一下，琴箱还没关上。")
        self.assertEqual(spk, "菈菈")
        self.assertEqual(clean, "等一下，琴箱还没关上。")

    def test_timestamp_like_text_is_not_a_speaker(self):
        spk, _ = extract_speaker_from_text("10:05 我们出发")
        self.assertIsNone(spk)

    def test_emotion_parenthesis_is_not_a_speaker(self):
        spk, clean = extract_speaker_from_text("(叹气) 这一天终于来了。")
        self.assertIsNone(spk)
        self.assertEqual(clean, "(叹气) 这一天终于来了。")


class TestOmniSpeakerMerge(unittest.TestCase):
    """The acoustic bridge: chunk planning, Omni output normalization, max-overlap
    binding (attribution), and metadata majority-vote naming — in that order."""

    def test_plan_parts_short_video_single_chunk(self):
        self.assertEqual(plan_parts(1_440_000), [(0, 1_440_000)])

    def test_plan_parts_long_video_chunks_at_45min(self):
        """130 min -> boundaries at 45/90 min; adjacent parts share a ±3s overlap."""
        self.assertEqual(
            plan_parts(7_800_000),
            [(0, 2_703_000), (2_697_000, 5_403_000), (5_397_000, 7_800_000)],
        )

    def test_plan_parts_chunk_seconds_with_overlap(self):
        """--chunk-seconds 40 on 148s -> 4 parts, adjacent sharing ±3s."""
        self.assertEqual(
            plan_parts(148_000, chunk_target_sec=40),
            [(0, 43_000), (37_000, 83_000), (77_000, 123_000), (117_000, 148_000)],
        )

    def test_plan_parts_snaps_boundaries_to_silence(self):
        self.assertEqual(
            plan_parts(148_000, chunk_target_sec=40, silence_points=[39.2, 79.4, 121.4]),
            [(0, 42_200), (36_200, 82_400), (76_400, 124_400), (118_400, 148_000)],
        )

    def test_parse_silences_uses_midpoints(self):
        log = ("[silencedetect @ 0x1] silence_start: 12.345\n"
               "[silencedetect @ 0x1] silence_end: 13.456 | silence_duration: 1.111\n"
               "[silencedetect @ 0x1] silence_start: 40.1\n")  # open at EOF -> ignored
        self.assertAlmostEqual(parse_silences(log)[0], 12.9005)
        self.assertEqual(len(parse_silences(log)), 1)

    def test_snap_boundaries_nearest_unused_silence(self):
        self.assertEqual(snap_boundaries([40.0, 80.0], [39.2, 79.6, 120.9]), [39.2, 79.6])
        self.assertEqual(snap_boundaries([40.0], [100.0]), [40.0])  # out of tolerance

    def test_plan_parts_folds_short_tail(self):
        """53.3 min -> 45-min part + an 8.3-min tail that folds back into it."""
        self.assertEqual(plan_parts(3_200_000), [(0, 3_200_000)])

    def test_normalize_segments_seconds_to_ms_with_offset(self):
        data = {"segments": [
            {"speaker": "Speaker 1", "start": 1.5, "end": 2.5, "text": "你好"},
            {"speaker": "", "start": 3.0, "end": 4.0},          # no label -> skipped
            {"speaker": "Speaker 2", "start": 5.0, "end": 5.0},  # empty span -> skipped
            {"speaker": "Speaker 2", "start": "bad", "end": 9.0},  # bad time -> skipped
        ]}
        turns = normalize_omni_segments(data, offset_ms=2_700_000)
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0]["start_ms"], 2_701_500)
        self.assertEqual(turns[0]["end_ms"], 2_702_500)
        self.assertEqual(turns[0]["text"], "你好")

    def _pturn(self, part, label, start, end, text=""):
        return {"part": part, "raw_label": label, "start_ms": start, "end_ms": end, "text": text}

    def test_cluster_map_links_shared_utterance_across_parts(self):
        turns = [self._pturn(0, "Speaker 1", 27_200, 39_500),
                 self._pturn(1, "Speaker 1", 40_880, 43_040)]
        merged = build_cluster_map(turns)
        self.assertNotEqual(merged[(0, "Speaker 1")], merged[(1, "Speaker 1")])
        # same utterance diarized by both parts inside the ±3s overlap -> linked
        turns.append(self._pturn(1, "Speaker 1", 38_000, 39_400))
        merged = build_cluster_map(turns)
        self.assertEqual(merged[(0, "Speaker 1")], merged[(1, "Speaker 1")])

    def test_cluster_map_ep3_regression_same_label_different_people(self):
        """v0.3.0 bug: part0's Speaker 2 (advisor) and part1's Speaker 2 (Nagahama)
        were merged by label string alone. Without co-occurrence evidence they
        must stay separate clusters."""
        turns = [self._pturn(0, "Speaker 2", 3_920, 25_040),
                 self._pturn(1, "Speaker 2", 53_000, 57_000),
                 self._pturn(0, "Speaker 1", 38_400, 39_520),
                 self._pturn(1, "Speaker 1", 40_880, 43_040)]
        merged = build_cluster_map(turns)
        self.assertNotEqual(merged[(0, "Speaker 2")], merged[(1, "Speaker 2")])
        self.assertNotEqual(merged[(0, "Speaker 1")], merged[(1, "Speaker 1")])

    def test_cluster_map_transitive_link_and_temporal_order(self):
        turns = [self._pturn(0, "A", 38_000, 39_500),
                 self._pturn(1, "B", 38_800, 39_600),    # co-occurs with part0
                 self._pturn(1, "B", 50_000, 52_000),
                 self._pturn(2, "C", 51_500, 52_500)]    # co-occurs with part1's later turn
        merged = build_cluster_map(turns)
        self.assertEqual(merged[(0, "A")], merged[(1, "B")])
        self.assertEqual(merged[(1, "B")], merged[(2, "C")])
        self.assertEqual(merged[(0, "A")], "SPEAKER_A1")  # earliest component wins A1

    def _turn(self, label, start, end, text=""):
        return {"raw_label": label, "cluster_id": label, "start_ms": start,
                "end_ms": end, "text": text}

    def test_bind_lines_picks_max_overlap_primary(self):
        turns = [self._turn("SPEAKER_A1", 0, 2000), self._turn("SPEAKER_A2", 2000, 4000)]
        rows, _ = bind_lines([{"text": "喂？", "start_ms": 1500, "end_ms": 3500}], turns)
        row = rows[0]
        self.assertEqual(row["cluster_id"], "SPEAKER_A2")   # 1500ms vs 500ms overlap
        self.assertEqual(row["confidence"], 0.75)           # ratio 0.75 -> mid tier
        self.assertIsNone(row["secondary_speaker"])          # A1 ratio 0.25 < 0.4
        self.assertEqual(row["method"], "acoustic_diarization")

    def test_bind_lines_secondary_speaker_on_heavy_overlap(self):
        turns = [self._turn("SPEAKER_A1", 0, 1400), self._turn("SPEAKER_A2", 1400, 2400)]
        rows, _ = bind_lines([{"text": " overlapping!", "start_ms": 1000, "end_ms": 2000}], turns)
        self.assertEqual(rows[0]["cluster_id"], "SPEAKER_A2")      # 600ms
        self.assertEqual(rows[0]["secondary_speaker"], "SPEAKER_A1")  # 400ms = 0.4
        self.assertEqual(rows[0]["confidence"], 0.75)

    def test_bind_lines_no_overlap_is_unattributed(self):
        rows, cluster_lines = bind_lines(
            [{"text": "（OP 主题歌）", "start_ms": 90_000, "end_ms": 91_000}],
            [self._turn("SPEAKER_A1", 0, 1000)])
        self.assertIsNone(rows[0]["speaker"])
        self.assertEqual(rows[0]["method"], "no_speech_overlap")
        self.assertEqual(cluster_lines, {})

    def test_bind_lines_text_agreement_flag(self):
        turns = [self._turn("SPEAKER_A1", 0, 2000, text="你好世界")]
        rows, _ = bind_lines([
            {"text": "你好，世界！", "start_ms": 0, "end_ms": 2000},
            {"text": "完全不同的一句话", "start_ms": 0, "end_ms": 2000},
        ], turns)
        self.assertTrue(rows[0]["text_agreement"])
        self.assertFalse(rows[1]["text_agreement"])

    def test_confidence_tiers(self):
        """ratio >= 0.80 -> 0.90; >= 0.50 -> 0.75; below -> 0.55."""
        turn = [self._turn("SPEAKER_A1", 1000, 4000)]

        def confidence_for(line_start, line_end):
            rows, _ = bind_lines(
                [{"text": "x", "start_ms": line_start, "end_ms": line_end}], turn)
            return rows[0]["confidence"]

        self.assertEqual(confidence_for(1000, 4000), 0.90)  # overlap 4000/4000 = 1.00
        self.assertEqual(confidence_for(0, 4000), 0.75)     # overlap 3000/4000 = 0.75
        self.assertEqual(confidence_for(0, 1800), 0.55)     # overlap 800/1800 ≈ 0.44

    def test_name_clusters_majority_vote_with_threshold(self):
        cluster_lines = {
            "SPEAKER_A1": [
                {"text": "a", "speaker": "菈菈"}, {"text": "b", "speaker": "菈菈"},
                {"text": "c", "speaker": "菈菈"}, {"text": "d", "speaker": "罗温"},
            ],
            "SPEAKER_A2": [{"text": "e", "speaker": "罗温"}],  # 1 vote < 2: stays unnamed
        }
        names, manifest = name_clusters(cluster_lines)
        self.assertEqual(names["SPEAKER_A1"], "菈菈")
        self.assertIsNone(names["SPEAKER_A2"])
        self.assertEqual(manifest, {"菈菈": 4})  # provisional clusters never enter

    def test_empty_turns_unattribute_everything(self):
        rows, cluster_lines = bind_lines(
            [{"text": "x", "start_ms": 0, "end_ms": 1000}], [])
        self.assertIsNone(rows[0]["speaker"])
        self.assertEqual(cluster_lines, {})

    def test_is_provisional_label_covers_acoustic_labels(self):
        self.assertTrue(is_provisional_label("SPEAKER_A1"))
        self.assertTrue(is_provisional_label(None))
        self.assertTrue(is_provisional_label("SPEAKER_UNKNOWN"))
        self.assertFalse(is_provisional_label("菈菈"))


class TestAlignTimeline(unittest.TestCase):

    def test_av_relationship_os_inference(self):
        scene = {"start_ms": 10000, "end_ms": 11000}
        sub = {"text": "你好？", "start_ms": 8000, "end_ms": 9000}
        self.assertEqual(infer_av_relationship(sub, scene), "OFF_SCREEN")


class TestSceneVisualVerified(unittest.TestCase):
    """P1 regression: failed_keyframes now flow shots.json -> scenes.json ->
    aligned_timeline; a macro scene stays visually verified iff at least one
    child shot has an extractable keyframe."""

    def test_no_failures_means_verified(self):
        from align_timeline import scene_visual_verified
        scene = {"scene_id": "SCENE_01", "child_shot_ids": [1, 2]}
        self.assertTrue(scene_visual_verified(scene, []))

    def test_all_children_failed_is_unverified(self):
        from align_timeline import scene_visual_verified
        scene = {"scene_id": "SCENE_03", "child_shot_ids": [7, 8]}
        self.assertFalse(scene_visual_verified(scene, [7, 8]))

    def test_partially_failed_still_verified(self):
        from align_timeline import scene_visual_verified
        scene = {"scene_id": "SCENE_03", "child_shot_ids": [7, 8]}
        self.assertTrue(scene_visual_verified(scene, [7]))

    def test_fallback_checks_scene_id_without_children(self):
        from align_timeline import scene_visual_verified
        self.assertFalse(scene_visual_verified({"scene_id": 9}, [9]))
        self.assertTrue(scene_visual_verified({"scene_id": "SCENE_09"}, [9]))


class TestSceneManifestBuilder(unittest.TestCase):

    def test_manifest_evidence_pack_with_resume_status(self):
        from build_scene_manifest import build_manifest

        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            visual = ws / ".cache" / "visual"
            kf = visual / "keyframes"
            kf.mkdir(parents=True)
            (visual / "scenes.json").write_text(json.dumps({
                "scenes": [{
                    "scene_id": "SCENE_01", "macro_index": 1,
                    "start_ms": 0, "end_ms": 6000,
                    "start_timecode": "00:00:00.000", "end_timecode": "00:00:06.000",
                    "duration_ms": 6000, "slugline": "SCENE 01", "child_shot_count": 2,
                }]
            }, ensure_ascii=False), encoding="utf-8")
            (visual / "shots.json").write_text(json.dumps({
                "scenes": [
                    {"scene_id": 1, "start_ms": 0, "end_ms": 3000, "keyframe": "shot_0001_000000ms.jpg"},
                    {"scene_id": 2, "start_ms": 3000, "end_ms": 6000, "keyframe": "shot_0002_003000ms.jpg"},
                ]
            }), encoding="utf-8")
            (ws / "materials").mkdir()
            (ws / "materials" / "bible.json").write_text(json.dumps({
                "characters": [{"name": "菈菈"}, {"name": "茉里"}],
                "speaker_whitelist": ["面试的店主"],
            }, ensure_ascii=False), encoding="utf-8")
            (kf / "shot_0001_000000ms.jpg").write_bytes(b"\xff\xd8fake")
            (kf / "shot_0002_003000ms.jpg").write_bytes(b"\xff\xd8fake")

            manifest = build_manifest(ws, max_keyframes=8)
            self.assertEqual(manifest["total_scenes"], 1)
            self.assertEqual(manifest["bible_names"], ["菈菈", "茉里", "面试的店主"])
            self.assertIn("[[SUB:", manifest["instructions"])
            scene = manifest["scenes"][0]
            # 640px thumbs are generated for the writer (cv2 present in dev env)
            self.assertTrue(scene["keyframes_thumbs"])
            self.assertTrue(all(k.startswith(str(ws)) for k in scene["keyframes_thumbs"]))
            self.assertEqual(scene["draft_status"], "missing")

            # Resume: once the scene TEXT (.md) exists the builder reports it written
            drafts_dir = ws / ".cache" / "scene_drafts"
            drafts_dir.mkdir(parents=True, exist_ok=True)
            (drafts_dir / "scene_01.md").write_text("## 第 1 场\n", encoding="utf-8")
            manifest2 = build_manifest(ws, max_keyframes=8)
            self.assertEqual(manifest2["scenes"][0]["draft_status"], "written")


if __name__ == "__main__":
    unittest.main()
