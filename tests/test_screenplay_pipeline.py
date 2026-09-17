#!/usr/bin/env python3
"""
tests/test_screenplay_pipeline.py - Pipeline regression tests against the CURRENT API.

Covers:
- unified speaker prefix parsing (subtitle_extractor is the single source of truth)
- two-way dash alternation (stable A/B pair, no phantom speaker explosion)
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
from speaker_diarize import extract_speaker_tags, is_provisional_label
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


class TestSpeakerDiarize(unittest.TestCase):

    def test_dash_alternation_uses_stable_pair(self):
        """Regression: dash-prefixed runs must toggle A/B, not increment a new
        speaker id per line (the old %8 logic exploded 2 people into 8 labels)."""
        subs = [
            {"text": "- 你到底想怎样？", "start_ms": 0, "end_ms": 1000},
            {"text": "- 我不想怎样。", "start_ms": 1200, "end_ms": 2000},
            {"text": "- 那你先放手。", "start_ms": 2200, "end_ms": 3000},
            {"text": "- 好。", "start_ms": 3200, "end_ms": 4000},
        ]
        segments, manifest, unattributed = extract_speaker_tags(subs)
        labels = [s["speaker"] for s in segments]
        self.assertEqual(labels, ["SPEAKER_00A", "SPEAKER_00B", "SPEAKER_00A", "SPEAKER_00B"])
        self.assertEqual(unattributed, 0)
        # Provisional labels must never pollute the characters manifest
        self.assertEqual(manifest, {})

    def test_separate_dash_runs_get_distinct_pairs(self):
        """Two different conversations must not share the same provisional pair:
        an intervening non-dash line closes the run, the next run gets 01A/01B."""
        subs = [
            {"text": "- 走吗？", "start_ms": 0, "end_ms": 800},
            {"text": "- 走。", "start_ms": 1000, "end_ms": 1600},
            {"text": "他们出发了。", "start_ms": 2000, "end_ms": 3000},
            {"text": "- 到了。", "start_ms": 40000, "end_ms": 41000},
            {"text": "- 嗯。", "start_ms": 42000, "end_ms": 43000},
        ]
        segments, _, _ = extract_speaker_tags(subs)
        labels = [s["speaker"] for s in segments]
        self.assertEqual(
            labels,
            ["SPEAKER_00A", "SPEAKER_00B", None, "SPEAKER_01A", "SPEAKER_01B"],
        )

    def test_long_silence_creates_no_phantom_speaker(self):
        """Regression: a silence gap is not evidence of a new speaker."""
        subs = [
            {"text": "今天天气不错。", "start_ms": 0, "end_ms": 1500},
            {"text": "是啊。", "start_ms": 12000, "end_ms": 13000},
        ]
        segments, manifest, unattributed = extract_speaker_tags(subs)
        self.assertIsNone(segments[0]["speaker"])
        self.assertIsNone(segments[1]["speaker"])
        self.assertEqual(unattributed, 2)
        self.assertEqual(manifest, {})

    def test_metadata_and_prefix_attribution(self):
        subs = [
            {"text": "走吧。", "start_ms": 0, "end_ms": 800, "speaker": "菈菈"},
            {"text": "【罗温】跟上。", "start_ms": 1000, "end_ms": 1800},
        ]
        segments, manifest, _ = extract_speaker_tags(subs)
        self.assertEqual(segments[0]["speaker"], "菈菈")
        self.assertEqual(segments[0]["method"], "subtitle_metadata")
        self.assertEqual(segments[1]["speaker"], "罗温")
        self.assertEqual(segments[1]["method"], "text_syntax")
        self.assertEqual(manifest, {"菈菈": 1, "罗温": 1})

    def test_is_provisional_label(self):
        self.assertTrue(is_provisional_label(None))
        self.assertTrue(is_provisional_label("SPEAKER_UNKNOWN"))
        self.assertTrue(is_provisional_label("SPEAKER_03A"))
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
