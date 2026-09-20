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

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS_DIR = (Path(__file__).parent.parent / "scripts").resolve()
sys.path.insert(0, str(SCRIPTS_DIR))


def _tmp_file(suffix: str) -> Path:
    """A safely unpredictable temp file (mkstemp), closed immediately for
    Path-level use in tests."""
    fd, name = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return Path(name)

from subtitle_extractor import extract_speaker_from_text
from speaker_diarize import (
    bind_lines,
    build_cluster_map,
    build_output,
    cmd_run,
    estimate_subtitle_offset_ms,
    is_provisional_label,
    make_workorder,
    name_clusters,
    normalize_omni_segments,
    parse_silences,
    plan_parts,
    snap_boundaries,
)
import omni_client
import op_ed
import workspace
from align_timeline import infer_av_relationship
from subtitle_extractor import select_embedded_stream
from omni_client import build_video_part, fit_video, understand_video_segment
from av_understand import (
    build_prompt,
    cmd_merge as av_merge,
    cmd_prepare as av_prepare,
    cmd_run as av_run,
    dedup_entries,
    has_substance,
    parse_note,
    plan_segments,
    scene_coverage,
)
import av_understand as av_understand_module


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

    def test_estimate_offset_and_bind_tolerate_subtitle_lead(self):
        """第三集 52.2s 案例复现：字幕整体超前 ~2.3s 时，换人边界的行会被前一
        说话人的杂散短尾句抢走；全局偏移估计 + 校正后应绑到真正的后一说话人。"""
        A, B = "SPEAKER_A1", "SPEAKER_A2"
        turns = [
            self._turn(A, 30_000, 32_000), self._turn(A, 33_000, 35_000),
            self._turn(A, 36_000, 38_000), self._turn(A, 38_000, 39_500),
            self._turn(A, 52_680, 53_080),   # 前一说话人的杂散短尾句
            self._turn(B, 54_680, 56_680),   # 后一说话人真正开口
        ]
        items = [
            {"text": "a", "start_ms": 27_700, "end_ms": 29_700},
            {"text": "b", "start_ms": 30_700, "end_ms": 32_700},
            {"text": "c", "start_ms": 33_700, "end_ms": 35_700},
            {"text": "d", "start_ms": 35_700, "end_ms": 37_200},
            {"text": "e", "start_ms": 27_800, "end_ms": 29_600},
            {"text": "大津！你一个后辈", "start_ms": 52_380, "end_ms": 54_380},  # contested
        ]
        self.assertEqual(estimate_subtitle_offset_ms(items, turns), 2_250)
        rows, _ = bind_lines(items, turns, offset_ms=2_250)
        self.assertEqual(rows[5]["cluster_id"], B)          # 校正后绑到长浜
        self.assertEqual(rows[5]["confidence"], 0.90)
        rows0, _ = bind_lines(items, turns, offset_ms=0)    # 回归对照：不校正则被抢
        self.assertEqual(rows0[5]["cluster_id"], A)

    def test_estimate_offset_needs_enough_lines(self):
        turns = [self._turn("SPEAKER_A1", 20_000, 22_000)]
        items = [{"text": "x", "start_ms": 17_700, "end_ms": 19_700}] * 3
        self.assertEqual(estimate_subtitle_offset_ms(items, turns), 0)  # <6 行不启用

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
            # Stage 3.7 evidence: present -> injected into the scene evidence pack
            (visual / "av_notes.json").write_text(json.dumps({
                "schema": "vts-av-notes/v1",
                "scene_notes": [{"scene_id": "SCENE_01", "start_ms": 0, "end_ms": 6000,
                                 "covered_pct": 100.0,
                                 "segments": [{"start_ms": 0, "end_ms": 6000,
                                               "visual": {"caption": "雨夜码头", "actions": [], "camera": [],
                                                          "scene_transition": "无"},
                                               "visible_text": [], "acoustic": {}, "uncertain": []}]}]
            }, ensure_ascii=False), encoding="utf-8")

            manifest = build_manifest(ws, max_keyframes=8)
            self.assertEqual(manifest["total_scenes"], 1)
            self.assertEqual(manifest["bible_names"], ["菈菈", "茉里", "面试的店主"])
            self.assertIn("[[SUB:", manifest["instructions"])
            scene = manifest["scenes"][0]
            self.assertIsNotNone(scene["av_notes"])
            self.assertEqual(scene["av_notes"]["covered_pct"], 100.0)
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


class TestOmniClient(unittest.TestCase):
    """Pure-function coverage of the direct-API client (no network, no ffmpeg)."""

    def test_select_audio_encoding_tiers(self):
        budget = omni_client.INLINE_RAW_BUDGET_BYTES
        self.assertEqual(omni_client.select_audio_encoding(10, budget, raw_size=100),
                         ("passthrough", None))
        self.assertEqual(omni_client.select_audio_encoding(200, 200 * omni_client.WAV_BYTES_PER_SEC),
                         ("wav", None))
        self.assertEqual(omni_client.select_audio_encoding(1440, budget), ("mp3", 40))  # 24 min
        self.assertEqual(omni_client.select_audio_encoding(2700, budget), ("mp3", 16))  # 45 min
        with self.assertRaises(ValueError):
            omni_client.select_audio_encoding(5400, budget)                             # 90 min

    def test_accumulate_sse_accumulates_and_stops(self):
        lines = [': keep-alive', '',
                 'data: {"choices":[{"delta":{"content":"Hello"}}]}',
                 'data: {bad json',
                 'data: {"choices":[{"delta":{"content":" world"}}]}',
                 'data: {"choices":[],"usage":{"total_tokens":7}}',
                 'data: [DONE]',
                 'data: {"choices":[{"delta":{"content":" ignored"}}]}']
        text, usage, finish = omni_client.accumulate_sse(lines)
        self.assertEqual(text, "Hello world")
        self.assertEqual(usage, {"total_tokens": 7})
        self.assertIsNone(finish)

    def test_accumulate_sse_reports_finish_reason(self):
        text, _, finish = omni_client.accumulate_sse([
            'data: {"choices":[{"delta":{"content":"{\\"segments\\":[]"},"finish_reason":"length"}]}',
            'data: [DONE]'])
        self.assertEqual(finish, "length")
        self.assertTrue(text)

    def test_accumulate_sse_accepts_bytes_lines(self):
        text, _, finish = omni_client.accumulate_sse(
            [b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}'])
        self.assertEqual(text, "ok")
        self.assertEqual(finish, "stop")

    def test_extract_json_payload_variants(self):
        self.assertEqual(omni_client.extract_json_payload('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(omni_client.extract_json_payload('Sure: {"a": 1} hope that helps'), {"a": 1})
        self.assertEqual(omni_client.extract_json_payload('{"a": [1, 2,]}'), {"a": [1, 2]})
        with self.assertRaises(ValueError):
            omni_client.extract_json_payload("no structure at all")
        with self.assertRaises(ValueError):
            omni_client.extract_json_payload("   ")

    def test_validate_endpoint_accepts_public_https(self):
        url = omni_client.validate_endpoint(
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            resolver=lambda host: ["8.8.8.8"])
        self.assertEqual(url, "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")

    def test_validate_endpoint_refuses_non_public(self):
        for url in ["http://dashscope.aliyuncs.com/v1",      # not https
                    "https://localhost/v1",
                    "https://metadata.local/v1",
                    "https://127.0.0.1/v1",
                    "https://192.168.1.1/v1",
                    "https://169.254.169.254/v1"]:           # cloud metadata
            with self.assertRaises(omni_client.OmniError, msg=url):
                omni_client.validate_endpoint(url, resolver=lambda host: ["8.8.8.8"])

    def test_validate_endpoint_refuses_private_dns_resolution(self):
        with self.assertRaises(omni_client.OmniError):
            omni_client.validate_endpoint("https://internal.example.com/v1",
                                          resolver=lambda host: ["10.0.0.5"])

    def test_resolve_public_host_blocks_loopback(self):
        fake = [(2, 1, 6, "", ("127.0.0.1", 443))]
        with mock.patch.object(omni_client.socket, "getaddrinfo", return_value=fake):
            with self.assertRaises(omni_client.OmniError):
                omni_client.resolve_public_host("evil.example.com")

    def test_resolve_public_host_accepts_global(self):
        fake = [(2, 1, 6, "", ("8.8.8.8", 443))]
        with mock.patch.object(omni_client.socket, "getaddrinfo", return_value=fake):
            self.assertEqual(omni_client.resolve_public_host("api.example.com"), ["8.8.8.8"])

    def test_omni_error_transient_classification(self):
        for kind in ("rate_limited", "server", "timeout", "connection", "empty"):
            self.assertTrue(omni_client.OmniError(kind).transient, kind)
        for kind in ("auth", "bad_request", "http"):
            self.assertFalse(omni_client.OmniError(kind).transient, kind)

    def test_call_omni_chat_retries_transient_then_succeeds(self):
        with mock.patch.object(omni_client, "validate_endpoint",
                               side_effect=lambda u: u + "/chat/completions"), \
             mock.patch.object(omni_client, "_post_stream",
                               side_effect=[omni_client.OmniError("server", 503),
                                            ("payload", {"total_tokens": 5})]), \
             mock.patch.object(omni_client.time, "sleep"):
            text, usage = omni_client.call_omni_chat(
                [{"role": "user", "content": "hi"}], api_key="k", attempts=3)
        self.assertEqual(text, "payload")
        self.assertEqual(usage, {"total_tokens": 5})

    def test_call_omni_chat_fails_fast_on_auth(self):
        calls = []

        def boom(*a, **kw):
            calls.append(1)
            raise omni_client.OmniError("auth", 401)

        with mock.patch.object(omni_client, "validate_endpoint",
                               side_effect=lambda u: u + "/chat/completions"), \
             mock.patch.object(omni_client, "_post_stream", side_effect=boom):
            with self.assertRaises(omni_client.OmniError):
                omni_client.call_omni_chat([{"role": "user", "content": "hi"}],
                                           api_key="k", attempts=3)
        self.assertEqual(len(calls), 1)

    def test_call_omni_chat_requires_key(self):
        with mock.patch.dict(os.environ, {"DASHSCOPE_API_KEY": ""}):
            with self.assertRaises(omni_client.OmniError):
                omni_client.call_omni_chat([{"role": "user", "content": "hi"}])


class TestSpeakerDiarizeRun(unittest.TestCase):
    """cmd_run integration on a temp workspace: no network (diarize_audio_file
    stubbed), no ffmpeg - only workorder plumbing, resume and exit codes."""

    def _make_ws(self, parts):
        handle = tempfile.TemporaryDirectory()
        self.addCleanup(handle.cleanup)
        root = Path(handle.name)
        audio_dir = root / ".cache" / "audio"
        audio_dir.mkdir(parents=True)
        work = {"status": "awaiting_omni_diarization",
                "num_speakers_hint": None, "language_hint": None, "parts": []}
        for i, (start_ms, end_ms) in enumerate(parts):
            work["parts"].append({
                "file": f"audio/source_audio.part{i:03d}.m4a",
                "start_ms": start_ms, "end_ms": end_ms,
                "output": f"audio/omni_diarized.part{i:03d}.json",
            })
            (audio_dir / f"source_audio.part{i:03d}.m4a").write_bytes(b"\x00" * 64)
        (audio_dir / "diarize_workorder.json").write_text(
            json.dumps(work, ensure_ascii=False, indent=2), encoding="utf-8")
        return root, audio_dir

    @staticmethod
    def _fake_result():
        return {"speakers": ["Speaker 1"],
                "segments": [{"speaker": "Speaker 1", "start": 0.0, "end": 1.0, "text": "hi"}],
                "meta": {"backend": "direct_api", "model": "qwen3.8-omni-flash"}}

    def _run(self, ws, force=False):
        err = io.StringIO()
        with self.assertRaises(SystemExit) as ctx:
            with contextlib.redirect_stderr(err):
                with mock.patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}):
                    cmd_run(str(ws), force=force)
        return ctx.exception.code, err.getvalue()

    def test_cmd_run_requires_workorder(self):
        with tempfile.TemporaryDirectory() as tmp:
            err = io.StringIO()
            with self.assertRaises(SystemExit) as ctx:
                with contextlib.redirect_stderr(err):
                    with mock.patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}):
                        cmd_run(tmp)
            self.assertEqual(ctx.exception.code, 1)

    def test_cmd_run_missing_key_exits_8(self):
        ws, _ = self._make_ws([(0, 60_000)])
        err = io.StringIO()
        with self.assertRaises(SystemExit) as ctx:
            with contextlib.redirect_stderr(err):
                with mock.patch.dict(os.environ, {"DASHSCOPE_API_KEY": ""}):
                    cmd_run(str(ws))
        self.assertEqual(ctx.exception.code, 8)
        self.assertIn("DASHSCOPE_API_KEY", err.getvalue())

    def test_cmd_run_skips_valid_outputs_without_key(self):
        ws, audio_dir = self._make_ws([(0, 60_000)])
        (audio_dir / "omni_diarized.part000.json").write_text(
            json.dumps(self._fake_result()), encoding="utf-8")
        err = io.StringIO()
        with self.assertRaises(SystemExit) as ctx:
            with contextlib.redirect_stderr(err):
                with mock.patch.dict(os.environ, {"DASHSCOPE_API_KEY": ""}):
                    cmd_run(str(ws))
        self.assertEqual(ctx.exception.code, 0)
        self.assertIn("[SKIP]", err.getvalue())

    def test_cmd_run_calls_model_and_writes_outputs(self):
        ws, audio_dir = self._make_ws([(0, 60_000)])
        with mock.patch.object(omni_client, "diarize_audio_file",
                               return_value=self._fake_result()) as stub:
            code, err = self._run(ws)
        self.assertEqual(code, 0)
        stub.assert_called_once()
        saved = json.loads((audio_dir / "omni_diarized.part000.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["meta"]["backend"], "direct_api")
        self.assertIn("[OK", err)

    def test_cmd_run_one_failure_does_not_stop_other_parts(self):
        ws, audio_dir = self._make_ws([(0, 60_000), (57_000, 120_000)])
        with mock.patch.object(omni_client, "diarize_audio_file",
                               side_effect=[omni_client.OmniError("server", 502),
                                            self._fake_result()]):
            code, err = self._run(ws)
        self.assertEqual(code, 7)
        self.assertFalse((audio_dir / "omni_diarized.part000.json").exists())
        self.assertTrue((audio_dir / "omni_diarized.part001.json").exists())
        self.assertIn("[ERROR]", err)

    def test_cmd_run_resume_keeps_finished_parts(self):
        ws, audio_dir = self._make_ws([(0, 60_000), (57_000, 120_000)])
        (audio_dir / "omni_diarized.part000.json").write_text(
            json.dumps(self._fake_result()), encoding="utf-8")
        with mock.patch.object(omni_client, "diarize_audio_file",
                               return_value=self._fake_result()) as stub:
            code, err = self._run(ws)
        self.assertEqual(code, 0)
        stub.assert_called_once()  # only part001 was re-diagnosed
        self.assertIn("[SKIP]", err)

    def test_cmd_run_force_redoes_everything(self):
        ws, audio_dir = self._make_ws([(0, 60_000)])
        (audio_dir / "omni_diarized.part000.json").write_text(
            json.dumps(self._fake_result()), encoding="utf-8")
        with mock.patch.object(omni_client, "diarize_audio_file",
                               return_value=self._fake_result()) as stub:
            code, err = self._run(ws, force=True)
        self.assertEqual(code, 0)
        stub.assert_called_once()

    def test_cmd_run_warns_on_overlong_part(self):
        ws, audio_dir = self._make_ws([(0, 1_800_000)])  # 30 min
        (audio_dir / "omni_diarized.part000.json").write_text(
            json.dumps(self._fake_result()), encoding="utf-8")
        code, err = self._run(ws)
        self.assertEqual(code, 0)
        self.assertIn("--chunk-seconds 1200", err)

    def test_build_output_reports_backend_provenance(self):
        out = build_output("v.mkv", [], [], {}, [], {}, {}, ["a.json"],
                           backend="direct_api",
                           part_metas=[{"backend": "direct_api", "model": "qwen3.8-omni-flash"}])
        self.assertEqual(out["diarization_source"]["backend"], "direct_api")
        self.assertEqual(out["diarization_source"]["model"], "qwen3.8-omni-flash")
        default = build_output("v.mkv", [], [], {}, [], {}, {}, ["a.json"])
        self.assertEqual(default["diarization_source"]["backend"], "mcp_tool")

    def test_make_workorder_note_mentions_both_paths(self):
        work = make_workorder("v.mkv", 60_000, [(0, 60_000)], None, None, no_audio=False)
        self.assertIn("`run`", work["note"])
        self.assertIn("omni_multi_speaker_asr", work["note"])


class TestSubtitleStreamSelection(unittest.TestCase):
    """Embedded track selection: user language first, signs/forced tracks skipped."""

    def test_prefers_user_language(self):
        streams = [{"index": 4, "tags": {"language": "eng", "title": ""}},
                   {"index": 18, "tags": {"language": "chi", "title": "Simplified"}}]
        sel, skipped, reason = select_embedded_stream(streams, ["chi", "zho"])
        self.assertEqual(sel["index"], 18)
        self.assertEqual(reason, "language_match:chi")
        self.assertEqual(skipped, [])

    def test_skips_forced_disposition_even_in_user_language(self):
        streams = [{"index": 3, "tags": {"language": "chi", "title": "Simplified"},
                    "disposition": {"forced": 1}},
                   {"index": 4, "tags": {"language": "eng", "title": ""}}]
        sel, skipped, reason = select_embedded_stream(streams, ["chi", "zho"])
        self.assertEqual(sel["index"], 4)
        self.assertEqual(reason, "language_fallback_first_non_sign")
        self.assertEqual(skipped[0]["why"], "disposition_forced")

    def test_skips_signs_titled_track(self):
        streams = [{"index": 4, "tags": {"language": "eng", "title": "Signs & Songs"}},
                   {"index": 18, "tags": {"language": "chi", "title": "Simplified"}}]
        sel, skipped, reason = select_embedded_stream(streams, ["chi", "zho"])
        self.assertEqual(sel["index"], 18)
        self.assertEqual(skipped[0]["why"], "title_keyword:sign")

    def test_falls_back_to_first_survivor_without_language_match(self):
        streams = [{"index": 7, "tags": {"language": "ger", "title": ""}},
                   {"index": 8, "tags": {"language": "spa", "title": ""}}]
        sel, _, reason = select_embedded_stream(streams, ["chi", "zho"])
        self.assertEqual(sel["index"], 7)
        self.assertEqual(reason, "language_fallback_first_non_sign")

    def test_last_resort_when_everything_looks_like_signs(self):
        streams = [{"index": 3, "tags": {"language": "eng", "title": "Forced"},
                    "disposition": {"forced": 1}},
                   {"index": 4, "tags": {"language": "eng", "title": "Signs"}}]
        sel, skipped, reason = select_embedded_stream(streams, ["chi", "zho"])
        self.assertEqual(sel["index"], 3)
        self.assertEqual(reason, "last_resort_all_look_like_signs")
        self.assertEqual(len(skipped), 2)  # one by disposition, one by title keyword


class TestWorkspaceDoctor(unittest.TestCase):
    """doctor's diarization preflight: presence-only key reporting (the key value
    must never appear in the report). ffmpeg/ffprobe are faked for determinism."""

    def _doctor(self, env, tools_ready=True):
        out = io.StringIO()
        which = (lambda name: f"/usr/bin/{name}") if tools_ready else (lambda name: None)
        fake = subprocess.CompletedProcess([], 0, stdout="ffmpeg version 7.0\n", stderr="")
        code = 0  # doctor returns normally on success; sys.exit(3) only when tools missing
        with mock.patch.dict(os.environ, env), \
             mock.patch.object(workspace.shutil, "which", side_effect=which), \
             mock.patch.object(workspace.subprocess, "run", return_value=fake), \
             contextlib.redirect_stdout(out):
            try:
                workspace.run_doctor_check()
            except SystemExit as e:
                code = e.code
        return code, out.getvalue()

    def test_doctor_reports_key_set_and_never_leaks_value(self):
        fake_key = "-".join(["test", "key", "12345"])  # assembled; never a real credential
        code, out = self._doctor({"DASHSCOPE_API_KEY": fake_key})
        self.assertEqual(code, 0)
        report = json.loads(out)["report"]
        self.assertEqual(report["diarization"]["dashscope_api_key"], "set")
        self.assertTrue(report["diarization"]["ready"])
        self.assertNotIn(fake_key, out)

    def test_doctor_reports_key_missing(self):
        code, out = self._doctor({"DASHSCOPE_API_KEY": ""})
        self.assertEqual(code, 0)  # key absence is reported, not fatal, at doctor level
        report = json.loads(out)["report"]
        self.assertEqual(report["diarization"]["dashscope_api_key"], "missing")
        self.assertFalse(report["diarization"]["ready"])

    def test_doctor_reports_endpoint_host_and_model(self):
        _, out = self._doctor({"DASHSCOPE_BASE_URL": "https://api.example.com/v1",
                               "V2S_OMNI_MODEL": "test-model"})
        report = json.loads(out)["report"]
        self.assertEqual(report["diarization"]["dashscope_base_url_host"], "api.example.com")
        self.assertEqual(report["diarization"]["model"], "test-model")

    def test_doctor_default_endpoint_host(self):
        _, out = self._doctor({})
        report = json.loads(out)["report"]
        self.assertEqual(report["diarization"]["dashscope_base_url_host"], "dashscope.aliyuncs.com")

    def test_doctor_exit_3_when_ffmpeg_missing(self):
        code, out = self._doctor({}, tools_ready=False)
        self.assertEqual(code, 3)
        report = json.loads(out)["report"]
        self.assertFalse(report["ffmpeg"]["ready"])
        self.assertFalse(report["ffprobe"]["ready"])


class TestAvUnderstand(unittest.TestCase):
    """Stage 3.7 pure planning + cmd_run/prepare integration (no network, no ffmpeg)."""

    def test_plan_segments_splits_without_exceeding_cap(self):
        """200s -> three windows all <= 90s: the old 45s tail fold let the last
        window balloon to 115s while three docs promised <= 90s."""
        segs = plan_segments([{"scene_id": "SCENE_01", "start_ms": 0, "end_ms": 200_000}])
        self.assertEqual([(s["start_ms"], s["end_ms"]) for s in segs],
                         [(0, 90_000), (85_000, 175_000), (170_000, 200_000)])
        self.assertTrue(all(s["end_ms"] - s["start_ms"] <= 90_000 for s in segs))

    def test_plan_segments_folds_tiny_tail_within_documented_exception(self):
        """A <10s tail folds into the last window: 91s stays ONE segment (90s cap
        + documented 10s fold exception) instead of paying a full call for a 1s
        sliver."""
        segs = plan_segments([{"scene_id": "SCENE_01", "start_ms": 0, "end_ms": 91_000}])
        self.assertEqual([(s["start_ms"], s["end_ms"]) for s in segs], [(0, 91_000)])

    def test_plan_segments_short_scene_single_window(self):
        segs = plan_segments([{"scene_id": "SCENE_01", "start_ms": 30_000, "end_ms": 90_000}])
        self.assertEqual([(s["start_ms"], s["end_ms"]) for s in segs], [(30_000, 90_000)])

    def test_plan_segments_never_cross_scene_boundaries(self):
        segs = plan_segments([{"scene_id": "A", "start_ms": 0, "end_ms": 100_000},
                              {"scene_id": "B", "start_ms": 100_000, "end_ms": 120_000}])
        self.assertTrue(all(s["scene_id"] in ("A", "B") for s in segs))
        self.assertEqual([s["end_ms"] for s in segs if s["scene_id"] == "A"], [90_000, 100_000])
        self.assertEqual([s["start_ms"] for s in segs if s["scene_id"] == "B"], [100_000])

    def test_parse_note_shifts_to_absolute_and_drops_garbage(self):
        note = parse_note({
            "visual": {"caption": "雨夜", "scene_transition": "硬切",
                       "actions": [{"start": 1.5, "end": 3.0, "who": "红衣女子", "what": "拔刀"},
                                   "not-a-dict",
                                   {"start": "abc", "end": 9},
                                   {"start": 5.0, "end": 2.0, "who": "x", "what": "倒序"}],
                       "camera": [{"start": 0, "end": 4, "movement": "缓推"}]},
            "visible_text": "should be a list",
            "acoustic": {"events": [{"start": 2.2, "what": "雷声"}], "music_mood": "紧张"},
            "uncertain": [7, "第 40s 附近人物身份无法确认"],
        }, 420_000, 510_000)
        self.assertEqual(note["visual"]["caption"], "雨夜")
        self.assertEqual(note["visual"]["actions"],
                         [{"who": "红衣女子", "what": "拔刀", "start": 421_500, "end": 423_000,
                           "mouth_state": "unknown", "mouth_motion": "unknown"}])
        self.assertEqual(note["visual"]["camera"][0]["start"], 420_000)
        self.assertEqual(note["visible_text"], [])
        self.assertEqual(note["acoustic"]["events"][0]["start"], 422_200)
        self.assertEqual(note["acoustic"]["music_mood"], "紧张")
        self.assertEqual(note["uncertain"], ["第 40s 附近人物身份无法确认"])

    def test_parse_note_accepts_key_aliases_and_time_strings(self):
        """Models drift between start/start_time and what/description; a MM:SS
        string timebase must parse instead of silently vanishing."""
        note = parse_note({
            "visual": {"actions": [
                {"start_time": 5.0, "end_time": 8.0, "who": "红衣女子", "description": "撑伞快走"},
                {"start": "00:01:05", "end": "00:01:08", "who": "老者", "what": "驻足"}]},
        }, 100_000, 190_000)
        self.assertEqual(note["visual"]["actions"],
                         [{"who": "红衣女子", "what": "撑伞快走", "start": 105_000, "end": 108_000,
                           "mouth_state": "unknown", "mouth_motion": "unknown"},
                          {"who": "老者", "what": "驻足", "start": 165_000, "end": 168_000,
                           "mouth_state": "unknown", "mouth_motion": "unknown"}])

    def test_parse_note_drops_out_of_timebase_and_counts(self):
        """A 200s timestamp inside a 90s window means the model ignored the local
        timebase - dropping it beats clamping it into a confident wrong timecode."""
        note = parse_note({
            "visual": {"actions": [
                {"start": 200.0, "end": 210.0, "who": "B", "what": "越界"},
                {"start": 2.0, "end": 3.0, "who": "D", "what": "正常"}]},
        }, 100_000, 190_000)
        self.assertEqual([a["what"] for a in note["visual"]["actions"]], ["正常"])
        self.assertEqual(note["dropped"]["actions"], 1)

    def test_parse_note_survives_non_dict_input(self):
        for garbage in (None, [], "x", 42):
            note = parse_note(garbage, 0, 1_000)
            self.assertEqual(note["visual"]["actions"], [])
            self.assertEqual(note["acoustic"]["events"], [])

    def test_scene_coverage_union(self):
        scene = {"start_ms": 0, "end_ms": 100_000}
        self.assertEqual(scene_coverage(scene, [(0, 50_000), (50_000, 100_000)])[0], 100.0)
        self.assertEqual(scene_coverage(scene, [(0, 60_000), (55_000, 90_000)])[0], 90.0)
        self.assertEqual(scene_coverage(scene, [(0, 40_000), (80_000, 100_000)])[0], 60.0)
        self.assertEqual(scene_coverage(scene, [])[0], 0.0)
        self.assertEqual(scene_coverage(scene, [(200_000, 300_000)])[0], 0.0)  # outside span

    def test_prompt_declares_time_basis_and_name_ban(self):
        prompt = build_prompt(420_000, 510_000)
        self.assertIn("0..90.0", prompt)
        self.assertIn("420.0..510.0", prompt)
        self.assertIn("NEVER output a real name", prompt)
        self.assertIn("Do NOT transcribe spoken dialogue", prompt)
        self.assertIn("Chinese", prompt)

    def _make_ws(self, scenes, segments):
        handle = tempfile.TemporaryDirectory()
        self.addCleanup(handle.cleanup)
        root = Path(handle.name)
        av_dir = root / ".cache" / "av"
        av_dir.mkdir(parents=True)
        (root / ".cache" / "visual").mkdir(parents=True)
        (root / ".cache" / "visual" / "scenes.json").write_text(
            json.dumps({"scenes": scenes}, ensure_ascii=False), encoding="utf-8")
        work = {"status": "ready", "segment_seconds": 90.0, "overlap_sec": 5.0,
                "scenes": scenes, "segments": []}
        for i, (sid, s_ms, e_ms) in enumerate(segments):
            work["segments"].append({"file": f"av/seg_{i:03d}.mp4", "scene_id": sid,
                                     "start_ms": s_ms, "end_ms": e_ms,
                                     "output": f"av/av_note_{i:03d}.json"})
            (av_dir / f"seg_{i:03d}.mp4").write_bytes(b"\x00" * 32)
        (av_dir / "av_workorder.json").write_text(
            json.dumps(work, ensure_ascii=False, indent=2), encoding="utf-8")
        return root, av_dir

    @staticmethod
    def _fake_note():
        return {"visual": {"caption": "雨夜", "actions": [{"start": 1.0, "end": 2.0,
                                                          "who": "红衣女子", "what": "转身"}],
                            "camera": [], "scene_transition": "无"},
                "visible_text": [], "acoustic": {"events": [], "music_mood": "无"},
                "uncertain": []}

    def _run(self, ws, force=False):
        err = io.StringIO()
        with self.assertRaises(SystemExit) as ctx:
            with contextlib.redirect_stderr(err):
                with mock.patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}):
                    av_run(str(ws), force=force)
        return ctx.exception.code, err.getvalue()

    def test_cmd_prepare_writes_workorder_and_segments(self):
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            (ws / "materials").mkdir()
            (ws / "materials" / "clip.mkv").write_bytes(b"\x00")
            (ws / ".cache").mkdir()
            (ws / ".cache" / "visual").mkdir()
            (ws / ".cache" / "visual" / "scenes.json").write_text(json.dumps(
                {"scenes": [{"scene_id": "SCENE_01", "start_ms": 0, "end_ms": 200_000}]}),
                encoding="utf-8")
            err = io.StringIO()

            def fake_cut(video, out_path, s_ms, e_ms):
                Path(out_path).write_bytes(b"\x00" * 32)

            with mock.patch.object(av_understand_module, "require_binaries", lambda: None), \
                 mock.patch.object(av_understand_module, "locate_video",
                                   return_value=str(ws / "materials" / "clip.mkv")), \
                 mock.patch.object(av_understand_module, "_cut_segment", side_effect=fake_cut), \
                 contextlib.redirect_stderr(err):
                with self.assertRaises(SystemExit) as ctx:
                    av_prepare(str(ws), 90.0, None)
            self.assertEqual(ctx.exception.code, 0)
            work = json.loads((ws / ".cache" / "av" / "av_workorder.json").read_text(encoding="utf-8"))
            self.assertEqual(len(work["segments"]), 3)  # 200s scene -> [90, 90, 20]s, all <= cap
            self.assertTrue((ws / ".cache" / "av" / "seg_000.mp4").is_file())

    def test_cmd_run_writes_notes_and_av_notes_json(self):
        ws, av_dir = self._make_ws(
            [{"scene_id": "SCENE_01", "start_ms": 0, "end_ms": 60_000}],
            [("SCENE_01", 0, 60_000)])
        with mock.patch.object(omni_client, "understand_video_segment",
                               return_value=(self._fake_note(), {"backend": "direct_api"})) as stub:
            code, err = self._run(ws)
        self.assertEqual(code, 0)
        stub.assert_called_once()
        self.assertTrue((av_dir / "av_note_000.json").is_file())
        notes = json.loads((ws / ".cache" / "visual" / "av_notes.json").read_text(encoding="utf-8"))
        self.assertEqual(notes["schema"], av_understand_module.AV_NOTES_SCHEMA)
        self.assertEqual(notes["scene_notes"][0]["covered_pct"], 100.0)
        self.assertEqual(notes["scene_notes"][0]["segments"][0]["visual"]["actions"][0]["start"],
                         1_000)  # local 1.0s -> absolute 1000ms
        self.assertTrue(notes["scene_notes"][0]["segments"][0]["substantive"])

    def test_cmd_run_missing_key_exits_8(self):
        ws, _ = self._make_ws([{"scene_id": "SCENE_01", "start_ms": 0, "end_ms": 60_000}],
                              [("SCENE_01", 0, 60_000)])
        err = io.StringIO()
        with self.assertRaises(SystemExit) as ctx:
            with contextlib.redirect_stderr(err):
                with mock.patch.dict(os.environ, {"DASHSCOPE_API_KEY": ""}):
                    av_run(str(ws))
        self.assertEqual(ctx.exception.code, 8)

    def test_cmd_run_failure_does_not_stop_other_segments(self):
        ws, av_dir = self._make_ws(
            [{"scene_id": "SCENE_01", "start_ms": 0, "end_ms": 120_000}],
            [("SCENE_01", 0, 90_000), ("SCENE_01", 85_000, 120_000)])
        with mock.patch.object(omni_client, "understand_video_segment",
                               side_effect=[omni_client.OmniError("server", 502),
                                            (self._fake_note(), {"backend": "direct_api"})]):
            code, err = self._run(ws)
        self.assertEqual(code, 7)
        self.assertFalse((av_dir / "av_note_000.json").exists())
        self.assertTrue((av_dir / "av_note_001.json").exists())
        self.assertIn("[ERROR]", err)

    def test_cmd_run_skips_valid_notes_without_key(self):
        ws, av_dir = self._make_ws(
            [{"scene_id": "SCENE_01", "start_ms": 0, "end_ms": 60_000}],
            [("SCENE_01", 0, 60_000)])
        (av_dir / "av_note_000.json").write_text(
            json.dumps(self._fake_note()), encoding="utf-8")
        err = io.StringIO()
        with self.assertRaises(SystemExit) as ctx:
            with contextlib.redirect_stderr(err):
                with mock.patch.dict(os.environ, {"DASHSCOPE_API_KEY": ""}):
                    av_run(str(ws))
        self.assertEqual(ctx.exception.code, 0)
        self.assertIn("[SKIP]", err.getvalue())

    def test_cmd_merge_missing_note_exits_7(self):
        ws, _ = self._make_ws(
            [{"scene_id": "SCENE_01", "start_ms": 0, "end_ms": 60_000}],
            [("SCENE_01", 0, 60_000)])  # no note file written
        with self.assertRaises(SystemExit) as ctx:
            with contextlib.redirect_stderr(io.StringIO()):
                av_merge(str(ws))
        self.assertEqual(ctx.exception.code, 7)


class TestOpEdWindows(unittest.TestCase):
    """OP/ED window config: bible.json is the single source; spans >=50% inside
    a window are non-narrative; filtered lines reindex contiguously."""

    def test_load_windows_from_bible(self):
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            (ws / "materials").mkdir()
            (ws / "materials" / "bible.json").write_text(json.dumps({
                "op_ed_windows": [
                    {"start_ms": 84000, "end_ms": 105000, "label": "OP"},
                    {"start_ms": 1440000, "end_ms": 1320000},           # inverted -> skipped
                    {"start_ms": "x", "end_ms": 5},                     # malformed -> skipped
                ]}), encoding="utf-8")
            windows = op_ed.load_windows(str(ws))
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["label"], "OP")

    def test_load_windows_without_config_is_noop(self):
        self.assertEqual(op_ed.load_windows(None), [])
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(op_ed.load_windows(td), [])

    def test_matching_label_threshold(self):
        windows = [{"start_ms": 84000, "end_ms": 105000, "label": "OP"}]
        self.assertEqual(op_ed.matching_label(84000, 105000, windows), "OP")     # fully inside
        self.assertEqual(op_ed.matching_label(84000, 104000, windows), "OP")     # >= 50%
        self.assertIsNone(op_ed.matching_label(83000, 85000, windows))           # 50% boundary line kept
        self.assertIsNone(op_ed.matching_label(0, 60000, windows))               # narrative

    def test_filter_items_reindexes_and_records(self):
        windows = [{"start_ms": 84000, "end_ms": 105000, "label": "OP"}]
        items = [{"index": 1, "start_ms": 1000, "end_ms": 2000, "text": "a"},
                 {"index": 2, "start_ms": 90000, "end_ms": 91000, "text": "歌词"},
                 {"index": 3, "start_ms": 120000, "end_ms": 121000, "text": "b"},
                 {"index": 4, "start_ms": 100000, "end_ms": 101000, "text": "歌词2"}]
        kept, meta = op_ed.filter_items(items, windows)
        self.assertEqual([it["text"] for it in kept], ["a", "b"])
        self.assertEqual([it["index"] for it in kept], [1, 2])   # contiguous [[SUB:n]] contract
        self.assertEqual(meta["removed"], {"OP": 2})
        self.assertEqual(meta["removed_total"], 2)

    def test_filter_items_no_windows_is_identity(self):
        items = [{"index": 1, "start_ms": 0, "end_ms": 1, "text": "a"}]
        kept, meta = op_ed.filter_items(items, [])
        self.assertIs(kept, items)
        self.assertEqual(meta, {})


class TestAvHardening(unittest.TestCase):
    """The review-driven fixes: substance gate, dedup, transport shape, orphan
    cleanup, no second model call for a JSON problem."""

    def test_has_substance_and_is_valid_note_reject_empty_evidence(self):
        self.assertFalse(has_substance(parse_note({}, 0, 1000)))
        self.assertFalse(has_substance(parse_note({"visual": {}}, 0, 1000)))
        self.assertTrue(has_substance(parse_note(
            {"visual": {"caption": "雨夜"}}, 0, 1000)))
        p = _tmp_file(".json")
        p.write_text(json.dumps({"raw": {}, "meta": {}}), encoding="utf-8")
        try:
            self.assertFalse(av_understand_module._is_valid_note(p))  # used to be True
        finally:
            p.unlink()

    def test_dedup_entries_merges_overlap_zone_duplicates(self):
        previous = [{"start": 0, "end": 90_000, "who": "红衣女子", "what": "转身看窗"}]
        entries = [
            {"start": 87_500, "end": 89_500, "who": "红衣女子", "what": "转身看窗。"},  # dup in overlap
            {"start": 95_000, "end": 99_000, "who": "红衣女子", "what": "坐下"},        # new
        ]
        kept = dedup_entries(entries, previous, "what")
        self.assertEqual([e["what"] for e in kept], ["坐下"])
        self.assertEqual(previous[0]["end"], 90_000)  # span extension stays inside the dup itself

    def test_dedup_entries_keeps_distinct_entries(self):
        previous = [{"start": 0, "end": 5_000, "who": "A", "what": "转身看窗"}]
        entries = [{"start": 1_000, "end": 4_000, "who": "B", "what": "系鞋带"}]
        self.assertEqual(len(dedup_entries(entries, previous, "what")), 1)

    def test_build_video_part_shape(self):
        p = _tmp_file(".mp4")
        p.write_bytes(b"\x00" * 16)
        try:
            part = build_video_part(str(p), fps=2.0, max_pixels=100_000)
            self.assertEqual(part["type"], "video_url")
            self.assertTrue(part["video_url"]["url"].startswith("data:video/mp4;base64,"))
            # fps/max_pixels MUST sit at the part's top level - the endpoint only
            # honors them there
            self.assertEqual(part["fps"], 2.0)
            self.assertEqual(part["max_pixels"], 100_000)
            self.assertNotIn("fps", part["video_url"])
            self.assertTrue(part["use_audio_in_video"])
        finally:
            p.unlink()

    def test_understand_video_segment_does_not_rebill_on_bad_json(self):
        p = _tmp_file(".mp4")
        p.write_bytes(b"\x00" * 16)  # tiny -> fit_video passthrough, no ffmpeg
        try:
            with mock.patch.object(omni_client, "call_omni_chat",
                                   return_value=("this is not json at all", {})) as stub:
                with self.assertRaises(ValueError):
                    understand_video_segment(str(p), prompt="p", api_key="k")
            self.assertEqual(stub.call_count, 1)  # no second model call for a JSON problem
        finally:
            p.unlink()

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg required")
    def test_fit_video_ladder_cleans_orphans(self):
        src = _tmp_file(".mp4")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-f", "lavfi", "-i", "testsrc=duration=2:size=640x480:rate=10",
                        "-c:v", "libx264", "-preset", "ultrafast", str(src)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
        try:
            payload, fmt = fit_video(str(src), budget=src.stat().st_size - 1)
            self.assertLessEqual(payload.stat().st_size, src.stat().st_size - 1)
            self.assertEqual(fmt, "mp4")
            leftovers = [o for o in src.parent.glob(f".fitv_{src.stem}_*.mp4") if o != payload]
            self.assertEqual(leftovers, [])           # success path leaves no busted tiers
            payload.unlink(missing_ok=True)
        finally:
            src.unlink(missing_ok=True)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg required")
    def test_fit_video_impossible_budget_raises_without_orphans(self):
        src = _tmp_file(".mp4")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-f", "lavfi", "-i", "testsrc=duration=2:size=640x480:rate=10",
                        "-c:v", "libx264", "-preset", "ultrafast", str(src)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
        try:
            with self.assertRaises(ValueError):
                fit_video(str(src), budget=1)
            self.assertEqual(list(src.parent.glob(f".fitv_{src.stem}_*.mp4")), [])  # every busted tier cleaned
        finally:
            src.unlink(missing_ok=True)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg required")
    def test_end_to_end_prepare_cut_merge_roundtrip(self):
        """The no-network E2E leg: real ffmpeg cut -> fake evidence -> merge ->
        av_notes.json on disk with a summary (not the whole document) on stdout."""
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            (ws / "materials").mkdir()
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                            "-f", "lavfi", "-i", "testsrc=duration=30:size=320x240:rate=10",
                            "-f", "lavfi", "-i", "sine=frequency=440:duration=30",
                            "-shortest", "-c:v", "libx264", "-preset", "ultrafast",
                            "-c:a", "aac", str(ws / "materials" / "clip.mp4")],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
            (ws / ".cache" / "visual").mkdir(parents=True)
            (ws / ".cache" / "visual" / "scenes.json").write_text(json.dumps(
                {"scenes": [{"scene_id": "SCENE_01", "start_ms": 0, "end_ms": 30_000}]}),
                encoding="utf-8")
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                with self.assertRaises(SystemExit) as ctx:
                    av_prepare(str(ws), 90.0, None)
            self.assertEqual(ctx.exception.code, 0)
            payload = ws / ".cache" / "av" / "seg_000.mp4"
            self.assertTrue(payload.is_file())
            probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                    "format=duration", "-of", "csv=p=0", str(payload)],
                                   stdout=subprocess.PIPE, text=True, timeout=30)
            self.assertAlmostEqual(float(probe.stdout.strip()), 30.0, delta=1.5)

            (ws / ".cache" / "av" / "av_note_000.json").write_text(json.dumps({
                "raw": {"visual": {"caption": "测试画面", "actions": [], "camera": [],
                                   "scene_transition": "无"},
                        "visible_text": [], "acoustic": {"events": [], "music_mood": "无"},
                        "uncertain": []},
                "meta": {"usage": {"total_tokens": 6397}}}), encoding="utf-8")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                av_merge(str(ws))
            summary = json.loads(out.getvalue())
            self.assertEqual(summary["scenes"], 1)
            self.assertEqual(summary["covered_min"], 100.0)
            self.assertEqual(summary["tokens"], 6397)
            self.assertTrue((ws / ".cache" / "visual" / "av_notes.json").is_file())
            self.assertNotIn('"scene_notes"', out.getvalue())  # stdout is a summary, not the document


class TestV052ContractGaps(unittest.TestCase):
    """Regressions for the v0.5.1 review: each one reproduces a defect that was
    verified on main before this branch."""

    # --- dedup must not rewrite an earlier segment's committed row ---------
    def test_dedup_does_not_rewrite_emitted_rows(self):
        pool = []
        seg1 = dedup_entries([{"start": 1_000, "end": 3_000, "who": "红衣女子", "what": "撑伞快走"}],
                             pool, "what")
        seg2 = dedup_entries([{"start": 2_500, "end": 9_000, "who": "红衣女子", "what": "撑伞快走"}],
                             pool, "what")
        self.assertEqual(seg2, [])                    # the overlap-zone duplicate is dropped
        self.assertEqual(seg1[0]["end"], 3_000)       # what seg1 returned stays verbatim
        self.assertEqual(pool[0]["end"], 9_000)       # the pool carries the merged span forward

    def test_dedup_chain_matches_the_widest_pool_span(self):
        """A third description of the same beat, 5 s after the first window ended, only
        merges if the pool - not just the previous row - holds the widened span."""
        pool = []
        dedup_entries([{"start": 1_000, "end": 3_000, "who": "A", "what": "撑伞快走"}], pool, "what")
        dedup_entries([{"start": 2_500, "end": 9_000, "who": "A", "what": "撑伞快走"}], pool, "what")
        third = dedup_entries([{"start": 8_000, "end": 9_500, "who": "A", "what": "撑伞快走"}],
                              pool, "what")
        self.assertEqual(third, [])

    # --- the resume gate must not mistake real evidence for emptiness ------
    @staticmethod
    def _note_file(payload):
        p = _tmp_file(".json")
        p.write_text(json.dumps({"raw": payload, "meta": {}}, ensure_ascii=False), encoding="utf-8")
        return p

    def test_resume_gate_accepts_late_timestamps_without_caption(self):
        """v0.5.1 parsed the candidate with a synthetic 1-second span, so a note whose
        only evidence is an action at 5 s read as empty and was re-billed on every run."""
        p = self._note_file({"visual": {"caption": "", "actions": [
            {"start": 5.0, "end": 8.0, "who": "红衣女子", "what": "撑伞快走"}]},
            "acoustic": {"music_mood": ""}})
        try:
            self.assertTrue(av_understand_module._is_valid_note(p))
        finally:
            p.unlink()

    def test_resume_gate_still_rejects_shapeless_notes(self):
        payloads = [{}, {"visual": {}}, {"visual": {"caption": "   "}, "uncertain": [""]},
                    {"visual": {"actions": [{}]}}, {"uncertain": [None, ""]}]
        for payload in payloads:
            p = self._note_file(payload)
            try:
                self.assertFalse(av_understand_module._is_valid_note(p), payload)
            finally:
                p.unlink()

    # --- OP/ED filtering must be auditable and must feed the echo check ----
    def test_filter_items_records_removed_text_and_reindexes(self):
        windows = [{"start_ms": 84_000, "end_ms": 105_000, "label": "OP"}]
        items = [{"index": 1, "start_ms": 85_000, "end_ms": 88_000, "text": "♪ 主题曲歌词 ♪"},
                 {"index": 2, "start_ms": 106_000, "end_ms": 109_000, "text": "你终于来了"}]
        kept, meta = op_ed.filter_items(items, windows)
        self.assertEqual([it["index"] for it in kept], [1])
        self.assertEqual(meta["removed_total"], 1)
        self.assertEqual(meta["removed_lines"],
                         [{"original_index": 1, "label": "OP", "start_ms": 85_000,
                           "end_ms": 88_000, "text": "♪ 主题曲歌词 ♪"}])

    def test_subtitle_norms_include_op_ed_filtered_lines(self):
        """A hard-subbed ED echoes the very lines the filter removed upstream; the
        echo check has to know about them or it stays blind exactly where it matters."""
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            (ws / ".cache" / "subtitles").mkdir(parents=True)
            (ws / ".cache" / "subtitles" / "extracted.json").write_text(json.dumps({
                "subtitle_count": 1,
                "items": [{"index": 1, "start_ms": 200_000, "end_ms": 203_000, "text": "你终于来了"}],
                "op_ed_filtered": {"removed_total": 1, "removed": {"ED": 1}, "removed_lines": [
                    {"original_index": 9, "label": "ED", "start_ms": 1_330_000,
                     "end_ms": 1_334_000, "text": "永远不会再醒来"}]},
            }, ensure_ascii=False), encoding="utf-8")
            norms = av_understand_module._load_subtitle_norms(str(ws))
        norm = av_understand_module._norm_text
        self.assertIn(norm("你终于来了"), norms)
        self.assertIn(norm("永远不会再醒来"), norms)

    # --- OP/ED scene vs the splice coverage contract -----------------------
    @staticmethod
    def _op_ed_ws(ws: Path, dialogues, schema="vts-av-notes/v2"):
        visual = ws / ".cache" / "visual"
        visual.mkdir(parents=True)
        (ws / ".cache" / "alignment").mkdir(parents=True)
        (visual / "scenes.json").write_text(json.dumps({"scenes": [{
            "scene_id": "SCENE_01", "macro_index": 1, "start_ms": 84_000, "end_ms": 106_500,
            "start_timecode": "00:01:24.000", "end_timecode": "00:01:46.500",
            "duration_ms": 22_500, "slugline": "OP", "child_shot_count": 2,
        }]}, ensure_ascii=False), encoding="utf-8")
        (visual / "shots.json").write_text(json.dumps({"scenes": []}), encoding="utf-8")
        (ws / ".cache" / "alignment" / "aligned_timeline.json").write_text(json.dumps({
            "shots": [{"shot_id": "SCENE_01", "dialogues": dialogues}]}), encoding="utf-8")
        (visual / "av_notes.json").write_text(json.dumps(
            {"schema": schema, "scene_notes": []}), encoding="utf-8")
        (ws / "materials").mkdir()
        (ws / "materials" / "bible.json").write_text(json.dumps({
            "characters": [{"name": "菈菈"}],
            "op_ed_windows": [{"start_ms": 84_000, "end_ms": 105_000, "label": "OP"}],
        }, ensure_ascii=False), encoding="utf-8")

    def test_op_ed_scene_with_surviving_line_is_not_stubbed(self):
        """The scene is >50% inside the OP window, but one line straddles the edge and
        survives. Stubbing it used to deadlock splice on a placeholder nobody was asked
        to weave (the writer only handles draft_status == "missing")."""
        from build_scene_manifest import build_manifest
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            self._op_ed_ws(ws, [{"sub_index": 1, "text": "喂，等等我", "start_ms": 104_900,
                                 "end_ms": 105_400}])
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                manifest = build_manifest(ws, max_keyframes=4)
            scene = manifest["scenes"][0]
            self.assertEqual(scene["draft_status"], "missing")
            self.assertEqual(scene["op_ed"], "OP")            # annotation survives
            self.assertEqual(len(scene["dialogues"]), 1)
            self.assertFalse((ws / ".cache" / "scene_drafts" / "scene_01.md").exists())
            self.assertIn("[WARN]", err.getvalue())
            self.assertIn("surviving dialogue", err.getvalue())

    def test_op_ed_scene_without_lines_still_gets_a_stub(self):
        from build_scene_manifest import build_manifest
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            self._op_ed_ws(ws, [])
            with contextlib.redirect_stderr(io.StringIO()):
                manifest = build_manifest(ws, max_keyframes=4)
            scene = manifest["scenes"][0]
            self.assertEqual(scene["draft_status"], "op_ed")
            self.assertEqual(scene["keyframes_thumbs"], [])
            stub = (ws / ".cache" / "scene_drafts" / "scene_01.md").read_text(encoding="utf-8")
            self.assertIn("（动画 OP——按配置略）", stub)
            self.assertNotIn("[[SUB:", stub)

    def test_stale_av_notes_schema_is_reported(self):
        import build_scene_manifest
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            self._op_ed_ws(ws, [], schema="vts-av-notes/v1")
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                build_scene_manifest.build_manifest(ws, max_keyframes=4)
            self.assertIn("vts-av-notes/v1", err.getvalue())
            self.assertIn(build_scene_manifest.EXPECTED_AV_NOTES_SCHEMA, err.getvalue())

    def test_manifest_schema_constant_matches_the_producer(self):
        import build_scene_manifest
        self.assertEqual(build_scene_manifest.EXPECTED_AV_NOTES_SCHEMA,
                         av_understand_module.AV_NOTES_SCHEMA)


if __name__ == "__main__":
    unittest.main()
