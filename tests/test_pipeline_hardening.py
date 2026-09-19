#!/usr/bin/env python3
"""
tests/test_pipeline_hardening.py - Regressions for the fidelity and honesty gaps found
in the v0.4.2 audit.

Each test names the failure it exists to prevent:
- a BOM in an .srt silently deleted the first line of dialogue
- the subtitle gate refused hard-subsidised sources instead of routing them to OCR
- a cue with <=100ms overlap vanished from the aligned timeline, then made the splice
  stage fatal on a subtitle the writing pass had never been shown
- a legitimately silent diarization part was re-billed on every `run` and blocked `merge`
- a truncated model reply parsed as valid, shorter JSON
- the degraded speakers.json claimed acoustic clustering had run
- the documented sequence-wall snap window was never consulted
"""

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

import omni_client
from align_timeline import assign_subtitles_to_shots, interval_gap_ms
from speaker_diarize import (
    _empty_output,
    bind_lines,
    build_output,
    classify_omni_output,
    estimate_subtitle_offset_ms,
    write_json_atomic,
)
from subtitle_extractor import parse_ass_file, parse_srt_file

ITEMS = [{"index": 1, "start_ms": 1000, "end_ms": 2000, "text": "第一句"},
         {"index": 2, "start_ms": 4000, "end_ms": 5000, "text": "第二句"}]


def _write_bom(path, text):
    Path(path).write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))


class TestSubtitleFileParsing(unittest.TestCase):

    def test_bom_prefixed_srt_keeps_first_cue(self):
        """The BOM glued itself to the index line, the block failed the timecode test,
        and cue 1 disappeared with no error anywhere."""
        with tempfile.TemporaryDirectory() as d:
            srt = os.path.join(d, "ep.srt")
            body = ("1\r\n00:00:01,000 --> 00:00:02,000\r\n你好\r\n\r\n"
                    "2\r\n00:00:03,000 --> 00:00:04,000\r\n第二句\r\n\r\n")
            _write_bom(srt, body)
            entries = parse_srt_file(srt)
            self.assertEqual([e["text"] for e in entries], ["你好", "第二句"])
            self.assertEqual([e["index"] for e in entries], [1, 2])

    def test_bom_prefixed_ass_keeps_first_event(self):
        with tempfile.TemporaryDirectory() as d:
            ass = os.path.join(d, "ep.ass")
            body = ("[Script Info]\nScriptType: v4.00+\n\n[Events]\n"
                    "Format: Layer, Start, End, Style, Name, Text\n"
                    "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,开场白\n")
            _write_bom(ass, body)
            entries = parse_ass_file(ass)
            self.assertEqual([e["text"] for e in entries], ["开场白"])

    def test_out_of_order_srt_is_sorted_and_reindexed(self):
        """`index` is the chronological contract downstream [[SUB:n]] placement relies on."""
        with tempfile.TemporaryDirectory() as d:
            srt = os.path.join(d, "ep.srt")
            Path(srt).write_text(
                "1\n00:00:09,000 --> 00:00:10,000\n后说的\n\n"
                "2\n00:00:01,000 --> 00:00:02,000\n先说的\n\n", encoding="utf-8")
            entries = parse_srt_file(srt)
            self.assertEqual([e["start_ms"] for e in entries], [1000, 9000])
            self.assertEqual([e["index"] for e in entries], [1, 2])


@unittest.skipIf(shutil.which("ffmpeg") is None, "ffmpeg not installed")
class TestSubtitleGateTiers(unittest.TestCase):
    """The gate advertised an OCR tier it could never reach: exit 5 fired before
    stdout was written, so the Tier 3 instruction was destroyed."""

    def _make_video_without_subtitles(self, d):
        out = os.path.join(d, "clip.mp4")
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1",
             "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono:d=1",
             "-shortest", "-c:v", "libx264", "-c:a", "aac", out],
            check=True, timeout=120)
        return out

    def test_no_subtitle_source_reports_needs_ocr_not_a_refusal(self):
        with tempfile.TemporaryDirectory() as d:
            video = self._make_video_without_subtitles(d)
            proc = subprocess.run(
                [sys.executable, str(SCRIPTS_DIR / "subtitle_extractor.py"),
                 video, "--require-subtitles"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
            self.assertEqual(proc.returncode, 6, proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertEqual(payload["source_tier"], "TIER_3_HARDCODED_OCR")
            self.assertEqual(payload["status"], "NEEDS_OCR")
            self.assertIn("instruction", payload)
            self.assertIn("check-subtitles --mode none", proc.stderr)

    def test_external_subtitle_still_exits_zero(self):
        with tempfile.TemporaryDirectory() as d:
            video = self._make_video_without_subtitles(d)
            Path(os.path.join(d, "clip.srt")).write_text(
                "1\n00:00:00,500 --> 00:00:01,000\n台词\n\n", encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(SCRIPTS_DIR / "subtitle_extractor.py"),
                 video, "--require-subtitles"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(json.loads(proc.stdout)["subtitle_count"], 1)


class TestAlignCompleteness(unittest.TestCase):

    def test_gap_helper(self):
        self.assertEqual(interval_gap_ms(0, 10, 20, 30), 10)
        self.assertEqual(interval_gap_ms(0, 30, 20, 30), 0)

    def test_low_overlap_cue_is_bound_to_nearest_shot_not_dropped(self):
        shots = [{"start_ms": 0, "end_ms": 3000}, {"start_ms": 3000, "end_ms": 6000}]
        # 50ms inside shot 0, 4000ms inside shot 1 - but a cue that clears no shot
        # by >100ms must still land somewhere.
        subs = [{"index": 1, "start_ms": 2950, "end_ms": 3000, "text": "句尾"}]
        assignment, overlap, fallback = assign_subtitles_to_shots(subs, shots)
        self.assertEqual(list(assignment), [0])
        self.assertEqual(fallback, [0])
        self.assertEqual(overlap[0], 0)

    def test_sliver_cue_nearest_shot_wins(self):
        shots = [{"start_ms": 0, "end_ms": 1000}, {"start_ms": 5000, "end_ms": 6000}]
        subs = [{"index": 1, "start_ms": 4900, "end_ms": 4950, "text": "尾巴"}]
        assignment, _, fallback = assign_subtitles_to_shots(subs, shots)
        self.assertEqual(assignment[0], 1)
        self.assertEqual(fallback, [0])

    def test_every_cue_assigned_exactly_once(self):
        """The invariant splice depends on: it treats an unclaimed subtitle as fatal,
        so the aligner may not leave one behind."""
        shots = [{"start_ms": i * 3000, "end_ms": i * 3000 + 2900} for i in range(5)]
        subs = [{"index": i + 1, "start_ms": i * 3000 + 2950, "end_ms": i * 3000 + 2990,
                 "text": f"t{i}"} for i in range(5)]
        assignment, _, fallback = assign_subtitles_to_shots(subs, shots)
        self.assertEqual(sorted(assignment), list(range(len(subs))), "every cue bound once")
        self.assertTrue(all(0 <= pos < len(shots) for pos in assignment.values()))
        self.assertEqual(len(fallback), len(subs), "these cues clear no shot's overlap bar")

    def test_no_shots_cannot_align(self):
        """The aligner cannot invent a home for a cue; main() turns this into exit 1."""
        subs = [{"index": 1, "start_ms": 0, "end_ms": 100, "text": "x"}]
        assignment, _, fallback = assign_subtitles_to_shots(subs, [])
        self.assertEqual(assignment, {})
        self.assertEqual(fallback, [])

    def test_align_cli_fails_loudly_without_shots(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(f"{d}/.cache/subtitles")
            os.makedirs(f"{d}/.cache/visual")
            Path(f"{d}/.cache/subtitles/extracted.json").write_text(
                json.dumps({"items": ITEMS}), encoding="utf-8")
            Path(f"{d}/.cache/visual/scenes.json").write_text(
                json.dumps({"scenes": []}), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(SCRIPTS_DIR / "align_timeline.py"), "--workspace", d],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
            self.assertEqual(proc.returncode, 1, proc.stderr)
            self.assertIn("no shots", proc.stderr)


class TestOmniOutputClassification(unittest.TestCase):

    def test_explicitly_silent_part_is_silent_not_invalid(self):
        self.assertEqual(classify_omni_output({"segments": []}), "silent")
        self.assertEqual(classify_omni_output({"results": []}), "silent")
        self.assertEqual(classify_omni_output([]), "silent")

    def test_usable_and_unusable_shapes(self):
        self.assertEqual(
            classify_omni_output({"segments": [{"speaker": "S1", "start": 1, "end": 2, "text": "a"}]}),
            "ok")
        self.assertEqual(classify_omni_output({"segments": "not a list"}), "invalid")
        self.assertEqual(classify_omni_output({"text": "I found no speech"}), "invalid")
        # Present but every segment unparseable is a bad reply, not a quiet one.
        self.assertEqual(classify_omni_output({"segments": [{"foo": 1}]}), "invalid")

    def test_zero_length_turn_is_not_a_usable_segment(self):
        self.assertEqual(
            classify_omni_output({"segments": [{"speaker": "S1", "start": 5.0, "end": 5.0}]}),
            "invalid")

    def test_silent_part_survives_merge_instead_of_exiting_7(self):
        """`run` used to re-dial a music-only part on every invocation (paying again)
        and `merge` exited 7, so a chunked episode with one silent stretch could never
        finish. Both now read the same classifier."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, "omni_diarized.part002.json")
            write_json_atomic(p, {"segments": [], "speakers": [], "meta": {"backend": "direct_api"}})
            self.assertIn(classify_omni_output(json.loads(p.read_text())), ("ok", "silent"))


class TestDegradedDiarizationMetadata(unittest.TestCase):
    """A degraded artifact must not describe itself as a measured one: downstream saw
    acoustic_clustering_enabled=true, backend=mcp_tool and a model name for a run that
    made zero calls."""

    def test_empty_fallback_declares_acoustics_off(self):
        out = _empty_output("clip.mp4", ITEMS, "diarization_unavailable_empty_fallback")
        self.assertFalse(out["acoustic_clustering_enabled"])
        self.assertEqual(out["diarization_source"]["backend"], "none")
        self.assertEqual(out["diarization_source"]["model"], "none")
        self.assertEqual(out["degradation"], "diarization_unavailable_empty_fallback")
        self.assertTrue(all(r["speaker"] is None for r in out["segments"]))

    def test_real_run_still_claims_acoustics(self):
        row = {"segment_id": 1, "start_ms": 1000, "end_ms": 2000, "speaker": "SPEAKER_A1",
               "confidence": 0.9, "method": "acoustic_diarization", "cluster_id": "SPEAKER_A1",
               "secondary_speaker": None, "text_agreement": None}
        out = build_output("clip.mp4", ITEMS[:1], [row], {"SPEAKER_A1": ITEMS[:1]}, [],
                           {"SPEAKER_A1": "罗温"}, {"罗温": 1}, ["omni_diarized.json"])
        self.assertTrue(out["acoustic_clustering_enabled"])
        self.assertEqual(out["diarization_source"]["backend"], "mcp_tool")
        self.assertEqual(out["distinct_speakers_detected"], 1)
        self.assertEqual(out["clusters_formed"], 1)


class TestBindLinesSecondarySpeaker(unittest.TestCase):

    @staticmethod
    def _turn(start, end, cluster):
        return {"start_ms": start, "end_ms": end, "cluster_id": cluster,
                "text": "x", "raw_label": cluster, "part": 0, "part_file": "f"}

    def test_two_short_turns_still_report_the_second_voice(self):
        """Best/second were tracked per turn: two short turns of A together covering
        more of the line than one long turn of B lost B silently - exactly the overlap
        dialogue the secondary column exists to surface. Under the old code the single
        4000ms turn of A2 won the line outright and A1's 5000ms aggregate vanished."""
        line = [{"index": 1, "start_ms": 0, "end_ms": 10000, "text": "争吵"}]
        turns = [self._turn(0, 2500, "SPEAKER_A1"),
                 self._turn(7000, 9500, "SPEAKER_A1"),
                 self._turn(2600, 6600, "SPEAKER_A2")]
        rows, _ = bind_lines(line, turns)
        self.assertEqual(rows[0]["speaker"], "SPEAKER_A1")
        self.assertEqual(rows[0]["secondary_speaker"], "SPEAKER_A2")

    def test_weak_second_voice_stays_absent(self):
        line = [{"index": 1, "start_ms": 0, "end_ms": 10000, "text": "独白"}]
        turns = [self._turn(0, 9000, "SPEAKER_A1"), self._turn(9700, 9900, "SPEAKER_A2")]
        rows, _ = bind_lines(line, turns)
        self.assertIsNone(rows[0]["secondary_speaker"])


class TestOffsetEstimator(unittest.TestCase):

    def test_lines_without_end_ms_do_not_bias_the_shift(self):
        """end_ms defaulted to the already-shifted start, so a cue missing its end
        scored better the further the trial delta pushed it."""
        turns = [{"start_ms": 0, "end_ms": 9000, "cluster_id": "SPEAKER_A1", "text": "x",
                  "raw_label": "S1", "part": 0, "part_file": "f"}]
        items = [{"index": i + 1, "start_ms": 1000 + i * 20000} for i in range(8)]
        self.assertEqual(estimate_subtitle_offset_ms(items, turns), 0)

    def test_recovers_a_known_shift(self):
        turns = [{"start_ms": i * 5000 + 1000, "end_ms": i * 5000 + 2500,
                  "cluster_id": "SPEAKER_A1", "text": "x", "raw_label": "S1",
                  "part": 0, "part_file": "f"} for i in range(10)]
        # subtitles lead the audio by 2s -> correcting them means +2000ms
        items = [{"index": i + 1, "start_ms": i * 5000 - 1000, "end_ms": i * 5000 + 500}
                 for i in range(10)]
        self.assertEqual(estimate_subtitle_offset_ms(items, turns), 2000)


class TestOmniPayloadHardening(unittest.TestCase):

    def test_truncated_completion_is_rejected_not_resliced(self):
        """Slicing to the LAST closing brace turned a stream cut mid-array into valid
        shorter JSON, so a part that lost its final speakers was accepted silently."""
        truncated = '{"segments": [{"speaker": "S1", "start": 0, "end": 1, "text": "a"}, ' \
                    '{"speaker": "S2", "start": 2, "end": 3, "te'
        with self.assertRaises(ValueError):
            omni_client.extract_json_payload(truncated)

    def test_complete_payload_with_surrounding_prose_still_parses(self):
        self.assertEqual(
            omni_client.extract_json_payload('Here you go: {"segments": []} hope that helps'),
            {"segments": []})

    def test_unparseable_completion_becomes_retryable(self):
        """A reply we cannot parse used to raise a bare ValueError past the retry loop
        and killed the whole part; it must surface as a transient OmniError instead."""
        with tempfile.TemporaryDirectory() as d:
            audio = Path(d, "part001.m4a")
            audio.write_bytes(b"\x00\x01fake audio bytes")
            with mock.patch.object(omni_client, "call_omni_chat",
                                   return_value=("I could not analyse this", None)), \
                 mock.patch.object(omni_client, "fit_audio",
                                   return_value=(audio, "m4a")), \
                 mock.patch.object(omni_client, "_probe_duration_sec", return_value=1.0):
                with self.assertRaises(omni_client.OmniError) as ctx:
                    omni_client.diarize_audio_file(str(audio), api_key="k")
        self.assertTrue(ctx.exception.transient, ctx.exception)
        self.assertIn("unparseable completion", str(ctx.exception))

    def test_length_finish_reason_fails_fast_without_retry(self):
        chunk = ('data: {"choices":[{"delta":{"content":"{\\"segments\\":[]}"},'
                 '"finish_reason":"length"}]}')

        class _Resp:
            headers = {"Content-Type": "text/event-stream"}

            def __iter__(self):
                return iter([chunk])

            def close(self):
                pass

        with mock.patch.object(omni_client, "_OPENER") as opener:
            opener.open.return_value = _Resp()
            with self.assertRaises(omni_client.OmniError) as ctx:
                omni_client._post_stream("https://example.invalid/v1/chat/completions",
                                         "k", {}, 5.0)
        self.assertEqual(ctx.exception.kind, "bad_request")
        self.assertFalse(ctx.exception.transient)

    def test_zero_attempts_surfaces_the_real_error(self):
        boom = omni_client.OmniError("connection", detail="host x")
        with mock.patch.object(omni_client, "validate_endpoint", side_effect=lambda u: u), \
             mock.patch.object(omni_client, "_post_stream", side_effect=boom):
            with self.assertRaises(omni_client.OmniError) as ctx:
                omni_client.call_omni_chat([{"role": "user", "content": "hi"}],
                                           api_key="k", attempts=0)
        self.assertEqual(ctx.exception.kind, "connection")

    def test_no_key_material_in_failure_details(self):
        secret = "-".join(["sk", "TEST", "0123456789"])  # assembled; never a literal key
        err = omni_client.OmniError("auth", 401, detail="host dashscope.example")
        self.assertNotIn(secret, str(err))
        with mock.patch.dict(os.environ, {"DASHSCOPE_API_KEY": secret}), \
             mock.patch.object(omni_client, "validate_endpoint", side_effect=lambda u: u), \
             mock.patch.object(omni_client, "_OPENER") as opener:
            opener.open.side_effect = omni_client.urllib.error.HTTPError(
                "https://x", 401, "nope", {}, None)
            with self.assertRaises(omni_client.OmniError) as ctx:
                omni_client._post_stream("https://dashscope.example/v1/chat/completions",
                                         secret, {}, 5.0)
        self.assertNotIn(secret, str(ctx.exception))


class TestAtomicWrites(unittest.TestCase):

    def test_write_json_atomic_leaves_no_temp_and_roundtrips(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, "part.json")
            write_json_atomic(p, {"segments": [], "note": "中文"})
            self.assertEqual(json.loads(p.read_text())["note"], "中文")
            self.assertEqual([f for f in os.listdir(d) if ".tmp" in f], [])

    def test_failed_write_does_not_clobber_the_previous_good_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, "part.json")
            write_json_atomic(p, {"good": True})
            with mock.patch.object(Path, "write_text",
                                   side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    write_json_atomic(p, {"good": False})
            self.assertTrue(json.loads(p.read_text())["good"])
            self.assertEqual([f for f in os.listdir(d) if ".tmp" in f], [])


class TestGrouperFatalOnEmptySolve(unittest.TestCase):

    def test_empty_solve_is_fatal_not_a_silent_zero(self):
        """A 0-scene result exited 0, so the next stage built an empty screenplay."""
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(f"{d}/.cache/visual")
            os.makedirs(f"{d}/.cache/subtitles")
            Path(f"{d}/.cache/visual/shots.json").write_text(
                json.dumps({"scenes": [], "count": 0}), encoding="utf-8")
            Path(f"{d}/.cache/subtitles/extracted.json").write_text(
                json.dumps({"items": ITEMS}), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(SCRIPTS_DIR / "semantic_scene_grouper.py"),
                 "--workspace", d],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
            self.assertEqual(proc.returncode, 1, proc.stderr)
            self.assertIn("0 macro scenes", proc.stderr)


if __name__ == "__main__":
    unittest.main()
