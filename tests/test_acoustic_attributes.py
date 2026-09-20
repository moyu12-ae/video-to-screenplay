#!/usr/bin/env python3
"""
tests/test_acoustic_attributes.py - v0.6 P0: the acoustic attribute channel.

The attributes (gender / age_band / timbre) are asked for inside the diarization
call that is already paid for, so the cost is zero and the risk is drift: a model
that answers "a youngish woman" instead of an enum value must not smuggle free
text into the evidence that §4.1 auto-merge will consume. These tests pin the
coercion to "unknown", the vote that excludes abstentions from the denominator,
and the compliance counter that tells us whether the prompt worked at all.
"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import build_scene_manifest
import omni_client
import speaker_diarize
from build_scene_manifest import build_manifest
from speaker_diarize import (
    ACOUSTIC_ATTR_MIN_SHARE,
    SPEAKERS_SCHEMA,
    aggregate_acoustic,
    make_workorder,
    normalize_omni_segments,
)


def _turn(label="Speaker 1", start=1.0, end=2.0, **kw):
    seg = {"speaker": label, "start": start, "end": end, "text": "こんにちは"}
    seg.update(kw)
    return seg


class TestSanitisation(unittest.TestCase):
    def test_valid_enum_values_survive(self):
        data = {"segments": [_turn(gender="female", age_band="teen", timbre="bright")]}
        turns = normalize_omni_segments(data, offset_ms=0)
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0]["gender"], "female")
        self.assertEqual(turns[0]["age_band"], "teen")
        self.assertEqual(turns[0]["timbre"], "bright")

    def test_free_text_and_absence_become_unknown_not_a_guess(self):
        data = {"segments": [
            _turn(gender="a youngish woman", timbre="明亮偏尖"),
            _turn(start=5.0, end=6.0),
        ]}
        turns = normalize_omni_segments(data, offset_ms=0)
        self.assertEqual([t["gender"] for t in turns], ["unknown", "unknown"])
        self.assertEqual([t["timbre"] for t in turns], ["unknown", "unknown"])

    def test_case_spacing_and_hyphens_are_tolerated(self):
        data = {"segments": [_turn(age_band=" Young-Adult ", gender="FEMALE")]}
        turn = normalize_omni_segments(data, offset_ms=0)[0]
        self.assertEqual(turn["age_band"], "young_adult")
        self.assertEqual(turn["gender"], "female")

    def test_every_enum_value_is_accepted(self):
        for field, allowed in speaker_diarize.ACOUSTIC_ENUMS.items():
            for value in sorted(allowed):
                got = speaker_diarize._sanitize_attr(value.upper(), field)
                self.assertEqual(got, value, f"{field}={value} must survive")


class TestAggregation(unittest.TestCase):
    def _turns(self, *overrides):
        base = {"gender": "unknown", "age_band": "unknown", "timbre": "unknown"}
        out = []
        for o in overrides:
            t = dict(base)
            t.update(o)
            out.append(t)
        return out

    def test_majority_wins_and_support_is_reported(self):
        turns = self._turns(*[{"gender": "female"}] * 3 + [{"gender": "male"}])
        agg = aggregate_acoustic(turns)
        self.assertEqual(agg["gender"], "female")
        self.assertEqual(agg["gender_support"], "3/4 turns")

    def test_unknown_votes_leave_the_denominator(self):
        """Missing data is not a vote against the winner - the rule §7 sets for the
        visual family, applied to acoustics."""
        turns = self._turns(*[{"gender": "female"}] * 2, *[{"gender": "unknown"}] * 8)
        agg = aggregate_acoustic(turns)
        self.assertEqual(agg["gender"], "female")
        self.assertEqual(agg["gender_support"], "2/2 turns")

    def test_genuine_mix_abstains(self):
        turns = self._turns({"gender": "female"}, {"gender": "male"},
                            {"gender": "male"}, {"gender": "female"})
        self.assertEqual(aggregate_acoustic(turns)["gender"], "unknown")

    def test_no_observations_is_unknown_not_empty(self):
        agg = aggregate_acoustic([])
        for field in speaker_diarize.ACOUSTIC_ATTRIBUTES:
            self.assertEqual(agg[field], "unknown")
            self.assertEqual(agg[f"{field}_support"], "0/0 turns")

    def test_share_threshold_is_the_named_constant(self):
        known = 10
        winner_votes = int(ACOUSTIC_ATTR_MIN_SHARE * known)
        turns = self._turns(*[{"timbre": "deep"}] * winner_votes,
                            *[{"timbre": "sharp"}] * (known - winner_votes))
        agg = aggregate_acoustic(turns)
        self.assertEqual(agg["timbre"], "deep")
        turns = self._turns(*[{"timbre": "deep"}] * (winner_votes - 1),
                            *[{"timbre": "sharp"}] * (known - winner_votes + 1))
        self.assertEqual(aggregate_acoustic(turns)["timbre"], "unknown")


class TestPromptContract(unittest.TestCase):
    def test_prompt_asks_for_the_three_fields_and_still_formats(self):
        # .format() on the enum braces raised KeyError once the lists were added;
        # a literal brace in this prompt must always be doubled.
        prompt = omni_client.DIARIZE_PROMPT.format(speakers=" Target 2 speakers.", lang="")
        for field, allowed in speaker_diarize.ACOUSTIC_ENUMS.items():
            self.assertIn(field, prompt)
            for value in sorted(allowed):
                self.assertIn(value, prompt)
        self.assertIn("never from what is being said", prompt)

    def test_producer_and_consumer_agree_on_schema(self):
        self.assertEqual(build_scene_manifest.EXPECTED_SPEAKERS_SCHEMA, SPEAKERS_SCHEMA)

    def test_workorder_tells_the_agent_the_expected_shape(self):
        order = make_workorder("ep.mkv", 30_000, [(0, 30_000)],
                              num_speakers=None, language=None, no_audio=False)
        schema = order["expected_segment_schema"]
        self.assertEqual(set(schema) - {"speaker", "start", "end", "text"},
                         set(speaker_diarize.ACOUSTIC_ATTRIBUTES))


class TestArtifactPlumbing(unittest.TestCase):
    def _speakers_doc(self, ws: Path, clusters, schema=SPEAKERS_SCHEMA):
        doc = {
            "schema": schema,
            "clusters": clusters,
            "characters_manifest": {},
            "speech_turns": [],
            "segments": [],
        }
        (ws / ".cache" / "audio").mkdir(parents=True, exist_ok=True)
        (ws / ".cache" / "audio" / "speakers.json").write_text(
            json.dumps(doc, ensure_ascii=False), encoding="utf-8")

    def _base_ws(self, ws: Path):
        visual = ws / ".cache" / "visual"
        visual.mkdir(parents=True, exist_ok=True)
        (ws / ".cache" / "alignment").mkdir(parents=True, exist_ok=True)
        (visual / "scenes.json").write_text(json.dumps({"scenes": [{
            "scene_id": "SCENE_01", "macro_index": 1, "start_ms": 0, "end_ms": 30_000,
            "start_timecode": "00:00:00.000", "end_timecode": "00:00:30.000",
            "duration_ms": 30_000, "slugline": "咖啡馆", "child_shot_count": 1,
        }]}, ensure_ascii=False), encoding="utf-8")
        (visual / "shots.json").write_text(json.dumps({"scenes": []}), encoding="utf-8")
        (ws / ".cache" / "alignment" / "aligned_timeline.json").write_text(
            json.dumps({"shots": []}), encoding="utf-8")
        (ws / "materials").mkdir(parents=True, exist_ok=True)
        (ws / "materials" / "bible.json").write_text(
            json.dumps({"characters": []}, ensure_ascii=False), encoding="utf-8")

    def test_profiles_reach_the_evidence_pack(self):
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            self._base_ws(ws)
            self._speakers_doc(ws, [{"cluster_id": "SPEAKER_A1", "name": None,
                                     "line_count": 4, "speech_ms": 9_000,
                                     "acoustic": {"gender": "male", "age_band": "young_adult",
                                                  "timbre": "bright",
                                                  "gender_support": "3/4 turns",
                                                  "age_band_support": "4/4 turns"}}])
            with contextlib.redirect_stderr(io.StringIO()):
                manifest = build_manifest(ws, max_keyframes=2)
            profile = manifest["speaker_profiles"][0]
            self.assertEqual(profile["gender"], "male")
            self.assertEqual(profile["age_band"], "young_adult")
            self.assertEqual(profile["support"], {"gender_support": "3/4 turns",
                                                 "age_band_support": "4/4 turns"})
            contract = manifest["instructions"]
            self.assertIn("speaker_profiles", contract)

    def test_stale_speakers_schema_is_reported_once_per_build(self):
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            self._base_ws(ws)
            self._speakers_doc(ws, [{"cluster_id": "SPEAKER_A1", "name": None,
                                     "line_count": 1, "speech_ms": 500}],
                               schema="vts-speakers/v1")
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                build_manifest(ws, max_keyframes=2)
            text = err.getvalue()
            self.assertIn("vts-speakers/v1", text)
            self.assertIn(SPEAKERS_SCHEMA, text)
            self.assertEqual(text.count("speakers.json schema"), 1)

    def test_current_schema_builds_without_the_drift_warning(self):
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            self._base_ws(ws)
            self._speakers_doc(ws, [{"cluster_id": "SPEAKER_A1", "name": None,
                                     "line_count": 1, "speech_ms": 500, "acoustic": {}}])
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                manifest = build_manifest(ws, max_keyframes=2)
            self.assertNotIn("speakers.json schema", err.getvalue())
            # Missing acoustic keys still surface as unknown, never as a guess.
            self.assertEqual(manifest["speaker_profiles"][0]["gender"], "unknown")


if __name__ == "__main__":
    unittest.main()
