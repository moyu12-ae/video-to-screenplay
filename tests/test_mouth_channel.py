#!/usr/bin/env python3
"""
tests/test_mouth_channel.py - v0.6 P3: the mouth-state channel and its gate.

Per-action mouth_state is a prompt-compliance bet, so the artifact carries its own
measurement and the resolver refuses to vote on the channel until that measurement
passes. The accident to prevent is subtle: an unanswered field reads as "nobody was
talking", which is a claim about the world rather than about our data.
"""

import sys
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import av_understand as av  # noqa: E402
import resolve_cast  # noqa: E402


def _note(actions, compliance=None):
    return {"schema": av.AV_NOTES_SCHEMA,
            "mouth_compliance": compliance if compliance is not None
            else av.mouth_compliance([{"segments": [{"visual": {"actions": actions}}]}]),
            "scene_notes": [{"scene_id": "SCENE_01",
                             "segments": [{"visual": {"actions": actions}}]}]}


class TestParsing(unittest.TestCase):
    def test_mouth_state_survives_when_it_is_legal(self):
        note = av.parse_note({"visual": {"actions": [
            {"start": 1, "end": 2, "who": "金发青年", "what": "说话", "mouth_state": "MOVING "}]}},
            420_000, 500_000)
        self.assertEqual(note["visual"]["actions"][0]["mouth_state"], "moving")

    def test_illegal_or_absent_state_is_unknown_and_counted(self):
        note = av.parse_note({"visual": {"actions": [
            {"start": 1, "end": 2, "who": "金发青年", "what": "说话", "mouth_state": "probably"},
            {"start": 3, "end": 4, "who": "金发青年", "what": "转身"}]}}, 420_000, 500_000)
        states = [a["mouth_state"] for a in note["visual"]["actions"]]
        self.assertEqual(states, ["unknown", "unknown"])
        self.assertEqual(note["dropped"]["mouth_state_invalid"], 2)

    def test_prompt_asks_for_the_enum_and_the_window_cap(self):
        prompt = av.build_prompt(0, 30_000)
        self.assertIn("mouth_state", prompt)
        for state in av.MOUTH_STATES:
            self.assertIn(state, prompt)
        self.assertIn("Never infer mouth_state from speech timing", prompt)
        self.assertIn("at most 3 seconds", prompt)


class TestComplianceGate(unittest.TestCase):
    def test_clean_channel_is_usable(self):
        actions = [{"start": i * 1000, "end": i * 1000 + 1500, "who": "金发青年",
                    "what": "说话", "mouth_state": "moving"} for i in range(10)]
        stats = av.mouth_compliance([{"segments": [{"visual": {"actions": actions}}]}])
        self.assertEqual(stats["valid_rate"], 1.0)
        self.assertEqual(stats["window_compliance"], 1.0)
        self.assertTrue(stats["usable"])

    def test_long_windows_break_the_channel(self):
        """A 9s "action" cannot say who is talking in it, so obeying the enum is
        not enough - the ≤3s rule is part of the same bet."""
        actions = [{"start": 0, "end": 9_000, "who": "金发青年", "what": "说话",
                    "mouth_state": "moving"}] * 10
        stats = av.mouth_compliance([{"segments": [{"visual": {"actions": actions}}]}])
        self.assertEqual(stats["window_compliance"], 0.0)
        self.assertFalse(stats["usable"])

    def test_empty_notes_are_not_a_passing_channel(self):
        stats = av.mouth_compliance([])
        self.assertEqual(stats["actions_total"], 0)
        self.assertFalse(stats["usable"])

    def test_unanswered_field_is_not_read_as_nobody_talking(self):
        ev = resolve_cast.visual_evidence(_note([{"start": 0, "end": 1_500, "who": "金发青年",
                                                  "what": "说话", "mouth_state": "unknown"}]))
        self.assertFalse(ev["available"])
        self.assertEqual(ev["anchors"]["positive_by_anchor"], {})
        self.assertEqual(ev["clips_positive"], 0)
        self.assertIn("contributes no votes", ev["note"])


class TestPerClusterWindowing(unittest.TestCase):
    """The design's own words: to ask whether A1 is the blond young man, evaluate
    the frames where A1 speaks - not the episode. Global aggregation would let a
    person talking during someone else's line vote here, and would make the
    over-split audit uncomputable."""

    SPEAKERS = {"speech_turns": [
        {"cluster_id": "SPEAKER_A1", "start_ms": 0, "end_ms": 2000},
        {"cluster_id": "SPEAKER_A1", "start_ms": 3000, "end_ms": 4000},
        {"cluster_id": "SPEAKER_A2", "start_ms": 6000, "end_ms": 7000}]}

    def test_votes_follow_each_cluster_windows(self):
        actions = [{"start": 1000, "end": 1500, "who": "金发青年", "mouth_state": "moving"},
                   {"start": 6200, "end": 6600, "who": "老年女性", "mouth_state": "moving"},
                   {"start": 3500, "end": 3900, "who": "金发青年", "mouth_state": "still"},
                   {"start": 9000, "end": 9500, "who": "路人", "mouth_state": "moving"}]
        votes = resolve_cast.cluster_visual_votes(self.SPEAKERS, _note(actions))
        self.assertEqual(votes, {"SPEAKER_A1": {"金发青年": 1},
                                 "SPEAKER_A2": {"老年女性": 1}},
                         "still must not vote, and a window outside every turn votes for nobody")

    def test_a_downgraded_channel_still_reports_to_the_human(self):
        """The sheet shows untrusted votes on purpose: a person reading "chewing"
        next to "mouth moving" is how the mastication confounder gets caught."""
        actions = [{"start": 1000, "end": 1500, "who": "黑发少女", "mouth_state": "moving"}]
        doc = resolve_cast.resolve(Path("/tmp"), {
            "approved": {},
            "acoustic": {"SPEAKER_A1": {"gender": "unknown", "age_band": "unknown",
                                        "timbre": "unknown", "speech_ms": 1,
                                        "line_count": 1, "named": False}},
            "address": {"events": [], "terms": {}, "not_speaker": {}},
            "visual": dict(resolve_cast.visual_evidence(_note(actions, compliance={"usable": False})),
                           by_cluster=resolve_cast.cluster_visual_votes(
                               {"speech_turns": [{"cluster_id": "SPEAKER_A1",
                                                  "start_ms": 0, "end_ms": 2000}]},
                               _note(actions)))})
        entry = doc["pending"][0]
        self.assertEqual(entry["visual_votes"], {"黑发少女": 1})
        self.assertFalse(entry["visual_trusted"])


class TestResolverConsumption(unittest.TestCase):
    ACTIONS = [{"start": 0, "end": 1_500, "who": "金发青年", "what": "说话", "mouth_state": "moving"},
               {"start": 2_000, "end": 3_400, "who": "金发青年", "what": "回头", "mouth_state": "still"},
               {"start": 4_000, "end": 5_200, "who": "黑发少女", "what": "站着", "mouth_state": "not_visible"}]

    def test_only_moving_counts_as_positive(self):
        ev = resolve_cast.visual_evidence(_note(self.ACTIONS))
        self.assertEqual(ev["anchors"]["positive_by_anchor"], {"金发青年": 1})
        self.assertNotIn("黑发少女", ev["anchors"]["positive_by_anchor"])
        self.assertEqual(ev["clips_total"], 1)
        self.assertEqual(ev["clips_visual_usable"], 1)
        self.assertTrue(ev["available"])

    def test_a_live_channel_with_two_families_can_inherit_an_identity(self):
        approved = {"entities": [{
            "id": "C2", "canonical_name": "托德", "status": "approved",
            "voice_profile": {"gender": "male", "age_band": "young_adult", "timbre": "low"},
            "visual_label": "金发青年", "visual_anchors": [{"desc": "金发、青绿外套"}],
            "approved_by": "human", "approved_at": "2026-09-19"}]}
        ev = {"approved": approved,
              "acoustic": {"SPEAKER_A1": {"gender": "male", "age_band": "young_adult",
                                          "timbre": "low", "speech_ms": 8000,
                                          "line_count": 4, "named": False}},
              "address": {"events": [], "terms": {}, "not_speaker": {}},
              "visual": resolve_cast.visual_evidence(_note(self.ACTIONS))}
        doc = resolve_cast.resolve(Path("/tmp"), ev)
        assignment = doc["clusters"][0]["assignment"]
        self.assertEqual(assignment["matched_via"], "approved_slot")
        self.assertEqual(assignment["families_supporting"], ["acoustic", "visual"])
        slot = doc["slots"][0]
        self.assertEqual(slot["visual_label"], "金发青年")
        self.assertEqual(slot["coverage"]["clips_positive"], 1)

    def test_downgraded_channel_cannot_satisfy_the_two_family_rule(self):
        """Same actions, but the compliance block says the channel is not usable:
        the cluster must then keep its own pending slot instead of borrowing one."""
        note = _note(self.ACTIONS, compliance={"usable": False})
        ev = resolve_cast.visual_evidence(note)
        self.assertFalse(ev["available"])
        slot = {"slot_id": "S1", "profile": {"gender": "male"}, "visual_label": "金发青年",
                "status": "approved", "entity_id": "C2"}
        families, reasons = resolve_cast.family_support(
            {"gender": "male"}, slot, {"positive_by_anchor": {"金发青年": 3}}, "托德",
            anchors_available=ev["available"])
        self.assertEqual(families, ["acoustic"], "the visual vote is withheld, not reversed")
        self.assertFalse(any("visual" in r for r in reasons))


if __name__ == "__main__":
    unittest.main()
