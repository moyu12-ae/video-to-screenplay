#!/usr/bin/env python3
"""
tests/test_cast_merge.py - v0.6 P4: merging evidence families without letting
missing data vote.

The formula in the design doc is small but every part of it earns its place:
unequal bases (acoustics sees all 17 windows, the mouth channel 6) must not let
the better-covered family dictate, a family with nothing to say must be dropped
rather than multiplied in as a zero, and a tie must abstain even when the
absolute counts look decisive.
"""

import json
import sys
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import resolve_cast  # noqa: E402
import splice_screenplay as sp  # noqa: E402


class TestNormalisation(unittest.TestCase):
    def test_unobserved_windows_flatten_instead_of_voting_against(self):
        dist = resolve_cast.normalize_distribution({"S1": 3, "S2": 1}, unobserved=11)
        self.assertAlmostEqual(dist["S1"], 3 / 15)
        self.assertAlmostEqual(dist["S2"], 1 / 15)
        self.assertAlmostEqual(sum(dist.values()), 4 / 15,
                               msg="the mass stays below 1 because most windows went unseen")

    def test_a_family_with_no_observations_is_dropped_not_zeroed(self):
        self.assertEqual(resolve_cast.normalize_distribution({"S1": 0, "S2": 0}, 6), {})

    def test_negative_unobserved_cannot_inflate_the_distribution(self):
        a = resolve_cast.normalize_distribution({"S1": 1}, -5)
        self.assertEqual(a, {"S1": 1.0})


class TestMerge(unittest.TestCase):
    def test_confident_zero_from_one_family_eliminates_the_candidate(self):
        """Narrowing the candidate set IS the value of this channel: acoustics
        likes both, the face was only ever seen on one."""
        scores = resolve_cast.merge_distributions(
            {"acoustic": {"S1": 0.30, "S2": 0.10}, "visual": {"S1": 0.20}},
            resolve_cast.FAMILY_WEIGHTS)
        self.assertGreater(scores["S1"], 0)
        self.assertEqual(scores.get("S2", 0.0), 0.0)

    def test_tie_abstains_regardless_of_absolute_counts(self):
        margin, ranked = resolve_cast.distribution_margin({"S1": 0.25, "S2": 0.25})
        self.assertEqual(margin, 0.0)
        self.assertEqual(ranked, ["S1", "S2"])

    def test_margin_is_relative_to_total_mass(self):
        margin, _ = resolve_cast.distribution_margin({"S1": 0.9, "S2": 0.1})
        self.assertAlmostEqual(margin, 0.8)
        self.assertEqual(resolve_cast.distribution_margin({})[0], 0.0)


class TestCandidatesSurviveAbstention(unittest.TestCase):
    def test_abstaining_still_reports_the_narrowed_set(self):
        """'It is one of these two' is not a failure to answer; recording only the
        winner would throw away everything the evidence actually established."""
        approved = {"entities": [
            {"id": f"C{i}", "canonical_name": n, "status": "approved",
             "voice_profile": {"gender": "female", "age_band": "teen", "timbre": "bright"},
             "visual_label": lbl, "approved_by": "human", "approved_at": "2026-09-19"}
            for i, n, lbl in [(1, "甲", "黑发少女"), (2, "乙", "棕发少女")]]}
        ev = {"approved": approved,
              "acoustic": {"SPEAKER_A1": {"gender": "female", "age_band": "teen",
                                          "timbre": "bright", "speech_ms": 4000,
                                          "line_count": 2, "named": False}},
              "address": {"events": [], "terms": {}, "not_speaker": {}},
              "visual": {"clips_total": 4, "clips_visual_usable": 4, "clips_positive": 4,
                         "anchors": {"positive_by_anchor": {"黑发少女": 2, "棕发少女": 2}},
                         "available": True}}
        doc = resolve_cast.resolve(Path("/tmp"), ev)
        assignment = doc["clusters"][0]["assignment"]
        self.assertEqual(assignment["matched_via"], "new_slot")
        self.assertEqual(assignment["status"], "unknown")
        self.assertEqual(set(assignment["candidates"]), {"S1", "S2"})
        self.assertEqual(assignment["margin"], 0.0)
        self.assertEqual(assignment["families_supporting"], ["acoustic", "visual"],
                         "both families voted; they just agreed on nothing")


class TestOverSplitAudit(unittest.TestCase):
    def _two_clusters_one_face(self):
        approved = {"entities": [{
            "id": "C1", "canonical_name": "托德", "status": "approved",
            "voice_profile": {"gender": "male", "age_band": "young_adult", "timbre": "low"},
            "visual_label": "金发青年", "approved_by": "human", "approved_at": "2026-09-19"}]}
        cluster = {"gender": "male", "age_band": "young_adult", "timbre": "low",
                   "speech_ms": 5000, "line_count": 3, "named": False}
        return {"approved": approved,
                "acoustic": {"SPEAKER_A1": dict(cluster), "SPEAKER_A2": dict(cluster)},
                "address": {"events": [], "terms": {}, "not_speaker": {}},
                "visual": {"clips_total": 3, "clips_visual_usable": 3, "clips_positive": 3,
                           "anchors": {"positive_by_anchor": {"金发青年": 3}},
                           "by_cluster": {"SPEAKER_A1": {"金发青年": 2},
                                          "SPEAKER_A2": {"金发青年": 1}},
                           "available": True}}

    def test_two_clusters_sharing_one_face_are_flagged(self):
        doc = resolve_cast.resolve(Path("/tmp"), self._two_clusters_one_face())
        suspects = doc["over_split_suspects"]
        self.assertEqual(len(suspects), 1)
        self.assertEqual(suspects[0]["clusters"], ["SPEAKER_A1", "SPEAKER_A2"])
        self.assertEqual(suspects[0]["visual_label"], "金发青年")

    def test_the_audit_works_before_any_cast_table_exists(self):
        """Measured on ep02 09:18: two female clusters both showed 红发校服少女
        talking during their lines. Keying this audit to signed-off slots made it
        invisible precisely on episode one, where over-splitting is the live risk -
        the signal costs nothing and needs no names."""
        ev = self._two_clusters_one_face()
        ev["approved"] = {"entities": []}
        ev["visual"]["by_cluster"] = {"SPEAKER_A1": {"红发校服少女": 3},
                                      "SPEAKER_A2": {"红发校服少女": 2}}
        doc = resolve_cast.resolve(Path("/tmp"), ev)
        self.assertEqual(doc["over_split_suspects"][0]["clusters"],
                         ["SPEAKER_A1", "SPEAKER_A2"])
        self.assertEqual(doc["over_split_suspects"][0]["visual_label"], "红发校服少女")
        self.assertEqual(doc["abstention"]["rate"], 1.0,
                         "no table means no auto-merge; the audit is separate")

    def test_the_audit_is_one_directional(self):
        """Several people inside one cluster cannot be seen from this evidence, so
        nothing may claim to detect it."""
        self.assertNotIn("under_split", json.dumps(self._two_clusters_one_face()))
        doc = resolve_cast.resolve(Path("/tmp"), self._two_clusters_one_face())
        self.assertNotIn("under_split", doc)


class TestAbstentionAccounting(unittest.TestCase):
    def test_abstention_rate_and_unmeasured_thresholds_travel_together(self):
        doc = resolve_cast.resolve(Path("/tmp"), {
            "approved": {},
            "acoustic": {"SPEAKER_A1": {"gender": "unknown", "age_band": "unknown",
                                        "timbre": "unknown", "speech_ms": 1,
                                        "line_count": 1, "named": False}},
            "address": {"events": [], "terms": {}, "not_speaker": {}},
            "visual": {"clips_total": 0, "clips_visual_usable": 0, "clips_positive": 0,
                       "anchors": {}, "available": False}})
        self.assertEqual(doc["abstention"]["abstained"], 1)
        self.assertEqual(doc["abstention"]["rate"], 1.0)
        self.assertFalse(doc["gates"]["thresholds_are_measured"])


class TestReportIntegration(unittest.TestCase):
    def test_summarize_cast_counts_the_three_statuses(self):
        doc = {"clusters": [
            {"assignment": {"status": "approved"}}, {"assignment": {"status": "candidate"}},
            {"assignment": {"status": "unknown"}}, {"assignment": {"status": "unknown"}}],
            "gates": {"thresholds_are_measured": False},
            "abstention": {"rate": 0.5}, "over_split_suspects": [], "table_version": "4"}
        summary = sp.summarize_cast(doc)
        self.assertEqual((summary["named"], summary["candidate"], summary["unknown"]), (1, 1, 2))
        self.assertEqual(summary["abstention_rate"], 0.5)
        self.assertEqual(summary["cast_version"], "4")
        self.assertFalse(summary["thresholds_are_measured"])


if __name__ == "__main__":
    unittest.main()
