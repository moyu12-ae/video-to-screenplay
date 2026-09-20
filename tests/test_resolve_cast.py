#!/usr/bin/env python3
"""
tests/test_resolve_cast.py - v0.6 P2: the identity gate.

Each test names the accident it exists to prevent. The recurring one in this
project's history is a missing piece of evidence being read as a negative:
「没拍到」 and 「不是他」 are different statements, and a resolver that conflates
them will quietly rule out the right character. The other one is a name appearing
with no human behind it - the exact failure that started the cast-table design.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import resolve_cast  # noqa: E402
import series  # noqa: E402
from av_understand import AV_NOTES_SCHEMA as AV_NOTES_SCHEMA_FIX  # noqa: E402

APPROVED = {
    "schema": "vts-cast/v1",
    "series": "再见菈菈",
    "version": "4",
    "entities": [
        {"id": "C1", "canonical_name": "茉里", "status": "approved", "aliases": ["マリー"],
         "voice_profile": {"gender": "female", "age_band": "teen", "timbre": "bright"},
         "visual_label": "黑发齐刘海少女",
         "visual_anchors": [{"desc": "黑发齐刘海", "first_seen_ms": 340000}],
         "approved_by": "human", "approved_at": "2026-09-19"},
        {"id": "C2", "canonical_name": "托德", "status": "approved", "aliases": [],
         "voice_profile": {"gender": "male", "age_band": "young_adult", "timbre": "low"},
         "visual_label": "金发青年",
         "visual_anchors": [{"desc": "金发、青绿外套", "first_seen_ms": 361000}],
         "approved_by": "human", "approved_at": "2026-09-19"},
    ],
}


def _evidence(clusters, terms=None, anchors=None, approved=None):
    return {
        "approved": APPROVED if approved is None else approved,
        "acoustic": clusters,
        "address": {"events": [], "terms": terms or {}, "not_speaker": {}},
        "visual": {"clips_total": 6, "clips_visual_usable": 4,
                   "clips_positive": sum((anchors or {}).values()),
                   "anchors": {"positive_by_anchor": anchors or {}, "states_seen": {}},
                   "available": bool(anchors)},
    }


CLUSTER_FEMALE_TEEN = {"gender": "female", "age_band": "teen", "timbre": "bright",
                       "speech_ms": 9000, "line_count": 4, "named": False}
CLUSTER_MALE_YOUNG = {"gender": "male", "age_band": "young_adult", "timbre": "low",
                      "speech_ms": 11000, "line_count": 6, "named": False}
CLUSTER_UNKNOWN = {"gender": "unknown", "age_band": "unknown", "timbre": "unknown",
                   "speech_ms": 1200, "line_count": 1, "named": False}


class TestMatchingPriority(unittest.TestCase):
    def test_single_family_never_merges_even_when_it_fits(self):
        """Acoustics alone match 茉里 exactly. §4.1 forbids the merge anyway:
        two characters of the same gender and age band is normal in a cast, and a
        wrong merge is inherited by every later episode while an extra question
        is only a cost."""
        doc = resolve_cast.resolve(Path("/tmp"), _evidence({"SPEAKER_A1": CLUSTER_FEMALE_TEEN}))
        cluster = doc["clusters"][0]
        self.assertEqual(cluster["assignment"]["matched_via"], "new_slot")
        self.assertEqual(cluster["assignment"]["status"], "unknown")
        self.assertIn("single-family auto-merge is forbidden", cluster["assignment"]["basis"][0])
        self.assertEqual(len(doc["pending"]), 1)

    def test_two_agreeing_families_inherit_the_approved_identity(self):
        ev = _evidence({"SPEAKER_A1": CLUSTER_FEMALE_TEEN}, anchors={"黑发齐刘海少女": 3})
        doc = resolve_cast.resolve(Path("/tmp"), ev)
        assignment = doc["clusters"][0]["assignment"]
        self.assertEqual(assignment["slot_id"], "S1")
        self.assertEqual(assignment["matched_via"], "approved_slot")
        self.assertEqual(assignment["status"], "approved")
        self.assertEqual(sorted(assignment["families_supporting"]), ["acoustic", "visual"])
        self.assertEqual(doc["pending"], [], "a reused identity must not interrupt the user")

    def test_ambiguity_creates_a_new_slot_instead_of_guessing(self):
        """Two approved profiles fit equally (same gender/age/timbre). The winner
        has no margin, so the cluster keeps its own pending slot."""
        twin = json.loads(json.dumps(APPROVED))
        twin["entities"].append({"id": "C3", "canonical_name": "茉里子", "status": "approved",
                                 "aliases": [],
                                 "voice_profile": {"gender": "female", "age_band": "teen",
                                                   "timbre": "bright"},
                                 "visual_label": "黑发双马尾",
                                 "visual_anchors": [{"desc": "黑发双马尾"}],
                                 "approved_by": "human", "approved_at": "2026-09-19"})
        ev = _evidence({"SPEAKER_A1": CLUSTER_FEMALE_TEEN},
                       anchors={"黑发齐刘海少女": 3, "黑发双马尾": 3}, approved=twin)
        doc = resolve_cast.resolve(Path("/tmp"), ev)
        assignment = doc["clusters"][0]["assignment"]
        self.assertEqual(assignment["matched_via"], "new_slot")
        self.assertIn("ambiguous", assignment["basis"][0])

    def test_no_positive_evidence_at_all_stays_unknown(self):
        doc = resolve_cast.resolve(Path("/tmp"), _evidence({"SPEAKER_A1": CLUSTER_UNKNOWN}))
        self.assertEqual(doc["clusters"][0]["assignment"]["status"], "unknown")
        self.assertEqual(doc["clusters"][0]["assignment"]["families_supporting"], [])


class TestMissingDataIsNotNegative(unittest.TestCase):
    def test_unknown_slot_attributes_do_not_rule_the_slot_out(self):
        """A slot whose profile is blank still matches on the one attribute the
        cluster does report - absence of data costs nothing, it only adds nothing."""
        families, reasons = resolve_cast.family_support(
            {"gender": "female"}, {"profile": {"gender": "female", "age_band": "unknown"}},
            {}, "无档案者")
        self.assertEqual(families, ["acoustic"])
        self.assertIn("gender=female", reasons[0])

    def test_uncovered_visual_clips_are_counted_not_scored(self):
        ev = _evidence({"SPEAKER_A1": CLUSTER_FEMALE_TEEN}, terms={"茉里": 9})
        doc = resolve_cast.resolve(Path("/tmp"), ev)
        slot = next(s for s in doc["slots"] if s["slot_id"] == "S1")
        self.assertEqual(slot["coverage"], {"clips_total": 6, "clips_visual_usable": 4,
                                           "clips_positive": 0})
        visual = doc["evidence_summary"]["visual"]
        self.assertEqual(visual["clips_total"], 6,
                         "uncovered clips must exist as a number, not as votes against")


class TestNamingIsNeverProduced(unittest.TestCase):
    def test_new_slots_never_carry_an_entity(self):
        doc = resolve_cast.resolve(Path("/tmp"), _evidence({"SPEAKER_A1": CLUSTER_UNKNOWN}))
        created = [s for s in doc["slots"] if s["origin"] == "this_episode"]
        self.assertTrue(created)
        for slot in created:
            self.assertIsNone(slot["entity_id"])
            self.assertNotEqual(slot["status"], "approved")

    def test_resolver_rejects_an_approved_entity_without_a_human_record(self):
        forged = {"entities": [{"id": "C1", "canonical_name": "茉里", "status": "approved",
                                "voice_profile": {}, "visual_anchors": []}]}
        with self.assertRaises(AssertionError) as ctx:
            resolve_cast.resolve(Path("/tmp"), _evidence({}, approved=forged))
        self.assertIn("human sign-off record", str(ctx.exception))

    def test_resolver_rejects_a_named_slot_it_created(self):
        doc = resolve_cast.resolve(Path("/tmp"), _evidence({"SPEAKER_A1": CLUSTER_FEMALE_TEEN}))
        created = next(s for s in doc["slots"] if s["origin"] == "this_episode")
        created["entity_id"] = "C1"  # simulate a resolver that learned to name
        with self.assertRaises(AssertionError) as ctx:
            resolve_cast._assert_no_forged_names(doc)
        self.assertIn("must never name a slot", str(ctx.exception))

    def test_no_address_term_is_ever_used_to_attribute_a_line(self):
        """An address term appears in the document as a naming candidate for the
        human table and as an exclusion - never as support for who spoke."""
        ev = _evidence({"SPEAKER_A1": CLUSTER_FEMALE_TEEN}, terms={"茉里": 2}, anchors={"黑发齐刘海少女": 3})
        ev["address"]["not_speaker"] = {"SPEAKER_A1": ["茉里"]}
        doc = resolve_cast.resolve(Path("/tmp"), ev)
        dumped = json.dumps(doc, ensure_ascii=False)
        self.assertNotIn("attribution", dumped)
        self.assertEqual(doc["clusters"][0]["assignment"]["matched_via"], "new_slot",
                         "ruled out by the vocative, so it cannot inherit 茉里's identity")
        self.assertNotIn("text_subtitle", doc["clusters"][0]["assignment"]["families_supporting"])


class TestAddressNegativeEvidence(unittest.TestCase):
    def test_speaker_of_an_addressing_line_is_who_the_cluster_is_not(self):
        """ep02 scene 1 verbatim shape: 「茉里 你交朋友了」 was written as 茉里
        speaking. The line belongs to A1, so A1 is NOT 茉里."""
        ev = _evidence({"SPEAKER_A1": CLUSTER_FEMALE_TEEN})
        ev["address"]["not_speaker"] = {"SPEAKER_A1": ["茉里"]}
        doc = resolve_cast.resolve(Path("/tmp"), ev)
        self.assertEqual(doc["clusters"][0]["not_speaker"], ["茉里"])


class TestSignOffClosesTheLoop(unittest.TestCase):
    """cast_signoff records {kind: signoff, cluster_id} with approved_by: human.
    Honouring that record is the only legal route by which a slot created in this
    episode can carry a name - without it the sign-off would evaporate on the
    next resolve, and the writer would be back to guessing."""

    SIGNED = {"entities": [{
        "id": "C7", "canonical_name": "托德", "status": "approved", "aliases": [],
        "voice_profile": {}, "visual_anchors": [], "approved_by": "human",
        "approved_at": "2026-09-20",
        "evidence": [{"kind": "signoff", "cluster_id": "SPEAKER_A1", "slot_id": "S3"}]}]}

    def test_a_signed_cluster_becomes_named_and_stops_being_pending(self):
        ev = _evidence({"SPEAKER_A1": CLUSTER_FEMALE_TEEN,
                        "SPEAKER_A2": CLUSTER_UNKNOWN}, approved=self.SIGNED)
        doc = resolve_cast.resolve(Path("/tmp"), ev)
        by_cluster = {c["cluster_id"]: c["assignment"] for c in doc["clusters"]}
        self.assertEqual(by_cluster["SPEAKER_A1"]["status"], "approved")
        self.assertEqual(by_cluster["SPEAKER_A1"]["matched_via"], "human_signoff")
        self.assertEqual(by_cluster["SPEAKER_A1"]["entity_name"], "托德")
        self.assertEqual(by_cluster["SPEAKER_A2"]["status"], "unknown")
        self.assertEqual([p["cluster_id"] for p in doc["pending"]], ["SPEAKER_A2"],
                         "the unsigned cluster stays a question")

    def test_a_name_with_no_signoff_record_is_still_rejected(self):
        doc = resolve_cast.resolve(Path("/tmp"), _evidence({"SPEAKER_A1": CLUSTER_UNKNOWN}))
        created = next(s for s in doc["slots"] if s["origin"] == "this_episode")
        created["entity_id"] = "C7"
        with self.assertRaises(AssertionError):
            resolve_cast._assert_no_forged_names(doc)


class TestArtifactAndGuards(unittest.TestCase):
    def _ws(self, root: Path):
        ws = root / "episodes" / "ep02"
        (ws / ".cache" / "audio").mkdir(parents=True)
        (ws / ".cache" / "visual").mkdir(parents=True)
        (ws / ".cache" / "alignment").mkdir(parents=True)
        (ws / "materials").mkdir(parents=True)
        (root / "cast.approved.json").write_text(json.dumps(APPROVED, ensure_ascii=False),
                                                 encoding="utf-8")
        (ws / ".cache" / "audio" / "speakers.json").write_text(json.dumps({
            "schema": "vts-speakers/v2",
            "clusters": [{"cluster_id": "SPEAKER_A1", "name": None, "line_count": 4,
                          "speech_ms": 9000,
                          "acoustic": {"gender": "female", "age_band": "teen",
                                       "timbre": "bright"}}]}, ensure_ascii=False),
            encoding="utf-8")
        (ws / ".cache" / "visual" / "av_notes.json").write_text(json.dumps(
            {"schema": AV_NOTES_SCHEMA_FIX, "scene_notes": []}), encoding="utf-8")
        (ws / ".cache" / "alignment" / "aligned_timeline.json").write_text(json.dumps(
            {"shots": [{"shot_id": "SCENE_01", "dialogues": [
                {"sub_index": 1, "text": "茉里 你交朋友了", "speaker": "SPEAKER_A1"}]}]},
            ensure_ascii=False), encoding="utf-8")
        series.bind_series(ws, root)
        return ws

    def test_cli_writes_cast_json_and_leaves_the_series_table_untouched(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ws = self._ws(root)
            before = (root / "cast.approved.json").read_bytes()
            res = subprocess.run(
                [sys.executable, str(SCRIPTS_DIR / "resolve_cast.py"), "-w", str(ws)],
                capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, res.stderr)
            summary = json.loads(res.stdout)
            self.assertEqual(summary["clusters"], 1)
            self.assertEqual(summary["pending_slots"], 1,
                             "one family is not enough to reuse an identity")
            self.assertIn("待人工签核", res.stderr)
            doc = json.loads((ws / ".cache" / "cast" / "cast.json").read_text(encoding="utf-8"))
            self.assertEqual(doc["schema"], "vts-cast/v1")
            self.assertEqual((root / "cast.approved.json").read_bytes(), before)
            self.assertFalse(doc["gates"]["thresholds_are_measured"],
                             "the margin in force is a starting point, not a result")

    def test_evidence_pack_is_read_through_the_series_binding(self):
        with tempfile.TemporaryDirectory() as td:
            ws = self._ws(Path(td))
            ev = resolve_cast.build_evidence(ws)
            self.assertEqual(sorted(ev["address"]["terms"]), ["茉里"])
            self.assertIn("茉里", series.approved_names(ws))


if __name__ == "__main__":
    unittest.main()
