#!/usr/bin/env python3
"""
tests/test_cast_signoff.py - v0.6 P5: the sign-off conversation.

The rule this file defends is that a name only ever comes from a human, so every
path where the script could have guessed instead has to route back to the user -
and nothing may be half-written when one of them fires.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import cast_signoff as so  # noqa: E402
import series  # noqa: E402

CAST_DOC = {
    "schema": "vts-cast/v1", "series": "再见菈菈", "version": None, "entities": [],
    "slots": [{"slot_id": f"S{i}", "profile": {"gender": "female", "age_band": "teen",
                                               "timbre": "bright", "visual": f"画像{i}"},
               "status": "pending", "entity_id": None, "origin": "this_episode"}
              for i in range(1, 6)],
    "clusters": [{"cluster_id": f"SPEAKER_A{i}",
                  "assignment": {"slot_id": f"S{i}", "status": "unknown",
                                 "matched_via": "new_slot", "candidates": {},
                                 "margin": 0.0, "families_supporting": [], "basis": []}}
                 for i in range(1, 6)],
    "pending": [{"slot_id": f"S{i}", "cluster_id": f"SPEAKER_A{i}", "reason": "新出现"}
                for i in range(1, 6)],
    "evidence_summary": {"address_terms": {"奶奶": 3}},
}

SLOTS = [{"slot_id": "S1", "candidates": ["托德"], "profile": {"gender": "male"}},
         {"slot_id": "S2", "candidates": ["奶奶"], "profile": {"gender": "female"}}]


class TestRounds(unittest.TestCase):
    def test_five_pending_slots_need_two_rounds_not_a_strand(self):
        """The pre-v0.6 rule was "ask at most 4, leave the rest unknown" - which
        means a first episode ships half-unnamed. Rounds instead of a cap."""
        first, window, remaining = so.render_prompt(CAST_DOC, offset=0)
        self.assertEqual(len(window), so.SLOTS_PER_ROUND)
        self.assertEqual(remaining, 1)
        self.assertIn("后面还有 1 个", first)
        second, window2, remaining2 = so.render_prompt(CAST_DOC, offset=so.SLOTS_PER_ROUND)
        self.assertEqual([s["slot_id"] for s in window2], ["S5"])
        self.assertEqual(remaining2, 0)
        self.assertNotIn("后面还有", second)

    def test_table_shows_profile_and_candidates_not_a_blank_question(self):
        table = so.render_table(SLOTS)
        self.assertIn("| S1 | male | 托德 |", table)
        self.assertIn("（无候选，建议保留描述性标签）",
                      so.render_table([{"slot_id": "S9", "candidates": [], "profile": {}}]))


class TestReplyPaths(unittest.TestCase):
    def test_five_distinct_outcomes(self):
        accepted = so.parse_reply("S1=托德 S2=跳过", SLOTS, [])
        self.assertEqual(accepted["decisions"], {"S1": "accept", "S1:name": "托德",
                                                "S2": "skip"})
        self.assertEqual((accepted["needs_confirmation"], accepted["unparsed"]), ([], []))

        novel = so.parse_reply("S1=艾拉", SLOTS, [])
        self.assertEqual(novel["decisions"], {})
        self.assertEqual(novel["needs_confirmation"][0]["slot_id"], "S1")
        self.assertIn("不在 S1 的候选里", novel["needs_confirmation"][0]["why"])

        ambiguous = so.parse_reply("第二个叫托德", SLOTS, [])
        self.assertEqual(ambiguous["needs_confirmation"][0]["slot_id"], "S2")
        self.assertIn("请确认槽位号", ambiguous["needs_confirmation"][0]["why"])

        unparsable = so.parse_reply("今天天气不错", SLOTS, [])
        self.assertEqual(unparsable["unparsed"], ["今天天气不错"])
        self.assertTrue(unparsable["template"], "must hand back an editable template")

        affirmation = so.parse_reply("S1 对", SLOTS, [])
        self.assertEqual(affirmation["decisions"], {"S1": "skip"},
                         "an affirmation keeps the descriptive label; it is not a name")

    def test_bare_interjections_never_become_cast_members(self):
        for reply, slot in [("S1 嗯", "S1"), ("S1 哦", "S1"), ("S1=好的", "S1"),
                            ("S2 没问题", "S2")]:
            outcome = so.parse_reply(reply, SLOTS, [])
            self.assertEqual(outcome["needs_confirmation"], [], reply)
            self.assertEqual(outcome["decisions"].get(slot), "skip", reply)

    def test_a_candidate_that_is_a_real_short_name_beats_the_filler_list(self):
        slots = [{"slot_id": "S1", "candidates": ["好"], "profile": {}}]
        self.assertEqual(so.parse_reply("S1=好", slots, [])["decisions"],
                         {"S1": "accept", "S1:name": "好"})

    def test_unknown_slot_reference_is_reported_once(self):
        outcome = so.parse_reply("S9=甲", SLOTS, [])
        self.assertEqual(len(outcome["unparsed"]), 1)

    def test_empty_reply_asks_again_instead_of_skipping_everything(self):
        outcome = so.parse_reply("", SLOTS, [])
        self.assertEqual(outcome["decisions"], {})
        self.assertIn("S1=", outcome["template"])


class TestApply(unittest.TestCase):
    def test_accepted_names_land_with_a_human_record_and_history(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            result = so.apply_round(root, CAST_DOC, SLOTS,
                                    {"S1": "accept", "S1:name": "托德",
                                     "S2": "accept", "S2:name": "奶奶"})
            doc = json.loads((root / "cast.approved.json").read_text(encoding="utf-8"))
            self.assertEqual(doc["version"], "1")
            names = {e["canonical_name"]: e for e in doc["entities"]}
            self.assertEqual(set(names), {"托德", "奶奶"})
            for entity in doc["entities"]:
                self.assertEqual(entity["approved_by"], "human")
                self.assertTrue(entity["approved_at"])
                self.assertEqual(entity["status"], "approved")
            self.assertEqual(len(doc["history"]), 1)
            self.assertEqual(result["approved_entity_count"], 2)

    def test_second_round_appends_a_version_and_keeps_the_first(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            so.apply_round(root, CAST_DOC, SLOTS, {"S1": "accept", "S1:name": "托德"})
            later = [{"slot_id": "S5", "candidates": [], "profile": {"gender": "unknown"}}]
            so.apply_round(root, CAST_DOC, later, {"S5": "accept", "S5:name": "系统音"})
            doc = json.loads((root / "cast.approved.json").read_text(encoding="utf-8"))
            self.assertEqual(doc["version"], "2")
            self.assertEqual(len(doc["history"]), 2)
            self.assertIn("托德", {e["canonical_name"] for e in doc["entities"]})
            self.assertIn("系统音", {e["canonical_name"] for e in doc["entities"]})

    def test_skips_write_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            result = so.apply_round(root, CAST_DOC, SLOTS, {"S1": "skip", "S2": "skip"})
            doc = json.loads((root / "cast.approved.json").read_text(encoding="utf-8"))
            self.assertEqual(doc["entities"], [])
            self.assertEqual(result["diff"], [])


class TestDraftBanner(unittest.TestCase):
    def test_unbound_is_loud(self):
        self.assertIn("未绑定", so.cmd_draft_banner(None, enforced=False))

    def test_pending_count_is_named(self):
        self.assertIn("5 个槽位为候选", so.cmd_draft_banner(CAST_DOC, enforced=True))

    def test_fully_signed_off_produces_no_notice(self):
        signed = dict(CAST_DOC, pending=[])
        self.assertEqual(so.cmd_draft_banner(signed, enforced=True), "")


class TestCLI(unittest.TestCase):
    def _run(self, *args):
        return subprocess.run([sys.executable, str(SCRIPTS_DIR / "cast_signoff.py"), *args],
                              capture_output=True, text=True)

    def test_refuses_an_unbound_workspace_naming_the_fix(self):
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td) / "ep02"
            (ws / ".cache" / "cast").mkdir(parents=True)
            (ws / ".cache" / "cast" / "cast.json").write_text(
                json.dumps(CAST_DOC, ensure_ascii=False), encoding="utf-8")
            res = self._run("-w", str(ws), "--render")
            self.assertEqual(res.returncode, so.EXIT_NOT_BOUND)
            self.assertIn("workspace.py series --bind", res.stderr)

    def test_render_then_apply_round_trips_through_the_series_table(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "series"
            ws = root / "episodes" / "ep02"
            (ws / ".cache" / "cast").mkdir(parents=True)
            (ws / "materials").mkdir(parents=True)
            (root / "cast.approved.json").write_text(
                json.dumps({"schema": "vts-cast/v1", "version": "0", "entities": []},
                           ensure_ascii=False), encoding="utf-8")
            series.bind_series(ws, root)
            (ws / ".cache" / "cast" / "cast.json").write_text(
                json.dumps(CAST_DOC, ensure_ascii=False), encoding="utf-8")

            rendered = self._run("-w", str(ws), "--render")
            self.assertEqual(rendered.returncode, 0, rendered.stderr)
            self.assertIn("| S1 |", rendered.stdout)

            blocked = self._run("-w", str(ws), "--reply", "S1=艾拉", "--apply")
            self.assertEqual(blocked.returncode, 0)
            self.assertIn("本轮未写盘", blocked.stderr)
            self.assertEqual(series.load_approved(ws)["entities"], [],
                             "an unconfirmed name must not reach the table")

            # 奶奶 is this round's only candidate because it arrives from the address
            # terms - the naming side of the vocative rule, exercised end to end.
            applied = self._run("-w", str(ws), "--reply", "S1=奶奶 S2=跳过", "--apply")
            self.assertIn("signed_off", applied.stdout, applied.stdout)
            table = series.load_approved(ws)
            self.assertEqual([e["canonical_name"] for e in table["entities"]], ["奶奶"])
            self.assertEqual(table["version"], "1")


if __name__ == "__main__":
    unittest.main()
