#!/usr/bin/env python3
"""
tests/test_label_taxonomy.py - v0.6 P1: label taxonomy and the vocative edge.

Two paired obligations from the design doc §13:
- the lint must stop punishing descriptive labels while name-shaped fabrication
  still gets caught (positive AND negative examples for every gate), and
- an address term may produce naming candidates and negative evidence, and must
  never be able to attribute a line to the person it names - the exact mistake
  that started this design (ep02 scene 1: 「茉里 你交朋友了」 written as 茉里
  speaking, when the speaker was the blond young man).
"""

import json
import sys
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import speaker_labels as taxonomy  # noqa: E402
import vocatives  # noqa: E402


class TestLabelTaxonomy(unittest.TestCase):
    def test_descriptive_shapes(self):
        for label in ["王子的声音", "威严之声", "系统音", "女声", "男声", "关西腔者",
                      "面试的店主", "陌生旅人", "青年男声", "路人（男）", "旁白", "奶奶",
                      "金发青年"]:
            self.assertFalse(taxonomy.looks_like_name(label), f"{label} is a description")

    def test_name_shapes(self):
        for label in ["茉里", "托德", "完全陌生的名字", "凉音", "SPEAKER_A1"]:
            if label == "SPEAKER_A1":
                self.assertEqual(taxonomy.classify_label(label), taxonomy.LABEL_KIND_PROVISIONAL)
                continue
            self.assertTrue(taxonomy.looks_like_name(label), f"{label} must trace")

    def test_head_parts_are_judged_separately(self):
        verdict = taxonomy.audit_head("菈菈、陌生旅人", {"菈菈"})
        self.assertEqual(verdict["traced"], ["菈菈"])
        self.assertEqual(verdict["descriptive"], ["陌生旅人"])
        self.assertEqual(verdict["untraced"], [])

    def test_trace_prefers_the_longest_known_surface(self):
        found = taxonomy.trace("菈菈与妈妈", {"菈", "菈菈"})
        self.assertEqual(found, "菈菈")
        self.assertIsNone(taxonomy.trace("茉里南", {"茉里"}), "prefix tracing is for >=2 char surfaces")

    def test_provisional_labels_are_their_own_kind(self):
        self.assertEqual(taxonomy.classify_label("SPEAKER_A3"), taxonomy.LABEL_KIND_PROVISIONAL)
        self.assertFalse(taxonomy.looks_like_name("SPEAKER_A3"))


class TestVocatives(unittest.TestCase):
    LINES = [
        {"index": 1, "text": "茉里 你交朋友了", "cluster_id": "SPEAKER_A1"},
        {"index": 2, "text": "啊，我没有啦", "cluster_id": "SPEAKER_A2"},
        {"index": 3, "text": "我知道了，奶奶", "cluster_id": "SPEAKER_A2"},
        {"index": 4, "text": "奶奶做的饭最好吃", "cluster_id": "SPEAKER_A1"},
        {"index": 5, "text": "茉里！", "cluster_id": "SPEAKER_A3"},
        {"index": 6, "text": "托德，过来帮忙", "cluster_id": "SPEAKER_A4"},
    ]
    NAMES = ["茉里", "奶奶", "托德"]

    def _result(self):
        return vocatives.extract_address_terms(self.LINES, self.NAMES)

    def test_address_positions_are_detected(self):
        terms = self._result()["terms"]
        self.assertEqual(terms["茉里"]["count"], 1, "the bare 「茉里！」 call is not an address line")
        self.assertEqual(terms["奶奶"]["count"], 1, "「奶奶做的饭」 is talked about, not addressed")
        self.assertEqual(terms["托德"]["count"], 1)

    def test_speaker_of_the_line_is_the_negative_not_the_positive(self):
        result = self._result()
        neg = vocatives.negative_evidence(result)
        self.assertEqual(neg.get("SPEAKER_A1"), {"茉里"})
        self.assertEqual(neg.get("SPEAKER_A2"), {"奶奶"})
        # The speaker is the one ruled out, not the person named: 「托德，过来帮忙」
        # was said BY A4, so A4 cannot be 托德.
        self.assertEqual(neg.get("SPEAKER_A4"), {"托德"})

    def test_roles_are_closed_and_attribution_is_impossible(self):
        result = self._result()
        dumped = json.dumps(result, ensure_ascii=False)
        self.assertNotIn("attribution", dumped)
        for ev in result["events"]:
            self.assertIn(ev["role"], vocatives.ROLES)

    def test_naming_candidates_are_counts_only(self):
        cands = vocatives.naming_candidates(self._result())
        self.assertEqual(cands["托德"], 1)
        self.assertNotIn("啊", cands)

    def test_no_known_names_yields_no_events_not_a_crash(self):
        result = vocatives.extract_address_terms(self.LINES, [])
        self.assertEqual(result["events"], [])
        self.assertEqual(result["known_names_used"], 0)

    def test_lines_without_speakers_only_produce_naming(self):
        lines = [{"index": 1, "text": "托德，过来帮忙"}]
        result = vocatives.extract_address_terms(lines, self.NAMES)
        self.assertEqual([e["role"] for e in result["events"]], ["naming"])
        self.assertEqual(vocatives.negative_evidence(result), {})

    def test_longest_term_wins_at_the_same_position(self):
        lines = [{"index": 1, "text": "茉里子，等等我", "cluster_id": "SPEAKER_A1"}]
        result = vocatives.extract_address_terms(lines, ["茉里", "茉里子"])
        self.assertEqual(list(result["terms"]), ["茉里子"])


class TestAddressTermsReachTheWriter(unittest.TestCase):
    """P1 is only real if the writer is shown the fact, not just if a module can
    compute it: the ep02 mistake happened in the writing pass."""

    def _ws(self, ws: Path):
        visual = ws / ".cache" / "visual"
        visual.mkdir(parents=True, exist_ok=True)
        (ws / ".cache" / "alignment").mkdir(parents=True, exist_ok=True)
        (visual / "scenes.json").write_text(json.dumps({"scenes": [{
            "scene_id": "SCENE_01", "macro_index": 1, "start_ms": 0, "end_ms": 30_000,
            "start_timecode": "00:00:00.000", "end_timecode": "00:00:30.000",
            "duration_ms": 30_000, "slugline": "中庭", "child_shot_count": 1,
        }]}, ensure_ascii=False), encoding="utf-8")
        (visual / "shots.json").write_text(json.dumps({"scenes": []}), encoding="utf-8")
        (ws / ".cache" / "alignment" / "aligned_timeline.json").write_text(json.dumps({
            "shots": [{"shot_id": "SCENE_01", "dialogues": [
                {"sub_index": 1, "text": "茉里 你交朋友了", "speaker": "SPEAKER_A1"},
                {"sub_index": 2, "text": "啊，我没有啦", "speaker": "SPEAKER_A2"},
            ]}]}, ensure_ascii=False), encoding="utf-8")
        (ws / "materials").mkdir(parents=True, exist_ok=True)
        (ws / "materials" / "bible.json").write_text(json.dumps(
            {"characters": [{"name": "茉里"}]}, ensure_ascii=False), encoding="utf-8")

    def test_manifest_carries_address_terms_and_the_contract_says_so(self):
        import contextlib
        import io
        import tempfile
        from build_scene_manifest import build_manifest
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            self._ws(ws)
            with contextlib.redirect_stderr(io.StringIO()):
                manifest = build_manifest(ws, max_keyframes=2)
        terms = {t["term"]: t for t in manifest["address_terms"]}
        self.assertIn("茉里", terms)
        self.assertEqual(terms["茉里"]["count"], 1)
        self.assertEqual(terms["茉里"]["spoken_by"], {"SPEAKER_A1": 1})
        self.assertIn("address_terms", manifest["instructions"])
        self.assertIn("绝不是那句话的说话人", manifest["instructions"])


if __name__ == "__main__":
    unittest.main()
