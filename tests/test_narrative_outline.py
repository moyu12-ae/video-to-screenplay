#!/usr/bin/env python3
"""
tests/test_narrative_outline.py - Phase 3.5 narrative outline work-order & validator.

The outline is the McKee sequence layer (value-shift units, not location units).
These tests pin: the work-order is emitted from extracted.json; the authored
structure must partition the subtitle stream contiguously with titles and value
arcs; optional acts must partition the sequence list; any violation is fatal.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = (Path(__file__).parent.parent / "scripts").resolve()
sys.path.insert(0, str(SCRIPTS_DIR))

import narrative_outline as no


def make_ws(items):
    tmp = tempfile.TemporaryDirectory()
    ws = Path(tmp.name)
    sub_dir = ws / ".cache" / "subtitles"
    sub_dir.mkdir(parents=True)
    (sub_dir / "extracted.json").write_text(
        json.dumps({"items": items}, ensure_ascii=False), encoding="utf-8"
    )
    return tmp, ws


ITEMS = [
    {"index": i, "start_ms": i * 5000, "end_ms": i * 5000 + 2000, "text": f"第{i}句"}
    for i in range(1, 7)
]

VALID = {
    "sequences": [
        {"seq_index": 1, "title": "铺垫", "value_from": "安稳", "value_to": "起疑",
         "start_sub": 1, "end_sub": 4, "evidence": "setup"},
        {"seq_index": 2, "title": "转折", "value_from": "起疑", "value_to": "决意",
         "start_sub": 5, "end_sub": 6, "evidence": "turn"},
    ]
}


class TestWorkorder(unittest.TestCase):

    def test_workorder_written_with_contract_and_stream(self):
        tmp, ws = make_ws(ITEMS)
        try:
            path = no.build_workorder(ws, ITEMS)
            self.assertTrue(path.is_file())
            doc = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("序列是叙事单位", doc["contract"])
            self.assertEqual(len(doc["subtitles"]), 6)
            self.assertEqual(doc["subtitles"][0]["sub_index"], 1)
        finally:
            tmp.cleanup()


class TestValidation(unittest.TestCase):

    def setUp(self):
        self.tmp, self.ws = make_ws(ITEMS)

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, doc):
        d = self.ws / ".cache" / "alignment"
        d.mkdir(parents=True, exist_ok=True)
        (d / "narrative_structure.json").write_text(
            json.dumps(doc, ensure_ascii=False), encoding="utf-8"
        )

    def test_valid_structure_passes_with_spans(self):
        self._write(VALID)
        result = no.validate_structure(self.ws, ITEMS)
        self.assertEqual(result["status"], "narrative_structure_valid")
        self.assertEqual(len(result["sequences"]), 2)
        self.assertEqual(result["sequences"][0]["sub_range"], [1, 4])
        self.assertIn("wall_to_next", result["sequences"][0])

    def test_gap_is_fatal(self):
        broken = json.loads(json.dumps(VALID, ensure_ascii=False))
        broken["sequences"][1]["start_sub"] = 6  # skips sub 5
        self._write(broken)
        with self.assertRaises(SystemExit):
            no.validate_structure(self.ws, ITEMS)

    def test_overlap_is_fatal(self):
        broken = json.loads(json.dumps(VALID, ensure_ascii=False))
        broken["sequences"][1]["start_sub"] = 4  # overlaps seq 1
        self._write(broken)
        with self.assertRaises(SystemExit):
            no.validate_structure(self.ws, ITEMS)

    def test_missing_value_arc_is_fatal(self):
        broken = json.loads(json.dumps(VALID, ensure_ascii=False))
        broken["sequences"][0]["value_to"] = ""
        self._write(broken)
        with self.assertRaises(SystemExit):
            no.validate_structure(self.ws, ITEMS)

    def test_incomplete_coverage_is_fatal(self):
        broken = json.loads(json.dumps(VALID, ensure_ascii=False))
        broken["sequences"][1]["end_sub"] = 5  # stream ends at 6
        self._write(broken)
        with self.assertRaises(SystemExit):
            no.validate_structure(self.ws, ITEMS)

    def test_missing_structure_exits_pending(self):
        with self.assertRaises(SystemExit) as ctx:
            no.validate_structure(self.ws, ITEMS)
        self.assertEqual(ctx.exception.code, 6)

    def test_bad_acts_coverage_is_fatal(self):
        broken = dict(VALID)
        broken["acts"] = [{"act_index": 1, "title": "第一幕", "start_seq": 1, "end_seq": 1}]
        self._write(broken)  # act stops at 1 but sequences reach 2
        with self.assertRaises(SystemExit):
            no.validate_structure(self.ws, ITEMS)


if __name__ == "__main__":
    unittest.main()
