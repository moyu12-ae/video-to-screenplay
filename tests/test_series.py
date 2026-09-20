#!/usr/bin/env python3
"""
tests/test_series.py - Regressions for the v0.6 P-1 series config directory.

The defect this exists to prevent: SKILL.md told users to put OP/ED windows in
materials/bible.json "once per season", while every reader looked only inside the
single-episode workspace - so the promise silently meant "copy the file per
episode". Binding must therefore be strictly additive: an unbound workspace has
to behave exactly like v0.5.x.
"""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import op_ed  # noqa: E402
import series  # noqa: E402


WORKSPACE_PY = SCRIPTS_DIR / "workspace.py"


class SeriesFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="v2s-series-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        self.root = self.tmp / "再見菈菈"
        self.root.mkdir()
        self.ep = self.tmp / "ep02"
        (self.ep / "materials").mkdir(parents=True)

    def write_series_windows(self, windows):
        (self.root / "op_ed_windows.json").write_text(
            json.dumps({"op_ed_windows": windows}, ensure_ascii=False), encoding="utf-8")

    def write_bible_windows(self, windows):
        (self.ep / "materials" / "bible.json").write_text(
            json.dumps({"op_ed_windows": windows}, ensure_ascii=False), encoding="utf-8")


class TestUnboundIsLegacy(SeriesFixture):
    def test_unbound_workspace_reads_bible_and_reports_unbound(self):
        self.write_bible_windows([{"start_ms": 0, "end_ms": 90000, "label": "OP"}])
        self.assertIsNone(series.resolve_series_root(self.ep))
        self.assertEqual([w["end_ms"] for w in series.load_op_ed_windows(self.ep)], [90000])
        status = series.series_status(self.ep)
        self.assertFalse(status["bound"])
        self.assertEqual(status["op_ed_source"], "bible")

    def test_no_config_means_no_windows(self):
        self.assertEqual(series.load_op_ed_windows(self.ep), [])
        self.assertEqual(op_ed.load_windows(str(self.ep)), [])
        self.assertEqual(series.series_status(self.ep)["op_ed_source"], "none")

    def test_dangling_pointer_falls_back_instead_of_crashing(self):
        series.write_json_atomic(
            series.pointer_path(self.ep),
            {"schema": series.SERIES_POINTER_SCHEMA, "series_root": str(self.tmp / "gone")})
        self.write_bible_windows([{"start_ms": 0, "end_ms": 90000, "label": "OP"}])
        stderr = _Capture()
        with stderr:
            windows = series.load_op_ed_windows(self.ep)
        self.assertEqual([w["end_ms"] for w in windows], [90000])
        self.assertIn("does not exist", stderr.text)
        self.assertFalse(series.series_status(self.ep)["bound"])


class TestBinding(SeriesFixture):
    def test_bind_writes_pointer_and_refuses_self(self):
        out = series.bind_series(self.ep, self.root)
        self.assertEqual(out["status"], "bound")
        doc = json.loads(series.pointer_path(self.ep).read_text(encoding="utf-8"))
        self.assertEqual(doc["schema"], series.SERIES_POINTER_SCHEMA)
        self.assertEqual(Path(doc["series_root"]).resolve(), self.root.resolve())
        with self.assertRaises(ValueError):
            series.bind_series(self.ep, self.ep)

    def test_series_windows_win_and_divergence_is_warned(self):
        self.write_bible_windows([{"start_ms": 0, "end_ms": 90000, "label": "OP"}])
        self.write_series_windows([{"start_ms": 0, "end_ms": 84000, "label": "OP"}])
        series.bind_series(self.ep, self.root)
        stderr = _Capture()
        with stderr:
            windows = op_ed.load_windows(str(self.ep))
        self.assertEqual([w["end_ms"] for w in windows], [84000])
        self.assertIn("differ", stderr.text)

    def test_agreement_does_not_warn(self):
        same = [{"start_ms": 0, "end_ms": 84000, "label": "OP"}]
        self.write_bible_windows(same)
        self.write_series_windows(same)
        series.bind_series(self.ep, self.root)
        stderr = _Capture()
        with stderr:
            op_ed.load_windows(str(self.ep))
        self.assertNotIn("differ", stderr.text)


class TestCastTable(SeriesFixture):
    TABLE = {
        "version": "1",
        "entities": [
            {"id": "C1", "canonical_name": "茉里", "status": "approved",
             "aliases": ["マリー", "Marie"]},
            {"id": "C2", "canonical_name": "托德", "status": "candidate", "aliases": []},
        ],
    }

    def _publish(self):
        # copy.deepcopy, not a shared class dict: one test mutates the version to
        # prove a later series-side edit is detected, and that must stay local.
        table = copy.deepcopy(self.TABLE)
        (self.root / "cast.approved.json").write_text(
            json.dumps(table, ensure_ascii=False), encoding="utf-8")
        return table

    def test_snapshot_records_digest_and_version(self):
        self._publish()
        series.bind_series(self.ep, self.root)
        snap = series.load_snapshot(self.ep)
        self.assertIsNotNone(snap)
        self.assertEqual(snap["version"], "1")
        self.assertEqual(snap["sha256"], series.sha256_of(self.root / "cast.approved.json"))

    def test_only_approved_entities_and_aliases_are_names(self):
        self._publish()
        series.bind_series(self.ep, self.root)
        names = series.approved_names(self.ep)
        self.assertIn("茉里", names)
        self.assertIn("Marie", names)
        self.assertNotIn("托德", names)

    def test_reading_the_table_never_mutates_it(self):
        self._publish()
        series.bind_series(self.ep, self.root)
        before = (self.root / "cast.approved.json").read_bytes()
        table = series.load_approved(self.ep)
        table["entities"].append({"id": "C9", "canonical_name": "编造", "status": "approved"})
        table["entities"][0]["canonical_name"] = "改掉了"
        self.assertEqual((self.root / "cast.approved.json").read_bytes(), before)

    def test_later_mutation_of_series_table_is_warned_not_applied_silently(self):
        table = self._publish()
        series.bind_series(self.ep, self.root)
        table["version"] = "2"
        (self.root / "cast.approved.json").write_text(
            json.dumps(table, ensure_ascii=False), encoding="utf-8")
        stderr = _Capture()
        with stderr:
            loaded = series.load_approved(self.ep)
        self.assertEqual(loaded["version"], "2")
        self.assertIn("changed after this workspace was bound", stderr.text)

    def test_unbound_without_snapshot_gives_empty_table(self):
        table = series.load_approved(self.ep)
        self.assertEqual(table["entities"], [])
        self.assertEqual(series.approved_names(self.ep), [])


class TestWorkspaceCLI(SeriesFixture):
    def _run(self, *args):
        res = subprocess.run([sys.executable, str(WORKSPACE_PY), *args],
                             capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)
        return res.stdout

    def test_init_with_series_creates_layout_and_pointer(self):
        out = json.loads(self._run("init", "-w", str(self.ep), "--series", str(self.root)))
        self.assertTrue((self.ep / "materials").is_dir())
        self.assertEqual(Path(out["series"]["series_root"]).resolve(), self.root.resolve())
        self.assertIsNone(out["series"]["cast"])  # nothing signed off yet

    def test_series_status_is_machine_readable(self):
        self._run("init", "-w", str(self.ep), "--series", str(self.root))
        out = json.loads(self._run("series", "-w", str(self.ep)))
        self.assertTrue(out["bound"])
        self.assertEqual(Path(out["series_root"]).resolve(), self.root.resolve())
        self.assertEqual(out["approved_entity_count"], 0)

    def test_bind_subcommand_snapshots_an_existing_table(self):
        self._run("init", "-w", str(self.ep))
        (self.root / "cast.approved.json").write_text(
            json.dumps({"version": "7", "entities": [
                {"id": "C1", "canonical_name": "茉里", "status": "approved", "aliases": []}]},
                ensure_ascii=False), encoding="utf-8")
        out = json.loads(self._run("series", "-w", str(self.ep), "--bind", str(self.root)))
        self.assertEqual(out["cast"]["version"], "7")
        status = json.loads(self._run("series", "-w", str(self.ep)))
        self.assertEqual(status["approved_entity_count"], 1)
        self.assertEqual(status["cast_version"], "7")


class _Capture:
    """Redirect sys.stderr into a buffer for the duration of a `with` block."""

    def __init__(self):
        self.text = ""
        self._saved = None
        self._file = None

    def __enter__(self):
        import io
        self._saved = sys.stderr
        self._file = io.StringIO()
        sys.stderr = self._file
        return self

    def __exit__(self, *exc):
        sys.stderr = self._saved
        self.text = self._file.getvalue()
        return False


if __name__ == "__main__":
    unittest.main()
