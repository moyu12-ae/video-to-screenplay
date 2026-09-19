#!/usr/bin/env python3
"""
tests/test_end_to_end.py - The five stages composed against a real ffmpeg-encode.

Unit tests pin each stage in isolation; this one exists because the stages disagreed
with each other about what "every line accounted for" meant, and no isolated test saw
it. It builds a short synthetic episode (hard colour cuts + distinct tones) whose
subtitle file is deliberately hostile - UTF-8 BOM, CRLF, HTML tags, a two-line cue and
a cue landing past the last cut - drives every stage, and asserts the finished
screenplay carries all of it verbatim.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = (Path(__file__).parent.parent / "scripts").resolve()
PY = sys.executable

TEXTS = ["第一句台词。", "第二句台词，跨在一处硬切上。", "<i>第三句</i>带标签。",
         "第四句\n分两行写。", "第五句台词。", "第六句。", "第七句台词稍长一些。",
         "第八句。", "第九句。", "第十句台词。", "第十一句。", "第十二句。",
         "第十三句。", "第十四句收尾。",
         "第十五句完全落在最后一个切点之后。"]


def timecode(ms):
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, milli = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{milli:03d}"


class TestEndToEndPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if shutil.which("ffmpeg") is None:
            raise unittest.SkipTest("ffmpeg not installed")
        cls.ws = Path(tempfile.mkdtemp(prefix="v2s-e2e-"))
        mat = cls.ws / "materials"
        mat.mkdir(parents=True)
        try:
            cls._build_episode(mat)
        except subprocess.CalledProcessError as e:
            raise unittest.SkipTest(f"could not synthesise the test episode: {e}")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.ws, ignore_errors=True)

    @classmethod
    def _build_episode(cls, mat):
        for i, (colour, tone) in enumerate(zip(
                ["red", "blue", "green", "yellow", "black", "teal"],
                [220, 440, 660, 880, 300, 500])):
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                 "-f", "lavfi", "-i", f"color=c={colour}:s=320x240:d=4:r=25",
                 "-f", "lavfi", "-i", f"sine=frequency={tone}:duration=4",
                 "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-c:a", "aac", "-ar", "16000", "-ac", "1", str(cls.ws / f"clip{i}.mp4")],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=180)
        (cls.ws / "concat.txt").write_text(
            "\n".join(f"file '{cls.ws}/clip{i}.mp4'" for i in range(6)), encoding="utf-8")
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat",
             "-safe", "0", "-i", str(cls.ws / "concat.txt"), "-c", "copy",
             str(mat / "episode.mp4")],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=180)

        cues, t = [], 500
        for i, body in enumerate(TEXTS, start=1):
            if i == 15:
                start, dur = 30000, 400   # past the end of a 24s film
            else:
                start, dur = t, (2600 if i == 2 else 1400)
                t += 1150
            cues.append(f"{i}\n{timecode(start)} --> {timecode(start + dur)}\n{body}\n")
        # BOM + CRLF: the combination that used to delete the first line of dialogue.
        (mat / "episode.srt").write_bytes(
            b"\xef\xbb\xbf" + ("\r\n".join(cues) + "\r\n").encode("utf-8"))
        (mat / "bible.json").write_text(
            json.dumps({"characters": ["甲", "乙"]}, ensure_ascii=False), encoding="utf-8")

    def _stage(self, script, *args, expect=0):
        proc = subprocess.run([PY, str(SCRIPTS / script), *args],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, timeout=600)
        self.assertEqual(proc.returncode, expect,
                         f"{script} exited {proc.returncode}\n{proc.stderr[-2000:]}")
        return proc

    def test_full_pipeline_preserves_every_dialogue_line(self):
        ws = str(self.ws)
        self._stage("workspace.py", "init", "--workspace", ws)
        self._stage("workspace.py", "probe", "--workspace", ws)

        shots = self._stage("scene_detect.py", "--workspace", ws, "--threshold", "0.30")
        (self.ws / ".cache/visual/shots.json").write_text(shots.stdout, encoding="utf-8")
        shot_doc = json.loads(shots.stdout)
        self.assertGreaterEqual(shot_doc["total_scenes"], 4,
                                "the colour cuts should be detectable")

        gate = self._stage("subtitle_extractor.py", "--workspace", ws, "--require-subtitles")
        (self.ws / ".cache/subtitles/extracted.json").write_text(gate.stdout, encoding="utf-8")
        parsed = json.loads(gate.stdout)
        self.assertEqual(parsed["source_tier"], "TIER_1_EXTERNAL")
        self.assertEqual(parsed["subtitle_count"], len(TEXTS),
                         "a BOM must not cost a line of dialogue")
        self.assertEqual(parsed["items"][0]["text"], TEXTS[0])

        # No DASHSCOPE_API_KEY in this environment by design: the documented
        # degradation route has to carry the run all the way to a finished script.
        self._stage("speaker_diarize.py", "--workspace", ws, "prepare", expect=6)
        env = dict(os.environ, DASHSCOPE_API_KEY="")
        proc = subprocess.run([PY, str(SCRIPTS / "speaker_diarize.py"), "--workspace", ws, "run"],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                              env=env, timeout=300)
        self.assertEqual(proc.returncode, 8, "run must refuse without a key")
        merged = self._stage("speaker_diarize.py", "--workspace", ws, "merge", "--empty-fallback")
        speakers = self.ws / ".cache/audio/speakers.json"
        speakers.write_text(merged.stdout, encoding="utf-8")
        sdoc = json.loads(merged.stdout)
        self.assertFalse(sdoc["acoustic_clustering_enabled"],
                         "a zero-call degraded artifact must not claim acoustic clustering")
        self.assertEqual(sdoc["diarization_source"]["backend"], "none")

        grp = self._stage("semantic_scene_grouper.py", "--workspace", ws)
        (self.ws / ".cache/visual/scenes.json").write_text(grp.stdout, encoding="utf-8")
        self.assertGreater(json.loads(grp.stdout)["total_macro_scenes"], 0)

        al = self._stage("align_timeline.py", "--workspace", ws)
        (self.ws / ".cache/alignment/aligned_timeline.json").write_text(al.stdout,
                                                                        encoding="utf-8")
        adoc = json.loads(al.stdout)
        self.assertEqual(adoc["cues_assigned"], adoc["total_dialogue_cues"],
                         "the aligner may not leave a cue for splice to trip over")
        self.assertGreaterEqual(adoc["cues_nearest_shot_fallback"], 1,
                                "the past-the-end cue should have used the fallback")

        self._stage("build_scene_manifest.py", "--workspace", ws)
        manifest = json.loads(
            (self.ws / ".cache/alignment/scene_manifest.json").read_text(encoding="utf-8"))
        self._write_drafts(manifest)

        self._stage("splice_screenplay.py", "--workspace", ws, "--title", "端到端 第01话")
        out_files = list((self.ws / "output").glob("*.md"))
        self.assertEqual(len(out_files), 1, "exactly one deliverable, in output/")
        text = out_files[0].read_text(encoding="utf-8")

        # --- fidelity: nothing lost, nothing invented, nothing left unresolved ---
        for i, line in enumerate(TEXTS, start=1):
            expected = line.replace("\n", " ").replace("<i>", "").replace("</i>", "")
            self.assertIn(expected, text, f"cue {i} missing from the screenplay")
        self.assertNotIn("SPEAKER_", text, "a provisional label reached the screenplay")
        # the fidelity appendix names the [[SUB:n]] convention in prose; the screenplay
        # body before it must have none left unresolved
        self.assertNotIn("[[SUB:", text.split("台词保真说明")[0],
                         "an unreplaced placeholder reached the screenplay")

    def _write_drafts(self, manifest):
        """Stand in for the multimodal writing pass: placeholders only, each sub_index
        exactly once, per the contract in build_scene_manifest.WRITING_CONTRACT."""
        drafts = self.ws / ".cache/scene_drafts"
        drafts.mkdir(exist_ok=True)
        for scene in manifest["scenes"]:
            idx = scene.get("scene_index") or scene.get("index")
            lines = [f"## 场 {idx}【端到端场景 {idx}】", "", "**内景·日**｜合成素材",
                     "**人物：** 甲", "", "△【甲】站在色块背景前，身体朝向画面左侧。", ""]
            rows, row = [], []
            for cue in scene["dialogues"]:
                if row and (row[0].get("speaker") or "") != (cue.get("speaker") or ""):
                    rows.append(row)
                    row = []
                row.append(cue)
            if row:
                rows.append(row)
            for row in rows:
                who = row[0].get("speaker") or "人物"
                if str(who).startswith("SPEAKER_"):
                    who = "人物"
                joined = "／".join(f"[[SUB:{c['sub_index']}]]" for c in row)
                lines += [f"**{who}**：{joined}", "", "△【甲】的手压在桌面上。", ""]
            (drafts / f"scene_{idx:02d}.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
