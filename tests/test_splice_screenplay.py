#!/usr/bin/env python3
"""
tests/test_splice_screenplay.py - Verbatim splicer validation & assembly tests.

The splice stage is the fidelity guarantee of the v2 pipeline: dialogue text is
injected from extracted.json into [[SUB:n]] placeholders written by the LLM.
These tests pin the fatal-coverage semantics (missing / duplicated / orphaned
placeholders must abort), the verbatim injection, the speaker-name lint, and
the assembly tables.
"""

import io
import json
import re
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

SCRIPTS_DIR = (Path(__file__).parent.parent / "scripts").resolve()
sys.path.insert(0, str(SCRIPTS_DIR))

import series  # noqa: E402
import splice_screenplay as sp  # noqa: E402


def write_scene(drafts_dir: Path, idx: int, text: str) -> Path:
    p = drafts_dir / f"scene_{idx:02d}.md"
    p.write_text(text, encoding="utf-8")
    return p


SCENE_1 = """## 第 1 场
> 概要：测试场一
**内景 · 测试房间 —— 夜** `[TC 00:00:00–00:00:10]`

**人物：** 菈菈

△ 【菈菈】（红发蓝瞳）坐在床边翻漫画。

**姑婆**
（拿着吸尘器）
[[SUB:1]]

**菈菈**
（错愕）
[[SUB:2]]
"""

SCENE_2 = """## 第 2 场
> 概要：测试场二
**外景 · 街道 —— 日** `[TC 00:00:10–00:00:20]`

**菈菈**
[[SUB:3]]

（字幕卡：[[SUB:4]]）
"""


class TestValidateAndSplice(unittest.TestCase):

    def setUp(self):
        self.verbatim = {1: "菈菈 我说你啊", 2: "你弄坏的窗户 花了十万才修好", 3: "真的吗", 4: "（未完待续）"}
        self.expected = [{1: "", 2: ""}, {3: "", 4: ""}]
        self.tmp = tempfile.TemporaryDirectory()
        self.drafts = Path(self.tmp.name) / "scene_drafts"
        self.drafts.mkdir(parents=True)
        write_scene(self.drafts, 1, SCENE_1)
        write_scene(self.drafts, 2, SCENE_2)

    def tearDown(self):
        self.tmp.cleanup()

    def test_splice_is_verbatim_and_complete(self):
        texts, warnings, count = sp.validate_and_splice(
            [self.drafts / "scene_01.md", self.drafts / "scene_02.md"], self.expected, self.verbatim
        )
        self.assertEqual(count, 4)
        self.assertIn("菈菈 我说你啊", texts[0])
        self.assertIn("你弄坏的窗户 花了十万才修好", texts[0])
        self.assertNotIn("[[SUB:", texts[0])
        self.assertIn("（字幕卡：（未完待续））", texts[1])

    def test_missing_placeholder_is_fatal(self):
        broken = SCENE_1.replace("[[SUB:2]]\n", "")
        write_scene(self.drafts, 1, broken)
        with self.assertRaises(SystemExit):
            sp.validate_and_splice(
                [self.drafts / "scene_01.md", self.drafts / "scene_02.md"], self.expected, self.verbatim
            )

    def test_duplicated_placeholder_is_fatal(self):
        write_scene(self.drafts, 1, SCENE_1 + "\n**菈菈**\n[[SUB:2]]\n")
        with self.assertRaises(SystemExit):
            sp.validate_and_splice(
                [self.drafts / "scene_01.md", self.drafts / "scene_02.md"], self.expected, self.verbatim
            )

    def test_placeholder_from_other_scene_is_fatal(self):
        write_scene(self.drafts, 1, SCENE_1 + "\n**路人**\n[[SUB:3]]\n")
        with self.assertRaises(SystemExit):
            sp.validate_and_splice(
                [self.drafts / "scene_01.md", self.drafts / "scene_02.md"], self.expected, self.verbatim
            )

    def test_orphan_subtitle_is_fatal(self):
        extra = dict(self.verbatim)
        extra[99] = "无家可归的一句"
        with self.assertRaises(SystemExit):
            sp.validate_and_splice(
                [self.drafts / "scene_01.md", self.drafts / "scene_02.md"], self.expected, extra
            )

    def test_out_of_order_placeholders_are_fatal(self):
        """SKILL.md requires 按序出现; coverage alone was checked as a set, so a draft
        could present 第二句 before 第一句 and still assemble cleanly."""
        swapped = (SCENE_1.replace("[[SUB:1]]", "[[TMP]]").replace("[[SUB:2]]", "[[SUB:1]]")
                   .replace("[[TMP]]", "[[SUB:2]]"))
        self.assertEqual(re.findall(r"SUB:(\d)", swapped), ["2", "1"])
        write_scene(self.drafts, 1, swapped)
        buf = io.StringIO()
        with self.assertRaises(SystemExit) as ctx:
            with redirect_stderr(buf):
                sp.validate_and_splice(
                    [self.drafts / "scene_01.md", self.drafts / "scene_02.md"],
                    self.expected, self.verbatim)
        self.assertEqual(ctx.exception.code, 1)
        self.assertIn("out of subtitle order", buf.getvalue())
        self.assertIn("[[SUB:2]] before [[SUB:1]]", buf.getvalue())

    def test_nonexistent_sub_index_is_fatal(self):
        write_scene(self.drafts, 1, SCENE_1 + "\n**菈菈**\n[[SUB:999]]\n")
        with self.assertRaises(SystemExit):
            sp.validate_and_splice(
                [self.drafts / "scene_01.md", self.drafts / "scene_02.md"], self.expected, self.verbatim
            )

    def test_structural_lint_warnings(self):
        write_scene(self.drafts, 1, "只有台词，没有场头。\n\n**菈菈**\n[[SUB:1]]\n[[SUB:2]]\n")
        _, warnings, _ = sp.validate_and_splice(
            [self.drafts / "scene_01.md", self.drafts / "scene_02.md"], self.expected, self.verbatim
        )
        joined = " ".join(warnings)
        self.assertIn("scene_01", joined)
        self.assertIn("H2", joined)

    def test_multiple_placeholders_on_one_merged_line(self):
        # Same-speaker cues merged with the full-width slash stay two verbatim cues.
        merged = (
            "## 场 1【测试场】\n**内景·夜**｜测试房间\n\n"
            "**姑婆**（拿着吸尘器）：[[SUB:1]]／[[SUB:2]]\n"
        )
        write_scene(self.drafts, 1, merged)
        write_scene(self.drafts, 2, SCENE_2)
        texts, _, count = sp.validate_and_splice(
            [self.drafts / "scene_01.md", self.drafts / "scene_02.md"], self.expected, self.verbatim
        )
        self.assertEqual(count, 4)
        self.assertIn("菈菈 我说你啊／你弄坏的窗户 花了十万才修好", texts[0])


class TestSpeakerLint(unittest.TestCase):

    def test_unknown_name_warns(self):
        allowed = {"菈菈", "茉里", "旁白"}
        text = "**菈菈**\n你好\n\n**完全陌生的名字**\n嗯\n"
        warnings = sp.lint_speaker_names([text], allowed)
        self.assertTrue(any("完全陌生的名字" in w for w in warnings))

    def test_known_names_pass(self):
        text = "**菈菈**\n你好\n\n**旁白**\n（画外）\n后来\n"
        self.assertEqual(sp.lint_speaker_names([text], {"菈菈", "茉里", "旁白"}), [])

    def test_merged_inline_head_is_linted(self):
        """A head naming several speakers is judged part by part - and only the
        name-shaped part is flagged. 「陌生旅人」 is a description, not a name
        (v0.6 P1 reversed the direction; the old rule flagged it)."""
        allowed = {"菈菈", "茉里"}
        slug = "**内景·日**｜房间\n\n"
        text = slug + "**菈菈**（躬身）：你好／再见\n\n**陌生旅人**：嗯\n"
        self.assertEqual(sp.lint_speaker_names([text], allowed), [])
        buckets = sp.audit_speaker_labels([text], allowed)
        self.assertIn("陌生旅人", buckets["descriptive"])   # counted, not punished
        self.assertIn("菈菈", buckets["named"])

        text2 = slug + "**菈菈**（躬身）：你好／再见\n\n**艾拉**：嗯\n"
        warnings = sp.lint_speaker_names([text2], allowed)
        self.assertTrue(any("艾拉" in w for w in warnings))

    def test_slug_and_character_lines_are_not_dialogue_heads(self):
        text = "**内景·日**｜测试房间\n\n**人物：** 菈菈、陌生旅客\n"
        self.assertEqual(sp.lint_speaker_names([text], set()), [])

    def test_bare_slug_line_is_not_a_dialogue_head(self):
        # `**黑场**` is the whole-scene slug of dialogue-free black scenes.
        text = "## 场 31【黑场】\n**黑场**\n\n△ 黑场。\n"
        self.assertEqual(sp.lint_speaker_names([text], set()), [])

    def test_paren_qualifier_traces_to_generic_base(self):
        # 「路人（男）」 is allowed wherever plain 路人 is.
        text = "**路人（男）**\n让让\n"
        self.assertEqual(sp.lint_speaker_names([text], {"路人"}), [])

    def test_voice_of_pattern_is_a_descriptive_label(self):
        # The writing contract permits descriptive tags like 「王子的声音」;
        # the lint targets fabricated PROPER nouns, not these.
        text = "**王子的声音**\n是谁\n"
        self.assertEqual(sp.lint_speaker_names([text], set()), [])

    def test_bible_prefix_match_requires_two_chars(self):
        slug = "**内景·日**｜房间\n\n"
        self.assertEqual(sp.lint_speaker_names([slug + "**菈菈与妈妈**\n好啊\n"], {"菈菈"}), [])
        warnings = sp.lint_speaker_names([slug + "**菈某与妈妈**\n好啊\n"], {"菈"})
        self.assertTrue(any("菈某与妈妈" in w for w in warnings))

    def test_descriptive_shape_is_not_a_naming_violation(self):
        """v0.6 P1 replaced the old rule, which flagged 「面试的店主」「女声」
        「系统音」 while passing 「威严的声音」 - equally descriptive labels got
        opposite treatment purely on suffix shape, so the cheapest way to satisfy
        the guard was a LONGER label rather than better evidence."""
        text = "**内景·日**｜店铺\n\n**面试的店主**\n请进\n"
        self.assertEqual(sp.lint_speaker_names([text], set()), [])
        self.assertIn("面试的店主", sp.audit_speaker_labels([text], set())["descriptive"])

    def test_the_reverse_bias_examples_all_pass(self):
        """The three labels ep02 measured as false alarms, verbatim."""
        text = "**内景·日**｜店铺\n\n**女声**：嗯\n\n**系统音**：请输入\n\n**关西腔者**：多谢\n"
        self.assertEqual(sp.lint_speaker_names([text], set()), [])

    def test_one_char_stem_before_a_descriptive_tail_still_traces(self):
        """「凉音」 reads as a name even though 音 is a descriptive tail - the >=2
        char stem is what stops a real character name from sailing through."""
        text = "**内景·日**｜店铺\n\n**凉音**：早\n"
        warnings = sp.lint_speaker_names([text], set())
        self.assertTrue(any("凉音" in w for w in warnings))


class TestSpeakerLineageGate(unittest.TestCase):
    """v0.6 P5: "in the table" is not the same as "signed off as this speaker".

    The accident that started this design reads, to a membership check, like a
    perfectly legal name: 茉里 IS a character in that show. What makes the label
    wrong is that no cluster was ever signed off as her - she was the person being
    talked to. Only a lineage check can tell those apart.
    """
    CAST = {
        "entities": [{"id": "C1", "canonical_name": "茉里", "status": "approved",
                      "aliases": ["マリー"]},
                     {"id": "C2", "canonical_name": "托德", "status": "approved", "aliases": []}],
        "slots": [{"slot_id": "S1", "entity_id": "C2"}],
        "clusters": [{"cluster_id": "SPEAKER_A1",
                      "assignment": {"slot_id": "S1", "status": "approved"}}],
    }

    def test_lineage_list_holds_only_signed_speakers(self):
        self.assertEqual(sp.cast_lineage_names(self.CAST), ["托德"])

    def test_a_real_character_still_fails_without_lineage(self):
        lineage = set(sp.cast_lineage_names(self.CAST))
        text = "**内景·日**｜早饭桌\n\n**茉里**：我没有\n"
        self.assertTrue(any("茉里" in w for w in
                            sp.lint_speaker_names([text], lineage, cast_enforced=True)))
        self.assertEqual(sp.lint_speaker_names([text.replace("**茉里**", "**托德**")],
                                               lineage, cast_enforced=True), [])

    def test_alias_of_a_signed_speaker_is_enough(self):
        cast = json.loads(json.dumps(self.CAST))
        cast["clusters"][0]["assignment"]["slot_id"] = "S0"
        cast["slots"].append({"slot_id": "S0", "entity_id": "C1"})
        lineage = set(sp.cast_lineage_names(cast))
        self.assertIn("マリー", lineage)
        self.assertEqual(sp.lint_speaker_names(["**内景·日**｜x\n\n**マリー**：嗯\n"],
                                               lineage, cast_enforced=True), [])

    def test_no_cast_document_means_no_lineage_requirement(self):
        self.assertIsNone(sp.cast_lineage_names(None))


class TestAttributionAudit(unittest.TestCase):
    """§6 rows 5-6: a draft must not quietly overrule cast.json. The lineage gate
    says a label traces to SOME signed entity; only the per-line audit can see
    that line 1's cluster was signed as someone else - the ep02 accident wearing
    a table. Every violation is waivable by an in-scene attribution-override
    comment, and nothing else waives it."""

    CAST = {
        "entities": [
            {"id": "C1", "canonical_name": "托德", "status": "approved", "aliases": []},
            {"id": "C2", "canonical_name": "玛丽亚", "status": "approved", "aliases": []},
        ],
        "clusters": [
            {"cluster_id": "SPEAKER_A1",
             "assignment": {"slot_id": "S1", "status": "approved", "entity_id": "C1"}},
            {"cluster_id": "SPEAKER_A2",
             "assignment": {"slot_id": "S2", "status": "approved", "entity_id": "C2"}},
            {"cluster_id": "SPEAKER_A3",
             "assignment": {"slot_id": "S3", "status": "unknown", "entity_id": None}},
        ],
    }
    DIALOGUES = [
        [{"sub_index": 1, "speaker": "SPEAKER_A1"},
         {"sub_index": 2, "speaker": "SPEAKER_A3"}],
        [],
    ]

    def _audit(self, texts, dialogues=None):
        return sp.audit_attribution(texts, self.DIALOGUES if dialogues is None else dialogues,
                                    self.CAST)

    def test_placeholder_maps_to_the_nearest_preceding_head(self):
        text = "**内景·日**｜x\n\n**托德**：[[SUB:1]]\n\n**奶奶**\n（放下碗）\n[[SUB:2]]\n"
        heads = sp._sub_index_heads(text)
        self.assertEqual(heads, {1: "托德", 2: "奶奶"})

    def test_placeholder_before_any_head_has_no_speaker(self):
        text = "**内景·日**｜x\n（字幕卡：[[SUB:4]]）\n**托德**：[[SUB:1]]\n"
        self.assertEqual(sp._sub_index_heads(text), {1: "托德"})

    def test_override_numbers_ignore_digits_inside_entity_ids(self):
        covers = sp.parse_overrides("<!-- attribution-override: A1 8,9 → C2 reason=呼语 -->")
        self.assertEqual(covers[0]["lines"], {8, 9}, "the 2 in C2 is not line 2")
        self.assertTrue(sp._covers(covers[0], 8, "SPEAKER_A1"))
        self.assertFalse(sp._covers(covers[0], 2, "SPEAKER_A2"))

    def test_re_attribution_to_another_signed_entity_is_a_violation(self):
        violations, count = self._audit(["**内景·日**｜x\n\n**玛丽亚**：[[SUB:1]]\n"])
        self.assertEqual(count, 0)
        self.assertEqual(len(violations), 1)
        self.assertIn("第 1 条", violations[0])
        self.assertIn("玛丽亚", violations[0])
        self.assertIn("托德", violations[0])

    def test_the_line_s_own_entity_passes(self):
        self.assertEqual(self._audit(["**内景·日**｜x\n\n**托德**：[[SUB:1]]\n"])[0], [])

    def test_an_override_comment_waives_the_re_attribution(self):
        text = ("**内景·日**｜x\n\n<!-- attribution-override: A1 1 → C2 reason=呼语 -->\n\n"
                "**奶奶**：[[SUB:1]]\n")
        violations, count = self._audit([text])
        self.assertEqual(violations, [])
        self.assertEqual(count, 1)

    def test_an_override_for_another_line_and_cluster_waives_nothing(self):
        text = "**内景·日**｜x\n\n<!-- attribution-override: A2 9 → C1 -->\n\n**玛丽亚**：[[SUB:1]]\n"
        self.assertEqual(len(self._audit([text])[0]), 1)

    def test_a_proper_name_on_an_unknown_cluster_is_a_violation(self):
        violations, _ = self._audit(["**内景·日**｜x\n\n**托德**：[[SUB:2]]\n"])
        self.assertEqual(len(violations), 1)
        self.assertIn("无定名", violations[0])

    def test_a_descriptive_head_reattributes_nothing(self):
        self.assertEqual(self._audit(["**内景·日**｜x\n\n**青年男声**：[[SUB:2]]\n"
                                      "**威严的声音**：[[SUB:1]]\n"])[0], [])

    def test_a_stale_cluster_id_is_skipped_not_fatal(self):
        dialogues = [[{"sub_index": 1, "speaker": "SPEAKER_Z9"}]]
        self.assertEqual(self._audit(["**内景·日**｜x\n\n**玛丽亚**：[[SUB:1]]\n"], dialogues)[0], [])

    def test_speaker_unknown_is_skipped(self):
        dialogues = [[{"sub_index": 1, "speaker": "SPEAKER_UNKNOWN"}]]
        self.assertEqual(self._audit(["**内景·日**｜x\n\n**玛丽亚**：[[SUB:1]]\n"], dialogues)[0], [])


class TestNamingGateExitCode(unittest.TestCase):
    """§13 scenario 3 drives the CLI, not the lint function: 5319ab4 accidentally
    removed the fatal exit while refactoring the appendix, and 317 function-level
    tests stayed green while the deliverable shipped with fabricated names and
    exit 0. These tests pin the process behaviour: with a signed table in force,
    a naming violation exits 9 and writes no deliverable."""

    def _ws(self, root: Path, first_head: str, first_line_extra: str = "") -> Path:
        ws = root / "episodes" / "ep02"
        (ws / ".cache" / "alignment").mkdir(parents=True)
        (ws / ".cache" / "subtitles").mkdir(parents=True)
        (ws / ".cache" / "cast").mkdir(parents=True)
        (ws / ".cache" / "scene_drafts").mkdir(parents=True)
        (ws / "materials").mkdir(parents=True)
        series_root = root / "series"
        series_root.mkdir(exist_ok=True)
        (series_root / "cast.approved.json").write_text(json.dumps({
            "schema": "vts-cast/v1", "version": "1", "entities": [
                {"id": "C1", "canonical_name": "托德", "status": "approved", "aliases": [],
                 "approved_by": "human", "approved_at": "2026-09-20", "evidence": []},
                {"id": "C2", "canonical_name": "玛丽亚", "status": "approved", "aliases": [],
                 "approved_by": "human", "approved_at": "2026-09-20", "evidence": []},
            ]}, ensure_ascii=False), encoding="utf-8")
        series.bind_series(ws, series_root)
        manifest = {"total_scenes": 1, "scenes": [{
            "sequence_title": "", "start_timecode": "00:00:00:00",
            "end_timecode": "00:00:10:00",
            "dialogues": [{"sub_index": 1, "speaker": "SPEAKER_A1"},
                          {"sub_index": 2, "speaker": "SPEAKER_A3"}]}]}
        (ws / ".cache" / "alignment" / "scene_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        (ws / ".cache" / "subtitles" / "extracted.json").write_text(json.dumps(
            {"source_detail": "test", "items": [{"index": 1, "text": "你好"},
                                                {"index": 2, "text": "下雨了"}]},
            ensure_ascii=False), encoding="utf-8")
        cast = {
            "schema": "vts-cast/v1", "entities": TestAttributionAudit.CAST["entities"],
            "clusters": TestAttributionAudit.CAST["clusters"], "slots": [],
        }
        (ws / ".cache" / "cast" / "cast.json").write_text(
            json.dumps(cast, ensure_ascii=False), encoding="utf-8")
        draft = (f"## 场 1【早饭桌】\n**内景·日**｜早饭桌\n\n"
                 f"{first_line_extra}**{first_head}**：[[SUB:1]]\n\n"
                 f"**青年男声**：[[SUB:2]]\n")
        (ws / ".cache" / "scene_drafts" / "scene_01.md").write_text(draft, encoding="utf-8")
        return ws

    def _run(self, ws: Path):
        return subprocess.run([sys.executable, str(SCRIPTS_DIR / "splice_screenplay.py"),
                               "-w", str(ws)], capture_output=True, text=True)

    def test_signed_names_exit_zero_and_write_the_deliverable(self):
        with tempfile.TemporaryDirectory() as td:
            ws = self._ws(Path(td), "托德")
            res = self._run(ws)
            self.assertEqual(res.returncode, 0, res.stderr)
            self.assertTrue((ws / "output" / "ep02_影视文学剧本.md").is_file())

    def test_untraced_proper_noun_exits_nine_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            ws = self._ws(Path(td), "艾拉")
            res = self._run(ws)
            self.assertEqual(res.returncode, 9, res.stderr)
            self.assertIn("艾拉", res.stderr)
            self.assertFalse((ws / "output").exists(),
                             "a violating run must not deliver a screenplay")

    def test_re_attribution_to_a_signed_but_wrong_entity_exits_nine(self):
        with tempfile.TemporaryDirectory() as td:
            ws = self._ws(Path(td), "玛丽亚")
            res = self._run(ws)
            self.assertEqual(res.returncode, 9, res.stderr)
            self.assertIn("attribution-override", res.stderr)

    def test_an_override_comment_turns_the_re_attribution_legal(self):
        with tempfile.TemporaryDirectory() as td:
            ws = self._ws(Path(td), "奶奶",
                          "<!-- attribution-override: A1 1 → C2 reason=呼语 -->\n")
            res = self._run(ws)
            self.assertEqual(res.returncode, 0, res.stderr)
            delivered = (ws / "output" / "ep02_影视文学剧本.md").read_text(encoding="utf-8")
            self.assertIn("留痕改判 1 处", delivered)


class TestFidelityAppendix(unittest.TestCase):
    """Every line must appear whether or not a cast.json exists. The first
    implementation put a conditional inside a string concatenation, so
    `A + B if cast else C + D` dropped the label-composition line precisely on
    the runs that most needed it."""

    BUCKETS = {"named": ["托德"], "descriptive": ["女声", "系统音"], "untraced": []}
    CAST = {"named": 1, "candidate": 0, "unknown": 3, "abstention_rate": 0.6,
            "over_split_suspects": 1, "thresholds_are_measured": False, "cast_version": "4"}

    def _text(self, cast_summary, enforced):
        return "".join(sp.fidelity_lines("| ON_SCREEN | 3 | x |", 15, 15,
                                        cast_summary, self.BUCKETS, enforced))

    def test_with_cast_table(self):
        text = self._text(self.CAST, True)
        self.assertIn("说话人标签构成", text)
        self.assertIn("演员表状态", text)
        self.assertIn("疑似过度切分 1 组", text)
        self.assertIn("尚未经测量", text)
        self.assertIn("表外专名为致命", text)

    def test_without_cast_document(self):
        text = self._text(None, False)
        self.assertIn("说话人标签构成", text)
        self.assertIn("可溯源 1 个、描述性（未定名）2 个、表外专名 0 个", text)
        self.assertNotIn("演员表状态", text)
        self.assertIn("仅告警", text)

    def test_line_counts_survive_zero_everything(self):
        text = self._text({"named": 0, "candidate": 0, "unknown": 0, "abstention_rate": 0.0,
                           "over_split_suspects": 0, "thresholds_are_measured": True,
                           "cast_version": None}, True)
        self.assertIn("弃权率 0%", text)
        self.assertIn("阈值来自测量", text)
        self.assertNotIn("疑似过度切分", text)
        self.assertIn("表版本 未签核", text)


class TestHeadingNormalization(unittest.TestCase):

    def test_legacy_draft_gets_title_from_slug_and_manifest_tc(self):
        text = "## 第 1 场\n> 概要：测试概要\n**内景 · 测试房间 —— 夜** `[TC 00:00:00–00:00:10]`\n\n**菈菈**\n你好\n"
        out = sp.normalize_scene_text(text, 4, "00:00:30", "00:00:40")
        self.assertIn("## 场 4【测试房间】（00:00:30 - 00:00:40）", out)
        self.assertNotIn("> 概要", out)

    def test_titled_heading_keeps_title_and_gets_manifest_tc(self):
        text = "## 场 7【海面·燃烧的帆船】\n**外景·夜**｜字幕卡\n正文\n"
        out = sp.normalize_scene_text(text, 7, "00:06:15", "00:07:42")
        self.assertIn("## 场 7【海面·燃烧的帆船】（00:06:15 - 00:07:42）", out)

    def test_writer_typed_tc_is_replaced_by_manifest_tc(self):
        text = "## 场 2【测试场】（99:99:99 - 99:99:99）\n正文\n"
        out = sp.normalize_scene_text(text, 2, "00:01:00", "00:02:00")
        self.assertIn("（00:01:00 - 00:02:00）", out)
        self.assertNotIn("99:99:99", out)

    def test_missing_heading_is_inserted(self):
        out = sp.normalize_scene_text("**内景·夜**｜房间\n\n**菈菈**：你好\n", 3, "00:05:00", "00:05:30")
        self.assertTrue(out.startswith("## 场 3【房间】（00:05:00 - 00:05:30）\n"))

    def test_draft_tc_fallback_when_manifest_empty(self):
        text = "## 场 1【测试场】（00:00:05 - 00:00:09）\n正文\n"
        out = sp.normalize_scene_text(text, 1, "", "")
        self.assertIn("（00:00:05 - 00:00:09）", out)


class TestAssemblyTables(unittest.TestCase):

    def test_overview_row(self):
        # build_overview numbers rows by position (scene files are spliced in order),
        # so a writer's mis-numbered heading cannot corrupt the overview table.
        text = (
            "## 场 3【烤箱风波】（00:13:31 - 00:14:02）\n"
            "**内景·日**｜蛋糕店·烘焙间\n\n**菈菈**：对不起\n"
        )
        overview = sp.build_overview([text], ["新人修行序列"])
        self.assertIn("| 场号 | 序列 | 时空 | 概要 | 时间码 |", overview)
        self.assertIn("第 1 场", overview)
        self.assertIn("新人修行序列", overview)
        self.assertIn("烤箱风波", overview)
        self.assertIn("内景·日｜蛋糕店·烘焙间", overview)
        self.assertIn("00:13:31 - 00:14:02", overview)

    def test_overview_without_outline_keeps_empty_sequence_cells(self):
        text = "## 场 1【测试场】（00:00:00 - 00:00:10）\n**内景·夜**｜房间\n\n**菈菈**：你好\n"
        overview = sp.build_overview([text])
        self.assertIn("| 第 1 场 |  | ", overview)

    def test_overview_escapes_pipes_in_cells(self):
        # A title containing a raw pipe must not corrupt the 场次总表 row.
        text = "## 场 1【海|面】（00:00:00 - 00:00:10）\n**内景·夜**｜房间\n\n**菈菈**：你好\n"
        overview = sp.build_overview([text])
        row = next(r for r in overview.splitlines() if r.startswith("| 第 1 场"))
        self.assertIn("海\\|面", row)

    def test_av_statistics(self):
        aligned = {"shots": [
            {"shot_type": "DIALOGUE_SHOT", "dialogues": [
                {"av_relationship": "ON_SCREEN"}, {"av_relationship": "OFF_SCREEN"}]},
            {"shot_type": "SILENT_ACTION", "dialogues": []},
        ]}
        rows = sp.av_statistics(aligned)
        by_name = {r[0]: r[1] for r in rows}
        self.assertEqual(by_name["ON_SCREEN"], 1)
        self.assertEqual(by_name["OFF_SCREEN"], 1)
        self.assertEqual(by_name["SILENT_ACTION（纯视听镜头）"], 1)


if __name__ == "__main__":
    unittest.main()
