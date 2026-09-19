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
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

SCRIPTS_DIR = (Path(__file__).parent.parent / "scripts").resolve()
sys.path.insert(0, str(SCRIPTS_DIR))

import splice_screenplay as sp


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
        allowed = {"菈菈", "茉里"}
        text = "**菈菈**（躬身）：你好／再见\n\n**陌生旅人**：嗯\n"
        warnings = sp.lint_speaker_names([text], allowed)
        self.assertTrue(any("陌生旅人" in w for w in warnings))

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

    def test_production_specific_label_warns_unless_whitelisted(self):
        # Burned-in show vocabulary was removed from the plugin whitelist:
        # such labels must come from the workspace bible (speaker_whitelist).
        text = "**内景·日**｜店铺\n\n**面试的店主**\n请进\n"
        self.assertTrue(any("面试的店主" in w for w in sp.lint_speaker_names([text], set())))
        self.assertEqual(sp.lint_speaker_names([text], {"面试的店主"}), [])


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
