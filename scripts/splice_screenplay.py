#!/usr/bin/env python3
"""
scripts/splice_screenplay.py - Verbatim Splicer, Validator & Episode Assembler

Final stage of the v2 pipeline. The LLM writes each scene's 场号制 screenplay
text with dialogue as [[SUB:n]] placeholders (never retyping lines); this script:

  1. VALIDATES every scene file: each expected sub_index appears exactly once,
     in no other scene; structural lint (场头 / [TC] / 概要 present); speaker
     names must come from the bible / manifest (anti-hallucination guard).
  2. SPLICES the verbatim subtitle text into the placeholders - dialogue
     fidelity by construction, not by trust.
  3. NORMALIZES each scene heading to `## 场 N【标题】（起 - 止）`: the title is
     the writer's creative choice, the timecode paren is injected from the
     manifest, and legacy '> 概要：' blockquotes are stripped.
  4. ASSEMBLES the episode document: metadata header, 场次总表, spliced scenes,
     appendix (audio-visual relationship statistics + fidelity report).

Structural violations are fatal (nonzero exit); naming issues are warnings.
All writes stay inside the workspace output/ directory.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PLACEHOLDER_RE = re.compile(r"\[\[SUB:(\d+)\]\]")
SCENE_HEADING_RE = re.compile(r"^##\s*(第\s*\d+\s*场.*)$", re.M)  # legacy draft heading
H2_TITLED_RE = re.compile(r"^##\s*场\s*\d+\s*【\s*([^】]*?)\s*】\s*(?:（([^）]*)）)?\s*$", re.M)
HEADING_LINE_RE = re.compile(r"^##\s*(?:场\s*\d+\s*【[^】]*】(?:（[^）]*）)?|第\s*\d+\s*场)[^\n]*$", re.M)
ANY_HEADING_RE = re.compile(r"^##\s*(?:场\s*\d+\s*【|第\s*\d+\s*场)", re.M)
SLUG_RE = re.compile(r"^\*\*(.+?)\*\*", re.M)  # first bold line of a scene = its slug (may carry a trailing `[TC]`)
SLUG_LINE_RE = re.compile(r"^\*\*([^*\n].*?)\*\*([^\n]*)$", re.M)  # slug incl. the tail after the bold span (｜注记)
TC_RE = re.compile(r"\[TC[^\]]*\]")
SUMMARY_RE = re.compile(r"^>\s*概要[：:]\s*(.+)$", re.M)
# Dialogue head: legacy bare-line form `**名字**` and merged inline form
# `**名字**（提示）：甲／乙`. Slug (`**内景·日**｜…`) and `**人物：** …` lines
# are not dialogue heads and must not match.
DIALOGUE_HEAD_RE = re.compile(
    r"^\*\*([^*\n]{1,24})\*\*(?:\s*$|\s*[（(][^）)]*[）)]\s*[：:]|\s*[：:])", re.M
)

# Generic, non-proper-noun speakers allowed without bible backing. Only role,
# kinship and narration words belong here - production-specific descriptive
# labels must come from the workspace (materials/bible.json "speaker_whitelist"
# or characters_manifest), not from this plugin-level list.
GENERIC_SPEAKERS = {
    "旁白", "解说", "画外音", "众人", "群臣", "众侍", "大家",
    "路人", "店员", "摊主", "店主", "顾客", "客人", "司机",
    "孩子", "家人", "家人们", "长辈", "少女", "少年", "同学",
    "侍从", "员工", "工作人员", "广播", "广播员", "广告",
    "童声", "呼声", "合唱", "神秘人物",
    "姐姐", "姐姐们", "哥哥", "父亲", "母亲", "奶奶", "爷爷",
}
# Descriptive (non-proper-noun) label patterns, e.g. 「王子的声音」「神秘的声音」.
# The writing contract permits descriptive tags for unnamed characters; the
# lint's job is to catch fabricated PROPER nouns, not these.
DESCRIPTIVE_SPEAKER_RE = re.compile(r"^[^（）]{1,12}(的声音|之声)$")
PAREN_QUALIFIER_RE = re.compile(r"[（(][^（）()]*[)）]\s*$")


def load_json(path: Path) -> Optional[Any]:
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        sys.stderr.write(f"[WARN] Failed to load {path}: {e}\n")
        return None


def collect_scene_files(drafts_dir: Path, total_scenes: int) -> List[Path]:
    files: List[Path] = []
    for idx in range(1, total_scenes + 1):
        p = drafts_dir / f"scene_{idx:02d}.md"
        if not p.is_file():
            sys.stderr.write(f"[FATAL] Missing scene text: {p}. The writing pass is incomplete.\n")
            sys.exit(1)
        files.append(p)
    return files


def validate_and_splice(
    scene_files: List[Path],
    expected_by_scene: List[Dict[int, str]],
    verbatim: Dict[int, str],
) -> Tuple[List[str], List[str], int]:
    """Validate coverage per scene, then replace placeholders with verbatim text.
    Returns (spliced_texts, warnings, spliced_count). Fatal on coverage errors."""
    warnings: List[str] = []
    spliced_texts: List[str] = []
    spliced_count = 0
    claimed: Dict[int, int] = {}

    for pos, (path, expected) in enumerate(zip(scene_files, expected_by_scene), start=1):
        text = path.read_text(encoding="utf-8")
        found = [int(m.group(1)) for m in PLACEHOLDER_RE.finditer(text)]

        seen: set = set()
        for n in found:
            if n not in verbatim:
                sys.stderr.write(f"[FATAL] scene_{pos:02d}.md references [[SUB:{n}]] which does not exist in extracted.json.\n")
                sys.exit(1)
            if n in seen:
                sys.stderr.write(f"[FATAL] scene_{pos:02d}.md uses [[SUB:{n}]] more than once.\n")
                sys.exit(1)
            seen.add(n)
            claimed[n] = claimed.get(n, 0) + 1

        missing = sorted(set(expected) - seen)
        extra = sorted(seen - set(expected))
        if missing:
            sys.stderr.write(
                f"[FATAL] scene_{pos:02d}.md is missing {len(missing)} dialogue placeholder(s): {missing[:12]}. "
                "Every subtitle of the scene must be woven in exactly once.\n"
            )
            sys.exit(1)
        if extra:
            sys.stderr.write(
                f"[FATAL] scene_{pos:02d}.md uses placeholder(s) belonging to other scenes: {extra[:12]}.\n"
            )
            sys.exit(1)

        if not ANY_HEADING_RE.search(text):
            warnings.append(f"scene_{pos:02d}: 缺少场次 H2 标题行（## 场 N【标题】）")

        spliced = PLACEHOLDER_RE.sub(lambda m: verbatim[int(m.group(1))], text)
        spliced_texts.append(spliced)
        spliced_count += len(found)

    orphans = sorted(set(verbatim) - set(claimed))
    if orphans:
        sys.stderr.write(
            f"[FATAL] {len(orphans)} subtitle(s) were never woven into any scene: {orphans[:12]}. "
            "Coverage must be complete before the episode can be assembled.\n"
        )
        sys.exit(1)

    return spliced_texts, warnings, spliced_count


def _fmt_tc(value: Any) -> str:
    """Normalize a timecode string to HH:MM:SS ('' when unparseable)."""
    m = re.match(r"(\d{1,2}:\d{2}:\d{2})", str(value or "").strip())
    return m.group(1) if m else ""


def _slug_line_text(text: str) -> str:
    """First bold line of a scene with its full tail: `内景·日｜蛋糕店·烘焙间`."""
    m = SLUG_LINE_RE.search(text)
    if not m:
        return ""
    return (m.group(1) + m.group(2)).strip()


def _title_from_slug(slug_text: str) -> str:
    """Fallback scene title: the location part of a slug line."""
    s = TC_RE.sub("", slug_text).replace("`", "").strip()
    s = re.sub(r"^\[?TC[^\]]*\]?", "", s).strip()
    if "｜" in s or "|" in s:
        s = re.split(r"[｜|]", s, maxsplit=1)[1]
    else:
        m = re.match(r"^(?:内景|外景|片头曲|片尾曲|片尾|黑场)\s*[··・]?\s*(.+?)(?:\s*——.*)?$", s)
        s = m.group(1) if m else re.split(r"\s*——", s)[0]
    return s.strip(" ··—-")


def normalize_scene_text(text: str, pos: int, start_tc: str, end_tc: str) -> str:
    """Enforce the deliverable heading format: `## 场 N【标题】（起 - 止）`.

    The title is the writer's creative choice (fallback: slug location); the
    timecode paren is deterministic data injected from scene_manifest — the
    writer never types TCs and any TC written into the heading is replaced.
    Legacy '> 概要：' blockquotes are stripped (summaries live in the overview).
    """
    text = re.sub(r"^>\s*概要[：:][^\n]*\n?", "", text, flags=re.M)
    m = H2_TITLED_RE.search(text)
    title = m.group(1).strip() if m else ""
    draft_tc = (m.group(2) or "").strip() if m else ""
    if not title:
        slug_text = _slug_line_text(text)
        title = _title_from_slug(slug_text) if slug_text else ""
    if not title:
        title = "未命名"
    tc_pair = f"{start_tc} - {end_tc}" if start_tc and end_tc else draft_tc
    head = f"## 场 {pos}【{title}】（{tc_pair}）" if tc_pair else f"## 场 {pos}【{title}】"
    if HEADING_LINE_RE.search(text):
        return HEADING_LINE_RE.sub(lambda _: head, text, count=1)
    return head + "\n" + text.lstrip("\n")


def _speaker_is_known(name: str, allowed: set) -> bool:
    """True when a dialogue-head name traces back to an allowed source.

    Tolerates three benign decorations, never invents approvals:
      - trailing qualifier in parens: 「路人（男）」 -> 路人
      - voice-of pattern: 「王子的声音」 -> descriptive label (contract-legal)
      - bible alias prefix (>=2 chars): bible「菈菈」 admits 「菈菈与妈妈」
    """
    candidates = [name]
    stripped = PAREN_QUALIFIER_RE.sub("", name).strip()
    if stripped and stripped != name:
        candidates.append(stripped)
    for cand in candidates:
        if cand in allowed:
            return True
        if DESCRIPTIVE_SPEAKER_RE.match(cand):
            return True
        if any(len(a) >= 2 and cand.startswith(a) for a in allowed):
            return True
    return False


def lint_speaker_names(spliced_texts: List[str], allowed: set) -> List[str]:
    """Anti-hallucination guard: dialogue header names must be known. Warnings only.
    Heads like 「茉里、菈菈」 are split on 、/／ and every part is checked.
    Each scene's first bold line is its slug (`**黑场**`, `**内景·日**｜…`) and is
    never a dialogue head."""
    warnings: List[str] = []
    used = set()
    for text in spliced_texts:
        slug = SLUG_LINE_RE.search(text)
        slug_start = slug.start() if slug else -1
        for m in DIALOGUE_HEAD_RE.finditer(text):
            if m.start() == slug_start:
                continue
            for part in re.split(r"[、/／]", m.group(1).strip()):
                part = part.strip()
                if part:
                    used.add(part)
    unknown = sorted(n for n in used if not _speaker_is_known(n, allowed))
    if unknown:
        warnings.append(f"未登记的说话人名称（请核对是否杜撰）: {', '.join(unknown[:10])}")
    return warnings


def _cell(value: Any) -> str:
    """Markdown table cell: escape pipes so titles like 【A|B】 cannot break the row."""
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def build_overview(spliced_texts: List[str], sequence_titles: Optional[List[str]] = None) -> str:
    rows = ["| 场号 | 序列 | 时空 | 概要 | 时间码 |", "| :--- | :--- | :--- | :--- | :--- |"]
    for i, text in enumerate(spliced_texts, start=1):
        h2 = H2_TITLED_RE.search(text)
        title = h2.group(1).strip() if h2 else ""
        tc = (h2.group(2) or "").strip() if h2 else ""
        slug_text = _slug_line_text(text)
        slug_text = re.sub(r"\s*`\[TC[^\]]*\]`\s*$", "", slug_text)
        if not tc:
            legacy_tc = TC_RE.search(text)
            tc = legacy_tc.group(0) if legacy_tc else ""
        seq = ""
        if sequence_titles is not None and i - 1 < len(sequence_titles):
            seq = str(sequence_titles[i - 1] or "")
        rows.append(
            f"| 第 {i} 场 | {_cell(seq)} "
            f"| {_cell(slug_text)} "
            f"| {_cell(title)} "
            f"| {_cell(tc)} |"
        )
    return "\n".join(rows)


def av_statistics(aligned: Dict[str, Any]) -> List[Tuple[str, int, str]]:
    counts: Dict[str, int] = {}
    for shot in aligned.get("shots", []):
        for dlg in shot.get("dialogues", []):
            rel = dlg.get("av_relationship", "ON_SCREEN")
            counts[rel] = counts.get(rel, 0) + 1
    silent = sum(1 for s in aligned.get("shots", []) if s.get("shot_type") == "SILENT_ACTION")
    desc = {
        "ON_SCREEN": "说话者在画面内",
        "OFF_SCREEN": "说话者在画框外（O.S.）",
        "VOICE_OVER": "非剧情空间声源（旁白/解说）",
        "INTERNAL_MONOLOGUE": "内心独白/字幕卡",
        "REACTION_SHOT": "台词延续到他人反应镜头",
    }
    rows = [(rel, n, desc.get(rel, "")) for rel, n in sorted(counts.items(), key=lambda kv: -kv[1])]
    rows.append(("SILENT_ACTION（纯视听镜头）", silent, "无对白的关键视听节拍"))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Splice verbatim dialogue into scene texts and assemble the episode screenplay")
    parser.add_argument("--workspace", "-w", required=True, help="Project workspace root")
    parser.add_argument("--title", "-t", default=None, help="Episode title (default: workspace name)")
    parser.add_argument("--output", "-o", default=None, help="Output markdown path (default: output/<title>_影视文学剧本.md)")
    args = parser.parse_args()

    ws = Path(args.workspace).resolve()
    if not ws.is_dir():
        sys.stderr.write(f"[FATAL] Workspace directory does not exist: {ws}\n")
        sys.exit(1)

    manifest = load_json(ws / ".cache" / "alignment" / "scene_manifest.json")
    if not isinstance(manifest, dict) or not manifest.get("scenes"):
        sys.stderr.write("[FATAL] scene_manifest.json missing or empty. Run build_scene_manifest.py first.\n")
        sys.exit(1)

    extracted = load_json(ws / ".cache" / "subtitles" / "extracted.json")
    if not isinstance(extracted, dict) or not extracted.get("items"):
        sys.stderr.write("[FATAL] extracted.json missing or empty. Run the subtitle stage first.\n")
        sys.exit(1)
    verbatim = {int(it["index"]): str(it["text"]).strip() for it in extracted["items"]}

    total_scenes = int(manifest["total_scenes"])
    scene_files = collect_scene_files(ws / ".cache" / "scene_drafts", total_scenes)

    expected_by_scene: List[Dict[int, str]] = []
    for sc in manifest["scenes"]:
        expected = {
            int(d["sub_index"]): verbatim.get(int(d["sub_index"]), "")
            for d in sc.get("dialogues", [])
            if d.get("sub_index") is not None and int(d["sub_index"]) in verbatim
        }
        expected_by_scene.append(expected)

    allowed_names = set(GENERIC_SPEAKERS) | set(manifest.get("bible_names") or []) | set(
        manifest.get("characters_manifest") or {}
    )

    spliced_texts, warnings, spliced_count = validate_and_splice(scene_files, expected_by_scene, verbatim)
    tc_pairs = [(_fmt_tc(sc.get("start_timecode")), _fmt_tc(sc.get("end_timecode"))) for sc in manifest["scenes"]]
    spliced_texts = [
        normalize_scene_text(text, pos, start_tc, end_tc)
        for pos, (text, (start_tc, end_tc)) in enumerate(zip(spliced_texts, tc_pairs), start=1)
    ]
    warnings += lint_speaker_names(spliced_texts, allowed_names)

    title = args.title or ws.name
    safe_title = re.sub(r'[\\/*?:"<>|]', "_", title)
    out_dir = (ws / "output").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (ws / "output" / f"{safe_title}_影视文学剧本.md")
    try:
        out_path.resolve().relative_to(ws)
    except ValueError:
        sys.stderr.write("[FATAL] Output path escaped the workspace containment.\n")
        sys.exit(2)

    shots_doc = load_json(ws / ".cache" / "visual" / "shots.json") or {}
    aligned = load_json(ws / ".cache" / "alignment" / "aligned_timeline.json") or {}

    header = [
        f"# 《{title}》 场号制影视文学剧本",
        "",
        "| 项目 | 内容 |",
        "| :--- | :--- |",
        f"| **台词来源** | {extracted.get('source_detail', '')}，{len(verbatim)} 条，由占位符逐字回填 |",
        f"| **分镜来源** | {shots_doc.get('total_scenes', '?')} 个镜头，{total_scenes} 个宏场景（LGSS-DP + 关键帧色板亲和度） |",
        f"| **体例** | 中文场号制（H2 场头【标题】+ △ 视听动作段 + 同一说话人连续台词以／合并；台词一字未改） |",
        f"| **生成** | Video-to-Screenplay Pipeline v2（证据包 → LLM 写作 → 占位符回填校验） |",
        "",
        "---",
        "",
        "## 场次总表",
        "",
        build_overview(
            spliced_texts,
            [str(sc.get("sequence_title") or "") for sc in manifest["scenes"]],
        ),
        "",
        "---",
        "",
    ]

    appendix_rows = "\n".join(f"| {rel} | {n} | {d} |" for rel, n, d in av_statistics(aligned))
    appendix = (
        "\n---\n\n"
        "## 附：本集声画关系统计\n\n"
        "| 关系类型 | 条数 | 说明 |\n| :--- | :--- | :--- |\n"
        f"{appendix_rows}\n\n"
        f"> **台词保真说明**：全片 {len(verbatim)} 条字幕以 [[SUB:n]] 占位符由脚本从字幕轨逐字回填，"
        f"本次拼装 {spliced_count}/{len(verbatim)} 条，覆盖率 {spliced_count / max(1, len(verbatim)):.1%}；"
        "台词文本未经过任何改写。\n"
    )

    body = "\n---\n\n".join(part.strip() + "\n" for part in spliced_texts)
    document = "\n".join(header) + "\n" + body + appendix
    out_path.write_text(document, encoding="utf-8")

    for w in warnings:
        sys.stderr.write(f"[WARN] {w}\n")
    sys.stdout.write(json.dumps({
        "status": "screenplay_written",
        "output": str(out_path),
        "scenes": total_scenes,
        "dialogue_spliced": spliced_count,
        "dialogue_total": len(verbatim),
        "warnings": warnings,
    }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
