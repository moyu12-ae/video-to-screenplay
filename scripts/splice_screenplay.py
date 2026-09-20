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

import cast_signoff
import series

EXIT_NAMING_VIOLATION = 9  # a name-shaped speaker label traced to nothing while a
#                              signed-off cast table is in force (v0.6 P1)

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

# The label taxonomy lives in one module so this lint and resolve_cast.py cannot
# disagree about what "looks like a proper noun".
import speaker_labels as labels  # noqa: E402

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

        reversed_pairs = [(a, b) for a, b in zip(found, found[1:]) if b < a]
        if reversed_pairs:
            # SKILL.md states the placeholders must appear 按序; only set membership was
            # checked, so a draft could reorder dialogue and the screenplay would read
            # out of chronological sequence while every check stayed green.
            sys.stderr.write(
                f"[FATAL] scene_{pos:02d}.md lists {len(reversed_pairs)} placeholder(s) out of subtitle "
                f"order (e.g. [[SUB:{reversed_pairs[0][0]}]] before [[SUB:{reversed_pairs[0][1]}]]). "
                "Dialogue must appear in subtitle-index (chronological) order within a scene.\n")
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


def cast_lineage_names(cast_doc: Optional[Dict[str, Any]]) -> Optional[List[str]]:
    """Names a HUMAN attached to a specific speaking cluster, plus their aliases.

    The difference from "names present in the cast table" is the whole point:
    茉里 can be a perfectly well-signed-off character while the cluster that spoke
    「茉里 你交朋友了」 was never signed off as her - the ep02 accident reads exactly
    like a legitimate name. Only lineage stops it, so the enforced allow-list is
    built from cluster assignments, not from the entity list.
    """
    if not isinstance(cast_doc, dict):
        return None
    entities = {str(e.get("id")): e for e in (cast_doc.get("entities") or [])
                if isinstance(e, dict)}
    slots = {str(s.get("slot_id")): s for s in (cast_doc.get("slots") or [])
             if isinstance(s, dict)}
    out: List[str] = []
    for cluster in cast_doc.get("clusters") or []:
        if not isinstance(cluster, dict):
            continue
        assignment = cluster.get("assignment") or {}
        if str(assignment.get("status")) != "approved":
            continue
        slot = slots.get(str(assignment.get("slot_id"))) or {}
        entity = entities.get(str(assignment.get("entity_id") or slot.get("entity_id"))) or {}
        name = str(entity.get("canonical_name") or assignment.get("entity_name") or "").strip()
        if not name:
            continue
        for surface in [name] + [str(a).strip() for a in (entity.get("aliases") or [])]:
            if surface and surface not in out:
                out.append(surface)
    return out


def _dialogue_heads(text: str) -> List[str]:
    """Every dialogue head in one scene, scene slug excluded.

    A head like 「**菈菈、陌生旅人**（躬身）：你好」 names more than one speaker, so
    each part is judged on its own.
    """
    heads: List[str] = []
    slug = SLUG_LINE_RE.search(text)
    slug_start = slug.start() if slug else -1
    for m in DIALOGUE_HEAD_RE.finditer(text):
        if m.start() == slug_start:
            continue
        heads.append(m.group(1).strip())
    return heads


def audit_speaker_labels(spliced_texts: List[str], allowed: set) -> Dict[str, List[str]]:
    """Classify every speaker label in the delivered text.

    Returns {"named": traced to the cast/bible, "descriptive": shape says it is a
    role/voice description, "untraced": name-shaped but traceable to nothing}.
    `untraced` is the only fatal-eligible bucket, and `descriptive` is the one the
    fidelity report must keep counting: a run that leans on descriptions is not
    a failure, it is an unfinished cast table, and pretending otherwise is what
    pushed writers towards longer labels rather than better evidence.
    """
    buckets: Dict[str, List[str]] = {"named": [], "descriptive": [], "untraced": []}
    for text in spliced_texts:
        for head in _dialogue_heads(text):
            verdict = labels.audit_head(head, allowed)
            for part in verdict["descriptive"]:
                if part not in buckets["descriptive"]:
                    buckets["descriptive"].append(part)
            for part in verdict["traced"]:
                if part not in buckets["named"]:
                    buckets["named"].append(part)
            for part in verdict["untraced"]:
                if part not in buckets["untraced"]:
                    buckets["untraced"].append(part)
    return buckets


def lint_speaker_names(spliced_texts: List[str], allowed: set,
                       cast_enforced: bool = False) -> List[str]:
    """Anti-hallucination guard, reversed direction (v0.6 P1).

    Before v0.6 a label passed only if it matched one narrow suffix pattern, so
    「女声」「系统音」「关西腔者」 were flagged while 「威严的声音」 sailed through -
    the guard punished short honest descriptions and rewarded long ones. Now a
    label is accepted when its SHAPE is descriptive, and only name-shaped labels
    have to trace back to a known surface. Without a cast table there is nothing
    to trace against, so this stays a warning that says why; with one, the caller
    escalates to a fatal exit.
    """
    buckets = audit_speaker_labels(spliced_texts, allowed)
    untraced = buckets["untraced"]
    if not untraced:
        return []
    listed = ", ".join(untraced[:10])
    if cast_enforced:
        return [f"表外专名说话人（演员表已生效，必须是这些之一）: {listed}"]
    return [f"未登记的说话人名称（请核对是否杜撰）: {listed}"]


OVERRIDE_COMMENT_RE = re.compile(r"<!--\s*attribution-override:\s*(.*?)-->", re.S)
# A line number must not be preceded by a letter, so the 2 inside 「C2」 can never
# waive a violation against line 2; cluster tokens are letter-prefixed by contrast.
_STANDALONE_NUM_RE = re.compile(r"(?<![A-Za-z])\d+")
_CLUSTER_TOKEN_RE = re.compile(r"[A-Za-z_]+\d+")


def _norm_cluster_token(token: str) -> str:
    return token.strip().upper().replace("SPEAKER_", "")


def parse_overrides(text: str) -> List[Dict[str, Any]]:
    """Well-formed `<!-- attribution-override: ... -->` comments in one scene draft.

    A comment covers line n when n appears as a standalone number (「8,9」 yes, the
    2 inside 「C2」 no) and covers a cluster when its id appears as a letter token
    (「A1」 matches SPEAKER_A1). A comment that parses to neither covers nothing,
    so a sloppy note can never silently waive a violation.
    """
    out: List[Dict[str, Any]] = []
    for m in OVERRIDE_COMMENT_RE.finditer(text):
        body = m.group(1)
        out.append({
            "lines": {int(x) for x in _STANDALONE_NUM_RE.findall(body)},
            "clusters": {_norm_cluster_token(t) for t in _CLUSTER_TOKEN_RE.findall(body)},
            "text": body.strip(),
        })
    return out


def _covers(override: Dict[str, Any], sub_index: int, cluster_id: str) -> bool:
    return sub_index in override["lines"] \
        or _norm_cluster_token(cluster_id) in override["clusters"]


def _sub_index_heads(text: str) -> Dict[int, str]:
    """Map each [[SUB:n]] to the dialogue head it sits under.

    A placeholder belongs to the nearest preceding head - the writing contract
    puts dialogue on the head's own line, so anything before the first head has
    no speaker to contradict and is skipped.
    """
    events: List[Tuple[int, str, Any]] = []
    slug = SLUG_LINE_RE.search(text)
    for m in DIALOGUE_HEAD_RE.finditer(text):
        if slug and m.start() == slug.start():
            continue
        events.append((m.start(), "head", m.group(1).strip()))
    for m in PLACEHOLDER_RE.finditer(text):
        events.append((m.start(), "sub", int(m.group(1))))
    events.sort(key=lambda e: e[0])
    out: Dict[int, str] = {}
    head: Optional[str] = None
    for _, kind, val in events:
        if kind == "head":
            head = val
        elif head is not None:
            out[int(val)] = head
    return out


def audit_attribution(draft_texts: List[str], dialogues_by_scene: List[List[Dict[str, Any]]],
                      cast_doc: Dict[str, Any]) -> Tuple[List[str], int]:
    """§6 rows 5-6: the draft must not quietly overrule the resolver.

    The lineage gate can only say a label traces to SOME signed entity; it cannot
    see that lines 8-9 sit under a head naming an entity OTHER than the one the
    line's own cluster was signed to - the ep02 accident, wearing a table. For
    each dialogue line, the head's name-shaped parts must name the line's own
    cluster entity. Two violations:

      - the cluster has an entity and the head names a different one (改判, row 5)
      - the cluster has no entity and the head names anyone at all (row 6)

    Both are waivable only by an in-scene attribution-override comment covering
    that line or its cluster: overtaking the machine is allowed, doing it
    invisibly is not. Descriptive parts re-attribute nothing. Returns
    (fatal violation strings, override comment count for the fidelity report).
    """
    entities = {str(e.get("id")): e for e in (cast_doc.get("entities") or []) if isinstance(e, dict)}
    cluster_entity: Dict[str, str] = {}
    surface_entity: Dict[str, str] = {}
    known_clusters: set = set()
    for c in cast_doc.get("clusters") or []:
        if isinstance(c, dict) and c.get("cluster_id"):
            known_clusters.add(str(c["cluster_id"]))
        assignment = c.get("assignment") if isinstance(c, dict) else None
        entity = entities.get(str((assignment or {}).get("entity_id") or "")) or {}
        name = str(entity.get("canonical_name") or (assignment or {}).get("entity_name") or "").strip()
        if isinstance(c, dict) and str((assignment or {}).get("status")) == "approved" and name:
            cluster_entity[str(c["cluster_id"])] = name
            for surface in [name] + [str(a).strip() for a in (entity.get("aliases") or [])]:
                if surface:
                    surface_entity[surface] = name
    overrides = [o for text in draft_texts for o in parse_overrides(text)]
    violations: List[str] = []
    for text, dialogues in zip(draft_texts, dialogues_by_scene):
        sub_heads = _sub_index_heads(text)
        for d in dialogues or []:
            if not isinstance(d, dict):
                continue
            speaker = str(d.get("speaker") or "").strip()
            raw = d.get("sub_index")
            if not speaker or speaker == "SPEAKER_UNKNOWN" or raw is None:
                continue
            if speaker not in known_clusters:
                continue  # a stale cast.json must not invent violations
            sub_index = int(raw)
            head = sub_heads.get(sub_index)
            if not head:
                continue
            parts = [p for p in labels.split_head(head) if labels.looks_like_name(p)]
            if not parts:
                continue
            expected = cluster_entity.get(speaker)
            traced = {}
            for part in parts:
                surface = labels.trace(part, surface_entity.keys())
                if surface:
                    traced[part] = surface_entity[surface]
            if not traced:
                continue  # nothing traces: that is the lineage gate's job, not re-attribution
            if expected is not None and expected in set(traced.values()):
                continue  # the head names this line's own entity (any of its surfaces)
            if any(_covers(o, sub_index, speaker) for o in overrides):
                continue
            expected_desc = f"演员表记为「{expected}」" if expected else "演员表中无定名"
            violations.append(
                f"第 {sub_index} 条台词说话人被写成 {'、'.join(sorted(traced))}，"
                f"但该簇{expected_desc}（簇 {speaker}）——如确要改判，须在本场景草稿内留痕："
                "<!-- attribution-override: ... -->")
    return sorted(dict.fromkeys(violations)), len(overrides)


def summarize_cast(cast_doc: Dict[str, Any]) -> Dict[str, Any]:
    """named / candidate / unknown per cluster, plus the resolver's own caveats."""
    statuses = [str((c.get("assignment") or {}).get("status") or "unknown")
                for c in (cast_doc.get("clusters") or []) if isinstance(c, dict)]
    gates = cast_doc.get("gates") or {}
    abstention = cast_doc.get("abstention") or {}
    return {
        "named": statuses.count("approved"),
        "candidate": statuses.count("candidate"),
        "unknown": statuses.count("unknown"),
        "abstention_rate": abstention.get("rate"),
        "over_split_suspects": len(cast_doc.get("over_split_suspects") or []),
        "thresholds_are_measured": bool(gates.get("thresholds_are_measured")),
        "cast_version": cast_doc.get("table_version") or cast_doc.get("version"),
    }


def fidelity_lines(appendix_rows: str, dialogue_total: int, spliced_count: int,
                   cast_summary: Optional[Dict[str, Any]],
                   label_buckets: Dict[str, List[str]], cast_enforced: bool,
                   override_count: int = 0) -> List[str]:
    """The fidelity appendix, one paragraph per list item.

    Every line here must survive regardless of whether a cast.json exists: the
    first version of this block used a conditional inside a string concatenation
    and silently dropped the label-composition line on exactly the runs where a
    cast table WAS present - the runs that most needed it.
    """
    lines = [
        "\n---\n\n"
        "## 附：本集声画关系统计\n\n"
        "| 关系类型 | 条数 | 说明 |\n| :--- | :--- | :--- |\n"
        f"{appendix_rows}\n\n"
        f"> **台词保真说明**：全片 {dialogue_total} 条字幕以 [[SUB:n]] 占位符由脚本从字幕轨逐字回填，"
        f"本次拼装 {spliced_count}/{dialogue_total} 条，"
        f"覆盖率 {spliced_count / max(1, dialogue_total):.1%}；台词文本未经过任何改写。\n",
    ]
    if cast_summary:
        cast_line = (">\n> **演员表状态**：已定名 {named} 簇、候选 {candidate} 簇、"
                     "未定名 {unknown} 簇").format(named=cast_summary["named"],
                                                  candidate=cast_summary["candidate"],
                                                  unknown=cast_summary["unknown"])
        if isinstance(cast_summary.get("abstention_rate"), float):
            cast_line += f"，弃权率 {cast_summary['abstention_rate']:.0%}"
        if cast_summary["over_split_suspects"]:
            cast_line += f"，疑似过度切分 {cast_summary['over_split_suspects']} 组"
        cast_line += ("；判定阈值**尚未经测量**。" if not cast_summary["thresholds_are_measured"]
                      else "；阈值来自测量。")
        cast_line += f"（表版本 {cast_summary['cast_version'] or '未签核'}）\n"
        lines.append(cast_line)
    lines.append(
        ">\n> **说话人标签构成**：可溯源 {named} 个、描述性（未定名）{descriptive} 个、"
        "表外专名 {untraced} 个".format(named=len(label_buckets["named"]),
                                        descriptive=len(label_buckets["descriptive"]),
                                        untraced=len(label_buckets["untraced"]))
        + (f"；留痕改判 {override_count} 处（attribution-override）" if override_count else "")
        + ("；演员表已生效，表外专名为致命" if cast_enforced
           else "；本工作区未绑定已签核演员表，故仅告警（成稿按未核验处理）") + "\n")
    return lines


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

    # v0.6: the approved cast table is a naming source like bible names, and its
    # presence is also what makes an untraceable name fatal instead of advisory -
    # without a table there is nothing to trace against, so the guard says so and
    # lets --draft output proceed (a gate nobody can satisfy gets routed around).
    cast_entities = [e for e in (series.load_approved(ws).get("entities") or [])
                     if isinstance(e, dict) and str(e.get("status") or "").lower() == "approved"]
    cast_enforced = bool(cast_entities)
    cast_doc = load_json(ws / ".cache" / "cast" / "cast.json")
    lineage = cast_lineage_names(cast_doc)
    # Once the resolver has run there IS a lineage record to check, so bible and
    # manifest names stop being a licence: they predate the table and cannot say
    # which cluster they belong to.
    allowed_names = (
        set(labels.GENERIC_SPEAKERS) | set(lineage)
        if cast_doc and lineage is not None
        else set(labels.GENERIC_SPEAKERS) | set(manifest.get("bible_names") or [])
        | set(manifest.get("characters_manifest") or {}) | set(series.approved_names(ws)))

    spliced_texts, warnings, spliced_count = validate_and_splice(scene_files, expected_by_scene, verbatim)
    tc_pairs = [(_fmt_tc(sc.get("start_timecode")), _fmt_tc(sc.get("end_timecode"))) for sc in manifest["scenes"]]
    spliced_texts = [
        normalize_scene_text(text, pos, start_tc, end_tc)
        for pos, (text, (start_tc, end_tc)) in enumerate(zip(spliced_texts, tc_pairs), start=1)
    ]
    naming_warnings = lint_speaker_names(spliced_texts, allowed_names, cast_enforced=cast_enforced)
    warnings += naming_warnings
    label_buckets = audit_speaker_labels(spliced_texts, allowed_names)
    # The machine's own uncertainty belongs in the deliverable, not just in a
    # scratch file: an episode where the resolver abstained on most speakers is not
    # "done, with warnings", and the reader needs to see that before trusting a
    # name that appears in the text.
    cast_summary = summarize_cast(cast_doc) if isinstance(cast_doc, dict) else None
    # --draft semantics: shipping without sign-off is allowed, being quiet about
    # it is not. The notice sits in the deliverable's own header table, because an
    # appendix note is the first thing a reader skips.
    cast_notice = cast_signoff.cmd_draft_banner(cast_doc, cast_enforced)

    # With a signed table in force, naming violations are not warnings: the lint
    # of the ep02 era was pointed at and ignored precisely because nothing stopped
    # the run. This exits non-zero BEFORE the deliverable is written - and the
    # regression test drives the CLI, not the lint function, because 5319ab4
    # showed a function-level test cannot pin a process behaviour.
    fatal_naming = list(naming_warnings) if cast_enforced else []
    override_count = 0
    if cast_enforced and isinstance(cast_doc, dict):
        drafts = [p.read_text(encoding="utf-8") for p in scene_files]
        scene_dialogues = [sc.get("dialogues") or [] for sc in manifest["scenes"]]
        attribution_violations, override_count = audit_attribution(drafts, scene_dialogues, cast_doc)
        fatal_naming += attribution_violations
    if fatal_naming:
        for line in fatal_naming:
            sys.stderr.write(f"[FATAL] {line}\n")
        sys.stderr.write("[ACTION] 回到阶段 3.8 签核，或在场景草稿内留痕改判"
                         "（<!-- attribution-override: ... -->），或解除系列绑定走 --draft 语义。\n")
        sys.exit(EXIT_NAMING_VIOLATION)


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
        "| **体例** | 中文场号制（H2 场头【标题】+ △ 视听动作段 + 同一说话人连续台词以／合并；台词一字未改） |",
        ("| **演员表** | ⚠️ " + cast_notice + " |") if cast_notice else
        "| **演员表** | 已签核"
        + (f"（表版本 {cast_summary['cast_version']}）" if cast_summary else "") + " |",
        "| **生成** | Video-to-Screenplay Pipeline v2（证据包 → LLM 写作 → 占位符回填校验） |",
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
    # Assembled from a list, not a conditional expression: `A + B if cond else C + D`
    # parses as `(A + B) if cond else (C + D)`, which silently deleted the label
    # composition line whenever a cast.json existed.
    appendix = "".join(fidelity_lines(appendix_rows, len(verbatim), spliced_count,
                                      cast_summary, label_buckets, cast_enforced,
                                      override_count))

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
