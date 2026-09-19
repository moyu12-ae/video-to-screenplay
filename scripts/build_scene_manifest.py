#!/usr/bin/env python3
"""
scripts/build_scene_manifest.py - Scene Evidence Pack & Writing Contract Builder

Bridges the deterministic pipeline and the LLM screenplay-writing pass.

Philosophy (post-v2 redesign): the LLM should WRITE the screenplay, not fill a
form. Python owns the ground truth; the model owns the prose:

  1. This script reads macro scenes, keyframes, the aligned timeline and the
     bible, then emits .cache/alignment/scene_manifest.json - a per-scene
     evidence pack: downscaled keyframe copies to Read, the scene's verbatim
     dialogue list (sub_index + timecodes), exemplar screenplay paths, and the
     writing contract.
  2. The agent Reads the evidence per scene and writes .cache/scene_drafts/
     scene_XX.md - the scene's actual 场号制 screenplay text, weaving △ action
     between dialogue blocks. Dialogue lines are written as [[SUB:n]]
     placeholders (n = the subtitle index), NEVER retyped.
  3. splice_screenplay.py splices the verbatim subtitle text into the
     placeholders (100% fidelity by construction), validates coverage and
     naming, and assembles the final episode document.

All writes stay strictly inside the resolved workspace (.cache/...), enforced
by containment checks before any file is touched.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import op_ed

THUMB_WIDTH = 640
# Kept in sync with av_understand.AV_NOTES_SCHEMA by a test (importing that module
# here would drag the network client into an offline stdlib-only stage).
EXPECTED_AV_NOTES_SCHEMA = "vts-av-notes/v2"

WRITING_CONTRACT = """\
## 写作合同（每场一个 scene_XX.md）

写「真正的剧本」，不是填表。成品版式以样张（exemplars，第 1/2 集剧本）为准，每场产出：

```
## 场 N【地点·事件标题】
**内景/外景·时辰**｜地点或注记

**人物：** 出场角色列表

△ 定场动作段（人物初登场写 △【角色名】（外观特征）……）

**角色名**（表演提示）：[[SUB:2]]／[[SUB:3]]／[[SUB:4]]

△ 织在台词之间的导演笔记……

**另一角色**（画外）：[[SUB:5]]
```

硬性规则：
1. 场头两行由你起笔：H2 标题 = 地点+事件（6~14 字，如【海面·燃烧的帆船】）；
   第二行 = **内景/外景·时辰**｜注记。不要写时间码、不要写「> 概要：」引用行——
   时间码由拼装器按清单自动注入 H2 行尾，概要只活在场次总表里。
2. 台词一律写 [[SUB:n]] 占位符（n 见本清单 dialogues[].sub_index），绝不抄写/改写台词原文。
   占位符每条恰好用一次，按时间顺序。同一角色连续说话时合并为一行，句间用全角斜杠「／」：
   `**乳母**（躬身）：[[SUB:2]]／[[SUB:3]]／[[SUB:4]]`；说话人切换才另起一行。
   画外/独白写在名字后的括号里：`**旁白**（画外）：[[SUB:1]]`、`**小满**（内心独白）：…`。
   字幕卡/招牌文字可内联：`（字幕卡：[[SUB:4]]）`。
3. △ 动作段是导演笔记，不是场景说明书。三类用途：演出指示（一句话该怎么说）、
   表演指导（表情与小动作）、镜头强调（特写/缓推/骤白/声画对位）。
   要有画面：写光、写物、写身体、写声音。基准密度——场首必有一段定场 △；
   台词的情绪转折处必须插一拍 △；超过 1.5 秒的无声节拍单独立 △ 并标时长。
   样张水准示例（照这个画质写，不封顶）：
   △ 铜镜只剩一点将熄的微光。云铮的音容浮现在光晕里。
   △ 小满双手按上镜面——镜的另一侧，一只涂着黑甲的手（影卫）与她隔镜相抵。青金的光纹在镜面炸开。
   禁止「两人交谈」「气氛尴尬」这类空泛句。
   若本场带 av_notes（声画理解证据）：△ 优先取材其中真实可见的动作/运镜/音效/
   屏显文字（时间码对得上的优先），关键帧管构图与人物外观；av_notes 里的描述
   是目击证据，绝不当台词写进正文，其中人物一律是描述性称呼、以台词与
   characters_manifest 的称呼为准。
4. 角色名只能来自 bible.characters / characters_manifest / 本场 dialogues 里出现过的称呼；
   认不出的人用描述性指称（路人（男）/神秘人物），绝不发明专有名词。
5. 视听绝对性：**目击者原则**——只写画面可见、耳朵可闻的内容；禁止心理描写与回忆
   （不写"她想起五年前……心中涌起悔恨"，写"她动作僵住，死死攥住拳头，指甲几乎
   陷进肉里，下颌肌肉紧绷"——演员能演、摄影机能拍）。**Notnot 原则**——银幕呈现
   "有什么"而非"没什么"：禁止"他没有回答/什么也没发生/房间里一个人也没有"式
   否定句，写"他面无表情地看着对方，保持沉默。空荡荡的房间唯有座钟摆动"。
6. 闪回/插入写 △【插入：…】/△【闪回：…】；字幕卡写（字幕卡：…）。
   draft_status == "op_ed" 的场次已由脚本写好单行片头/片尾标注，不要动它；
   若某场带 op_ed 标注却仍有 dialogues（台词恰好压在窗口边界上存活下来），
   照常织入这些 [[SUB:n]]，并在场头下方单起一行写（动画 OP）/（动画 ED）。
7. 叙事上下文：本清单若带 sequence_title/sequence_value（所属序列及其价值弧，
   如「误会与和解：负面 -> 正面」），先读它——场标题与 △ 要服务这条价值弧；
   同一序列内的相邻场要写出递进，禁止写成平行复读。
8. 开写前：Read 本场全部 keyframes_thumbs（一张不落）、概览 dialogues、（若存在）
   上一场 scene_(N-1).md 的结尾以保持连贯；人物造型拿不准时查 exemplar 范例剧本的人物表。
"""


def load_json(path: Path) -> Optional[Any]:
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        sys.stderr.write(f"[WARN] Failed to load {path}: {e}\n")
        return None


def sample_keyframes(
    shots: List[Dict[str, Any]],
    start_ms: int,
    end_ms: int,
    keyframes_dir: Path,
    limit: int,
) -> List[str]:
    """Evenly sample up to `limit` keyframe abs paths of the child shots inside [start_ms, end_ms]."""
    inside = [
        s for s in shots
        if int(s.get("start_ms", 0)) >= start_ms and int(s.get("start_ms", 0)) < end_ms
    ]
    if not inside:
        inside = [
            s for s in shots
            if int(s.get("end_ms", 0)) > start_ms and int(s.get("start_ms", 0)) < end_ms
        ]
    if not inside:
        return []
    if len(inside) > limit:
        step = len(inside) / float(limit)
        inside = [inside[int(i * step)] for i in range(limit)]
    paths: List[str] = []
    for s in inside:
        fname = str(s.get("keyframe") or "")
        if not fname:
            continue
        p = keyframes_dir / fname
        if p.is_file():
            paths.append(str(p))
    return paths


def make_thumbs(
    paths: List[str], out_dir: Path, width: int
) -> List[str]:
    """Downscaled copies for the writer's context budget. Missing cv2 -> return originals."""
    try:
        import cv2
    except ImportError:
        return paths
    out_dir.mkdir(parents=True, exist_ok=True)
    thumbs: List[str] = []
    for i, src in enumerate(paths):
        dst = out_dir / f"kf_{i:02d}.jpg"
        try:
            img = cv2.imread(src)
            if img is None:
                continue
            h, w = img.shape[:2]
            if w > width:
                img = cv2.resize(img, (width, int(h * width / w)))
            cv2.imwrite(str(dst), img)
            thumbs.append(str(dst))
        except Exception:
            continue
    return thumbs or paths


def detect_exemplars(ws: Path) -> List[str]:
    """Heuristic: previous episodes' screenplays are the best format exemplars."""
    found: List[str] = []
    parent = ws.parent
    try:
        for p in sorted(parent.glob("*.md")):
            name = p.name
            if ("剧本" in name) and (ws.name not in name):
                found.append(str(p))
    except OSError:
        pass
    return found[:2]


def build_manifest(ws: Path, max_keyframes: int) -> Dict[str, Any]:
    scenes_path = ws / ".cache" / "visual" / "scenes.json"
    shots_path = ws / ".cache" / "visual" / "shots.json"
    aligned_path = ws / ".cache" / "alignment" / "aligned_timeline.json"
    speakers_path = ws / ".cache" / "audio" / "speakers.json"
    keyframes_dir = ws / ".cache" / "visual" / "keyframes"
    thumbs_dir = ws / ".cache" / "scene_drafts" / "thumbs"
    drafts_dir = ws / ".cache" / "scene_drafts"

    macro_doc = load_json(scenes_path)
    macro_scenes: List[Dict[str, Any]] = (
        macro_doc.get("scenes", []) if isinstance(macro_doc, dict) else (macro_doc or [])
    )
    if not macro_scenes:
        sys.stderr.write(f"[FATAL] No macro scenes found in {scenes_path}. Run semantic_scene_grouper.py first.\n")
        sys.exit(1)

    shots_doc = load_json(shots_path)
    shots: List[Dict[str, Any]] = (
        shots_doc.get("scenes", []) if isinstance(shots_doc, dict) else (shots_doc or [])
    )

    aligned_doc = load_json(aligned_path)
    aligned_rows: List[Dict[str, Any]] = (
        aligned_doc.get("shots", []) if isinstance(aligned_doc, dict) else (aligned_doc or [])
    )
    dialogues_by_scene: Dict[str, List[Dict[str, Any]]] = {}
    for row in aligned_rows:
        sid = str(row.get("shot_id") or "")
        if sid:
            dialogues_by_scene[sid] = row.get("dialogues", []) or []

    speakers_doc = load_json(speakers_path)
    characters_manifest: Dict[str, Any] = {}
    if isinstance(speakers_doc, dict):
        characters_manifest = speakers_doc.get("characters_manifest", {}) or {}

    bible_doc = load_json(ws / "materials" / "bible.json") or {}
    bible_names: List[str] = []
    for c in bible_doc.get("characters", []):
        if isinstance(c, dict) and c.get("name"):
            bible_names.append(c["name"])
            bible_names.extend(a for a in (c.get("aliases") or []) if a)
    # Production-specific descriptive speaker labels (e.g. 「面试的店主」) belong
    # here, in the workspace's bible - never hardcoded in the plugin.
    bible_names.extend(str(w) for w in (bible_doc.get("speaker_whitelist") or []) if str(w).strip())

    exemplars = detect_exemplars(ws)
    if not exemplars:
        sys.stderr.write("[INFO] No exemplar screenplays found next to the workspace; writer relies on the style card only.\n")

    # Stage 3.7 evidence (optional): scene-grounded AV notes from av_understand.py
    av_doc = load_json(ws / ".cache" / "visual" / "av_notes.json") or {}
    if av_doc and av_doc.get("schema") != EXPECTED_AV_NOTES_SCHEMA:
        sys.stderr.write(f"[WARN] av_notes.json schema is {av_doc.get('schema')!r}, this build writes "
                         f"{EXPECTED_AV_NOTES_SCHEMA!r} - stale evidence may lack dedup/substance fields; "
                         "rerun av_understand.py merge.\n")
    av_by_scene = {str(n.get("scene_id")): n for n in (av_doc.get("scene_notes") or [])
                   if isinstance(n, dict) and n.get("scene_id")}
    if av_by_scene:
        sys.stderr.write(f"[INFO] AV understanding notes loaded for {len(av_by_scene)} scene(s); "
                         "they are WRITING EVIDENCE - never dialogue text.\n")
    else:
        sys.stderr.write("[INFO] No av_notes.json - writer falls back to keyframes-only evidence.\n")

    drafts_dir.mkdir(parents=True, exist_ok=True)

    # OP/ED windows (v0.5.1): scenes sitting inside a configured window are not
    # narrative - the builder writes their one-line stub itself, so the writer
    # never sees them and splice still gets a file per scene.
    op_ed_windows = op_ed.load_windows(ws)

    manifest_scenes: List[Dict[str, Any]] = []
    for sc in macro_scenes:
        sid = str(sc.get("scene_id") or "")
        idx = int(sc.get("macro_index") or len(manifest_scenes) + 1)
        start_ms = int(sc.get("start_ms", 0))
        end_ms = int(sc.get("end_ms", 0))
        draft_name = f"scene_{idx:02d}.md"
        draft_path = drafts_dir / draft_name
        originals = sample_keyframes(shots, start_ms, end_ms, keyframes_dir, max_keyframes)
        op_ed_label = op_ed.matching_label(start_ms, end_ms, op_ed_windows)
        scene_dialogues = dialogues_by_scene.get(sid, [])
        # A stub carries no [[SUB:n]], so it is only legal when the window filter left
        # this scene with nothing to say. A line straddling the window edge survives
        # (<=50% inside), and stubbing its scene would make splice FATAL on a
        # placeholder no writer was ever asked to weave.
        stub_eligible = bool(op_ed_label) and not scene_dialogues
        if stub_eligible and not draft_path.is_file():
            # The stub IS the deliverable for these scenes: one line in the final
            # screenplay saying what the window is, nothing for the writer to do.
            draft_path.write_text(f"## 场 {idx}【{op_ed_label}】\n\n（动画 {op_ed_label}——按配置略）\n",
                                  encoding="utf-8")
        elif op_ed_label and scene_dialogues:
            sys.stderr.write(
                f"[WARN] scene_{idx:02d} sits in the {op_ed_label} window but keeps "
                f"{len(scene_dialogues)} surviving dialogue line(s) - authored as a normal "
                "scene, because a stub cannot carry [[SUB:n]] placeholders.\n")
        manifest_scenes.append({
            "scene_index": idx,
            "scene_id": sid,
            "slugline_hint": sc.get("slugline", ""),
            "start_timecode": sc.get("start_timecode", ""),
            "end_timecode": sc.get("end_timecode", ""),
            "duration_ms": sc.get("duration_ms", max(0, end_ms - start_ms)),
            "sequence_index": sc.get("sequence_index"),
            "sequence_title": sc.get("sequence_title", ""),
            "sequence_value": sc.get("sequence_value", ""),
            "keyframes_thumbs": [] if stub_eligible else make_thumbs(
                originals, thumbs_dir / f"scene_{idx:02d}", THUMB_WIDTH),
            "av_notes": av_by_scene.get(sid),
            "op_ed": op_ed_label,
            "child_shot_count": sc.get("child_shot_count"),
            "dialogues": scene_dialogues,
            "draft_path": str(draft_path),
            "draft_status": ("op_ed" if stub_eligible else
                             ("written" if draft_path.is_file() else "missing")),
        })

    return {
        "version": 2,
        "workspace": str(ws),
        "drafts_dir": str(drafts_dir),
        "instructions": WRITING_CONTRACT,
        "bible_names": bible_names,
        "characters_manifest": characters_manifest,
        "exemplars": exemplars,
        "total_scenes": len(manifest_scenes),
        "scenes": manifest_scenes,
    }


def main():
    parser = argparse.ArgumentParser(description="Build the scene evidence pack + writing contract (chunked manifest)")
    parser.add_argument("--workspace", "-w", required=True, help="Project workspace root")
    parser.add_argument("--max-keyframes", type=int, default=12, help="Max keyframes sampled per scene (default: 12)")
    args = parser.parse_args()

    ws = Path(args.workspace).resolve()
    if not ws.is_dir():
        sys.stderr.write(f"[FATAL] Workspace directory does not exist: {ws}\n")
        sys.exit(1)

    manifest = build_manifest(ws, max(1, args.max_keyframes))

    # Containment: the manifest target must resolve inside the resolved workspace,
    # otherwise abort instead of writing (defense against malformed workspace args).
    out_path = (ws / ".cache" / "alignment" / "scene_manifest.json")
    if out_path.resolve().parent != ws / ".cache" / "alignment":
        sys.stderr.write(f"[FATAL] Manifest path escaped the workspace containment: {out_path}\n")
        sys.exit(2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    written = sum(1 for s in manifest["scenes"] if s["draft_status"] == "written")
    sys.stdout.write(json.dumps({
        "status": "manifest_written",
        "manifest": str(out_path),
        "total_scenes": manifest["total_scenes"],
        "drafts_written": written,
        "drafts_missing": manifest["total_scenes"] - written,
        "exemplars": manifest["exemplars"],
        "next_step": "Agent reads evidence per scene, writes scene_XX.md screenplay text, then runs splice_screenplay.py"
    }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
