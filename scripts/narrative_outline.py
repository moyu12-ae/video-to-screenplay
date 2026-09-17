#!/usr/bin/env python3
"""
scripts/narrative_outline.py - Narrative Outline Work-order & Validator (Phase 3.5)

McKee layer of the segmentation hierarchy: sequences (and optionally acts) are
NARRATIVE units defined by value shifts, not by location. Beat builds scene,
scenes build a sequence (2-5 scenes, escalating impact, ending on a sequence
climax), sequences build an act. A scene whose value charge is unchanged from
start to end is a non-event and belongs to its neighbour.

Deterministic responsibilities (LLM comprehension stays in the agent):

  1. PREPARE - emit .cache/alignment/narrative_workorder.json: the full dialogue
     stream (sub_index + timecode + text), the outline contract, and the target
     schema for .cache/alignment/narrative_structure.json.
  2. CHECK   - validate the agent-authored structure file: sequences must
     partition the subtitle stream (contiguous, ascending, no gaps/overlaps)
     and carry title + value_from/value_to; optional acts must partition the
     sequence list. Prints the time spans the scene grouper consumes.

The outline is ADVISORY to the scene DP (sequence walls + per-sequence solving);
audio-visual evidence still decides the fine cuts inside each sequence.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

OUTLINE_CONTRACT = """\
## 叙事大纲合同（McKee 层级：节拍→场景→序列→幕）

序列是叙事单位，不是地点单位：一个序列是一组场景（典型 2-5 场），冲击力逐场递增，
收在一个序列高潮上。给每个序列起一个题目（为的是明确它为什么必须在片内），
并写明押上台面的价值从什么转向什么（如 希望→绝望、误解→和解、受挫→重燃）。
切分边界应落在转折点上；从开头到结尾价值毫无变化的段是"非事件"，并入相邻序列。
单集（约 24 分钟）通常 4-8 个序列；只有更长的影片才需要更粗的幕层（acts 可选，
按内容决定层数，不为层级而层级）。

写法：start_sub = 序列第一条台词的 sub_index，end_sub = 最后一条；序列按序无缝
覆盖全部台词（下一条的 start_sub = 上一条的 end_sub + 1）。无对白段落（OP/ED/
动作节拍）按时间自动归入相邻序列，无需也不得单独成序列。

输出到 .cache/alignment/narrative_structure.json：
{
  "sequences": [
    {"seq_index": 1, "title": "序列题目", "value_from": "正面/负面或具体价值",
     "value_to": "…", "start_sub": 1, "end_sub": 60, "evidence": "一句话依据"}
 ],
  "acts": [{"act_index": 1, "title": "…", "start_seq": 1, "end_seq": 3}]
}
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


def format_tc(ms: int) -> str:
    total_sec = int(ms) // 1000
    return f"{total_sec // 3600:02d}:{(total_sec // 60) % 60:02d}:{total_sec % 60:02d}"


def build_workorder(ws: Path, items: List[Dict[str, Any]]) -> Path:
    out_dir = (ws / ".cache" / "alignment").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "narrative_workorder.json"
    try:
        out_path.resolve().relative_to(ws.resolve())
    except ValueError:
        sys.stderr.write("[FATAL] Work-order path escaped workspace containment.\n")
        sys.exit(2)
    subs = [
        {
            "sub_index": int(it["index"]),
            "tc": f"{format_tc(it.get('start_ms', 0))}-{format_tc(it.get('end_ms', 0))}",
            "text": str(it.get("text", "")).strip(),
        }
        for it in items
    ]
    doc = {
        "version": 1,
        "contract": OUTLINE_CONTRACT,
        "target": ".cache/alignment/narrative_structure.json",
        "schema": {
            "sequences": ["seq_index", "title", "value_from", "value_to", "start_sub", "end_sub", "evidence"],
            "acts (optional)": ["act_index", "title", "start_seq", "end_seq"],
        },
        "subtitles": subs,
    }
    out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def validate_structure(ws: Path, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    path = (ws / ".cache" / "alignment" / "narrative_structure.json").resolve()
    doc = load_json(path)
    if not isinstance(doc, dict) or not doc.get("sequences"):
        sys.stderr.write(
            "[PENDING] narrative_structure.json missing or empty. Read narrative_workorder.json, "
            "author the outline per its contract, then re-run this script.\n"
        )
        sys.exit(6)

    items = sorted(items, key=lambda x: int(x["index"]))
    by_index = {int(it["index"]): it for it in items}
    min_idx, max_idx = int(items[0]["index"]), int(items[-1]["index"])

    seqs = doc["sequences"]
    for pos, seq in enumerate(seqs, start=1):
        for field in ("title", "value_from", "value_to"):
            if not str(seq.get(field, "")).strip():
                sys.stderr.write(f"[FATAL] sequence {pos}: missing {field}.\n")
                sys.exit(1)
        if int(seq.get("seq_index", -1)) != pos:
            sys.stderr.write(f"[FATAL] sequence {pos}: seq_index must be 1..N in order.\n")
            sys.exit(1)
        s, e = int(seq.get("start_sub", -1)), int(seq.get("end_sub", -1))
        if s not in by_index or e not in by_index or e < s:
            sys.stderr.write(f"[FATAL] sequence {pos}: bad sub range [{s}, {e}].\n")
            sys.exit(1)
        expected_start = min_idx if pos == 1 else int(seqs[pos - 2]["end_sub"]) + 1
        if s != expected_start:
            sys.stderr.write(
                f"[FATAL] sequence {pos}: start_sub {s} breaks contiguity (expected {expected_start}). "
                "Sequences must partition the subtitle stream without gaps or overlaps.\n"
            )
            sys.exit(1)
    if int(seqs[-1]["end_sub"]) != max_idx:
        sys.stderr.write(
            f"[FATAL] last sequence ends at sub {seqs[-1]['end_sub']} but stream ends at {max_idx}.\n"
        )
        sys.exit(1)

    acts = doc.get("acts") or []
    if acts:
        n_seqs = len(seqs)
        cursor = 1
        for act in acts:
            s, e = int(act.get("start_seq", -1)), int(act.get("end_seq", -1))
            if s != cursor or e < s or e > n_seqs:
                sys.stderr.write(
                    f"[FATAL] act '{act.get('title', '')}': [{s}, {e}] breaks sequence coverage 1..{n_seqs}.\n"
                )
                sys.exit(1)
            cursor = e + 1
        if cursor != n_seqs + 1:
            sys.stderr.write(f"[FATAL] acts stop at sequence {cursor - 1}; coverage must reach {n_seqs}.\n")
            sys.exit(1)

    summary_seqs = []
    for pos, seq in enumerate(seqs, start=1):
        s_ms = int(by_index[int(seq["start_sub"])]["start_ms"])
        e_ms = int(by_index[int(seq["end_sub"])]["end_ms"])
        wall = ""
        if pos < len(seqs):
            nxt = by_index[int(seqs[pos]["start_sub"])]
            wall = format_tc((e_ms + int(nxt["start_ms"])) // 2)
        summary_seqs.append({
            "seq_index": pos,
            "title": str(seq["title"]).strip(),
            "value": f"{seq['value_from']} -> {seq['value_to']}",
            "sub_range": [int(seq["start_sub"]), int(seq["end_sub"])],
            "time": f"{format_tc(s_ms)}-{format_tc(e_ms)}",
            "wall_to_next": wall,
        })
    return {"status": "narrative_structure_valid", "sequences": summary_seqs, "acts": len(acts)}


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3.5 narrative outline: emit the agent work-order, validate the authored structure"
    )
    parser.add_argument("--workspace", "-w", required=True, help="Project workspace root")
    args = parser.parse_args()

    ws = Path(args.workspace).resolve()
    if not ws.is_dir():
        sys.stderr.write(f"[FATAL] Workspace directory does not exist: {ws}\n")
        sys.exit(1)

    extracted = load_json(ws / ".cache" / "subtitles" / "extracted.json")
    if not isinstance(extracted, dict) or not extracted.get("items"):
        sys.stderr.write("[FATAL] extracted.json missing or empty. Run the subtitle stage first.\n")
        sys.exit(1)

    workorder = build_workorder(ws, extracted["items"])
    result: Dict[str, Any] = {"workorder": str(workorder), "subtitles": len(extracted["items"])}

    check = validate_structure(ws, extracted["items"])
    result.update(check)
    sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
