#!/usr/bin/env python3
"""
scripts/av_understand.py - Scene-grounded Audio-Visual Understanding (Qwen3.8-Omni)

Stage 3.7, OPTIONAL: watch each macro scene (segments of <=90 s when a scene runs
longer) and produce structured evidence notes - ACTIONS / CAMERA / SOUND /
ON-SCREEN TEXT - that stage 4 weaves into real △ direction lines instead of
guessing motion from static keyframes. Mirrors speaker_diarize.py's workorder
pattern, but every step is script-driven (no agent-side MCP hop, no exit 6):

  prepare : read scenes.json, split long scenes, cut + transcode inline-friendly
            segment payloads (480p CRF ladder -> inline budget), write
            av_workorder.json, EXIT 0.
  run     : direct DashScope call per segment via omni_client (needs
            DASHSCOPE_API_KEY, exit 8 without it), per-segment resume, one
            failure never stops the others, then merge in place.
  merge   : map segment-relative seconds back onto the episode timeline, group
            by scene, verify coverage (gaps WARN and fall back to keyframes),
            print av_notes.json.

HARD RULES (evidence layer, same red line as the acoustic transcripts):
- AV notes are EVIDENCE for the writer. They NEVER become dialogue text (that is
  [[SUB:n]] verbatim from subtitles), NEVER re-attribute speakers (that is the
  acoustic layer), and NEVER replace keyframes (that is the visual track).
- People are described by visible epithets ("红衣女子"); the model is forbidden
  from naming anyone - naming belongs to the scene-writing pass.

Exit codes: 0 success | 1 bad input | 3 ffmpeg/ffprobe missing | 4 segment cut failed
            7 outputs missing/invalid at run/merge stage | 8 DASHSCOPE_API_KEY missing.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import omni_client
from speaker_diarize import (
    EXIT_BAD_INPUT,
    EXIT_EXTRACT_FAILED,
    EXIT_MISSING_FFMPEG,
    EXIT_MISSING_KEY,
    EXIT_OK,
    EXIT_OMNI_OUTPUT_INVALID,
    locate_video,
    require_binaries,
)

AV_SEGMENT_SEC = 90.0          # max seconds per segment (video2note's quality tier)
AV_OVERLAP_SEC = 5.0           # between segments of the SAME scene only
AV_NOTES_SCHEMA = "vts-av-notes/v1"

AV_UNDERSTAND_PROMPT = """You are a rigorous, objective audio-visual analyst for a screenplay
reconstruction pipeline. Watch this clip and report ONLY what is truly visible and audible.

Time basis: the clip's local timeline runs 0..$__SPAN__$ seconds, which maps to source
$__START__$..$__END__$ seconds of the episode. Output LOCAL-relative seconds; the caller adds the offset.

HARD RULES:
- Describe only what is truly on screen / in the soundtrack. Never invent details; anything you
  cannot confirm goes into "uncertain".
- Refer to people by DESCRIPTIVE EPITHETS visible in the frame (e.g. "红衣女子", "拄拐老者").
  NEVER output a real name, even if you believe you recognize the person or a name appears in
  on-screen text.
- Do NOT transcribe spoken dialogue. Speech content is handled elsewhere; your job is ONLY the
  non-dialogue channels below. (Lyrics you hear belong in acoustic, not in dialogue.)
- All "what" / "movement" / "appearance" / "music_mood" / "uncertain" values must be written in
  natural Chinese - they feed a Chinese screenplay writer. Concrete and filmable: write light,
  objects, bodies, sound - never inner feelings.
- Output ONLY one JSON object inside a ```json fence, exactly this shape:
{"visual": {"caption": "<2-3 句中文，概括整段画面>",
   "actions": [{"start": <sec>, "end": <sec>, "who": "<描述性称呼>", "what": "<中文，具体可演的动作>"}],
   "camera": [{"start": <sec>, "end": <sec>, "movement": "<中文：景别/推拉摇移/手持/固定>"}],
   "scene_transition": "<中文：硬切/叠化/无>"},
 "visible_text": [{"start": <sec>, "end": <sec>, "text": "<屏显文字逐字>",
   "appearance": "<中文：位置/字体/颜色>"}],
 "acoustic": {"events": [{"start": <sec>, "what": "<中文：关门/脚步/雨声/杯碎等>"}],
   "music_mood": "<中文：音乐风格与情绪走向；无音乐写 无>"},
 "uncertain": ["<中文：任何无法确认之处>"]}
- Any list may be empty; "music_mood" may be "无". Never write anything outside the JSON.
"""


def build_prompt(start_ms: int, end_ms: int) -> str:
    span = (end_ms - start_ms) / 1000.0
    return (AV_UNDERSTAND_PROMPT
            .replace("$__SPAN__$", f"{span:.1f}")
            .replace("$__START__$", f"{start_ms / 1000.0:.1f}")
            .replace("$__END__$", f"{end_ms / 1000.0:.1f}"))


# ---------------------------------------------------------------------------
# pure planning / mapping / coverage
# ---------------------------------------------------------------------------

def plan_segments(scenes: List[Dict[str, Any]], segment_sec: float = AV_SEGMENT_SEC,
                  overlap_sec: float = AV_OVERLAP_SEC) -> List[Dict[str, Any]]:
    """Split each scene into <=segment_sec windows with overlap_sec shared between
    consecutive windows of the SAME scene; a short tail folds back into the last
    window instead of spawning a sliver. Pure; deterministic. Segments never
    cross scene boundaries."""
    segments: List[Dict[str, Any]] = []
    ov = int(overlap_sec * 1000)
    max_len = int(segment_sec * 1000)
    for sc in scenes:
        sid = str(sc.get("scene_id", ""))
        s0, s1 = int(sc.get("start_ms", 0)), int(sc.get("end_ms", 0))
        if s1 <= s0:
            continue
        start = s0
        while start < s1:
            end = min(s1, start + max_len)
            if s1 - end < max_len // 2:      # short tail folds into this window
                end = s1
            segments.append({"scene_id": sid, "start_ms": start, "end_ms": end})
            if end >= s1:
                break
            start = end - ov
            if start <= segments[-1]["start_ms"]:   # degenerate guard: always advance
                start = end
    return segments


def parse_note(data: Any, start_ms: int, end_ms: int) -> Dict[str, Any]:
    """Coerce one model reply into the note schema and shift LOCAL-relative
    seconds onto the absolute episode timeline (ms). Malformed entries are
    dropped; missing keys become empty defaults. Never raises."""
    span_s = max(0.001, (end_ms - start_ms) / 1000.0)

    def sec(v: Any) -> Optional[float]:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        if f != f or f in (float("inf"), float("-inf")):   # NaN / inf guard
            return None
        return min(max(f, 0.0), span_s)

    def shift(entries: Any, keys: Tuple[str, ...] = ("start", "end")) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not isinstance(entries, list):
            return out
        for e in entries:
            if not isinstance(e, dict):
                continue
            row = {k: v for k, v in e.items() if k not in keys}
            s = sec(e.get(keys[0]))
            if s is None:
                continue
            # point events (sound cues) carry only "start"; an explicit inverted
            # window stays malformed and is dropped
            t = sec(e.get(keys[1])) if keys[1] in e else s
            if t is None or t < s:
                continue
            row[keys[0]] = start_ms + int(s * 1000)
            row[keys[1]] = start_ms + int(t * 1000)
            out.append(row)
        return out

    data = data if isinstance(data, dict) else {}
    visual = data.get("visual") if isinstance(data.get("visual"), dict) else {}
    acoustic = data.get("acoustic") if isinstance(data.get("acoustic"), dict) else {}
    return {
        "visual": {
            "caption": str(visual.get("caption") or ""),
            "actions": shift(visual.get("actions")),
            "camera": shift(visual.get("camera")),
            "scene_transition": str(visual.get("scene_transition") or ""),
        },
        "visible_text": shift(data.get("visible_text"), ("start", "end")),
        "acoustic": {
            "events": shift(acoustic.get("events"), ("start", "end")),
            "music_mood": str(acoustic.get("music_mood") or ""),
        },
        "uncertain": [str(u).strip() for u in data.get("uncertain", [])
                      if isinstance(u, str) and u.strip()]
        if isinstance(data.get("uncertain"), list) else [],
    }


def scene_coverage(scene: Dict[str, Any], intervals: List[Tuple[int, int]]) -> Tuple[float, int]:
    """Covered percentage of a scene's span by the union of intervals, plus the
    union size in ms. Pure."""
    s0, s1 = int(scene.get("start_ms", 0)), int(scene.get("end_ms", 0))
    span = max(1, s1 - s0)
    clipped = sorted((max(s0, a), min(s1, b)) for a, b in intervals if b > s0 and a < s1)
    covered = 0
    cur_a = cur_b = None
    for a, b in clipped:
        if cur_b is None or a > cur_b:
            if cur_b is not None:
                covered += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    if cur_b is not None:
        covered += cur_b - cur_a
    return round(100.0 * covered / span, 1), covered


def load_scenes(ws: Optional[str]) -> List[Dict[str, Any]]:
    path = Path(ws, ".cache", "visual", "scenes.json") if ws else Path(".cache", "visual", "scenes.json")
    if not path.is_file():
        sys.stderr.write(f"[FATAL] scenes.json not found: {path} (run semantic_scene_grouper.py first)\n")
        sys.exit(EXIT_BAD_INPUT)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        sys.stderr.write(f"[FATAL] Cannot parse scenes.json: {e}\n")
        sys.exit(EXIT_BAD_INPUT)
    scenes = doc.get("scenes") if isinstance(doc, dict) else doc
    if not isinstance(scenes, list) or not scenes:
        sys.stderr.write("[FATAL] scenes.json contains no scenes\n")
        sys.exit(EXIT_BAD_INPUT)
    return [s for s in scenes if isinstance(s, dict) and s.get("scene_id")]


# ---------------------------------------------------------------------------
# actions
# ---------------------------------------------------------------------------

def _cut_segment(video: str, out_path: str, start_ms: int, end_ms: int) -> None:
    """Accurate cut (-ss before -i re-encode discards from the prior keyframe)
    straight to the inline-friendly 480p tier; heavier downshifts go through
    omni_client.fit_video when this tier still exceeds the budget."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-ss", f"{start_ms / 1000.0:.3f}", "-i", video,
           "-t", f"{(end_ms - start_ms) / 1000.0:.3f}",
           "-vf", "scale=-2:480", "-c:v", "libx264", "-crf", "26", "-preset", "veryfast",
           "-c:a", "aac", "-b:a", f"{omni_client.VIDEO_AUDIO_KBPS}k", "-ar", "16000", "-ac", "1",
           "-movflags", "+faststart", "-y", out_path]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                              timeout=omni_client.FFMPEG_TIMEOUT_SEC, check=False)
    except subprocess.TimeoutExpired:
        sys.stderr.write(f"[FATAL] ffmpeg segment cut timed out: {out_path}\n")
        sys.exit(EXIT_EXTRACT_FAILED)
    if proc.returncode != 0 or not os.path.isfile(out_path):
        lines = (proc.stderr or b"").decode("utf-8", "replace").strip().splitlines()
        tail = lines[-1] if lines else "unknown ffmpeg error"
        sys.stderr.write(f"[FATAL] ffmpeg failed to cut {out_path}: {tail}\n")
        sys.exit(EXIT_EXTRACT_FAILED)
    payload, _fmt = omni_client.fit_video(out_path)
    if Path(payload) != Path(out_path):
        Path(payload).replace(out_path)


def cmd_prepare(ws: Optional[str], segment_sec: float, video_override: Optional[str]) -> None:
    require_binaries()
    video = locate_video(ws, video_override)
    if not video:
        sys.stderr.write("[FATAL] No video file found in materials/ (or --video path invalid)\n")
        sys.exit(EXIT_BAD_INPUT)
    scenes = load_scenes(ws)
    segments = plan_segments(scenes, segment_sec)

    av_dir = Path(ws, ".cache", "av") if ws else Path(".cache", "av")
    os.makedirs(av_dir, exist_ok=True)
    entries: List[Dict[str, Any]] = []
    for i, seg in enumerate(segments):
        name = f"seg_{i:03d}.mp4"
        out_name = f"av_note_{i:03d}.json"
        p = (av_dir / name).resolve()
        if ws:
            try:
                p.relative_to(Path(ws).resolve())
            except ValueError:
                sys.stderr.write(f"[FATAL] Segment path escaped the workspace containment: {p}\n")
                sys.exit(2)
        _cut_segment(video, str(p), seg["start_ms"], seg["end_ms"])
        entries.append({"file": f"av/{name}", "scene_id": seg["scene_id"],
                        "start_ms": seg["start_ms"], "end_ms": seg["end_ms"],
                        "output": f"av/{out_name}"})
        sys.stderr.write(f"[CUT ] {name}: {seg['scene_id']} "
                         f"{seg['start_ms'] / 1000.0:.1f}-{seg['end_ms'] / 1000.0:.1f}s\n")

    work = {
        "status": "ready",
        "video_path": video,
        "segment_seconds": segment_sec,
        "overlap_sec": AV_OVERLAP_SEC,
        "expected_schema": {"visual": ["caption", "actions", "camera", "scene_transition"],
                            "visible_text": [], "acoustic": ["events", "music_mood"],
                            "uncertain": []},
        "scenes": [{"scene_id": s["scene_id"], "start_ms": s.get("start_ms", 0),
                    "end_ms": s.get("end_ms", 0)} for s in scenes],
        "segments": entries,
        "note": ("Each segment: `run` dials DashScope directly (needs DASHSCOPE_API_KEY). "
                 "Notes are EVIDENCE for the writing pass - never dialogue text, never "
                 "speaker re-attribution."),
    }
    workorder_path = (av_dir / "av_workorder.json").resolve()
    if ws:
        try:
            workorder_path.relative_to(Path(ws).resolve())
        except ValueError:
            sys.stderr.write(f"[FATAL] Workorder path escaped the workspace containment: {workorder_path}\n")
            sys.exit(2)
    workorder_path.write_text(json.dumps(work, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sys.stderr.write(f"[DONE ] {len(entries)} segment(s) cut; workorder: {workorder_path}\n"
                     "[NEXT ] av_understand.py --workspace <ws> run\n")
    sys.exit(EXIT_OK)


def _workorder(ws: Optional[str]) -> Tuple[Path, Dict[str, Any]]:
    av_dir = Path(ws, ".cache", "av") if ws else Path(".cache", "av")
    workorder_path = (av_dir / "av_workorder.json").resolve()
    if ws:
        try:
            workorder_path.relative_to(Path(ws).resolve())
        except ValueError:
            sys.stderr.write(f"[FATAL] Workorder path escaped the workspace containment: {workorder_path}\n")
            sys.exit(2)
    if not workorder_path.is_file():
        sys.stderr.write(f"[FATAL] av_workorder.json not found: {workorder_path} (run `prepare` first)\n")
        sys.exit(EXIT_BAD_INPUT)
    try:
        work = json.loads(workorder_path.read_text(encoding="utf-8"))
    except Exception as e:
        sys.stderr.write(f"[FATAL] Cannot parse workorder: {e}\n")
        sys.exit(EXIT_OMNI_OUTPUT_INVALID)
    return av_dir, work


def _is_valid_note(pth: Path) -> bool:
    try:
        data = json.loads(pth.read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(data, dict) and (isinstance(data.get("raw"), dict)
                                       or isinstance(data.get("visual"), dict))


def cmd_run(ws: Optional[str], force: bool = False) -> None:
    av_dir, work = _workorder(ws)
    segments = [s for s in work.get("segments", []) if isinstance(s, dict)]
    if not segments:
        sys.stderr.write("[FATAL] Workorder lists no segments\n")
        sys.exit(EXIT_OMNI_OUTPUT_INVALID)

    def contained(pth: Path, what: str) -> Path:
        if ws:
            try:
                pth.relative_to(Path(ws).resolve())
            except ValueError:
                sys.stderr.write(f"[FATAL] {what} escaped the workspace containment: {pth}\n")
                sys.exit(2)
        return pth

    def out_path(entry: Dict[str, Any]) -> Path:
        return contained((av_dir / os.path.basename(entry["output"])).resolve(), "AV note path")

    def seg_path(entry: Dict[str, Any]) -> Path:
        return contained((av_dir / os.path.basename(entry["file"])).resolve(), "Segment path")

    pending = [e for e in segments if force or not (out_path(e).is_file() and _is_valid_note(out_path(e)))]
    key = omni_client.resolve_key()
    if pending and not key:
        sys.stderr.write(
            "[FATAL] DASHSCOPE_API_KEY is not set; direct AV understanding is impossible.\n"
            "[FATAL] Options: (a) export DASHSCOPE_API_KEY=... and rerun `run`;\n"
            "        (b) skip this optional pass - stage 4 falls back to keyframes-only;\n"
            "        (c) the writer proceeds without av_notes.\n")
        sys.exit(EXIT_MISSING_KEY)

    failed: List[str] = []
    for i, entry in enumerate(segments):
        o_p = out_path(entry)
        if not force and o_p.is_file() and _is_valid_note(o_p):
            sys.stderr.write(f"[SKIP] seg_{i:03d}: {o_p.name} already present and valid (--force to redo)\n")
            continue
        s_p = seg_path(entry)
        if not s_p.is_file():
            sys.stderr.write(f"[ERROR] seg_{i:03d}: segment file missing: {s_p.name}\n")
            failed.append(f"seg_{i:03d} file missing")
            continue
        sys.stderr.write(f"[RUN ] seg_{i:03d}: {entry['scene_id']} "
                         f"{entry['start_ms'] / 1000.0:.1f}-{entry['end_ms'] / 1000.0:.1f}s -> {o_p.name}\n")
        try:
            data, meta = omni_client.understand_video_segment(
                str(s_p), prompt=build_prompt(int(entry["start_ms"]), int(entry["end_ms"])), api_key=key)
            # the RAW model reply is the evidence on disk; parse_note runs exactly
            # once, at merge time (double-shifting the timeline is a real bug class)
            o_p.write_text(json.dumps({"raw": data, "meta": meta},
                                      ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            probe = parse_note(data, int(entry["start_ms"]), int(entry["end_ms"]))
            sys.stderr.write(f"[OK  ] seg_{i:03d}: {len(probe['visual']['actions'])} actions, "
                             f"{len(probe['visible_text'])} on-screen text -> {o_p.name}\n")
        except omni_client.OmniError as e:
            sys.stderr.write(f"[ERROR] seg_{i:03d}: {e}\n")
            failed.append(f"seg_{i:03d} {e.kind}")
            continue
        except Exception as e:  # noqa: BLE001 - one segment must not stop the others
            sys.stderr.write(f"[ERROR] seg_{i:03d}: {type(e).__name__}: {str(e)[:200]}\n")
            failed.append(f"seg_{i:03d} {type(e).__name__}")
            continue

    missing = [os.path.basename(e["output"]) for e in segments if not out_path(e).is_file()]
    if missing or failed:
        sys.stderr.write(f"[FATAL] run incomplete: {len(failed)} failed call(s); missing: "
                         f"{', '.join(missing) if missing else '(none saved)'}\n"
                         "[FATAL] Rerun `run` to retry only the missing segments (resume), or skip "
                         "this optional pass - stage 4 falls back to keyframes-only.\n")
        sys.exit(EXIT_OMNI_OUTPUT_INVALID)
    cmd_merge(ws)
    sys.exit(EXIT_OK)


def cmd_merge(ws: Optional[str]) -> None:
    av_dir, work = _workorder(ws)
    scenes = {str(s["scene_id"]): s for s in work.get("scenes", []) if isinstance(s, dict)}
    scene_segments: Dict[str, List[Dict[str, Any]]] = {}
    for i, entry in enumerate(work.get("segments", [])):
        p = (av_dir / os.path.basename(entry["output"])).resolve()
        if ws:
            try:
                p.relative_to(Path(ws).resolve())
            except ValueError:
                sys.stderr.write(f"[FATAL] AV note path escaped the workspace containment: {p}\n")
                sys.exit(2)
        if not p.is_file():
            sys.stderr.write(f"[FATAL] Missing AV note: {p.name} (rerun `run`)\n")
            sys.exit(EXIT_OMNI_OUTPUT_INVALID)
        try:
            loaded = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            sys.stderr.write(f"[FATAL] AV note {p.name} is not valid JSON: {e}\n")
            sys.exit(EXIT_OMNI_OUTPUT_INVALID)
        raw = loaded.get("raw") if isinstance(loaded, dict) else None
        if raw is None:  # tolerate a pre-parsed note (single-shift assumption broken; prefer raw)
            raw = loaded if isinstance(loaded, dict) and "visual" in loaded else None
        if not isinstance(raw, dict):
            sys.stderr.write(f"[FATAL] AV note {p.name} has no usable payload\n")
            sys.exit(EXIT_OMNI_OUTPUT_INVALID)
        note = parse_note(raw, int(entry["start_ms"]), int(entry["end_ms"]))
        sid = str(entry.get("scene_id", ""))
        scene_segments.setdefault(sid, []).append({
            "start_ms": int(entry["start_ms"]), "end_ms": int(entry["end_ms"]),
            "visual": note["visual"], "visible_text": note["visible_text"],
            "acoustic": note["acoustic"], "uncertain": note["uncertain"],
        })

    scene_notes = []
    for sid, sc in scenes.items():
        covered_pct, _ = scene_coverage(sc, [(seg["start_ms"], seg["end_ms"])
                                             for seg in scene_segments.get(sid, [])])
        if covered_pct < 100.0:
            sys.stderr.write(f"[WARN] Scene {sid} coverage {covered_pct}% - gaps fall back to "
                             "keyframes-only evidence for the writer\n")
        scene_notes.append({
            "scene_id": sid,
            "start_ms": int(sc.get("start_ms", 0)),
            "end_ms": int(sc.get("end_ms", 0)),
            "covered_pct": covered_pct,
            "segments": sorted(scene_segments.get(sid, []), key=lambda s: s["start_ms"]),
        })

    payload = json.dumps({
        "schema": AV_NOTES_SCHEMA,
        "scene_notes": scene_notes,
    }, ensure_ascii=False, indent=2) + "\n"
    notes_path = (Path(ws, ".cache", "visual", "av_notes.json") if ws
                  else Path(".cache", "visual", "av_notes.json")).resolve()
    if ws:
        try:
            notes_path.relative_to(Path(ws).resolve())
        except ValueError:
            sys.stderr.write(f"[FATAL] AV notes path escaped the workspace containment: {notes_path}\n")
            sys.exit(2)
    notes_path.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scene-grounded AV understanding pass (Qwen3.8-Omni): actions / camera / "
                    "sound / on-screen text as WRITING EVIDENCE - never dialogue text.")
    parser.add_argument("action", nargs="?", default=None, choices=["prepare", "run", "merge"],
                        help="prepare: split scenes + cut segment payloads + workorder. "
                             "run: direct DashScope per segment + merge (needs DASHSCOPE_API_KEY). "
                             "merge: assemble av_notes.json from saved notes")
    parser.add_argument("--workspace", "-w", default=None, help="Workspace root directory")
    parser.add_argument("--video", "-v", default=None, help="Override video path (prepare)")
    parser.add_argument("--segment-seconds", type=float, default=AV_SEGMENT_SEC,
                        help=f"Max seconds per segment (default: {AV_SEGMENT_SEC:.0f})")
    parser.add_argument("--force", action="store_true",
                        help="run only: re-understand segments whose notes already exist")
    args = parser.parse_args()

    ws = os.path.abspath(args.workspace) if args.workspace else None
    if args.action == "prepare":
        cmd_prepare(ws, args.segment_seconds, args.video)
    elif args.action == "run":
        cmd_run(ws, force=args.force)
    elif args.action == "merge":
        cmd_merge(ws)
        sys.exit(EXIT_OK)
    else:
        parser.error("action required: prepare | run | merge (see --help)")


if __name__ == "__main__":
    main()
