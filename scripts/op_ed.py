#!/usr/bin/env python3
"""
scripts/op_ed.py - OP/ED window configuration and matching (v0.5.1).

Windows come from ONE place: materials/bible.json →

    {"op_ed_windows": [{"start_ms": 84000, "end_ms": 105000, "label": "OP"},
                       {"start_ms": 1320000, "end_ms": 1440000, "label": "ED"}]}

Measured once per series, reused for the whole season. No configuration means
no filtering anywhere - every stage keeps its pre-0.5.1 behaviour.

Semantics: a span (scene, dialogue line, speech turn, AV segment) is treated as
non-narrative when >= OP_ED_OVERLAP_RATIO of it falls inside a window. Filtered
subtitles never vanish silently: the extractor records what it removed, and the
final screenplay marks the position with a one-line （动画 OP/ED） note instead
of narrative scenes.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

OP_ED_OVERLAP_RATIO = 0.5


def load_windows(ws: Optional[str]) -> List[Dict[str, Any]]:
    """Read op_ed_windows from materials/bible.json. Malformed entries are
    skipped with a warning; missing file/field means no filtering."""
    if not ws:
        return []
    bible = Path(ws, "materials", "bible.json")
    if not bible.is_file():
        return []
    try:
        doc = json.loads(bible.read_text(encoding="utf-8"))
    except Exception as e:
        sys.stderr.write(f"[WARN] Cannot parse bible.json, OP/ED windows ignored: {e}\n")
        return []
    raw = doc.get("op_ed_windows") if isinstance(doc, dict) else None
    if not isinstance(raw, list):
        return []
    windows: List[Dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            s, e = int(entry["start_ms"]), int(entry["end_ms"])
        except (KeyError, TypeError, ValueError):
            sys.stderr.write(f"[WARN] Skipping malformed op_ed_windows entry: {entry!r}\n")
            continue
        if e <= s:
            sys.stderr.write(f"[WARN] Skipping inverted op_ed_windows entry: {entry!r}\n")
            continue
        windows.append({"start_ms": s, "end_ms": e,
                        "label": str(entry.get("label") or ("ED" if s > 0 else "OP"))})
    return windows


def overlap_ratio(start_ms: int, end_ms: int, window: Dict[str, Any]) -> float:
    """Fraction of [start_ms, end_ms] covered by one window. Pure."""
    span = max(1, int(end_ms) - int(start_ms))
    ov = min(int(end_ms), int(window["end_ms"])) - max(int(start_ms), int(window["start_ms"]))
    return max(0.0, ov) / span


def matching_label(start_ms: int, end_ms: int, windows: List[Dict[str, Any]],
                   threshold: float = OP_ED_OVERLAP_RATIO) -> Optional[str]:
    """Return the window label when STRICTLY MORE than `threshold` of the span
    sits inside a window; None when the span is narrative (or no windows
    configured). Strictly-greater keeps a boundary-straddling dialogue line
    (exactly half inside) on the narrative side. Pure."""
    best: Tuple[float, str] = (0.0, "")
    for w in windows:
        ratio = overlap_ratio(start_ms, end_ms, w)
        if ratio > threshold and ratio > best[0]:
            best = (ratio, str(w.get("label") or "OP"))
    return best[1] or None


def filter_items(items: List[Dict[str, Any]], windows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Drop subtitle lines that fall inside OP/ED windows and reindex the
    survivors 1..N (the [[SUB:n]] contract needs contiguous indices). Returns
    (kept_items, filter_meta). Pure apart from the returned metadata."""
    if not windows:
        return items, {}
    kept: List[Dict[str, Any]] = []
    removed: Dict[str, int] = {}
    for item in items:
        start = int(item.get("start_ms", 0))
        end = int(item.get("end_ms", start))
        label = matching_label(start, max(end, start + 1), windows, threshold=0.5)
        if label:
            removed[label] = removed.get(label, 0) + 1
        else:
            kept.append(item)
    for i, item in enumerate(kept, start=1):
        item["index"] = i
    meta = {"windows": [dict(w) for w in windows], "removed": removed,
            "removed_total": sum(removed.values())}
    return kept, meta
