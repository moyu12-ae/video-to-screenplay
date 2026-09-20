#!/usr/bin/env python3
"""
scripts/op_ed.py - OP/ED window configuration and matching (v0.5.1).

Windows come from ONE reading path, series.load_op_ed_windows():

    {"op_ed_windows": [{"start_ms": 84000, "end_ms": 105000, "label": "OP"},
                       {"start_ms": 1320000, "end_ms": 1440000, "label": "ED"}]}

in <series_root>/op_ed_windows.json when the workspace is bound to a series
(v0.6 P-1, configured once for the whole season), else materials/bible.json in
the workspace - the legacy per-episode location, still fully supported.

Measured once per series, reused for the whole season. No configuration means
no filtering anywhere - every stage keeps its pre-0.5.1 behaviour.

Semantics: a span (scene, dialogue line, speech turn, AV segment) is treated as
non-narrative when >= OP_ED_OVERLAP_RATIO of it falls inside a window. Filtered
subtitles never vanish silently: the extractor records what it removed, and the
final screenplay marks the position with a one-line （动画 OP/ED） note instead
of narrative scenes.
"""

import sys
from typing import Any, Dict, List, Optional, Tuple

import series

OP_ED_OVERLAP_RATIO = 0.5


def load_windows(ws: Optional[str]) -> List[Dict[str, Any]]:
    """Read OP/ED windows (series directory first, legacy bible.json second).
    Malformed entries are skipped with a warning; no windows means no
    filtering."""
    if not ws:
        return []
    windows: List[Dict[str, Any]] = []
    for entry in series.load_op_ed_windows(ws):
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
    removed_lines: List[Dict[str, Any]] = []
    for item in items:
        start = int(item.get("start_ms", 0))
        end = int(item.get("end_ms", start))
        label = matching_label(start, max(end, start + 1), windows, threshold=0.5)
        if label:
            removed[label] = removed.get(label, 0) + 1
            removed_lines.append({"original_index": item.get("index"), "label": label,
                                  "start_ms": start, "end_ms": max(end, start + 1),
                                  "text": str(item.get("text") or "")})
        else:
            kept.append(item)
    for i, item in enumerate(kept, start=1):
        item["index"] = i
    # The dropped lines stay in the record: counts alone cannot be audited or undone,
    # and they are the reference set for the hard-subtitle echo check at merge.
    meta = {"windows": [dict(w) for w in windows], "removed": removed,
            "removed_total": sum(removed.values()), "removed_lines": removed_lines}
    return kept, meta
