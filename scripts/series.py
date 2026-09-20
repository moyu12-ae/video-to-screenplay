#!/usr/bin/env python3
"""
scripts/series.py - Series-level configuration directory (v0.6 P-1).

Before v0.6 every episode workspace carried its own materials/bible.json, so a
season-long setting (OP/ED windows, and from v0.6 the approved cast table) had
to be copied by hand per episode. SKILL.md promised "configure once, reuse for
the whole season" while no code looked outside the workspace - a promise with no
implementation behind it. This module is that implementation.

Layout:

    <series_root>/
    ├── cast.approved.json     signed-off cast table (read-only for episodes)
    ├── op_ed_windows.json     [{"start_ms":..,"end_ms":..,"label":"OP"}]
    └── episodes/epNN/         ordinary workspaces (materials/ .cache/ output/)
        └── .v2s-series        pointer back to the series root

A workspace with no .v2s-series behaves exactly as it did in v0.5.x: series
files are simply absent, nothing errors. Precedence is series file first, then
the legacy in-workspace copy, so existing workspaces keep working unmodified.

Nothing here reads or writes credentials. The cast table holds names and
evidence, never API keys.
"""

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

SERIES_POINTER_SCHEMA = "vts-series-pointer/v1"
POINTER_FILENAME = ".v2s-series"
CAST_FILENAME = "cast.approved.json"
OPED_FILENAME = "op_ed_windows.json"
# The workspace keeps a read-only copy of the table it was resolved against, so
# a re-run months later reproduces the naming even if the series table moved on.
CAST_SNAPSHOT_REL = Path(".cache", "cast.series.snapshot.json")


def _warn(msg: str) -> None:
    sys.stderr.write(f"[WARN] {msg}\n")


def write_json_atomic(path: Path, payload: Any) -> None:
    """Sibling temp file + os.replace, so a crash never leaves half a pointer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Optional[Any]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        _warn(f"Cannot parse {path}: {e}")
        return None


# --------------------------------------------------------------------------
# series root resolution
# --------------------------------------------------------------------------

def pointer_path(workspace: Path) -> Path:
    return Path(workspace) / POINTER_FILENAME


def resolve_series_root(workspace: Optional[str | Path]) -> Optional[Path]:
    """Return the bound series root, or None when unbound. A pointer to a
    directory that no longer exists warns and yields None - an episode must
    still run with its own materials."""
    if not workspace:
        return None
    ptr = pointer_path(Path(workspace))
    if not ptr.is_file():
        return None
    doc = read_json(ptr)
    if not isinstance(doc, dict):
        return None
    raw = str(doc.get("series_root") or "").strip()
    if not raw:
        _warn(f"{ptr} has no series_root; ignoring series configuration")
        return None
    root = Path(os.path.expanduser(raw))
    if not root.is_dir():
        _warn(f"series root {root} in {ptr} does not exist; falling back to workspace-local config")
        return None
    return root


def bind_series(workspace: str | Path, series_root: str | Path) -> Dict[str, Any]:
    """Write the pointer and snapshot the current approved cast table.
    Refuses a series root that is the workspace itself (that would make the
    pointer and its target ambiguous)."""
    ws = Path(os.path.abspath(str(workspace)))
    root = Path(os.path.abspath(str(series_root)))
    if not ws.is_dir():
        raise ValueError(f"workspace does not exist: {ws}")
    if root == ws:
        raise ValueError("series root must not be the workspace itself")
    root.mkdir(parents=True, exist_ok=True)
    for name in (CAST_FILENAME, OPED_FILENAME):
        candidate = root / name
        if candidate.is_file() and candidate.stat().st_size == 0:
            raise ValueError(f"series file is empty, refusing to bind: {candidate}")

    snapshot = snapshot_cast(ws, root)
    pointer = {
        "schema": SERIES_POINTER_SCHEMA,
        "series_root": str(root),
        "series_name": _series_name(root),
        "cast_snapshot": str(snapshot["rel_path"]) if snapshot else None,
        "cast_version": snapshot.get("version") if snapshot else None,
        "cast_sha256": snapshot.get("sha256") if snapshot else None,
    }
    write_json_atomic(pointer_path(ws), pointer)
    return {"status": "bound", "workspace": str(ws), "series_root": str(root),
            "pointer": str(pointer_path(ws)), "cast": snapshot}


def _series_name(root: Path) -> str:
    meta = read_json(root / "series.json")
    if isinstance(meta, dict) and meta.get("name"):
        return str(meta["name"])
    return root.name


# --------------------------------------------------------------------------
# cast table (approved, read-only for episodes)
# --------------------------------------------------------------------------

def approved_cast_path(series_root: Optional[Path]) -> Optional[Path]:
    if not series_root:
        return None
    candidate = series_root / CAST_FILENAME
    return candidate if candidate.is_file() else None


def snapshot_cast(workspace: Path, series_root: Path) -> Optional[Dict[str, Any]]:
    """Copy the series cast table into the workspace .cache/ and record its
    version + digest. Returns None when the series has no table yet (first
    episode, nothing signed off)."""
    src = series_root / CAST_FILENAME
    if not src.is_file():
        return None
    doc = read_json(src)
    if not isinstance(doc, dict):
        _warn(f"{src} is not a JSON object; snapshot skipped")
        return None
    rel = CAST_SNAPSHOT_REL
    dest = workspace / rel
    payload = {
        "schema": SERIES_POINTER_SCHEMA,
        "copied_from": str(src),
        "sha256": sha256_of(src),
        "version": doc.get("version"),
        "table": doc,
    }
    write_json_atomic(dest, payload)
    return {"rel_path": str(rel), "abs_path": str(dest),
            "version": doc.get("version"), "sha256": payload["sha256"]}


def load_snapshot(workspace: Optional[str | Path]) -> Optional[Dict[str, Any]]:
    if not workspace:
        return None
    doc = read_json(Path(workspace) / CAST_SNAPSHOT_REL)
    return doc if isinstance(doc, dict) and isinstance(doc.get("table"), dict) else None


def load_approved(workspace: Optional[str | Path]) -> Dict[str, Any]:
    """The cast table an episode may use. Series file when bound and present,
    otherwise the workspace snapshot taken at bind time, otherwise empty.
    Never mutates the series file - callers get a copy."""
    empty = {"schema": "vts-cast/v1", "entities": [], "slots": [], "clusters": [], "pending": []}
    if not workspace:
        return empty
    ws = Path(workspace)
    root = resolve_series_root(ws)
    src = approved_cast_path(root)
    if src is None:
        snap = load_snapshot(ws)
        if snap:
            table = snap["table"]
            return {**empty, **table}
        return empty
    doc = read_json(src)
    if doc is None:
        return empty
    if not isinstance(doc, dict):
        _warn(f"{src} is not a JSON object; treating the cast table as empty")
        return empty
    merged = {**empty, **doc}
    snap = load_snapshot(ws)
    if snap and snap.get("sha256") and merged.get("version") != "draft":
        if snap["sha256"] != sha256_of(src):
            _warn(f"series cast table changed after this workspace was bound "
                  f"(snapshot {str(snap['sha256'])[:12]}… version {snap.get('version')}, "
                  f"current {sha256_of(src)[:12]}… version {merged.get('version')}); "
                  f"re-run 'workspace.py series --bind' or 'resolve_cast.py --refresh-snapshot' "
                  f"to make the change explicit")
    return merged


def approved_names(workspace: Optional[str | Path]) -> List[str]:
    """Canonical names plus every alias, used by the naming gate."""
    out: List[str] = []
    for entity in load_approved(workspace).get("entities") or []:
        if not isinstance(entity, dict):
            continue
        if str(entity.get("status") or "").lower() != "approved":
            continue
        name = str(entity.get("canonical_name") or "").strip()
        if name:
            out.append(name)
        for alias in entity.get("aliases") or []:
            alias = str(alias).strip()
            if alias:
                out.append(alias)
    return out


# --------------------------------------------------------------------------
# OP/ED windows: series file first, legacy bible.json as fallback
# --------------------------------------------------------------------------

def load_op_ed_windows(workspace: Optional[str | Path]) -> List[Dict[str, Any]]:
    """Series-level op_ed_windows.json when bound; else materials/bible.json.
    When both exist and disagree, the series file wins and the difference is
    reported - silent divergence across a season is how episodes end up with
    half their opening credits in the screenplay."""
    if not workspace:
        return []
    ws = Path(workspace)
    root = resolve_series_root(ws)
    series_windows: Optional[List[Dict[str, Any]]] = None
    if root:
        doc = read_json(root / OPED_FILENAME)
        if isinstance(doc, dict):
            raw = doc.get("op_ed_windows")
        elif isinstance(doc, list):
            raw = doc
        else:
            raw = None
        if isinstance(raw, list):
            series_windows = [w for w in raw if isinstance(w, dict)]

    legacy = _legacy_bible_windows(ws)
    if series_windows is not None:
        if legacy and _window_keys(legacy) != _window_keys(series_windows):
            _warn("OP/ED windows exist in both the series directory and "
                  "materials/bible.json and differ; using the series directory - "
                  "delete the bible.json copy to silence this")
        return series_windows
    return legacy or []


def _legacy_bible_windows(ws: Path) -> List[Dict[str, Any]]:
    doc = read_json(ws / "materials" / "bible.json")
    if not isinstance(doc, dict):
        return []
    raw = doc.get("op_ed_windows")
    return [w for w in raw if isinstance(w, dict)] if isinstance(raw, list) else []


def _window_keys(windows: List[Dict[str, Any]]) -> List[Any]:
    return sorted((w.get("start_ms"), w.get("end_ms"), w.get("label")) for w in windows)


def series_status(workspace: Optional[str | Path]) -> Dict[str, Any]:
    """What an episode actually resolves to - printed so the agent can tell the
    user which cast table is in force before naming anything."""
    ws = Path(workspace) if workspace else None
    root = resolve_series_root(ws) if ws else None
    snap = load_snapshot(ws) if ws else None
    windows = load_op_ed_windows(ws) if ws else []
    return {
        "status": "series_status",
        "workspace": str(ws) if ws else None,
        "bound": bool(root),
        "series_root": str(root) if root else None,
        "series_cast": str(approved_cast_path(root)) if root else None,
        "cast_version": (load_approved(ws).get("version") if ws else None),
        "cast_snapshot": {"version": snap.get("version"), "sha256": snap.get("sha256")} if snap else None,
        "approved_entity_count": len([e for e in (load_approved(ws).get("entities") or [])
                                      if isinstance(e, dict)
                                      and str(e.get("status")).lower() == "approved"]) if ws else 0,
        "op_ed_window_count": len(windows),
        "op_ed_source": ("series" if root and (root / OPED_FILENAME).is_file()
                         else "bible" if windows else "none"),
    }
