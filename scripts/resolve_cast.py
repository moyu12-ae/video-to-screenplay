#!/usr/bin/env python3
"""
scripts/resolve_cast.py - The single place speaker identity is decided (v0.6 P2).

Every upstream stage produces EVIDENCE and nothing produces a name: the diarizer
knows a voice, the AV pass knows a visible body, the subtitles know who is being
addressed. Until now the writing pass silently fused those three and the fusion
result - a character name - appeared in the delivered screenplay with no record
of how it was reached. This module owns the fusion step and can only converge on
what the evidence already says.

Two invariants, both test-enforced:

  1. A cluster is attached to an EXISTING slot only under §4.1 of the design doc:
     positive evidence from at least two families, compatible known attributes,
     and a clear winner. Ambiguity creates a NEW pending slot - over-merging is
     irreversible (every later episode inherits the wrong identity) while an
     extra question is only a cost.
  2. Names are never produced here. A slot gains a name only through a human
     sign-off record (`approved_by: "human"`), which this module reads and
     refuses to forge.

Unknown is a real value, not a failure: attributes stay "unknown" and uncovered
windows are counted rather than scored against a candidate.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import series
import vocatives

CAST_SCHEMA = "vts-cast/v1"
EXIT_OK = 0
EXIT_BAD_INPUT = 1
EXIT_CONTAINMENT = 2
EXIT_NAMING_VIOLATION = 9

# Families of evidence. Two families must AGREE before a cluster may inherit an
# existing identity (§4.1); a single family never justifies an irreversible merge.
FAMILIES = ("acoustic", "visual", "text_native", "text_subtitle", "external")
MIN_FAMILIES_FOR_AUTO_MATCH = 2

# A slot-vs-slot contest is decided only with a clear winner; below this the
# cluster gets its own pending slot instead of a guess. NOTE: this is the
# continuity decision (which slot am I?), NOT the naming decision - naming is
# never automatic (§6). It is a starting point for measurement, not a result:
# the design doc's §13 requires precision/recall on labelled windows before any
# threshold here is presented as validated.
AUTO_MATCH_MIN_MARGIN = 0.34


def _load(ws: Path, rel: str) -> Any:
    path = ws / rel
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        sys.stderr.write(f"[WARN] Cannot parse {path}: {e}\n")
        return None


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# evidence collection (producers stay producers; this only reads their output)
# --------------------------------------------------------------------------

def acoustic_evidence(speakers_doc: Any) -> Dict[str, Dict[str, Any]]:
    """cluster_id -> known acoustic attributes. Absent/unknown stays unknown."""
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(speakers_doc, dict):
        return out
    for cl in speakers_doc.get("clusters") or []:
        if not isinstance(cl, dict) or not cl.get("cluster_id"):
            continue
        ac = cl.get("acoustic") or {}
        out[str(cl["cluster_id"])] = {
            "gender": str(ac.get("gender") or "unknown"),
            "age_band": str(ac.get("age_band") or "unknown"),
            "timbre": str(ac.get("timbre") or "unknown"),
            "speech_ms": int(cl.get("speech_ms") or 0),
            "line_count": int(cl.get("line_count") or 0),
            "named": bool(cl.get("name")),
        }
    return out


def address_evidence(aligned_doc: Any, known_names: List[str]) -> Dict[str, Any]:
    """Who is addressed, and therefore who each speaking cluster is NOT."""
    lines: List[Dict[str, Any]] = []
    if isinstance(aligned_doc, dict):
        for shot in aligned_doc.get("shots") or aligned_doc.get("scenes") or []:
            if isinstance(shot, dict):
                lines.extend(d for d in (shot.get("dialogues") or []) if isinstance(d, dict))
    result = vocatives.extract_address_terms(lines, known_names, index_key="sub_index",
                                             speaker_key="speaker")
    return {"events": result["events"],
            "terms": vocatives.naming_candidates(result),
            "not_speaker": {k: sorted(v) for k, v in vocatives.negative_evidence(result).items()}}


def visual_evidence(av_doc: Any) -> Dict[str, Any]:
    """Who was visibly talking, keyed by the DESCRIPTIVE EPITHET the AV pass is
    required to use (its prompt forbids real names).

    Only `moving` counts as positive. `still` is weak (limited animation loops a
    closed mouth over a whole second) and `not_visible` / absent coverage are
    missing data - all three are counted, none of them scores against a candidate,
    per the one-sided rule. A slot reaches this evidence through `visual_label`,
    the epithet a human attached to that entity at sign-off: the mapping from
    "金发青年" to a name is a human judgement, never a string match here.
    """
    notes = [n for n in ((av_doc or {}).get("scene_notes") or []) if isinstance(n, dict)] \
        if isinstance(av_doc, dict) else []
    compliance = (av_doc or {}).get("mouth_compliance") if isinstance(av_doc, dict) else None
    clips_total = clips_usable = clips_positive = 0
    positive: Dict[str, int] = {}
    states_seen = {"moving": 0, "still": 0, "not_visible": 0, "unknown": 0}
    for note in notes:
        for seg in note.get("segments") or []:
            actions = ((seg.get("visual") or {}).get("actions") or [])
            if not actions:
                continue
            clips_total += 1
            usable = any(str(a.get("mouth_state")) in ("moving", "still", "not_visible")
                         for a in actions)
            clips_usable += int(usable)
            if any(str(a.get("mouth_state")) == "moving" for a in actions):
                clips_positive += 1
            for action in actions:
                state = str(action.get("mouth_state") or "unknown")
                if state in states_seen:
                    states_seen[state] += 1
                if state == "moving" and str(action.get("who") or "").strip():
                    who = str(action["who"]).strip()
                    positive[who] = positive.get(who, 0) + 1
    usable_channel = bool(compliance.get("usable")) if isinstance(compliance, dict) else False
    return {"clips_total": clips_total, "clips_visual_usable": clips_usable,
            "clips_positive": clips_positive,
            "anchors": {"positive_by_anchor": positive, "states_seen": states_seen},
            "available": usable_channel,
            "note": "visual family is live" if usable_channel else
                    "mouth_state compliance below the P3 gate (or no v3 notes yet): "
                    "the visual family contributes no votes and cannot satisfy §4.1"}


# --------------------------------------------------------------------------
# slots, and the §4.1 three-tier match
# --------------------------------------------------------------------------

def slots_from_approved(approved: Dict[str, Any],
                      visual: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """One slot per approved entity, carrying its profile. `entity_id` is set
    because a human already signed it off - that is the only legal way a slot
    can arrive with a name."""
    visual = visual or {"clips_total": 0, "clips_visual_usable": 0,
                        "anchors": {"positive_by_anchor": {}}}
    slots: List[Dict[str, Any]] = []
    for entity in approved.get("entities") or []:
        if not isinstance(entity, dict):
            continue
        if str(entity.get("status") or "").lower() != "approved":
            continue
        voice = entity.get("voice_profile") or {}
        slots.append({
            "slot_id": f"S{len(slots) + 1}",
            "profile": {
                "gender": str(voice.get("gender") or "unknown"),
                "age_band": str(voice.get("age_band") or "unknown"),
                "timbre": str(voice.get("timbre") or "unknown"),
                "visual": "; ".join(str(a.get("desc") or "") for a in
                                    (entity.get("visual_anchors") or []) if isinstance(a, dict)),
            },
            "status": "approved",
            "entity_id": entity.get("id"),
            # The descriptive epithet a human mapped to this entity at sign-off.
            # The AV pass may never output a name, so this field is the only legal
            # bridge between "金发青年" and an entity.
            "visual_label": str(entity.get("visual_label") or "").strip() or None,
            # Coverage is per slot: how many clips were usable at all, and how many
            # gave a positive sighting of THIS anchor. Uncovered clips stay in the
            # denominator so "nobody saw them" is visible as a number.
            "coverage": {"clips_total": visual.get("clips_total", 0),
                         "clips_visual_usable": visual.get("clips_visual_usable", 0),
                         "clips_positive": int((visual.get("anchors") or {})
                                               .get("positive_by_anchor", {})
                                               .get(str(entity.get("visual_label") or ""), 0))},
            "origin": "approved_table",
        })
    return slots


def _compatible(known: str, other: str) -> Optional[bool]:
    """True = both known and equal; None = at least one side is unknown, which is
    MISSING data (never a mismatch); False = both known and different."""
    if known == "unknown" or other == "unknown":
        return None
    return known == other


def family_support(cluster: Dict[str, Any], slot: Dict[str, Any],
                   anchors: Dict[str, Any], entity_name: Optional[str],
                   anchors_available: bool = True) -> Tuple[List[str], List[str]]:
    """Return (families supporting, reasons). Positive evidence only:
    an absent observation is never a vote for or against."""
    families: List[str] = []
    reasons: List[str] = []
    profile = slot.get("profile") or {}

    compat = {f: _compatible(str(cluster.get(f) or "unknown"), str(profile.get(f) or "unknown"))
              for f in ("gender", "age_band", "timbre")}
    known_agree = [f for f, v in compat.items() if v is True]
    if known_agree:
        families.append("acoustic")
        reasons.append("acoustic." + ",".join(f"{f}={cluster.get(f)}" for f in known_agree))

    label = slot.get("visual_label")
    positive = anchors.get("positive_by_anchor") or {}
    if anchors_available and label and positive.get(label):
        families.append("visual")
        reasons.append(f"visual.mouth={positive[label]} clips of {label} talking")

    # Deliberately NO text_subtitle support here. An address term says the
    # LISTENER is 茉里, and the cluster being scored just spoke that line, so the
    # only thing it can contribute is the exclusion applied by the caller. Turning
    # it into support would rebuild the exact mistake this design came from - the
    # ep02 line 「茉里 你交朋友了」 attributed to 茉里. Naming candidates from these
    # terms go to the human sign-off table instead (§7, role=naming).
    return families, reasons


def match_cluster(cluster_id: str, cluster: Dict[str, Any], slots: List[Dict[str, Any]],
                  ev: Dict[str, Any]) -> Dict[str, Any]:
    """§4.1: ① approved slot → ② pending slot → ③ brand-new slot.

    ① and ② both require >=2 supporting families and a margin over the runner-up;
    anything else is ambiguity, and ambiguity creates a pending slot instead of
    merging - a wrong merge is inherited by every later episode.
    """
    not_speaker = ev["address"].get("not_speaker") or {}
    anchors = ev["visual"]["anchors"]
    entity_by_slot = {s["slot_id"]: s.get("entity_id") for s in slots}
    scored: List[Tuple[float, str, List[str], List[str]]] = []
    for slot in slots:
        entity_name = _entity_name(ev["approved"], entity_by_slot.get(slot["slot_id"]))
        if entity_name and entity_name in (not_speaker.get(cluster_id) or []):
            continue  # this cluster spoke a line addressing them: ruled out, not scored
        families, reasons = family_support(cluster, slot, anchors, entity_name,
                                            anchors_available=bool(ev["visual"].get("available")))
        if not families:
            continue
        scored.append((len(families) + _exact_attribute_bonus(cluster, slot),
                       slot["slot_id"], families, reasons))

    scored.sort(key=lambda x: (-x[0], x[1]))
    if not scored:
        return _new_slot_result(cluster_id, cluster, "no slot has any positive evidence")
    top_score, top_slot, families, reasons = scored[0]
    runner = scored[1][0] if len(scored) > 1 else 0
    margin = (top_score - runner) / max(1.0, top_score)
    if len(families) < MIN_FAMILIES_FOR_AUTO_MATCH:
        return _new_slot_result(cluster_id, cluster,
                                f"only {len(families)} family supports {top_slot}; "
                                "single-family auto-merge is forbidden (§4.1)")
    if margin < AUTO_MATCH_MIN_MARGIN:
        return _new_slot_result(cluster_id, cluster,
                                f"{top_slot} vs next by margin {margin:.2f} < "
                                f"{AUTO_MATCH_MIN_MARGIN} - ambiguous, keep separate")
    status = next((s.get("status") for s in slots if s["slot_id"] == top_slot), "pending")
    return {
        "slot_id": top_slot,
        "status": "approved" if status == "approved" else "candidate",
        "matched_via": "approved_slot" if status == "approved" else "pending_slot",
        "candidates": {top_slot: top_score},
        "margin": round(margin, 3),
        "families_supporting": sorted(set(families)),
        "basis": reasons + [f"cluster {cluster_id} attributes: "
                            + ", ".join(f"{k}={cluster.get(k)}"
                                        for k in ("gender", "age_band", "timbre"))],
        "not_speaker": sorted(not_speaker.get(cluster_id) or []),
    }


def _exact_attribute_bonus(cluster: Dict[str, Any], slot: Dict[str, Any]) -> float:
    profile = slot.get("profile") or {}
    return sum(1 for f in ("gender", "age_band", "timbre")
               if _compatible(str(cluster.get(f) or "unknown"), str(profile.get(f) or "unknown")) is True) * 0.1


def _entity_name(approved: Dict[str, Any], entity_id: Optional[str]) -> Optional[str]:
    if not entity_id:
        return None
    for entity in approved.get("entities") or []:
        if isinstance(entity, dict) and entity.get("id") == entity_id:
            return str(entity.get("canonical_name") or "") or None
    return None


def _new_slot_result(cluster_id: str, cluster: Dict[str, Any], why: str) -> Dict[str, Any]:
    return {
        "slot_id": None, "status": "unknown", "matched_via": "new_slot",
        "candidates": {}, "margin": 0.0, "families_supporting": [],
        "basis": [why],
        "new_slot_profile": {k: cluster.get(k, "unknown") for k in ("gender", "age_band", "timbre")},
        "reason": why,
    }


# --------------------------------------------------------------------------
# the resolver
# --------------------------------------------------------------------------

def resolve(ws: Path, ev: Dict[str, Any]) -> Dict[str, Any]:
    approved = ev["approved"]
    """Fold every cluster into a slot under §4.1 and emit the cast document."""
    slots = slots_from_approved(approved, ev.get("visual"))
    entities = [dict(e) for e in (approved.get("entities") or []) if isinstance(e, dict)]
    assignments: List[Dict[str, Any]] = []
    pending: List[Dict[str, Any]] = []
    new_slot_index = len(slots)

    for cluster_id, cluster in sorted((ev["acoustic"] or {}).items()):
        result = match_cluster(cluster_id, cluster, slots, ev)
        if result["slot_id"] is None:
            new_slot_index += 1
            slot_id = f"S{new_slot_index}"
            slots.append({
                "slot_id": slot_id,
                "profile": dict(result["new_slot_profile"]),
                "status": "pending", "entity_id": None,
                "coverage": {"clips_total": ev["visual"]["clips_total"],
                             "clips_visual_usable": ev["visual"]["clips_visual_usable"],
                             "clips_positive": ev["visual"]["clips_positive"]},
                "origin": "this_episode",
            })
            result["slot_id"] = slot_id
            pending.append({"slot_id": slot_id, "cluster_id": cluster_id,
                            "reason": result["reason"]})
        assignments.append({
            "cluster_id": cluster_id,
            "assignment": {k: result[k] for k in
                           ("slot_id", "status", "matched_via", "candidates", "margin",
                            "families_supporting", "basis")},
            # Ruled-out names travel with the cluster on every path, including
            # the new-slot path - a human reading `pending` needs to see that the
            # cluster already cannot BE the person it addressed.
            "not_speaker": sorted((ev["address"].get("not_speaker") or {}).get(cluster_id) or []),
            "speech_ms": cluster.get("speech_ms", 0),
            "line_count": cluster.get("line_count", 0),
        })

    doc = {
        "schema": CAST_SCHEMA,
        "series": approved.get("series") or Path(ws).name,
        "version": approved.get("version"),
        "table_version": approved.get("version"),
        "entities": entities,
        "slots": slots,
        "clusters": assignments,
        "pending": pending,
        "evidence_summary": {
            "acoustic_clusters": len(ev["acoustic"]),
            "address_terms": ev["address"]["terms"],
            "visual": {k: ev["visual"][k] for k in
                       ("clips_total", "clips_visual_usable", "clips_positive", "available")},
        },
        "gates": {
            "auto_match_min_margin": AUTO_MATCH_MIN_MARGIN,
            "min_families_for_auto_match": MIN_FAMILIES_FOR_AUTO_MATCH,
            "thresholds_are_measured": False,
            "cast_table_present": bool(entities),
        },
    }
    _assert_no_forged_names(doc)
    return doc


def _assert_no_forged_names(doc: Dict[str, Any]) -> None:
    """Gate: every name in the document must sit on a human-signed entity record."""
    named_slots = {s["slot_id"]: s for s in doc["slots"]}
    for entity in doc["entities"]:
        if str(entity.get("status") or "").lower() != "approved":
            continue
        if entity.get("approved_by") != "human" or not entity.get("approved_at"):
            raise AssertionError(
                f"entity {entity.get('id')!r} carries a name without a human sign-off record")
    for assignment in doc["clusters"]:
        slot = named_slots.get(assignment["assignment"]["slot_id"])
        if slot and slot.get("entity_id") and slot.get("origin") != "approved_table":
            raise AssertionError(
                f"slot {slot['slot_id']} has an entity_id but was not seeded from the "
                "approved table - the resolver must never name a slot it created")


def build_evidence(ws: Path) -> Dict[str, Any]:
    approved = series.load_approved(ws)
    speakers_doc = _load(ws, ".cache/audio/speakers.json")
    av_doc = _load(ws, ".cache/visual/av_notes.json")
    aligned_doc = _load(ws, ".cache/alignment/aligned_timeline.json")
    manifest = _load(ws, ".cache/alignment/scene_manifest.json") or {}
    known = list(series.approved_names(ws))
    if isinstance(manifest, dict):
        known += [str(n) for n in (manifest.get("bible_names") or []) if str(n).strip()]
        known += [str(n) for n in (manifest.get("characters_manifest") or {}) if str(n).strip()]
    return {
        "approved": approved,
        "acoustic": acoustic_evidence(speakers_doc),
        "address": address_evidence(aligned_doc, sorted(set(known))),
        "visual": visual_evidence(av_doc),
    }


def cmd_resolve(ws: Path, out_rel: str = ".cache/cast/cast.json",
                refresh_snapshot: bool = False) -> Dict[str, Any]:
    root = series.resolve_series_root(ws)
    if refresh_snapshot and root:
        series.bind_series(ws, root)
    approved_path = series.approved_cast_path(root) if root else None
    before = series.sha256_of(approved_path) if approved_path else None

    ev = build_evidence(ws)
    doc = resolve(ws, ev)

    out_path = ws / out_rel
    try:
        out_path.resolve().relative_to(ws.resolve())
    except ValueError:
        sys.stderr.write("[FATAL] cast.json path escaped the workspace containment.\n")
        sys.exit(EXIT_CONTAINMENT)
    _write_json_atomic(out_path, doc)

    if approved_path and series.sha256_of(approved_path) != before:
        sys.stderr.write("[FATAL] cast.approved.json was modified by resolution; aborting.\n")
        sys.exit(EXIT_NAMING_VIOLATION)

    doc["output"] = str(out_path)
    return doc


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve speaker identity into cast.json (no naming)")
    parser.add_argument("--workspace", "-w", required=True)
    parser.add_argument("--refresh-snapshot", action="store_true",
                        help="Re-snapshot the series cast table before resolving")
    parser.add_argument("--out", default=".cache/cast/cast.json")
    args = parser.parse_args()

    ws = Path(args.workspace).resolve()
    if not ws.is_dir():
        sys.stderr.write(f"[FATAL] Workspace does not exist: {ws}\n")
        sys.exit(EXIT_BAD_INPUT)
    doc = cmd_resolve(ws, args.out, args.refresh_snapshot)
    summary = {
        "status": "cast_resolved",
        "output": doc["output"],
        "clusters": len(doc["clusters"]),
        "slots": len(doc["slots"]),
        "pending_slots": len(doc["pending"]),
        "auto_matched": sum(1 for c in doc["clusters"]
                            if c["assignment"]["matched_via"] != "new_slot"),
        "new_slots": sum(1 for c in doc["clusters"]
                         if c["assignment"]["matched_via"] == "new_slot"),
        "cast_table_present": doc["gates"]["cast_table_present"],
        "thresholds_are_measured": doc["gates"]["thresholds_are_measured"],
        "visual_family_available": doc["evidence_summary"]["visual"]["available"],
    }
    sys.stdout.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if doc["pending"]:
        sys.stderr.write(
            f"[INFO] {len(doc['pending'])} 个槽位待人工签核（机器不给名字，见 §6 第 3-4 行）。\n")
    sys.exit(EXIT_OK)


if __name__ == "__main__":
    main()
