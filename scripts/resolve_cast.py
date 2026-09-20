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
# Equal weights on purpose: the design doc's §11.7 says the visual family's weight
# must come from measured precision/recall, and there is no such number yet. Equal
# weighting is the honest neutral, and `thresholds_are_measured: false` travels with
# every document produced under it.
FAMILY_WEIGHTS: Dict[str, float] = {family: 1.0 for family in FAMILIES}


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
                # Only mouth motion caused by SPEECH is speaker evidence; chewing
                # is mouth motion too and was the majority of "moving" in ep02.
                if state == "moving" and str(action.get("mouth_motion") or "") == "speaking" \
                        and str(action.get("who") or "").strip():
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


def cluster_visual_votes(speakers_doc: Any, av_doc: Any) -> Dict[str, Dict[str, int]]:
    """Per-cluster visual positives, taken from THAT cluster's speech windows.

    This is the shape the design asked for: to ask whether A1 is the blond young
    man, evaluate the frames where A1 is speaking and nobody else's. Aggregating
    mouth evidence over the whole episode instead would let a person talking in
    another cluster vote for this one, and would make the over-split audit
    ("two clusters, same face") impossible to compute at all.
    """
    actions: List[Dict[str, Any]] = []
    for note in ((av_doc or {}).get("scene_notes") or []) if isinstance(av_doc, dict) else []:
        if not isinstance(note, dict):
            continue
        for seg in note.get("segments") or []:
            if not isinstance(seg, dict):
                continue
            actions.extend(a for a in ((seg.get("visual") or {}).get("actions") or [])
                           if isinstance(a, dict))
    turns = [t for t in ((speakers_doc or {}).get("speech_turns") or [])
             if isinstance(t, dict)] if isinstance(speakers_doc, dict) else []
    out: Dict[str, Dict[str, int]] = {}
    for turn in turns:
        cluster = str(turn.get("cluster_id") or "")
        if not cluster:
            continue
        t0, t1 = int(turn.get("start_ms", 0)), int(turn.get("end_ms", 0))
        for action in actions:
            if str(action.get("mouth_state")) != "moving" \
                    or str(action.get("mouth_motion") or "") != "speaking":
                continue
            a0, a1 = int(action.get("start", 0)), int(action.get("end", 0))
            if min(t1, a1) - max(t0, a0) <= 0:
                continue
            label = str(action.get("who") or "").strip()
            if not label:
                continue
            bucket = out.setdefault(cluster, {})
            bucket[label] = bucket.get(label, 0) + 1
    return out



# --------------------------------------------------------------------------
# P4: same-base distributions and their merge
# --------------------------------------------------------------------------

def normalize_distribution(counts: Dict[str, float], unobserved: float) -> Dict[str, float]:
    """p_f(x) = n_f(x) / (SUM_y n_f(y) + u_f).

    `unobserved` goes in the DENOMINATOR and never the numerator: a window where
    nothing could be seen flattens this family's distribution - i.e. it reduces
    how much this family is allowed to say - instead of voting against anybody.
    A family with no observations at all returns {} so it is dropped from the
    merge entirely; multiplying by an all-zero distribution would zero every
    candidate and leave the margin as 0/0.
    """
    total = float(sum(counts.values()))
    if total <= 0:
        return {}
    denom = total + max(0.0, float(unobserved))
    return {k: v / denom for k, v in counts.items() if v > 0}


def merge_distributions(distributions: Dict[str, Dict[str, float]],
                        weights: Dict[str, float]) -> Dict[str, float]:
    """score(x) = PRODUCT_f p_f(x) ** w_f, i.e. a weighted sum in log space.

    A candidate a family does not mention gets p=0 for that family, which is what
    "narrow the candidate set" means operationally: one family's confident zero
    takes a candidate out of the running, while its silence (an absent family)
    does not. Weights are equal until §11.7's measurement lands - and that
    honest-default is exactly why `thresholds_are_measured` ships as False.
    """
    candidates = set()
    for dist in distributions.values():
        candidates |= set(dist)
    scores: Dict[str, float] = {}
    for cand in candidates:
        score = 1.0
        for family, dist in distributions.items():
            weight = float(weights.get(family, 1.0))
            score *= float(dist.get(cand, 0.0)) ** weight
        scores[cand] = score
    return scores


def distribution_margin(scores: Dict[str, float]) -> Tuple[float, List[str]]:
    """margin = (top1 - top2) / SUM, plus the ranked candidates.

    Dividing by the total rather than by top1 keeps a flat 1-vs-1 tie at 0 no
    matter how many observations there were; a large absolute lead that is only
    half the mass still reads as weak, which is the point.
    """
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    total = sum(scores.values())
    if not ranked or total <= 0:
        return 0.0, [k for k, _ in ranked]
    top = ranked[0][1]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    return (top - second) / total, [k for k, _ in ranked]


def acoustic_distribution(cluster: Dict[str, Any], slots: List[Dict[str, Any]]
                          ) -> Tuple[Dict[str, float], float]:
    """Per-slot count of attributes both sides actually report and agree on."""
    counts: Dict[str, float] = {}
    unobserved = 0.0
    for slot in slots:
        profile = slot.get("profile") or {}
        agree = 0
        for field in ("gender", "age_band", "timbre"):
            verdict = _compatible(str(cluster.get(field) or "unknown"),
                                  str(profile.get(field) or "unknown"))
            if verdict is True:
                agree += 1
            elif verdict is None:
                unobserved += 1
        if agree:
            counts[str(slot["slot_id"])] = float(agree)
    return counts, unobserved


def visual_distribution(slots: List[Dict[str, Any]], anchors: Dict[str, Any],
                        clips_total: int, available: bool,
                        cluster_votes: Optional[Dict[str, int]] = None
                        ) -> Tuple[Dict[str, float], float]:
    """Per-slot count of clips where that slot's own label was seen talking.

    `still` and `not_visible` are deliberately absent: under limited animation a
    closed mouth over one second proves little, and an unseen face proves nothing.
    """
    if not available:
        return {}, 0.0
    positive = cluster_votes if cluster_votes is not None \
        else ((anchors or {}).get("positive_by_anchor") or {})
    counts: Dict[str, float] = {}
    seen = 0.0
    for slot in slots:
        label = slot.get("visual_label")
        hits = int(positive.get(label, 0) or 0) if label else 0
        seen += hits
        if hits:
            counts[str(slot["slot_id"])] = float(hits)
    return counts, float(max(0, int(clips_total) - int(seen)))

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

    ① and ② both require >=2 supporting families and a clear margin on the merged
    distribution; anything else is ambiguity, and ambiguity creates a pending slot
    instead of merging - a wrong merge is inherited by every later episode.
    """
    not_speaker = ev["address"].get("not_speaker") or {}
    ruled_out = set(not_speaker.get(cluster_id) or [])
    entity_of = {s["slot_id"]: _entity_name(ev["approved"], s.get("entity_id")) for s in slots}
    eligible = [s for s in slots if not (entity_of.get(s["slot_id"]) and
                                         entity_of[s["slot_id"]] in ruled_out)]
    visual = ev["visual"]
    ac_counts, ac_unobserved = acoustic_distribution(cluster, eligible)
    per_cluster = visual.get("by_cluster")
    if isinstance(per_cluster, dict):
        # A cluster whose speech windows overlap no speaking face must get an EMPTY
        # distribution, not the episode-wide counts. `.get()` returning None used to
        # fall back to global positives, which is precisely the cross-cluster
        # contamination this exists to remove (measured: two off-screen clusters
        # inherited 金发青年's votes).
        cluster_votes = per_cluster.get(cluster_id, {})
    else:
        cluster_votes = None  # pre-v4 artifact: no per-window evidence at all
    vis_counts, vis_unobserved = visual_distribution(
        eligible, visual.get("anchors") or {}, int(visual.get("clips_total") or 0),
        bool(visual.get("available")), cluster_votes)
    distributions = {
        "acoustic": normalize_distribution(ac_counts, ac_unobserved),
        "visual": normalize_distribution(vis_counts, vis_unobserved),
    }
    distributions = {f: d for f, d in distributions.items() if d}
    if not distributions:
        return _new_slot_result(cluster_id, cluster,
                                "no slot has any positive evidence"
                                + (f"; {len(slots) - len(eligible)} slot(s) ruled out by 呼语"
                                   if len(eligible) != len(slots) else ""))

    scores = merge_distributions(distributions, FAMILY_WEIGHTS)
    margin, ranked = distribution_margin(scores)
    winner = ranked[0]
    families = sorted(f for f, dist in distributions.items() if dist.get(winner, 0) > 0)
    reasons = []
    if distributions.get("acoustic"):
        reasons.append("acoustic attributes agreed with " + ", ".join(
            s["slot_id"] for s in eligible if ac_counts.get(str(s["slot_id"]))))
    if distributions.get("visual"):
        reasons.append("visual mouth evidence on that slot's own label")
    candidates = {slot_id: round(scores[slot_id], 4) for slot_id in ranked if scores[slot_id] > 0}

    if len(families) < MIN_FAMILIES_FOR_AUTO_MATCH:
        return _new_slot_result(cluster_id, cluster,
                                f"only {len(families)} family supports {winner}; "
                                "single-family auto-merge is forbidden (§4.1)",
                                candidates=candidates, families=families, margin=margin)
    if margin < AUTO_MATCH_MIN_MARGIN:
        return _new_slot_result(cluster_id, cluster,
                                f"{winner} vs next by margin {margin:.2f} < "
                                f"{AUTO_MATCH_MIN_MARGIN} - ambiguous, keep separate",
                                candidates=candidates, families=families, margin=margin)

    status = next((s.get("status") for s in slots if s["slot_id"] == winner), "pending")
    winner_slot = next((s for s in slots if s["slot_id"] == winner), {})
    return {
        # The lineage the naming gate checks: this cluster, this slot, and the
        # signed-off entity behind it. A name that is merely PRESENT in the table
        # is not lineage - that distinction is what catches the ep02 accident
        # (茉里 is a real character; A1 was never signed off as her).
        "entity_id": winner_slot.get("entity_id"),
        "entity_name": _entity_name(ev["approved"], winner_slot.get("entity_id")),
        "slot_id": winner,
        "status": "approved" if status == "approved" else "candidate",
        "matched_via": "approved_slot" if status == "approved" else "pending_slot",
        "candidates": candidates,
        "margin": round(margin, 3),
        "families_supporting": families,
        "basis": reasons + [f"cluster {cluster_id} attributes: "
                            + ", ".join(f"{k}={cluster.get(k)}"
                                        for k in ("gender", "age_band", "timbre"))],
        "visual_votes": dict(distributions.get("visual") or {}),
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


def _new_slot_result(cluster_id: str, cluster: Dict[str, Any], why: str,
                     candidates: Optional[Dict[str, float]] = None,
                     families: Optional[List[str]] = None,
                     margin: float = 0.0) -> Dict[str, Any]:
    return {
        # The narrowed candidate set survives even when the cluster gets its own
        # slot: "it is one of these two" is the information the evidence actually
        # carries, and dropping it would throw away the channel's whole value.
        "slot_id": None, "status": "unknown", "matched_via": "new_slot",
        "candidates": candidates or {}, "margin": round(margin, 3),
        "families_supporting": families or [],
        "basis": [why],
        "new_slot_profile": {k: cluster.get(k, "unknown") for k in ("gender", "age_band", "timbre")},
        "reason": why,
        "visual_votes_this_cluster": {},
    }


# --------------------------------------------------------------------------
# the resolver
# --------------------------------------------------------------------------

def resolve(ws: Path, ev: Dict[str, Any]) -> Dict[str, Any]:
    approved = ev["approved"]
    """Fold every cluster into a slot under §4.1 and emit the cast document."""
    # A sign-off record is how a name gets back into the loop: cast_signoff writes
    # {kind: "signoff", cluster_id, slot_id} with approved_by: "human", and honoring
    # it is the ONLY way a slot created in this episode can carry an entity.
    signoffs: Dict[str, Dict[str, Any]] = {}
    for entity in approved.get("entities") or []:
        if not isinstance(entity, dict) or str(entity.get("status") or "").lower() != "approved":
            continue
        for record in entity.get("evidence") or []:
            if isinstance(record, dict) and record.get("kind") == "signoff" and record.get("cluster_id"):
                signoffs[str(record["cluster_id"])] = entity
    slots = slots_from_approved(approved, ev.get("visual"))
    entities = [dict(e) for e in (approved.get("entities") or []) if isinstance(e, dict)]
    assignments: List[Dict[str, Any]] = []
    pending: List[Dict[str, Any]] = []
    new_slot_index = len(slots)
    visual_votes_by_cluster: Dict[str, Dict[str, float]] = {}

    for cluster_id, cluster in sorted((ev["acoustic"] or {}).items()):
        result = match_cluster(cluster_id, cluster, slots, ev)
        visual_votes_by_cluster[cluster_id] = dict(result.get("visual_votes") or {})
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
            signed = signoffs.get(cluster_id)
            if signed:
                slots[-1]["entity_id"] = signed.get("id")
                slots[-1]["status"] = "approved"
                result["status"] = "approved"
                result["matched_via"] = "human_signoff"
                result["entity_id"] = signed.get("id")
                result["entity_name"] = str(signed.get("canonical_name") or "") or None
                result["basis"] = list(result.get("basis") or []) + [
                    f"human sign-off record on {cluster_id} -> {result['entity_name']}"]
            else:
                pending.append({"slot_id": slot_id, "cluster_id": cluster_id,
                                "reason": result["reason"],
                                # What the user should actually look at: who was seen
                                # talking during THIS cluster's speech windows. Shown
                                # even when the channel is downgraded, because a human
                                # reading "chewing" next to "moving" is exactly how that
                                # confounder gets caught.
                                "visual_votes": dict((ev["visual"].get("by_cluster") or {})
                                                     .get(cluster_id) or {}),
                                "visual_trusted": bool(ev["visual"].get("available"))})
        visual_top = None
        if visual_votes_by_cluster.get(cluster_id):
            visual_top = max(visual_votes_by_cluster[cluster_id].items(),
                             key=lambda kv: (-kv[1], kv[0]))[0]
        assignments.append({
            "cluster_id": cluster_id,
            "visual_top_slot": visual_top,
            "assignment": {k: result.get(k) for k in
                           ("slot_id", "status", "matched_via", "candidates", "margin",
                            "families_supporting", "basis", "entity_id", "entity_name")},
            # Ruled-out names travel with the cluster on every path, including
            # the new-slot path - a human reading `pending` needs to see that the
            # cluster already cannot BE the person it addressed.
            "not_speaker": sorted((ev["address"].get("not_speaker") or {}).get(cluster_id) or []),
            "speech_ms": cluster.get("speech_ms", 0),
            "line_count": cluster.get("line_count", 0),
        })

    # Reverse audit of the separation layer: one person spread over several
    # clusters is the anomaly we CAN see (two clusters, same face seen talking).
    # The opposite - one cluster carrying several people - is invisible from this
    # evidence, so no field claims to detect it.
    by_visual_top: Dict[str, List[str]] = {}
    for assignment in assignments:
        top = assignment.get("visual_top_slot")
        if top:
            by_visual_top.setdefault(top, []).append(assignment["cluster_id"])
    over_split = [{"slot_id": slot, "clusters": sorted(clusters),
                   "note": "两个人类簇的视觉正证据指向同一画像，怀疑声学过度切分"}
                  for slot, clusters in sorted(by_visual_top.items()) if len(clusters) > 1]

    abstained = [a for a in assignments if a["assignment"]["status"] == "unknown"]
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
        "over_split_suspects": over_split,
        "abstention": {"clusters": len(assignments), "abstained": len(abstained),
                       "rate": round(len(abstained) / len(assignments), 3) if assignments else 0.0},
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
    signed_clusters = {c["cluster_id"] for c in doc["clusters"]
                       if c["assignment"].get("matched_via") == "human_signoff"}
    for assignment in doc["clusters"]:
        slot = named_slots.get(assignment["assignment"]["slot_id"])
        if not slot or not slot.get("entity_id"):
            continue
        if slot.get("origin") == "approved_table":
            continue
        if assignment["cluster_id"] in signed_clusters:
            continue  # a human attached this name to this cluster; that is legal
        raise AssertionError(
            f"slot {slot['slot_id']} has an entity_id but was not seeded from the approved "
            "table and carries no human sign-off record - the resolver must never name a slot")


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
    visual = visual_evidence(av_doc)
    # Pending slots have no signed-off visual_label yet, so per-cluster votes cannot
    # attach to anything: on episode one the visual family is structurally unable to
    # vote, which is why the sign-off sheet must show the user what each cluster was
    # seen doing rather than pretending a distribution exists.
    visual["by_cluster"] = cluster_visual_votes(speakers_doc, av_doc)
    return {
        "approved": approved,
        "acoustic": acoustic_evidence(speakers_doc),
        "address": address_evidence(aligned_doc, sorted(set(known))),
        "visual": visual,
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
        "abstained": doc["abstention"]["abstained"],
        "abstention_rate": doc["abstention"]["rate"],
        "over_split_suspects": len(doc["over_split_suspects"]),
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
