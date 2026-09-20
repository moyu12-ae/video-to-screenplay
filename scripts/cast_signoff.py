#!/usr/bin/env python3
"""
scripts/cast_signoff.py - The one human interaction the制度 exists for (v0.6 P5).

The machine's job is to narrow the question; the human's job is to answer it. So
this script never asks "who is this?" - it renders "one of these two, or skip",
and it refuses to write anything it had to guess at:

  round-based    4 slots per screen because that is what reads without scrolling,
                 then "continue?" - NOT a cap that strands half the cast unnamed
                 on episode one (a 35s slice of ep02 already yields 5 clusters).
  five outcomes  accepted / needs confirmation / ambiguous reference / unparsable /
                 user rejects the diff. Each has a distinct reply; "unparsable"
                 never becomes a silent skip, and nothing is ever half-written.
  names by hand  a name that is not among the candidates is created only after an
                 explicit confirmation, because inventing a name is the failure
                 this whole stage exists to make impossible.

Writes go to <series_root>/cast.approved.json atomically with a version history,
and the workspace keeps the snapshot it was resolved against, so "why does episode
5 call this person that?" stays answerable.
"""

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import series

SLOTS_PER_ROUND = 4
DECISION_ACCEPT = "accept"          # name came from this slot's own candidates
DECISION_CONFIRM_NEW = "confirm_new"  # a name nobody proposed - needs confirmation
DECISION_SKIP = "skip"
EXIT_OK = 0
EXIT_BAD_INPUT = 1
EXIT_NOT_BOUND = 3

_SLOT_REF = re.compile(r"(?<![A-Za-z0-9])S(\d+)(?![0-9])")
_ASSIGN = re.compile(r"S(\d+)\s*[=＝:：]\s*([^\s，,;；]+)")
_ORDER_WORDS = {"第一": 1, "第二": 2, "第三": 3, "第四": 4, "第五": 5,
                "第六": 6, "第七": 7, "第八": 8, "第九": 9, "第十": 10}
_SKIP_WORDS = ("跳过", "不知道", "略", "skip", "待定")
_AFFIRM = ("对", "是", "确认", "ok", "yes", "保持", "对的", "没问题")
# Bare interjections. Not filler to be skipped as "no answer": they are a human
# saying "keep what the machine described", so they must not become a NAME.
_FILLER = ("嗯", "哦", "啊", "呢", "吧", "好", "行", "可", "以", "哈")
# Anything spelled entirely out of acknowledgement characters ("好的", "可行", "嗯哈")
# is a human agreeing, never a cast member - checked before the length rule so a
# two-char reply cannot mint a name.
_ACK_CHARS = set("对是确认保持跳过好的行可以嗯哦啊呢吧哈没问题yesok")


def _label_for_cast_slot(cast_doc: Dict[str, Any], slot_id: str) -> Dict[str, Any]:
    for slot in cast_doc.get("slots") or []:
        if isinstance(slot, dict) and slot.get("slot_id") == slot_id:
            return slot
    return {}


def pending_slots(cast_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Slots that still need a human, in cluster order, deduplicated."""
    out: List[Dict[str, Any]] = []
    seen = set()
    for entry in cast_doc.get("pending") or []:
        slot_id = entry.get("slot_id") if isinstance(entry, dict) else entry
        if not slot_id or slot_id in seen:
            continue
        seen.add(slot_id)
        slot = _label_for_cast_slot(cast_doc, str(slot_id))
        cluster = (entry.get("cluster_id") if isinstance(entry, dict) else None) or ""
        excluded = _ruled_out_for(cast_doc, cluster)
        candidates = _candidates_for(cast_doc, str(slot_id), cluster)
        out.append({"slot_id": str(slot_id), "cluster_id": cluster,
                    "profile": slot.get("profile") or {}, "candidates": candidates,
                    # Shown so a human cannot re-commit the original mistake: this
                    # cluster spoke lines ADDRESSING these people, so it is not them.
                    "ruled_out": excluded,
                    "visual_votes": (entry.get("visual_votes") or {}) if isinstance(entry, dict) else {},
                    "visual_trusted": bool(entry.get("visual_trusted")) if isinstance(entry, dict) else False,
                    "reason": (entry.get("reason") if isinstance(entry, dict) else "") or ""})
    return out


def _candidates_for(cast_doc: Dict[str, Any], slot_id: str, cluster_id: str) -> List[str]:
    """Candidate NAMES for the human, never a guess.

    Two sources, both already vetted: entity names whose visual_label or acoustic
    profile this slot positively matched, and the address terms the resolver
    recorded (naming evidence - the doc forbids using them as attribution, and
    this is the naming side of that rule).
    """
    names: List[str] = []
    for cluster in cast_doc.get("clusters") or []:
        if not isinstance(cluster, dict):
            continue
        if cluster_id and cluster.get("cluster_id") != cluster_id:
            continue
        assignment = cluster.get("assignment") or {}
        for entity in cast_doc.get("entities") or []:
            if not isinstance(entity, dict):
                continue
            name = str(entity.get("canonical_name") or "")
            label = str(entity.get("visual_label") or "")
            if not name:
                continue
            if label and label in str(assignment.get("basis") or "") and name not in names:
                names.append(name)
    terms = (cast_doc.get("evidence_summary") or {}).get("address_terms") or {}
    for term, count in sorted(terms.items(), key=lambda kv: -int(kv[1] or 0)):
        if str(term) not in names:
            names.append(str(term))
    # An address term names the LISTENER. Showing it under the cluster that spoke
    # it would be the ep02 accident wearing a UI, so it is filtered out here too.
    ruled_out = set(_ruled_out_for(cast_doc, cluster_id))
    return [n for n in names[:3] if n not in ruled_out]


def _ruled_out_for(cast_doc: Dict[str, Any], cluster_id: str) -> List[str]:
    for cluster in cast_doc.get("clusters") or []:
        if isinstance(cluster, dict) and cluster.get("cluster_id") == cluster_id:
            return [str(t) for t in (cluster.get("not_speaker") or [])]
    return []


def render_table(slots: List[Dict[str, Any]]) -> str:
    rows = ["| 槽位 | 画像（机器给的证据） | 候选（最多 3 个） | 为什么需要你 |",
            "| :--- | :--- | :--- | :--- |"]
    for slot in slots:
        profile = slot.get("profile") or {}
        desc = "；".join(str(profile.get(k) or "") for k in ("gender", "age_band", "timbre")
                        if profile.get(k) and profile.get(k) != "unknown") or "（无可用画像）"
        cands = "／".join(slot.get("candidates") or []) or "（无候选，建议保留描述性标签）"
        ruled = slot.get("ruled_out") or []
        if ruled:
            cands += f"（不是 {'、'.join(ruled)}——这簇台词里在叫他们）"
        votes = slot.get("visual_votes") or {}
        if votes:
            seen = "、".join(f"{k}×{v}" for k, v in sorted(votes.items(), key=lambda kv: -kv[1])[:3])
            desc += f"｜说话期间动嘴的画像：{seen}"
            if not slot.get("visual_trusted"):
                desc += "（该信道未过 P3 遵从门，仅供你目视核对）"
        rows.append(f"| {slot['slot_id']} | {desc} | {cands} | {slot.get('reason') or '新出现'} |")
    return "\n".join(rows)


def _example(window: List[Dict[str, Any]]) -> str:
    """An example that names the slots actually on screen - the first draft showed
    `S1=托德 S2=奶奶` while the round displayed S3/S4, and the human's answer was
    then correctly rejected as 'no such slot this round'."""
    slots = [str(s.get("slot_id")) for s in window[:2]]
    if not slots:
        return "`（本轮无待答槽位）`"
    tail = " ".join(f"{sid}=跳过" for sid in slots[1:])
    return "`" + " ".join([f"{slots[0]}=名字"] + ([tail] if tail else [])) + "`"


def render_prompt(cast_doc: Dict[str, Any], offset: int = 0
                  ) -> Tuple[str, List[Dict[str, Any]], int]:
    """One round of the sign-off conversation, plus what remains."""
    pending = pending_slots(cast_doc)
    window = pending[offset:offset + SLOTS_PER_ROUND]
    table = render_table(window)
    remaining = len(pending) - offset - len(window)
    header = (f"本集识别出 {len((cast_doc.get('clusters') or []))} 个说话人："
              f"已定名 {sum(1 for c in (cast_doc.get('clusters') or [])
                            if (c.get('assignment') or {}).get('status') == 'approved')} 个、"
              f"待你定名 {len(pending)} 个。\n\n{table}\n\n"
              + "回话示例：" + _example(window) + "，或整句「"
              + (f"{window[0]['slot_id']} 那个金发的就叫…" if window else "…") + "」")
    if remaining > 0:
        header += f"\n\n（本轮只问 {SLOTS_PER_ROUND} 个，后面还有 {remaining} 个，看完这轮再问你要不要继续。）"
    return header, window, remaining


def parse_reply(reply: str, slots: List[Dict[str, Any]],
                approved_names: List[str]) -> Dict[str, Any]:
    """Turn natural language into decisions, or into a question.

    Ambiguity is returned as `needs_confirmation` rather than resolved by
    guessing - the whole point of the gate is that a name comes from a person,
    so a reply we cannot place must come back to them.
    """
    by_slot = {s["slot_id"]: s for s in slots}
    decisions: Dict[str, str] = {}
    needs_confirmation: List[Dict[str, Any]] = []
    unparsed: List[str] = []
    text = (reply or "").strip()
    if not text:
        return {"decisions": {}, "needs_confirmation": [], "unparsed": ["<空回复>"],
                "template": _template(slots)}

    handled: set = set()
    for m in _ASSIGN.finditer(text):
        slot_id, value = f"S{m.group(1)}", m.group(2).strip("。.！!")
        _record(slot_id, value, by_slot, approved_names, decisions, needs_confirmation, unparsed)
        handled.add(slot_id)
    confirmed = {n.get("slot_id") for n in needs_confirmation}
    for m in _SLOT_REF.finditer(text):
        slot_id = f"S{m.group(1)}"
        if slot_id in handled or slot_id in confirmed:
            continue
        handled.add(slot_id)
        value = _value_after_slot(text, m.start(), m.end())
        tail = text[m.end():]
        if value:
            _record(slot_id, value, by_slot, approved_names, decisions, needs_confirmation, unparsed)
        elif any(w in tail[:12] for w in _AFFIRM):
            _record(slot_id, "保持描述性标签", by_slot, approved_names, decisions,
                    needs_confirmation, unparsed)
    if not _SLOT_REF.search(text):
        for word, position in _ORDER_WORDS.items():
            if word in text and position <= len(by_slot):
                proposal = list(by_slot)[position - 1]
                needs_confirmation.append({
                    "slot_id": proposal, "name": None, "position": word,
                    "why": f"「{word}」按出现顺序读作 {proposal}，请确认槽位号再继续"})
                break
    if any(skip in text for skip in ("剩下的跳过", "都跳过", "全部跳过")):
        for slot_id in by_slot:
            if slot_id not in decisions:
                decisions[slot_id] = DECISION_SKIP
    leftovers = [tok for tok in re.split(r"[\s，,。;；]+", text)
                 if tok and not _SLOT_REF.search(tok) and tok not in decisions]
    if not decisions and not needs_confirmation and not unparsed:
        unparsed.append(text)
    unparsed[:] = list(dict.fromkeys(unparsed))
    return {"decisions": decisions, "needs_confirmation": needs_confirmation,
            "unparsed": unparsed, "template": _template(slots) if unparsed else "",
            "leftovers": leftovers}


def _template(slots: List[Dict[str, Any]]) -> str:
    return " ".join(f"{s['slot_id']}=" for s in slots) + "（填名字或 跳过）"


def _value_after_slot(text: str, start: int, end: int) -> str:
    tail = text[end:end + 24].strip(" 。,，：:")
    if not tail:
        return ""
    for stop in ("，", ",", "；", ";", " ", "。"):
        idx = tail.find(stop)
        if idx > 0:
            tail = tail[:idx]
            break
    return tail.strip("「」\"'（）()】")


def _record(slot_id: str, value: str, by_slot: Dict[str, Any], approved_names: List[str],
            decisions: Dict[str, str], needs_confirmation: List[Dict[str, Any]],
            unparsed: List[str]) -> None:
    if slot_id not in by_slot:
        unparsed.append(f"{slot_id}={value}（本轮没有这个槽位）")
        return
    candidates = [str(c) for c in by_slot[slot_id].get("candidates") or []]
    # A real candidate wins first, so a character actually named 好 or 可 cannot be
    # swallowed by the filler list below.
    if value and value in candidates:
        decisions[slot_id] = DECISION_ACCEPT
        decisions[f"{slot_id}:name"] = value
        return
    if not value or value in _AFFIRM or value in _FILLER or len(value) == 1\
            or set(value) <= _ACK_CHARS\
            or any(w in value for w in _SKIP_WORDS) or "描述性" in value or value == "保持":
        decisions[slot_id] = DECISION_SKIP
        return
    # 「S1 那个金发的叫托德」 is unambiguous even though the whole tail is not a
    # name: exactly one candidate appears inside it. Matching on ONE candidate
    # keeps this a read, not a guess - two candidates go to needs_confirmation.
    inside = [c for c in candidates if c and c in value]
    if len(inside) == 1:
        decisions[slot_id] = DECISION_ACCEPT
        decisions[f"{slot_id}:name"] = inside[0]
        return
    # A name the table already knows may still be buried in a chatty reply
    # ("就叫托德"). Surfacing it as the PROPOSED name keeps the question honest -
    # it is still confirmation-gated, but the human reads a name, not a sentence.
    inside_table = [n for n in approved_names if n and n in value]
    if len(inside_table) == 1:
        value = inside_table[0]
    needs_confirmation.append({
        "slot_id": slot_id, "name": value,
        "why": (f"「{value}」不在 {slot_id} 的候选里"
                + ("（但它已在演员表里）" if value in approved_names else "（表也没有这个条目）")
                + "。确认为该槽位新建/改判这个定名吗？")})


def load_history(series_root: Path) -> List[Dict[str, Any]]:
    doc = series.read_json(series_root / series.CAST_FILENAME)
    return list(doc.get("history") or []) if isinstance(doc, dict) else []


def apply_round(series_root: Path, cast_doc: Dict[str, Any], slots: List[Dict[str, Any]],
                decisions: Dict[str, str]) -> Dict[str, Any]:
    """Write the accepted names, append a history entry, return the diff."""
    # Read the series file directly: load_approved resolves through a WORKSPACE
    # pointer, and a sign-off writes to the series root itself.
    table = series.read_json(series_root / series.CAST_FILENAME) or {}
    entities: List[Dict[str, Any]] = [dict(e) for e in (table.get("entities") or [])
                                      if isinstance(e, dict)]
    by_name = {str(e.get("canonical_name") or ""): e for e in entities}
    diff: List[Dict[str, Any]] = []
    for slot in slots:
        slot_id = slot["slot_id"]
        kind = decisions.get(slot_id)
        if kind in (None, DECISION_SKIP):
            continue
        name = decisions.get(f"{slot_id}:name")
        if not name:
            continue
        existing = by_name.get(name)
        if existing is None:
            entity = {"id": f"C{len(entities) + 1}", "canonical_name": name,
                      "status": "approved", "aliases": [],
                      "voice_profile": {k: slot.get("profile", {}).get(k, "unknown")
                                        for k in ("gender", "age_band", "timbre")},
                      "visual_anchors": [], "evidence": [
                          {"kind": "signoff", "slot_id": slot_id,
                           "cluster_id": slot.get("cluster_id"), "date": date.today().isoformat()}],
                      "approved_by": "human", "approved_at": date.today().isoformat()}
            entities.append(entity)
            by_name[name] = entity
            diff.append({"slot_id": slot_id, "change": "new_entity", "name": name})
        else:
            diff.append({"slot_id": slot_id, "change": "linked_existing", "name": name})
        existing = by_name[name]
        # The lineage record is what makes the link survive the next resolve: an
        # entity that merely EXISTS in the table says nothing about which speaker
        # it is, and that distinction is the whole ep02 accident.
        record = {"kind": "signoff", "slot_id": slot_id, "cluster_id": slot.get("cluster_id"),
                  "date": date.today().isoformat()}
        evidence = existing.setdefault("evidence", [])
        if isinstance(evidence, list) and not any(
                isinstance(e, dict) and e.get("kind") == "signoff"
                and e.get("slot_id") == slot_id and e.get("cluster_id") == record["cluster_id"]
                for e in evidence):
            evidence.append(record)
        label = (slot.get("profile") or {}).get("visual") or ""
        if label and not existing.get("visual_label"):
            existing["visual_label"] = str(label)
            diff.append({"slot_id": slot_id, "change": "visual_label", "name": str(label)})

    history = load_history(series_root)
    version = str(len(history) + 1)
    doc = {
        "schema": "vts-cast/v1",
        "series": cast_doc.get("series"),
        "version": version,
        "entities": entities,
        "history": history + [{"version": version, "date": date.today().isoformat(),
                               "diff": diff}],
    }
    series.write_json_atomic(series_root / series.CAST_FILENAME, doc)
    return {"status": "signed_off", "version": version, "diff": diff,
            "approved_entity_count": len(entities)}


def cmd_draft_banner(cast_doc: Optional[Dict[str, Any]], enforced: bool) -> str:
    """What the delivered screenplay must say when the table was not signed off."""
    if not enforced:
        return "演员表未经核验（本工作区未绑定已签核演员表）"
    pending = len((cast_doc or {}).get("pending") or [])
    if pending:
        return f"演员表未经核验（{pending} 个槽位为候选）"
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Cast sign-off: render, parse, apply")
    parser.add_argument("--workspace", "-w", required=True)
    parser.add_argument("--render", action="store_true", help="Print this round's table")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--cast", default=".cache/cast/cast.json")
    parser.add_argument("--reply", default=None, help="Natural-language answer to parse")
    parser.add_argument("--apply", action="store_true",
                        help="Write a parsed reply into the series cast table")
    parser.add_argument("--confirm", action="append", default=[], metavar="SLOT=NAME",
                        help="Names the user confirmed after being asked (repeatable); "
                             "without them a round holding needs_confirmation writes nothing")
    args = parser.parse_args()

    ws = Path(args.workspace).resolve()
    root = series.resolve_series_root(ws)
    if not root:
        sys.stderr.write("[FATAL] 本工作区未绑定系列目录，演员表无处可写。"
                         "先运行 workspace.py series --bind <系列根>。\n")
        sys.exit(EXIT_NOT_BOUND)
    cast_path = ws / args.cast
    if not cast_path.is_file():
        sys.stderr.write(f"[FATAL] 缺少 {cast_path}，先运行 resolve_cast.py。\n")
        sys.exit(EXIT_BAD_INPUT)
    cast_doc = json.loads(cast_path.read_text(encoding="utf-8"))

    if args.reply is not None:
        window = pending_slots(cast_doc)[args.offset:args.offset + SLOTS_PER_ROUND]
        outcome = parse_reply(args.reply, window, series.approved_names(ws))
        outcome["slots_in_round"] = [s["slot_id"] for s in window]
        for pair in args.confirm:
            slot_id, _, name = pair.partition("=")
            slot_id, name = slot_id.strip(), name.strip()
            if not slot_id or not name:
                continue
            if slot_id not in {w["slot_id"] for w in window}:
                continue
            outcome["decisions"][slot_id] = DECISION_ACCEPT
            outcome["decisions"][f"{slot_id}:name"] = name
            outcome["needs_confirmation"] = [n for n in outcome["needs_confirmation"]
                                             if n.get("slot_id") != slot_id]
        sys.stdout.write(json.dumps(outcome, ensure_ascii=False, indent=2) + "\n")
        if args.apply:
            if outcome["needs_confirmation"] or outcome["unparsed"]:
                sys.stderr.write("[INFO] 有未确认或未解析的条目，本轮未写盘（演员表要么完整接受，要么不动）。\n")
                sys.exit(EXIT_OK)
            result = apply_round(root, cast_doc, window, outcome["decisions"])
            sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            sys.stderr.write("[INFO] 已写入，请核对上面 diff；不对就运行 "
                             "workspace.py 的历史版本恢复（cast.approved.json 带 history）。\n")
        sys.exit(EXIT_OK)

    prompt, window, remaining = render_prompt(cast_doc, args.offset)
    sys.stdout.write(prompt + "\n")
    if args.render:
        sys.stdout.write(json.dumps({"slots_in_round": [s["slot_id"] for s in window],
                                     "remaining": remaining,
                                     "slots_per_round": SLOTS_PER_ROUND},
                                    ensure_ascii=False) + "\n")
    sys.exit(EXIT_OK)


if __name__ == "__main__":
    main()
