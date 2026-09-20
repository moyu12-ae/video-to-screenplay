#!/usr/bin/env python3
"""
scripts/speaker_labels.py - One taxonomy for dialogue-head speaker labels (v0.6 P1).

The pre-v0.6 lint recognised descriptive labels by ONE suffix shape
(`「…的声音」/「…之声」`) and treated everything else as a possible proper noun.
On ep02 that inverted the incentive: the writer invented no names and was still
flagged for 「关西腔者」「女声」「系统音」, while 「威严的声音」 passed - so the
cheapest way to satisfy the guard was to write a LONGER label, not a better-
evidenced one.

From v0.6 the direction is reversed: a label is descriptive by SHAPE (generic
role/kinship word, voice-of pattern, or a descriptor ending in a role/age/voice
noun), and only a label that looks like a NAME has to trace back to a signed-off
cast entry. Tracing is fatal only when a cast table exists to trace against;
without one the pipeline degrades to a warning and says so, because a gate that
cannot be satisfied makes people route around the whole process.

One definition lives here so the lint (splice) and the resolver (resolve_cast)
cannot drift into disagreeing about what "looks like a name" means.
"""

import re
from typing import Dict, Iterable, List, Optional, Set

# Generic, non-proper-noun speakers allowed without any cast-table backing. Only
# role, kinship and narration words belong here - production-specific descriptive
# labels must come from the workspace, not from this plugin-level list.
GENERIC_SPEAKERS: Set[str] = {
    "旁白", "解说", "画外音", "众人", "群臣", "众侍", "大家",
    "路人", "店员", "摊主", "店主", "顾客", "客人", "司机",
    "孩子", "家人", "家人们", "长辈", "少女", "少年", "同学",
    "侍从", "员工", "工作人员", "广播", "广播员", "广告",
    "童声", "呼声", "合唱", "神秘人物", "女声", "男声",
    "姐姐", "姐姐们", "哥哥", "父亲", "母亲", "奶奶", "爷爷",
}

# Voice-of patterns: 「王子的声音」/「威严之声」.
DESCRIPTIVE_SPEAKER_RE = re.compile(r"^[^（）]{1,12}(的声音|之声)$")
# A trailing qualifier is a writer's description, not a name: 「路人（男）」.
PAREN_QUALIFIER_RE = re.compile(r"[（(][^（）()]*[)）]\s*$")
# Descriptor tails. The stem must be >=2 chars so a real name ending in a
# descriptive tail ("凉音") still has to trace instead of passing unexamined.
# This is the deliberate trade-off: over-tracing costs a lookup, under-tracing
# costs a fabricated name in the delivered screenplay.
DESCRIPTIVE_SHAPE_RE = re.compile(
    r"^[^（）()]{2,12}(的声音|之声|者|声|音|人|人物|男|女|青年|少年|少女|儿童|孩子|"
    r"老人|老者|男人|女人|男生|女生|店员|摊主|店主|司机|学生|老师|警官|士兵|"
    r"机器人|主持|播报员|旁白|系统)$"
)
# 「A／B」 and 「A、B」 heads are two speakers; each part is judged on its own.
HEAD_SPLIT_RE = re.compile(r"[、/／]")

PROVISIONAL_LABEL_RE = re.compile(r"^SPEAKER_[A-Z0-9]+$")

LABEL_KIND_GENERIC = "generic"
LABEL_KIND_DESCRIPTIVE = "descriptive"
LABEL_KIND_NAME = "name"
LABEL_KIND_PROVISIONAL = "provisional"


def strip_qualifier(label: str) -> str:
    return PAREN_QUALIFIER_RE.sub("", label).strip()


def classify_label(label: str) -> str:
    """What kind of thing is this speaker label, structurally?"""
    token = (label or "").strip()
    if not token:
        return LABEL_KIND_DESCRIPTIVE
    if PROVISIONAL_LABEL_RE.match(token):
        return LABEL_KIND_PROVISIONAL
    if token in GENERIC_SPEAKERS or strip_qualifier(token) in GENERIC_SPEAKERS:
        return LABEL_KIND_GENERIC
    if DESCRIPTIVE_SPEAKER_RE.match(token) or DESCRIPTIVE_SHAPE_RE.match(token):
        return LABEL_KIND_DESCRIPTIVE
    if any(DESCRIPTIVE_SPEAKER_RE.match(p) or DESCRIPTIVE_SHAPE_RE.match(p)
           for p in split_head(token) if p):
        return LABEL_KIND_DESCRIPTIVE
    return LABEL_KIND_NAME


def is_descriptive(label: str) -> bool:
    return classify_label(label) in (LABEL_KIND_GENERIC, LABEL_KIND_DESCRIPTIVE)


def looks_like_name(label: str) -> bool:
    return classify_label(label) == LABEL_KIND_NAME


def split_head(head: str) -> List[str]:
    """A dialogue head can name several speakers; judge each part separately."""
    return [p.strip() for p in HEAD_SPLIT_RE.split((head or "").strip()) if p.strip()]


def trace(label: str, known_names: Iterable[str]) -> Optional[str]:
    """Return the known surface this label traces to, or None.

    Exact match first, then the benign decorations the writing contract allows:
    a stripped parenthetical, and a >=2-char known prefix whose remainder is a
    connector (bible「菈菈」 admits 「菈菈与妈妈」 but never 「茉里南」).
    """
    allowed: Set[str] = {str(n).strip() for n in known_names if str(n).strip()}
    token = (label or "").strip()
    if not token:
        return None
    if token in allowed:
        return token
    stripped = strip_qualifier(token)
    if stripped and stripped in allowed:
        return stripped
    for cand in (token, stripped):
        matches = [a for a in allowed
                   if len(a) >= 2 and cand.startswith(a) and _is_compound_tail(cand[len(a):])]
        if matches:
            return max(matches, key=len)
    return None


# A known surface may head a compound label (「菈菈与妈妈」), but only when what
# follows it is a connector or a qualifier. Without that guard 「茉里南」 would
# trace to 「茉里」, and a name-shaped fabrication is exactly what this gate exists
# to catch.
_COMPOUND_TAIL = ("与", "和", "跟", "及", "、", "／", "/", "（", "(", " ", "们")


def _is_compound_tail(tail: str) -> bool:
    return not tail or tail.startswith(_COMPOUND_TAIL)


def audit_head(head: str, known_names: Iterable[str]) -> Dict[str, List[str]]:
    """Split a head into parts that need no tracing and parts that failed to
    trace. The latter are the only candidates for a fatal naming error."""
    out = {"descriptive": [], "traced": [], "untraced": []}
    for part in split_head(head):
        kind = classify_label(part)
        if kind in (LABEL_KIND_GENERIC, LABEL_KIND_DESCRIPTIVE):
            out["descriptive"].append(part)
            continue
        found = trace(part, known_names)
        out["traced" if found else "untraced"].append(part)
    return out
