#!/usr/bin/env python3
"""
scripts/vocatives.py - Address-term extraction from subtitles (v0.6 P1).

A name in address position points at the person being TALKED TO, never at the
person talking. That single fact is the cheapest speaker evidence available - it
costs no API call, only a string scan - and it is also the mistake that started
this whole design: on ep02 scene 1 a line beginning 「茉里 你交朋友了」 was written
as 茉里 speaking, when 茉里 is the addressee and the speaker was the blond young
man. So this module is deliberately narrow about what it emits:

  role="negative"  the cluster that SPOKE this line is not the person named
  role="naming"    the term is a candidate NAME for some cluster (whose, this
                   module does not and must not decide)

There is no third role. An "attribution" edge - term used to pick the speaker of
the line - is exactly the forbidden direction, so nothing here can produce it and
a test asserts the literal string never appears in the output.

Detection is against a KNOWN name list (approved cast + aliases + bible/manifest
names) rather than an open NLP parse: an open parse would have to decide what
counts as a name, which is the very judgement we took away from the model.
"""

import re
from typing import Any, Dict, Iterable, List, Set

ROLES = ("naming", "negative")

# Separators that mark a term as being called out rather than talked about:
# 「茉里，你听我说」 / 「我知道了，茉里」 / 「茉里 你交朋友了」.
_SEP_CHARS = "，,、：:！!？?…—·~～  \t\"“”「」『』（）()【】"
_SEP_CLASS = re.escape(_SEP_CHARS)
_BOUNDARY = f"(?:^|[{_SEP_CLASS}])"
_AFTER = f"(?:$|[{_SEP_CLASS}])"
_VOCATIVE_TPL = _BOUNDARY + "({term})" + _AFTER

# A term is only an address when something else survives in the line: a bare
# 「茉里！」 caption, a shot title, or the speaker's own name in a heading is
# someone being named, not someone being spoken to.
_MIN_REMAINDER_CHARS = 1


def _escape(term: str) -> str:
    return re.escape(term)


def compile_patterns(terms: Iterable[str]) -> List[Dict[str, Any]]:
    """Longest terms first, so 「茉里子」 wins over 「茉里」 at the same position."""
    ordered = sorted({str(t).strip() for t in terms if str(t).strip()}, key=len, reverse=True)
    return [{"term": t, "re": re.compile(_VOCATIVE_TPL.format(term=_escape(t)), re.M)}
            for t in ordered]


def extract_address_terms(lines: List[Dict[str, Any]], known_names: Iterable[str],
                          text_key: str = "text", index_key: str = "index",
                          speaker_key: str = "cluster_id") -> Dict[str, Any]:
    """Scan subtitle lines for names in address position.

    `lines` are dicts carrying at least text + index; `cluster_id` is optional
    (without it, negative evidence cannot be attributed to anyone and only the
    term tally is produced).
    """
    patterns = compile_patterns(known_names)
    events: List[Dict[str, Any]] = []
    terms: Dict[str, Dict[str, Any]] = {}
    if not patterns:
        return {"events": [], "terms": {}, "lines_scanned": 0,
                "known_names_used": 0}

    for line in lines:
        text = str(line.get(text_key) or "")
        if not text:
            continue
        speaker = line.get(speaker_key)
        for entry in patterns:
            term = str(entry["term"])
            for m in entry["re"].finditer(text):
                remainder = (text[:m.start()] + text[m.end():]).strip(_SEP_CHARS + "\n").strip()
                if len(remainder) < _MIN_REMAINDER_CHARS:
                    continue
                position = "start" if not text[:m.start()].strip(_SEP_CHARS) else (
                    "end" if not text[m.end():].strip(_SEP_CHARS) else "mid")
                rec = {"line_index": line.get(index_key), "term": term,
                       "position": position, "speaker_cluster": speaker,
                       "text": text}
                if speaker:
                    rec["role"] = "negative"
                else:
                    rec["role"] = "naming"
                events.append(rec)
                bucket = terms.setdefault(term, {"term": term, "count": 0,
                                                 "spoken_by": {}, "positions": {},
                                                 "roles": set()})
                bucket["count"] += 1
                if speaker:
                    bucket["spoken_by"][str(speaker)] = bucket["spoken_by"].get(str(speaker), 0) + 1
                bucket["positions"][position] = bucket["positions"].get(position, 0) + 1
                bucket["roles"].add(rec["role"])

    for bucket in terms.values():
        bucket["roles"] = sorted(bucket["roles"])
    return {"events": events, "terms": terms, "lines_scanned": len(lines),
            "known_names_used": len(patterns)}


def negative_evidence(result: Dict[str, Any]) -> Dict[str, Set[str]]:
    """cluster -> terms that cluster cannot BE (it spoke lines addressing them)."""
    out: Dict[str, Set[str]] = {}
    for ev in result.get("events") or []:
        speaker = ev.get("speaker_cluster")
        if speaker and ev.get("role") == "negative":
            out.setdefault(str(speaker), set()).add(str(ev["term"]))
    return out


def naming_candidates(result: Dict[str, Any]) -> Dict[str, int]:
    """term -> how many times someone was addressed by it.

    A count only; which cluster it names is the resolver's job, and 'attribution'
    is not among the roles this module can emit (see ROLES).
    """
    return {term: int(bucket.get("count") or 0) for term, bucket in
            (result.get("terms") or {}).items()}
