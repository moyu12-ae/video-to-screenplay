---
description: Splice verbatim dialogue into the scene drafts, assemble the episode screenplay, and run the delivery checklist before handing the document over.
argument-hint: <workspace path> --title "第 N 话 …"
---

# /video-to-screenplay:splice

Stage 5 only. Assumes every `draft_status == "missing"` scene has been authored
(`/video-to-screenplay:build`). Details in `skills/video-to-screenplay/SKILL.md` §阶段 5.

```bash
python3 scripts/splice_screenplay.py --workspace "<ws>" --title "第 N 话 …"
```

Each `[[SUB:n]]` is replaced by the verbatim subtitle text, so fidelity is structural — paraphrase is
not representable. Missing, duplicated, out-of-order or orphaned placeholders are **fatal** and name the
offending indices; a scene inside an OP/ED window carries the one-line （动画 OP）/（动画 ED） marker
instead of narrative content.

## Delivery checklist (all must hold before reporting done)

- Exit code 0 and `dialogue_spliced == dialogue_total`.
- Every speaker-name lint warning accounted for; production-specific epithets go into
  `materials/bible.json` → `speaker_whitelist`, never into the plugin.
- Output document lands in `<ws>/output/`; nothing scattered in the project root; zero `SPEAKER_`
  strings in the final `.md`.
- State to the user: subtitle source tier, shot/scene counts, whether the AV-understanding pass ran, and
  any degraded inputs (e.g. blank speaker columns when no key was configured) — never present a degraded
  artifact as a full one.
