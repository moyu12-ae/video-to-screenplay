---
description: Create the workspace for an episode, run the environment and API-key preflight, and settle the subtitle decision gate before anything is computed.
argument-hint: <workspace path> [video path]
---

# /video-to-screenplay:init

Stage 1 only: set up, then stop for the two gates. Full rules live in
`skills/video-to-screenplay/SKILL.md` §阶段 1 — read it before deviating from the steps below.

Paths below are relative to this plugin's root (the directory holding
`skills/video-to-screenplay/SKILL.md`).

## Steps

```bash
python3 scripts/workspace.py init  --workspace "<ws>"
python3 scripts/workspace.py probe --workspace "<ws>"
python3 scripts/workspace.py doctor
```

1. **Key gate.** `doctor` reports `diarization.dashscope_api_key` as `set`/`missing` (never the value)
   plus the host and model a key would be sent to. On `missing`, ask the user to choose — configure the
   key, or continue with blank speaker columns, or stop. Never degrade silently.
2. **Egress disclosure.** Before any Omni pass is enabled, state what leaves the machine per
   `SECURITY.md` and get confirmation.
3. **Subtitle gate.** Decide the tier: external `.srt`/`.ass` file → embedded soft stream → OCR. If only
   OCR can serve the source, the extractor exits 6 with the Tier 3 instruction on stdout; follow it.
   Refuse dialogue-free footage only after the user confirms it (`check-subtitles --mode none`, exit 5).
4. Offer the optional OP/ED windows (`materials/bible.json` → `op_ed_windows`) once per series, so
   opening/ending lyrics never become "scenes" or singer-clusters.

## Stop-checkpoint

Report to the user: workspace path, `doctor` readiness per dependency, the chosen subtitle tier, and
whether the two optional passes (narrative outline, AV understanding) will run. Do not proceed into
stage 2 inside this command.
