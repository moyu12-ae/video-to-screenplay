---
description: Run the deterministic pipeline from extraction through scene grouping and the per-scene evidence packs, then author every scene draft with placeholder dialogue.
argument-hint: <workspace path>
---

# /video-to-screenplay:build

Stages 2 → 4. The contracts, exit codes and checkpoints are in
`skills/video-to-screenplay/SKILL.md` §阶段 2-4 — read that section; this file only fixes the order.
Paths are relative to this plugin's root.

## Steps

```bash
# stage 2 — visual track in the background, dialogue track in the foreground
python3 scripts/scene_detect.py --workspace "<ws>" --threshold 0.35 > "<ws>/.cache/visual/shots.json" &
PID_VISUAL=$!
python3 scripts/subtitle_extractor.py --workspace "<ws>" --require-subtitles \
    > "<ws>/.cache/subtitles/extracted.json"
python3 scripts/speaker_diarize.py --workspace "<ws>" prepare \
    > "<ws>/.cache/audio/diarize_workorder.json"
wait $PID_VISUAL
```

Then, in order: `speaker_diarize.py run` (or the MCP fallback; `merge --empty-fallback` when neither is
available), `semantic_scene_grouper.py`, `align_timeline.py`. Optional but recommended:
`narrative_outline.py` before the grouper (McKee sequence layer), and `av_understand.py prepare` + `run`
after it — preview the AV pass cost and get the user's confirmation first, and run it in the background.

```bash
python3 scripts/build_scene_manifest.py --workspace "<ws>"
```

Finally author every scene with `draft_status == "missing"`: read its thumbnails, `av_notes` and
verbatim dialogue, write `.cache/scene_drafts/scene_XX.md`.

## Hard rules while writing

- Dialogue is **only** `[[SUB:n]]` placeholders — one per `sub_index`, exactly once, in time order.
  Never retype a line.
- Character names come only from the bible / manifest / the names appearing in dialogue; otherwise use
  a visible epithet. Never render a provisional `SPEAKER_*` label as a name.
- Witness rule: write only what a camera could capture and a microphone could hear — no interiority,
  no negated action. `av_notes` are evidence, never dialogue text.
- Deliverables go to `<ws>/output/`, intermediates only to `<ws>/.cache/`.

## Stop-checkpoint

Report: `total_scenes`, `dialogue_total` vs the subtitle count, whether acoustic attribution is real or
degraded (`acoustic_clustering_enabled`), AV coverage per scene, and the count of drafts still missing.
Do not splice in this command — assembling is `/video-to-screenplay:splice`.
