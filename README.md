# Video to Screenplay (`video-to-screenplay`)

English | [简体中文](README_CN.md)

A ZCode plugin that reverse-engineers videos and anime episodes into production-standard Asian 场号制 screenplays. Deterministic Python stages compute everything measurable (cuts, timecodes, dialogue alignment); perception is delegated to general multimodal models — scene understanding (place, time of day, characters, staging) is supplied by the agent itself, while speaker attribution comes from **Qwen3.8-Omni acoustic timbre clustering** (MCP `omni_multi_speaker_asr`, degrading gracefully to fully-unattributed when unconfigured). Dialogue text always comes exclusively from subtitles via `[[SUB:n]]` verbatim splicing — acoustics and OCR never touch the dialogue text itself.

## How It Works

1. **Clean Workspace** — `materials/` (read-only inputs) → `.cache/` (disposable intermediates) → `output/` (final screenplay only).
2. **Subtitle Gate** — external file → embedded soft stream → OCR gateway; pure visual footage is hard-refused (exit 5). The pipeline never invents dialogue.
3. **Dual-Track Extraction** — FFmpeg shot cuts + keyframes run concurrently with subtitle ingestion and **acoustic speaker diarization** (ffmpeg extracts 16 kHz mono audio → MCP `omni_multi_speaker_asr` clusters voices by timbre, auto-chunked past 50 minutes → deterministic max-overlap binding to subtitle lines). Attribution is 100% acoustic; subtitle metadata (ASS Actor, `【角色】` prefixes) only votes to NAME the acoustic clusters. Without the MCP tool, speakers degrade to all-null and the pipeline keeps running.
4. **LGSS-Inspired Scene Grouping** — a 1D DP solver folds 200+ physical cuts into macro scenes. Dialogue-crossed boundaries carry a large soft penalty (never a hard ban — the solver cannot deadlock into a single-scene collapse, and any forced cut is reported). With numpy/opencv installed, an HSV keyframe-palette distance (a lightweight "place" proxy in the spirit of LGSS, CVPR 2020) sharpens boundaries; without them it degrades to silence/duration heuristics.
5. **Millisecond Alignment** — every dialogue cue is assigned to exactly one shot (max temporal overlap), with `ON_SCREEN` / `OFF_SCREEN` / `VOICE_OVER` / `INTERNAL_MONOLOGUE` flags.
6. **Narrative Outline (optional, McKee sequence layer)** — `narrative_outline.py` emits a dialogue-stream work-order; the agent authors the sequence skeleton by value shift (title + value from→to + sub range), and the grouper switches to hierarchical mode: sequence walls snap to the nearest physical cut, then each sequence is solved independently. Without an outline the flat solve is unchanged.
7. **Scene Understanding & Writing (chunked, resumable)** — `build_scene_manifest.py` emits a per-scene **evidence pack** (640px keyframe thumbnails, verbatim dialogue with timecodes, bible character names, previous-episode exemplars, the owning sequence's value arc) embedding the full writing contract; the multimodal agent reads keyframes per scene and authors real 场号制 screenplay text as `scene_drafts/scene_XX.md` — `△` action paragraphs woven between dialogue, with every line represented by a `[[SUB:n]]` placeholder (dialogue is never retyped).
8. **Verbatim Splice & Assembly** — `splice_screenplay.py` replaces each `[[SUB:n]]` with the verbatim subtitle text — fidelity is guaranteed **by construction**; paraphrase is structurally impossible. Missing/duplicated/misplaced placeholders are fatal errors naming the offending indices. The assembled document carries a metadata header, 场次总表 (with the sequence column), spliced scenes, and an appendix of audio-visual statistics + fidelity report, written to `output/<Title>_影视文学剧本.md`.

## Requirements

- Python 3.10+
- FFmpeg / ffprobe on PATH (`brew install ffmpeg`)
- Recommended: the `api` plugin of [Qwen-MM-Plugins](https://github.com/QwenLM/Qwen-MM-Plugins) (with `DASHSCOPE_API_KEY`) — enables Qwen3.8-Omni acoustic speaker diarization; without it the speaker columns stay blank and the pipeline still runs
- Optional: `pip install -r requirements.txt` (numpy + opencv-python-headless) — enables the visual place affinity in the scene grouper. Everything else is pure standard library.

## Usage

Run the skill inside ZCode:

```
/video-to-screenplay
```

Or drive the stages directly:

```bash
python3 scripts/workspace.py init  --workspace "<ws>"
python3 scripts/workspace.py probe --workspace "<ws>"
python3 scripts/scene_detect.py --workspace "<ws>" --threshold 0.35 > "<ws>/.cache/visual/shots.json"
python3 scripts/subtitle_extractor.py --workspace "<ws>" --require-subtitles > "<ws>/.cache/subtitles/extracted.json"
python3 scripts/speaker_diarize.py --workspace "<ws>" prepare > "<ws>/.cache/audio/diarize_workorder.json"  # exit 6
#   ↑ the agent then calls MCP omni_multi_speaker_asr per workorder part,
#     saving each returned JSON block verbatim to .cache/audio/omni_diarized[.partNNN].json
python3 scripts/speaker_diarize.py --workspace "<ws>" merge > "<ws>/.cache/audio/speakers.json"
#   ↑ without the MCP tool: merge --empty-fallback (all-null speakers degradation)
python3 scripts/semantic_scene_grouper.py --workspace "<ws>" > "<ws>/.cache/visual/scenes.json"
python3 scripts/narrative_outline.py --workspace "<ws>"   # optional: agent authors the outline, then re-run to validate
python3 scripts/align_timeline.py --workspace "<ws>" > "<ws>/.cache/alignment/aligned_timeline.json"
python3 scripts/build_scene_manifest.py --workspace "<ws>"
# agent authors .cache/scene_drafts/scene_XX.md per the manifest (dialogue as [[SUB:n]] placeholders)
python3 scripts/splice_screenplay.py --workspace "<ws>" --title "<Title>"
```

## Development

```bash
python3 -m pytest tests/ -q
```

See [SKILL.md](skills/video-to-screenplay/SKILL.md) for the full agent workflow, the scene-writing contract, and the failure-mode matrix.

## License

MIT — see [LICENSE](LICENSE).
