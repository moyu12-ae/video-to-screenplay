# Video to Screenplay (`video-to-screenplay`)

English | [简体中文](README_CN.md)

A ZCode plugin that reverse-engineers videos and anime episodes into production-standard Asian 场号制 screenplays. Deterministic Python stages compute everything measurable (cuts, timecodes, dialogue alignment); perception is delegated to general multimodal models — scene understanding (place, time of day, characters, staging) is supplied by the agent itself, while speaker attribution comes from **Qwen3.8-Omni acoustic timbre clustering** (dialed directly from the pipeline via `speaker_diarize.py run`, with the MCP tool `omni_multi_speaker_asr` as a fallback, and degrading gracefully to fully-unattributed when unconfigured). Dialogue text always comes exclusively from subtitles via `[[SUB:n]]` verbatim splicing — acoustics and OCR never touch the dialogue text itself.

## ⚠️ Aliyun API Key Required

**Configure an Aliyun DashScope API key before using this plugin.**

1. Create one in the [Aliyun Bailian console](https://bailian.console.aliyun.com/) (API-KEY management).
2. Export it: `export DASHSCOPE_API_KEY="sk-..."`
3. Optional — point to another OpenAI-compatible endpoint: `export DASHSCOPE_BASE_URL="https://..."`

The key powers both Qwen3.8-Omni capabilities: acoustic speaker diarization and the optional AV-understanding pass. Stage 1 preflight-checks it: when it is missing, the pipeline asks you explicitly — configure the key, or consciously continue with blank speaker columns (`speaker_diarize.py run` exits 8 as a second guard).

### What leaves your machine (and where the key goes)

This plugin uploads source content to a third-party API. `SECURITY.md` is the full declaration; `scripts/config_spec.py` is the machine-readable source of truth and a test fails if either drifts.

| Variable | Default | What it controls |
| :--- | :--- | :--- |
| `DASHSCOPE_API_KEY` | unset | Bearer credential for both Omni passes; read in exactly one function, never persisted, never logged |
| `DASHSCOPE_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | **which host receives the key and the media** (https + public addresses only) |
| `V2S_OMNI_MODEL` | `qwen3.8-omni-flash` | which omnimodal model interprets your footage |
| `V2S_OMNI_ATTEMPTS` | `3` | retry budget per request (transient errors only) |
| `V2S_OMNI_TIMEOUT_SEC` | `1800` | wall-clock ceiling per streaming request |

Uploaded: 16 kHz mono audio parts (diarization) and per-scene video segments of ≤90 s at 480p with a mono audio track (AV understanding). Never uploaded for transcription: the dialogue — subtitles are parsed locally and spliced verbatim, and the AV prompt forbids transcribing speech. `workspace.py doctor` reports the key state together with the host and model it will be sent to.

## How It Works

1. **Clean Workspace** — `materials/` (read-only inputs) → `.cache/` (disposable intermediates) → `output/` (final screenplay only).
2. **Subtitle Gate** — external file → embedded soft stream → OCR tier. When only the OCR tier can serve the source, the extractor prints the Tier 3 instruction and exits 6 (awaiting the OCR pass), which is a route, not a refusal; footage with no dialogue at all is refused only after the user confirms it (`workspace.py check-subtitles --mode none`, exit 5). The pipeline never invents dialogue. Configured OP/ED windows (bible.json → `op_ed_windows`) drop opening/ending lyrics at extraction — a one-line （动画 OP/ED） marker replaces them in the final screenplay; a window scene that still keeps a boundary-straddling dialogue line is authored as a normal scene (a stub cannot carry `[[SUB:n]]`) with that same one-line marker added.
3. **Dual-Track Extraction** — FFmpeg shot cuts + keyframes run concurrently with subtitle ingestion and **acoustic speaker diarization** (ffmpeg extracts 16 kHz mono audio → a direct DashScope client streams the same Qwen3.8-Omni diarization request — code-owned retries/backoff, per-part resume, no client tool-window limits; MCP `omni_multi_speaker_asr` remains as a fallback → deterministic binding by aggregate per-cluster overlap, so a co-speaker covering ≥40% of a line survives as `secondary_speaker`). Multi-part runs snap cut points to silence, overlap adjacent parts by ±3 s, and reconcile cross-part identities only on co-occurrence evidence (the same utterance diarized in both parts — union-find, prefer splitting over wrong merging). A part the model reports as speech-free counts as finished: reruns skip it instead of paying for it again. Attribution is 100% acoustic; subtitle metadata only votes to NAME the clusters. Without an API key (or the MCP fallback), speakers degrade to all-null and the pipeline keeps running — and the degraded `speakers.json` says so (`acoustic_clustering_enabled: false`, `backend: "none"`).
4. **LGSS-Inspired Scene Grouping** — a 1D DP solver folds 200+ physical cuts into macro scenes. Dialogue-crossed boundaries carry a large soft penalty (never a hard ban — the solver cannot deadlock into a single-scene collapse, and any forced cut is reported). With numpy/opencv installed, an HSV keyframe-palette distance (a lightweight "place" proxy in the spirit of LGSS, CVPR 2020) sharpens boundaries; without them it degrades to silence/duration heuristics.
5. **Millisecond Alignment** — every dialogue cue is assigned to exactly one shot (max temporal overlap; a cue that clears no shot's overlap bar is bound to the temporally nearest one and reported, so the aligner can never hand the splicer a subtitle nobody was asked to place), with `ON_SCREEN` / `OFF_SCREEN` / `VOICE_OVER` / `INTERNAL_MONOLOGUE` flags.
6. **Narrative Outline (optional, McKee sequence layer)** — `narrative_outline.py` emits a dialogue-stream work-order; the agent authors the sequence skeleton by value shift (title + value from→to + sub range), and the grouper switches to hierarchical mode: sequence walls snap to the nearest physical cut **within a 15 s window** (a wall with no cut that close is placed at the nearest preceding cut and reported with `within_snap_window: false`), then each sequence is solved independently. Without an outline the flat solve is unchanged.
7. **AV Understanding (optional, recommended)** — `av_understand.py` makes the model actually WATCH each macro scene (segments of ≤90 s, with a documented +10 s sliver-fold exception, cut from the episode and transcoded down a CRF ladder into the inline budget) and return structured evidence notes — actions, camera language, sound events, on-screen text — with per-entry timecodes mapped back to the episode timeline and a coverage check per scene. Dialogue transcription and character naming are explicitly forbidden in this pass (people appear as visible epithets like "the woman in red"); the notes are a **writing-evidence layer**, never dialogue text.
8. **Scene Understanding & Writing (chunked, resumable)** — `build_scene_manifest.py` emits a per-scene **evidence pack** (640px keyframe thumbnails, verbatim dialogue with timecodes, AV understanding notes when available, bible character names, previous-episode exemplars, the owning sequence's value arc) embedding the full writing contract; the multimodal agent reads keyframes + notes per scene and authors real 场号制 screenplay text as `scene_drafts/scene_XX.md` — `△` action paragraphs grounded in what was actually seen and heard, with every line represented by a `[[SUB:n]]` placeholder (dialogue is never retyped).
9. **Verbatim Splice & Assembly** — `splice_screenplay.py` replaces each `[[SUB:n]]` with the verbatim subtitle text — fidelity is guaranteed **by construction**; paraphrase is structurally impossible. Missing/duplicated/misplaced placeholders are fatal errors naming the offending indices. The assembled document carries a metadata header, 场次总表 (with the sequence column), spliced scenes, and an appendix of audio-visual statistics + fidelity report, written to `output/<Title>_影视文学剧本.md`.

## Requirements

- Python 3.10+
- FFmpeg / ffprobe on PATH (`brew install ffmpeg`)
- Recommended: `DASHSCOPE_API_KEY` in the environment — enables direct-API Qwen3.8-Omni acoustic speaker diarization (`speaker_diarize.py run`: no client tool-window limits, exponential-backoff retries, per-part resume). Fallback: the `api` plugin of [Qwen-MM-Plugins](https://github.com/QwenLM/Qwen-MM-Plugins) (MCP). Without either, the speaker columns stay blank and the pipeline still runs
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
#   ↑ exit 6 with a Tier 3 payload on stdout = the source is hard-subsidised; run the OCR pass
#     and write extracted.json yourself. exit 5 (pure-visual refusal) belongs to
#     `workspace.py check-subtitles --mode none`, after the user confirms there is no dialogue.
python3 scripts/speaker_diarize.py --workspace "<ws>" prepare > "<ws>/.cache/audio/diarize_workorder.json"  # exit 6
python3 scripts/speaker_diarize.py --workspace "<ws>" run
#   ↑ path A (recommended): dials DashScope directly (DASHSCOPE_API_KEY in the env;
#     code-owned retries/backoff, per-part resume — rerun `run` to retry only missing parts).
#   ↑ path B (fallback): the agent calls MCP omni_multi_speaker_asr per workorder part,
#     saving each returned JSON block verbatim to .cache/audio/omni_diarized[.partNNN].json
python3 scripts/speaker_diarize.py --workspace "<ws>" merge > "<ws>/.cache/audio/speakers.json"
#   ↑ without any diarization backend: merge --empty-fallback (all-null speakers degradation)
python3 scripts/semantic_scene_grouper.py --workspace "<ws>" > "<ws>/.cache/visual/scenes.json"
python3 scripts/narrative_outline.py --workspace "<ws>"   # optional: agent authors the outline, then re-run to validate
python3 scripts/align_timeline.py --workspace "<ws>" > "<ws>/.cache/alignment/aligned_timeline.json"
python3 scripts/build_scene_manifest.py --workspace "<ws>"
# agent authors .cache/scene_drafts/scene_XX.md per the manifest (dialogue as [[SUB:n]] placeholders)
python3 scripts/splice_screenplay.py --workspace "<ws>" --title "<Title>"
```

## Development

```bash
python3 -m pytest tests/ -q                 # unit + end-to-end (needs ffmpeg)
python3 -m ruff check scripts tests --select F
```

`.github/workflows/ci.yml` runs the suite twice per OS — once with the optional
numpy/opencv pair, once without it — because the grouper's stdlib degradation path is
part of the contract. `tests/test_end_to_end.py` synthesises a short episode with
ffmpeg and drives all five stages, so stage-to-stage contract breaks (a subtitle the
aligner dropped but the splicer demanded, say) cannot ship unnoticed.

See [SKILL.md](skills/video-to-screenplay/SKILL.md) for the full agent workflow, the scene-writing contract, and the failure-mode matrix.

## Acknowledgements

- [Qwen-MM-Plugins](https://github.com/QwenLM/Qwen-MM-Plugins) (Apache-2.0) — the direct DashScope client in `scripts/omni_client.py` is adapted from its request-shape / audio-fitting / retry implementation; the diarization prompt is reproduced verbatim.
- [NarratoAI](https://github.com/linyqh/NarratoAI) (MIT) — engineering patterns for transient-only retries and failure isolation.

## License

MIT — see [LICENSE](LICENSE). Third-party adaptations are attributed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
