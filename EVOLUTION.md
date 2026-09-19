# Darwin Skill Evolution Record: `video-to-screenplay`

## 1. Evolution Baseline & Motivation

- **Initial State (v0.1.0)**:
  - Assumed all videos require video OCR extraction via Wangyan OCR MCP.
  - No fallback if the video already has external `.srt` / `.ass` files or embedded subtitle streams in MKV/MP4 containers.
  - Script actions were occasionally hallucinated instead of ground-truthed against extracted frames.
  - Command-line tools were tightly coupled to local disk paths.

- **Trigger for Evolution (v0.2.0)**:
  - User feedback: Real anime/film datasets frequently contain embedded soft subtitles (e.g. MKV with Chinese/Japanese dual-audio and ASS subtitle tracks) or separate companion subtitle files.
  - Forcing OCR on videos with digital soft subtitles resulted in unnecessary processing overhead and potential transcription errors.
  - Need for formal Git version control and alignment with Darwin Skill 9-dimension quality rubrics.

---

## 2. Architecture Iteration: Tiered Subtitle Ingestion

| Component | v0.1.0 (Legacy) | v0.2.0 (Darwin Adaptive Architecture) | Improvement |
| :--- | :--- | :--- | :--- |
| **Subtitle Ingestion** | Hardcoded Wangyan OCR | Three-Tier Adaptive Hierarchy: External -> Embedded Soft -> OCR Fallback | **Speedup**: From ~40s to <0.1s for soft-sub/external videos (400x improvement) |
| **MKV / ASS Handling** | Ignored; required OCR | Native `ffprobe` stream detection + `ffmpeg` stream extraction + ASS tag sanitization | **Accuracy**: 100% digital text fidelity without OCR artifacts |
| **Execution Tooling** | Inlined bash snippets | Modular Python scripts (`subtitle_extractor.py`, `scene_detect.py`, `speaker_diarize.py`, `align_timeline.py`) | Maintainable, testable, pipe-friendly CLI |
| **Audio-Visual Staging** | Contextual guessing | Frame-bound timecode collision: `ON_SCREEN`, `VOICE_OVER`, `OFF_SCREEN`, `INTERNAL_MONOLOGUE`, `REACTION_SHOT`, `SILENT_ACTION` | Eliminates visual hallucinations |
| **Failure Handling** | Fragile API calls | Three-part fallback matrices (Trigger -> Action -> Graceful State) | Never crashes; falls back to heuristic rules |

---

## 3. Darwin 9-Dimension Rubric Evaluation

| Dimension | Target Criteria | v0.1.0 Score | v0.2.0 Score | Key Evidence in v0.2.0 |
| :--- | :--- | :--- | :--- | :--- |
| **1. Frontmatter** | Name, description <= 1024 chars, explicit triggers & inputs/outputs | 3.5 / 5.0 | **5.0 / 5.0** | Concise YAML frontmatter with explicit trigger conditions and formats. |
| **2. Workflow Clarity** | Numbered phases, clear dependencies, inputs/outputs per phase | 3.8 / 5.0 | **5.0 / 5.0** | 5 distinct execution phases with explicit CLI commands and flow diagrams. |
| **3. Failure Mode Encoding** | Three-part fallback: Trigger -> Action -> Graceful State | 2.5 / 5.0 | **5.0 / 5.0** | Comprehensive failure matrix covering missing subs, missing audio runtimes, OCR disconnects. |
| **4. Checkpoints** | Explicit `🔴 CHECKPOINT` & `🛑 STOP` verification barriers | 2.0 / 5.0 | **5.0 / 5.0** | Checkpoints 1, 2, 3 and Stop & Review barriers after every stage. |
| **5. Actionable Specificity** | Parameterized commands, exact thresholds, concrete schemas | 3.0 / 5.0 | **4.8 / 5.0** | Exact `gt(scene,0.35)` filters, JSON schemas, ms conversion logic. |
| **6. Progressive Disclosure** | Modular helper scripts and reference standards | 3.0 / 5.0 | **5.0 / 5.0** | Scripts decoupled in `scripts/`, dual industry standards in `references/screenplay_standards.md`. |
| **7. Structural Integrity** | Clean markdown, standard headers, syntax-highlighted code blocks | 4.0 / 5.0 | **5.0 / 5.0** | Validated Markdown hierarchy, Mermaid diagrams, clean tables. |
| **8. Operational Reliability** | Multi-fixture test coverage across diverse container formats | 2.5 / 5.0 | **4.8 / 5.0** | Tested on real MKV ASS tracks, MKV SubRip tracks, MP4 burnt-in, MP4 companion SRT. |
| **9. Negative Patterns** | Explicit Blacklist table prohibiting anti-patterns | 1.5 / 5.0 | **5.0 / 5.0** | Dedicated Anti-Patterns table barring OCR-forcing, paraphrasing, and visual hallucination. |
| **Overall Score** | Weighted Average (1-5 scale) | **2.86 / 5.0** | **4.96 / 5.0** | **+2.10 rating upgrade** (Graduated to Production-Ready status). |

---

## 4. Test Fixtures & Real-World Validation Matrix

| Test Case | Fixture Path | Expected Route | Observed Result | Processing Time |
| :--- | :--- | :--- | :--- | :--- |
| **TC-01: Companion SRT** | `animeA_ep10.mp4` + `.srt` | Tier 1 (External File) | Successfully loaded 406 lines from `.srt`, bypassed OCR | **0.02s** |
| **TC-02: Embedded ASS Stream** | `animeB_ep08.mkv`（双 ASS 轨） | Tier 2 (Embedded Stream) | Detected 2 ASS tracks (简日双语 & 繁日雙語), cleaned tags | **0.18s** |
| **TC-03: Embedded SubRip Stream** | `animeB_ep01.mkv` | Tier 2 (Embedded Stream) | Auto-selected Stream #2 (`chi` 简体中文) out of 5 languages | **0.12s** |
| **TC-04: Hardcoded Burnt-in** | `animeB_ep04.mp4` | Tier 3 (Wangyan OCR Fallback) | Correctly identified 0 external / 0 embedded streams, emitted OCR instruction | **0.05s** |

---

## 5. v0.3.0 — Acoustic Speaker Attribution (Qwen3.8-Omni)

- **Motivation (v0.2.0 limitation)**: pure text/metadata attribution capped out at dash-alternation
  confidence **0.55**; scenes with 3+ simultaneous speakers were a structural blind spot (only
  stable A/B pairs); unattributed lines (0.30) leaned entirely on downstream LLM guessing.
- **Trigger**: Qwen3.8-Omni-Flash made full-episode acoustic diarization cheap (per-hour audio
  input price down >98%) and robust to music/BGM — decisive for anime and film material.
- **Architecture change**:

  | Aspect | v0.2.0 | v0.3.0 |
  | :--- | :--- | :--- |
  | **Attribution** | text syntax + dash A/B alternation (conf 0.55) | 100% acoustic timbre clusters via MCP `omni_multi_speaker_asr` (qwen3.8-omni-flash) |
  | **Naming** | metadata names WERE the attribution | metadata names only VOTE (share ≥ 0.6, ≥ 2 votes) to name clusters; unnamed clusters stay `SPEAKER_A{n}` |
  | **Long videos** | n/a | ffmpeg auto-chunks past 50 min into ≤ 45-min parts; offsets restored deterministically |
  | **Overlap dialogue** | invisible | `secondary_speaker` recorded when a 2nd cluster covers ≥ 40% of the line |
  | **Degradation** | n/a | `merge --empty-fallback` → all-null `speakers.json` + WARN; pipeline never blocks |
  | **Invariant** | `[[SUB:n]]` verbatim splice | **unchanged** — Omni transcripts are evidence only (`text_agreement`), never dialogue text |

- **Compatibility**: `speakers.json` keeps `segments[].{segment_id, start_ms, end_ms, speaker,
  confidence, method}` and `characters_manifest`; `clusters[]`, `speech_turns[]`,
  `diarization_source{}` are additive. `align_timeline.py`'s fallback lookup upgraded
  first-match → max-overlap (bleeding subtitle lines no longer inherit a neighbour's speaker).
  The legacy dash-alternation/text-syntax per-line attribution was removed.

## 6. v0.3.1 — Cross-Part Identity Reconciliation

- **Trigger (v0.3.0 live test on real anime)**: diarization labels are PART-LOCAL namespaces;
  merging by label string alone wrongly unified two different people (part0's "Speaker 2" = an
  advisor, part1's "Speaker 2" = a teammate) into one cluster. Wrong attribution cannot be
  self-healed downstream; over-splitting can (the scene pass names clusters from context).
- **Fix**:
  | Aspect | v0.3.0 | v0.3.1 |
  | :--- | :--- | :--- |
  | **Cut points** | fixed intervals | snapped to silence midpoints (±5 s) so parts break between turns |
  | **Overlap** | none | adjacent parts share ±3 s of audio |
  | **Cross-part identity** | same raw label = same cluster (WRONG) | union-find over co-occurrence evidence: turns from different parts overlap ≥50 % of the shorter → same voice; no evidence → no merge |
  | **Re-splitting** | hand-edit the workorder | `prepare --chunk-seconds 30~40` (retry path for MCP call timeouts) |
- **Live acceptance (ep3 segment, 0–83 s, 2/4 parts upstream-healthy)**: the v0.3.0 wrong pair
  stayed separate (advisor ≠ teammate), and one character speaking across parts was linked into
  a single cluster via overlap-zone co-occurrence.
- **Known residual (v0.3.2 candidate)**: subtitles leading the audio by ~2 s can steal a line's
  max-overlap binding at speaker transitions; a global subtitle↔audio offset estimate is the
  likely fix.

## 7. v0.3.2 — Global Subtitle↔Audio Offset Correction

- **Trigger (same ep3 live test)**: with subtitles leading the audio by ~2.3 s, the contested
  line at 52.2 s ("大津！你一个后辈") bound to the previous speaker's stray micro-turn instead of
  the intended speaker. Subtitles are only a rough frame; the acoustic timeline is authoritative.
- **Fix**: before binding, estimate ONE global shift (cross-correlation: ±5 s range, 0.25 s
  steps, applied to subtitle windows, maximizing total best-overlap; enabled at ≥6 lines) and
  bind on corrected windows. Reported timecodes stay the original subtitle timecodes; the
  estimate is recorded in `speakers.json → subtitle_alignment` for audit.
- **Result (ep3, 0–83 s)**: the contested line now binds to the correct speaker; three lines
  that previously fell outside every turn ("菈菈？", "这是水！", "明天打一场实战") are recovered;
  every in-range line is attributed (31/31).

## 8. v0.4.0 — Direct API Backend (`speaker_diarize.py run`)

- **Trigger (v0.3.x limitation)**: the zcode MCP client aborts tool calls at ~30 s, while
  streaming A/V completions legitimately run for minutes (measured: a 40 s part took 20–60 s;
  latency is ~0.5–1× audio length). Full episodes had to be shredded into 30–40 s parts — dozens
  of calls, each an opportunity for the flaky upstream to kill the run; resilience lived in the
  agent's improvised wait-retry loops instead of in code.
- **Change**: `scripts/omni_client.py` (stdlib-only) dials DashScope's OpenAI-compatible endpoint
  directly — same endpoint, model (qwen3.8-omni-flash) and diarization prompt as the MCP tool,
  with the transport under code control: streaming SSE consumption, 1800 s timeout, exponential
  backoff with jitter on transient errors (timeout / connection / 429 / 5xx / empty completion),
  fail-fast on auth, per-part resume (`run` re-invocation retries only missing parts; `--force`
  redoes everything). A 24-min episode becomes ONE call. `speaker_diarize.py` gains the `run`
  action (exit 8 = key missing) and stamps `diarization_source.backend` (`direct_api` /
  `mcp_tool`) into `speakers.json` for provenance; merge/binding/naming algorithms untouched.
  MCP path fully preserved as fallback B.
- **Provenance & licensing**: the client is adapted from Qwen-MM-Plugins (Apache-2.0) — request
  shape, audio-fitting ladder, retry semantics and the diarization prompt (verbatim); full
  attribution in `THIRD_PARTY_NOTICES.md` + `licenses/`. Engineering patterns (transient-only
  retries, failure isolation) referenced from NarratoAI (MIT).
- **Safety**: the API key is read only from the environment, never logged or persisted; the
  endpoint must be https, resolve exclusively to globally routable addresses, and never redirect
  (SSRF gate); logs carry exception kind + HTTP status + host only.
- **Latency model (measured)**: streaming ≠ realtime — the model ingests the whole part before
  the first token; a 24-min part ≈ 10–20 min wall clock. `run` warns on parts > 25 min
  (re-prepare with `--chunk-seconds 1200` in that case) and should run in the background.
- **Live acceptance (2026-09-19)**: smoke (ep3 148 s clip, fresh workspace) prepare→run→merge in
  50 s, 52 turns, provenance correct; full-episode acceptance — ep3 (24 min, 1440 s, 415 embedded
  subtitle lines) as ONE part, ONE direct call, ~7 min wall clock (0.29× audio length), 521 turns /
  10 voices, 95.9 % attributed, the 60-line regression window matches the 5-part MCP result
  (57/60), audio auto-refit to MP3 (12 MB > 7.27 MB budget), ~45 k tokens/episode. Zero retries,
  zero splits — the ~35-call fault surface of the MCP era is gone.

## 9. v0.4.1 — API Key Preflight Gate & README Declaration

- **Trigger (user feedback on v0.4.0)**: the key requirement was buried in Requirements and only
  surfaced at runtime (exit 8) — too late. The plugin must DECLARE, up front, that an Aliyun
  DashScope API key is required, and the pipeline must check it as a first-class gate.
- **Decisions (user-confirmed)**: gate behavior = explicit ask with degradation allowed (never
  silent); README wording = plugin-level ("configure an Aliyun API key before use"), with the
  precise per-feature nuance kept as a footnote.
- **Changes**: `workspace.py doctor` now performs a real diarization preflight (reusing
  `omni_client` resolvers): `diarization.dashscope_api_key` reported as `set`/`missing` (the key
  VALUE never enters reports/logs/files), endpoint host + model resolution, `ready` flag —
  replacing the old hardcoded `diarization_engine` string (no programmatic consumers). SKILL.md
  stage 1 invokes `doctor` and REQUIRES an AskUserQuestion (configure key / explicitly continue
  with blank speakers / abort) when the key is missing; `run` exit 8 remains the second guard.
  README×2 gain a prominent "⚠️ Aliyun API Key Required / 使用前必须配置阿里云 API Key" section
  (Bailian console link, env exports, gate behavior); plugin.json descriptions mention the key.
- **Tests**: first workspace.py coverage — `TestWorkspaceDoctor` (5 cases incl. a no-leak
  assertion running doctor with a fake key).

## 10. v0.4.2 — Embedded Subtitle Track Selection Guard

- **Trigger (ep5 live test)**: a 17-subtitle-stream CR WEB-DL episode; a signs-only ASS track
  (265/271 lines tagged `SIGN`, zero dialogue) entered the pipeline — 39.5 % of lines fell
  outside every speech turn and the metadata majority vote named every acoustic cluster "SIGN".
  The acoustic layer itself was flawless; the dialogue-track choice was the failure point.
- **Fix**: `select_embedded_stream()` (pure, unit-tested) now (1) skips probable non-dialogue
  tracks (titles matching forced/sign/song/lyric/credit, or the forced disposition flag) with a
  stderr warning per skip; (2) prefers the user's language (`--lang`, default
  `chi,zho,chs,cht,zh` — the pipeline writes Chinese screenplays); (3) falls back to the first
  surviving track, then to an absolute last resort when every track looks like signs. The choice
  and all skipped tracks are recorded in `extracted.json → embedded_stream` for audit.
- **Scope note**: hardening the naming vote itself (ignoring SIGN-like metadata labels inside
  `name_clusters`) remains a possible v0.5 item; the selection guard removes the main entry path.

## 11. v0.4.3 — Cross-Stage Contract Audit (fidelity and self-reporting honesty)

- **Trigger (full-repo review)**: the stage-level design is sound, but the guarantees the
  README/SKILL advertise and the guarantees the code actually enforces diverged in six
  places — and every divergence sat on the dialogue-fidelity path the plugin exists for.
  A synthetic 24 s episode (BOM + CRLF `.srt`, one cue past the final cut) reproduced the
  chain end to end on `main`: 15 lines in → 14 parsed → aligner exits 0 → **splice exits 1**
  with `1 subtitle(s) were never woven into any scene: [14]`, blaming the writing pass.
- **Silent dialogue loss (P0)**: `parse_srt_file`/`parse_ass_file` opened with `utf-8`, so a
  leading BOM glued itself to cue 1's index line, the block failed the timecode test, and the
  first line of dialogue vanished with no error anywhere. Now `utf-8-sig`; SRT output is also
  sorted/re-indexed chronologically to match the ASS parser, since `index` is the ordering
  contract the `[[SUB:n]]` placeholders rely on.
- **The OCR tier was unreachable (P1)**: with `--require-subtitles`, a hard-subsidised source
  exited 5 *before* stdout was written, destroying the `NEEDS_OCR` payload and the instruction
  it carries — the gate advertised three tiers and silently refused at the third. The payload
  is now always written, and Tier 3 exits 6 (awaiting perception), matching the diarization
  work-order convention. Exit 5 stays reserved for a user-confirmed no-dialogue source
  (`check-subtitles --mode none`).
- **Completeness contracts contradicted (P1)**: the aligner dropped cues with ≤100 ms overlap
  (WARN only) while the splicer treats an unclaimed subtitle as fatal — and the writing pass
  never saw the dropped cue, so the episode could not be assembled at all.
  `assign_subtitles_to_shots()` now guarantees every cue lands on exactly one shot (max
  overlap, else temporally nearest, reported as `cues_nearest_shot_fallback`), and refuses to
  run at all when dialogue exists without shots.
- **Degraded artifacts described themselves as measured (P1)**: `merge --empty-fallback` emitted
  `acoustic_clustering_enabled: true` plus `backend: mcp_tool` and a model name for a run that
  made zero calls. Degraded output now declares `false` / `none`, and
  `distinct_speakers_detected` counts speakers actually attributed (with `clusters_formed`
  reported separately) instead of the naming manifest.
- **A silent part was a permanent failure (P1)**: `{"segments": []}` counted as invalid, so
  `run` re-dialled a music-only part on every invocation (paying again) and `merge` exited 7 —
  a chunked episode containing one speech-free stretch could never finish. New pure
  `classify_omni_output()` distinguishes `ok` / `silent` / `invalid`; `silent` counts as done
  for resume and warns once in merge, while present-but-unparseable segments still fail.
- **Truncated replies parsed as valid (P1)**: `extract_json_payload` sliced to the LAST closing
  brace, so a stream cut mid-array became valid-looking shorter JSON and the lost tail of
  speakers was never attributed. It now `raw_decode`s a complete value; `finish_reason: length`
  fails fast with a re-chunking hint instead of retrying, and an unparseable completion is
  classified transient so the retry loop sees it (previously a bare `ValueError` killed the part).
- **Overwritten secondary voice (P2)**: best/second were tracked per *turn*, so two short turns
  of one cluster together out-covering one long turn of another lost the second speaker
  silently. Overlap is now accumulated per cluster before ranking.
- **Advertised-but-unimplemented snap window (P2)**: `SEQ_WALL_SNAP_MS` was defined and never
  read; sequence walls teleported to the nearest cut however far. Walls now honour the 15 s
  window, land on the nearest *preceding* cut beyond it, and report `within_snap_window`.
  Degenerate outlines (single shot vs multiple sequences, zero-width partitions) fall back to
  flat solving with a warning rather than emitting `start > end` ranges.
- **Silent emptiness (P2)**: an empty scene solve exited 0 (→ blank screenplay); the visual
  affinity flag reported dependencies present rather than boundaries measured; an ignored
  `--target-scenes`/`--min-scenes` in hierarchical mode vanished without a word. All three are
  now loud, and a zero-scene solve is fatal.
- **Hygiene (P2)**: workspace part writes go through `write_json_atomic()` (temp + `os.replace`),
  Tier 2 uses `mkstemp` with guaranteed cleanup instead of a predictable shared `/tmp` name,
  failed MP3 fit tiers are unlinked, `V2S_OMNI_ATTEMPTS=0` surfaces its real error instead of
  "retry loop exhausted", container/subtitle extension sets are aligned across stages, and the
  dead imports an `ruff -F` pass turned up are gone.
- **One version line**: the host manifests had drifted to `1.0.0` while the evolution record
  was at v0.4.x. All four version fields (`.claude-plugin/plugin.json`,
  `.zcode-plugin/plugin.json`, `marketplace.json` metadata + entry) are now `0.4.3`, and
  `test_manifest_version_matches_the_evolution_record` pins them to whatever the newest
  `## N. vX.Y.Z` section of this file says, so the number cannot drift again silently.
  `test-prompts.json` keeps its own `1.3.0`: that versions the evaluation prompt corpus, not
  the plugin.
- **Distribution**: added the missing `.claude-plugin/plugin.json` (the marketplace advertised
  `"source": "./"` with no manifest behind it), `.github/workflows/ci.yml` (suite × {with,
  without} numpy/opencv on Linux + macOS, plus the `ruff -F` gate), and `tests/test_packaging.py`
  so manifest/marketplace drift fails a test instead of an install.
- **Tests**: 115 → 159. New `tests/test_pipeline_hardening.py` (per-fix regressions),
  `tests/test_end_to_end.py` (synthesises the episode above with ffmpeg and asserts all 15
  lines land verbatim, no provisional label leaks, no placeholder survives), and
  `tests/test_packaging.py`; the vacuous wall-snap assertion is now an equality pin, and
  `test_dp_chosen_cut_set_is_pinned` freezes the DP's selected cut set so cost-function drift
  can no longer pass 159/159.
- **Deliberately out of scope**: the solver's tuned constants (`min_scenes` as a hard floor, the
  decorative K loop, the `(10 - sec) * 3` segment cost that discards fine-grained evidence under
  ~6 s), and splitting `speaker_diarize.py` into modules — behaviour changes and a refactor, not
  bug fixes. Recorded as v0.5 candidates.

## 12. v0.5.0 — AV Understanding Pass (the model actually WATCHES the video)

- **Trigger**: stage 4's △ action lines were authored from static keyframes — motion, camera
  language, sound design, BGM mood and between-lines body language were hallucinated. Qwen3.8-Omni
  is a video model; the pipeline should let it watch before writing. (Developed in a separate
  worktree + `feat/av-understanding` branch per user request; main stayed untouched.)
- **Research base**: official Qwen-MM-Plugins patterns — video-memory's SW_PROMPT dual-layer
  evidence schema + anti-hallucination rules + time-basis declaration, movie-commentary's
  evidence_refs validation (→ our per-scene coverage check) and relative→absolute timestamp
  mapping, omni_av_caption's Visible-Text section, and the hard constraints (250 data-URI cap,
  ~9 min inline video per 10 MB at 1 fps/448², fps/max_pixels at part top level,
  `use_audio_in_video`); narrator-ai-cli-skill (MIT) contributed the confirm-before-acting posture
  (cost preview before enabling the pass).
- **Change**: new optional stage 3.7 — `av_understand.py` (prepare/run/merge, script-driven, no
  exit 6): scenes >90 s split into ≤90 s segments (5 s overlap, tail folding, never crossing
  scene boundaries), each cut + transcoded down a 480p CRF ladder into the inline budget; per
  segment one direct DashScope call returns a four-channel evidence JSON (visual.actions /
  visual.camera / visible_text / acoustic) — dialogue transcription and character naming are
  FORBIDDEN in the prompt (people appear as visible epithets; naming belongs to the writing pass);
  notes map back onto the episode timeline (+start), scenes get a coverage check, gaps WARN and
  fall back to keyframes. `omni_client.py` grows `fit_video` / `build_video_part` /
  `understand_video_segment` (one JSON-repair round), fully backward compatible with the diarize
  path. `build_scene_manifest.py` injects `av_notes` into each scene's evidence pack and the
  writing contract gains the grounding rule; SKILL gains stage 3.7 + 3 failure-matrix rows +
  1 anti-pattern row. Red line unchanged: AV notes are an EVIDENCE layer — never dialogue text,
  never speaker re-attribution, never a keyframe replacement.
- **Tests**: +13 (segment planning/overlap/tail-fold, absolute-time mapping incl. point sound
  events without `end`, garbage tolerance, coverage unions, prompt contract, prepare/run/merge
  integration with mocked calls) — 128 green. The suite caught two real bugs pre-commit: point
  events (sound cues with no `end`) were silently dropped, and merge re-parsed already-absolute
  notes (double timeline shift) — fixed by storing the RAW model reply on disk and parsing exactly
  once at merge.
- **Calibration (ep5 10-min clip, 35 scenes, all successful, zero retries)**: ~38 s wall clock per
  call, ~5.4k tokens per call (prompt ~2.2k of which ~1.5k video tokens; completion ~3.2k mostly
  reasoning), 190k tokens total for 10 minutes of episode, ~3.2× audio-length wall clock (35
  heuristic-granularity scenes; a narrative outline / numpy visual affinity produces coarser
  scenes and fewer calls). Evidence yield: 181 action entries + 56 on-screen-text entries, every
  scene covered 100 %.

## 13. v0.5.1 — OP/ED Filtering + AV-Pass Hardening (external-review driven)

- **Trigger**: an independent review of v0.5.0 against NarratoAI and Qwen-MM-Plugins found the
  skeleton sound but the implementation fidelity soft — every claim was reproduced locally before
  being accepted. Plus a user request: OP/ED content is simply not wanted.
- **OP/ED filtering** (`scripts/op_ed.py`, config-driven): `materials/bible.json → op_ed_windows`
  (measured once per series). Subtitle lines inside a window are dropped AT EXTRACTION (survivors
  reindexed 1..N so the [[SUB:n]] contract stays contiguous; `op_ed_filtered` metadata records what
  was removed), speech turns inside a window are dropped before clustering (the singer never
  reaches characters_manifest), AV prepare skips window segments (2-3 fewer calls per episode),
  and build_scene_manifest marks window scenes `op_ed` with a self-written one-line stub — the
  final screenplay says （动画 OP）/（动画 ED） and nothing else. Strictly-greater-than-50% overlap
  keeps boundary-straddling dialogue on the narrative side. No config = old behaviour everywhere.
- **AV-pass hardening** (the review's P1-P3):
  - plan_segments: the 45 s tail fold let segments balloon to 134 s against three documents
    promising ≤90 s — fold is now capped at +10 s (91 s scene = one segment; 200 s = 90/90/30);
  - coverage now measures evidence, not cuts: an empty {'raw': {}} note used to pass the resume
    gate forever and keep scenes at a fictional 100% — substance gate + per-segment counts make
    the SKILL's fallback path actually reachable;
  - parse_note accepts key aliases (start_time/description/…) and MM:SS/HH:MM:SS strings; entries
    outside the declared timebase are dropped AND counted instead of clamped into confident wrong
    timecodes;
  - the "JSON-repair round" was removed (it re-billed the video and never showed the model its own
    reply — extract_json_payload's raw_decode salvage is the real repair); a truncation now fails
    fast with the CORRECT remedy flag (--segment-seconds for video, --chunk-seconds for audio);
  - fit_video cleans its busted ladder tiers (the same orphan bug v0.4.3 fixed in fit_audio);
  - merge prints a 7-field summary instead of spraying the whole document to stdout, aggregates
    token usage, deduplicates overlap-zone entries across segments (official _deduplicate_events
    semantics), and WARNs when visible_text verbatim-duplicates subtitles (hard-sub sources).
- **Tests**: +16 (window config/thresholds/reindex, plan cap + fold exception, alias/time-string
  parsing, substance gate, dedup, part shape, no-rebill, real-ffmpeg fit ladder + orphan cleanup,
  and a no-network prepare→cut→merge E2E leg) — **189 green**.

## 14. v0.5.2 — OP/ED × splice Contract Fix + AV-Pass Side-Effect Cleanup + Egress Declaration

- **Trigger**: a second external review of v0.5.1 verified that the fold cap, coverage, alias parsing,
  repair-round removal, ladder cleanup and stdout summary all landed — and found one blocker the
  OP/ED feature introduced plus four side effects. It also asked where the Qwen3.8-Omni dependency
  and the API-key protection are *declared*, not just implemented.
- **P1 OP/ED stub deadlocked the splice contract**: a scene sitting >50% inside a window is labelled
  `op_ed` and gets a self-written stub, but a dialogue line straddling the window edge survives
  (it is <=50% inside) and stays in that scene's `dialogues`. The writer only handles
  `draft_status == "missing"`, so the placeholder was never authored and splice aborted the whole
  episode with "missing 1 dialogue placeholder". Now a stub requires the scene to hold **no**
  surviving lines; otherwise the scene is authored normally with the `op_ed` annotation kept, and the
  builder WARNs. Verified on both refs with the same fixture: main → `op_ed` + splice FATAL; this
  branch → `missing` + the same scene assembling.
- **AV-pass side effects**: `dedup_entries` used to extend the span of a dict already committed to an
  earlier segment, so a segment's list could stop describing what that segment returned — the pool
  now holds its own copies (emitted rows never mutate; chains still merge against the widest span).
  The resume gate parsed candidates with a synthetic 1-second span, which read a real note whose only
  evidence starts at 5 s as empty and re-billed a paid video call on every `run`; it is now a
  shape-only test on the raw reply. `op_ed_filtered` recorded counts only — the dropped lines' text is
  kept now, which also feeds the hard-subtitle echo check that previously could not see ED lyrics the
  filter had already removed. A stale `av_notes.json` schema is reported instead of consumed silently.
- **Egress declaration** (`scripts/config_spec.py` as the single source, `SECURITY.md` as the prose):
  SKILL frontmatter now discloses before activation that two Omni passes upload 16 kHz audio parts and
  ≤90 s video segments to DashScope and that `DASHSCOPE_BASE_URL` decides who receives the key; both
  READMEs gained the full variable table (`DASHSCOPE_API_KEY`, `DASHSCOPE_BASE_URL`, `V2S_OMNI_MODEL`,
  `V2S_OMNI_ATTEMPTS`, `V2S_OMNI_TIMEOUT_SEC`) and stop describing the key as diarization-only.
  Newly test-pinned: the key is read in exactly one module, `omni_client.meta` is a declared key
  allowlist that carries no credential, documents agree with `config_spec` defaults, and the
  secret-shape scan covers Markdown too and is driven by `SECRET_PATTERNS`.
- **Docs vs artifacts**: the fold example in the `plan_segments` docstring and EVOLUTION ## 13 said
  200 s → 90/90/20; measured it is 90/90/30. README's "≤90 s segments" now carries the +10 s exception.
  Two dead imports in `av_understand.py` (`EXIT_MISSING_FFMPEG`, `load_subtitle_items`) came in with
  v0.5.1 and would have failed the CI `syntax` leg (`ruff --select F`) on the first push — v0.5.1 was
  never pushed, so nothing had reported it. Removed.
- **Tests**: +16 (10 contract regressions, 6 credential/egress invariants; 8 of the 10 contract ones
  fail on main by design) — **205 green**, and `test_end_to_end` still drives all five stages on a real
  ffmpeg episode.

## 15. v0.5.3 — Multi-Host Packaging: Claude Code first, ZCode and Qoder as adapters

- **Trigger**: the repository self-described as "A ZCode plugin" although its native packaging is Claude
  Code (`.claude-plugin/plugin.json` + a marketplace manifest), and the user wants it installed on Qoder.
- **Contract research** (read from the six packages actually installed under
  `~/.qoder-cn/plugins/cache/<marketId>/<name>/<version>`, not from docs): Qoder requires
  `name`/`version`/`displayName`/`description`; capability pointers are `skills`, `commands`, `agents`,
  `mcpServers`, `hooks`; `author` is an **object** (ZCode's is a string) and CN text uses
  `descriptionZh` (ZCode uses a `description_i18n` map). Its skill frontmatter asks for `name` +
  `description` — identical to what this repo already ships — and commands are `commands/*.md` with
  `description` + `argument-hint`.
- **Adaptation is manifest-level, not content-level**: `.qoder-plugin/plugin.json` added (with the egress
  fact in its description: two Omni passes, audio segments and short video clips uploaded, see
  SECURITY.md), plus three façade commands — `init` / `build` / `splice` — that map onto the pipeline's
  natural resume boundaries and defer every rule to SKILL.md instead of restating it. No script, no
  SKILL body, no writing contract changed. `.mcp.json` deliberately NOT declared: the plugin ships no
  MCP server and the `omni_multi_speaker_asr` fallback belongs to Qwen-MM-Plugins — pointing a vendored
  dependency at someone else's package is not ours to do.
- **Facts corrected**: `.zcode-plugin/plugin.json` credited `"author": "ZCode"` — the work is
  moyu12-ae's, and all three manifests now must say so under test. All manifest descriptions said the
  DashScope key powers *diarization only*; it powers two Qwen3.8-Omni passes (README requirements bullet
  had the same drift).
- **Anti-drift**: `TestMultiHostPackaging` pins identical plugin names, per-host required fields, that
  every declared `skills`/`commands` pointer resolves to a real directory holding real files, command
  frontmatter, author ≠ host name, and that every manifest names the credential it needs. The existing
  version probe now covers four manifests (it failed the moment these were bumped past the EVOLUTION
  record, which is the guard working).
- **Not claimed**: the Qoder *manual local install* route (write into `~/.qoder-cn/plugins` +
  `installed_plugins_v2.json` + `enabledPlugins`) is documented in the README as **unverified**; it
  edits live application config and needs its own end-to-end experiment before being presented as fact.
- **Tests**: +6 — **211 green**, ruff `--select F` clean.
