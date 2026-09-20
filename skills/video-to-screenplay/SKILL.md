---
name: video-to-screenplay
description: 将动漫、电影或电视剧视频转化为制作级中文场号制剧本。采用干净工作区协议（materials/ → .cache/ → output/）、前置字幕决策门与 API key 前置检查门、FFmpeg 场景切点关键帧提取、Qwen3.8-Omni 声学说话人分离（run 直连 DashScope，MCP omni_multi_speaker_asr 回退，均不可用时全 null 降级）、LGSS 式动态规划场景分组（含关键帧色板亲和度）、可选声画理解 pass（Omni 看视频产出动作/镜头/声学/屏显文字证据）、多模态场景理解 pass，以及毫秒级时间线对齐。数据外发声明：两处 Qwen3.8-Omni 能力需要 DASHSCOPE_API_KEY，会把 16kHz 音频分片与 ≤90 秒视频段上传至 DashScope（默认 dashscope.aliyuncs.com，可经 DASHSCOPE_BASE_URL 改变——启用前必须按 SECURITY.md 向用户披露并获得确认）；台词只来自字幕，模型输出绝不作为台词文本。当用户提供视频素材或要求逆向还原剧本时使用。
---

# 视频转剧本流水线（`video-to-screenplay`）

一条逆向还原流水线：把视频反编译为制作级亚洲场号制剧本，附带关键帧真实画面依据、声学说话人归属、画外台词标注与逐字台词。确定性的 Python 阶段负责一切可量化的计算；感知型任务统一交给通用多模态模型——场景理解由 agent 本身逐场完成，说话人**归属**由声学分离完成（`speaker_diarize.py run` 经 `omni_client.py` 直连 DashScope 调 Qwen3.8-Omni 按音色聚类，对 BGM/背景音鲁棒；MCP `omni_multi_speaker_asr` 为回退），角色**命名**只由字幕元数据多数票或场景理解 pass 给出。台词文本始终只来自字幕，经 `[[SUB:n]]` 占位符逐字拼装——声学与 OCR 都绝不参与台词文本本身。

---

## 1. 干净工作区架构（根目录零污染）

```text
📁 <项目根>/
├── 📂 materials/                  [只读] 视频、配套 .srt/.ass、可选 bible.json
├── 📂 output/                     [仅最终交付物]
└── 📂 .cache/
    ├── subtitles/extracted.json   规范化台词 + 时间码
    ├── visual/
    │   ├── shots.json             物理镜头切点 + 关键帧文件名
    │   ├── scenes.json            宏场景（LGSS 式动态规划求解）
    │   ├── av_notes.json          声画理解逐场证据（阶段 3.7，可选）
    │   └── keyframes/             镜头缩略图（shot_XXXX_XXXXXXms.jpg）
    ├── audio/
    │   ├── speakers.json          声学说话人标签 + characters_manifest
    │   ├── diarize_workorder.json 说话人工作单（prepare 产物，指引 run / MCP 调用）
    │   ├── source_audio*.m4a      16k 单声道抽取音频（>50 分钟自动分片）
    │   └── omni_diarized*.json    声学分离原始返回（run 直连或 MCP，证据，只读）
    ├── av/
    │   ├── av_workorder.json      声画理解工作单（prepare 产物）
    │   ├── seg_*.mp4              场景段 payload（480p CRF 阶梯转码）
    │   └── av_note_*.json         每段原始模型证据（只读）
    ├── alignment/
    │   ├── aligned_timeline.json  镜头↔台词主对齐（每条台词恰好分配一次）
    │   └── scene_manifest.json    场景证据包 + 写作契约（阶段 4）
    ├── scene_drafts/              阶段 4 产物：scene_XX.md 剧本文本 + thumbs/
    └── debug/                     临时检查产物
```

**契约**：交付物只进 `output/`；`materials/` 只读；一切中间文件都在 `.cache/` 内。

---

## 2. 流水线总览

```
[materials/*]
     │  阶段 1：工作区初始化 + 探测 + API key 前置检查 + 字幕决策门（AskUserQuestion / 快速失败退出码 5）
     │
     ├─ 视觉轨：    scene_detect.py       → shots.json + keyframes/
     └─ 台词轨：    subtitle_extractor.py → extracted.json
                    speaker_diarize.py prepare → 抽音频 + 工作单（退出码 6）
                    agent 调 MCP omni_multi_speaker_asr → omni_diarized*.json
                    speaker_diarize.py merge   → speakers.json（声学归属 + 元数据命名）
     │  阶段 3：汇流
     ├── semantic_scene_grouper.py → scenes.json   （软惩罚动态规划 + 可选 HSV 场景亲和度）
     │  阶段 3.5：叙事大纲（可选，McKee 序列层：价值转折单位，非地点单位）
     ├── narrative_outline.py → 工作单 → agent 写 narrative_structure.json → 校验
     │   （grouper 按序列墙分段、段内 DP 精切；无大纲则整体平切）
     ├── align_timeline.py         → aligned_timeline.json（每条台词只分配一次，取最大重叠）
     │  阶段 3.7：声画理解（可选，推荐——让 AI 真正看视频再写 △）
     ├── av_understand.py  prepare → run → merge → av_notes.json（动作/镜头/声学/屏显文字证据 + mouth_state 遵从率）
     │  阶段 3.8：演员表收敛（v0.6，唯一命名权威；只出候选，绝不产名字）
     ├── resolve_cast.py           → cast.json（槽位 / 候选分布 / margin / 弃权 / 过切分审计）
     ├── cast_signoff.py --render  → 一轮 4 个槽位的候选表（agent 打印给用户，用户回一句自然语言）
     ├── cast_signoff.py --apply   → <系列根>/cast.approved.json（带 version 与 history，工作区只读快照）
     │  阶段 4：场景写作（多模态 LLM 撰写真实剧本，分块可续跑）
     ├── build_scene_manifest.py   → scene_manifest.json（证据包 + 写作契约）
     ├── agent 读关键帧+av_notes+台词 → 写 scene_drafts/scene_XX.md（台词以 [[SUB:n]] 占位）
     │  阶段 5：逐字拼装与成稿
     └── splice_screenplay.py      → output/<标题>_影视文学剧本.md（表外专名致命，退出码 9）
```

---

## 3. 执行阶段

### 阶段 1 —— 工作区初始化、API key 前置检查与字幕决策门

```bash
python3 scripts/workspace.py init  --workspace "<ws>"
python3 scripts/workspace.py probe --workspace "<ws>"
python3 scripts/workspace.py doctor
```

- 🔑 **API key 前置检查**：`doctor` 报告 `diarization.dashscope_api_key`（只含 `"set"`/`"missing"`，绝不含 key 值）。为 `missing` 时**必须**用 `AskUserQuestion` 让用户三选一：① 现在配置 `DASHSCOPE_API_KEY` 后重跑检查；② 明确选择"无声学归属继续"（说话人列留空；阶段 2 仍可走 MCP 回退路径 B）；③ 中止流水线。**绝不静默降级**。
- 🎬 **OP/ED 窗口（可选，推荐）**：用户不想要 OP/ED 内容时，二选一配置：
  - **整季通用**（v0.6 起）：`python3 scripts/workspace.py init --workspace "<ws>" --series "<系列根>"`，
    把窗口写进 `<系列根>/op_ed_windows.json`。同一系列目录下的每一集都自动沿用，不必逐集拷贝；
    绑定还会把当时的演员表快照进 `.cache/cast.series.snapshot.json`（可复现）。
  - **单集**：写在 `materials/bible.json` 的 `op_ed_windows` 字段（旧版行为，完全支持）。
  两处都有且**不一致**时用系列目录并在 stderr 告警；一致则静默。
  ```json
  {"op_ed_windows": [{"start_ms": 84000, "end_ms": 105000, "label": "OP"},
                     {"start_ms": 1320000, "end_ms": 1440000, "label": "ED"}]}
  ```
  配置后：字幕行在提取时即被过滤（`extracted.json` 记录 `op_ed_filtered` 明细含**逐条被删文本**，存活行重编号保持 [[SUB:n]] 连续）、声学分离丢弃窗口内语音段（歌手不再混入角色表）、声画理解跳过窗口内场景（每集省 2-3 次调用）、manifest 把窗口内**且无幸存台词**的场景标为 `op_ed` 并写好单行 stub——成稿在相应位置只出现一行 **（动画 OP）/（动画 ED）**。若某场落在窗口内却留有压边台词（该台词过半在窗口外，故存活），它按普通场景撰写并带 `op_ed` 注记（stub 装不下 `[[SUB:n]]`，强行 stub 会让拼装致命退出）。不配置 = 行为与旧版完全一致。

probe 会报告 `ffprobe_available`、外部字幕文件与内嵌字幕流。随后通过 `AskUserQuestion` 询问用户：
- **外挂字幕**（一级）：在 `materials/` 中检出 `.srt`/`.ass`。
- **内嵌软字幕流**（二级）：用 ffmpeg 抽取（要求 `ffprobe_available: true`）。多语言轨时自动**优先用户语言**（`--lang`，默认 `chi,zho,chs,cht,zh`），并自动跳过 Forced / Signs / 歌曲类字幕牌轨（每条跳过都有 stderr 告警）；最终选择与被跳过的轨记录在 extracted.json 的 `embedded_stream` 字段备查。
- **硬字幕 OCR**（三级）：望言 OCR MCP（`POST /import → /predet → /pipeline → /export`）。OCR 完成后由你把导出文本按规范 schema 落盘到 `.cache/subtitles/extracted.json`——`{"source_tier": "TIER_3_HARDCODED_OCR", "video_path": …, "items": [{"index": 1, "start_ms": …, "end_ms": …, "text": …}, …]}`（index 从 1 连续递增，时间用毫秒整数）。后续所有阶段只认这个文件。
- 🟡 **只剩 OCR 档时不是拒止**：外挂与内封都落空 → `subtitle_extractor.py --require-subtitles` 先把 Tier 3 载荷（含 `status: NEEDS_OCR` 与 OCR 调用指令）打到 stdout，再以**退出码 6** 结束，等待你跑 OCR 并落盘 extracted.json。**绝不**在这一步退出码 5。
- 🛑 **快速失败**：只有当你确认素材真的无对白（纯画面）时，才运行 `python3 scripts/workspace.py check-subtitles --mode none`，它以退出码 5 结束。绝不编造台词。

### 阶段 2 —— 双轨提取（并发）+ 声学说话人分离

```bash
# 视觉轨（后台）
python3 scripts/scene_detect.py --workspace "<ws>" --threshold 0.35 > "<ws>/.cache/visual/shots.json" &
PID_VISUAL=$!
# 台词轨（前台）
python3 scripts/subtitle_extractor.py --workspace "<ws>" --require-subtitles > "<ws>/.cache/subtitles/extracted.json"
# 声学说话人三步：prepare（抽音频 + 工作单）→ run 直连（或 MCP 回退）→ merge（绑定 + 命名）
python3 scripts/speaker_diarize.py --workspace "<ws>" prepare > "<ws>/.cache/audio/diarize_workorder.json"
wait $PID_VISUAL
```

- `scene_detect.py` 把镜头 JSON 打印到 **stdout**（纯过滤器；按示例重定向）。缺 FFmpeg → 退出码 3。
- `speaker_diarize.py prepare` 用 ffmpeg 抽 16k 单声道音频；多分片时切点**吸附静音点**（silencedetect，±5s 容差）、相邻分片**重叠 ±3 秒**（merge 据此跨片对齐身份）；默认 >50 分钟自动分片，直连单调用死于超时时用 `prepare --chunk-seconds 1200` 重切。写 `diarize_workorder.json` 后**退出码 6**——等待感知完成（路径 A 或 B）。无音频流的工作单 `status=no_audio_stream`。

**声学分离（两条路径，产物同 schema）**：

- **路径 A（推荐）`run` 直连**——脚本经 `omni_client.py` 直调 DashScope（key 只从环境变量 `DASHSCOPE_API_KEY` 读，缺失退出码 8 并给出三条出路）：
  ```bash
  python3 scripts/speaker_diarize.py --workspace "<ws>" run
  ```
  无客户端工具窗口——流式延迟约 0.5–1×音频时长（24 分钟单片 ≈ 10–20 分钟），**用后台任务执行**；瞬态错误（超时/连接/429/5xx/空补全）代码化指数退避（`V2S_OMNI_ATTEMPTS` 默认 3），单片失败不阻塞其余分片；**断点续跑**——重跑 `run` 只补缺失分片，`--force` 全部重做。单分片 >25 分钟会 WARN（改用 `prepare --chunk-seconds 1200` 重切）。日志只含异常类型+状态码+主机名，key 绝不落盘、绝不进日志。
- **路径 B（回退）MCP 工具**——对工作单 `parts[]` 的每一片调用 MCP `omni_multi_speaker_asr`（Qwen-MM-Plugins `api` 插件；默认模型 qwen3.8-omni-flash）：`file_path` = 分片绝对路径，`format: "json"`；`num_speakers` 仅当 bible 明确人数时传；`language` 默认不传（自动检测）。**上下文卫生**：此循环是纯机械动作（调工具 → 存文件），**委托一个子代理执行**——每片的工具返回（JSON 与 SRT 双份文本）只进子代理上下文，主会话只收"全部已保存"的一句结果；把每个返回的 JSON block 原样保存到工作单指定的 `output` 路径。
- **静默片是完成片**：某片音频确实无人声时，模型返回 `{"segments": []}` 即视为该片已完成——重跑 `run` 会跳过它（`[SKIP] ... reported no speech`），不再为同一片段重复付费；merge 端只 WARN 不再以退出码 7 阻断。
- 两条路径产物一致：`{"speakers": [...], "segments": [{"speaker","start","end","text"}]}`（秒制；直连多一个 `meta` 溯源块，merge 端忽略）。分离质量由模型音色聚类保证（对音乐/背景音鲁棒）；**Omni 转写文本只作证据**（`text_agreement` 校验用），绝不进剧本正文。

```bash
python3 scripts/speaker_diarize.py --workspace "<ws>" merge > "<ws>/.cache/audio/speakers.json"
```

- `merge` 确定性执行：分片时间偏移还原 → **跨片身份对齐**（不同分片的标签是局部命名空间，只有重叠区里同一段语音被两片各自标出——共现证据——才经 union-find 合并；无证据不合并，宁拆不并）→ **全局字幕-音频偏移估计**（字幕只是粗框，常整体超前音频 1–3 秒；±5s/0.25s 步长做整体滞后扫描、取总覆盖最大者，≥6 行才启用）→ 每条字幕按**校正后最大时间重叠**绑定音色簇（`SPEAKER_A1…`，重叠 ≥40% 的次簇记入 `secondary_speaker`（重叠按声学簇**累计**后再排序，故同簇多段短语音合计过半时不会漏记）→ 元数据多数票（份额 ≥0.6 且 ≥2 票）给簇**起名**。**归属 100% 归声学，元数据只起名、绝不改判归属**；Omni 转写与字幕的一致性记入 `text_agreement` 仅作报告。
- 无 DASHSCOPE_API_KEY 且无 MCP 插件 / 无音频流 → 改跑 `merge --empty-fallback`：生成说话人全 `null` 的合法 `speakers.json` + WARN，流水线继续（等价于"未归属"状态），成稿说话人列留空。该产物自报底细：`acoustic_clustering_enabled: false`、`diarization_source.backend: "none"`——**读它的人必须看这两个字段**，降级产物绝不自称声学分结果。

🔴 **检查点**：`shots.json` 已生成、`failed_keyframes[]` 已记录；`extracted.json` 非空；`speakers.json` 已产出（声学合并或 `--empty-fallback` 皆可）。

### 阶段 3 —— 汇流：语义分组与时间线对齐

```bash
python3 scripts/semantic_scene_grouper.py --workspace "<ws>" > "<ws>/.cache/visual/scenes.json"
python3 scripts/align_timeline.py --workspace "<ws>" > "<ws>/.cache/alignment/aligned_timeline.json"
```

- **分组器**：跨越台词的切点带一个较大且有限的惩罚（软约束——求解器绝不死锁成整集单一场景）；输出中的 `forced_dialogue_cuts` 列出最优解不得不切开的台词保护边界，stderr 会告警。装好 numpy/opencv 后，HSV 关键帧色板距离（LGSS 的 "place" 代理）会锐化边界；否则退化为静默/时长启发式并打印 `[INFO]`。`visual_affinity_enabled` 反映**实际可测边界数**（依赖装了但关键帧全缺 → false，并 WARN）；`visual_boundary_coverage` 给出 [可测, 总边界]。**零宏场景是致命错误**（退出码 1）——空 scenes.json 只会让成稿静默为空。
- **层级模式**会忽略 `--target-scenes`/`--min-scenes`（每序列按自身粒度求解）——用到这两个 flag 时 stderr 明确告警，不静默吞参数。
- **对齐器**：每条台词被分配给且仅分配给一个镜头（取时间重叠最大者）；达不到重叠门槛（>100ms）的台词**不会被丢弃**，而是绑到时间最近的镜头并点名告警（`cues_nearest_shot_fallback`）——拼装器把"从未被认领的字幕"判为致命错误，丢一条就等于把整集卡死在最后一步。有台词但无任何镜头 → 退出码 1，绝不产出空洞时间线。逐行给出 `OFF_SCREEN`/`VOICE_OVER`/`INTERNAL_MONOLOGUE` 标记。

### 阶段 3.5 —— 叙事大纲（可选；McKee 序列层）

```bash
python3 scripts/narrative_outline.py --workspace "<ws>"
```

- 首次运行生成 `.cache/alignment/narrative_workorder.json`（全部台词流 + 大纲合同 + 目标 schema），并因 `narrative_structure.json` 缺失以**退出码 6** 提示待写。
- agent 依合同写大纲：序列 = 价值转折单位（2-5 场递增收在序列高潮；给每个序列定题目；写明价值 from→to；边界落在转折点上；单集约 4-8 个序列；无对白段按时间自动归入相邻序列）。序列必须无缝覆盖全部 `sub_index`。
- 重跑校验通过后，grouper 自动启用**层级模式**：序列墙吸附到 **15 秒窗口内**的最近物理切点；窗口内找不到切点时落在其之前最近的切点，并在 `wall_snaps[]` 标记 `within_snap_window: false` + stderr 告警（绝不假装吸附成功）。逐序列独立求解场景（粒度自适应），每场继承 `sequence_title`/`sequence_value`，禁止跨序列成场；单镜头配多序列大纲等退化输入会回落平切而非崩溃；无大纲则与旧版完全一致的平切。
- 价值判定是理解任务，归 agent；切点归 DP——单模型约束下的分工。

🔴 **检查点**：`total_macro_scenes` 合理（典型 8~35），`aligned_timeline.json` 覆盖全部台词。

### 阶段 3.7 —— 声画理解（可选，推荐：让 AI 真正看视频再写 △）

```bash
python3 scripts/av_understand.py --workspace "<ws>" prepare   # 切段 + 转码，exit 0
python3 scripts/av_understand.py --workspace "<ws>" run       # 直连逐段理解 + merge，缺 key exit 8
```

- **prepare**：读 scenes.json，>90 秒的场景切成 ≤90 秒段（`--segment-seconds` 可调；段间 5 秒重叠、段不跨场景；**短尾折叠例外至 +10 秒**——91 秒的场景仍是 1 段，避免为 1 秒碎尾多付一次调用），逐段转码 480p/CRF 阶梯 MP4（超 inline 预算自动降档），写 `av_workorder.json`；配置了 OP/ED 窗口时自动跳过窗口内的段（记录在 `skipped_op_ed`）。**启用前用 AskUserQuestion 预告成本**（实测校准，ep5 10 分钟片/35 段：每段 ~38 秒墙钟、~5.4k tokens，合计 ~3.2× 片长墙钟、~190k tokens——段数 = 工作单 segments 数；**后台执行**），经用户确认再执行。
- **run**：逐段直连 DashScope。模型只产出四通道结构化证据——`visual.actions`（谁做了什么）、`visual.camera`（景别/运镜）、`visible_text`（屏显文字逐字+起止）、`acoustic`（音效事件+音乐情绪）——prompt 里**明令不做台词转写、不给人物起名**（人物一律描述性称呼如"红衣女子"，命名权归场景写作 pass）；不确定之处进 `uncertain`。逐段断点续跑、单片失败不阻塞（`--force` 全部重做）。
- **merge（run 内置）**：段内相对秒 +start 映射回源时间轴 → 按场景聚合 → **覆盖率校验**（<100% 的场景 WARN，缺口区间由写作者回退关键帧证据）→ `.cache/visual/av_notes.json`。
- 🔴 **红线**：av_notes 是**证据层**——绝不成为台词文本（台词永远 `[[SUB:n]]` 逐字来自字幕）、绝不改判声学归属、绝不替代关键帧。
- 跳过本 pass：阶段 4 退回纯关键帧证据，流水线其余不受影响。

### 阶段 3.8 —— 演员表：唯一命名权威（v0.6，设计见 `references/cast_glossary_v0.6.md`）

```bash
python3 scripts/resolve_cast.py     --workspace "<ws>"                       # 只收敛，不产名字
python3 scripts/cast_signoff.py     --workspace "<ws>" --render              # 打印本轮候选表（≤4 槽位）
python3 scripts/cast_signoff.py     --workspace "<ws>" --reply "<用户原话>" --apply
python3 scripts/workspace.py        series  --workspace "<ws>"               # 看这集实际用的是哪张表
```

- **为什么要这一层**：v0.5 之前"角色名只能来自 bible/manifest/台词"是**叮嘱**，没有检查器兜底，实测必然失败（写作者把被称呼者"茉里"当成说话人并据此改判声学归属）。现在命名权在人手里，成稿由机器强制。
- **三道强制闸门（致命项都有反例测试）**：① `resolve_cast.py` 永不产出名字——簇只能进槽位，`slots → entities` 必须有 `approved_by: "human"` 记录；② §4.1 三级匹配（**只用正证据、至少两族、含糊即新建 pending**，因为过度合并会被后续每一集继承、不可逆）；③ `splice_screenplay.py` 表外专名**退出码 9**（无表可查时降为告警并在成稿标注，见下）。
- **🔴 检查点（必须问用户，`pending` 为空则不问）**：阶段 3.7/3.8 跑完后打印候选表并问：
  > 本集识别出 N 个说话人：已定名 x 个（沿用系列演员表）、待你定名 y 个、无法判定 z 个。
  > 现在定名，还是先出草稿（成稿会标注"演员表未经核验"）？

  一轮最多 4 个槽位，**问完问"后面还有 y 个，继续吗？"**——不是"只许问 4 个、剩下算未定名"。
- **回话必须解析成五种结局之一**：明确对应候选 → 写盘 + 回显 diff；提到表外新名字 → **不写**、回问确认；按位置指代（"第二个"）→ 回显解析出的槽位号请确认；完全解析不出 → 重打表 + 给可改的行内模板；用户对 diff 说"不对" → 整轮不写（写盘是原子替换，不存在半条记录）。**绝不静默跳过、绝不部分写入**。
- **`--draft` 降级语义**：未绑定系列或用户不签核也能出稿，但成稿**头部表**里必须出现 `⚠️ 演员表未经核验（N 个槽位为候选）`（不是只写在附录——附录是读者最先跳过的地方）。理由：没有降级通道，人就会为了省事绕过流程，表反而形同虚设。
- **呼语的单向性**：台词里的人名是**被称呼者**。它只能作为①某槽位的命名候选、②"说这句话的簇不是他"的负证据；**永远不能**用来判定"这句话是谁说的"（`vocatives.py` 的 `role` 枚举里根本没有这个值，测试钉住）。
- **描述性标签不等于失败**：`女声`／`系统音`／`面试的店主` 一律放行（形状判定），但会被计数进成稿的"说话人标签构成"——一集里描述性标签多，说明演员表没做完，而不是写作者不乖。
- 跳过本阶段 = 全部走 `--draft` 标注，流水线其余不受影响。

### 阶段 4 —— 场景写作（由 LLM 撰写真正的剧本，分块可续跑）

```bash
python3 scripts/build_scene_manifest.py --workspace "<ws>"
```

manifest 是逐场**证据包**：`keyframes_thumbs`（640px 降采样帧）、该场逐字 `dialogues`（各带 `sub_index` + 时间码）、`av_notes`（若跑过阶段 3.7：逐段动作/运镜/声学/屏显文字证据，时间码已映射回源时间轴）、`bible_names`，以及 `exemplars`（前几集剧本——读其中一份作为格式参照）。对每个 `draft_status == "missing"` 的场景：

1. `Read` 它的缩略图、av_notes（如有）、台词，以及（如有）上一场 `scene_(N-1).md` 的结尾以保持连贯。
2. 写 `.cache/scene_drafts/scene_XX.md` —— 该场真正的场号制剧本文本：
   - `## 场 N【地点·事件标题】` H2 场头（标题由你起名；时间码由拼装器注入行尾，不要手写 TC）+ `**内景/外景·时辰**｜注记` 时空行 + `**人物：**` 行；不写 `> 概要：` 引用行；
   - 台词行按说话人合并：同一角色连续说话写一行、句间用全角斜杠——`**角色名**（提示）：[[SUB:2]]／[[SUB:3]]`；说话人切换才另起一行；
   - `△` 动作段是导演笔记，织在台词之间：演出指示、表演指导、镜头强调——写光、物、身体、声音；人物首次上镜写作 `△【角色】（特征）`；
   - `（画外）/【内心独白】` 括注、`△【插入：…】` 闪回、`（字幕卡：…）` 字幕。
3. 硬规则：**绝不复打台词**——只允许占位符，每个 `sub_index` 恰好一次、按序出现（顺序倒置现在是 `splice` 的致命错误，会点名出错的两条下标）；角色名只能来自 bible/manifest/台词中的称呼（陌生人物用描述性标签，绝不杜撰专有名词）；**目击者原则**——只写摄影机能拍到、录音机能录到的内容，零心理描写与回忆（不写"她想起……心中悔恨"，写"她动作僵住，攥紧拳头，指甲陷进肉里"——演员能演、摄影机能拍）；**Notnot 原则**——写"有什么"不写"没什么"，禁否定式动作句（不写"他没有回答"，写"他保持沉默"）；△ 禁止"两人交谈"式空泛句；若本场带 av_notes——△ 优先取材其中真实可见的动作/运镜/音效/屏显文字（时间码对得上的优先），av_notes 的描述绝不当台词写进正文，其中人物是描述性称呼、真名以台词与 characters_manifest 为准。

完整契约内嵌于 `manifest.instructions`。随时可重跑 builder——它会报告 `drafts_written` / `drafts_missing`（检查点/续跑）。

### 阶段 5 —— 逐字拼装与成稿

```bash
python3 scripts/splice_screenplay.py --workspace "<ws>" --title "第 N 话 …"
```

- 把每个 `[[SUB:n]]` 替换为逐字字幕文本——**结构天然保证保真**；复打台词在结构上不可能发生。
- 场头规范化：每场 H2 统一为 `## 场 N【标题】（起 - 止）`——标题保留写作阶段起的名字，时间码从 scene_manifest 确定性注入；遗留的 `> 概要：` 引用行会被剥离。
- 校验覆盖：缺失、重复、错位或孤儿的占位符都是**致命**错误，并点名出错的下标；说话人名单 lint 对 bible/manifest/通用角色词之外的名字告警。制作专属的描述性称呼（如「面试的店主」）写进 `materials/bible.json` 的 `"speaker_whitelist": [...]` 数组即可消除告警——插件内置白名单只含通用角色词，绝不烧入单部作品的词汇。
- 组装整集文档：元数据头（字幕来源、镜头/场景计数）、场次总表、拼装后的各场、附录（声画关系统计 + 保真报告）。
- 输出：`output/<标题>_影视文学剧本.md`。配置了 OP/ED 窗口时，成稿在相应位置只含一行 **（动画 OP）/（动画 ED）** 标注（窗口内仍留有压边台词的场按普通场景撰写，并在场头下方加同一行标注），保真报告记录被过滤的行数与窗口明细。
- 🛑 **停下复查**：splice 退出码 0 且 `dialogue_spliced == dialogue_total`；确认所有 lint 告警已知晓；交付物在 `output/`，项目根目录无散落文件，最终 `.md` 中零 `SPEAKER_` 字符串。
- 🛑 **命名复查（v0.6）**：成稿头部「演员表」一行写的是「已签核」而非「⚠️ 未核验」；若为未核验，必须是**用户明确选择**先出稿，而不是这一阶段被悄悄跳过；正文里每一个专名说话人都能在 `cast.approved.json` 查到 `approved_by: "human"` 的记录。

---

## 4. 失败模式与兜底矩阵

| 失败情形 | 信号 | 响应 |
| :--- | :--- | :--- |
| 完全没有字幕 | 用户确认素材无对白 | 硬性拒绝：`workspace.py check-subtitles --mode none` 退出码 5，请求在 `materials/` 放置 `.srt`/`.ass` |
| 只有硬字幕（无外挂/无内封） | 提取器 **退出码 6** + stdout 带 `status: NEEDS_OCR` | 这是出路不是拒止：按 Tier 3 指令跑 OCR，把结果落盘 `extracted.json` 后继续；**不要**误当作无字幕直接退出码 5 |
| 阶段 1 未配置 DASHSCOPE_API_KEY | doctor 报告 `diarization.dashscope_api_key=missing` | AskUserQuestion 显式三选一（配 key 重跑 / 降级继续 / 中止）；选择降级后 `run` 仍会以退出码 8 二次拦截 |
| 无 DASHSCOPE_API_KEY（路径 A 首次执行） | run **退出码 8** | 三选一：配 key 重跑 run；走路径 B 调 MCP；`merge --empty-fallback` 全 null 继续 |
| 直连单片网络瞬态（超时/连接/429/5xx/空补全） | run 内已自动指数退避重试（`V2S_OMNI_ATTEMPTS` 默认 3） | 重试耗尽的分片报 ERROR 但不阻塞其余；处置后重跑 `run` 只补缺 |
| 直连单调用近超时天花板（超长分片） | run 对 >25 分钟分片 WARN；调用死于超时 | `prepare --chunk-seconds 1200` 重切后重跑 `run` |
| Omni 输出缺失/非法/分片不全 | merge **退出码 7** 并点名文件 | 重跑 `run`（断点续跑只补缺）或按工作单补做对应分片的 MCP 调用，再重跑 `merge` |
| 某个分片确实无人声（片头曲/纯动作段） | `run` 打 `[SKIP] ... (reported no speech)`；merge 只 WARN | 静默片算已完成，**不要** `--force` 反复重调付费接口；若怀疑漏识别，用 `prepare --chunk-seconds` 重切该段 |
| 单次 MCP 调用超时/失败（路径 B：客户端执行窗口、上游波动） | 工具执行超时或连接中断 | `prepare --chunk-seconds 30~40` 重切后逐片重调（上游不稳时等待 1–3 分钟再试），或改走路径 A |
| 视频无音频流 | 工作单 `status=no_audio_stream` | 直接 `merge --empty-fallback`；对白只存在于字幕层，流水线不受影响 |
| Omni 返回零语音段 | merge WARN `omni_returned_no_speech` | 音轨可能为纯音乐/环境声；声学层留空不阻塞，后续阶段照常 |
| 缺 ffprobe | probe `ffprobe_available: false` | 告警；内嵌字幕流选项不可用；`brew install ffmpeg` |
| 内嵌轨疑似纯字幕牌（SIGN/Forced） | extracted.json 的 `embedded_stream.reason=last_resort_all_look_like_signs`，或行文本多为画面文字 | 用 `--lang` 指定其他语言轨重抽；多轨源可显式映射对白轨重剪（`ffmpeg -map 0:<idx>`） |
| 片源有 OP/ED 但成稿出现歌词"场景"或歌手簇 | 未配置 `op_ed_windows` | 在 bible.json 配置窗口后重跑——字幕过滤/声学丢弃/av 跳过/单行标注全链路自动生效 |
| 声画理解无 key / 用户选择跳过 | av run **退出码 8** / AskUserQuestion 选跳过 | 阶段 4 退回纯关键帧证据，流水线其余不受影响 |
| 声画理解单段调用失败 | av run 逐段 `[ERROR]` 不阻塞其余 | 重跑 `run`（断点续跑只补缺）；或放弃该 pass |
| 场景声画覆盖缺口 | av_notes.json 中该场景 `covered_pct<100` + merge WARN | 缺口区间回退关键帧证据；必要时 `--segment-seconds` 调小重切 `prepare` |
| 缺 FFmpeg | scene_detect 退出码 3 / doctor 退出码 3 | 停止，原样上报，绝不借用无关 MCP |
| 分组器零宏场景 | **退出码 1** 并报出镜头/台词数 | 空 scenes.json 会让成稿静默为空：检查 shots.json 时间码；层级模式下检查大纲 sub 区间是否覆盖全片 |
| 台词落在所有镜头之外 | 对齐器 WARN + `cues_nearest_shot_fallback` | 无需干预：自动绑到最近镜头；若计数异常偏高，说明切镜时间码或字幕时间基有问题 |
| 分组器 `forced_dialogue_cuts > 0` | stderr WARN | 可能出现台词中途切场；检查字幕对齐 / `--max-scenes` 余量 |
| 镜头数超出 `40 × max_scenes` | （旧版会静默塌缩成 1 场） | 求解窗已自动加宽至 `⌈镜头数 / max_scenes⌉` 自愈；若仍见"collapse into a single scene" WARN，调大 `--max-scenes` |
| 关键帧抓取失败 | `failed_keyframes[]`（shots.json → scenes.json → aligned_timeline 全程透传） | 无已验证帧的场在时间线里 `visual_verified: false`；草稿不得描写它们 |
| 场景文本缺失/不完整 | splice 致命错误并点名下标 | 写作 pass 必须让每条字幕恰好覆盖一次后才能成稿 |
| 叙事大纲未写 | narrative_outline 退出码 6 | 可选层：按工作单补 `narrative_structure.json`，或直接以平切继续 |
| **成稿出现表外专名说话人** | splice **退出码 9** 并逐个点名 | 演员表闸门在生效：回阶段 3.8 签核该槽位，或（用户明确同意先出稿）解除系列绑定走 `--draft` 标注。**绝不**为了过闸门往 bible.json 里塞名字 |
| 说话人全是描述性标签（`女声`／`系统音`／`面试的店主`） | 成稿「说话人标签构成」里描述性占比高 | 不是错误，是演员表没做完：把 pending 槽位问完再定名 |
| `resolve_cast.py` 有 `pending` 但无人签核 | cast.json 的 `abstention.rate` | 成稿头部会标 ⚠️ 未核验；用户选择跳过即可继续，**不得由 agent 代为定名** |
| 演员表无处可写 | cast_signoff **退出码 3**（未绑定系列） | `workspace.py series --workspace <ws> --bind <系列根>` 后重试 |
| 视觉族不参与判定 | av_notes 的 `mouth_compliance.usable=false` | 提示词遵从未达标（或还没有 v3 产物）：按设计文档 §10 P3 降级为只做拼接视频探测，别把「没测出」当「没人说话」 |

---

## 5. 禁止行为与反模式

| 反模式 | 强制纠正 |
| :--- | :--- |
| 根目录乱丢文件 | 一切非交付物都在 `.cache/` 内 |
| 为无声画面编造台词 | 硬性拒绝；无字幕 → 不开工 |
| 把暂定标签渲染成真名 | `SPEAKER_*`（含声学 `SPEAKER_A*`）/ 未归属 → `人物`，直到场景理解 pass 给出命名 |
| 用 Omni 转写文本代替字幕台词 | 转写仅作 `text_agreement` 证据；台词一律 `[[SUB:n]]` 逐字来自字幕 |
| 把 av_notes 的描述当台词写进正文 | av_notes 是目击证据层——台词永远 `[[SUB:n]]` 逐字来自字幕；其中人物是描述性称呼，真名以台词/manifest 为准 |
| 把 API key 写进代码/命令行/证据文件 | key 只从环境变量读（无 key 时 run 以退出码 8 拒绝）；日志只含异常类型+状态码+主机名 |
| 让降级产物冒充声学结论 | `merge --empty-fallback` 写出的 `speakers.json` 必须自报 `acoustic_clustering_enabled: false` + `backend: "none"`；下游判断说话人列是否可用只看这两个字段，绝不靠"有没有名字"猜 |
| 在对齐阶段丢弃低重叠台词 | 每条台词必须有着落（重叠不足则绑最近镜头并上报）：拼装器把"从未被认领"判为致命，丢一条就等于把整集卡死在最后一步 |
| 把静默分片当失败反复重调 | 模型明确返回空 `segments` 就是完成态；反复 `run`/`--force` 只会重复付费 |
| 用字幕元数据改判声学归属 | 元数据只给声学簇**投票起名**（份额 ≥0.6、≥2 票）；"谁在说"永远由音色聚类决定 |
| 幻觉动作行 | 动作草稿只写关键帧可见内容；未验证的镜头保持未验证 |
| 在 DP 中硬禁台词切点 | 软惩罚 + `forced_dialogue_cuts` 报告——绝不用硬 INF 静默地把整集塌缩成一场 |
| 跳过场景写作 pass | splice 会中止，直到每场文本覆盖其全部字幕；必须先跑阶段 4，若跳过须如实说明 |
| 把台词复打进场景文本 | 绝不——台词经 [[SUB:n]] 拼装逐字流入；改写在结构上不可能 |
| 用 `> 概要：` 引用行或在草稿手写时间码 | 概要只进场次总表；TC 由拼装器注入 H2 场头 |
| 把交付物写到任意路径 | 只输出到 `<ws>/output/` |
