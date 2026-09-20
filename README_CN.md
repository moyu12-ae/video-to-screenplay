# 影视逆向剧本流水线插件 (`video-to-screenplay`)

[English](README.md) | 简体中文

一个 **Claude Code 插件**：把长视频与番剧逆向还原为标准亚洲场号制影视剧本——同一个仓库打包给三种 agent 宿主（Claude Code 原生走 `.claude-plugin/`，另附 ZCode 与 Qoder 适配）。确定性 Python 阶段负责一切可测量的计算（切镜、时间码、台词对齐）；感知型任务统一交给通用多模态大模型——场景理解（地点、时辰、出场人物、场面调度）由 agent 本身逐场完成，说话人归属由 **Qwen3.8-Omni 按音色声学聚类**（由流水线自身直连 DashScope 完成——`speaker_diarize.py run`；MCP 工具 `omni_multi_speaker_asr` 作回退；未配置时优雅降级为全部未归属）完成。台词文本永远只来自字幕、经 `[[SUB:n]]` 占位符逐字拼装——声学与 OCR 都不碰台词文本本身。

## ⚠️ 使用前必须配置阿里云 API Key

**使用本插件前，请先配置阿里云 DashScope API Key。**

1. 在[阿里云百炼控制台](https://bailian.console.aliyun.com/)的 API-KEY 管理页创建密钥；
2. 导出环境变量：`export DASHSCOPE_API_KEY="sk-..."`；
3. （可选）指向其他 OpenAI 兼容端点：`export DASHSCOPE_BASE_URL="https://..."`。

该 Key 同时支撑两处 Qwen3.8-Omni 能力：声学说话人分离，以及可选的声画理解 pass。阶段 1 会做前置检查：未配置时流水线会显式询问——配置 Key 后重跑，或明确选择"说话人列留空"继续（`speaker_diarize.py run` 以退出码 8 二次拦截）。

### 什么会离开你的机器（以及 Key 会发往何处）

本插件会把素材上传给第三方 API。完整声明见 `SECURITY.md`；`scripts/config_spec.py` 是机器可读的唯一事实源，文档与它漂移时测试会失败。

| 环境变量 | 默认值 | 控制什么 |
| :--- | :--- | :--- |
| `DASHSCOPE_API_KEY` | 未设置 | 两处 Omni pass 的 Bearer 凭据；全仓库只在一处函数读取，绝不落盘、绝不进日志 |
| `DASHSCOPE_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | **哪个主机接收 key 与媒体**（强制 https + 公网地址） |
| `V2S_OMNI_MODEL` | `qwen3.8-omni-flash` | 用哪个全模态模型解读你的画面 |
| `V2S_OMNI_ATTEMPTS` | `3` | 单次请求的重试预算（仅限瞬时错误） |
| `V2S_OMNI_TIMEOUT_SEC` | `1800` | 单个流式请求的墙钟上限 |

会上传：16 kHz 单声道音频分片（声学分离）、逐场 ≤90 秒 480p 含单声道音轨的视频段（声画理解）。
不会作为转写内容上传：台词——字幕在本地解析并逐字拼装，AV prompt 明令禁止转写对白。
`workspace.py doctor` 在报告 key 状态的同时报告它将去往的主机与所用模型。

## 工作原理

1. **纯净三层工作区** —— `materials/`（只读输入）→ `.cache/`（可随时清空的中间产物）→ `output/`（只放最终剧本）。
2. **字幕门禁** —— 外挂字幕 → 容器内封软字幕 → OCR 档。只有 OCR 能服务时，提取器把 Tier 3 指令打到 stdout 并以**退出码 6**（等待感知阶段）结束——这是**出路，不是拒止**；只有在用户确认素材真的无对白之后，才由 `workspace.py check-subtitles --mode none` 以退出码 5 硬性拒止。绝不凭空捏造台词。 配置 OP/ED 窗口（v0.6 起可放 `<系列根>/op_ed_windows.json` 全季通用，也可留在单集 `bible.json`）后，片头/片尾歌词在提取时即被过滤——成稿只留一行（动画 OP/ED）标注；若某场落在窗口内却仍留有压边台词，它按普通场景撰写（stub 装不下 `[[SUB:n]]`），并同样加上这一行标注。
3. **音画双轨并行** —— FFmpeg 切镜与关键帧提取，和字幕提取、**声学说话人分离**（ffmpeg 抽 16k 单声道音频 → 直连客户端流式调用同一个 Qwen3.8-Omni 分离请求——代码化重试/退避、逐片断点续跑、无客户端工具窗口限制；MCP `omni_multi_speaker_asr` 保留为回退 → 按声学簇**累计重叠**绑定到字幕行，故一条台词上覆盖 ≥40% 的第二声源会记入 `secondary_speaker`）并发执行。多分片时切点吸附静音、相邻片重叠 ±3 秒，跨片身份只认"重叠区同段语音"的共现证据（union-find 对齐，宁拆不并）；**归属 100% 归声学**，字幕元数据只给声学簇投票起名。模型报告某片「无人声」时该片记为已完成，重跑只补缺、不会为同一片段重复计费。**归属 100% 归声学**；没有 API key（也没有 MCP 回退）时说话人全 null 降级，流水线不阻塞，且降级产物会自报底细（`acoustic_clustering_enabled: false`、`backend: "none"`）。同一次**已付费**调用还会返回**闭合枚举**的声学属性（`gender`／`age_band`／`timbre`，不确定一律 `unknown`），簇级按多数票聚合、`unknown` 票不进分母，并自带「模型到底答了几条」的遵从率计数。
4. **LGSS 思想场景聚类** —— 一维 DP 求解器把 200+ 物理切镜折叠为宏场景。对白穿越的剪切点施加**大额软惩罚**（绝非硬禁止——求解器不可能死锁成"整集一场"，任何被迫切口都会显式上报）。安装 numpy/opencv 后，启用关键帧 HSV 调色板距离（LGSS "place" 模态的轻量代理，理念源自 LGSS, CVPR 2020）锐化边界；未安装则优雅降级为静默/时长启发式。
5. **毫秒级对齐** —— 每条台词按最大时间重叠唯一归属到一个镜头；达不到重叠门槛的台词不会被丢弃，而是绑定到时间上最近的镜头并上报（对齐器因此绝不可能把「没人被要求放置」的字幕丢给拼装器），并标注 `ON_SCREEN` / `OFF_SCREEN` / `VOICE_OVER` / `INTERNAL_MONOLOGUE`。
6. **叙事大纲（可选，McKee 序列层）** —— `narrative_outline.py` 生成台词流工作单；agent 按「价值转折」判定写出序列骨架（题目 + 价值 from→to + 起止台词），grouper 随即切换为层级模式：序列墙在 **15 秒窗口内**吸附最近物理切点（窗口内无切点时落到其之前最近的切点，并以 `within_snap_window: false` 上报）、逐序列独立求解场景。无大纲则整体平切，行为不变。
7. **声画理解（可选，推荐）** —— `av_understand.py` 让模型真正"看"每个宏场景（>90 秒自动切段、CRF 阶梯转码塞进 inline 预算），返回结构化证据笔记——动作、镜头语言、音效事件、屏显文字——逐条时间码映射回源时间轴，并做逐场覆盖率校验。此 pass 明令**不做台词转写、不给人物起名**（人物一律"红衣女子"式可见称呼）；笔记是**写作证据层**，绝不成为台词文本。每条动作还带 `mouth_state`（`moving`／`still`／`not_visible`，窗口 ≤3 秒），merge 会报告模型的遵从率；不达标时视觉族**主动弃权**，而不是把「没看到」当成「没人在说话」。
8. **演员表——唯一命名权威（v0.6）** —— 借鉴本地化术语表的纪律：证据生产者只提候选、人签核、检查器强制执行。`resolve_cast.py` 把声学属性、口型证据与呼语折成**每簇的候选分布**（绝不是单一判决）；一个簇只有在**两个证据族一致且margin 清晰**时才能沿用已有身份，含糊就新建 pending 槽位——因为过度合并会被后续每一集继承，而多问一次只是成本。名字**只能**来自人的回答：`cast_signoff.py` 按轮次（一轮 4 个槽位、五种结局、有未确认项就整轮不写盘）写入`<系列根>/cast.approved.json`，带版本与历史，并在工作区留只读快照以保证可复现。`splice_screenplay.py` 随后把「名字形状却溯源不到签核条目」的说话人判为**致命（退出码 9）**，而诚实的描述性标签（`女声`、`系统音`）按形状放行并被计数而非惩罚——旧闸门恰好相反，它训练写作者去写**更长**的标签而不是**更有证据**的标签。不签核也能出稿，但成稿头部会写明演员表未核验；台词里出现的人名**永远不是**「这句话由他说出」的证据——呼语指向的是听者。
9. **场景理解与写作（分块、可断点续跑）** —— `build_scene_manifest.py` 生成逐场**证据包**（640px 关键帧缩略图、逐字台词与时间码、声画理解笔记（如有）、bible 人物名单、前集剧本范例、所属序列的价值弧）并内嵌完整写作契约；多模态 agent 逐场读图与笔记，撰写真正的场号制剧本文本 `scene_drafts/scene_XX.md`——`△` 动作段取材于真实看到听到的内容，台词一律以 `[[SUB:n]]` 占位符表示，**绝不复打**。
10. **逐字拼装成片** —— `splice_screenplay.py` 把每个 `[[SUB:n]]` 替换为逐字字幕原文——台词保真是**结构保证**的，改写在结构上不可能；缺失/重复/错位的占位符一律致命报错并点名下标。成稿含元数据头、场次总表（含序列列）、拼装正文与声画统计/保真度附录，输出 `output/<标题>_影视文学剧本.md`。

## 环境要求

- Python 3.10+
- PATH 上有 FFmpeg / ffprobe（`brew install ffmpeg`）
- 推荐：环境变量 `DASHSCOPE_API_KEY`——同时启用两处直连 API 的 Qwen3.8-Omni 能力：声学说话人分离（`speaker_diarize.py run`：无客户端工具窗口限制、指数退避重试、逐片断点续跑）与可选声画理解（`av_understand.py run`）。分离的回退方案：[Qwen-MM-Plugins](https://github.com/QwenLM/Qwen-MM-Plugins) 的 `api` 插件（MCP）。两者皆无时说话人列留空，流水线照常运行
- 可选：`pip install -r requirements.txt`（numpy + opencv-python-headless）——启用场景聚类的视觉 place 亲和度。其余全部纯标准库。

## 使用

### 按宿主安装

一个仓库、三份宿主清单——技能正文、脚本与写作合同完全共用，只有清单按宿主不同。`tests/test_packaging.py`
钉住三份清单的 name/version 一致。

| 宿主 | 清单 | 安装方式 |
| :--- | :--- | :--- |
| Claude Code | `.claude-plugin/plugin.json` + `marketplace.json` | `/plugins` → 把本仓库添加为市场（`moyu12-ae/video-to-screenplay`）后安装 |
| ZCode | `.zcode-plugin/plugin.json` | 同以往，从本仓库安装 |
| Qoder | `.qoder-plugin/plugin.json` | 添加市场安装，或本地安装——**已实测可用**：把包拷到 `~/.qoder-cn/plugins/local/video-to-screenplay/<版本>/`（含 `.qoder-plugin/`、`skills/`、`commands/`、`scripts/`、`SECURITY.md`），在 `~/.qoder-cn/plugins/installed_plugins_v2.json` 加一条 `video-to-screenplay@local`（`scope: user`、绝对 `installPath`、`marketId: "local"`），在 `~/.qoder-cn/settings.json` 里置 `enabledPlugins["video-to-screenplay@local"] = true`，然后**重启 Qoder**——插件集只在进程启动时读取一次，生效后技能会以 `video-to-screenplay:video-to-screenplay` 出现。包是快照拷贝，仓库改动后需重新拷贝 |

### 宿主内使用

技能会在任何"把这个视频逆向成剧本"的请求下激活。三个命令对应流水线的自然续跑边界：

```
/video-to-screenplay:init      # 工作区 + 环境与 API key 前置检查 + 字幕决策门
/video-to-screenplay:build     # 阶段 2-4：双轨提取、分组、对齐、证据包、逐场撰写
/video-to-screenplay:splice    # 阶段 5：逐字拼装 + 交付复查清单
```

> **宿主差异（2026-09-19 在 Qoder 桌面端实测）：**该宿主的插件命令**既不列出也不执行**——`@`/`/` 选择器
> 只列插件、技能、MCP server 与 agent，从不列命令；完整敲 `/video-to-screenplay:init` 会当作普通文本发进来。
> 这不是打包缺陷：同一个选择器里 `context7` 的 `/context7:docs` 同样不显示，而它的插件、MCP server、agent
> 都在，且我们的命令文件与该参照包形状完全一致。在 Qoder 下直接说要做哪一段即可（"跑阶段 1"、"把证据包做出来"）——
> `SKILL.md` 里有同样的阶段顺序、命令与检查点。命令在会展示斜杠命令的宿主（如 Claude Code）下可用。

或直接运行技能：

```
/video-to-screenplay
```

或直接驱动各阶段：

```bash
python3 scripts/workspace.py init  --workspace "<ws>"
python3 scripts/workspace.py probe --workspace "<ws>"
python3 scripts/scene_detect.py --workspace "<ws>" --threshold 0.35 > "<ws>/.cache/visual/shots.json"
python3 scripts/subtitle_extractor.py --workspace "<ws>" --require-subtitles > "<ws>/.cache/subtitles/extracted.json"
#   ↑ 退出码 6 + stdout 里的 Tier 3 载荷 = 硬字幕片：请跑 OCR 档并自行落盘 extracted.json；
#     退出码 5（纯画面拒止）属于 `workspace.py check-subtitles --mode none`，在用户确认无对白之后
python3 scripts/speaker_diarize.py --workspace "<ws>" prepare > "<ws>/.cache/audio/diarize_workorder.json"  # 退出码 6
python3 scripts/speaker_diarize.py --workspace "<ws>" run
#   ↑ 路径 A（推荐）：直连 DashScope（环境变量 DASHSCOPE_API_KEY；
#     代码化重试/退避、逐片断点续跑——重跑 run 只补缺失分片）。
#   ↑ 路径 B（回退）：由 agent 对工作单每个分片调用 MCP omni_multi_speaker_asr，
#     把返回 JSON 原样存到 .cache/audio/omni_diarized[.partNNN].json
python3 scripts/speaker_diarize.py --workspace "<ws>" merge > "<ws>/.cache/audio/speakers.json"
#   ↑ 两种后端皆不可用时改跑：merge --empty-fallback（说话人全 null 降级）
python3 scripts/semantic_scene_grouper.py --workspace "<ws>" > "<ws>/.cache/visual/scenes.json"
python3 scripts/narrative_outline.py --workspace "<ws>"   # 可选：agent 依工作单写叙事大纲后重跑校验
python3 scripts/align_timeline.py --workspace "<ws>" > "<ws>/.cache/alignment/aligned_timeline.json"
python3 scripts/build_scene_manifest.py --workspace "<ws>"
# agent 依据清单逐场撰写 .cache/scene_drafts/scene_XX.md（台词用 [[SUB:n]] 占位）
python3 scripts/splice_screenplay.py --workspace "<ws>" --title "<标题>"
```

## 开发

```bash
python3 -m pytest tests/ -q                 # 单元 + 端到端（需 ffmpeg）
python3 -m ruff check scripts tests --select F
```

`.github/workflows/ci.yml` 在每个操作系统上跑两遍测试——装与不装可选的 numpy/opencv——因为分组器的纯标准库降级路径本身就是契约的一部分。`tests/test_end_to_end.py` 用 ffmpeg 合成一段短片并跑完五个阶段，阶段之间的合同冲突（例如对齐器丢掉、拼装器却要求存在的台词）无法悄悄上线。

完整 agent 工作流、场景写作契约与故障矩阵见 [SKILL.md](skills/video-to-screenplay/SKILL.md)。

## 致谢

- [Qwen-MM-Plugins](https://github.com/QwenLM/Qwen-MM-Plugins)（Apache-2.0）——`scripts/omni_client.py` 的直连客户端改编自其请求构造/音频装配/重试实现；分离提示词逐字保留。
- [NarratoAI](https://github.com/linyqh/NarratoAI)（MIT）——瞬态重试与失败隔离的工程模式参考。

## 许可证

MIT，见 [LICENSE](LICENSE)。第三方改编归属见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
