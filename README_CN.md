# 影视逆向剧本流水线插件 (`video-to-screenplay`)

[English](README.md) | 简体中文

一个 ZCode 插件：把长视频与番剧逆向还原为标准亚洲场号制影视剧本。确定性 Python 阶段负责一切可测量的计算（切镜、时间码、台词对齐）；感知型任务统一交给通用多模态大模型——场景理解（地点、时辰、出场人物、场面调度）由 agent 本身逐场完成，说话人归属由 **Qwen3.8-Omni 按音色声学聚类**（由流水线自身直连 DashScope 完成——`speaker_diarize.py run`；MCP 工具 `omni_multi_speaker_asr` 作回退；未配置时优雅降级为全部未归属）完成。台词文本永远只来自字幕、经 `[[SUB:n]]` 占位符逐字拼装——声学与 OCR 都不碰台词文本本身。

## ⚠️ 使用前必须配置阿里云 API Key

**使用本插件前，请先配置阿里云 DashScope API Key。**

1. 在[阿里云百炼控制台](https://bailian.console.aliyun.com/)的 API-KEY 管理页创建密钥；
2. 导出环境变量：`export DASHSCOPE_API_KEY="sk-..."`；
3. （可选）指向其他 OpenAI 兼容端点：`export DASHSCOPE_BASE_URL="https://..."`。

该 Key 用于 Qwen3.8-Omni 声学说话人分离。阶段 1 会做前置检查：未配置时流水线会显式询问——配置 Key 后重跑，或明确选择"说话人列留空"继续（`speaker_diarize.py run` 以退出码 8 二次拦截）。

## 工作原理

1. **纯净三层工作区** —— `materials/`（只读输入）→ `.cache/`（可随时清空的中间产物）→ `output/`（只放最终剧本）。
2. **字幕门禁** —— 外挂字幕 → 容器内封软字幕 → OCR 档。只有 OCR 能服务时，提取器把 Tier 3 指令打到 stdout 并以**退出码 6**（等待感知阶段）结束——这是**出路，不是拒止**；只有在用户确认素材真的无对白之后，才由 `workspace.py check-subtitles --mode none` 以退出码 5 硬性拒止。绝不凭空捏造台词。
3. **音画双轨并行** —— FFmpeg 切镜与关键帧提取，和字幕提取、**声学说话人分离**（ffmpeg 抽 16k 单声道音频 → 直连客户端流式调用同一个 Qwen3.8-Omni 分离请求——代码化重试/退避、逐片断点续跑、无客户端工具窗口限制；MCP `omni_multi_speaker_asr` 保留为回退 → 按声学簇**累计重叠**绑定到字幕行，故一条台词上覆盖 ≥40% 的第二声源会记入 `secondary_speaker`）并发执行。多分片时切点吸附静音、相邻片重叠 ±3 秒，跨片身份只认"重叠区同段语音"的共现证据（union-find 对齐，宁拆不并）；**归属 100% 归声学**，字幕元数据只给声学簇投票起名。模型报告某片「无人声」时该片记为已完成，重跑只补缺、不会为同一片段重复计费。**归属 100% 归声学**；没有 API key（也没有 MCP 回退）时说话人全 null 降级，流水线不阻塞，且降级产物会自报底细（`acoustic_clustering_enabled: false`、`backend: "none"`）。
4. **LGSS 思想场景聚类** —— 一维 DP 求解器把 200+ 物理切镜折叠为宏场景。对白穿越的剪切点施加**大额软惩罚**（绝非硬禁止——求解器不可能死锁成"整集一场"，任何被迫切口都会显式上报）。安装 numpy/opencv 后，启用关键帧 HSV 调色板距离（LGSS "place" 模态的轻量代理，理念源自 LGSS, CVPR 2020）锐化边界；未安装则优雅降级为静默/时长启发式。
5. **毫秒级对齐** —— 每条台词按最大时间重叠唯一归属到一个镜头；达不到重叠门槛的台词不会被丢弃，而是绑定到时间上最近的镜头并上报（对齐器因此绝不可能把「没人被要求放置」的字幕丢给拼装器），并标注 `ON_SCREEN` / `OFF_SCREEN` / `VOICE_OVER` / `INTERNAL_MONOLOGUE`。
6. **叙事大纲（可选，McKee 序列层）** —— `narrative_outline.py` 生成台词流工作单；agent 按「价值转折」判定写出序列骨架（题目 + 价值 from→to + 起止台词），grouper 随即切换为层级模式：序列墙在 **15 秒窗口内**吸附最近物理切点（窗口内无切点时落到其之前最近的切点，并以 `within_snap_window: false` 上报）、逐序列独立求解场景。无大纲则整体平切，行为不变。
7. **场景理解与写作（分块、可断点续跑）** —— `build_scene_manifest.py` 生成逐场**证据包**（640px 关键帧缩略图、逐字台词与时间码、bible 人物名单、前集剧本范例、所属序列的价值弧）并内嵌完整写作契约；多模态 agent 逐场读图，撰写真正的场号制剧本文本 `scene_drafts/scene_XX.md`——`△` 动作段与台词自然交织，台词一律以 `[[SUB:n]]` 占位符表示，**绝不复打**。
8. **逐字拼装成片** —— `splice_screenplay.py` 把每个 `[[SUB:n]]` 替换为逐字字幕原文——台词保真是**结构保证**的，改写在结构上不可能；缺失/重复/错位的占位符一律致命报错并点名下标。成稿含元数据头、场次总表（含序列列）、拼装正文与声画统计/保真度附录，输出 `output/<标题>_影视文学剧本.md`。

## 环境要求

- Python 3.10+
- PATH 上有 FFmpeg / ffprobe（`brew install ffmpeg`）
- 推荐：环境变量 `DASHSCOPE_API_KEY`——启用直连 API 的 Qwen3.8-Omni 声学说话人分离（`speaker_diarize.py run`：无客户端工具窗口限制、指数退避重试、逐片断点续跑）。回退：[Qwen-MM-Plugins](https://github.com/QwenLM/Qwen-MM-Plugins) 的 `api` 插件（MCP）。两者皆无时说话人列留空，流水线照常运行
- 可选：`pip install -r requirements.txt`（numpy + opencv-python-headless）——启用场景聚类的视觉 place 亲和度。其余全部纯标准库。

## 使用

在 ZCode 内运行技能：

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
