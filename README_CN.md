# 影视逆向剧本流水线插件 (`video-to-screenplay`)

[English](README.md) | 简体中文

一个 ZCode 插件：把长视频与番剧逆向还原为标准亚洲场号制影视剧本。确定性 Python 阶段负责一切可测量的计算（切镜、时间码、台词对齐）；场景理解（地点、时辰、出场人物、场面调度）由**唯一一个通用多模态大模型**（即 agent 本身）逐场完成。无专用模型、无 torch、无声学依赖。

## 工作原理

1. **纯净三层工作区** —— `materials/`（只读输入）→ `.cache/`（可随时清空的中间产物）→ `output/`（只放最终剧本）。
2. **字幕门禁** —— 外挂字幕 → 容器内封软字幕 → OCR 网关；纯画面无字幕一律硬性拒止（退出码 5），绝不凭空捏造台词。
3. **音画双轨并行** —— FFmpeg 切镜与关键帧提取，和字幕提取、纯文本说话人标注（ASS Actor/Name 字段、【角色】/角色：前缀、`-` 破折号稳定 A/B 交替）并发执行。
4. **LGSS 思想场景聚类** —— 一维 DP 求解器把 200+ 物理切镜折叠为宏场景。对白穿越的剪切点施加**大额软惩罚**（绝非硬禁止——求解器不可能死锁成"整集一场"，任何被迫切口都会显式上报）。安装 numpy/opencv 后，启用关键帧 HSV 调色板距离（LGSS "place" 模态的轻量代理，理念源自 LGSS, CVPR 2020）锐化边界；未安装则优雅降级为静默/时长启发式。
5. **毫秒级对齐** —— 每条台词按最大时间重叠唯一归属到一个镜头，并标注 `ON_SCREEN` / `OFF_SCREEN` / `VOICE_OVER` / `INTERNAL_MONOLOGUE`。
6. **叙事大纲（可选，McKee 序列层）** —— `narrative_outline.py` 生成台词流工作单；agent 按「价值转折」判定写出序列骨架（题目 + 价值 from→to + 起止台词），grouper 随即切换为层级模式：序列墙吸附最近物理切点、逐序列独立求解场景。无大纲则整体平切，行为不变。
7. **场景理解与写作（分块、可断点续跑）** —— `build_scene_manifest.py` 生成逐场**证据包**（640px 关键帧缩略图、逐字台词与时间码、bible 人物名单、前集剧本范例、所属序列的价值弧）并内嵌完整写作契约；多模态 agent 逐场读图，撰写真正的场号制剧本文本 `scene_drafts/scene_XX.md`——`△` 动作段与台词自然交织，台词一律以 `[[SUB:n]]` 占位符表示，**绝不复打**。
8. **逐字拼装成片** —— `splice_screenplay.py` 把每个 `[[SUB:n]]` 替换为逐字字幕原文——台词保真是**结构保证**的，改写在结构上不可能；缺失/重复/错位的占位符一律致命报错并点名下标。成稿含元数据头、场次总表（含序列列）、拼装正文与声画统计/保真度附录，输出 `output/<标题>_影视文学剧本.md`。

## 环境要求

- Python 3.10+
- PATH 上有 FFmpeg / ffprobe（`brew install ffmpeg`）
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
python3 scripts/speaker_diarize.py --workspace "<ws>" > "<ws>/.cache/audio/speakers.json"
python3 scripts/semantic_scene_grouper.py --workspace "<ws>" > "<ws>/.cache/visual/scenes.json"
python3 scripts/narrative_outline.py --workspace "<ws>"   # 可选：agent 依工作单写叙事大纲后重跑校验
python3 scripts/align_timeline.py --workspace "<ws>" > "<ws>/.cache/alignment/aligned_timeline.json"
python3 scripts/build_scene_manifest.py --workspace "<ws>"
# agent 依据清单逐场撰写 .cache/scene_drafts/scene_XX.md（台词用 [[SUB:n]] 占位）
python3 scripts/splice_screenplay.py --workspace "<ws>" --title "<标题>"
```

## 开发

```bash
python3 -m pytest tests/ -q
```

完整 agent 工作流、场景写作契约与故障矩阵见 [SKILL.md](skills/video-to-screenplay/SKILL.md)。

## 许可证

MIT，见 [LICENSE](LICENSE)。
