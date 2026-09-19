# Qwen3.8-Omni 与凭据保护声明

本插件会把**你的素材内容上传到第三方 API**。这份文件说清楚：上传什么、发到哪里、
凭据怎么存放、以及哪些约束是被测试钉住的、哪些只是约定。

唯一事实源是 `scripts/config_spec.py`（`ENV_VARS` / `DATA_EGRESS` / `SECRET_PATTERNS`）。
本文件与 `README.md` / `README_CN.md` 的对应小节由 `tests/test_packaging.py` 保证不漂移。

## 1. 这个插件依赖 Qwen3.8-Omni（全模态模型）

两处能力**只有**在 Qwen3.8-Omni 上才成立，它们构成"真正听懂、真正看过"的部分：

| 能力 | 脚本 | 上传的东西 |
| :--- | :--- | :--- |
| 声学说话人分离（归属 100% 归声学） | `speaker_diarize.py run` | 16 kHz 单声道音频，按时间分片 |
| 声画理解（动作/运镜/音效/屏显文字证据） | `av_understand.py run` | 逐场 ≤90 秒视频段（480p/CRF 阶梯，含单声道 16 kHz 音轨），base64 内联 |

其余阶段全部在本地完成：切镜与关键帧（ffmpeg）、字幕解析（SRT/ASS）、场景分组（DP 求解器）、
剧本撰写（由运行本插件的 agent 自己完成）、逐字拼装。

## 2. API Key 的保护（硬约束）

- `DASHSCOPE_API_KEY` **只从进程环境变量读取**，且全仓库只有 `omni_client.resolve_key()` 一处读它
  （`tests/test_packaging.py` 扫描 `scripts/` 的 `os.environ` 使用点来钉住这条）。
- 绝不写入磁盘：不进工作区、不进 `.cache/`、不进 `output/`、不进任何证据文件（`speakers.json`、
  `av_notes.json`、`*_workorder.json` 的 `meta` 字段是白名单形状：backend / model /
  video_encoding / endpoint_host / usage）。
- 绝不进 URL、日志、错误消息。失败信息只携带异常种类 + HTTP 状态码 + 主机名。
- 绝不出现在版本库里：`SECRET_PATTERNS`（`sk-`、`LTAI`、手写 `Bearer` 头）对全部 tracked 文件扫描；
  测试夹具也用拼接构造假 key，以免自身触发。
- 缺失时不静默降级：`run` 以**退出码 8** 拒绝，`workspace.py doctor` 报 `dashscope_api_key: missing`，
  agent 必须先经用户明确选择才继续（说话人列留空）。

## 3. Key 会被发往哪里（可核查）

`doctor` 在报告 key 状态的同时报告**它将去往的主机**与**将使用的模型**，不显示 key 本身：

```json
"diarization": {"dashscope_api_key": "set",
                "dashscope_base_url_host": "dashscope.aliyuncs.com",
                "model": "qwen3.8-omni-flash"}
```

`DASHSCOPE_BASE_URL` 与 `V2S_OMNI_MODEL` 是**决定数据去向**的开关，因此与 key 同等对待：
`validate_endpoint()` 强制 https，并拒绝回环 / 私有 / link-local / 保留地址与非公网主机名
（同时排除云元数据端点）。把 key 指向别的端点前，你是自愿的，且这里看得见。

### 全部可配置变量（与 `scripts/config_spec.py` 一致，测试保证）

| 变量 | 默认值 | 必要性 | 控制什么 |
| :--- | :--- | :--- | :--- |
| `DASHSCOPE_API_KEY` | 未设置 | 直连路径必需 | 两处 Omni pass 的 Bearer 凭据；缺失时 `run` 退出码 8 |
| `DASHSCOPE_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | 可选 | 哪个主机接收 key 与媒体 |
| `V2S_OMNI_MODEL` | `qwen3.8-omni-flash` | 可选 | 哪个全模态模型解读素材 |
| `V2S_OMNI_ATTEMPTS` | `3` | 可选 | 单请求重试预算（仅瞬时错误） |
| `V2S_OMNI_TIMEOUT_SEC` | `1800` | 可选 | 单个流式请求的墙钟上限 |

## 4. 内容与身份的红线

- **台词只来自字幕**。声学转写与声画理解都**不参与**台词文本；正文里每一句台词都是 `[[SUB:n]]`
  逐字替换的结果，复写在结构上不可能。
- AV pass 的 prompt 明令**不转写对白**，其产出是证据层；屏显文字与字幕逐字重复时 merge 会告警
  （硬字幕片源防护）。
- AV pass **不给人物命名**，只给可见称呼（如"红衣女子"）；命名权在字幕元数据投票与写作阶段。
- 临时媒体文件写在 `.cache/` 内并做工作区逃逸校验；每段原始回复落盘即为证据，
  时间轴只在 merge 位移一次。

## 5. 花费

每一次 API 调用都会计量并落在摘要里（`av_understand.py merge` 的 `tokens` 字段、
`speaker_diarize` 的分片元数据）。实测校准与启用前的成本预告见 SKILL 的阶段 2 与阶段 3.7：
**默认先 AskUserQuestion 取得确认，再执行任何多段 API pass。**

## 6. 已钉住 vs 仅约定

| 约束 | 状态 |
| :--- | :--- |
| key 只在 `resolve_key()` 一处读取 | 测试钉住 |
| tracked 文件无 key 形状字符串 | 测试钉住 |
| 端点必须 https + 公网 | 代码 + 测试 |
| `meta` 字段不含凭据 | 测试钉住（键白名单） |
| 文档与 `config_spec` 一致 | 测试钉住 |
| 历史提交里的密钥 | **未**扫描——仓库内的扫描只看 tracked 内容；要覆盖历史需要 CI 里的 gitleaks/trufflehog |
| 上传内容在使用方的留存策略 | 由 DashScope 侧条款决定，本插件不承诺 |
