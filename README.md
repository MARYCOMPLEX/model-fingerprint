# model-fingerprint

判断一个 OpenAI 兼容端点**实际**在服务哪个模型，而不是相信它自己报的名字。

```bash
model-fingerprint --url https://api.example.com/v1 --key sk-... --model <model-id>
```

一行命令，任意 `url` / `key` / `model` 组合。

## 为什么需要它

很多中转站和聚合网关会把便宜的模型挂在贵模型的 id 后面卖，模型自己也会
胡说自己是谁（DeepSeek V4 系列经常自称 OpenAI / Anthropic / Gemini）。本工具
不看自述，只看**可测量的行为**：

1. **词表指纹（最强证据）**——在提示词末尾追加一段文本，测量
   `usage.prompt_tokens` 的增量。增量只取决于分词器，与各家的聊天模板、系统
   提示词无关。用官方 tokenizer 逐条比对即可判断词表家族。
2. **模板复现（定位到代）**——同一词表家族内部（V3 / V4 / V4.1 共用一套
   常规文本词表），把官方 `encode_messages` 渲染后的 token 数与服务器返回的
   `prompt_tokens` 精确比对，精确命中即定位到具体代次。
3. **能力探测**——视觉输入、函数调用、约 15 万 token 的 needle-in-haystack，
   用于佐证代次（例如 V4-Flash-0731 是纯文本）。
4. **自述与陷阱题**——只记录、低权重，并在与测量结论冲突时给出警告。

## 安装

```bash
pip install -r requirements.txt        # 或 pip install .
```

离线可用：三套 DeepSeek 词表与 Qwen3 词表已压缩打包在
`model_fingerprint/data/`（约 9 MB），无需联网。

## 用法

```bash
# 完整探测（含 15 万 token 长上下文，最慢也最全）
model-fingerprint --url https://api.cline.bot/api/v1 --key sk-... \
  --model cline-pass/deepseek-v4.1-flash

# 快速模式：词表 + 模板 + 自述，十来次请求
model-fingerprint ... --quick

# 保存报告
model-fingerprint ... --json report.json --markdown report.md

# 只看某个端点有哪些模型 id
model-fingerprint --list-models --url https://api.deepseek.com/v1 --key sk-...

# 用厂商预置地址
model-fingerprint --provider openai --key sk-... --model gpt-5.5
model-fingerprint --provider deepseek --key sk-... --model deepseek-chat
model-fingerprint --url http://localhost:8000/v1 --model my-local-model   # vLLM / LM Studio / Ollama
```

### 环境变量

三个参数都可以用环境变量提供，CI 里只需导出一次：

```bash
export MODEL_FINGERPRINT_URL=https://api.example.com/v1
export MODEL_FINGERPRINT_MODEL=some-model-id
export MODEL_FINGERPRINT_KEY=sk-...
model-fingerprint --quick --json out.json
```

密钥查找顺序：`--key` → `MODEL_FINGERPRINT_KEY` / `OPENAI_API_KEY` /
`CLINE_API_KEY` → 当前目录的 `.model_fingerprint_key` / `.cline_key` /
`.openai_key`。

### 其他参数

| 参数 | 作用 |
|------|------|
| `--only tokenizer [--only structure ...]` | 只跑指定检查组 |
| `--skip endpoint` | 跳过某个检查组 |
| `--no-long-context` | 跳过 15 万 token 的 needle 测试 |
| `--header NAME:VALUE` | 附加 HTTP 头（重复可用），用于需要额外鉴权头的端点 |
| `--timeout 300` / `--retries 6` | 慢端点调大 |
| `--list-families` | 查看支持哪些家族、哪些词表已就绪 |
| `--download-data` | 从 HuggingFace 重新拉取参考词表 |

退出码：`0` 有结论；`1` 运行中出错；`2` 缺少密钥。

## 输出

```
  endpoint   https://api.cline.bot/api/v1
  model id   cline-pass/deepseek-v4.1-flash
  verdict    DeepSeek V4.1 (DeepSeek)
  confidence 100%

  evidence
    + fragment token deltas match the DeepSeek V4.1 vocabulary on 10/10 probes
    + server prompt_tokens reproduce the DeepSeek V4.1 chat template exactly on 15/15 prompts
    + accepts image input and reads a solid colour correctly
    + emits OpenAI-style tool calls

  caveats
    ! gateway escapes fullwidth special characters in user content, ...
    ! self-report 'OpenAI' contradicts the tokenizer/template evidence for DeepSeek; ...
```

`--markdown` 会输出带 delta 矩阵和模板残差表的报告，适合贴进 issue 或 PR。

## 判定逻辑

| 层级 | 证据 | 能回答什么 |
|------|------|-----------|
| 1 | 片段 token 增量（10 条） | 词表属于哪个家族（DeepSeek / Qwen / OpenAI / Llama） |
| 2 | 绝对值模板复现（15 条） | 家族内是哪一代（V3 / V4 / V4.1） |
| 3 | 视觉 / 工具 / 长上下文 | 佐证代次（多模态、上下文长度） |
| 4 | 自述、陷阱题 | 仅记录，出现矛盾时警告 |
| 5 | 网关元数据 | 实际路由到的 provider / backend model |

特殊 token（`<｜System｜>` 等）本可以进一步区分同族词表，但不少网关会把
用户内容里的全角 `｜` 转成 ASCII `|`，此时这一行对所有词表结果都一样。
工具会自动识别这种改写并把整类行排除出评分，转而依赖第 2 层。

## 支持的家族

| key | 家族 | 词表来源 |
|-----|------|---------|
| `deepseek_v41` | DeepSeek V4.1 | 打包 |
| `deepseek_v4` | DeepSeek V4 | 打包 |
| `deepseek_v3` | DeepSeek V3 / R1 | 打包 |
| `qwen3` | Qwen 3 | 打包 |
| `openai` / `openai_o200k` | OpenAI GPT | `tiktoken`（可选依赖） |
| `llama3` | Llama 3 | `--download-data` |
| `kimi_k2` | Kimi K2 | `--download-data` |

新增家族：在 `model_fingerprint/families.py` 的 `FAMILIES` 里加一条，把
`tokenizer.json` 压缩成 `data/<name>.json.gz`（或指定 `tiktoken` 编码名）。
若该家族有官方 `encode_messages`，放到 `data/` 里并在 `template_module` 声明，
就能用上第 2 层精确定位。

## 开发

```bash
python -m pytest tests -q
```

测试不访问网络：客户端行为、信封解析、重试策略、模板产物、判定逻辑均为离线单测。

## 免责

结论基于可测量的 token 行为，属于**行为指纹**，不是对权重文件的哈希校验。
它能可靠区分家族与代次，但不能证明部署者没有做过微调或代理层改写。
置信度低时请结合 `caveats` 一起看。
