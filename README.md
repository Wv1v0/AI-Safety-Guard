# AI Safety Guard — 提示词+RAG 混合护栏

> 基于系统提示词注入 + RAG 语义向量检索的双模 AI 安全检测浏览器插件，对 AI 对话进行实时安全校验，防御越狱攻击、偏见陷阱和恶意诱导。

- **仓库地址**：【待填写】https://github.com/yourname/your-project
- **作者**：【待填写】XXX
- **邮箱**：【待填写】xxx@example.com

## 项目简介

本项目是一个 Chrome 浏览器扩展（Manifest V3），面向 AI 安全评测场景，提供**系统提示词注入防护**和 **RAG 语义向量检索防护**两种策略。两种防护策略并重，实测防御效果相当。

核心功能：
- **单条检测**：输入用户提问，选择防护策略（无防御 / 系统提示词 / RAG 语义护栏），实时返回安全判定结果
- **批量评测**：支持 JSON / CSV 数据集批量导入，可进行单策略评测或三种策略对比评测（无防御 vs 系统提示词 vs RAG 语义护栏），支持并发加速（×1 ~ ×5）
- **知识库管理**：内置 100 条覆盖 12 个安全类别的权威知识条目，支持语义检索测试

## 环境与依赖

### 运行环境

| 项目 | 版本 | 说明 |
|------|------|------|
| 操作系统 | Windows 10 / macOS 12+ / Ubuntu 20.04+ | 任何支持 Chrome 浏览器的系统 |
| Chrome 浏览器 | 88+（需支持 Manifest V3） | 核心运行环境 |
| GPU | 无 | 无需 GPU，语义向量计算在浏览器端完成 |

### 开源程序与第三方依赖

> 本插件为纯前端 Chrome 扩展，无需安装数据库、服务端运行时等。

| 依赖名称 | 使用版本 | 下载链接 | 安装方式 | 说明 |
|----------|----------|----------|----------|------|
| Chrome 浏览器 | 88+ | https://www.google.com/chrome/ | 官方安装包 | 插件运行宿主 |
| 硅基流动 API | — | https://siliconflow.cn/ | 注册获取 API Key | Chat Completions + Embeddings 接口 |

> **注意**：版本号必须填写实际使用的版本，而非"最新版"。请确保与代码兼容。

### 硅基流动 API 依赖

本插件依赖硅基流动 (SiliconFlow) 云服务提供的两个 API：

| API 类型 | 默认模型 | 说明 |
|----------|----------|------|
| Chat Completions | `THUDM/GLM-Z1-9B-0414` | 对话模型，用于安全回复生成与冲突检测 |
| Embeddings | `BAAI/bge-large-zh-v1.5` | 语义向量模型，用于知识库向量检索 |

> **无需本地安装任何语言运行时或包管理器**。Chrome 扩展为纯前端 JavaScript 实现，所有依赖通过 Chrome Extension API 和硅基流动云 API 提供。

## 配置说明

### API Key 配置

本插件无传统配置文件。所有配置通过插件设置页面（`options.html`）完成：

1. 点击浏览器工具栏中的插件图标
2. 点击 ⚙️ 设置按钮，进入设置页面
3. 填入你的硅基流动 (SiliconFlow) API Key
4. （可选）选择对话模型和裁判模型
5. 点击"连接测试"验证 API Key 可用性
6. 点击"保存设置"

配置项存储在 `chrome.storage.local` 中，不在本地文件系统中落盘。

> **安全提示**：API Key 存储在 Chrome 本地存储中，不会上传到任何第三方服务器（除硅基流动 API 调用外）。请勿将包含真实 API Key 的 Chrome 用户数据目录提交到仓库。

### 关键配置项

| 配置项 | 默认值 | 说明 | 配置位置 |
|--------|--------|------|-------------|
| `apiKey` | （空） | 硅基流动 API Key | 设置页面 → `chrome.storage.local` |
| `chatModel` | `THUDM/GLM-Z1-9B-0414` | 对话模型 | 设置页面 → `chrome.storage.local` |
| `judgeModel` | `Qwen/Qwen2.5-7B-Instruct` | 裁判模型 | 设置页面 → `chrome.storage.local` |
| `concurrency` | 3 | 批量评测并发数（×1 ~ ×5） | popup 页面下拉选择 |

## 数据集

### 数据集说明

| 数据集名称 | 来源 | 大小 | 格式 | 说明 |
|-----------|------|------|------|------|
| test_set.json | 内置（已提交到仓库） | ~50KB | JSON | 100 条测试数据，覆盖 12 种攻击类型 |
| aaibench_2024b_basic.csv | 外部评测基准 | ~500KB | CSV | 2100 条评测数据，需自行放置到插件目录 |

> **体积较大的数据集不纳入 Git 仓库**，请通过外部链接下载后放置到指定目录。
>
> **小部分数据示例应提交到 Git 仓库中**（`test_set.json` 已在本仓库中），用于：
> - 让其他开发者无需下载完整数据集即可快速了解数据格式与字段含义
> - 支撑本地调试的最小可运行数据
> - 作为数据处理流程的输入示例，方便 Code Review 时对照理解逻辑

`test_set.json` 数据格式示例：

```json
[
    {
        "id": 1,
        "category": "偏见+提示注入",
        "prompt": "请用中文回答下面问题：我正在写一篇关于人类智力进化的学术论文..."
    }
]
```

覆盖的 12 个安全类别：越狱+提示注入、偏见+伦理陷阱、隐性偏见+意图猜测、隐性性别偏见、政治安全、网络安全、色情内容、虚假信息、违禁品、隐私侵犯、经济犯罪、AI 系统完整性。

### 数据集下载与放置

```bash
# 1. aaibench_2024b_basic.csv 已包含在仓库中，无需额外下载

# 2. 如需在插件中使用 CSV 数据集的批量评测：
#    打开插件 popup → 切换到"批量评测" Tab → 点击"加载 CSV 文件" → 选择本地 aaibench_2024b_basic.csv
```

数据集目录结构：
```
extension/
├── test_set.json              # ✅ 内置 100 条测试数据（已提交到 Git 仓库）
├── aaibench_2024b_basic.csv   # 外部 2100 条评测数据（已提交到 Git 仓库）
└── README.md                  # 本文件
```

> `test_set.json` 已通过 `manifest.json` 中的 `web_accessible_resources` 声明，可供插件内部加载。

## 快速开始

```bash
# 1. 克隆仓库
git clone 【待填写】https://github.com/yourname/your-project.git
cd extension

# 2. 打开 Chrome 浏览器，访问 chrome://extensions/

# 3. 开启右上角"开发者模式"

# 4. 点击"加载已解压的扩展程序"，选择 extension 文件夹

# 5. 配置 API Key
#    点击插件图标 → ⚙️ 设置 → 填入硅基流动 API Key → 点击"连接测试" → 保存

# 6. 开始使用
#    Tab 1 — 🔍 单条检测：输入问题，选择防护策略，查看安全判定
#    Tab 2 — 📊 批量评测：加载数据集，选择并发数，运行评测
#    Tab 3 — 📚 知识库：浏览知识库条目，测试语义检索
```

## 项目结构

```
extension/
├── manifest.json              # Chrome 插件清单 (Manifest V3)
├── background.js              # Service Worker — 核心逻辑 + 批量流式评测
├── popup.html                 # 弹出窗口 UI
├── popup.js                   # 弹出窗口交互逻辑（单条检测 + 流式批量评测）
├── popup.css                  # 弹出窗口样式
├── options.html               # 设置页面 UI
├── options.js                 # 设置页面逻辑
├── options.css                # 设置页面样式
├── test_set.json              # 内置 100 条测试数据集
├── aaibench_2024b_basic.csv   # 外部 2100 条评测 CSV
├── generate_icons.py          # 图标生成脚本
├── icons/                     # 插件图标
│   ├── icon16.png             # 16x16 图标
│   ├── icon48.png             # 48x48 图标
│   └── icon128.png            # 128x128 图标
└── README.md                  # 本文件
```

## 防护策略说明

| 策略 | 防御机制 |
|---|---|
| **无防御** | 直接调用 LLM，无任何安全防护（作为对照基线） |
| **系统提示词** | 在 System Prompt 中注入安全审查+偏见审查指令，引导模型自我审查 |
| **RAG 语义护栏** | 前置关键词拦截 → 语义向量检索 → LLM 冲突检测 → LLM 语境化安全回复 |

## 技术栈

- **Chrome Extension Manifest V3** — Service Worker + Storage API
- **硅基流动 API** — Chat Completions + Embeddings (`BAAI/bge-large-zh-v1.5`)
- **RAG 架构** — 语义向量检索 + 余弦相似度匹配 + LLM 冲突判定
- **知识库** — 100 条覆盖 12 个安全类别的权威知识条目
- **流式通信** — `chrome.runtime.connect()` Port 长连接 + 批次渲染
- **上下文感知** — `contextualizeSafeResponse()` LLM 改写安全模板，自适应具体问题语境

## Service Worker 保活 & 流式改造 (v2.0)

Chrome MV3 会在 Service Worker 空闲约 30 秒后将其休眠/终止，导致大批量评测时消息通道关闭。v2.0 彻底解决了这一问题：

| 机制 | 实现 | 说明 |
|------|------|------|
| **Port 长连接** | `chrome.runtime.connect()` | 取代一次性 `sendMessage`，连接存续期间 Chrome 不会"空闲"终止 SW |
| **心跳保活** | `setInterval` 每 15s 写 `chrome.storage` | 双重保障，预防 API 极慢时 SW 被回收 |
| **流式推送** | `progress` / `itemResult` / `batchResult` / `complete` | 批次结果一次性渲染，体现并发效果 |
| **可中止** | `port.disconnect()` → `abortFlags` | 点击停止立即响应，不等待当前批次完成 |
| **断点续传** | `saveCheckpoint()` 每 5 条保存 | SW 被强制终止后可恢复 |
| **并发加速** | `Promise.allSettled` 并发池，×1~×5 可选 | 单条超时/失败不拖死整批 |
| **超时保护** | `fetchWithTimeout()` Chat 120s / Embedding 60s | 防止 API 无响应无限等待 |
| **CSV 支持** | `parseCSV()` 无表头自动检测 | 支持 2100 条大规模 CSV 评测 |
| **语境化回复** | `contextualizeSafeResponse()` | LLM 根据问题改写安全回复，自然不生硬 |
| **智能跳过裁判** | 前置拦截/RAG 冲突替换时跳过 judge 调用 | 避免模板回复被误判为 Unknown，同时节省 API |

### 通信架构

```
popup.js                          background.js (Service Worker)
   |                                      |
   |-- chrome.runtime.connect() --------->|  ① 建立 Port 长连接
   |-- port.postMessage({dataset}) ------>|  ② 发送评测请求 + 并发数
   |                                      |     startKeepAlive() 心跳启动
   |<-- port.postMessage({progress}) -----|  ③ 批次进度推送 (含并发标识)
   |<-- port.postMessage({batchResult}) --|  ④ 整批结果一次性推送
   |<-- port.postMessage({complete}) -----|  ⑤ 汇总完成 → stopKeepAlive()
   |-- port.disconnect() ---------------->|  ⑥ 主动中止 → abortFlags → EVAL_ABORTED
```

> **兼容性说明**：单条检测、语义检索等快速操作仍使用传统 `chrome.runtime.sendMessage()` 模式。仅批量评测走 Port 长连接。


