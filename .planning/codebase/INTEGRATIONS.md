# 外部集成 (Integrations)

**分析日期:** 2026-07-29

## API / 服务集成

### Hugging Face Hub

| 属性 | 详情 |
|------|------|
| 集成 | Hugging Face Hub |
| 方向 | 入站 (下载模型) / 出站 (上传模型) |
| 数据流 | 模型权重 (safetensors 格式) + config.json |
| 用途 | 预训练模型分发与下载，通过 `PyTorchModelHubMixin` 实现 |
| SDK/Client | `huggingface_hub==0.33.1` |
| 认证 | 无需认证（公开模型），上传需 HuggingFace token |
| 关键函数 | `KronosTokenizer.from_pretrained()`, `Kronos.from_pretrained()`, `model.save_pretrained()` |
| 证据文件 | `model/kronos.py` (第4行, 第13行, 第180行), `webui/app.py` (第645-646行) |

**模型仓库:**
- `NeoQuasar/Kronos-Tokenizer-2k` — Kronos-mini 专用 tokenizer
- `NeoQuasar/Kronos-Tokenizer-base` — Kronos-small/base 通用 tokenizer
- `NeoQuasar/Kronos-mini` — 4.1M 参数轻量模型
- `NeoQuasar/Kronos-small` — 24.7M 参数小模型
- `NeoQuasar/Kronos-base` — 102.3M 参数基础模型

**具体用法:**
```python
# 加载 tokenizer
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
# 加载模型
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
# 保存微调后模型
model.save_pretrained(save_path)
```

### 东方财富网 API

| 属性 | 详情 |
|------|------|
| 集成 | 东方财富网 K线数据 API |
| 方向 | 入站 (获取数据) |
| 数据流 | JSONP 格式的 K线历史数据 |
| 用途 | 通过 HTTP 获取 A 股 K线数据（日/周/月等周期） |
| SDK/Client | `requests` (直接 HTTP 调用) |
| 认证 | 无需认证 |
| API 端点 | `http://push2his.eastmoney.com/api/qt/stock/kline/get` |
| 证据文件 | `examples/get_akshare_date_2024-2025_x.py` (第44-70行) |

**请求参数:**
- `secid`: 市场代码.股票代码 (如 `0.002354` 深交所, `1.600580` 上交所)
- `klt`: K线周期 (101=日线)
- `fqt`: 复权方式 (1=前复权)
- `beg`/`end`: 日期范围

**响应处理:**
- 响应为 JSONP 格式，需手动提取 JSON 数据
- K线数据以逗号分隔字符串形式返回

### Qlib 量化投资平台

| 属性 | 详情 |
|------|------|
| 集成 | Microsoft Qlib |
| 方向 | 入站 (获取数据) + 出站 (回测) |
| 数据流 | A股 K线数据加载 + 回测执行 |
| 用途 | A股数据获取、预处理、回测策略执行 |
| SDK/Client | `pyqlib` (pip install pyqlib) |
| 认证 | 无需认证，本地数据 |
| 证据文件 | `finetune/qlib_data_preprocess.py` (第6-8行), `finetune/qlib_test.py` (第14-20行) |

**核心功能:**
- `qlib.init()`: 初始化 Qlib 环境，指定数据路径
- `D.calendar()`: 获取交易日历
- `QlibDataLoader`: 批量加载股票数据
- `TopkDropoutStrategy`: Top-K 丢弃策略
- `backtest()`: 回测执行引擎
- `risk_analysis()`: 风险分析报告

**配置:**
```python
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
```
证据文件: `finetune/config.py` (第13行), `finetune/qlib_data_preprocess.py` (第28行)

### Comet.ml 实验追踪

| 属性 | 详情 |
|------|------|
| 集成 | Comet.ml |
| 方向 | 出站 (上传指标) |
| 数据流 | 训练损失、学习率、验证损失等指标 |
| 用途 | 训练过程可视化与实验管理 |
| SDK/Client | `comet_ml` |
| 认证 | API Key (通过 `config.comet_config['api_key']` 设置) |
| 证据文件 | `finetune/train_tokenizer.py` (第14行, 第237-246行), `finetune/config.py` (第75-83行) |

**使用方式:**
- 可选集成，通过 `config.use_comet = True/False` 控制
- 记录指标: `train_tokenizer_loss_batch`, `val_tokenizer_loss_epoch`, `train_predictor_loss_batch` 等
- 模型保存: `logger.log_model("best_model", save_path)`

**注意事项:**
- API Key 建议从环境变量读取: `os.getenv("COMET_API_KEY")`
- 默认配置中 API Key 为占位符 `"YOUR_COMET_API_KEY"`

## 数据源集成

### AKShare 金融数据接口

| 数据源 | 类型 | 数据格式 | 用途 | 证据文件 |
|--------|------|----------|------|---------|
| AKShare | Python API | pandas DataFrame | A股历史行情获取、指数数据获取、个股信息查询 | `examples/prediction_cn_markets_day.py` (第28行), `examples/prediction_new.py` (第13行) |

**核心函数:**
- `ak.stock_zh_a_hist(symbol, period, adjust)`: 获取 A股历史行情
- `ak.stock_zh_index_hist(symbol, period)`: 获取指数历史行情
- `ak.stock_individual_info_em(symbol)`: 获取个股基本信息

**数据字段映射:**
- `日期` -> `timestamps`
- `开盘` / `收盘` / `最高` / `最低` -> `open` / `close` / `high` / `low`
- `成交量` / `成交额` -> `volume` / `amount`

### BaoStock 证券数据接口

| 数据源 | 类型 | 数据格式 | 用途 | 证据文件 |
|--------|------|----------|------|---------|
| BaoStock | Python API | 列表数据 | A股历史K线数据（备用数据源） | `examples/get_akshare_date_2024-2025_x.py` (第261行) |

**核心函数:**
- `bs.login()` / `bs.logout()`: 登录/登出
- `bs.query_history_k_data_plus()`: 查询历史K线数据
- 市场代码: `sz.股票代码` (深交所), `sh.股票代码` (上交所)

### CSV 文件数据

| 数据源 | 类型 | 数据格式 | 用途 | 证据文件 |
|--------|------|----------|------|---------|
| CSV 文件 | 本地文件 | CSV (含 timestamps, open, high, low, close, volume, amount 列) | 自定义K线数据加载 | `examples/prediction_example.py` (第49行), `finetune_csv/finetune_base_model.py` (第53行) |

**必需列:**
- `timestamps`: 时间戳 (datetime)
- `open`, `high`, `low`, `close`: OHLC 价格
- `volume`, `amount`: 成交量与成交额 (可选)

### Qlib 本地数据

| 数据源 | 类型 | 数据格式 | 用途 | 证据文件 |
|--------|------|----------|------|---------|
| Qlib 数据目录 | 本地文件 | Qlib 二进制格式 | A股历史行情数据 | `finetune/qlib_data_preprocess.py` (第28行) |

**数据路径:**
- 默认: `~/.qlib/qlib_data/cn_data`
- 通过 `config.qlib_data_path` 配置

## 第三方库集成

| 库名 | 用途 | 版本 | 关键功能 | 证据文件 |
|------|------|----------|---------|---------|
| einops | 张量维度变换 | 0.8.1 | `rearrange()`, `reduce()` — 用于 BSQuantizer 中的分组量化操作 | `model/module.py` (第3行, 第135行, 第151行, 第176行, 第193行) |
| PyTorch | 深度学习核心 | >=2.0.0 | `nn.Module`, `F.scaled_dot_product_attention`, `DistributedDataParallel`, `torch.distributed` | `model/kronos.py` (第3行), `model/module.py` (第4行) |
| huggingface_hub | 模型分发 | 0.33.1 | `PyTorchModelHubMixin` — 自动处理模型下载、上传、缓存 | `model/kronos.py` (第4行, 第13行, 第180行) |
| safetensors | 模型存储 | 0.6.2 | 安全序列化格式，用于 `save_pretrained()` | `requirements.txt` (第10行) |
| matplotlib | 静态图表 | 3.9.3 | 预测结果折线图、回测曲线、K线图 | `examples/prediction_example.py` (第2行), `finetune/qlib_test.py` (第12行) |
| Plotly | 交互式图表 | N/A | Web UI 中的K线蜡烛图 (Candlestick) | `webui/app.py` (第5-6行, 第227行) |
| Flask | Web 服务 | N/A | HTTP API 端点，模型加载/预测接口 | `webui/app.py` (第7行, 第24行) |
| Flask-CORS | 跨域支持 | N/A | 允许跨域请求 | `webui/app.py` (第8行, 第25行) |
| tqdm | 进度条 | 4.67.1 | 自回归推理进度显示、训练循环进度 | `model/kronos.py` (第7行, 第416-419行) |
| PyYAML | YAML 解析 | N/A | 微调配置文件加载 | `finetune_csv/config_loader.py` (第2行, 第19行) |
| pytest | 测试框架 | N/A | 回归测试、MSE 测试 | `tests/test_kronos_regression.py` (第6行) |
| requests | HTTP 客户端 | N/A | 东方财富网 API 调用 | `examples/get_akshare_date_2024-2025_x.py` (第3行, 第69行) |
| tkinter | 桌面 GUI | (Python内置) | 股票预测桌面应用 | `examples/prediction_new_GUI.py` (第14行, 第35行) |

## 可选/条件集成

| 名称 | 条件 | 用途 | 证据文件 |
|------|------|------|---------|
| Comet.ml | `config.use_comet = True` | 训练实验追踪与可视化 | `finetune/config.py` (第75行), `finetune/train_tokenizer.py` (第237行) |
| Qlib | 安装 `pyqlib` | A股数据获取与回测 | `README.md` (第236行), `finetune/qlib_data_preprocess.py` (第6行) |
| AKShare | 安装 `akshare` | A股实时行情数据获取 | `examples/prediction_cn_markets_day.py` (第28行) |
| BaoStock | 安装 `baostock` | A股历史K线数据（备用） | `examples/get_akshare_date_2024-2025_x.py` (第261行) |
| Flask + Plotly | 安装 `flask`, `flask_cors`, `plotly` | Web UI 预测界面 | `webui/run.py` (第12-16行) |
| tkinter | Python 内置 | 桌面 GUI 预测应用 | `examples/prediction_new_GUI.py` (第14行) |
| CUDA GPU | `torch.cuda.is_available()` | GPU 加速推理与训练 | `model/kronos.py` (第496-497行) |
| Apple MPS | `torch.backends.mps.is_available()` | Apple Silicon GPU 推理 | `model/kronos.py` (第498-499行) |
| DDP 多 GPU | `torchrun` 启动 | 分布式训练 | `finetune/train_tokenizer.py` (第276行注释), `finetune_csv/train_sequential.py` (第41行) |

## 数据流集成图

### 推理流程

```
数据源 (CSV / AKShare / 东方财富)
    │
    ▼
pandas DataFrame (OHLCV)
    │
    ▼
KronosPredictor.predict()
    ├── calc_time_stamps() → 时间特征
    ├── 归一化 (mean/std)
    ├── KronosTokenizer.encode() → 量化 token
    ├── auto_regressive_inference() → 自回归推理
    │   ├── Kronos.decode_s1() → s1 logits
    │   ├── sample_from_logits() → s1 token
    │   ├── Kronos.decode_s2() → s2 logits
    │   ├── sample_from_logits() → s2 token
    │   └── KronosTokenizer.decode() → 重建数据
    ├── 反归一化
    └── 输出 DataFrame
```

### 训练流程 (Qlib 微调)

```
Qlib 本地数据
    │
    ▼
qlib_data_preprocess.py → pickle 文件
    │
    ▼
QlibDataset (PyTorch Dataset)
    │
    ▼
DataLoader + DistributedSampler
    │
    ▼
train_tokenizer.py (KronosTokenizer 训练)
    │
    ▼
train_predictor.py (Kronos 训练)
    │
    ▼
qlib_test.py (推理 + 回测)
    ├── QlibBacktest.run_single_backtest()
    └── TopkDropoutStrategy → risk_analysis()
```

### 训练流程 (CSV 微调)

```
CSV 文件
    │
    ▼
config_loader.py (YAML 配置加载)
    │
    ▼
CustomKlineDataset (PyTorch Dataset)
    │
    ▼
train_sequential.py (顺序训练)
    ├── finetune_tokenizer.py (Tokenizer 训练)
    └── finetune_base_model.py (Predictor 训练)
```

## Web UI API 端点

| 端点 | 方法 | 用途 | 证据文件 |
|------|------|------|---------|
| `/` | GET | 渲染主页 | `webui/app.py` (第330-332行) |
| `/api/data-files` | GET | 获取可用数据文件列表 | `webui/app.py` (第335-338行) |
| `/api/load-data` | POST | 加载指定数据文件 | `webui/app.py` (第341-402行) |
| `/api/predict` | POST | 执行预测 | `webui/app.py` (第404-624行) |
| `/api/load-model` | POST | 加载 Kronos 模型 | `webui/app.py` (第626-663行) |
| `/api/available-models` | GET | 获取可用模型列表 | `webui/app.py` (第665-671行) |
| `/api/model-status` | GET | 获取模型状态 | `webui/app.py` (第673-698行) |

**Web UI 运行:** `python webui/run.py` → 启动 Flask 开发服务器 (端口 7070)

---

*集成审计: 2026-07-29*
