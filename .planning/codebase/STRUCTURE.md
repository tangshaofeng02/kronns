# 项目结构 (Structure)

**分析日期:** 2026-07-29

## 目录树

```
kronos/                              # 项目根目录 — Kronos 金融基础模型
├── model/                           # 核心模型代码 (Tokenizer + Predictor + 推理)
│   ├── __init__.py                  # 模型工厂: 导出KronosTokenizer/Kronos/KronosPredictor
│   ├── kronos.py                    # 主要模型定义: KronosTokenizer/Kronos/KronosPredictor + 推理函数
│   └── module.py                    # 模型组件: BSQuantizer/BinarySphericalQuantizer/TransformerBlock/DualHead等
├── finetune/                        # Qlib数据微调 (基于Qlib量化框架)
│   ├── config.py                    # 硬编码配置类 Config (数据/训练/模型路径/回测参数)
│   ├── dataset.py                   # QlibDataset: 从pickle加载金融时间序列滑窗数据
│   ├── train_predictor.py           # Predictor微调训练脚本 (DDP支持, Comet ML日志)
│   ├── train_tokenizer.py           # Tokenizer微调训练脚本 (DDP支持, Comet ML日志)
│   ├── qlib_data_preprocess.py      # Qlib数据预处理器: 加载/处理/划分数据为pickle
│   ├── qlib_test.py                 # 推理+回测: QlibTestDataset + QlibBacktest + generate_predictions
│   └── utils/                       # 训练共享工具
│       ├── __init__.py              # 空文件
│       └── training_utils.py        # DDP设置/清理, 种子, 模型大小计算, 时间格式化
├── finetune_csv/                    # CSV自定义数据微调 (更灵活的YAML配置)
│   ├── README.md                    # 英文使用说明
│   ├── README_CN.md                 # 中文使用说明
│   ├── config_loader.py             # ConfigLoader(YAML解析) + CustomFinetuneConfig(配置类)
│   ├── finetune_base_model.py       # Predictor微调 + CustomKlineDataset + setup_logging
│   ├── finetune_tokenizer.py        # Tokenizer微调 + setup_logging
│   ├── train_sequential.py          # SequentialTrainer: 两阶段顺序训练编排器
│   ├── configs/                     # YAML配置文件
│   │   └── config_ali09988_candle-5min.yaml  # 阿里巴巴港股5分钟K线配置模板
│   ├── data/                        # 示例数据
│   │   └── HK_ali_09988_kline_5min_all.csv   # 阿里巴巴港股5分钟K线历史数据
│   └── examples/                    # 微调结果可视化示例
│       ├── HK_ali_09988_kline_5min_all_historical_*.png  # 训练结果图表(5张)
├── webui/                           # Web界面 (Flask REST API + 前端)
│   ├── app.py                       # Flask应用: 数据加载/模型加载/预测/图表生成/结果保存
│   ├── run.py                       # 启动脚本: 检查依赖/自动安装/启动Flask
│   ├── start.sh                     # Shell启动脚本: 检查Python/依赖/启动
│   ├── requirements.txt             # WebUI依赖: flask/flask-cors/pandas/numpy/plotly/torch/huggingface_hub
│   ├── README.md                    # WebUI说明
│   ├── templates/                   # HTML模板
│   │   └── index.html               # 前端单页面 (K线预测交互界面)
│   └── prediction_results/          # 预测结果存储 (JSON文件)
│       └── prediction_*.json        # 30个历史预测结果JSON文件 (2025-08-26)
├── examples/                        # 使用示例脚本
│   ├── prediction_example.py        # 基础预测示例 (单条序列, HuggingFace Hub模型)
│   ├── prediction_batch_example.py  # 批量预测示例 (多条序列并行)
│   ├── prediction_wo_vol_example.py # 无成交量预测示例 (仅OHLC)
│   ├── prediction_cn_markets_day.py # A股日线预测 (akshare数据, 价格涨跌限制)
│   ├── prediction_akshare_2024-2025.py # 增强版A股预测 (akshare + 市场因素分析)
│   ├── prediction_new.py            # 综合版预测 (多维度市场因素增强)
│   ├── prediction_new_GUI.py        # tkinter GUI版预测系统
│   ├── get_akshare_date_2024-2025_x.py # akshare数据下载脚本
│   ├── get_date_new.py              # 数据获取辅助脚本
│   ├── run_backtest_kronos.py       # 回测脚本 (KronosBacktester类, 交易信号/策略)
│   └── yuce/                        # 预测结果与报告
│       ├── historical_backtest.py   # 历史回测验证脚本
│       ├── *_comprehensive_analysis_report.json  # 综合分析报告 (4个股票)
│       └── *_optimized_prediction.png  # 预测图表 (4个股票)
├── tests/                           # 回归测试
│   ├── test_kronos_regression.py    # pytest回归测试: 精确数值对比 + MSE对比
│   └── data/                        # 测试数据
│       ├── generate_regression_output.py  # 生成回归基准输出脚本
│       ├── regression_input.csv     # 测试输入数据
│       ├── regression_output_256.csv # 256上下文长度基准输出
│       └── regression_output_512.csv # 512上下文长度基准输出
├── figures/                         # 论文/README图片
│   ├── logo.png                     # Kronos logo
│   ├── overview.png                 # 架构概览图
│   ├── prediction_example.png       # 预测示例图
│   └── backtest_result_example.png  # 回测结果示例图
├── requirements.txt                 # 核心依赖: numpy/pandas/torch/einops/huggingface_hub/matplotlib/tqdm/safetensors
├── .gitignore                       # Git忽略规则 (Python缓存/模型文件/数据文件/.env等)
├── LICENSE                          # 许可证文件
└── README.md                        # 项目说明 (模型介绍/模型Zoo/使用指南)
```

## 目录用途说明

| 目录 | 用途 | 关键文件 | 说明 |
|------|------|---------|------|
| `model/` | 核心模型定义与推理 | `kronos.py`, `module.py`, `__init__.py` | 所有模型类和组件的定义，是项目最核心的代码 |
| `finetune/` | Qlib数据微调 | `train_tokenizer.py`, `train_predictor.py`, `dataset.py`, `config.py` | 基于Qlib量化框架的微调流程，依赖Qlib和pickle数据 |
| `finetune/utils/` | 微调训练工具 | `training_utils.py` | DDP设置、种子、模型大小计算等共享工具函数 |
| `finetune_csv/` | CSV自定义数据微调 | `train_sequential.py`, `config_loader.py`, `finetune_tokenizer.py`, `finetune_base_model.py` | 更灵活的YAML配置微调方案，不依赖Qlib |
| `finetune_csv/configs/` | YAML配置模板 | `config_ali09988_candle-5min.yaml` | 微调参数配置文件，可复制修改 |
| `finetune_csv/data/` | 示例CSV数据 | `HK_ali_09988_kline_5min_all.csv` | 阿里巴巴港股5分钟K线数据样例 |
| `finetune_csv/examples/` | 微调结果图表 | `HK_ali_09988_*.png` | 微调后模型预测可视化结果 |
| `webui/` | Web预测界面 | `app.py`, `run.py`, `start.sh` | Flask REST API服务，提供数据加载/预测/可视化 |
| `webui/templates/` | 前端HTML | `index.html` | 单页面前端，K线预测交互界面 |
| `webui/prediction_results/` | 预测结果JSON | `prediction_*.json` | 历史预测结果自动保存目录 |
| `examples/` | 使用示例脚本 | `prediction_example.py`, `prediction_new.py` | 各种场景下的预测示例，从基础到增强版 |
| `examples/yuce/` | 预测结果与报告 | `*_report.json`, `*_prediction.png` | 综合分析报告和优化预测图表 |
| `tests/` | 回归测试 | `test_kronos_regression.py`, `data/` | pytest回归测试确保模型输出一致性 |
| `tests/data/` | 测试数据与基准 | `regression_input.csv`, `regression_output_*.csv` | 固定输入和基准输出用于回归测试 |
| `figures/` | 论文插图 | `overview.png`, `logo.png` | README和论文使用的图片资源 |

## 入口文件

| 文件 | 用途 | 说明 |
|------|------|------|
| `model/__init__.py` | 模型包入口 | 导出 KronosTokenizer, Kronos, KronosPredictor; 提供 `get_model_class()` 工厂函数 |
| `model/kronos.py` | 模型定义入口 | 定义三个核心类: KronosTokenizer(13行), Kronos(180行), KronosPredictor(482行) + 推理函数 |
| `finetune/train_tokenizer.py` | Qlib Tokenizer训练入口 | `torchrun` 启动DDP训练, main()函数读取Config并调用train_model() |
| `finetune/train_predictor.py` | Qlib Predictor训练入口 | `torchrun` 启动DDP训练, 依赖微调后的Tokenizer |
| `finetune/qlib_data_preprocess.py` | Qlib数据预处理入口 | 生成pickle文件供dataset.py加载 |
| `finetune/qlib_test.py` | 推理+回测入口 | 加载微调模型→推理→Qlib回测→保存结果 |
| `finetune_csv/train_sequential.py` | CSV两阶段训练入口 | SequentialTrainer编排Tokenizer→Predictor顺序训练 |
| `finetune_csv/finetune_tokenizer.py` | CSV Tokenizer单独训练入口 | 单GPU或多GPU微调Tokenizer |
| `finetune_csv/finetune_base_model.py` | CSV Predictor单独训练入口 | 包含CustomKlineDataset + train_model函数 |
| `webui/app.py` | Web服务入口 | Flask应用, 端口7070, `/api/load-model`, `/api/predict` 等路由 |
| `webui/run.py` | Web启动脚本 | 检查依赖/自动安装/启动Flask/自动打开浏览器 |
| `examples/prediction_example.py` | 基础预测示例入口 | 最简单的单条预测使用范例 |
| `examples/prediction_cn_markets_day.py` | A股预测入口 | akshare数据 + Kronos预测 + 涨跌限制后处理 |
| `tests/test_kronos_regression.py` | 测试入口 | pytest回归测试, 2组测试(精确对比 + MSE对比) |

## 配置文件

| 文件 | 用途 | 说明 |
|------|------|------|
| `finetune/config.py` | Qlib微调配置 | Python Config类, 硬编码参数, 包含TODO占位符路径 |
| `finetune_csv/configs/config_ali09988_candle-5min.yaml` | CSV微调配置模板 | YAML格式, 含数据/训练/模型路径/实验/设备五大配置块 |
| `finetune_csv/config_loader.py` | YAML配置解析 | ConfigLoader(YAML加载+动态路径解析) + CustomFinetuneConfig(参数提取) |
| `requirements.txt` | 核心依赖 | numpy, pandas, torch>=2.0.0, einops==0.8.1, huggingface_hub==0.33.1, matplotlib==3.9.3, tqdm==4.67.1, safetensors==0.6.2 |
| `webui/requirements.txt` | WebUI依赖 | flask==2.3.3, flask-cors==4.0.0, pandas==2.2.2, numpy>=1.26.0, plotly==5.17.0, torch>=2.1.0, huggingface_hub==0.33.1 |
| `.gitignore` | Git忽略配置 | Python缓存、模型文件(.pth/.pt/.bin)、数据文件(.feather/.parquet/.h5)、.env、IDE文件 |
| `webui/start.sh` | Shell启动配置 | 检查Python/依赖/启动命令 |

**HuggingFace Hub模型配置 (远程):**
- `NeoQuasar/Kronos-Tokenizer-base` — Tokenizer模型 (配合Kronos-small/base)
- `NeoQuasar/Kronos-Tokenizer-2k` — Tokenizer模型 (配合Kronos-mini, 2048上下文)
- `NeoQuasar/Kronos-mini` — 4.1M参数预测模型
- `NeoQuasar/Kronos-small` — 24.7M参数预测模型
- `NeoQuasar/Kronos-base` — 102.3M参数预测模型

## 测试文件

| 文件 | 用途 | 说明 |
|------|------|------|
| `tests/test_kronos_regression.py` | 回归测试 | 2组测试: (1) 不同上下文长度(256/512)的精确输出对比 (2) MSE误差阈值对比 |
| `tests/data/generate_regression_output.py` | 生成基准数据 | 用固定seed+模型生成regression_output_*.csv作为基准 |
| `tests/data/regression_input.csv` | 测试输入数据 | K线数据CSV, 包含timestamps+OHLCV列 |
| `tests/data/regression_output_256.csv` | 256长度基准输出 | 固定模型revision + seed生成的基准预测结果 |
| `tests/data/regression_output_512.csv` | 512长度基准输出 | 固定模型revision + seed生成的基准预测结果 |

**测试参数:**
- 模型: `NeoQuasar/Kronos-small`, revision=`901c26c1332695a2a8f243eb2f37243a37bea320`
- Tokenizer: `NeoQuasar/Kronos-Tokenizer-base`, revision=`0e0117387f39004a9016484a186a908917e22426`
- Seed: 123, Device: CPU
- 精确对比: rtol=1e-5
- MSE对比: 预期值[0.008979, 0.003741], tolerance=0.000001

## 文件依赖关系图

### 核心模型依赖关系

```text
model/__init__.py
    └── model/kronos.py (KronosTokenizer, Kronos, KronosPredictor)
        ├── model/module.py (所有组件类, 通过 "from model.module import *" 导入)
        │   ├── BinarySphericalQuantizer ← DifferentiableEntropyFunction
        │   ├── BSQuantizer ← BinarySphericalQuantizer
        │   ├── RMSNorm
        │   ├── FeedForward (SwiGLU: w1 + w3 → silu → w2)
        │   ├── RotaryPositionalEmbedding
        │   ├── MultiHeadAttentionWithRoPE ← RotaryPositionalEmbedding
        │   ├── MultiHeadCrossAttentionWithRoPE ← RotaryPositionalEmbedding
        │   ├── HierarchicalEmbedding ← nn.Embedding, nn.Linear
        │   ├── DependencyAwareLayer ← MultiHeadCrossAttentionWithRoPE, RMSNorm
        │   ├── TransformerBlock ← MultiHeadAttentionWithRoPE, FeedForward, RMSNorm
        │   ├── DualHead ← nn.Linear, F.cross_entropy
        │   ├── FixedEmbedding ← nn.Embedding
        │   ├── TemporalEmbedding ← FixedEmbedding/nn.Embedding
        │   └── codebook_entropy ← DifferentiableEntropyFunction
        │
        ├── numpy (np.mean, np.clip)
        ├── pandas (pd.DataFrame, pd.Series)
        ├── torch (nn.Module, F.softmax, F.scaled_dot_product_attention)
        ├── einops (rearrange, reduce)
        ├── huggingface_hub (PyTorchModelHubMixin)
        └── tqdm (trange)
```

### 微调依赖关系 (Qlib)

```text
finetune/train_tokenizer.py
    ├── finetune/config.py → Config.__dict__
    ├── finetune/dataset.py → QlibDataset
    │   └── finetune/config.py → Config
    ├── model/kronos.py → KronosTokenizer
    ├── finetune/utils/training_utils.py → setup_ddp, cleanup_ddp, set_seed, get_model_size, format_time
    └── comet_ml → Experiment (可选)

finetune/train_predictor.py
    ├── finetune/config.py → Config.__dict__
    ├── finetune/dataset.py → QlibDataset
    ├── model/kronos.py → KronosTokenizer, Kronos
    ├── finetune/utils/training_utils.py → setup_ddp, cleanup_ddp, set_seed, get_model_size, format_time
    └── comet_ml → Experiment (可选)

finetune/qlib_data_preprocess.py
    ├── finetune/config.py → Config
    ├── qlib (外部库)
    └── tqdm → trange

finetune/qlib_test.py
    ├── finetune/config.py → Config
    ├── model/kronos.py → Kronos, KronosTokenizer, auto_regressive_inference
    ├── qlib (外部库: backtest, executor, strategy, risk_analysis)
    └── argparse
```

### 微调依赖关系 (CSV)

```text
finetune_csv/train_sequential.py
    ├── model/__init__.py → Kronos, KronosTokenizer, KronosPredictor
    ├── finetune_csv/config_loader.py → CustomFinetuneConfig
    ├── finetune_csv/finetune_tokenizer.py → train_tokenizer, set_seed, setup_logging
    ├── finetune_csv/finetune_base_model.py → train_model, create_dataloaders, setup_logging, CustomKlineDataset
    └── argparse

finetune_csv/finetune_tokenizer.py
    ├── model/__init__.py → KronosTokenizer
    ├── finetune_csv/config_loader.py → CustomFinetuneConfig
    └── finetune_csv/finetune_base_model.py → CustomKlineDataset
    └── torch.distributed (可选DDP)

finetune_csv/finetune_base_model.py
    ├── model/__init__.py → Kronos, KronosTokenizer, KronosPredictor
    ├── finetune_csv/config_loader.py → CustomFinetuneConfig
    ├── CustomKlineDataset (内置)
    └── torch.distributed (可选DDP)

finetune_csv/config_loader.py
    └── yaml (PyYAML)
    └── os, typing
```

### WebUI依赖关系

```text
webui/app.py
    ├── model/__init__.py → Kronos, KronosTokenizer, KronosPredictor
    ├── flask → Flask, render_template, request, jsonify
    ├── flask_cors → CORS
    ├── plotly → go.Figure, plotly.utils.PlotlyJSONEncoder
    └── pandas, numpy, json, datetime

webui/run.py
    ├── webui/app.py → app
    ├── subprocess, webbrowser
    └── model/__init__.py (导入检查)

webui/templates/index.html
    └── 通过Flask render_template渲染
    └── 调用 /api/* REST端点
```

### 示例脚本依赖关系

```text
examples/prediction_example.py
    └── model/__init__.py → Kronos, KronosTokenizer, KronosPredictor

examples/prediction_batch_example.py
    └── model/__init__.py → Kronos, KronosTokenizer, KronosPredictor

examples/prediction_wo_vol_example.py
    └── model/__init__.py → Kronos, KronosTokenizer, KronosPredictor

examples/prediction_cn_markets_day.py
    ├── model/__init__.py → Kronos, KronosTokenizer, KronosPredictor
    └── akshare (外部库: stock_zh_a_hist)

examples/prediction_new.py (综合版)
    ├── model/__init__.py → Kronos, KronosTokenizer, KronosPredictor
    ├── akshare (外部库)
    ├── numpy, pandas, matplotlib, json

examples/prediction_akshare_2024-2025.py
    ├── model/__init__.py → Kronos, KronosTokenizer, KronosPredictor
    └── akshare (外部库)

examples/prediction_new_GUI.py (tkinter)
    ├── model/__init__.py → Kronos, KronosTokenizer, KronosPredictor
    ├── tkinter (Python GUI库)
    ├── matplotlib.backends.backend_tkagg

examples/run_backtest_kronos.py
    └── numpy, pandas, matplotlib (独立回测类, 不依赖Kronos模型)

examples/yuce/historical_backtest.py
    └── numpy, pandas, matplotlib (历史回测验证)
```

### 测试依赖关系

```text
tests/test_kronos_regression.py
    ├── model/__init__.py → Kronos, KronosTokenizer, KronosPredictor
    ├── pytest (测试框架)
    ├── numpy.testing → assert_allclose
    └── tests/data/regression_input.csv + regression_output_*.csv

tests/data/generate_regression_output.py
    ├── model/__init__.py → Kronos, KronosTokenizer, KronosPredictor
    └── tests/data/regression_input.csv
```

## 命名约定

### 文件命名

**Python模块:**
- 核心模型: `kronos.py`, `module.py` — 简短小写名称
- 训练脚本: `train_*.py`, `finetune_*.py` — 动词+目标命名
- 数据处理: `*_data_preprocess.py`, `dataset.py` — 功能描述命名
- 配置: `config.py`, `config_loader.py` — 配置相关统一前缀
- 测试: `test_kronos_regression.py` — pytest标准 `test_*` 前缀
- 示例: `prediction_*.py`, `get_*.py`, `run_*.py` — 功能+场景命名

**配置文件:**
- YAML: `config_<描述>_candle-<时间粒度>.yaml` — 如 `config_ali09988_candle-5min.yaml`
- CSV数据: `<市场>_<股票名>_<股票代码>_kline_<时间粒度>_all.csv` — 如 `HK_ali_09988_kline_5min_all.csv`

**测试数据:**
- `regression_input.csv`, `regression_output_<上下文长度>.csv` — 明确的输入/输出命名

### 目录命名

- 核心代码: `model/` — 简短小写
- 微调: `finetune/`, `finetune_csv/` — 功能+数据源区分
- 工具: `utils/` — 标准Python工具目录
- Web界面: `webui/` — 功能描述
- 示例: `examples/` — 标准Python示例目录
- 测试: `tests/` — pytest标准目录
- 图片: `figures/` — 文档图片
- 配置: `configs/` — 配置文件目录
- 数据: `data/` — 数据文件目录

## 新代码放置指南

### 新增功能代码

**新模型组件 (如新的注意力机制):**
- 主要代码: `model/module.py` — 添加新组件类
- 在 `model/kronos.py` 中引用新组件

**新预测方法 (如新的推理策略):**
- 主要代码: `model/kronos.py` — 在 KronosPredictor 类中添加新方法
- 如果是独立推理函数，在 `model/kronos.py` 中添加新函数

**新微调方案 (如新的训练策略):**
- 如果基于Qlib: `finetune/` 目录下新建脚本
- 如果基于CSV: `finetune_csv/` 目录下新建脚本 + `configs/` 下新建YAML
- 共享工具: `finetune/utils/training_utils.py` 或 `finetune_csv/` 内对应文件

**新示例脚本:**
- 位置: `examples/` 目录
- 命名: `prediction_<场景描述>.py`
- 首行必须包含: `sys.path.append("../")` 和 `from model import Kronos, KronosTokenizer, KronosPredictor`

**新测试:**
- 位置: `tests/` 目录
- 命名: `test_<功能描述>.py`
- 测试数据: `tests/data/` 目录

**新Web API端点:**
- 位置: `webui/app.py` — 添加新的 `@app.route`
- 前端更新: `webui/templates/index.html`

### 新组件/模块放置

**新数据集类:**
- Qlib数据集: `finetune/dataset.py` — 添加新类或修改QlibDataset
- CSV数据集: `finetune_csv/finetune_base_model.py` — CustomKlineDataset已在此，可扩展或新建

**新配置方案:**
- YAML配置: `finetune_csv/config_loader.py` — 扩展CustomFinetuneConfig
- 配置模板: `finetune_csv/configs/` — 新建YAML文件

**共享辅助函数:**
- Qlib微调: `finetune/utils/training_utils.py`
- CSV微调: 在对应的 `finetune_csv/finetune_*.py` 中内联定义，或新建 `finetune_csv/utils/` 目录

### 新数据文件放置

**训练数据:**
- Qlib: 通过 `finetune/qlib_data_preprocess.py` 生成，保存到Config.dataset_path指定的目录
- CSV: 直接放置，在YAML配置中指定 `data.data_path`

**示例数据:**
- 位置: `finetune_csv/data/` (CSV微调示例数据)
- 位置: `examples/yuce/` (预测结果数据)
- 位置: `tests/data/` (测试数据)

**注意:** `.gitignore` 排除 `.feather`, `.parquet`, `.h5`, `.hdf5`, `.pth`, `.pt`, `.ckpt`, `.bin` 等大文件

## 特殊目录

**`webui/prediction_results/`:**
- 用途: WebUI预测结果的自动存储目录
- 生成: 是 — 每次WebUI预测自动生成JSON文件
- 提交: 是 — 目录中包含历史预测结果文件
- 命名: `prediction_<时间戳>.json` 格式 (如 `prediction_20250826_163800.json`)

**`finetune_csv/examples/`:**
- 用途: 微调结果可视化示例图片
- 生成: 是 — 微调训练后生成
- 提交: 是 — 作为README展示图片

**`figures/`:**
- 用途: 论文和README使用的图片资源
- 生成: 否 — 手动准备的图片
- 提交: 是 — 项目文档插图

**`tests/data/`:**
- 用途: 回归测试的固定输入和基准输出数据
- 生成: 是 — 通过 `generate_regression_output.py` 生成
- 提交: 是 — 必须提交以确保测试可重复性

**`finetune_csv/data/`:**
- 用途: CSV微调的示例训练数据
- 生成: 否 — 外部获取的数据
- 提交: 是 — 提供了阿里巴巴港股样例数据

---

*结构分析: 2026-07-29*
