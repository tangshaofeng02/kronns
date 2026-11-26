# 技术栈 (Stack)

**分析日期:** 2026-07-29

## 语言与运行时

| 语言 | 版本 | 用途 | 证据文件 |
|------|------|------|---------|
| Python | 3.10+ | 全项目主要语言，模型推理、训练、Web UI、数据处理 | `README.md` (第89行) |
| HTML/JavaScript | N/A | Web UI 前端页面 | `webui/templates/index.html` |
| YAML | N/A | 微调配置文件 | `finetune_csv/configs/config_ali09988_candle-5min.yaml` |

## 框架与库

### 核心深度学习

| 名称 | 版本 | 用途 | 证据文件 |
|------|------|------|---------|
| PyTorch | >=2.0.0 | 深度学习框架，所有模型构建、训练与推理 | `requirements.txt` (第3行), `model/kronos.py` (第3行) |
| PyTorch DDP | (PyTorch内置) | 分布式数据并行训练 | `finetune/train_tokenizer.py` (第13行), `finetune/train_predictor.py` (第11行) |
| einops | 0.8.1 | 张量操作库，用于 rearrange/reduce 等维度变换 | `requirements.txt` (第5行), `model/module.py` (第3行) |

### 模型分发与存储

| 名称 | 版本 | 用途 | 证据文件 |
|------|------|------|---------|
| huggingface_hub | 0.33.1 | 模型下载与上传，通过 PyTorchModelHubMixin 实现预训练模型加载 | `requirements.txt` (第6行), `model/kronos.py` (第4行) |
| safetensors | 0.6.2 | 模型权重安全序列化格式 | `requirements.txt` (第10行) |

### 数据处理

| 名称 | 版本 | 用途 | 证据文件 |
|------|------|------|---------|
| numpy | N/A | 数值计算，归一化、统计、数组操作 | `requirements.txt` (第1行), `model/kronos.py` (第1行) |
| pandas | 2.2.2 | 数据表格处理，K线数据读写与时序操作 | `requirements.txt` (第8行), `model/kronos.py` (第2行) |

### 可视化

| 名称 | 版本 | 用途 | 证据文件 |
|------|------|------|---------|
| matplotlib | 3.9.3 | 静态图表绘制，预测结果可视化、回测曲线 | `requirements.txt` (第7行), `examples/prediction_example.py` (第2行) |
| Plotly | N/A | 交互式图表，Web UI 中的K线图 | `webui/app.py` (第5-6行) |

### 进度与日志

| 名称 | 版本 | 用途 | 证据文件 |
|------|------|------|---------|
| tqdm | 4.67.1 | 训练与推理进度条 | `requirements.txt` (第9行), `model/kronos.py` (第7行) |
| comet_ml | N/A | 实验追踪与日志记录（可选） | `finetune/train_tokenizer.py` (第14行), `finetune/config.py` (第75行) |

### Web 服务

| 名称 | 版本 | 用途 | 证据文件 |
|------|------|------|---------|
| Flask | N/A | Web UI 后端 HTTP 服务 | `webui/app.py` (第7行) |
| Flask-CORS | N/A | 跨域资源共享支持 | `webui/app.py` (第8行) |

### 金融数据源

| 名称 | 版本 | 用途 | 证据文件 |
|------|------|------|---------|
| Qlib (pyqlib) | N/A | 微软量化投资平台，用于A股数据获取与回测 | `finetune/qlib_data_preprocess.py` (第6行), `finetune/qlib_test.py` (第14-20行) |
| AKShare | N/A | 中国金融数据接口，用于A股实时行情获取 | `examples/prediction_cn_markets_day.py` (第28行), `examples/prediction_new.py` (第13行) |
| BaoStock | N/A | 证券数据接口（备用数据源） | `examples/get_akshare_date_2024-2025_x.py` (第261行) |

### GUI 框架

| 名称 | 版本 | 用途 | 证据文件 |
|------|------|------|---------|
| tkinter | (Python内置) | 桌面图形界面，股票预测GUI | `examples/prediction_new_GUI.py` (第14行) |

### 配置与序列化

| 名称 | 版本 | 用途 | 证据文件 |
|------|------|------|---------|
| PyYAML | N/A | YAML 配置文件解析 | `finetune_csv/config_loader.py` (第2行) |
| pickle | (Python内置) | 数据集序列化存储 | `finetune/dataset.py` (第1行), `finetune/qlib_test.py` (第5行) |
| json | (Python内置) | 配置与报告序列化 | `finetune/train_tokenizer.py` (第3行), `webui/app.py` (第4行) |

### 测试

| 名称 | 版本 | 用途 | 证据文件 |
|------|------|------|---------|
| pytest | N/A | 测试框架，回归测试 | `tests/test_kronos_regression.py` (第6行) |

### HTTP 数据获取

| 名称 | 版本 | 用途 | 证据文件 |
|------|------|------|---------|
| requests | N/A | HTTP 请求，从东方财富网获取股票数据 | `examples/get_akshare_date_2024-2025_x.py` (第3行) |

## 开发工具链

| 工具 | 版本 | 用途 | 证据文件 |
|------|------|------|---------|
| torchrun | (PyTorch内置) | 多 GPU 分布式训练启动器 | `README.md` (第270-277行), `finetune/train_tokenizer.py` (第276行注释) |
| pip | N/A | Python 包管理器 | `README.md` (第93行) |
| argparse | (Python内置) | 命令行参数解析 | `finetune/qlib_test.py` (第304行), `finetune_csv/train_sequential.py` (第321行) |

## 数据存储

| 类型 | 产品 | 用途 | 证据文件 |
|------|------|------|---------|
| CSV 文件 | 本地文件系统 | K线数据存储与加载 | `examples/prediction_example.py` (第49行), `finetune_csv/finetune_base_model.py` (第53行) |
| Pickle 文件 | 本地文件系统 | 处理后的训练/验证/测试数据集 | `finetune/qlib_data_preprocess.py` (第114-119行) |
| JSON 文件 | 本地文件系统 | 训练摘要、市场分析报告、预测结果 | `finetune/train_tokenizer.py` (第267行), `webui/app.py` (第199行) |
| HuggingFace Hub | 远程模型仓库 | 预训练模型分发与下载 | `model/kronos.py` (第4行), `README.md` (第77-82行) |
| Safetensors | 本地文件系统 | 模型权重安全持久化 | `requirements.txt` (第10行) |

## AI/ML 组件

| 名称 | 类型 | 用途 | 证据文件 |
|------|------|------|---------|
| KronosTokenizer | 自编码器+量化器 | 将连续 OHLCV 数据量化为离散 token，含 BinarySphericalQuantizer | `model/kronos.py` (第13-178行) |
| Kronos | 自回归 Transformer | 基于 token 的自回归预测模型，含 HierarchicalEmbedding + DependencyAwareLayer | `model/kronos.py` (第180-329行) |
| KronosPredictor | 推理封装器 | 端到端预测流水线，含数据预处理、归一化、推理、反归一化 | `model/kronos.py` (第482-661行) |
| BinarySphericalQuantizer | 量化器 | 二元球面量化，将连续表征量化为 {-1, +1} 二进制码 | `model/module.py` (第39-223行) |
| BSQuantizer | 量化器封装 | 二级量化器，支持 s1/s2 分离的半量化 | `model/module.py` (第225-254行) |
| HierarchicalEmbedding | 嵌入层 | 层级 token 嵌入，将 s1/s2 两个子 token 的嵌入融合 | `model/module.py` (第400-443行) |
| DualHead | 输出头 | 双头预测，分别输出 s1 和 s2 的 logits | `model/module.py` (第486-513行) |
| DependencyAwareLayer | 交叉注意力层 | s1 与 s2 之间的依赖建模，通过交叉注意力实现 | `model/module.py` (第446-462行) |
| TemporalEmbedding | 时间嵌入 | 提取分钟/小时/星期/日/月等时间特征 | `model/module.py` (第536-562行) |
| TransformerBlock | Transformer 块 | 自注意力 + FFN，含 RMSNorm 前归一化 | `model/module.py` (第465-483行) |
| RotaryPositionalEmbedding | 位置编码 | 旋转位置编码 (RoPE) | `model/module.py` (第284-312行) |
| RMSNorm | 归一化层 | 均方根归一化 | `model/module.py` (第257-268行) |
| FeedForward | 前馈层 | SwiGLU 激活的前馈网络 | `model/module.py` (第271-281行) |

## 预训练模型

| 模型名称 | Tokenizer | 上下文长度 | 参数量 | HuggingFace ID |
|----------|-----------|-----------|--------|---------------|
| Kronos-mini | Kronos-Tokenizer-2k | 2048 | 4.1M | NeoQuasar/Kronos-mini |
| Kronos-small | Kronos-Tokenizer-base | 512 | 24.7M | NeoQuasar/Kronos-small |
| Kronos-base | Kronos-Tokenizer-base | 512 | 102.3M | NeoQuasar/Kronos-base |
| Kronos-large | Kronos-Tokenizer-base | 512 | 499.2M | 未开源 |

证据文件: `README.md` (第77-82行), `webui/app.py` (第33-58行)

## 配置

**环境:**
- Python 3.10+，通过 pip 安装依赖
- GPU: 支持 CUDA（训练必须）、MPS（推理可选）、CPU（推理可选）
- 依赖声明: `requirements.txt`

**构建:**
- 无独立构建步骤，纯 Python 项目
- 模型通过 `Kronos.from_pretrained()` / `KronosTokenizer.from_pretrained()` 从 HuggingFace Hub 自动下载

**训练配置:**
- Qlib 微调: `finetune/config.py` 中的 `Config` 类
- CSV 微调: `finetune_csv/configs/` 目录下的 YAML 文件
- 通过 `torchrun` 启动多 GPU DDP 训练

## 平台要求

**开发:**
- Python 3.10+
- GPU (推荐 NVIDIA CUDA，训练必须)
- 磁盘空间: 模型权重约 100MB-500MB

**生产:**
- CPU 推理可行但缓慢
- GPU 推荐用于实时预测
- Web UI 运行在 Flask 开发服务器 (端口 7070)

---

*栈分析: 2026-07-29*
