<!-- refreshed: 2026-07-29 -->
# 架构 (Architecture)

**分析日期:** 2026-07-29

## 系统概览

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                          应用层 (Application Layer)                       │
├───────────────────────┬──────────────────────┬──────────────────────────┤
│   Web UI (Flask)      │   预测示例脚本         │   回测系统               │
│  `webui/app.py`       │  `examples/`          │  `finetune/qlib_test.py` │
│  REST API 接口         │  单条/批量预测         │  QlibBacktest            │
└───────────┬───────────┴──────────┬───────────┴──────────┬───────────────┘
            │                      │                       │
            ▼                      ▼                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        推理层 (Inference Layer)                           │
│                                                                          │
│  KronosPredictor ──────── auto_regressive_inference ────── sample_from_logits │
│  `model/kronos.py:482`   `model/kronos.py:389`            `model/kronos.py:373` │
│                                                                          │
│  负责: 数据预处理/归一化 → Tokenizer编码 → 自回归解码 → 反归一化 → 输出     │
└────────────────────┬─────────────────────────────┬────────────────────────┘
                     │                             │
                     ▼                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                       模型层 (Model Layer)                                │
│                                                                          │
│  ┌───────────────────────┐     ┌──────────────────────────────────────┐  │
│  │   KronosTokenizer     │     │         Kronos (Predictor)           │  │
│  │  `model/kronos.py:13` │     │        `model/kronos.py:180`        │  │
│  │                       │     │                                      │  │
│  │  Encoder → Quantizer  │     │  HierarchicalEmbedding → Transformer │  │
│  │  → Decoder            │     │  → DependencyAwareLayer → DualHead   │  │
│  └───────────────────────┘     └──────────────────────────────────────┘  │
│                    │                            │                        │
│                    ▼                            ▼                        │
┌──────────────────────────────────────────────────────────────────────────┐
│                     组件层 (Component Layer)                               │
│                                                                          │
│  ┌──────────────┐ ┌──────────────┐ ┌────────────────┐ ┌─────────────┐   │
│  │ BSQuantizer  │ │ TransformerBlock│ │ DualHead       │ │ TemporalEmb │   │
│  │ + BinarySph  │ │ + RoPE Attn   │ │ (s1 + s2 logits)│ │ (时间特征)   │   │
│  │ Quantizer    │ │ + RMSNorm     │ │                │ │             │   │
│  │ + HierarchEmb│ │ + FeedForward │ │                │ │             │   │
│  └──────────────┘ └──────────────┘ ┌────────────────┐ └─────────────┘   │
│                                    │ DependencyAware │                    │
│                                    │ Layer (交叉注意力) │                    │
│                                    └────────────────┘                     │
└──────────────────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                     训练层 (Training Layer)                               │
│                                                                          │
│  ┌─────────────────────────┐  ┌──────────────────────────────────┐      │
│  │  finetune/ (Qlib数据)    │  │  finetune_csv/ (CSV自定义数据)    │      │
│  │  train_tokenizer.py      │  │  train_sequential.py             │      │
│  │  train_predictor.py      │  │  finetune_tokenizer.py           │      │
│  │  qlib_data_preprocess.py │  │  finetune_base_model.py          │      │
│  │  dataset.py (QlibDataset)│  │  config_loader.py (YAML配置)      │      │
│  └─────────────────────────┘  └──────────────────────────────────┘      │
│                                                                          │
│  共享: utils/training_utils.py (DDP设置/种子/模型大小计算)                  │
└──────────────────────────────────────────────────────────────────────────┘
```

## 架构风格

**两阶段自回归基础模型架构 (Two-stage Autoregressive Foundation Model)**

Kronos 采用"Tokenizer + Predictor"两阶段设计，类似于 NLP 中的分词器+语言模型范式：

1. **第一阶段 (Tokenizer)**：将连续的 OHLCV 金融K线数据量化为层级离散token（s1粗粒度 + s2细粒度）
2. **第二阶段 (Predictor)**：自回归 Transformer 语言模型，基于量化token序列进行预测

**核心特征:**
- 层级token化设计 (Hierarchical Tokenization): s1_bits 粗粒度 token + s2_bits 细粒度 token
- Binary Spherical Quantization (BSQ): 将连续特征量化为二进制球面码本
- 依赖感知层 (DependencyAwareLayer): s2 token 通过交叉注意力条件化于 s1 token
- 自回归推理: 逐步生成 s1 → s2 token，支持 top-k/top-p 采样
- 实例级归一化: 每条时间序列独立归一化，避免跨序列信息泄露

## 核心组件

| 组件 | 职责 | 关键文件 | 依赖 |
|------|------|---------|------|
| **KronosTokenizer** | 将OHLCV连续数据量化为离散token，支持编码/解码 | `model/kronos.py:13-178` | BSQuantizer, TransformerBlock |
| **Kronos** | 自回归Transformer预测模型，层级双头输出 | `model/kronos.py:180-329` | HierarchicalEmbedding, DualHead, DependencyAwareLayer |
| **KronosPredictor** | 高层推理封装，处理数据预处理/归一化/预测/反归一化 | `model/kronos.py:482-661` | KronosTokenizer, Kronos |
| **BSQuantizer** | Binary Spherical Quantizer，将连续向量量化为二进制码 | `model/module.py:225-254` | BinarySphericalQuantizer |
| **BinarySphericalQuantizer** | 核心量化算法，实现球面二进制码本 + 熵正则 | `model/module.py:39-223` | DifferentiableEntropyFunction |
| **HierarchicalEmbedding** | 层级token嵌入，将s1/s2 token融合为d_model维向量 | `model/module.py:400-443` | nn.Embedding, nn.Linear |
| **DualHead** | 双头输出层，分别预测s1和s2 token logits | `model/module.py:486-513` | nn.Linear, F.cross_entropy |
| **DependencyAwareLayer** | s2对s1的条件依赖层，使用交叉注意力 | `model/module.py:446-462` | MultiHeadCrossAttentionWithRoPE, RMSNorm |
| **TemporalEmbedding** | 时间特征嵌入（分钟/小时/周/日/月） | `model/module.py:536-563` | FixedEmbedding |
| **TransformerBlock** | 标准Transformer块（自注意力+FFN+RoPE） | `model/module.py:465-483` | MultiHeadAttentionWithRoPE, FeedForward, RMSNorm |
| **CustomKlineDataset** | CSV数据集加载，时间序列滑窗采样 | `finetune_csv/finetune_base_model.py:25-133` | pandas, numpy |
| **QlibDataset** | Qlib金融数据集加载，预计算所有滑窗索引 | `finetune/dataset.py:9-139` | pickle, pandas, Config |
| **CustomFinetuneConfig** | YAML配置加载器，解析微调参数 | `finetune_csv/config_loader.py:109-268` | yaml, ConfigLoader |
| **Config** | Qlib微调配置类，硬编码参数 | `finetune/config.py:1-132` | os |
| **WebUI (Flask)** | Web界面，模型加载/预测/可视化 | `webui/app.py:1-709` | Flask, Plotly, KronosPredictor |

## 模型设计

### KronosTokenizer (`model/kronos.py:13-178`)

```text
输入: x [batch, seq_len, d_in=6]
  │
  ▼
nn.Linear(d_in → d_model)               # embed层
  │
  ▼
TransformerBlock × (n_enc_layers - 1)    # 编码器
  │
  ▼
nn.Linear(d_model → codebook_dim)        # quant_embed层，映射到量化空间
  │
  ▼
BSQuantizer(s1_bits + s2_bits)           # 核心量化: 连续→离散
  │  输出: bsq_loss, quantized, z_indices
  ├── quantized[:, :s1_bits] ──► nn.Linear(s1_bits → d_model) ──► Decoder ──► head ──► z_pre (粗粒度重建)
  └
  └── quantized[:,:]       ──► nn.Linear(codebook_dim → d_model) ──► Decoder ──► head ──► z (完整重建)
```

**关键方法:**
- `encode(x, half=False)`: 将x编码为z_indices（half=True时返回s1/s2分离索引）
- `decode(x, half=False)`: 将z_indices解码回d_in维数据
- `indices_to_bits(x, half=False)`: 将索引转换为二进制码本表示

### Kronos Predictor (`model/kronos.py:180-329`)

```text
输入: s1_ids, s2_ids [batch, seq_len], stamp [batch, seq_len, 5]
  │
  ▼
HierarchicalEmbedding(s1, s2 → d_model)   # 层级嵌入融合
  │ + TemporalEmbedding(stamp)             # 时间特征叠加
  │ + token_dropout                        # token级dropout
  ▼
TransformerBlock × n_layers               # 自注意力Transformer层
  │ (使用RoPE旋转位置编码)
  ▼
RMSNorm                                   # 归一化
  │
  ├──► head(x) ──► s1_logits [batch, seq_len, 2^s1_bits]   # s1 头
  │
  ├──► sample_from_logits(s1_logits) ──► sample_s1_ids     # 从s1采样
  │
  └──► emb_s1(sample_s1_ids) ──► sibling_embed             # s1嵌入作为s2条件
        │
        ▼
  DependencyAwareLayer(context, sibling_embed)              # s2条件化于s1
        │
        ▼
  cond_forward(x2) ──► s2_logits [batch, seq_len, 2^s2_bits]  # s2 头
```

**关键方法:**
- `forward(s1_ids, s2_ids, stamp, padding_mask, use_teacher_forcing, s1_targets)`: 完整前向推理，返回(s1_logits, s2_logits)
- `decode_s1(s1_ids, s2_ids, stamp, padding_mask)`: 仅解码s1，返回(s1_logits, context)
- `decode_s2(context, s1_ids, padding_mask)`: 条件解码s2，返回s2_logits

### 推理流程 (`model/kronos.py:389-469`)

```text
auto_regressive_inference 流程:
  1. 输入数据clip → 重复sample_count次 → tokenizer.encode(half=True)获取s1/s2索引
  2. 维护滚动缓冲区(pre_buffer, post_buffer)，长度max_context
  3. 对每个预测步i:
     a. 构造当前窗口输入token + 时间戳stamp
     b. model.decode_s1 → s1_logits → sample_from_logits → sample_s1_ids
     c. model.decode_s2(context, sample_s1_ids) → s2_logits → sample_from_logits → sample_s2_ids
     d. 更新滚动缓冲区
  4. 解码全部token → tokenizer.decode → reshape(sample_count) → 取均值
  5. 返回preds
```

### KronosPredictor (`model/kronos.py:482-661`)

```text
predict() 流程:
  1. 验证DataFrame列（必须含open/high/low/close，volume/amount可选补零）
  2. 计算时间戳特征 calc_time_stamps (minute/hour/weekday/day/month)
  3. 实例级归一化: x = (x - mean) / (std + 1e-5)，clip到[-5, 5]
  4. 调用 generate() → auto_regressive_inference
  5. 反归一化: preds = preds * (std + 1e-5) + mean
  6. 构造输出DataFrame，列名: open/high/low/close/volume/amount

predict_batch() 流程:
  1. 验证所有输入一致性（相同lookback长度、相同pred_len）
  2. 对每条序列独立归一化
  3. stack为batch → generate()
  4. 对每条序列独立反归一化
```

## 数据流

### 主要预测数据流

1. **用户输入** — pandas DataFrame (OHLCV) + 时间戳 Series (`examples/prediction_example.py`)
2. **KronosPredictor.predict()** — 数据验证 + 归一化 + 时间特征提取 (`model/kronos.py:519-559`)
3. **auto_regressive_inference** — Tokenizer编码 + 自回归逐步解码 (`model/kronos.py:389-469`)
4. **KronosTokenizer.encode** — 连续数据→Encoder→Quantizer→离散索引 (`model/kronos.py:142-159`)
5. **Kronos.decode_s1/decode_s2** — Transformer推理 + 采样 (`model/kronos.py:278-328`)
6. **KronosTokenizer.decode** — 索引→二进制码→Decoder→连续数据 (`model/kronos.py:161-177`)
7. **反归一化** — 均值/标准差恢复 + 输出DataFrame (`model/kronos.py:555-559`)

### 微调训练数据流 (Qlib)

1. **QlibDataPreprocessor** — 从Qlib加载原始数据，生成pickle (`finetune/qlib_data_preprocess.py`)
2. **QlibDataset** — 从pickle加载，滑动窗口采样，实例级归一化 (`finetune/dataset.py`)
3. **train_tokenizer.py** — DataLoader → KronosTokenizer.forward → MSE+BSQ损失 → AdamW优化 (`finetune/train_tokenizer.py`)
4. **train_predictor.py** — DataLoader → tokenizer.encode → Kronos.forward → DualHead.compute_loss (`finetune/train_predictor.py`)

### 微调训练数据流 (CSV自定义)

1. **CustomKlineDataset** — 从CSV加载，时间序列滑窗，按比例划分训练/验证 (`finetune_csv/finetune_base_model.py:25-133`)
2. **CustomFinetuneConfig** — 从YAML配置文件加载参数 (`finetune_csv/config_loader.py:109-268`)
3. **train_sequential.py** — SequentialTrainer编排两阶段训练 (`finetune_csv/train_sequential.py:18-361`)
4. **finetune_tokenizer.py** — 单独Tokenizer微调 (`finetune_csv/finetune_tokenizer.py`)
5. **finetune_base_model.py** — 单独Predictor微调 (`finetune_csv/finetune_base_model.py`)

### WebUI数据流

1. **浏览器请求** — `/api/load-model` 加载模型 (`webui/app.py:626-663`)
2. **Flask路由** — `/api/predict` 执行预测 (`webui/app.py:404-624`)
3. **KronosPredictor.predict** — 核心推理 (`model/kronos.py:519-559`)
4. **Plotly JSON图表** — create_prediction_chart → JSON响应 (`webui/app.py:209-328`)
5. **预测结果保存** — JSON文件存储到prediction_results目录 (`webui/app.py:125-207`)

## 配置架构

### Qlib微调配置 (`finetune/config.py`)

**方式:** Python类 `Config` 硬编码参数，通过 `__dict__` 传递给训练函数

**关键配置项:**
- 数据参数: qlib_data_path, instrument, lookback_window, predict_window
- 训练参数: epochs, batch_size, learning_rate (tokenizer: 2e-4, predictor: 4e-5)
- 模型路径: pretrained_tokenizer_path, pretrained_predictor_path
- 回测参数: backtest_n_symbol_hold, inference_T/top_p/top_k/sample_count

### CSV微调配置 (`finetune_csv/config_loader.py` + `finetune_csv/configs/`)

**方式:** YAML配置文件 → ConfigLoader → CustomFinetuneConfig

**配置层级:**
- `data`: data_path, lookback_window, predict_window, max_context, clip, train/val/test比例
- `training`: tokenizer_epochs, basemodel_epochs, batch_size, learning_rate, adam参数
- `model_paths`: pretrained路径, save路径, exp_name（动态路径模板 `{exp_name}`）
- `experiment`: train_tokenizer/train_basemodel开关, skip_existing, use_comet
- `device`: use_cuda, device_id

### 模型配置 (HuggingFace Hub)

**方式:** 通过 `PyTorchModelHubMixin` 从HuggingFace Hub加载，模型架构参数在 `config.json` 中定义

**模型 Zoo:**
| 模型 | Tokenizer | 上下文长度 | 参数量 | Hub ID |
|------|-----------|-----------|--------|---------|
| Kronos-mini | Kronos-Tokenizer-2k | 2048 | 4.1M | NeoQuasar/Kronos-mini |
| Kronos-small | Kronos-Tokenizer-base | 512 | 24.7M | NeoQuasar/Kronos-small |
| Kronos-base | Kronos-Tokenizer-base | 512 | 102.3M | NeoQuasar/Kronos-base |

## 执行流程

### 预测流程（从入口到输出）

1. **导入模型类** — `from model import Kronos, KronosTokenizer, KronosPredictor` (`model/__init__.py`)
2. **加载模型** — `KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")` / `Kronos.from_pretrained("NeoQuasar/Kronos-small")`
3. **创建Predictor** — `KronosPredictor(model, tokenizer, max_context=512)` (`model/kronos.py:482-506`)
4. **准备数据** — 读取CSV → pandas DataFrame → 提取OHLCV列和时间戳列
5. **调用predict** — `predictor.predict(df, x_timestamp, y_timestamp, pred_len, T, top_p, sample_count)` (`model/kronos.py:519-559`)
6. **auto_regressive_inference** — 编码 → 滑动窗口逐步解码 → 解码回连续数据 (`model/kronos.py:389-469`)
7. **输出** — pred_df: DataFrame (open/high/low/close/volume/amount), 索引为y_timestamp

### Tokenizer微调流程 (Qlib)

1. **数据预处理** — `python finetune/qlib_data_preprocess.py` 生成pickle文件
2. **启动训练** — `torchrun --standalone --nproc_per_node=N finetune/train_tokenizer.py`
3. **DDP初始化** — setup_ddp() → dist.init_process_group(nccl) (`finetune/utils/training_utils.py:9-32`)
4. **模型加载** — KronosTokenizer.from_pretrained → DDP包装 (`finetune/train_tokenizer.py:251-253`)
5. **训练循环** — MSE重建损失 + BSQ量化损失 → AdamW + OneCycleLR → 最佳模型保存 (`finetune/train_tokenizer.py:74-215`)
6. **Comet日志** — 可选的Comet ML实验跟踪 (`finetune/train_tokenizer.py:238-246`)

### Predictor微调流程 (Qlib)

1. **依赖Tokenizer** — 加载微调后的Tokenizer (`finetune/train_predictor.py:213-214`)
2. **启动训练** — `torchrun --standalone --nproc_per_node=N finetune/train_predictor.py`
3. **训练循环** — tokenizer.encode(half=True) → Kronos.forward → DualHead.compute_loss → CE损失 (`finetune/train_predictor.py:60-179`)

### CSV微调流程

1. **编写YAML配置** — `finetune_csv/configs/config_*.yaml` 指定数据路径和参数
2. **顺序训练** — `python finetune_csv/train_sequential.py --config configs/config.yaml`
3. **阶段1: Tokenizer** — `finetune_csv/finetune_tokenizer.py` → MSE + BSQ损失
4. **阶段2: Predictor** — `finetune_csv/finetune_base_model.py` → CE损失 (条件依赖s1→s2)
5. **支持DDP** — `torchrun --standalone --nproc_per_node=N train_sequential.py`

## 模式与原则

### 核心设计模式

**1. 层级token化模式 (Hierarchical Tokenization)**
- 将K线数据量化为两级token: s1(粗粒度，高位bits) + s2(细粒度，低位bits)
- s1捕捉宏观趋势，s2捕捉微观波动
- 实现: `BSQuantizer` (`model/module.py:225-254`) + `HierarchicalEmbedding` (`model/module.py:400-443`)

**2. 条件依赖解码模式 (Conditional Dependency Decoding)**
- s2 token的生成条件化于s1 token
- 通过交叉注意力 (`DependencyAwareLayer`, `model/module.py:446-462`) 实现
- DualHead分别输出s1和s2 logits

**3. 两阶段训练模式 (Two-stage Training)**
- 先训练Tokenizer (重建损失 + 量化损失)
- 再训练Predictor (交叉熵损失，Tokenizer冻结)
- 顺序编排: `SequentialTrainer` (`finetune_csv/train_sequential.py:18-361`)

**4. 自回归推理模式 (Autoregressive Inference)**
- 逐步生成token: 先s1再s2
- 支持top-k/top-p/nucleus采样控制生成多样性
- 滚动缓冲区管理超过max_context的长序列 (`model/kronos.py:408-454`)

**5. 实例级归一化模式 (Instance-level Normalization)**
- 每条时间序列独立计算均值和标准差
- 预测后反归一化恢复原始量级
- 防止跨序列信息泄露 (`model/kronos.py:544-547`, `finetune/dataset.py:109-117`)

**6. 模型工厂模式 (Model Factory)**
- `model/__init__.py` 定义 `model_dict` 和 `get_model_class()`
- 支持通过名称查找模型类
- 兼容HuggingFace Hub的 `PyTorchModelHubMixin`

**7. DDP分布式训练模式 (Distributed Data Parallel)**
- 所有训练脚本支持 `torchrun` 多GPU启动
- `setup_ddp/cleanup_ddp` (`finetune/utils/training_utils.py:9-38`)
- DistributedSampler + all_reduce 验证损失聚合

### 架构约束

- **序列长度限制**: Kronos-small/base最大上下文512, Kronos-mini最大2048
- **特征维度固定**: d_in=6 (open/high/low/close/volume/amount)
- **量化参数固定**: codebook_dim = s1_bits + s2_bits (默认10+10=20)
- **层级token化**: s1_bits和s2_bits决定码本大小 (2^10=1024 vocab per head)
- **无状态推理**: KronosPredictor不维护内部状态，每次predict独立归一化
- **全局模型实例**: WebUI使用全局变量存储tokenizer/model/predictor (`webui/app.py:28-30`)

### 反模式

**1. sys.path.append hack**

- **现象:** 多个文件使用 `sys.path.append("../")` 或 `sys.path.append(os.path.dirname(...))` 来解决导入问题
- **问题:** 依赖运行时目录位置，不同启动方式可能导致导入失败
- **涉及文件:** `model/kronos.py:6`, `finetune/train_tokenizer.py:18`, `finetune/train_predictor.py:15`, `webui/app.py:15`, `examples/*.py`
- **改进方式:** 使用正规的Python包结构（setup.py/pyproject.toml）+ 相对导入

**2. 全局状态管理 (WebUI)**

- **现象:** `webui/app.py` 使用全局变量 `tokenizer`, `model`, `predictor` 存储模型实例
- **问题:** Flask多线程环境下全局状态不安全，多请求并发可能导致竞态条件
- **改进方式:** 使用Flask应用上下文或单独的ModelManager类封装

**3. 硬编码配置 (Qlib Config)**

- **现象:** `finetune/config.py` 的Config类硬编码所有参数，包括TODO占位符路径
- **问题:** 不灵活，修改配置需要编辑源代码
- **改进方式:** 已有YAML配置方案 (`finetune_csv/config_loader.py`)，Qlib微调应也采用YAML

### 错误处理

**策略:** 验证优先 + 异常抛出

**模式:**
- KronosPredictor.predict() 验证输入DataFrame列和NaN值，直接raise ValueError (`model/kronos.py:521-536`)
- WebUI Flask路由用try/except包裹，返回JSON error响应 (`webui/app.py:401-402, 623-624`)
- 训练脚本使用dist.barrier()同步进程，异常时destroy_process_group (`finetune_csv/train_sequential.py:345-357`)
- 数据预处理使用fillna(method='ffill')填充缺失值 (`finetune_csv/finetune_base_model.py:69`)

### 跨领域关注点

**日志:** 
- Qlib微调: Comet ML (可选) + print输出 (`finetune/train_tokenizer.py:238-246`)
- CSV微调: Python logging模块 + RotatingFileHandler (`finetune_csv/finetune_tokenizer.py:49-90`)
- WebUI: print输出 + JSON文件存储 (`webui/app.py:125-207`)

**验证:** 
- 输入数据验证: DataFrame列检查、NaN检查 (`model/kronos.py:521-536`)
- 训练数据验证: 滑窗长度检查 (`finetune/dataset.py:78-80`)
- 回归测试: 固定seed + 精确数值对比 (`tests/test_kronos_regression.py`)

**认证:** 
- HuggingFace Hub模型下载: 无认证（公开模型）
- WebUI: 无认证（本地开发服务器）

---

*架构分析: 2026-07-29*
