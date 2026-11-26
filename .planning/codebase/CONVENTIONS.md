# 编码规范 (Conventions)

**分析日期:** 2026-07-29

## 代码风格

**语言规范:**
- Python 3.10+ (项目 README 明确要求)
- 类型注解使用 Python 3.10+ 语法，如 `tuple[KronosTokenizer, Kronos]` (`finetune/qlib_test.py:207`)、`dict[str, pd.DataFrame]` (`finetune/qlib_test.py:239`)

**格式化工具:**
- 未检测到专门的格式化配置文件 (无 `.prettierrc`、`black` 配置、`pyproject.toml` 等)
- 无 `.editorconfig`
- 项目依赖隐式格式化约定，依赖开发者自行保持一致性

**Linting 工具:**
- 未检测到 `.eslintrc`、`flake8`、`ruff`、`pylint` 配置文件
- 无 CI 中强制 lint 检查
- 代码中存在 `assert` 语句用于参数校验 (`model/module.py:58-62,83,168`)，但这更多是 PyTorch 模型开发惯例而非 linting 规范

**关键风格特征:**
- 行长度不统一：部分文件超 200 行长函数，部分文件短小紧凑
- 混合使用空格缩进（均为 4 空格标准 Python 缩进）
- 字符串使用双引号和单引号混用，无统一规范
- 注释使用英文为主，`examples/run_backtest_kronos.py` 和 `finetune_csv/` 的配置使用中文

## 命名规范

| 类别 | 规范 | 示例 |
|------|------|------|
| **类名** | PascalCase (大驼峰) | `KronosTokenizer` (`model/kronos.py:13`), `BinarySphericalQuantizer` (`model/module.py:39`), `QlibDataset` (`finetune/dataset.py:9`), `SequentialTrainer` (`finetune_csv/train_sequential.py:18`), `KronosBacktester` (`examples/run_backtest_kronos.py:16`) |
| **函数名** | snake_case (小写下划线) | `auto_regressive_inference` (`model/kronos.py:389`), `top_k_top_p_filtering` (`model/kronos.py:331`), `sample_from_logits` (`model/kronos.py:373`), `calc_time_stamps` (`model/kronos.py:472`), `set_seed` (`tests/test_kronos_regression.py:36`) |
| **私有方法** | 前缀单下划线 | `_init_weights` (`model/kronos.py:225`), `_norm` (`model/module.py:263`), `_rotate_half` (`model/module.py:310`), `_load_and_preprocess_data` (`finetune_csv/finetune_base_model.py:45`), `_setup_device` (`finetune_csv/train_sequential.py:29`) |
| **模块级常量** | UPPER_SNAKE_CASE (大写下划线) | `TEST_DATA_ROOT` (`tests/test_kronos_regression.py:12`), `MAX_CTX_LEN` (`tests/test_kronos_regression.py:32`), `FEATURE_NAMES` (`tests/test_kronos_regression.py:20`), `SEED` (`tests/test_kronos_regression.py:33`) |
| **实例属性** | snake_case | `self.price_cols` (`model/kronos.py:489`), `self.max_context` (`model/kronos.py:486`), `self.ffn_dropout_p` (`model/kronos.py:207`) |
| **变量名** | snake_case | `pred_len`, `x_timestamp`, `y_timestamp`, `sample_count`, `bsq_loss` |
| **模型超参数名** | snake_case, 紧凑缩写 | `s1_bits`, `s2_bits`, `ff_dim`, `d_model`, `n_heads`, `n_enc_layers`, `ffn_dropout_p`, `attn_dropout_p`, `resid_dropout_p`, `learn_te` |
| **配置属性** | snake_case | `qlib_data_path`, `tokenizer_learning_rate`, `predictor_learning_rate` (`finetune/config.py`) |
| **类实例变量（Config 类）** | snake_case | `self.dataset_path`, `self.save_path`, `self.pretrained_tokenizer_path` (`finetune/config.py`) |

**命名不一致情况:**
- `finetune/config.py` 中 `self.feature_list` 使用 `vol`/`amt` 而非 `volume`/`amount`，与 `model/kronos.py` 的 `self.price_cols` 中的 `volume`/`amount` 不一致
- `KronosPredictor` 使用 `self.amt_vol` (`model/kronos.py:492`) 命名，而非 `self.amount_col`，风格不统一
- `examples/` 下部分文件函数命名不遵循 snake_case：如 `prediction_new.py` 中的 `fetchRealStockData` (实际内部使用 snake_case：`fetch_real_stock_data`)

## 文件组织规范

**文件命名:**
- 核心模型代码：snake_case 简洁命名，如 `kronos.py`, `module.py`
- 训练脚本：snake_case 动词开头，如 `train_predictor.py`, `train_tokenizer.py`, `train_sequential.py`
- 数据处理：snake_case，如 `qlib_data_preprocess.py`, `dataset.py`
- 测试文件：`test_` 前缀，如 `test_kronos_regression.py`
- 配置文件：snake_case，如 `config.py`, `config_loader.py`
- YAML 配置：snake_case 描述性命名，如 `config_ali09988_candle-5min.yaml`
- 示例脚本：snake_case 动词开头，如 `prediction_example.py`, `run_backtest_kronos.py`

**目录组织规则:**
- `model/` — 核心模型定义，独立子包（含 `__init__.py`）
- `tests/` — 测试文件和数据分离（`tests/data/`）
- `finetune/` — Qlib 微调流水线（每个脚本独立功能）
- `finetune/utils/` — 微调共享工具（含 `__init__.py`）
- `finetune_csv/` — CSV 微调流水线（独立子目录，含 configs、data、examples）
- `examples/` — 使用示例脚本
- `webui/` — Web 界面（Flask app + templates）

**模块入口:**
- 每个模块通过 `__init__.py` 导出关键类：`model/__init__.py` 导出 `KronosTokenizer, Kronos, KronosPredictor`
- `finetune/utils/__init__.py` 为空文件，仅声明为包

## 导入规范

**import 顺序 (广泛观察到的模式):**

1. **标准库模块** — `import os, sys, random, math, json, time, pickle`
2. **第三方库** — `import numpy as np`, `import pandas as pd`, `import torch`
3. **第三方工具库** — `from tqdm import trange/tqdm`, `from huggingface_hub import PyTorchModelHubMixin`, `from einops import rearrange, reduce`
4. **PyTorch 子模块** — `import torch.nn as nn`, `import torch.nn.functional as F`, `from torch.autograd import Function`, `import torch.distributed as dist`
5. **项目内部模块** — `from model.module import *`, `from config import Config`, `from model import Kronos, KronosTokenizer, KronosPredictor`

**典型 import 示例 (`model/kronos.py`):**
```python
import numpy as np
import pandas as pd
import torch
from huggingface_hub import PyTorchModelHubMixin
import sys

from tqdm import trange

sys.path.append("../")
from model.module import *
```

**典型 import 示例 (`finetune/train_predictor.py`):**
```python
import os
import sys
import json
import time
from time import gmtime, strftime
import torch.distributed as dist
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import comet_ml

# Ensure project root is in path
sys.path.append('../')
from config import Config
from dataset import QlibDataset
from model.kronos import KronosTokenizer, Kronos
# Import shared utilities
from utils.training_utils import (
    setup_ddp,
    cleanup_ddp,
    set_seed,
    get_model_size,
    format_time
)
```

**路径操作模式:**
- 使用 `sys.path.append('../')` 或 `sys.path.append("../")` 将项目根目录添加到 Python 路径
- 此模式在几乎所有非根目录脚本中使用：`finetune/`, `examples/`, `webui/`, `finetune_csv/`, `tests/data/`
- 这是项目中最显著的导入约定，但也是潜在的技术债（依赖运行时路径修改而非正确的包安装）

**Path Aliases:**
- 无 `PYTHONPATH` 配置
- 无 `pyproject.toml` 或 `setup.py` 定义包路径
- 所有路径通过运行时 `sys.path.append` 解决

**Wildcard Import:**
- `model/kronos.py:10` 使用 `from model.module import *`，这是唯一一处通配导入
- `model/module.py` 定义了约 15 个类/函数，通配导入会将它们全部引入 `kronos.py` 的命名空间

## 类型系统

**类型注解使用情况:**
- **部分使用**：类型注解在关键位置使用，但并非全面覆盖
- **使用场景**：函数签名中的参数和返回值类型、Config 类属性

**具体模式:**

```python
# 函数签名类型注解（完整）
def set_seed(seed: int) -> None:                               # tests/test_kronos_regression.py:36
def set_seed(seed: int, rank: int = 0):                        # finetune/utils/training_utils.py:41
def get_model_size(model: torch.nn.Module) -> str:             # finetune/utils/training_utils.py:62
def format_time(seconds: float) -> str:                        # finetune/utils/training_utils.py:105
def load_models(config: dict) -> tuple[KronosTokenizer, Kronos]:  # finetune/qlib_test.py:207 (Python 3.10+ 语法)
def generate_predictions(config: dict, test_data: dict) -> dict[str, pd.DataFrame]:  # finetune/qlib_test.py:239

# Dataset 类类型注解
def __init__(self, data_type: str = 'train'):                  # finetune/dataset.py:23
def __len__(self) -> int:                                       # finetune/dataset.py:88
def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:  # finetune/dataset.py:92

# 模型类参数缺乏类型注解
def __init__(self, d_in, d_model, n_heads, ff_dim, ...):       # model/kronos.py:40 — 无类型注解
def forward(self, x):                                           # model/kronos.py:74 — 无类型注解
```

**类型注解规则:**
- 工具函数 (`finetune/utils/training_utils.py`, `tests/`) 使用完整类型注解
- PyTorch 模型类 (`model/module.py`, `model/kronos.py`) 不使用参数类型注解（这是 PyTorch 生态的常见惯例）
- 新增代码 (`finetune_csv/`) 也遵循部分注解模式
- ConfigLoader 使用 `Dict[str, Any]` 类型注解 (`finetune_csv/config_loader.py`)

**新代码应遵循:**
- 公共 API 函数添加完整类型注解（参数 + 返回值）
- PyTorch 模型类参数可不注解，但 `forward()` 方法的参数应使用 `torch.Tensor` 类型注解
- 使用 Python 3.10+ 泛型语法 (`list[X]` 而非 `List[X]`)

## 错误处理规范

**异常类型:**
- 主要使用 `ValueError` 进行输入校验
- 使用 `RuntimeError` 进行运行环境检查
- 使用 `FileNotFoundError` 进行文件路径校验
- 使用 `NotImplementedError` 进行未实现功能标记
- 不使用自定义异常类

**核心校验模式 (`model/kronos.py:519-646`):**
```python
# 输入类型校验
if not isinstance(df, pd.DataFrame):
    raise ValueError("Input must be a pandas DataFrame.")

# 必要列校验
if not all(col in df.columns for col in self.price_cols):
    raise ValueError(f"Price columns {self.price_cols} not found in DataFrame.")

# NaN 检测
if df[self.price_cols + [self.vol_col, self.amt_vol]].isnull().values.any():
    raise ValueError("Input DataFrame contains NaN values in price or volume columns.")

# 长度一致性校验
if not (len(df_list) == len(x_timestamp_list) == len(y_timestamp_list)):
    raise ValueError("df_list, x_timestamp_list, y_timestamp_list must have consistent lengths.")
```

**assert 用法 (`model/module.py`):**
```python
assert self.embed_dim % group_size == 0, "embed_dim must be divisible by group_size"
assert z.shape[-1] == self.embed_dim, f"Expected {self.embed_dim} dimensions, got {z.shape[-1]}"
assert isinstance(s2_bits, int) and s2_bits > 0, "s2_bits must be a positive integer"
```
- `assert` 仅用于 PyTorch 模型内部的形状/维度校验，不用于业务逻辑校验
- 这是 PyTorch 生态的常见模式

**训练脚本环境校验:**
```python
if "WORLD_SIZE" not in os.environ:
    raise RuntimeError("This script must be launched with `torchrun`.")
```
- (`finetune/train_predictor.py:241`, `finetune/train_tokenizer.py:278`)

**错误处理缺失领域:**
- `webui/app.py` 使用 `try/except Exception as e` 的宽泛捕获，返回 JSON 错误响应
- `examples/` 脚本基本无错误处理（示例代码，非生产级别）
- 模型推理流程 (`auto_regressive_inference`) 无异常处理，依赖调用方提供有效输入

## 文档规范

**注释风格:**
- 使用 Google-style docstring（`Args:`, `Returns:`, `Raises:` 段）
- 模型类使用多行 docstring 描述参数
- 工具函数使用单行或多行 docstring
- 代码注释使用英文为主

**Docstring 格式 (`model/kronos.py:15-38`):**
```python
class KronosTokenizer(nn.Module, PyTorchModelHubMixin):
    """
    KronosTokenizer module for tokenizing input data using a hybrid quantization approach.

    This tokenizer utilizes a combination of encoder and decoder Transformer blocks
    along with the Binary Spherical Quantization (BSQuantizer) to compress and decompress input data.

    Args:
           d_in (int): Input dimension.
           d_model (int): Model dimension.
           n_heads (int): Number of attention heads.
           ff_dim (int): Feed-forward dimension.
           ...
    """
```

**Docstring 覆盖情况:**
- `model/kronos.py`：类和关键方法均有 docstring（`KronosTokenizer`, `Kronos`, `forward`, `encode`, `decode`, `indices_to_bits`, `decode_s1`, `decode_s2`）
- `model/module.py`：部分方法有简短 docstring（如 `codes_to_indexes:163`, `indexes_to_codes:179`），大部分无 docstring
- `finetune/` 目录：所有公共函数和类都有详细 docstring（README 中注明部分注释由 AI 生成）
- `finetune_csv/` 目录：`config_loader.py` 有 docstring，训练脚本函数有 docstring
- `tests/` 目录：测试函数无 docstring（通过函数名和变量名自解释）
- `examples/` 目录：部分脚本有中文注释（`run_backtest_kronos.py` 使用中文注释和中文变量名）

**Docstring 规则:**
- 新增类必须使用 Google-style docstring，包含 `Args` 段描述每个参数的类型和含义
- 新增公共函数必须包含 `Args` 和 `Returns` 段
- 私有方法 (`_method`) 可以不写 docstring，但应有简短注释
- 模型 `forward()` 方法必须说明输入输出形状

**行内注释:**
- 使用 `#` 单行注释
- 注释与代码之间保持一个空格
- 短注释放在代码行末，长注释放在代码行上方
- Section 分隔使用 `# ===...===` 模式 (`finetune/config.py:9-11`)
- 变量说明紧跟在赋值后 (`model/kronos.py:55 — `self.codebook_dim = s1_bits + s2_bits # Total dimension of the codebook after quantization``)

## 配置管理规范

**配置文件格式:**

| 目录 | 格式 | 文件 | 说明 |
|------|------|------|------|
| `finetune/` | Python 类 | `config.py` | 使用 `Config` 类定义所有超参数和路径 |
| `finetune_csv/` | YAML + Python Loader | `configs/config_ali09988_candle-5min.yaml` + `config_loader.py` | YAML 配置文件 + `CustomFinetuneConfig` 类解析 |
| `model/` | HuggingFace config.json | 自动生成 | 模型参数通过 `from_pretrained()` 从 HF Hub 加载 |
| 根目录 | `requirements.txt` | `requirements.txt` | 依赖列表（部分版本锁定，部分不锁定） |

**Config 类模式 (`finetune/config.py`):**
```python
class Config:
    def __init__(self):
        self.qlib_data_path = "~/.qlib/qlib_data/cn_data"
        self.instrument = 'csi300'
        self.epochs = 30
        self.batch_size = 50
        # ... 其他属性
```
- 所有配置在 `__init__` 中硬编码默认值
- 使用 `TODO` 注释标记需要用户修改的路径 (`finetune/config.py:12,40,81,99`)
- 训练脚本通过 `Config().__dict__` 将配置转为字典传递 (`finetune/train_predictor.py:244`)
- 私有方法用于辅助计算 (`_set_benchmark` — `finetune/config.py:122`)

**YAML 配置模式 (`finetune_csv/config_loader.py`):**
```yaml
data:
  data_path: "/xxxx/Kronos/finetune_csv/data/HK_ali_09988_kline_5min_all.csv"
  lookback_window: 512
  predict_window: 48
training:
  tokenizer_epochs: 30
  basemodel_epochs: 20
model_paths:
  pretrained_tokenizer: "/xxx/Kronos/pretrained/Kronos-Tokenizer-base"
  exp_name: "HK_ali_09988_kline_5min_all"
```
- 分层结构：`data`, `training`, `model_paths`, `experiment`, `device`, `distributed`
- `ConfigLoader` 提供 `get_data_config()`, `get_training_config()` 等分类访问方法
- 支持动态路径解析：`{exp_name}` 模板替换和空字符串自动填充 (`config_loader.py:25-48`)
- 训练脚本通过 `CustomFinetuneConfig(config_path)` 加载，调用 `get_tokenizer_config()` / `get_basemodel_config()` 获取子配置字典

**配置加载方式:**
- 训练脚本不使用 argparse（除了 `finetune/qlib_test.py` 的 `--device` 参数）
- `finetune_csv/train_sequential.py` 使用 argparse 加载 `--config` 路径和 `--skip-*` 标志
- 环境变量用于分布式训练 (`WORLD_SIZE`, `RANK`, `LOCAL_RANK`)，不用于配置参数

## Git 规范

**分支命名:**
- 主分支：`master` (非 `main`)
- 功能分支：多种风格混用
  - `fix/batch-dimension-training` — `fix/` 前缀 + 描述
  - `fix/issue-231-sample-from-logits-topk-bug` — `fix/issue-{编号}-` 前缀
  - `codex/fix-webui-python312-deps` — `codex/` 前缀
  - `feature/cn-markets` — `feature/` 前缀
  - `my-feature-branch` — 无前缀
  - `wxs/feature/dev` — `{用户}/feature/dev` 格式（当前分支）

**提交信息格式:**
- 无统一格式规范，但部分提交遵循 Conventional Commits 风格
- 观察到的模式:
  - `fix: preserve batch dimension in tokenizer and predictor training` — `fix:` 前缀
  - `feat: China A-share market` — `feat:` 前缀
  - `Fix data leakage in normalization window (#227)` — 无前缀 + PR编号
  - `Auto-detect device for easier getting started` — 无前缀的描述性消息
  - `修改内容` — 中文提交消息（无前缀）
  - `Refactor auto_regressive_inference to reduce memory allocations and cpu-gpy syncs.` — 无前缀的动词开头描述

**推荐提交消息格式（基于现有惯例推断）:**
```
fix: 简要描述修复的问题
feat: 简要描述新功能
refactor: 简要描述重构内容
```

**Pull Request 格式:**
- PR 编号在提交消息中引用 (`#227`, `#231`, `#243`)
- PR 标题遵循类似提交消息的描述性格式

## 特殊模式

### 模型注册模式
```python
# model/__init__.py
model_dict = {
    'kronos_tokenizer': KronosTokenizer,
    'kronos': Kronos,
    'kronos_predictor': KronosPredictor
}

def get_model_class(model_name):
    if model_name in model_dict:
        return model_dict[model_name]
    else:
        raise NotImplementedError
```
- 使用字典注册模式而非工厂函数，支持通过字符串名称获取模型类
- 打印错误信息后抛出异常 (`model/__init__.py:14-15`)

### HuggingFace Hub 模型加载
```python
# 使用 PyTorchModelHubMixin 实现的 from_pretrained 方法
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
```
- 所有模型类继承 `PyTorchModelHubMixin` (`model/kronos.py:13,180`)
- 支持从 HuggingFace Hub 加载预训练模型和特定 revision
- 测试中使用 `revision` 参数锁定模型版本确保回归一致性 (`tests/test_kronos_regression.py:30-31`)

### 分布式训练模式 (DDP)
```python
# torchrun 启动要求
if "WORLD_SIZE" not in os.environ:
    raise RuntimeError("This script must be launched with `torchrun`.")

# DDP 初始化
rank, world_size, local_rank = setup_ddp()
model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

# 分布式数据加载
train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=train_sampler)

# 分布式验证指标聚合
dist.all_reduce(val_loss_sum_tensor, op=dist.ReduceOp.SUM)
```
- 所有训练脚本强制 `torchrun` 启动（`finetune/` 目录）
- `finetune_csv/` 支持单 GPU 和 DDP 模式（`train_sequential.py:40-48`）
- DDP 初始化/清理封装在 `setup_ddp()`/`cleanup_ddp()` (`finetune/utils/training_utils.py`)

### 随机种子设置模式
```python
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```
- 此模式在多处重复定义：`tests/test_kronos_regression.py:36-42`, `tests/data/generate_regression_output.py:26-32`, `finetune/utils/training_utils.py:41-59`, `finetune_csv/finetune_tokenizer.py:24-32`
- 分布式版本增加 rank 偏移：`actual_seed = seed + rank` (`finetune/utils/training_utils.py:50`)
- Dataset 类使用独立 `py_rng` 避免干扰全局随机状态 (`finetune/dataset.py:31`)

### 数据归一化模式
```python
# 实例级归一化（仅使用 lookback 窗口统计，防止数据泄漏）
x_mean = np.mean(x, axis=0)
x_std = np.std(x, axis=0)
x = (x - x_mean) / (x_std + 1e-5)
x = np.clip(x, -self.clip, self.clip)
```
- 所有预测和训练流程使用相同的归一化模式 (`model/kronos.py:544-547`, `finetune/dataset.py:109-117`)
- 使用 `1e-5` 防止除零
- 使用 `clip` 参数（默认 5.0）限制极端值
- 训练数据集使用 lookback 窗口统计归一化（而非全窗口），避免数据泄漏 (`finetune/dataset.py:108-117`)

### 滑动窗口数据加载模式
```python
class QlibDataset(Dataset):
    def __init__(self, data_type: str = 'train'):
        self.window = self.config.lookback_window + self.config.predict_window + 1
        self.indices = []  # 预计算所有 (symbol, start_index) 对
        for symbol in self.symbols:
            for i in range(num_samples):
                self.indices.append((symbol, i))

    def __getitem__(self, idx: int):
        random_idx = self.py_rng.randint(0, len(self.indices) - 1)
        symbol, start_idx = self.indices[random_idx]
```
- 预计算所有有效滑动窗口索引，训练时随机采样
- 使用独立的 `py_rng` 保证 epoch 间可复现的随机采样 (`finetune/dataset.py:77-86`)
- 每个样本包含 lookback + predict 窗口

### 自回归推理模式
```python
def auto_regressive_inference(tokenizer, model, x, x_stamp, y_stamp, max_context, pred_len, ...):
    # 滑动窗口 + buffer 模式
    pre_buffer = x_token[0].new_zeros(batch_size, max_context)
    post_buffer = x_token[1].new_zeros(batch_size, max_context)
    for i in range(pred_len):
        # 1. 编码当前输入 → s1_logits, context
        # 2. 从 s1_logits 采样 → sample_s1
        # 3. 用 sample_s1 解码 s2 → s2_logits
        # 4. 从 s2_logits 采样 → sample_s2
        # 5. 将新 token 添加到 buffer
```
- 使用滑动窗口 buffer 管理 (`model/kronos.py:408-454`)
- 支持 `sample_count` 多次采样取平均 (`model/kronos.py:465-468`)
- 支持 top-k / top-p 过滤 (`model/kronos.py:331-371`)
- 分层解码：先 s1 再 s2（依赖感知层 DependencyAwareLayer）

### 日志记录模式
- `finetune/` 使用 `print()` 直接输出 + `comet_ml` 实验追踪
- `finetune_csv/` 使用 Python `logging` 模块 + `RotatingFileHandler` (`finetune_csv/finetune_tokenizer.py:49-65`)
- `webui/` 使用 `print()` 输出
- 测试使用 `print()` 输出中间结果 (`tests/test_kronos_regression.py:87,138`)
- 无统一的日志框架

### Flask Web 应用模式
```python
app = Flask(__name__)
CORS(app)

# 全局模型实例
tokenizer = None
model = None
predictor = None

# RESTful API 路由
@app.route('/api/predict', methods=['POST'])
def predict():
    ...

@app.route('/api/load-model', methods=['POST'])
def load_model():
    global tokenizer, model, predictor
    ...
```
- 使用全局变量存储模型实例 (`webui/app.py:28-30`)
- API 路由使用 `/api/` 前缀 (`webui/app.py:330-698`)
- 使用 `try/except Exception` 宽泛捕获，返回 JSON 错误
- 使用 Plotly 生成图表，通过 `plotly.utils.PlotlyJSONEncoder` 序列化为 JSON

### 微调流水线模式
- **Qlib 流水线 (`finetune/`):** 数据预处理 → 分词器训练 → 预测器训练 → 回测
- **CSV 流水线 (`finetune_csv/`):** 配置加载 → 分词器训练 → 基础模型训练（SequentialTrainer 编排）
- 两个流水线使用不同的配置方式和数据加载逻辑，但共享相同的模型架构

---

*规范分析: 2026-07-29*
