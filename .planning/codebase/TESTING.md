# 测试 (Testing)

**分析日期:** 2026-07-29

## 测试框架

| 框架 | 版本 | 用途 |
|------|------|------|
| pytest | 未指定版本 (通过 requirements.txt 间接依赖) | 测试运行器 |
| numpy.testing | numpy (requirements.txt 无版本锁定) | 数值精度断言 |
| torch.no_grad() | torch>=2.0.0 | 推理模式上下文管理 |

**备注:** `pytest` 未在 `requirements.txt` 中列出，但测试文件 `tests/test_kronos_regression.py` 使用了 `import pytest`。这意味着 pytest 是开发依赖，需单独安装。

**配置文件:**
- 无 `pytest.ini`, `conftest.py`, `setup.cfg`, `pyproject.toml` 等测试配置文件
- 测试参数（如容差、种子）硬编码在测试文件顶部的常量中

## 测试结构

**测试文件分布:**

```
tests/
├── test_kronos_regression.py          # 回归测试主文件 (140 行)
└── data/
    ├── generate_regression_output.py  # 回归基线数据生成脚本 (91 行)
    ├── regression_input.csv           # 测试输入数据 (2500 行)
    ├── regression_output_256.csv      # 256上下文长度的预期输出 (8 行数据)
    └── regression_output_512.csv      # 512上下文长度的预期输出 (8 行数据)
```

**目录特征:**
- 测试文件集中在 `tests/` 目录（独立于源代码目录）
- 测试数据放在 `tests/data/` 子目录
- 无 `conftest.py` — 无共享 fixture
- 无 `__init__.py` — tests 不是 Python 包

**finetune/qlib_test.py:**
- 文件名为 `qlib_test.py`，但不是 pytest 测试文件
- 这是一个推理 + 回测脚本（362行），而非单元/集成测试
- 包含 `QlibTestDataset`, `QlibBacktest`, `generate_predictions()` 等功能代码
- 通过 `argparse` 接受 `--device` 参数，作为独立脚本运行
- 运行命令: `python finetune/qlib_test.py --device cuda:0`

## 测试类型

| 类型 | 数量 | 覆盖范围 | 证据文件 |
|------|------|---------|---------|
| **回归精度测试** | 2 个测试函数, 4 个参数化变体 | KronosPredictor 的 predict 输出精度 | `tests/test_kronos_regression.py:46-88` |
| **MSE 回归测试** | 1 个测试函数, 2 个参数化变体 | KronosPredictor 的 MSE 指标一致性 | `tests/test_kronos_regression.py:90-140` |
| **单元测试** | 0 | 无 | 不存在 |
| **集成测试** | 0 (qlib_test.py 是脚本而非测试) | 无 | 不存在 |
| **E2E 测试** | 0 | 无 | 不存在 |
| **性能测试** | 0 | 无 | 不存在 |

## 详细测试分析

### test_kronos_predictor_regression (精度回归测试)

**位置:** `tests/test_kronos_regression.py:46-88`

**参数化:** `@pytest.mark.parametrize("context_len", TEST_CTX_LEN)` — 产生 2 个变体:
- `context_len=512`
- `context_len=256`

**测试流程:**
1. 设置随机种子 (`SEED=123`)
2. 加载输入 CSV 和预期输出 CSV
3. 从 HuggingFace Hub 加载模型和 tokenizer（锁定 revision）
4. 创建 `KronosPredictor` 实例
5. 使用 `torch.no_grad()` 执行推理
6. 将预测结果与预期值比较，使用 `np.testing.assert_allclose(rtol=1e-5)` 验证

**关键参数:**
```python
TEST_CTX_LEN = [512, 256]           # 测试的上下文长度
PRED_LEN = 8                         # 预测长度
REL_TOLERANCE = 1e-5                 # 相对容差
FEATURE_NAMES = ["open", "high", "low", "close", "volume", "amount"]
MODEL_REVISION = "901c26c1332695a2a8f243eb2f37243a37bea320"
TOKENIZER_REVISION = "0e0117387f39004a9016484a186a908917e22426"
SEED = 123
DEVICE = "cpu"
```

**设计特点:**
- 使用 HuggingFace revision 锁定模型版本，确保基线数据与模型输出一致
- 测试数据（输入和预期输出）以 CSV 文件形式存储在 `tests/data/`
- 使用确定性采样参数 (`top_k=1`, `top_p=1.0`, `sample_count=1`, `T=1.0`) — 等价于贪心解码
- 仅在 CPU 上运行（`DEVICE = "cpu"`）

### test_kronos_predictor_mse (MSE 回归测试)

**位置:** `tests/test_kronos_regression.py:90-140`

**参数化:** `@pytest.mark.parametrize("context_len, expected_mse", zip(MSE_CTX_LEN, MSE_EXPECTED))` — 产生 2 个变体:
- `(context_len=512, expected_mse=0.008979)`
- `(context_len=256, expected_mse=0.003741)`

**测试流程:**
1. 设置随机种子
2. 加载模型和 tokenizer（同 regression 测试）
3. 从有效区域随机采样 4 个起始点
4. 对每个起始点执行预测，计算 MSE
5. 验证平均 MSE 与预期值的差异在容差范围内 (`MSE_TOLERANCE=0.000001`)

**关键参数:**
```python
MSE_SAMPLE_SIZE = 4                  # 采样点数
MSE_CTX_LEN = [512, 256]             # 上下文长度
MSE_EXPECTED = [0.008979, 0.003741]  # 预期 MSE 值
MSE_PRED_LEN = 30                    # MSE 测试的预测长度（不同于精度测试）
MSE_TOLERANCE = 0.000001             # MSE 容差
MSE_FEATURE_NAMES = ["open", "high", "low", "close"]  # 仅价格列（不含 volume/amount）
```

**设计特点:**
- 从数据集随机采样而非固定起始点，但使用固定种子保证可复现性
- MSE 容差极小 (0.000001)，对数值稳定性要求高
- 仅验证 4 个价格列 (OHLC)，不包含 volume/amount

### generate_regression_output.py (基线数据生成)

**位置:** `tests/data/generate_regression_output.py`

**用途:** 生成回归测试的预期输出 CSV 文件

**运行方式:** 作为独立脚本执行 (`python tests/data/generate_regression_output.py`)

**关键特征:**
- 与测试文件共享相同的模型加载和推理逻辑
- 使用相同的种子和确定性参数
- 输出 CSV 保存到 `tests/data/regression_output_{ctx_len}.csv`
- 包含输入数据行数校验 (`df.shape[0] < MAX_CTX_LEN + PRED_LEN`)
- 包含输出形状校验 (`pred_df.shape != (PRED_LEN, len(FEATURE_NAMES))`)

**注意事项:** 基线数据生成脚本与测试文件有代码重复（`set_seed` 函数、配置常量、模型加载逻辑）

## 测试覆盖

**覆盖率数据:** 无覆盖率工具配置，无覆盖率报告

**覆盖率目标:** 无强制覆盖率要求

**现有覆盖范围评估:**

| 组件 | 是否被测试 | 测试范围 | 测试文件 |
|------|-----------|---------|---------|
| **KronosPredictor.predict()** | 是 | 精度验证 + MSE 验证 | `tests/test_kronos_regression.py` |
| **KronosPredictor.predict_batch()** | 否 | 无测试 | 不存在 |
| **KronosTokenizer.encode()** | 间接（通过 Predictor 调用） | 无独立测试 | 不存在 |
| **KronosTokenizer.decode()** | 间接（通过 Predictor 调用） | 无独立测试 | 不存在 |
| **Kronos.forward()** | 否 | 无独立测试 | 不存在 |
| **Kronos.decode_s1()** | 否 | 无独立测试 | 不存在 |
| **Kronos.decode_s2()** | 否 | 无独立测试 | 不存在 |
| **BinarySphericalQuantizer** | 否 | 无独立测试 | 不存在 |
| **top_k_top_p_filtering()** | 否 | 无独立测试 | 不存在 |
| **sample_from_logits()** | 否 | 无独立测试 | 不存在 |
| **auto_regressive_inference()** | 间接（通过 Predictor 调用） | 无独立测试 | 不存在 |
| **calc_time_stamps()** | 否 | 无独立测试 | 不存在 |
| **KronosPredictor 输入校验** | 否 | ValueError 未被测试 | 不存在 |
| **KronosPredictor 边界条件** | 否 | 空 DataFrame、NaN、短序列 | 不存在 |
| **QlibDataset** | 否 | 无独立测试 | 不存在 |
| **CustomKlineDataset** | 否 | 无独立测试 | 不存在 |
| **Config/ConfigLoader** | 否 | 无独立测试 | 不存在 |
| **training_utils** | 否 | 无独立测试 | 不存在 |
| **WebUI API endpoints** | 否 | 无独立测试 | 不存在 |

## 运行测试

**运行命令:**

```bash
# 运行全部测试
pytest tests/

# 运行特定测试文件
pytest tests/test_kronos_regression.py

# 运行特定测试函数
pytest tests/test_kronos_regression.py::test_kronos_predictor_regression

# 运行特定参数化变体
pytest tests/test_kronos_regression.py::test_kronos_predictor_regression[512]

# 运行 MSE 测试
pytest tests/test_kronos_regression.py::test_kronos_predictor_mse

# 生成覆盖率报告 (需安装 pytest-cov)
pytest tests/ --cov=model --cov-report=html
```

**注意事项:**
- 测试需要从 HuggingFace Hub 下载模型（首次运行约需数分钟）
- 测试仅在 CPU 上运行，不需要 GPU
- 测试依赖 `tests/data/regression_input.csv` 和 `tests/data/regression_output_*.csv` 文件
- pytest 需额外安装（未在 requirements.txt 中列出）
- 测试使用确定性参数 (`top_k=1`, `top_p=1.0`, `T=1.0`) 确保结果可复现

**生成基线数据:**
```bash
# 重新生成回归测试的预期输出（模型更新后需要）
python tests/data/generate_regression_output.py
```

**watch 模式:** 未配置（无 pytest-watch）

**并行执行:** 未配置（无 pytest-xdist）

## 测试数据

**测试数据来源与管理:**

| 数据文件 | 行数 | 用途 | 路径 |
|---------|------|------|------|
| regression_input.csv | 2500 | 测试输入时间序列数据 | `tests/data/regression_input.csv` |
| regression_output_512.csv | 8 (数据行) | 512上下文长度的预期预测输出 | `tests/data/regression_output_512.csv` |
| regression_output_256.csv | 8 (数据行) | 256上下文长度的预期预测输出 | `tests/data/regression_output_256.csv` |

**数据格式:**
- 输入 CSV: 包含 `timestamps` 列 + `open, high, low, close, volume, amount` 列
- 输出 CSV: 包含 `timestamps` 列 + `open, high, low, close, volume, amount` 列
- timestamps 为 ISO 格式日期时间字符串

**数据生成方式:**
- `tests/data/generate_regression_output.py` 脚本从 HuggingFace 模型生成预期输出
- 输入数据 (`regression_input.csv`) 来源不明，可能是从项目示例数据 (`data/XSHG_5min_600977.csv`) 派生

**数据管理策略:**
- 基线数据作为 CSV 文件存储在 Git 仓库中（小文件，仅 8 行数据）
- 输入数据也存储在 Git 仓库中（2500 行，约数十 KB）
- 模型文件从 HuggingFace Hub 下载（不存储在仓库中）
- 模型 revision 通过 git hash 锁定，确保基线数据与模型版本对应

**Fixtures:**
- 无 pytest fixtures（无 `conftest.py`）
- 测试配置通过模块级常量定义
- 模型加载在每个测试函数内部执行（无共享 fixture）

## CI/CD

**自动化测试配置:** 无

- 不存在 `.github/workflows/` 目录
- 不存在 `.github/` 目录
- 不存在 `Jenkinsfile`, `Makefile`, `tox.ini` 等CI配置
- 无 GitHub Actions, CircleCI, Travis CI 或其他 CI 服务配置

**PR 测试流程:**
- PR 合入无自动化测试门控
- 依赖人工审查和手动测试
- PR 描述中偶尔引用测试结果（如 `#173 from AnMakc/regression_test`）

**本地测试建议:**
- 提交前手动运行 `pytest tests/`
- 修改模型代码后，可能需要重新生成基线数据

## 测试缺失领域

### 严重缺失（高优先级）

**1. 输入校验测试 — KronosPredictor.validate_inputs:**
- 未测试位置: `model/kronos.py:522-535`
- 未覆盖功能:
  - `ValueError("Input must be a pandas DataFrame.")` — 非 DataFrame 输入
  - `ValueError(f"Price columns {self.price_cols} not found in DataFrame.")` — 缺少必要列
  - `ValueError("Input DataFrame contains NaN values in price or volume columns.")` — NaN 值
- 风险: 边界条件错误可能在使用方未注意到时静默通过或产生错误输出
- 建议: 添加 `test_kronos_predictor_input_validation.py`

**2. predict_batch() 测试:**
- 未测试位置: `model/kronos.py:562-661`
- 未覆盖功能:
  - 批量预测的输出一致性（与 predict 结果是否一致）
  - 长度不一致时的 ValueError
  - 单条序列批量预测 vs 多条序列批量预测
  - volume/amount 缺失时的自动填充行为
- 风险: predict_batch 是核心功能，且代码逻辑复杂（660行文件中的100行方法），无任何测试覆盖
- 建议: 添加 `test_kronos_predictor_batch.py`

**3. 模型核心组件测试:**
- 未测试位置: `model/module.py`（570行）
- 未覆盖功能:
  - `BinarySphericalQuantizer` 的量化/反量化正确性
  - `HierarchicalEmbedding` 的 token 分割和融合 (`split_token`)
  - `DependencyAwareLayer` 的交叉注意力
  - `DualHead` 的损失计算 (`compute_loss`)
  - `RMSNorm` 的归一化行为
  - `TemporalEmbedding` 的时间特征提取
- 风险: 这些组件是模型推理的核心，任何错误都可能导致预测结果偏移而不被察觉
- 建议: 添加 `test_kronos_components.py`，使用小模型参数验证各组件输出

### 中等缺失（中优先级）

**4. 采样逻辑测试:**
- 未测试位置: `model/kronos.py:331-387`
- 未覆盖功能:
  - `top_k_top_p_filtering()` 在各种 top_k/top_p 组合下的行为
  - `sample_from_logits()` 的温度缩放效果
  - `top_k=1` 是否正确返回最高概率 token
  - `top_p` 过滤是否保留累积概率 >= threshold 的 token
- 风险: 采样逻辑直接影响预测质量，且曾经有 bug（`fix/issue-231-sample-from-logits-topk-bug`）
- 建议: 添加 `test_sampling.py`

**5. auto_regressive_inference 测试:**
- 未测试位置: `model/kronos.py:389-469`
- 未覆盖功能:
  - 滑动窗口 buffer 的边界行为（当序列长度 < max_context / > max_context）
  - 多次采样 (`sample_count>1`) 的平均行为
  - 长序列推理的内存管理
- 风景: 推理逻辑是用户最常使用的功能路径，但仅通过 Predictor 间接覆盖

**6. 归一化/反归一化测试:**
- 未测试位置: `model/kronos.py:544-557`
- 未覆盖功能:
  - 归一化后反归一化是否恢复原始值
  - `1e-5` 防除零的效果
  - clip 阈值的边界行为
  - volume/amount 自动计算 (`df[self.amt_vol] = df[self.vol_col] * df[self.price_cols].mean(axis=1)`)

**7. 数据集类测试:**
- 未测试位置: `finetune/dataset.py`, `finetune_csv/finetune_base_model.py` (`CustomKlineDataset`)
- 未覆盖功能:
  - 滑动窗口索引是否正确
  - 数据分割是否无泄漏
  - 归一化是否仅使用 lookback 窗口统计量
  - 随机采样的一致性

### 低优先级缺失

**8. 配置类测试:**
- 未测试位置: `finetune/config.py`, `finetune_csv/config_loader.py`
- 未覆盖功能:
  - Config 属性默认值合理性
  - ConfigLoader 的 YAML 解析和路径解析
  - 动态路径模板替换

**9. WebUI API 测试:**
- 未测试位置: `webui/app.py`
- 未覆盖功能:
  - `/api/predict` 的各种输入组合
  - `/api/load-model` 的模型加载
  - `/api/data-files` 的文件扫描
  - 错误响应格式

**10. 训练脚本测试:**
- 未测试位置: `finetune/train_predictor.py`, `finetune/train_tokenizer.py`, `finetune_csv/`
- 未覆盖功能:
  - 训练循环是否正确执行
  - 模型保存和加载是否一致
  - DDP 环境初始化
  - 梯度累积逻辑

## 测试模式与惯例

### 确定性测试模式
```python
# 测试中强制确定性
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 推理中使用确定性参数
with torch.no_grad():
    pred_df = predictor.predict(
        df=context_features,
        x_timestamp=x_timestamp,
        y_timestamp=future_timestamp,
        pred_len=expected.shape[0],
        T=1.0,          # 固定温度
        top_k=1,         # 确定性选择最高概率
        top_p=1.0,       # 不做概率过滤
        verbose=False,
        sample_count=1,  # 单次采样
    )
```
- 所有涉及随机性的测试必须在开头调用 `set_seed()`
- 推理参数使用确定性设置：`top_k=1`, `top_p=1.0`, `T=1.0`, `sample_count=1`
- 仅在 CPU 上运行测试 (`DEVICE = "cpu"`)

### 数值精度断言模式
```python
# 相对容差断言（精度回归测试）
np.testing.assert_allclose(obtained, expected, rtol=REL_TOLERANCE)  # rtol=1e-5

# 自定义容差断言（MSE 测试）
assert abs(mse_diff) <= MSE_TOLERANCE, f"MSE {mse} differs from expected {expected_mse}"
```
- 使用 `np.testing.assert_allclose` 进行数值比较（而非 `np.allclose`）
- MSE 测试使用 `assert abs(diff) <= tolerance` 模式
- 容差值硬编码在模块级常量中

### 参数化测试模式
```python
# 简单参数化 — 单参数
@pytest.mark.parametrize("context_len", TEST_CTX_LEN)
def test_kronos_predictor_regression(context_len):
    ...

# 复合参数化 — 多参数 zip
@pytest.mark.parametrize("context_len, expected_mse", zip(MSE_CTX_LEN, MSE_EXPECTED))
def test_kronos_predictor_mse(context_len, expected_mse):
    ...
```
- 使用 `pytest.mark.parametrize` 进行参数化测试
- 简单参数化：单个参数列表
- 复合参数化：使用 `zip` 组合多个参数列表
- 参数列表定义在模块级常量中

### 模型锁定模式
```python
MODEL_REVISION = "901c26c1332695a2a8f243eb2f37243a37bea320"
TOKENIZER_REVISION = "0e0117387f39004a9016484a186a908917e22426"

tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base", revision=TOKENIZER_REVISION)
model = Kronos.from_pretrained("NeoQuasar/Kronos-small", revision=MODEL_REVISION)
```
- 使用 HuggingFace `revision` 参数锁定特定 git commit
- 确保: 模型版本不变 → 输出不变 → 基线数据有效
- 如果模型更新，需要同时更新 `MODEL_REVISION`/`TOKENIZER_REVISION` 和重新生成基线数据

### 错误测试建议模式（新增测试应遵循）

```python
# 输入校验测试
def test_predictor_raises_on_non_dataframe():
    with pytest.raises(ValueError, match="Input must be a pandas DataFrame"):
        predictor.predict(df=[1, 2, 3], ...)

def test_predictor_raises_on_missing_columns():
    df = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(ValueError, match="Price columns"):
        predictor.predict(df=df, ...)

def test_predictor_raises_on_nan_values():
    df = pd.DataFrame({"open": [1, None], "high": [2, 3], "low": [1, 2], "close": [2, 3]})
    with pytest.raises(ValueError, match="NaN values"):
        predictor.predict(df=df, ...)

# 采样逻辑测试
def test_top_k_filtering_keeps_top_k():
    logits = torch.randn(1, 100)
    filtered = top_k_top_p_filtering(logits, top_k=10)
    # 验证只有10个非-inf值
    non_inf_count = (filtered != -float("Inf")).sum().item()
    assert non_inf_count == 10
```

## 测试工具与辅助

**已有的测试辅助代码:**

| 辅助 | 位置 | 用途 |
|------|------|------|
| `set_seed()` | `tests/test_kronos_regression.py:36-42` | 设置全局随机种子 |
| `set_seed()` | `tests/data/generate_regression_output.py:26-32` | 设置全局随机种子（重复定义） |
| `generate_regression_output.py` | `tests/data/generate_regression_output.py` | 基线数据生成脚本 |

**缺失的测试辅助:**
- 无 `conftest.py` — 无共享 fixture（模型加载应在 fixture 中共享）
- 无 pytest markers — 无标记区分慢测试/快测试/CPU/GPU
- 无测试参数化工厂 — 无从配置文件生成参数化变体的机制
- 无 mock/patch 工具 — 无对 HuggingFace Hub 下载的 mock

**建议添加的 conftest.py:**
```python
# tests/conftest.py
import pytest
import torch

@pytest.fixture(scope="session")
def tokenizer():
    from model import KronosTokenizer
    tok = KronosTokenizer.from_pretrained(
        "NeoQuasar/Kronos-Tokenizer-base",
        revision="0e0117387f39004a9016484a186a908917e22426"
    )
    tok.eval()
    return tok

@pytest.fixture(scope="session")
def model():
    from model import Kronos
    m = Kronos.from_pretrained(
        "NeoQuasar/Kronos-small",
        revision="901c26c1332695a2a8f243eb2f37243a37bea320"
    )
    m.eval()
    return m

@pytest.fixture(scope="session")
def predictor(tokenizer, model):
    from model import KronosPredictor
    return KronosPredictor(model, tokenizer, device="cpu", max_context=512)
```

---

*测试分析: 2026-07-29*
