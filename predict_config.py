import os


class PredictConfig:
    """
    Kronos 预测配置项

    所有影响预测结果的参数都在这里集中管理。
    每个参数都有详细说明，包括：作用、影响范围、推荐值、注意事项。

    参数影响层级：
    ┌─────────────────────────────────────────────────────────┐
    │  数据输入 → 归一化 → Tokenizer编码 → 模型推理 → 采样解码 → 反归一化  │
    │   (1)        (2)        (3)          (4)        (5)        (6)      │
    └─────────────────────────────────────────────────────────┘
    """

    def __init__(self):

        # =================================================================
        # (1) 数据输入参数
        # =================================================================
        # 这些参数决定了模型"看到什么数据"，直接影响预测的输入质量。

        # --- 股票基本信息 ---
        self.stock_code = "301293"
        # 股票代码。用于数据获取和结果标识。
        # 深交所: 0/2/3 开头（如 000001, 300499）
        # 上交所: 6/9 开头（如 600036）

        self.stock_name = "高澜股份"
        # 股票名称，仅用于显示和结果文件命名，不影响预测逻辑。

        # --- 数据源 ---
        self.data_source = "eastmoney"
        # 数据获取渠道。可选: "eastmoney"（东方财富API）, "akshare"（AKShare库）, "csv"（本地CSV文件）
        # 东方财富API无需额外依赖，但可能被限流；AKShare需 pip install akshare。
        # 使用 csv 时需同时设置 data_path。

        self.data_path = "./examples/data/300499_stock_data.csv"
        # 本地 CSV 数据文件路径。当 data_source="csv" 时使用。
        # CSV 必须包含列: timestamps, open, high, low, close, volume, amount
        # timestamps 为日期格式，如 "2024-01-02"

        self.data_start_date = "20260101"
        # 数据获取起始日期。格式: YYYYMMDD
        # 建议至少获取 2 年以上数据，确保 lookback 窗口有足够的历史数据。

        self.data_end_date = "20260730"
        # 数据获取截止日期。格式: YYYYMMDD
        # 通常设为当前日期或最近交易日。

        self.data_freq = "daily"
        # 数据频率。可选: "daily"（日线）, "5min"（5分钟线）等
        # 注意：Kronos 预训练模型主要基于日线数据训练，使用其他频率可能需要微调。
        # 日线数据的时间特征: minute=0, hour=0（所有日线数据都是0）
        # 分钟线数据的时间特征: minute/hour 有实际值，模型能捕捉日内模式。

        self.data_adjust = "qfq"
        # 复权方式。可选: "qfq"（前复权）, "hfq"（后复权）, ""（不复权）
        # 前复权：保持当前价格不变，历史价格向前调整。最常用于预测。
        # 后复权：保持历史价格不变，当前价格向后调整。适合计算真实收益。
        # 不复权：原始价格，含除权除息缺口，不适合预测。

        # --- 输入窗口 ---
        self.lookback = 120
        # 回看窗口长度（输入历史数据的行数）。
        # 模型将使用最近 lookback 条数据作为输入。
        # 范围: 10 ~ max_context（small/base=512, mini=2048）
        # 影响: lookback 越大，模型看到的上下文越多，对趋势判断越准确。
        #       但 lookback 超过 max_context 时，超出部分会被截断。
        # 推荐: 日线建议 200~512；分钟线建议 400~2048。
        # 注意: lookback 不能超过可用数据量，否则报错。

        self.pred_len = 10
        # 预测步数（预测未来多少个时间步）。
        # 日线: 1步=1个交易日；5分钟线: 1步=5分钟。
        # 影响: pred_len 越大，预测越远，误差累积越严重。
        # 推荐: 日线建议 5~30；分钟线建议 10~60。
        # 注意: 预测远期时，建议配合低温度(T)和高采样数(sample_count)来减少随机性。

        # =================================================================
        # (2) 归一化参数
        # =================================================================
        # 归一化是预测准确性的关键。Kronos 使用 z-score 归一化：
        #   x_norm = (x - mean) / (std + 1e-5)
        #   x_norm = clip(x_norm, -clip, clip)
        # 反归一化在预测后执行：
        #   x_pred = x_norm * (std + 1e-5) + mean

        self.clip = 5.0
        # 归一化后的裁剪范围。将超出 [-clip, +clip] 的值截断。
        # 作用: 防止极端异常值（如停牌后复牌涨跌停）干扰模型。
        # 影响: clip 过小 → 正常波动被截断，丢失信息；clip 过大 → 异常值干扰模型。
        # 推荐: 5.0（覆盖约99.99997%的正态分布范围）。
        # 代码位置: auto_regressive_inference:391, KronosPredictor.predict:547

        # 归一化基准说明（不可配置，但需要理解）：
        # mean 和 std 是基于输入窗口（lookback 条数据）计算的全特征均值和标准差。
        # 这意味着不同的 lookback 窗口会产生不同的归一化基准，
        # 进而导致不同的预测结果。这是 lookback 影响预测的一个重要原因。

        # =================================================================
        # (3) 模型参数
        # =================================================================
        # 决定使用哪个预训练模型，以及模型的最大上下文长度。

        self.model_name = "Kronos-base"
        # 预训练模型名称。可选:
        #   "Kronos-mini"  — 4.1M 参数，  上下文 2048，最快但精度最低
        #   "Kronos-small" — 24.7M 参数， 上下文 512， 速度和精度平衡
        #   "Kronos-base"  — 102.3M 参数， 上下文 512，  最慢但精度最高
        # 影响: 模型越大，对复杂模式的捕捉能力越强，预测越准确，但推理越慢。
        # 注意: Tokenizer 必须与模型匹配。
        #   mini  → Kronos-Tokenizer-2k
        #   small/base → Kronos-Tokenizer-base

        self.model_path = "./pretrained/Kronos-base"
        # 模型本地路径。可以是本地目录或 HuggingFace Hub 模型 ID（如 "NeoQuasar/Kronos-base"）。
        # 首次使用会自动下载到本地。

        self.tokenizer_path = "./pretrained/Kronos-Tokenizer-base"
        # Tokenizer 本地路径。同上，可以是本地路径或 HF Hub ID。

        self.max_context = 512
        # 模型最大上下文长度（能处理的最大 token 序列长度）。
        # 由模型架构决定，不可随意修改：
        #   Kronos-mini:  2048
        #   Kronos-small: 512
        #   Kronos-base:  512
        # 影响: lookback + pred_len 不能超过 max_context，超出部分会被滑动窗口截断。
        # 代码位置: auto_regressive_inference:408-414（buffer 机制）

        # =================================================================
        # (4) 采样控制参数（核心调参区）
        # =================================================================
        # 这些参数控制模型如何从预测的概率分布中选取 token。
        # Kronos 的每个时间步从 2^10=1024 个候选 token 中采样，
        # 采样策略直接决定预测结果的稳定性和多样性。
        #
        # 采样流程:
        #   logits = model_output / T              ← 温度缩放
        #   logits = top_k_top_p_filtering(logits)  ← 过滤低概率 token
        #   probs  = softmax(logits)               ← 转概率分布
        #   token  = multinomial(probs)             ← 随机采样
        #   token  = argmax(probs)                  ← 贪心采样（sample_logits=False时）

        self.T = 0.5
        # 温度（Temperature）。控制采样概率分布的"锐利程度"。
        # 公式: logits = logits / T
        # 影响:
        #   T < 1.0 → 分布更尖锐，高概率 token 更容易被选中，结果更确定、更保守
        #   T = 1.0 → 原始分布，不做调整
        #   T > 1.0 → 分布更平坦，低概率 token 也有机会被选中，结果更随机、更多样
        # 推荐:
        #   保守预测: 0.5~0.6（趋近贪心，结果稳定）
        #   均衡预测: 0.7~0.9（适度随机）
        #   探索性预测: 1.0~1.2（更多样化，但波动大）
        # 注意: T 越低，多次运行结果越一致；T 越高，结果越不稳定。
        # 代码位置: sample_from_logits:374

        self.top_k = 0
        # Top-K 过滤。只保留概率最高的 K 个 token，其余设为 -inf。
        # top_k=0 表示不启用 Top-K 过滤。
        # 影响: top_k 越小，候选 token 越少，结果越确定；越大越随机。
        # 推荐: 通常设为 0，用 top_p 代替（top_p 更灵活）。
        # 注意: top_k 和 top_p 可以同时使用，但通常只用一个。
        # 代码位置: top_k_top_p_filtering:347-352

        self.top_p = 0.95
        # Top-P（核采样/Nucleus Sampling）过滤。
        # 将 token 按概率从高到低排序，只保留累积概率 ≥ top_p 的最小集合。
        # top_p=1.0 表示不启用过滤（保留所有 token）。
        # 影响: top_p 越小，候选 token 越少，结果越确定；越大越随机。
        # 推荐:
        #   保守预测: 0.85~0.9（只考虑概率前 85~90% 的 token）
        #   均衡预测: 0.9~0.95
        #   探索性预测: 0.95~1.0
        # 代码位置: top_k_top_p_filtering:354-370

        self.sample_count = 15
        # 并行采样次数。每次预测生成 sample_count 条路径，最终取均值。
        # 影响: sample_count 越大，均值越稳定，随机性越小；但计算量线性增加。
        # 推荐:
        #   保守预测: 10~20（结果稳定）
        #   均衡预测: 5~10
        #   快速预测: 1~3（结果波动大，但速度快）
        # 原理: 代码将输入数据复制 sample_count 份，并行推理，最后 reshape 并取均值。
        #   preds = z.reshape(-1, sample_count, ...) → np.mean(preds, axis=1)
        # 代码位置: auto_regressive_inference:394-396, 465-467

        self.sample_logits = True
        # 是否使用随机采样。False 时使用贪心解码（argmax）。
        # True:  token = multinomial(probs)   — 随机采样，每次结果不同
        # False: token = argmax(probs)         — 贪心解码，结果确定
        # 影响: 设为 False 时，结果完全确定（同一输入永远得到同一输出），
        #       但可能陷入重复模式。设为 True 时结果有多样性。
        # 推荐: 保守预测用 True + 低 T + 高 sample_count；确定性结果用 False。
        # 注意: 当前代码中此参数未暴露为配置项，默认 True。
        #       如需修改，需在 sample_from_logits 函数中更改。
        # 代码位置: sample_from_logits:381-384

        # =================================================================
        # (5) 推理设备参数
        # =================================================================

        self.device = "cpu"
        # 推理设备。可选: "cpu", "cuda:0", "cuda:1", ..., "mps"
        # cpu:  所有设备都支持，速度最慢
        # cuda: NVIDIA GPU，速度最快
        # mps:  Apple Silicon GPU 加速，速度中等
        # 注意: Mac 上 mps 可能存在兼容性问题（如段错误），建议先用 cpu 测试。
        # 影响: 不影响预测结果，只影响速度。

        # =================================================================
        # (6) 输出参数
        # =================================================================

        self.output_dir = "./examples/output"
        # 预测结果保存目录。

        self.verbose = True
        # 是否显示推理进度条。

        # =================================================================
        # 以下为微调（Finetune）相关参数
        # 仅供 finetune 流程使用，直接预测时无需关注
        # =================================================================

        # --- 数据集划分 ---
        self.train_time_range = ["2024-01-01", "2025-12-31"]
        # 训练集时间范围。格式: [起始日期, 结束日期]
        # 训练集用于模型学习数据分布。

        self.val_time_range = ["2025-09-01", "2026-06-30"]
        # 验证集时间范围。与训练集有重叠是为了覆盖 lookback_window 的缓冲。
        # 验证集用于训练过程中监控过拟合，选择最优模型。

        self.test_time_range = ["2026-04-01", "2026-08-01"]
        # 测试集时间范围。模型训练过程中不可见，用于最终评估。

        # --- 训练超参数 ---
        self.epochs = 30
        # 训练轮数。一个 epoch = 遍历一次训练数据。
        # 推荐: tokenizer 20~30, predictor 10~20。
        # 过多 epoch 会过拟合，过少则欠拟合。

        self.batch_size = 32
        # 每个训练步的样本数。
        # 影响: 越大训练越稳定，但显存占用越高。
        # 推荐: GPU 8~16GB 用 32~64；CPU 用 8~16。

        self.tokenizer_learning_rate = 2e-4
        # Tokenizer 学习率。控制 tokenizer 微调的步长。
        # 推荐: 1e-4 ~ 5e-4。太大则训练不稳定，太小则收敛慢。

        self.predictor_learning_rate = 4e-5
        # Predictor 学习率。控制主模型微调的步长。
        # 推荐: 1e-5 ~ 1e-4。通常比 tokenizer 学习率小一个数量级，
        # 因为 predictor 参数量大，需要更小的步长避免破坏预训练知识。

        self.adam_beta1 = 0.9
        # AdamW 优化器的一阶矩估计衰减率。
        # 控制梯度动量的衰减速度。0.9 是标准值，通常无需修改。

        self.adam_beta2 = 0.95
        # AdamW 优化器的二阶矩估计衰减率。
        # 控制梯度平方的衰减速度。0.95 是 LLM 常用值，略低于标准值 0.999。

        self.adam_weight_decay = 0.1
        # AdamW 权重衰减（L2 正则化）。
        # 防止过拟合。0.1 是 Transformer 常用值。

        self.accumulation_steps = 1
        # 梯度累积步数。模拟更大的 batch_size。
        # 等效 batch_size = batch_size × accumulation_steps。
        # 显存不足时增大此值，保持等效 batch_size 不变。

        self.seed = 42
        # 随机种子。影响数据采样、权重初始化、dropout 等随机过程。
        # 相同 seed + 相同配置 → 可复现结果。

        # --- 实验日志 ---
        self.use_comet = False
        # 是否使用 Comet ML 记录实验日志。
        # True 需要注册 https://www.comet.com 获取 API Key。
        # False 则跳过日志记录。

        self.comet_config = {
            "api_key": "",
            "project_name": "Kronos-Finetune",
            "workspace": ""
        }
        # Comet ML 配置。api_key 从环境变量获取更安全: os.getenv("COMET_API_KEY")

        # --- 模型保存 ---
        self.save_path = "./outputs/models"
        # 微调模型保存根目录。

        # --- 回测参数 ---
        self.backtest_n_symbol_hold = 50
        # 组合持仓股票数量。

        self.backtest_n_symbol_drop = 5
        # 每次调仓剔除的股票数量。

        self.backtest_hold_thresh = 5
        # 最小持仓天数。避免频繁换手。

        self.backtest_benchmark = "SH000300"
        # 回测基准指数。csi300 对应 SH000300。

    def get_predict_params(self):
        """获取预测所需的关键参数字典"""
        return {
            "stock_code": self.stock_code,
            "stock_name": self.stock_name,
            "lookback": self.lookback,
            "pred_len": self.pred_len,
            "model_name": self.model_name,
            "model_path": self.model_path,
            "tokenizer_path": self.tokenizer_path,
            "max_context": self.max_context,
            "clip": self.clip,
            "T": self.T,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "sample_count": self.sample_count,
            "device": self.device,
            "verbose": self.verbose,
        }

    def print_summary(self):
        """打印当前配置摘要"""
        p = self.get_predict_params()
        print("=" * 60)
        print(f"  {p['stock_name']}({p['stock_code']}) 预测配置")
        print("=" * 60)
        print(f"  模型: {p['model_name']}")
        print(f"  设备: {p['device']}")
        print(f"  回看窗口: {p['lookback']}")
        print(f"  预测步数: {p['pred_len']}")
        print(f"  最大上下文: {p['max_context']}")
        print(f"  裁剪范围: ±{p['clip']}")
        print(f"  温度(T): {p['T']}")
        print(f"  Top-K: {p['top_k']}")
        print(f"  Top-P: {p['top_p']}")
        print(f"  采样次数: {p['sample_count']}")
        print("=" * 60)


# 预设配置模板
PRESETS = {
    "conservative": {
        # 保守预测：结果最稳定，适合实际决策参考
        "T": 0.5,
        "top_p": 0.9,
        "top_k": 0,
        "sample_count": 15,
        "lookback": 400,
    },
    "balanced": {
        # 均衡预测：稳定性和多样性兼顾
        "T": 0.7,
        "top_p": 0.92,
        "top_k": 0,
        "sample_count": 10,
        "lookback": 300,
    },
    "exploratory": {
        # 探索性预测：更多样化，适合发现可能性
        "T": 1.0,
        "top_p": 0.95,
        "top_k": 0,
        "sample_count": 5,
        "lookback": 200,
    },
    "deterministic": {
        # 确定性预测：每次结果相同（需修改 sample_logits=False）
        "T": 0.3,
        "top_p": 0.85,
        "top_k": 0,
        "sample_count": 1,
        "lookback": 400,
    },
}


def apply_preset(config: PredictConfig, preset_name: str) -> PredictConfig:
    """将预设配置应用到 Config 对象"""
    if preset_name not in PRESETS:
        raise ValueError(f"未知预设: {preset_name}，可选: {list(PRESETS.keys())}")
    for key, value in PRESETS[preset_name].items():
        setattr(config, key, value)
    return config
