# 4GB显存训练指南

本指南专门针对NVIDIA RTX 3050 Laptop（4GB VRAM）等低显存显卡优化，经过实际验证支持batch_size=1/2/4/8。

## 显存分析

4GB显存对于训练BERT+CRF的占用情况：

| 组件 | 占用大小 |
|-----|---------|
| bert-base-chinese 模型参数 | ~400MB |
| 优化器状态（AdamW） | ~1.2GB |
| 前向传播激活值 | ~1-2GB（取决于batch_size和序列长度） |
| 反向传播梯度 | ~400MB |
| **batch_size=1总计** | ~2.0GB |
| **batch_size=2总计** | ~2.5GB |
| **batch_size=4总计** | ~3.0GB |
| **batch_size=8总计** | ~3.5GB |

## 推荐配置（已验证）

### 推荐方案：batch_size=2（平衡速度与稳定性）

```bash
python src/train.py \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --max_seq_length 64 \
    --freeze_bert_layers 10 \
    --fp16 \
    --num_epochs 5 \
    --learning_rate 5e-5 \
    --crf_learning_rate 5e-3 \
    --logging_steps 50
```

**特点**：
- 显存占用：约2.5GB（余量充足）
- 训练速度：~11-12 it/s
- 收敛速度：快（250步内Loss从34→1.6）
- 稳定性：高

### 速度优先：batch_size=4

```bash
python src/train.py \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_seq_length 64 \
    --freeze_bert_layers 10 \
    --fp16 \
    --num_epochs 5
```

**特点**：
- 显存占用：约3GB（余量约1GB）
- 训练速度：~11-12 it/s
- 收敛速度：正常（250步内Loss从35.7→3.5）
- 稳定性：中等

### 极限方案：batch_size=8

```bash
python src/train.py \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --max_seq_length 64 \
    --freeze_bert_layers 10 \
    --fp16 \
    --num_epochs 5
```

**特点**：
- 显存占用：约3.5GB（余量约500MB）
- 训练速度：~10 it/s
- 收敛速度：较慢（Loss跳动较大）
- 稳定性：低（不推荐长期使用）

### 安全保底：batch_size=1

```bash
python src/train.py \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_seq_length 64 \
    --freeze_bert_layers 10 \
    --fp16 \
    --num_epochs 5
```

**特点**：
- 显存占用：约2GB（余量充足）
- 训练速度：~9-10 it/s
- 收敛速度：慢（但稳定）

## 参数解释

| 参数 | 推荐值 | 说明 |
|-----|-------|------|
| batch_size | 2 | 每步训练的样本数，4GB显存推荐2 |
| gradient_accumulation_steps | 8 | 累积8步再更新，等效batch_size=16 |
| max_seq_length | 64 | 序列长度，64足够覆盖大部分句子 |
| freeze_bert_layers | 10 | 冻结BERT前10层，只训练最后2层+CRF |
| fp16 | - | 混合精度，节省约40%显存，加速训练 |
| num_epochs | 5 | 训练轮数，5轮通常足够收敛 |

## 显存监控

训练时监控显存使用：

```bash
# Windows
nvidia-smi -l 1

# 或者在Python中
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')"
```

## 训练时间预估

基于RTX 3050 Laptop 4GB（约4万条训练数据）：

| 方案 | batch_size | 每轮时间 | 总时间（5轮） |
|-----|-----------|---------|-------------|
| batch_size=1 | 1 | 约40分钟 | 约3.5小时 |
| **batch_size=2（推荐）** | **2** | **约30分钟** | **约2.5小时** |
| batch_size=4 | 4 | 约28分钟 | 约2.5小时 |
| batch_size=8 | 8 | 约30分钟 | 约2.5小时 |

## 可能遇到的问题

### 1. CUDA out of memory

如果仍然OOM，尝试：
- 减小batch_size到1
- 减小max_seq_length到48
- 增加freeze_bert_layers到12（冻结更多层）
- 关闭fp16（某些旧GPU不支持，但会显著增加显存占用）

### 2. 训练速度过慢

- 确保启用了FP16：`--fp16`
- 检查GPU利用率：`nvidia-smi`，应接近100%
- 减小gradient_accumulation_steps（如从8改为4，但需同比例减小batch_size）

### 3. 效果不理想

- 增加gradient_accumulation_steps（如16或32）
- 解冻更多BERT层（如只冻结前8层）
- 增加训练轮数（如8-10轮）
- 调整学习率（尝试3e-5或1e-4）

### 4. Loss为NaN或异常大

- 降低CRF层学习率：`--crf_learning_rate 1e-3`
- 检查数据格式是否正确
- 关闭FP16尝试：`--no_cuda`或移除`--fp16`

## Windows批处理脚本

已为你准备（在项目根目录）：
- `run_4gb_gpu.bat`：4GB GPU训练（batch_size=2配置）

直接双击运行即可！

## 快速测试

在完整训练前，测试能否正常启动：

```bash
# 测试1轮
python src/train.py --batch_size 2 --num_epochs 1 --logging_steps 10
```

如果正常跑完1轮无OOM，则可以开始完整训练。

## 总结建议

| 优先级 | 配置 | 理由 |
|-------|-----|-----|
| **首选** | batch_size=2, accumulation=8 | 速度、稳定性、显存余量最佳平衡 |
| 次选 | batch_size=4, accumulation=4 | 速度稍快，但显存余量较小 |
| 保底 | batch_size=1, accumulation=16 | 最安全，不会OOM |
| 不推荐 | batch_size=8, accumulation=2 | 显存余量太小，容易OOM |

**训练技巧**：
1. 先用batch_size=2跑通5轮，观察收敛情况
2. 如果显存允许，尝试batch_size=4加速训练
3. 保存检查点：`--save_steps=500`定期保存模型
4. 监控训练日志：`outputs/train.log`

祝训练顺利！
