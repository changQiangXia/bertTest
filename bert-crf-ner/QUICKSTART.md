# BERT+CRF 中文NER - 快速开始指南

## 环境准备

确保已激活conda环境：

```bash
conda activate hciaML
```

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

## 2. 准备数据集

将MSRA-NER数据集放入 `data/msra_ner/` 目录：
```
data/msra_ner/
├── train.txt    # 训练集
└── test.txt     # 测试集
```

数据格式（CoNLL格式，每行一个词和标签，空行分隔句子）：
```
北 B-LOC
京 I-LOC
是 O
中 B-ORG
国 I-ORG
... 
```

## 3. 训练模型

### 推荐配置（4GB-8GB显存）

```bash
python src/train.py --batch_size 2 --gradient_accumulation_steps 8 --fp16
```

### 完整训练参数（5轮，约2-3小时）

```bash
python src/train.py \
    --model_name bert-base-chinese \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_epochs 5 \
    --learning_rate 5e-5 \
    --crf_learning_rate 5e-3 \
    --max_seq_length 64 \
    --fp16 \
    --freeze_bert_layers 10 \
    --warmup_ratio 0.1 \
    --logging_steps 50 \
    --output_dir outputs
```

### 显存与batch_size对照表

| 显存大小 | batch_size | gradient_accumulation_steps | 其他参数 |
|---------|-----------|---------------------------|---------|
| 4GB | 2 | 8 | freeze_bert_layers=10, fp16 |
| 4GB | 4 | 4 | freeze_bert_layers=10, fp16 |
| 6GB | 4 | 4 | freeze_bert_layers=6, fp16 |
| 8GB+ | 8 | 2 | freeze_bert_layers=6, fp16 |
| 16GB+ | 16 | 1 | 无需冻结 |

训练完成后，最佳模型将保存在 `outputs/best_model/`。

## 4. 评估模型

```bash
python src/evaluate.py --model_path outputs/best_model
```

## 5. 推理测试

### 单条文本

```bash
python src/predict.py --model_path outputs/best_model --text "北京是中国的首都"
```

### 交互模式

```bash
python src/predict.py --model_path outputs/best_model
```

### 批量推理

```bash
python src/predict.py --model_path outputs/best_model --input_file test_texts.txt --output_file results.txt
```

## 项目结构说明

```
bert-crf-ner/
├── src/
│   ├── train.py           # 训练脚本（支持梯度累积、FP16、层冻结）
│   ├── model.py           # BERT+CRF模型（手动实现CRF层）
│   ├── data_utils.py      # 数据加载（支持本地文件+Hub下载）
│   ├── evaluate.py        # 评估脚本
│   └── predict.py         # 推理脚本
├── data/
│   └── msra_ner/          # MSRA-NER数据集
├── outputs/               # 训练输出（模型、日志）
└── requirements.txt       # 依赖列表
```

## 常见问题

### Q1: 显存不足怎么办？

使用以下参数组合（从大到小尝试）：
```bash
# 方案1: 减小batch_size
python src/train.py --batch_size 1 --gradient_accumulation_steps 16 --fp16

# 方案2: 增加冻结层数
python src/train.py --batch_size 2 --gradient_accumulation_steps 8 --freeze_bert_layers 10 --fp16

# 方案3: 缩短序列长度
python src/train.py --batch_size 2 --max_seq_length 48 --fp16

# 方案4: 完全冻结BERT（仅训练CRF）
python src/train.py --batch_size 4 --freeze_bert --fp16
```

### Q2: 如何调整学习率？

BERT层和CRF层使用不同学习率：
```bash
python src/train.py \
    --learning_rate 5e-5 \      # BERT层学习率（较小）
    --crf_learning_rate 5e-3    # CRF层学习率（较大）
```

### Q3: 数据集加载失败？

确保数据文件路径正确：
```
data/msra_ner/train.txt
data/msra_ner/test.txt
```

或使用HuggingFace Hub（设置镜像）：
```bash
set HF_ENDPOINT=https://hf-mirror.com  # Windows
python src/train.py
```

### Q4: 训练速度慢？

- 确保使用GPU：`nvidia-smi` 查看GPU利用率
- 启用FP16：`--fp16` 可加速2-3倍
- 增大batch_size（如果显存允许）
- 减小gradient_accumulation_steps（如从8改为4）

### Q5: 如何恢复训练？

当前版本不支持自动恢复，可以手动修改train.py加载检查点：
```python
# 在模型初始化后添加
model.load_state_dict(torch.load("outputs/checkpoint-1000/pytorch_model.bin"))
```

## 训练监控

查看训练日志：
```bash
tail -f outputs/train.log
```

监控显存使用：
```bash
nvidia-smi -l 1
```
