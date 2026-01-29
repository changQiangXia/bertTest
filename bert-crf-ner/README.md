# 中文BERT+CRF命名实体识别 (NER)

基于PyTorch和Hugging Face Transformers的中文命名实体识别项目，使用BERT+CRF模型在MSRA-NER数据集上进行训练。

## 项目特点

- **模型架构**：bert-base-chinese + 手动实现CRF层
- **支持实体**：PER（人物）、LOC（地点）、ORG（机构）
- **数据集**：MSRA-NER（支持本地文件加载）
- **标注格式**：BIO（Begin-Inside-Outside）
- **低显存优化**：支持4GB显存训练，已验证batch_size=1/2/4/8均可运行

## 任务说明

### 什么是命名实体识别（NER）？

命名实体识别是**序列标注任务**，与文本分类任务有本质区别：

| 对比维度 | 文本分类任务 | 序列标注任务（NER） |
|---------|-------------|-------------------|
| **输入** | 完整文本序列 | 文本序列 |
| **输出** | 单个类别标签 | 每个token一个标签 |
| **粒度** | 句子级 | token级 |
| **依赖关系** | 独立预测 | 标签间有依赖关系（如B-PER后不能接B-LOC） |

### BIO标注格式

```
北京[LOC]是中华人民共和国[ORG]的首都

原始文本：  北京    是    中华人民共和国    的    首都
BIO标注：  B-LOC   O    B-ORG I-ORG I-ORG I-ORG  O   O
```

- **B-**：实体开始（Begin）
- **I-**：实体内部（Inside）
- **O**：非实体（Outside）

## 环境要求

- Python >= 3.7
- PyTorch >= 1.13.0
- transformers >= 4.21.0（Python 3.7兼容版本）
- datasets >= 2.0.0
- seqeval
- accelerate

## 安装依赖

```bash
conda activate hciaML
pip install -r requirements.txt
```

## 数据集准备

项目使用MSRA-NER数据集，支持两种方式加载：

### 方式1：本地文件（推荐）

将MSRA-NER数据集文件放入 `data/msra_ner/` 目录：
```
data/msra_ner/
├── train.txt    # 训练集
└── test.txt     # 测试集
```

数据格式（CoNLL格式）：
```
北 B-LOC
京 I-LOC
的 O
... 
```

### 方式2：HuggingFace Hub（需要网络访问）

设置镜像站（国内访问）：
```bash
set HF_ENDPOINT=https://hf-mirror.com  # Windows
export HF_ENDPOINT=https://hf-mirror.com  # Linux/Mac
```

## 快速开始

### 训练模型

```bash
# 正常训练（推荐，batch_size=2，平衡速度与显存）
python src/train.py --batch_size 2 --gradient_accumulation_steps 8 --fp16

# 4GB显存极限配置
python src/train.py --batch_size 8 --gradient_accumulation_steps 2 --fp16

# 完整参数示例
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
    --output_dir outputs
```

### 评估模型

```bash
python src/evaluate.py --model_path outputs/best_model
```

### 推理测试

```bash
# 单条文本
python src/predict.py --model_path outputs/best_model --text "北京是中国的首都"

# 交互模式
python src/predict.py --model_path outputs/best_model

# 批量推理
python src/predict.py --model_path outputs/best_model --input_file test_texts.txt --output_file results.txt
```

## 显存优化技巧

本项目实现了以下显存优化技巧：

1. **梯度累积（Gradient Accumulation）**：小batch_size配合大accumulation_steps，保持有效batch_size=16
2. **混合精度训练（FP16）**：使用torch.cuda.amp自动混合精度，节省约40%显存
3. **梯度裁剪（Gradient Clipping）**：防止梯度爆炸，默认max_grad_norm=1.0
4. **冻结BERT层**：可选冻结前N层BERT参数，只训练顶层+CRF

### 4GB显存配置推荐

| 配置 | batch_size | gradient_accumulation_steps | 显存占用 | 推荐度 |
|-----|-----------|---------------------------|---------|-------|
| 稳定训练 | 2 | 8 | ~2.5GB | ⭐⭐⭐ |
| 速度优先 | 4 | 4 | ~3GB | ⭐⭐ |
| 极限测试 | 8 | 2 | ~3.5GB | ⭐ |

详见 [4GB_GPU_GUIDE.md](4GB_GPU_GUIDE.md)

## 项目结构

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
│       ├── train.txt
│       ├── test.txt
│       └── README.md
├── outputs/               # 训练输出（模型、日志）
├── README.md              # 本文件
├── QUICKSTART.md          # 快速开始指南
├── 4GB_GPU_GUIDE.md       # 4GB显存训练指南
└── requirements.txt       # 依赖列表
```

## CRF层的作用

CRF（条件随机场）层在序列标注中的核心作用：

1. **建模标签依赖**：学习标签间的转移规则（如B-PER后只能接I-PER或O）
2. **全局最优解码**：使用Viterbi算法寻找全局最优标签序列
3. **约束非法序列**：避免产生如"B-PER I-LOC"这样的非法标注

本项目手动实现了CRF层的前向算法（计算partition function）和Viterbi解码，支持：
- 转移矩阵学习
- 发射分数计算
- 考虑padding mask的序列解码

## 技术问题解决记录

1. **本地数据集加载**：HuggingFace Hub 401错误时自动回退到本地文件加载
2. **Tokenizer兼容性**：非fast tokenizer没有word_ids()，手动构建word_id映射
3. **CRF实现修复**：mask类型转换为bool，处理-100标签索引过滤
4. **编码问题**：Windows UTF-8编码设置，避免中文乱码
5. **显存优化**：FP16混合精度 + 层冻结 + 梯度累积

## 参考

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [ChineseNLPCorpus](https://github.com/brightmart/nlp_chinese_corpus)
- [MSRA-NER Dataset](https://huggingface.co/datasets/chinesenlpcorpus/msra_ner)
