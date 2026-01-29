"""
配置文件：定义模型、训练、数据相关的所有超参数
"""
import os
from dataclasses import dataclass
from typing import List


@dataclass
class ModelConfig:
    """模型配置类"""
    # 预训练模型名称（Hugging Face模型仓库）
    bert_model_name: str = "bert-base-chinese"
    
    # 模型保存路径
    model_save_path: str = "outputs/best_model"
    
    # 最大序列长度（BERT输入限制为512，中文NER通常128足够）
    max_seq_length: int = 128
    
    # dropout率
    dropout_rate: float = 0.1
    
    # 是否冻结BERT参数（用于微调时节省显存）
    freeze_bert: bool = False
    
    # 冻结BERT前N层（0表示不冻结，12表示冻结全部）
    freeze_bert_layers: int = 0


@dataclass
class DataConfig:
    """数据配置类"""
    # 数据集名称（Hugging Face datasets格式）
    dataset_name: str = "chinesenlpcorpus/msra_ner"
    
    # 数据缓存目录
    data_cache_dir: str = "data"
    
    # 标签列表（BIO格式）
    # B-XX: 实体开始, I-XX: 实体内部, O: 非实体
    labels: List[str] = None
    
    def __post_init__(self):
        # 定义所有可能的标签
        # PER: 人名, LOC: 地名, ORG: 机构名
        self.labels = [
            "O",           # 非实体
            "B-PER", "I-PER",  # 人名
            "B-LOC", "I-LOC",  # 地名
            "B-ORG", "I-ORG",  # 机构名
        ]
    
    @property
    def num_labels(self) -> int:
        """返回标签数量"""
        return len(self.labels)
    
    @property
    def label2id(self) -> dict:
        """标签到ID的映射"""
        return {label: i for i, label in enumerate(self.labels)}
    
    @property
    def id2label(self) -> dict:
        """ID到标签的映射"""
        return {i: label for i, label in enumerate(self.labels)}


@dataclass
class TrainingConfig:
    """训练配置类"""
    # 批次大小（根据显存调整，16GB显存可设置16-32）
    batch_size: int = 16
    
    # 训练轮数
    num_epochs: int = 5
    
    # 学习率（BERT微调通常使用较小学习率，如2e-5到5e-5）
    learning_rate: float = 3e-5
    
    # CRF层学习率（通常比BERT高一些）
    crf_learning_rate: float = 1e-3
    
    # 权重衰减（L2正则化）
    weight_decay: float = 0.01
    
    #  warmup比例（前N%步线性增加学习率）
    warmup_ratio: float = 0.1
    
    # 梯度累积步数（用于显存不足时模拟大batch）
    # 例如：batch_size=4, accumulation_steps=4 -> 等效batch_size=16
    gradient_accumulation_steps: int = 1
    
    # 最大梯度范数（梯度裁剪，防止梯度爆炸）
    max_grad_norm: float = 1.0
    
    # 是否使用混合精度训练（FP16，可节省显存并加速）
    fp16: bool = True
    
    # 每多少步记录日志
    logging_steps: int = 100
    
    # 每多少步保存模型检查点
    save_steps: int = 500
    
    # 评估策略: "steps" 或 "epoch"
    evaluation_strategy: str = "epoch"
    
    # 保存策略: "steps" 或 "epoch"
    save_strategy: str = "epoch"
    
    # 早停耐心值（验证集F1不提升多少轮后停止）
    early_stopping_patience: int = 3
    
    # 随机种子（保证可复现）
    seed: int = 42
    
    # 训练设备（"cuda", "cpu", 或 None自动检测）
    device: str = None
    
    def __post_init__(self):
        if self.device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    """全局配置类"""
    model: ModelConfig = None
    data: DataConfig = None
    training: TrainingConfig = None
    
    def __init__(self):
        self.model = ModelConfig()
        self.data = DataConfig()
        self.training = TrainingConfig()


# 全局配置实例
config = Config()
