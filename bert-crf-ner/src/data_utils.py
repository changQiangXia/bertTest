"""
数据加载与预处理工具

核心问题：BERT分词器可能将一个汉字拆分成多个subword，需要处理token与标注的对齐

示例：
原始文本：  "李明去了北京大学"
原始标注：  B-PER I-PER O O B-ORG I-ORG I-ORG I-ORG
BERT分词： [CLS] 李 明 去 了 北 京 大 学 [SEP]
           ↑注意：中文BERT通常按字分词，但某些字符可能被拆分

处理策略：
- 只保留第一个subword的标注，其余subword标注设为-100（忽略）
- 或者将标注复制到所有subword

MSRA-NER数据集格式：
- tokens: ["李", "明", "去", "了", "北", "京", "大", "学"]
- ner_tags: [1, 2, 0, 0, 3, 4, 4, 4]  (数字对应BIO标签)
"""
import os
import sys
import io

# 设置UTF-8编码 (Windows兼容)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from datasets import load_dataset
import numpy as np


@dataclass
class NERExample:
    """NER样本数据结构"""
    tokens: List[str]        # 分词后的token列表
    labels: List[str]        # BIO标签列表
    text: str = ""           # 原始文本（可选）


class MSRANERDataset(Dataset):
    """
    MSRA-NER数据集封装类
    
    处理流程：
    1. 加载原始数据（Hugging Face datasets格式）
    2. 使用BERT tokenizer进行编码
    3. 处理token与label的对齐
    4. 转换为模型输入格式
    """
    
    # MSRA-NER的标签映射（数字ID -> BIO标签）
    ID_TO_LABEL = {
        0: "O",
        1: "B-PER", 2: "I-PER",
        3: "B-LOC", 4: "I-LOC",
        5: "B-ORG", 6: "I-ORG",
    }
    
    LABEL_TO_ID = {v: k for k, v in ID_TO_LABEL.items()}
    
    def __init__(
        self,
        data,
        tokenizer: BertTokenizer,
        max_length: int = 128,
        label_to_id: Optional[Dict[str, int]] = None
    ):
        """
        初始化数据集
        
        Args:
            data: Hugging Face datasets数据
            tokenizer: BERT tokenizer
            max_length: 最大序列长度
            label_to_id: 标签到ID的映射
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_to_id = label_to_id or self.LABEL_TO_ID
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        
        # 预处理所有样本
        self.processed_data = self._preprocess()
    
    def _preprocess(self) -> List[Dict]:
        """
        预处理所有样本
        
        核心逻辑：
        1. 将原始token拼接成文本
        2. 使用tokenizer编码，记录每个token对应的word索引
        3. 根据word索引将label映射到token
        4. 非首个subword的label设为-100（在计算loss时忽略）
        """
        processed = []
        
        for example in self.data:
            tokens = example["tokens"]
            # MSRA数据集使用ner_tags字段
            labels = [self.ID_TO_LABEL[tag] for tag in example["ner_tags"]]
            
            # 使用tokenizer的word_ids功能进行对齐
            # word_ids()返回每个token对应的原始word索引
            encoding = self.tokenizer(
                tokens,
                is_split_into_words=True,  # 表示输入已分词
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
            )
            
            # 获取word索引映射（兼容非fast tokenizer）
            try:
                word_ids = encoding.word_ids()
            except ValueError:
                # 对于非fast tokenizer，手动构建word_ids
                # 中文BERT按字分词，token数量 = word数量 + 2 ([CLS]和[SEP])
                # 考虑截断情况：最多 max_length-2 个words
                max_words = self.max_length - 2
                num_words = min(len(tokens), max_words)
                word_ids = [None]  # [CLS]
                for i in range(num_words):
                    word_ids.append(i)  # 每个字对应一个word
                word_ids.append(None)  # [SEP]
                # 填充到max_length
                while len(word_ids) < self.max_length:
                    word_ids.append(None)  # [PAD]
            
            # 构建token级别的标签
            label_ids = []
            previous_word_idx = None
            
            for word_idx in word_ids:
                if word_idx is None:
                    # 特殊token ([CLS], [SEP], [PAD]) 设为-100
                    label_ids.append(-100)
                elif word_idx >= len(labels):
                    # 超出标签范围，设为-100
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    # 新word的第一个token，使用对应label
                    label = labels[word_idx]
                    label_ids.append(self.label_to_id[label])
                else:
                    # 同一个word的后续subword，设为-100
                    # 策略：也可以设为与第一个subword相同的label
                    label_ids.append(-100)
                
                previous_word_idx = word_idx
            
            processed.append({
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "token_type_ids": encoding.get("token_type_ids", [0] * len(encoding["input_ids"])),
                "labels": label_ids,
                "tokens": tokens,
                "original_labels": labels,
            })
        
        return processed
    
    def __len__(self) -> int:
        return len(self.processed_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.processed_data[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(item["token_type_ids"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
        }


def load_msra_ner_data(cache_dir: str = "data") -> Tuple[List, List, List]:
    """
    加载MSRA-NER数据集（支持本地文件）
    
    Args:
        cache_dir: 数据缓存目录
    
    Returns:
        train_data, val_data, test_data
    """
    import os
    from datasets import Dataset, DatasetDict
    
    print("正在加载MSRA-NER数据集...")
    
    # 首先尝试从Hugging Face加载
    try:
        dataset = load_dataset("chinesenlpcorpus/msra_ner", cache_dir=cache_dir)
        train_data = dataset["train"]
        test_data = dataset["test"]
        print("从Hugging Face Hub加载数据成功")
    except Exception as e:
        print(f"从Hub加载失败: {e}")
        print("尝试从本地文件加载...")
        
        # 从本地文件加载
        data_dir = os.path.join(cache_dir, "msra_ner")
        train_file = os.path.join(data_dir, "train.txt")
        test_file = os.path.join(data_dir, "test.txt")
        
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"找不到训练数据文件: {train_file}")
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"找不到测试数据文件: {test_file}")
        
        # 解析本地数据文件
        def parse_conll_file(filepath):
            """解析CoNLL格式文件"""
            samples = []
            tokens = []
            tags = []
            
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line == "":
                        # 空行表示一个样本结束
                        if tokens:
                            samples.append({
                                "tokens": tokens,
                                "ner_tags": [int(tag) for tag in tags]
                            })
                            tokens = []
                            tags = []
                    else:
                        parts = line.split('\t')
                        if len(parts) == 2:
                            token, tag = parts
                            tokens.append(token)
                            # 将标签字符串转换为数字ID
                            tag_map = {"O": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3, "I-LOC": 4, "B-ORG": 5, "I-ORG": 6}
                            tags.append(tag_map.get(tag, 0))
            
            # 处理最后一个样本（如果没有以空行结尾）
            if tokens:
                samples.append({
                    "tokens": tokens,
                    "ner_tags": [int(tag) for tag in tags]
                })
            
            return samples
        
        print(f"正在解析训练文件: {train_file}")
        train_samples = parse_conll_file(train_file)
        print(f"正在解析测试文件: {test_file}")
        test_samples = parse_conll_file(test_file)
        
        train_data = Dataset.from_dict({k: [d[k] for d in train_samples] for k in train_samples[0]})
        test_data = Dataset.from_dict({k: [d[k] for d in test_samples] for k in test_samples[0]})
        print("本地数据加载成功")
    
    # MSRA没有验证集，从训练集划分10%作为验证集
    train_val_split = train_data.train_test_split(test_size=0.1, seed=42)
    train_data = train_val_split["train"]
    val_data = train_val_split["test"]
    
    print(f"数据集加载完成！")
    print(f"训练集: {len(train_data)} 条")
    print(f"验证集: {len(val_data)} 条")
    print(f"测试集: {len(test_data)} 条")
    
    # 打印一个样本示例
    example = train_data[0]
    print(f"\n样本示例:")
    print(f"Tokens: {example['tokens'][:20]}...")
    print(f"NER Tags: {example['ner_tags'][:20]}...")
    
    return train_data, val_data, test_data


def create_data_loaders(
    train_data,
    val_data,
    test_data,
    tokenizer: BertTokenizer,
    batch_size: int = 16,
    max_length: int = 128,
    label_to_id: Optional[Dict[str, int]] = None,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建DataLoader
    
    Args:
        train_data: 训练数据
        val_data: 验证数据
        test_data: 测试数据
        tokenizer: BERT tokenizer
        batch_size: 批次大小
        max_length: 最大序列长度
        label_to_id: 标签映射
        num_workers: 数据加载线程数
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # 创建Dataset
    train_dataset = MSRANERDataset(train_data, tokenizer, max_length, label_to_id)
    val_dataset = MSRANERDataset(val_data, tokenizer, max_length, label_to_id)
    test_dataset = MSRANERDataset(test_data, tokenizer, max_length, label_to_id)
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader


def align_predictions(
    predictions: List[List[int]],
    label_ids: torch.Tensor,
    id_to_label: Dict[int, str]
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    将模型预测和真实标签对齐，移除特殊token
    
    Args:
        predictions: 模型预测的label ID序列列表
        label_ids: 真实的label ID张量，-100表示忽略
        id_to_label: ID到标签的映射
    
    Returns:
        pred_labels_list, true_labels_list（都是字符串标签）
    """
    pred_labels_list = []
    true_labels_list = []
    
    for pred_ids, true_ids in zip(predictions, label_ids):
        pred_labels = []
        true_labels = []
        
        for pred_id, true_id in zip(pred_ids, true_ids):
            # 跳过-100（padding和特殊token）
            if true_id == -100:
                continue
            
            pred_labels.append(id_to_label.get(pred_id, "O"))
            true_labels.append(id_to_label.get(true_id.item(), "O"))
        
        pred_labels_list.append(pred_labels)
        true_labels_list.append(true_labels)
    
    return pred_labels_list, true_labels_list


def get_labels() -> List[str]:
    """返回所有BIO标签"""
    return [
        "O",
        "B-PER", "I-PER",
        "B-LOC", "I-LOC",
        "B-ORG", "I-ORG",
    ]


if __name__ == "__main__":
    # 测试数据加载
    from transformers import BertTokenizer
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    train_data, val_data, test_data = load_msra_ner_data()
    
    # 创建dataset测试
    dataset = MSRANERDataset(train_data, tokenizer, max_length=128)
    print(f"\nDataset长度: {len(dataset)}")
    
    # 取一个样本查看
    sample = dataset[0]
    print(f"\n处理后的样本:")
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Labels shape: {sample['labels'].shape}")
    print(f"Labels: {sample['labels']}")
