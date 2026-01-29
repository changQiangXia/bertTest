"""
模型推理脚本

提供两种推理方式：
1. 单条文本推理：输入一段文本，输出实体标注结果
2. 批量推理：从文件读取文本列表，批量处理

推理流程：
1. 文本分词（使用BERT tokenizer）
2. BERT编码提取特征
3. CRF层Viterbi解码得到最优标签序列
4. 将标签序列转换为实体标注结果
5. 格式化输出（如"北京[LOC]是中国的首都"）
"""
import os
import sys
import argparse
import json
import torch
from transformers import BertTokenizer
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.dirname(__file__))

from model import BertCRFForNER


class NERPredictor:
    """
    NER预测器封装类
    
    提供简洁的API用于实体识别推理
    """
    
    def __init__(self, model_path: str, device: str = None):
        """
        初始化预测器
        
        Args:
            model_path: 模型目录路径
            device: 计算设备（"cuda"或"cpu"）
        """
        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"使用设备: {self.device}")
        
        # 加载tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        
        # 加载标签映射
        label_map_path = os.path.join(model_path, "label_map.json")
        if os.path.exists(label_map_path):
            with open(label_map_path, "r", encoding="utf-8") as f:
                label_map = json.load(f)
            self.label_to_id = label_map["label_to_id"]
            self.id_to_label = {int(k): v for k, v in label_map["id_to_label"].items()}
        else:
            # 使用默认标签
            labels = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]
            self.label_to_id = {label: i for i, label in enumerate(labels)}
            self.id_to_label = {i: label for i, label in enumerate(labels)}
        
        self.num_labels = len(self.label_to_id)
        
        # 加载模型
        self.model = BertCRFForNER.from_pretrained(model_path, num_labels=self.num_labels)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"模型加载完成，支持标签: {list(self.label_to_id.keys())}")
    
    def predict(self, text: str, max_length: int = 128) -> List[Tuple[str, str]]:
        """
        对单条文本进行实体识别
        
        Args:
            text: 输入文本
            max_length: 最大序列长度
        
        Returns:
            实体列表，每个元素为(word, entity_type)元组
        """
        # 分词（中文BERT通常按字分词）
        tokens = list(text)  # 按字符分割
        
        # 使用tokenizer编码
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # 获取word到token的映射
        word_ids = encoding.word_ids()
        
        # 推理
        with torch.no_grad():
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            
            # 获取发射分数
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            emissions = outputs[0]
            
            # Viterbi解码
            predictions = self.model.decode(emissions, attention_mask.bool())
        
        # 解析预测结果
        pred_ids = predictions[0]  # 取第一个（也是唯一一个）样本
        
        # 构建token级别的标签序列
        token_labels = []
        for i, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            if i < len(pred_ids):
                token_labels.append(self.id_to_label.get(pred_ids[i], "O"))
            else:
                token_labels.append("O")
        
        # 合并实体
        entities = self._extract_entities(tokens, token_labels)
        
        return entities
    
    def _extract_entities(self, tokens: List[str], labels: List[str]) -> List[Tuple[str, str]]:
        """
        从标签序列中提取实体
        
        Args:
            tokens: token列表
            labels: 标签列表
        
        Returns:
            实体列表，每个元素为(entity_text, entity_type)
        """
        entities = []
        current_entity = []
        current_type = None
        
        for token, label in zip(tokens, labels):
            if label.startswith("B-"):
                # 新实体开始，保存之前的实体
                if current_entity:
                    entities.append(("".join(current_entity), current_type))
                current_entity = [token]
                current_type = label[2:]
            elif label.startswith("I-"):
                # 实体内部
                entity_type = label[2:]
                if current_type == entity_type:
                    current_entity.append(token)
                else:
                    # I-标签类型与当前实体类型不匹配，视为新实体开始
                    if current_entity:
                        entities.append(("".join(current_entity), current_type))
                    current_entity = [token]
                    current_type = entity_type
            else:
                # O标签，实体结束
                if current_entity:
                    entities.append(("".join(current_entity), current_type))
                    current_entity = []
                    current_type = None
        
        # 处理最后一个实体
        if current_entity:
            entities.append(("".join(current_entity), current_type))
        
        return entities
    
    def format_result(self, text: str, entities: List[Tuple[str, str]]) -> str:
        """
        格式化输出结果（如"北京[LOC]是中国的首都"）
        
        Args:
            text: 原始文本
            entities: 实体列表
        
        Returns:
            格式化后的字符串
        """
        # 按实体在文本中出现的顺序排序
        # 这里简化处理，假设实体按顺序提取
        result = text
        offset = 0
        
        for entity_text, entity_type in entities:
            # 找到实体位置
            pos = result.find(entity_text, offset)
            if pos != -1:
                # 插入标记
                marked = f"{entity_text}[{entity_type}]"
                result = result[:pos] + marked + result[pos + len(entity_text):]
                offset = pos + len(marked)
        
        return result
    
    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[List[Tuple[str, str]]]:
        """
        批量推理
        
        Args:
            texts: 文本列表
            batch_size: 批次大小
        
        Returns:
            每个文本的实体列表
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_results = [self.predict(text) for text in batch_texts]
            results.extend(batch_results)
        return results


def main():
    parser = argparse.ArgumentParser(description="BERT+CRF中文NER推理")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型路径")
    parser.add_argument("--text", type=str, default=None,
                        help="输入文本（单条推理）")
    parser.add_argument("--input_file", type=str, default=None,
                        help="输入文件路径（每行一个文本，批量推理）")
    parser.add_argument("--output_file", type=str, default=None,
                        help="输出文件路径")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="最大序列长度")
    parser.add_argument("--no_cuda", action="store_true",
                        help="不使用CUDA")
    
    args = parser.parse_args()
    
    # 初始化预测器
    device = "cpu" if args.no_cuda else None
    predictor = NERPredictor(args.model_path, device=device)
    
    # 单条推理
    if args.text:
        print(f"\n输入文本: {args.text}")
        entities = predictor.predict(args.text, args.max_seq_length)
        formatted = predictor.format_result(args.text, entities)
        
        print(f"\n识别结果:")
        print(f"格式化输出: {formatted}")
        print(f"\n实体列表:")
        for entity, entity_type in entities:
            print(f"  - {entity}: {entity_type}")
    
    # 批量推理
    elif args.input_file:
        if not os.path.exists(args.input_file):
            print(f"错误: 文件不存在 {args.input_file}")
            return
        
        print(f"\n批量推理: {args.input_file}")
        
        # 读取输入文件
        with open(args.input_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        
        print(f"共 {len(texts)} 条文本")
        
        # 批量预测
        results = predictor.predict_batch(texts)
        
        # 输出结果
        output_lines = []
        for text, entities in zip(texts, results):
            formatted = predictor.format_result(text, entities)
            entity_str = ", ".join([f"{e}[{t}]" for e, t in entities])
            output_lines.append(f"文本: {text}")
            output_lines.append(f"标注: {formatted}")
            output_lines.append(f"实体: {entity_str}")
            output_lines.append("")
        
        # 保存或打印结果
        if args.output_file:
            os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
            with open(args.output_file, "w", encoding="utf-8") as f:
                f.write("\n".join(output_lines))
            print(f"\n结果已保存到: {args.output_file}")
        else:
            print("\n" + "\n".join(output_lines))
    
    else:
        # 交互模式
        print("\n进入交互模式（输入'quit'退出）:")
        while True:
            text = input("\n请输入文本: ").strip()
            if text.lower() in ["quit", "exit", "q"]:
                break
            if not text:
                continue
            
            entities = predictor.predict(text, args.max_seq_length)
            formatted = predictor.format_result(text, entities)
            
            print(f"格式化输出: {formatted}")
            print(f"实体列表:")
            for entity, entity_type in entities:
                print(f"  - {entity}: {entity_type}")


if __name__ == "__main__":
    main()
