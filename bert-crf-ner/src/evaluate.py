"""
模型评估脚本

评估指标说明：
- Token-level F1: 每个token的预测准确性
- Entity-level F1: 整个实体的识别准确性（更严格）

使用seqeval库计算实体级指标：
- Precision: 正确识别的实体数 / 识别的实体总数
- Recall: 正确识别的实体数 / 真实的实体总数
- F1-score: Precision和Recall的调和平均

实体匹配规则：
- 只有当实体的边界和类型都正确时才算正确
- 例如：真实是"B-PER I-PER"，预测是"B-LOC I-LOC" -> 错误
- 例如：真实是"B-PER I-PER"，预测是"B-PER I-PER" -> 正确
"""
import os
import sys
import argparse
import json
import torch
from transformers import BertTokenizer
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from model import BertCRFForNER
from data_utils import load_msra_ner_data, create_data_loaders, align_predictions
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score


def evaluate_model(args):
    """评估模型"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"使用设备: {device}")
    
    # 加载tokenizer
    print("加载模型和tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    
    # 加载标签映射
    label_map_path = os.path.join(args.model_path, "label_map.json")
    if os.path.exists(label_map_path):
        with open(label_map_path, "r", encoding="utf-8") as f:
            label_map = json.load(f)
        label_to_id = label_map["label_to_id"]
        id_to_label = {int(k): v for k, v in label_map["id_to_label"].items()}
    else:
        # 使用默认标签
        labels = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]
        label_to_id = {label: i for i, label in enumerate(labels)}
        id_to_label = {i: label for i, label in enumerate(labels)}
    
    num_labels = len(label_to_id)
    
    # 加载模型
    model = BertCRFForNER.from_pretrained(args.model_path, num_labels=num_labels)
    model.to(device)
    model.eval()
    
    print(f"标签数量: {num_labels}")
    print(f"标签映射: {id_to_label}")
    
    # 加载测试数据
    print("加载测试数据...")
    _, _, test_data = load_msra_ner_data(args.data_dir)
    
    # 创建测试数据加载器
    _, _, test_loader = create_data_loaders(
        [], [], test_data,
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_seq_length,
        label_to_id=label_to_id,
        num_workers=0,
    )
    
    # 评估
    print("开始评估...")
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # 获取发射分数
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            emissions = outputs[0]
            
            # Viterbi解码
            predictions = model.decode(emissions, attention_mask.bool())
            
            # 对齐预测和标签
            pred_list, true_list = align_predictions(predictions, labels, id_to_label)
            all_predictions.extend(pred_list)
            all_labels.extend(true_list)
    
    # 计算指标
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    
    print("\n" + "="*50)
    print("评估结果")
    print("="*50)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\n详细分类报告:")
    print(classification_report(all_labels, all_predictions))
    
    # 保存结果
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1-Score: {f1:.4f}\n\n")
            f.write(classification_report(all_labels, all_predictions))
        print(f"\n结果已保存到: {args.output_file}")


def main():
    parser = argparse.ArgumentParser(description="评估BERT+CRF中文NER模型")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型路径（包含pytorch_model.bin和config.json）")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="数据缓存目录")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批次大小")
    parser.add_argument("--no_cuda", action="store_true",
                        help="不使用CUDA")
    parser.add_argument("--output_file", type=str, default="outputs/eval_results.txt",
                        help="结果输出文件")
    
    args = parser.parse_args()
    evaluate_model(args)


if __name__ == "__main__":
    main()
