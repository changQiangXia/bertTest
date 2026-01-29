"""
BERT+CRF模型训练脚本

训练流程：
1. 加载数据和预处理
2. 构建BERT+CRF模型
3. 定义优化器和学习率调度器
4. 训练循环：
   - 前向传播计算CRF损失
   - 反向传播更新参数
   - 验证集评估F1分数
   - 保存最佳模型
5. 支持显存优化技巧：梯度累积、混合精度、梯度裁剪

显存优化技巧说明：
1. 梯度累积：用小batch_size模拟大batch_size
   - 例如：batch_size=4, accumulation_steps=4 -> 等效batch_size=16
   - 每accumulation_steps次前向传播后才进行一次参数更新

2. 混合精度训练（FP16）：
   - 使用半精度浮点数（16位）代替全精度（32位）
   - 可节省约50%显存，且通常不损失精度
   - PyTorch使用torch.cuda.amp自动管理

3. 梯度裁剪：
   - 限制梯度的最大范数，防止梯度爆炸
   - 设置max_grad_norm=1.0
"""
import os
import sys
import io

# 设置UTF-8编码 (Windows兼容)
if sys.platform == 'win32':
    import ctypes
    # 设置Windows控制台为UTF-8模式
    ctypes.windll.kernel32.SetConsoleCP(65001)
    ctypes.windll.kernel32.SetConsoleOutputCP(65001)
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import argparse
import logging
from typing import Dict, Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import (
    BertTokenizer,
    BertConfig,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
import numpy as np

# 添加src到路径
sys.path.insert(0, os.path.dirname(__file__))

from model import BertCRFForNER
from data_utils import load_msra_ner_data, create_data_loaders, align_predictions, get_labels
from seqeval.metrics import classification_report, f1_score


def setup_logging(output_dir: str) -> logging.Logger:
    """设置日志"""
    os.makedirs(output_dir, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # 清除已有处理器
    if logger.handlers:
        logger.handlers.clear()
    
    # 文件处理器 (使用io.open确保UTF-8编码)
    log_file = os.path.join(output_dir, "train.log")
    file_stream = io.open(log_file, 'a', encoding='utf-8')
    fh = logging.StreamHandler(file_stream)
    fh.setLevel(logging.INFO)
    
    # 控制台处理器 (UTF-8编码)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    # 格式化
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def evaluate(
    model: BertCRFForNER,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    id_to_label: Dict[int, str],
    logger: logging.Logger
) -> Tuple[float, Dict]:
    """
    在验证集上评估模型
    
    使用seqeval库计算实体级F1分数（不是token级）
    
    Args:
        model: BERT+CRF模型
        dataloader: 验证数据加载器
        device: 计算设备
        id_to_label: ID到标签的映射
        logger: 日志记录器
    
    Returns:
        f1_score, report
    """
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # 将数据移到设备
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs[0]
            emissions = outputs[1]
            
            total_loss += loss.item()
            
            # Viterbi解码
            predictions = model.decode(emissions, attention_mask.bool())
            
            # 对齐预测和标签
            pred_list, true_list = align_predictions(predictions, labels, id_to_label)
            all_predictions.extend(pred_list)
            all_labels.extend(true_list)
    
    # 计算实体级F1
    f1 = f1_score(all_labels, all_predictions)
    report = classification_report(all_labels, all_predictions)
    avg_loss = total_loss / len(dataloader)
    
    return f1, report, avg_loss


def train(args):
    """主训练函数"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    
    # 如果在CPU上运行，禁用fp16
    if device.type == "cpu" and args.fp16:
        args.fp16 = False
        print("警告: CPU模式下自动禁用FP16混合精度训练")
    
    # 设置日志
    logger = setup_logging(args.output_dir)
    logger.info(f"使用设备: {device}")
    logger.info(f"训练参数: {args}")
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 加载tokenizer
    logger.info("加载BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    
    # 获取标签
    labels = get_labels()
    label_to_id = {label: i for i, label in enumerate(labels)}
    id_to_label = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)
    
    logger.info(f"标签列表: {labels}")
    logger.info(f"标签数量: {num_labels}")
    
    # 加载数据
    logger.info("加载数据集...")
    train_data, val_data, test_data = load_msra_ner_data(args.data_dir)
    
    # 创建数据加载器
    logger.info("创建数据加载器...")
    train_loader, val_loader, _ = create_data_loaders(
        train_data,
        val_data,
        test_data,
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_seq_length,
        label_to_id=label_to_id,
        num_workers=args.num_workers,
    )
    
    # 加载模型
    logger.info("初始化BERT+CRF模型...")
    bert_config = BertConfig.from_pretrained(args.model_name)
    model = BertCRFForNER.from_pretrained(
        args.model_name,
        config=bert_config,
        num_labels=num_labels
    )
    model.to(device)
    
    # 冻结BERT参数（可选）
    if args.freeze_bert:
        logger.info("冻结BERT参数...")
        for param in model.bert.parameters():
            param.requires_grad = False
    
    # 只冻结前N层
    if args.freeze_bert_layers > 0:
        logger.info(f"冻结BERT前{args.freeze_bert_layers}层...")
        for layer in model.bert.encoder.layer[:args.freeze_bert_layers]:
            for param in layer.parameters():
                param.requires_grad = False
    
    # 可训练参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"总参数: {total_params:,}, 可训练参数: {trainable_params:,}")
    
    # 优化器
    # 使用不同的学习率：BERT使用较小学习率，CRF层使用较大学习率
    bert_params = []
    crf_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'transitions' in name or 'classifier' in name:
                crf_params.append(param)
            else:
                bert_params.append(param)
    
    optimizer_grouped_parameters = [
        {'params': bert_params, 'lr': args.learning_rate},
        {'params': crf_params, 'lr': args.crf_learning_rate}
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, weight_decay=args.weight_decay)
    
    # 学习率调度器
    total_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    logger.info(f"总训练步数: {total_steps}, Warmup步数: {warmup_steps}")
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if args.fp16 and torch.cuda.is_available() else None
    
    # 训练循环
    logger.info("开始训练...")
    best_f1 = 0
    global_step = 0
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}", 
                        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                        ncols=100, ascii=' =')
        
        for step, batch in enumerate(progress_bar):
            # 将数据移到设备
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # 混合精度前向传播
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs[0]
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs[0]
            
            # 梯度累积：损失除以累积步数
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            # 反向传播
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            epoch_loss += loss.item()
            
            # 梯度累积：只在特定步数更新参数
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # 梯度裁剪
                if args.max_grad_norm > 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                # 更新参数
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # 更新进度条
                progress_bar.set_postfix({
                    'loss': f'{loss.item() * args.gradient_accumulation_steps:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
                
                # 记录日志
                if global_step % args.logging_steps == 0:
                    logger.info(
                        f"Epoch {epoch+1}, Step {global_step}, "
                        f"Loss: {loss.item() * args.gradient_accumulation_steps:.4f}, "
                        f"LR: {scheduler.get_last_lr()[0]:.2e}"
                    )
        
        # 每个epoch结束，评估验证集
        avg_train_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} 平均训练损失: {avg_train_loss:.4f}")
        
        # 验证
        val_f1, val_report, val_loss = evaluate(model, val_loader, device, id_to_label, logger)
        logger.info(f"Epoch {epoch+1} 验证损失: {val_loss:.4f}, F1: {val_f1:.4f}")
        logger.info(f"验证集分类报告:\n{val_report}")
        
        # 保存最佳模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_path = os.path.join(args.output_dir, "best_model")
            os.makedirs(save_path, exist_ok=True)
            
            # 保存模型和tokenizer
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            
            # 保存标签映射
            import json
            with open(os.path.join(save_path, "label_map.json"), "w", encoding="utf-8") as f:
                json.dump({"label_to_id": label_to_id, "id_to_label": id_to_label}, f, ensure_ascii=False)
            
            logger.info(f"保存最佳模型到 {save_path}, F1: {best_f1:.4f}")
    
    logger.info(f"训练完成！最佳验证F1: {best_f1:.4f}")


def main():
    parser = argparse.ArgumentParser(description="训练BERT+CRF中文NER模型")
    
    # 模型参数
    parser.add_argument("--model_name", type=str, default="bert-base-chinese",
                        help="预训练BERT模型名称")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="最大序列长度")
    
    # 数据参数
    parser.add_argument("--data_dir", type=str, default="data",
                        help="数据缓存目录")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=16,
                        help="训练批次大小")
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=3e-5,
                        help="BERT学习率")
    parser.add_argument("--crf_learning_rate", type=float, default=1e-3,
                        help="CRF层学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="权重衰减")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="warmup比例")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="最大梯度范数（梯度裁剪）")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="梯度累积步数（用于显存优化）")
    
    # 显存优化
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="使用混合精度训练")
    parser.add_argument("--no_cuda", action="store_true",
                        help="不使用CUDA")
    parser.add_argument("--freeze_bert", action="store_true",
                        help="冻结BERT参数")
    parser.add_argument("--freeze_bert_layers", type=int, default=0,
                        help="冻结BERT前N层")
    
    # 其他参数
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="输出目录")
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="日志记录步数")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="数据加载线程数")
    
    args = parser.parse_args()
    
    train(args)


if __name__ == "__main__":
    main()
