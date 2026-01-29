#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPU显存测试脚本

在正式训练前测试你的GPU是否足够运行BERT+CRF模型
"""
import torch
import sys

def test_gpu_memory():
    print("=" * 60)
    print("GPU显存测试")
    print("=" * 60)
    
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("\n❌ 没有检测到CUDA设备")
        print("请使用CPU训练模式: python src/train.py --no_cuda")
        return False
    
    # 获取GPU信息
    device_id = 0
    gpu_name = torch.cuda.get_device_name(device_id)
    total_memory = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)  # GB
    
    print(f"\n✓ 检测到GPU: {gpu_name}")
    print(f"✓ 总显存: {total_memory:.2f} GB")
    
    # 尝试加载模型
    print("\n尝试加载BERT+CRF模型...")
    
    try:
        from transformers import BertTokenizer, BertConfig
        from src.model import BertCRFForNER
        
        # 加载tokenizer
        print("  - 加载Tokenizer...")
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        
        # 加载模型配置
        print("  - 加载模型配置...")
        config = BertConfig.from_pretrained("bert-base-chinese")
        
        # 创建模型并移到GPU
        print("  - 创建BERT+CRF模型...")
        model = BertCRFForNER.from_pretrained(
            "bert-base-chinese", 
            config=config, 
            num_labels=7
        )
        model.cuda()
        
        # 查看模型加载后的显存占用
        memory_after_load = torch.cuda.memory_allocated() / (1024**3)
        print(f"  ✓ 模型加载完成，显存占用: {memory_after_load:.2f} GB")
        
        # 测试前向传播
        print("\n测试前向传播（batch_size=1, seq_len=64）...")
        
        test_text = "北京是中国的首都，位于华北地区。"
        inputs = tokenizer(
            test_text, 
            return_tensors="pt", 
            max_length=64, 
            truncation=True, 
            padding="max_length"
        )
        inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # 前向传播
        with torch.no_grad():
            outputs = model(**inputs)
        
        memory_after_forward = torch.cuda.memory_allocated() / (1024**3)
        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
        
        print(f"  ✓ 前向传播成功")
        print(f"  - 当前显存占用: {memory_after_forward:.2f} GB")
        print(f"  - 峰值显存占用: {peak_memory:.2f} GB")
        
        # 测试反向传播（训练）
        print("\n测试反向传播（训练）...")
        
        # 创建需要梯度的输入
        inputs_train = tokenizer(
            test_text, 
            return_tensors="pt", 
            max_length=64, 
            truncation=True, 
            padding="max_length"
        )
        inputs_train = {k: v.cuda() for k, v in inputs_train.items()}
        labels = torch.randint(0, 7, (1, 64)).cuda()  # 随机标签
        
        # 前向+反向
        outputs = model(labels=labels, **inputs_train)
        loss = outputs[0]
        loss.backward()
        
        memory_after_backward = torch.cuda.memory_allocated() / (1024**3)
        peak_memory_train = torch.cuda.max_memory_allocated() / (1024**3)
        
        print(f"  ✓ 反向传播成功")
        print(f"  - 当前显存占用: {memory_after_backward:.2f} GB")
        print(f"  - 峰值显存占用: {peak_memory_train:.2f} GB")
        
        # 评估显存是否足够
        print("\n" + "=" * 60)
        print("评估结果")
        print("=" * 60)
        
        available_memory = total_memory * 0.9  # 留10%余量
        
        if peak_memory_train < available_memory:
            print(f"\n✅ 显存充足！可以训练！")
            print(f"   建议使用参数：")
            print(f"   --batch_size 1")
            print(f"   --max_seq_length 64")
            print(f"   --gradient_accumulation_steps 16")
            if peak_memory_train < total_memory * 0.7:
                print(f"   甚至可以尝试更大的batch_size!")
        else:
            print(f"\n⚠️ 显存可能不足，建议：")
            print(f"   1. 减小max_seq_length到48或32")
            print(f"   2. 冻结BERT层 --freeze_bert_layers 10")
            print(f"   3. 使用CPU训练 --no_cuda")
        
        # 显存优化建议
        print("\n" + "=" * 60)
        print("显存优化建议（按优先级）")
        print("=" * 60)
        print("1. --batch_size 1          # 必须设置为1")
        print("2. --max_seq_length 64      # 64通常足够，可减小到48")
        print("3. --freeze_bert_layers 10  # 冻结前10层")
        print("4. --fp16                   # 混合精度训练")
        print("5. --gradient_accumulation_steps 16  # 累积梯度")
        
        return True
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n❌ 显存不足！")
        print(f"错误: {str(e)}")
        print(f"\n建议：")
        print(f"1. 重启电脑释放显存")
        print(f"2. 关闭其他占用显存的程序")
        print(f"3. 使用CPU训练: python src/train.py --no_cuda")
        return False
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_gpu_memory()
    sys.exit(0 if success else 1)
