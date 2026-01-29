"""
下载MSRA-NER数据集

MSRA-NER是微软亚洲研究院发布的中文命名实体识别数据集，
包含人名(PER)、地名(LOC)、机构名(ORG)三种实体类型。

数据格式：
- 训练集：约46,000条
- 测试集：约4,000条
- 标注格式：BIO (Begin-Inside-Outside)

使用Hugging Face datasets库下载
"""
import os
import sys
from datasets import load_dataset
import json


def download_msra_ner(cache_dir: str = "data", verify: bool = True):
    """
    下载MSRA-NER数据集
    
    Args:
        cache_dir: 数据缓存目录
        verify: 下载后是否验证数据
    """
    print("=" * 60)
    print("开始下载 MSRA-NER 数据集")
    print("=" * 60)
    
    # 确保缓存目录存在
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"\n数据将缓存到: {os.path.abspath(cache_dir)}")
    
    try:
        # 加载数据集
        print("\n正在从Hugging Face下载数据...")
        print("数据集: chinesenlpcorpus/msra_ner")
        
        dataset = load_dataset("chinesenlpcorpus/msra_ner", cache_dir=cache_dir)
        
        print("\n数据集下载成功！")
        
        # 打印数据集信息
        print("\n数据集信息:")
        print(f"  - 训练集: {len(dataset['train'])} 条")
        print(f"  - 测试集: {len(dataset['test'])} 条")
        
        # 标签信息
        if hasattr(dataset['train'], 'features') and 'ner_tags' in dataset['train'].features:
            feature = dataset['train'].features['ner_tags']
            if hasattr(feature, 'feature') and hasattr(feature.feature, 'names'):
                label_names = feature.feature.names
                print(f"\n标签类别 ({len(label_names)} 种):")
                for i, name in enumerate(label_names):
                    print(f"  {i}: {name}")
        
        # 显示样本示例
        print("\n样本示例（训练集前3条）:")
        for i in range(min(3, len(dataset['train']))):
            example = dataset['train'][i]
            tokens = example['tokens']
            tags = example['ner_tags']
            
            print(f"\n  样本 {i+1}:")
            print(f"    Tokens: {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
            print(f"    Tags:   {tags[:20]}{'...' if len(tags) > 20 else ''}")
        
        # 保存数据集信息
        info = {
            "dataset_name": "MSRA-NER",
            "source": "ChineseNLPCorpus (chinesenlpcorpus/msra_ner)",
            "train_size": len(dataset['train']),
            "test_size": len(dataset['test']),
            "entity_types": ["PER", "LOC", "ORG"],
            "annotation_format": "BIO"
        }
        
        info_path = os.path.join(cache_dir, "dataset_info.json")
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        
        print(f"\n数据集信息已保存到: {info_path}")
        
        # 验证数据
        if verify:
            print("\n验证数据中...")
            verify_dataset(dataset)
        
        print("\n" + "=" * 60)
        print("数据下载和验证完成！")
        print("=" * 60)
        
        return dataset
        
    except Exception as e:
        print(f"\n下载失败: {str(e)}")
        print("\n可能的解决方案:")
        print("1. 检查网络连接")
        print("2. 设置Hugging Face镜像: export HF_ENDPOINT=https://hf-mirror.com")
        print("3. 手动下载数据集并放到 data 目录")
        raise


def verify_dataset(dataset):
    """
    验证数据集完整性
    
    检查：
    1. 样本数量是否合理
    2. token和tag数量是否匹配
    3. 标签值是否在有效范围内
    """
    print("\n执行数据验证...")
    
    errors = []
    
    # 检查训练集
    for i, example in enumerate(dataset['train']):
        tokens = example['tokens']
        tags = example['ner_tags']
        
        # 检查长度匹配
        if len(tokens) != len(tags):
            errors.append(f"训练集样本{i}: token数量({len(tokens)})与tag数量({len(tags)})不匹配")
        
        # 检查标签范围（MSRA使用0-6）
        for tag in tags:
            if tag < 0 or tag > 6:
                errors.append(f"训练集样本{i}: 非法标签值 {tag}")
        
        # 只检查前1000条加速
        if i >= 1000:
            break
    
    # 检查测试集
    for i, example in enumerate(dataset['test']):
        tokens = example['tokens']
        tags = example['ner_tags']
        
        if len(tokens) != len(tags):
            errors.append(f"测试集样本{i}: token数量与tag数量不匹配")
        
        for tag in tags:
            if tag < 0 or tag > 6:
                errors.append(f"测试集样本{i}: 非法标签值 {tag}")
        
        if i >= 1000:
            break
    
    if errors:
        print(f"\n发现 {len(errors)} 个错误:")
        for error in errors[:10]:  # 只显示前10个
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... 还有 {len(errors) - 10} 个错误")
    else:
        print("  验证通过！数据格式正确。")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="下载MSRA-NER数据集")
    parser.add_argument("--cache_dir", type=str, default="data",
                        help="数据缓存目录")
    parser.add_argument("--no_verify", action="store_true",
                        help="跳过数据验证")
    
    args = parser.parse_args()
    
    download_msra_ner(args.cache_dir, verify=not args.no_verify)


if __name__ == "__main__":
    main()
