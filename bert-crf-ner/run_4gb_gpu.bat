@echo off
chcp 65001 >nul
echo ========================================
echo BERT+CRF 中文NER - 4GB显存专用训练脚本
echo ========================================
echo 优化策略：
echo   - batch_size=1（最小批次）
echo   - gradient_accumulation_steps=16（累积等效batch_size=16）
echo   - max_seq_length=64（缩短序列长度）
echo   - freeze_bert_layers=10（冻结前10层，只训练最后2层+CRF）
echo   - fp16混合精度训练
echo ========================================

call conda activate hciaML

echo [1/3] 安装依赖...
call pip install -r requirements.txt

echo [2/3] 下载数据...
call python src/download_data.py --cache_dir data

echo [3/3] 开始训练（4GB显存优化模式）...
python src/train.py ^
    --model_name bert-base-chinese ^
    --batch_size 1 ^
    --gradient_accumulation_steps 16 ^
    --num_epochs 5 ^
    --learning_rate 5e-5 ^
    --crf_learning_rate 5e-3 ^
    --max_seq_length 64 ^
    --fp16 ^
    --freeze_bert_layers 10 ^
    --weight_decay 0.0 ^
    --warmup_ratio 0.1 ^
    --logging_steps 50 ^
    --output_dir outputs ^
    --num_workers 0

echo ========================================
echo 训练完成！
echo ========================================
pause
