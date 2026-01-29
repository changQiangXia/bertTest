@echo off
chcp 65001 >nul
echo ========================================
echo BERT+CRF 中文NER - 极致低显存模式
echo ========================================
echo 适用场景：4GB显存仍然OOM
echo 策略：完全冻结BERT，只训练CRF层
echo ========================================

call conda activate hciaML

python src/train.py ^
    --model_name bert-base-chinese ^
    --batch_size 1 ^
    --gradient_accumulation_steps 8 ^
    --num_epochs 5 ^
    --learning_rate 0.001 ^
    --max_seq_length 48 ^
    --fp16 ^
    --freeze_bert ^
    --weight_decay 0.0 ^
    --output_dir outputs

echo 训练完成！
pause
