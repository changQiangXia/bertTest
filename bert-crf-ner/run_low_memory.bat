@echo off
chcp 65001 >nul
echo ========================================
echo BERT+CRF 中文NER训练脚本 - 低显存模式
echo ========================================
echo 本脚本适用于显存不足的情况（如8GB显存）
echo 使用: batch_size=4 + gradient_accumulation_steps=4
echo 等效batch_size = 4 * 4 = 16
echo ========================================

call conda activate hciaML

call pip install -r requirements.txt

call python src/download_data.py --cache_dir data

call python src/train.py --model_name bert-base-chinese --batch_size 4 --gradient_accumulation_steps 4 --num_epochs 5 --learning_rate 3e-5 --max_seq_length 128 --fp16 --freeze_bert_layers 6 --output_dir outputs

echo 训练完成！
pause
