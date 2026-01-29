@echo off
chcp 65001 >nul
echo ========================================
echo BERT+CRF 中文NER - CPU训练模式
echo ========================================
echo 如果4GB显存仍然OOM（内存溢出），请使用此CPU版本
echo 速度较慢但更稳定
echo ========================================

call conda activate hciaML

echo [1/3] 安装依赖...
call pip install -r requirements.txt

echo [2/3] 下载数据...
call python src/download_data.py --cache_dir data

echo [3/3] 开始CPU训练...
python src/train.py ^
    --model_name bert-base-chinese ^
    --batch_size 8 ^
    --gradient_accumulation_steps 2 ^
    --num_epochs 3 ^
    --learning_rate 5e-5 ^
    --max_seq_length 128 ^
    --no_cuda ^
    --output_dir outputs

echo ========================================
echo CPU训练完成！
echo ========================================
pause
