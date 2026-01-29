@echo off
chcp 65001 >nul
echo ========================================
echo BERT+CRF 中文NER训练脚本
echo ========================================

echo [1/4] 激活conda环境: hciaML
call conda activate hciaML
if errorlevel 1 (
    echo 错误: 无法激活环境hciaML
    exit /b 1
)

echo [2/4] 安装依赖
call pip install -r requirements.txt
if errorlevel 1 (
    echo 警告: 部分依赖安装失败，尝试继续...
)

echo [3/4] 下载MSRA-NER数据集
call python src/download_data.py --cache_dir data
if errorlevel 1 (
    echo 错误: 数据下载失败
    exit /b 1
)

echo [4/4] 开始训练模型
call python src/train.py --model_name bert-base-chinese --batch_size 16 --num_epochs 5 --learning_rate 3e-5 --max_seq_length 128 --fp16 --output_dir outputs

echo.
echo ========================================
echo 训练完成!
echo 最佳模型保存在: outputs/best_model
echo ========================================
pause
