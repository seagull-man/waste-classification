"""
路径配置文件 - 统一管理项目中的路径

使用方法：
from path_config import DATA_PATH, RUNS_PATH

该文件会自动检测操作系统，选择正确的路径
"""

import os
import platform

# 检测操作系统
is_windows = platform.system() == 'Windows'

# 基础路径
if is_windows:
    BASE_PATH = "C:/Users/11237/Desktop/final2/autodl-tmp"
else:
    BASE_PATH = "/root/autodl-tmp"

# 数据集路径
DATA_PATH = os.path.join(BASE_PATH, "garbage_4cls")

# 原始数据集路径
ORIGINAL_DATA_PATH = os.path.join(BASE_PATH, "garbage_datasets")

# 训练结果保存路径
RUNS_PATH = os.path.join(BASE_PATH, "runs")

# 分类任务路径
CLASSIFY_PATH = os.path.join(RUNS_PATH, "classify")

# 模型权重路径
WEIGHTS_PATH = os.path.join(CLASSIFY_PATH, "weights")

def get_model_path(model_name):
    """获取指定模型的路径"""
    return os.path.join(CLASSIFY_PATH, model_name)

def get_weights_path(model_name, weight_type="best"):
    """获取指定模型的权重路径"""
    return os.path.join(CLASSIFY_PATH, model_name, "weights", f"{weight_type}.pt")

def get_results_path(model_name):
    """获取指定模型的训练结果CSV路径"""
    return os.path.join(CLASSIFY_PATH, model_name, "results.csv")

# 测试集路径
TEST_DATA_PATH = os.path.join(DATA_PATH, "test")

print(f"📂 基础路径: {BASE_PATH}")
print(f"📊 数据集路径: {DATA_PATH}")
print(f"📁 训练结果路径: {RUNS_PATH}")
