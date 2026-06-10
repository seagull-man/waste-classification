"""
全局配置文件 - 路径、超参数、类别名统一管理
"""
import os
import torch

# ==================== 路径配置 ====================
BASE_PATH = "/root/autodl-tmp"
DATA_PATH = os.path.join(BASE_PATH, "garbage_4cls")
MODEL_PATH = os.path.join(BASE_PATH, "yolov8n-cls.pt")
OUTPUT_DIR = os.path.join(BASE_PATH, "waste-classification", "prune_attention")
RUNS_DIR = os.path.join(OUTPUT_DIR, "runs")

# 确保输出目录存在
os.makedirs(RUNS_DIR, exist_ok=True)

# ==================== 数据集配置 ====================
CLASS_NAMES = ["hazardous", "kitchen", "other", "recyclable"]  # 按字母序
NUM_CLASSES = 4

# ==================== 训练超参数 ====================
BATCH_SIZE = 64          # 48GB 显存可以用大 batch
IMGSZ = 224              # YOLOv8-cls 推荐 224
EPOCHS = 100             # 训练轮数
LR0 = 0.001              # 初始学习率
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.937
WARMUP_EPOCHS = 3
PATIENCE = 15            # 早停

# ==================== 设备配置 ====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_ID = 0

# ==================== 剪枝配置 ====================
PRUNE_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5]  # 尝试的剪枝比例
FINE_TUNE_EPOCHS = 30   # 剪枝后微调轮数

# ==================== 注意力配置 ====================
ATTENTION_TYPES = ["se", "cbam", "eca"]
# 注意力模块已内置在 ultralytics (nn/modules/block.py, nn/modules/conv.py) 中
# YAML 定义文件: yolov8n-cls-{se,cbam,eca}.yaml

# ==================== 评估配置 ====================
WARMUP_ITERS = 50        # FPS 测试预热迭代次数
TEST_ITERS = 200         # FPS 测试正式迭代次数