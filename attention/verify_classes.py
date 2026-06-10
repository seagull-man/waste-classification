"""
验证并显示模型的真实类别顺序
运行这个脚本可以查看模型实际的类别名称和顺序
"""

import sys
import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from cbam_modules import CBAM, WrapperWithCBAM, ChannelAttention, SpatialAttention
from ultralytics import YOLO

MODEL_PATH = os.path.join(CURRENT_DIR, "runs", "classify", "new_cbam_fixed", "weights", "best.pt")

print("=" * 70)
print("正在加载模型并获取真实类别顺序...")
model = YOLO(MODEL_PATH)

print("\n模型实际类别顺序 (model.names):")
print("-" * 70)
for idx, name in model.names.items():
    print(f"  索引 {idx}: 英文名称 = '{name}'")

print("\n" + "=" * 70)
print("如果显示的类别顺序与预期不符，请修改 streamlit_app.py 中的")
print("CLASS_INFO_BY_NAME 字典，让每个英文名称对应到正确的中文类别。")
print("=" * 70)
