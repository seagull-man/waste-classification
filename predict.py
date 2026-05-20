from ultralytics import YOLO
import os

# 加载你训练好的最佳模型
model = YOLO(r"D:\Waste-Classification-using-YOLOv8 dataset\runs\classify\waste_yolov8_4cls14\weights\last.pt")

# 测试文件夹
test_dir = r"D:\Waste-Classification-using-YOLOv8 dataset\test_images"

# 对每张图预测
results = model.predict(
    source=test_dir,
    save=True,          # 自动保存带标签的图
    show=False,         # 不弹窗（避免卡住）
    device='cuda'
)

# 打印结果
class_names = ['hazardous', 'kitchen', 'other', 'recyclable']  # 替换成你数据集的实际类别顺序！

for i, result in enumerate(results):
    img_name = os.path.basename(result.path)
    pred_class_id = result.probs.top1
    confidence = result.probs.top1conf.item()
    pred_class = class_names[pred_class_id]
    print(f"{img_name} → {pred_class} (belief: {confidence:.2f})")