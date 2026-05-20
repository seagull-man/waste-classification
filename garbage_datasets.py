from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import os

class GarbageDetector:
    def __init__(self):
        self.model = None
        self.class_names = []
    
    def train(self, data_yaml_path, weights_path=None):
        """
        训练模型或继续训练
        :param data_yaml_path: 数据配置文件路径
        :param weights_path: 断点权重路径
        """
        if weights_path:
            # 如果需要断点续训，加载指定权重
            self.model = YOLO(weights_path)
            results = self.model.train(
              data=data_yaml_path,
              epochs=50,  # 继续训练的轮数
              imgsz=1024,
              batch=32,
              workers=8,
              patience=10,
              device='0' if torch.cuda.is_available() else 'cpu',
              pretrained=False,  # 断点续训时不需要再加载预训练
              optimizer='AdamW',
              lr0=0.001,
              weight_decay=0.0005
            )
        else:
          # 创建YOLO模型
          self.model = YOLO('yolov8s.pt')  # 使用yolov8s架构创建新模型
          # 开始训练
          results = self.model.train(
              data=data_yaml_path,
              epochs=100,            # 训练轮数
              imgsz=1024,            # 图片尺寸
              batch=32,             # 批次大小
              workers=8,            # 数据加载器的工作进程数
              patience=10,         # 早停策略
              device='0' if torch.cuda.is_available() else 'cpu',  # 使用GPU或CPU
              pretrained=True,        # 使用预训练权重
              optimizer='AdamW',      # 使用AdamW优化器
              #   对于batch size较大（如32及以上），推荐lr0=0.001或lr0=0.002
              #   对于batch size较小（如16或更小），可适当降低到lr0=0.0005~0.001
              lr0=0.001,             # 初始学习率
              weight_decay=0.0005     # 权重衰减
          )
        
    def predict(self, image_paths):
        """
        对多张图片进行预测并在一个画布上用matplotlib展示
        :param image_paths: 图片路径或路径列表
        """
        if self.model is None:
            raise ValueError("模型未训练，请先训练模型")

        # 支持单张图片字符串输入
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        num_images = len(image_paths)
        cols = min(3, num_images)
        rows = (num_images + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
        if num_images == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, image_path in enumerate(image_paths):
            img = mpimg.imread(image_path)
            ax = axes[idx]
            ax.imshow(img)
            ax.set_title(os.path.basename(image_path))
            ax.axis('off')

            results = self.model(image_path)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='g', facecolor='none')
                    ax.add_patch(rect)
                    label = f'{self.class_names[cls]} {conf:.2f}'
                    ax.text(x1, y1 - 5, label, color='g', fontsize=8, backgroundcolor='w')

        # 隐藏多余的子图
        for j in range(idx + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    detector = GarbageDetector()
    
    # 如果有预训练模型，可以加载
    if os.path.exists('best.pt'):
      detector.model = YOLO('best.pt')
    else:
      # 训练模型
      detector.train("data.yaml", weights_path=None)
    #   detector.train("data3.yaml", weights_path="runs/detect/train/weights/best.pt")

    # 预测单张或多张图片
    # detector.predict([
    #     "datasets/test/images/Batteries_20.jpg",
    #     "datasets/test/images/Cardboard_35.jpg",
    #     "datasets/test/images/Glassbottles_6.jpg",
    #     "datasets/test/images/Metal_12.jpg",
    #     "datasets/test/images/Paper_T_5.jpg",
    #     "datasets/test/images/Plastic_2_7.jpg",
    #     "datasets/test/images/Zip-topcan_24.jpg"
    # ])