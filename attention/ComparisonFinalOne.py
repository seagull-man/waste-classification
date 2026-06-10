#!/usr/bin/env python3
import os, time
import torch, torch.nn as nn
import pandas as pd
from PIL import Image
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, recall_score
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ── 注册自定义注意力模块 ──────────────────────────────────────────
import ultralytics.nn.tasks as tasks_module

class SEBlock(nn.Module):
    def __init__(self, *args):
        super().__init__()
        reduction = 16
        for arg in args:
            if isinstance(arg, int) and arg > 0:
                reduction = arg
                break
        self._reduction = reduction
        self._initialized = False

    def _initialize(self, channels):
        hidden_channels = max(channels // self._reduction, 1)
        self.gamma = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels, bias=False),
            nn.Sigmoid()
        )
        self._initialized = True

    def forward(self, x):
        if not self._initialized:
            self._initialize(x.size(1))
            self.to(device=x.device, dtype=x.dtype)
        identity = x
        b, c, _, _ = x.shape
        weight = self.avg_pool(x).view(b, c)
        weight = self.fc(weight).view(b, c, 1, 1)
        refined = x * weight
        return identity + self.gamma * (refined - identity)


class ECABlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        channels = None
        gamma = 2
        b = 1
        if args:
            for arg in args:
                if isinstance(arg, int) and arg > 0 and channels is None:
                    channels = arg
        if 'c1' in kwargs:
            channels = kwargs['c1']
        elif 'c2' in kwargs and channels is None:
            channels = kwargs['c2']
        elif 'channels' in kwargs:
            channels = kwargs['channels']
        if channels is None:
            self._initialized = False
            self._gamma = gamma
            self._b = b
            self.avg_pool = None
            self.conv = None
            self.sigmoid = None
        else:
            self._initialize(channels, gamma, b)

    def _initialize(self, channels, gamma=2, b=1):
        t = int(abs((torch.log2(torch.tensor(channels)) + b) / gamma))
        kernel_size = t if t % 2 == 1 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self._initialized = True

    def forward(self, x):
        if not hasattr(self, '_initialized') or not self._initialized:
            self._initialize(x.size(1), getattr(self, '_gamma', 2), getattr(self, '_b', 1))
        y = self.avg_pool(x).squeeze(-1)
        y = y.transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y

tasks_module.__dict__["SEBlock"] = SEBlock
tasks_module.__dict__["ECABlock"] = ECABlock

# ── 配置 ──────────────────────────────────────────────────────────
DATA_DIR = "/root/autodl-tmp/garbage_4cls"
RUNS_DIR = "/root/autodl-tmp/waste-classification/attention/runs/classify"
SAVE_DIR = "/root/autodl-tmp/waste-classification/attention/comparison_results"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

MODELS = {
    "YOLOv8n (Baseline)": os.path.join(RUNS_DIR, "waste_cls_original"),
    "YOLOv8n + ECA":      "/root/autodl-tmp/runs/classify/waste_cls_yolov8n_eca",
    "YOLOv8n + CBAM":     os.path.join(RUNS_DIR, "waste_cls_yolov8n_cbam_replace"),
    "YOLOv8n + SE":       os.path.join(RUNS_DIR, "waste_cls_yolov8n_se_real-6"),
}

# 图像预处理
VAL_TFM = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
])


# ── 工具函数 ───────────────────────────────────────────────────────
def load_model(path):
    model = YOLO(path)
    model.to(DEVICE)
    return model

def get_best_from_csv(csv_path, col):
    try:
        df = pd.read_csv(csv_path)
        return float(df[col].max()) if col in df.columns else 0.0
    except Exception:
        return 0.0

def collect_val_samples(data_dir):
    """收集 val 目录下所有图片路径和标签"""
    val_dir = os.path.join(data_dir, "val")
    class_dirs = sorted(
        d for d in os.listdir(val_dir)
        if os.path.isdir(os.path.join(val_dir, d))
    )
    label_map = {d: i for i, d in enumerate(class_dirs)}

    img_paths, labels = [], []
    for cls_name, label_id in label_map.items():
        cls_dir = os.path.join(val_dir, cls_name)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                img_paths.append(os.path.join(cls_dir, fname))
                labels.append(label_id)
    return img_paths, labels, dict(enumerate(class_dirs))


# ── 真实指标：全量 val 推理 ────────────────────────────────────────
def evaluate_full_metrics(model, img_paths, labels, class_names, batch_size=64):
    total = len(labels)
    print(f"  📸 验证集图片: {total} 张, 类别: {class_names}")

    all_preds = []
    for i in range(0, total, batch_size):
        batch_paths = img_paths[i:i + batch_size]
        batch_imgs = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                batch_imgs.append(VAL_TFM(img))
            except Exception:
                batch_imgs.append(torch.zeros(3, 320, 320))
        inp = torch.stack(batch_imgs).to(DEVICE)
        with torch.no_grad():
            out = model.model(inp)
        if isinstance(out, (list, tuple)):
            out = out[0]
        preds = out.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        if (i // batch_size + 1) % 20 == 0:
            print(f"  ⏳ 推理进度: {min(i + batch_size, total)}/{total}")

    acc = accuracy_score(labels, all_preds)
    f1  = f1_score(labels, all_preds, average="macro")
    rec = recall_score(labels, all_preds, average="macro")
    return acc, f1, rec


# ── FPS 测量 ───────────────────────────────────────────────────────
def measure_fps(model, runs=200):
    dummy = torch.randn(1, 3, 320, 320).to(DEVICE)
    for _ in range(10):
        _ = model.predict(dummy, device=DEVICE, verbose=False)
    if DEVICE == "cuda:0":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(runs):
        _ = model.predict(dummy, device=DEVICE, verbose=False)
    if DEVICE == "cuda:0":
        torch.cuda.synchronize()
    return runs / (time.time() - t0)


# ── 绘图 ───────────────────────────────────────────────────────────
def plot_results(results, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    names = [r["name"] for r in results]
    accs  = [r["accuracy"] for r in results]
    f1s   = [r["f1"]       for r in results]
    recs  = [r["recall"]   for r in results]
    fpss  = [r["fps"]      for r in results]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    def bar(ax, vals, title, ylabel):
        bars = ax.bar(names, vals, color=colors, edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2.,
                b.get_height() + max(vals) * 0.005,
                f"{v:.4f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold"
            )
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_xticklabels(names, rotation=15, fontsize=11)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    _, axes = plt.subplots(2, 2, figsize=(14, 11))
    bar(axes[0, 0], accs, "Top-1 Accuracy", "Accuracy")
    bar(axes[0, 1], f1s,  "F1 Score (macro)", "F1")
    bar(axes[1, 0], recs, "Recall (macro)", "Recall")
    bar(axes[1, 1], fpss, "Inference FPS", "FPS (imgs/s)")
    plt.tight_layout(pad=2)
    plt.savefig(os.path.join(save_dir, "comparison.png"), dpi=200)
    plt.close()
    print(f"\n✅ 图表已保存: {save_dir}/comparison.png")


# ── 主函数 ─────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  Model Comparison - Accuracy / F1 / Recall / FPS")
    print("=" * 70)

    # 预收集 val 数据（所有模型共享）
    img_paths, labels, class_names = collect_val_samples(DATA_DIR)

    results = []

    for name, run_dir in MODELS.items():
        best_pt = os.path.join(run_dir, "weights", "best.pt")
        csv_f = os.path.join(run_dir, "results.csv")

        print(f"\n{'─' * 60}")
        print(f"📦 {name}")
        print(f"  路径: {run_dir}")

        if not os.path.exists(best_pt):
            print(f"  ❌ best.pt 不存在, 跳过")
            continue

        csv_acc = get_best_from_csv(csv_f, "metrics/accuracy_top1")
        print(f"  📊 CSV best accuracy_top1: {csv_acc:.4f}")

        model = load_model(best_pt)
        params = sum(p.numel() for p in model.model.parameters())
        print(f"  🧩 参数量: {params:,}")

        print(f"  🔍 全量验证集评估...")
        real_acc, real_f1, real_rec = evaluate_full_metrics(
            model, img_paths, labels, class_names
        )
        print(f"  ✅ Accuracy: {real_acc:.4f} | F1: {real_f1:.4f} | Recall: {real_rec:.4f}")

        fps = measure_fps(model, runs=300)
        print(f"  ⚡ FPS: {fps:.1f} imgs/s")

        results.append({
            "name": name,
            "accuracy": real_acc,
            "f1":       real_f1,
            "recall":   real_rec,
            "fps":      fps,
            "params":   params,
            "csv_top1": csv_acc,
        })

    if not results:
        print("\n❌ 没有成功评估任何模型")
        return

    # ── 终端表格 ──
    print("\n" + "=" * 90)
    header = f"{'Model':<25} {'Accuracy':>10} {'F1':>10} {'Recall':>10} {'FPS':>10} {'Params':>12}"
    print(header)
    print("-" * 90)
    for r in results:
        print(
            f"{r['name']:<25} {r['accuracy']:>10.4f} {r['f1']:>10.4f} "
            f"{r['recall']:>10.4f} {r['fps']:>10.1f} {r['params']:>12,}"
        )
    print("=" * 90)

    # ── CSV ──
    os.makedirs(SAVE_DIR, exist_ok=True)
    df = pd.DataFrame(results)
    csv_out = os.path.join(SAVE_DIR, "comparison_results.csv")
    df.to_csv(csv_out, index=False, encoding="utf-8-sig")
    print(f"\n✅ CSV 已保存: {csv_out}")

    # ── Markdown ──
    md_out = os.path.join(SAVE_DIR, "comparison_results.md")
    with open(md_out, "w", encoding="utf-8") as f:
        f.write("# 模型对比结果\n\n")
        f.write("| Model | Accuracy | F1 | Recall | FPS | Params |\n")
        f.write("|-------|----------|----|--------|-----|--------|\n")
        for r in results:
            f.write(
                f"| {r['name']} | {r['accuracy']:.4f} | {r['f1']:.4f} "
                f"| {r['recall']:.4f} | {r['fps']:.1f} | {r['params']:,} |\n"
            )
    print(f"✅ Markdown 已保存: {md_out}")

    # ── 绘图 ──
    plot_results(results, SAVE_DIR)

    print("\n✅ 全部完成!")


if __name__ == "__main__":
    main()