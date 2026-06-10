"""
垃圾四分类交互式推理系统
模型: YOLOv8n-cls + ECA 注意力机制
"""

import sys
import os
import time
from PIL import Image

# 确保项目路径在 sys.path 中，以便加载自定义模块
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

import streamlit as st
import torch
import torch.nn as nn
from ultralytics import YOLO

# 必须先注册自定义模块，否则加载 .pt 时会报错
import ultralytics.nn.tasks as tasks_module

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

tasks_module.__dict__["ECABlock"] = ECABlock

# ==================== 配置 ====================
MODEL_PATH = os.path.join(CURRENT_DIR, "runs", "classify", "waste_cls_yolov8n_eca-2", "weights", "best.pt")
IMGSZ = 320
CONFIDENCE_THRESHOLD = 0.5

# 类别信息: 英文名 → {中文名, 英文名, 颜色, 图标, 示例, 投放指南}
CLASS_INFO_BY_NAME = {
    "recyclable": {
        "name": "可回收垃圾",
        "en_name": "Recyclable",
        "color": "#2E86AB",
        "icon": "♻️",
        "examples": "废纸、塑料瓶、金属罐、玻璃瓶、旧衣物",
        "guide": "请投放到蓝色垃圾桶。投递前请清洗干净、沥干水分。",
    },
    "kitchen": {
        "name": "厨余垃圾",
        "en_name": "Kitchen Waste",
        "color": "#A23B72",
        "icon": "🍎",
        "examples": "剩菜剩饭、果皮、菜叶、骨头、茶渣",
        "guide": "请投放到绿色垃圾桶。沥干水分后投放，勿混入塑料袋。",
    },
    "hazardous": {
        "name": "有害垃圾",
        "en_name": "Hazardous Waste",
        "color": "#F18F01",
        "icon": "☣️",
        "examples": "废电池、灯泡、过期药品、油漆桶、水银温度计",
        "guide": "请投放到红色垃圾桶。轻拿轻放，破损物品请包裹后投放。",
    },
    "other": {
        "name": "其他垃圾",
        "en_name": "Other Waste",
        "color": "#555555",
        "icon": "🗑️",
        "examples": "污染纸张、一次性餐具、尘土、破旧陶瓷",
        "guide": "请投放到灰色垃圾桶。尽量沥干水分后投放。",
    },
}


# ==================== 模型加载 ====================
@st.cache_resource
def load_model(model_path: str):
    """加载 CBAM 模型，返回 (model, class_info)"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    model = YOLO(model_path)
    class_info = {}
    for idx, name in model.names.items():
        info = CLASS_INFO_BY_NAME.get(name)
        class_info[idx] = info.copy() if info else {
            "name": name, "en_name": name,
            "color": "#666", "icon": "📦", "examples": "", "guide": "",
        }
    return model, class_info


def predict_image(model, class_info: dict, image: Image.Image):
    """对图片进行推理，返回结构化预测结果"""
    result = model.predict(image, imgsz=IMGSZ, verbose=False)[0]

    if not (hasattr(result, "probs") and result.probs is not None):
        return None

    probs = result.probs.data.cpu().numpy()
    top1_idx = int(result.probs.top1)
    top1_conf = float(result.probs.top1conf)

    class_probs = sorted(
        [
            {
                "index": i,
                "name": class_info[i]["name"],
                "en_name": class_info[i]["en_name"],
                "icon": class_info[i]["icon"],
                "color": class_info[i]["color"],
                "probability": float(probs[i]),
            }
            for i in range(len(probs))
        ],
        key=lambda x: x["probability"],
        reverse=True,
    )

    top = class_info[top1_idx]
    return {
        "top1_idx": top1_idx,
        "top1_name": top["name"],
        "top1_en": top["en_name"],
        "top1_icon": top["icon"],
        "top1_conf": top1_conf,
        "top1_color": top["color"],
        "top1_examples": top["examples"],
        "top1_guide": top.get("guide", ""),
        "all_probs": class_probs,
    }


# ==================== UI 组件 ====================
def render_result_card(prediction: dict):
    """渲染预测结果卡片"""
    color = prediction["top1_color"]
    st.markdown(
        f"""
        <div style='
            background: linear-gradient(135deg, {color}22, {color}44);
            border: 2px solid {color};
            border-radius: 16px;
            padding: 1.5rem;
            text-align: center;
            margin-bottom: 1rem;
        '>
            <div style='font-size: 3rem;'>{prediction["top1_icon"]}</div>
            <div style='font-size: 2rem; font-weight: bold; color: {color};'>
                {prediction["top1_name"]}
            </div>
            <div style='font-size: 1rem; color: #888;'>
                {prediction["top1_en"]}
            </div>
            <div style='font-size: 2.5rem; font-weight: bold; margin-top: 0.5rem;'>
                {prediction["top1_conf"] * 100:.1f}%
            </div>
            <div style='font-size: 0.85rem; color: #888; margin-top: 0.3rem;'>
                置信度
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prob_bars(probs: list, top1_idx: int):
    """渲染各类别概率分布"""
    st.markdown("---")
    st.subheader("📊 各类别概率分布")

    for item in probs:
        idx = item["index"]
        st.markdown(
            f"""<div style='display: flex; align-items: center; margin-bottom: 0.4rem;'>
                <span style='width: 30px;'>{item["icon"]}</span>
                <span style='width: 80px; font-weight: {"bold" if idx == top1_idx else "normal"};'>
                    {item["name"]}
                </span>
            </div>""",
            unsafe_allow_html=True,
        )
        st.progress(
            item["probability"],
            text=f"{item['probability'] * 100:.1f}%{' ← 最佳' if idx == top1_idx else ''}",
        )


def render_guide(prediction: dict):
    """渲染投放指南"""
    top1_idx = prediction["top1_idx"]
    top1_examples = prediction["top1_examples"]
    top1_guide = prediction["top1_guide"]

    st.markdown("---")
    st.subheader("📋 处理建议")
    if top1_examples:
        st.markdown(f"**常见示例**: {top1_examples}")
    st.info(top1_guide or "请按照当地规定投放。")


# ==================== 页面配置 ====================
st.set_page_config(
    page_title="垃圾四分类识别系统",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 顶部标题
st.markdown(
    """
    <div style='text-align: center; padding: 1rem 0;'>
        <h1 style='font-size: 2.2rem; margin-bottom: 0.5rem;'>
            ♻️ 智能垃圾分类识别系统
        </h1>
        <p style='color: #888; font-size: 1rem;'>
            基于 YOLOv8n-cls + ECA 注意力机制的垃圾四分类模型
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()

# ==================== 加载模型 ====================
with st.spinner("正在加载模型..."):
    try:
        model, class_info = load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        st.stop()

# ==================== 侧边栏 ====================
with st.sidebar:
    st.header("ℹ️ 系统信息")
    st.info("YOLOv8n-cls + Residual CBAM")
    st.caption(f"输入尺寸: {IMGSZ} × {IMGSZ}  |  置信度阈值: {int(CONFIDENCE_THRESHOLD * 100)}%")

    st.divider()
    st.header("📂 垃圾四分类类别")
    for info in class_info.values():
        st.markdown(
            f"**{info['icon']} {info['name']}**  \n"
            f"<small style='color: {info['color']};'>{info['en_name']}</small>  \n"
            f"<small>{info['examples']}</small>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.success("✅ 模型已加载")
    st.caption("© 2025 Waste Classification System")

# ==================== 主界面 ====================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 上传图片")
    uploaded_file = st.file_uploader(
        "选择一张垃圾图片进行识别",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="支持 JPG、PNG、BMP、WEBP 格式",
    )

    if uploaded_file is None:
        st.markdown("---")
        st.caption("或使用摄像头拍照:")
        if st.toggle("📷 使用摄像头拍照", value=False):
            camera_image = st.camera_input("拍照")
            if camera_image is not None:
                uploaded_file = camera_image
                st.rerun()

# ==================== 推理逻辑 ====================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.image(image, caption="上传的图片", use_container_width=True)

    with col2:
        st.subheader("🔍 识别结果")

        with st.spinner("正在识别..."):
            t0 = time.time()
            prediction = predict_image(model, class_info, image)
            elapsed = time.time() - t0

        if prediction is None:
            st.error("预测失败，请检查模型是否正确加载。")
        else:
            is_confident = prediction["top1_conf"] >= CONFIDENCE_THRESHOLD

            if is_confident:
                render_result_card(prediction)
            else:
                st.warning(
                    f"置信度较低 ({prediction['top1_conf'] * 100:.1f}% "
                    f"< {int(CONFIDENCE_THRESHOLD * 100)}%)，结果仅供参考。"
                )

            st.caption(f"推理耗时: {elapsed * 1000:.0f}ms")

            render_prob_bars(prediction["all_probs"], prediction["top1_idx"])

            if is_confident:
                render_guide(prediction)
else:
    with col2:
        st.subheader("🔍 识别结果")
        st.info("👈 请先上传一张图片或拍照")
        st.markdown(
            """
            <div style='
                background: #f0f0f0;
                border-radius: 16px;
                padding: 3rem 1rem;
                text-align: center;
                color: #aaa;
            '>
                <div style='font-size: 4rem;'>📸</div>
                <div style='font-size: 1.1rem; margin-top: 1rem;'>
                    上传图片后<br>这里将显示识别结果
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# 底部
st.divider()
st.caption("YOLOv8n-cls + ECA | 垃圾四分类模型 | imgsz=320")