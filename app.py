# app.py - 智能垃圾分类可视化系统（四分类专用）
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import tempfile

# ================== 页面配置 ==================
st.set_page_config(
    page_title="智能垃圾分类系统",
    page_icon="🗑️",
    layout="centered"
)

st.title("🗑️ 智能垃圾分类系统")
st.markdown("基于 YOLOv8 的四分类垃圾识别模型")

# ================== 加载模型 ==================
@st.cache_resource
def load_model():
    # ⚠️ 请确保这个路径指向你训练好的四分类 best.pt
    model_path = r"runs\classify\waste_cls_original-2\weights\best.pt"
    if not os.path.exists(model_path):
        st.error(f"❌ 模型文件不存在，请检查路径：\n{os.path.abspath(model_path)}")
        st.stop()
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"❌ 模型加载失败：{e}")
        st.stop()

model = load_model()

# ================== 四大类中文映射 ==================
# 注意：YOLOv8 四分类模型输出的是 ['hazardous', 'kitchen', 'recyclable', 'other']
CHINESE_MAP = {
    'hazardous': '有害垃圾',
    'kitchen': '厨余垃圾',
    'recyclable': '可回收物',
    'other': '其他垃圾'
}

TIPS = {
    "有害垃圾": "请投入 **红色垃圾桶**，避免破损泄漏。",
    "厨余垃圾": "请投入 **绿色垃圾桶**，沥干水分、去除包装。",
    "可回收物": "请投入 **蓝色垃圾桶**，保持清洁干燥。",
    "其他垃圾": "请投入 **灰色/黑色垃圾桶**。"
}

COLORS = {
    "有害垃圾": "#F44336",   # 红
    "厨余垃圾": "#FF9800",   # 橙
    "可回收物": "#4CAF50",   # 绿
    "其他垃圾": "#9E9E9E"    # 灰
}

# ================== 上传与预测 ==================
uploaded_file = st.file_uploader(
    "📤 上传一张垃圾图片（JPG/PNG）",
    type=["jpg", "jpeg", "png"],
    help="支持手机拍摄的真实垃圾照片"
)

if uploaded_file is not None:
    # 显示原图
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 你上传的图片", use_container_width=True)

    # 保存临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    # 预测
    with st.spinner("🧠 AI 正在分析垃圾类型..."):
        results = model.predict(source=temp_path, verbose=False)
        pred_idx = results[0].probs.top1
        pred_class_en = results[0].names[pred_idx]  # 如 'recyclable'
        confidence = float(results[0].probs.top1conf.item())

    # 清理临时文件
    os.unlink(temp_path)

    # 转为中文
    main_class = CHINESE_MAP.get(pred_class_en, "未知类别")

    # 显示结果
    st.success(f"✅ 识别结果：**{main_class}**")
    st.info(f"🔍 英文类别：{pred_class_en} | 置信度：{confidence:.2f}")

    # 投放建议
    tip = TIPS.get(main_class, "请按当地规定分类投放。")
    st.markdown(f"### 💡 投放建议\n{tip}")

    # 彩色标签
    color = COLORS.get(main_class, "#607D8B")
    st.markdown(
        f'<div style="padding:12px; background-color:{color}; color:white; '
        f'border-radius:8px; margin-top:10px; font-weight:bold;">'
        f'📌 最终分类：{main_class}</div>',
        unsafe_allow_html=True
    )

# ================== 底部说明 ==================
st.markdown("---")
st.caption("© 2026 课程设计作品 | 基于 YOLOv8 + Streamlit")