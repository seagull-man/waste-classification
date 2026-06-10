# convert_to_4class.py
import os
import shutil
from pathlib import Path

DATASET_ROOT = Path("/root/autodl-tmp/garbage_datasets")
OUTPUT_ROOT = Path("../garbage_4cls")

ORIGINAL_NAMES = [
    'FastFoodBox', 'SoiledPlastic', 'Cigarette', 'Toothpick', 'Flowerpot',
    'BambooChopstics', 'Meal', 'Bone', 'FruitPeel', 'Pulp', 'Tea', 'Vegetable',
    'Eggshell', 'FishBone', 'Powerbank', 'Bag', 'CosmeticBottles', 'Toys',
    'PlasticBowl', 'PlasticHanger', 'PaperBags', 'PlugWire', 'OldClothes',
    'Can', 'Pillow', 'PlushToys', 'ShampooBottle', 'GlassCup', 'Shoes',
    'Anvil', 'Cardboard', 'SeasoningBottle', 'Bottle', 'MetalFoodCans',
    'Pot', 'EdibleOilBarrel', 'DrinkBottle', 'DryBattery', 'Ointment',
    'ExpiredDrugs'
]

CATEGORY_MAPPING = {}
for name in ORIGINAL_NAMES:
    if name in ['DryBattery', 'Ointment', 'ExpiredDrugs']:
        CATEGORY_MAPPING[name] = 'hazardous'
    elif name in ['Meal', 'Bone', 'FruitPeel', 'Pulp', 'Tea', 'Vegetable', 'Eggshell', 'FishBone']:
        CATEGORY_MAPPING[name] = 'kitchen'
    elif name in ['FastFoodBox', 'SoiledPlastic', 'Cigarette', 'Toothpick', 'Flowerpot', 'BambooChopstics']:
        CATEGORY_MAPPING[name] = 'other'
    else:
        CATEGORY_MAPPING[name] = 'recyclable'

def convert_split(split):
    images_dir = DATASET_ROOT / "datasets" / "images" / split
    labels_dir = DATASET_ROOT / "datasets" / "labels" / split
    output_dir = OUTPUT_ROOT / split

    for cls in ['recyclable', 'hazardous', 'kitchen', 'other']:
        (output_dir / cls).mkdir(parents=True, exist_ok=True)

    label_files = list(labels_dir.glob("*.txt"))
    total = len(label_files)  # <-- 这行必须有！
    print(f"[INFO] 开始处理 {split} 集，共 {total} 张图片...")

    count = 0
    for i, label_file in enumerate(label_files):
        if i % 500 == 0:
            print(f"    已处理 {i}/{total} 张...")

        image_stem = label_file.stem
        image_file = None
        for ext in ['.jpg', '.jpeg', '.png']:
            candidate = images_dir / (image_stem + ext)
            if candidate.exists():
                image_file = candidate
                break
        if not image_file:
            continue

        with open(label_file, 'r') as f:
            line = f.readline().strip()
            if not line:
                continue
            cls_id = int(line.split()[0])
            if cls_id >= len(ORIGINAL_NAMES):
                continue

        original_class = ORIGINAL_NAMES[cls_id]
        target_class = CATEGORY_MAPPING[original_class]
        dest = output_dir / target_class / image_file.name
        shutil.copy(image_file, dest)
        count += 1

    print(f"[DONE] {split} 集完成！成功处理 {count} / {total} 张图片\n")

if __name__ == "__main__":
    convert_split("train")
    convert_split("val")
    print("[SUCCESS] 四分类数据集已生成到 'garbage_4cls' 文件夹！")