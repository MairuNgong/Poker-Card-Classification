import os
import re
import csv
import shutil
import random
from pathlib import Path
from collections import defaultdict

SOURCE_DIR = Path("augmented_poker_data")     # โฟลเดอร์ข้อมูล "ปัจจุบัน" ของคุณ
OUTPUT_DIR = Path("data")     # โฟลเดอร์ผลลัพธ์ที่แยก train/val/test

CLASSES = ["spade", "heart", "diamond", "club"]

# สัดส่วน split
SPLIT = {"train": 0.8, "val": 0.1, "test": 0.1}

# เลือกโหมดการอ่าน label
# MODE = "class_subfolders"  # เคส 1: SOURCE_DIR/spades/*.jpg ...
# MODE = "flat_filename"     # เคส 2: SOURCE_DIR/*.jpg และชื่อไฟล์มีคำว่า spades/hearts/...
# MODE = "csv_labels"        # เคส 3: มี labels.csv ใน SOURCE_DIR
MODE = "class_subfolders"

# ถ้า MODE="csv_labels" ตั้งชื่อไฟล์ CSV
CSV_NAME = "labels.csv"  # format: filename,label

# นามสกุลรูปที่รับ
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# seed ให้ผลนิ่ง
SEED = 42
random.seed(SEED)


# -----------------------------
# Helper
# -----------------------------
def ensure_dirs():
    for split in SPLIT.keys():
        for c in CLASSES:
            (OUTPUT_DIR / split / c).mkdir(parents=True, exist_ok=True)

def copy_to_split(items_by_class):
    """
    items_by_class: dict[class] -> list[Path]
    ทำ stratified split แบบง่าย: แยกตามคลาสแล้วสุ่ม+แบ่งสัดส่วน
    """
    ensure_dirs()

    for c, paths in items_by_class.items():
        if c not in CLASSES:
            continue

        paths = list(paths)
        random.shuffle(paths)

        n = len(paths)
        n_train = int(n * SPLIT["train"])
        n_val = int(n * SPLIT["val"])
        # ที่เหลือเป็น test
        train_paths = paths[:n_train]
        val_paths = paths[n_train:n_train + n_val]
        test_paths = paths[n_train + n_val:]

        print(f"[{c}] total={n} train={len(train_paths)} val={len(val_paths)} test={len(test_paths)}")

        for split_name, split_paths in [("train", train_paths), ("val", val_paths), ("test", test_paths)]:
            for p in split_paths:
                dst = OUTPUT_DIR / split_name / c / p.name
                shutil.copy2(p, dst)

def read_class_subfolders():
    """
    SOURCE_DIR/
      spades/*.jpg
      hearts/*.jpg
      ...
    """
    items = defaultdict(list)
    for c in CLASSES:
        class_dir = SOURCE_DIR / c
        if not class_dir.exists():
            continue
        for p in class_dir.rglob("*"):
            if p.suffix.lower() in IMG_EXTS and p.is_file():
                items[c].append(p)
    return items

def read_flat_filename():
    """
    SOURCE_DIR/*.jpg
    label จากชื่อไฟล์ เช่น:
      img_001_spades.jpg
      hearts-123.png
    """
    items = defaultdict(list)
    pattern = re.compile(r"(spade|heart|diamond|club)", re.IGNORECASE)

    for p in SOURCE_DIR.rglob("*"):
        if p.suffix.lower() in IMG_EXTS and p.is_file():
            m = pattern.search(p.name)
            if not m:
                continue
            label = m.group(1).lower()
            items[label].append(p)
    return items

def read_csv_labels():
    """
    SOURCE_DIR/labels.csv  (filename,label)
    โดย filename อ้างถึงไฟล์ใน SOURCE_DIR หรือ subfolders
    """
    csv_path = SOURCE_DIR / CSV_NAME
    if not csv_path.exists():
        raise FileNotFoundError(f"Not found: {csv_path}")

    items = defaultdict(list)
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    # รองรับมี/ไม่มี header
    # ถ้าแถวแรกมีคำว่า filename/label ให้ข้าม
    start_idx = 1 if rows and ("filename" in rows[0][0].lower() or "label" in rows[0][-1].lower()) else 0

    for r in rows[start_idx:]:
        if len(r) < 2:
            continue
        fname, label = r[0].strip(), r[1].strip().lower()
        if label not in CLASSES:
            continue

        # หาไฟล์จากชื่อ
        candidate = SOURCE_DIR / fname
        if not candidate.exists():
            # เผื่ออยู่ subfolder: ค้นหาแบบ rglob
            found = list(SOURCE_DIR.rglob(fname))
            if not found:
                continue
            candidate = found[0]

        if candidate.suffix.lower() in IMG_EXTS and candidate.is_file():
            items[label].append(candidate)

    return items


def main():
    if OUTPUT_DIR.exists():
        # ลบทิ้งก่อนเพื่อกันข้อมูลค้าง
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if MODE == "class_subfolders":
        items = read_class_subfolders()
    elif MODE == "flat_filename":
        items = read_flat_filename()
    elif MODE == "csv_labels":
        items = read_csv_labels()
    else:
        raise ValueError("MODE must be one of: class_subfolders | flat_filename | csv_labels")

    # ตรวจว่าแต่ละคลาสมีรูปไหม
    for c in CLASSES:
        print(f"Found {len(items.get(c, []))} images for class '{c}'")

    copy_to_split(items)

    print("\n✅ Done! Output structure:")
    print(OUTPUT_DIR.resolve())
    print("Now use train/val/test from data_out/ with your training script.")

if __name__ == "__main__":
    main()
