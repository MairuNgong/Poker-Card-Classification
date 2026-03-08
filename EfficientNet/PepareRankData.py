# PepareRankData.py (FIXED)
import shutil
import random
import re
from pathlib import Path
from collections import defaultdict

# -----------------------------
# CONFIG
# -----------------------------
SOURCE_DIR = Path("augmented_poker_data")
OUTPUT_DIR = Path("data_rank")

SUITS = ["spade", "heart", "diamond", "club"]
RANKS = ["2","3","4","5","6","7","8","9","10","a","j","q","k"]

SPLIT = {"train": 0.8, "val": 0.1, "test": 0.1}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

SEED = 42
random.seed(SEED)

# ✅ IMPORTANT:
# สร้าง "group key" เพื่อกันรูปที่มาจากต้นฉบับเดียวกันกระจายคนละ split
# คุณต้องปรับกติกา strip suffix ให้เข้ากับชื่อไฟล์ augment ของคุณ
AUG_SUFFIX_PATTERNS = [
    r"_aug\d+$",
    r"_flip$", r"_fliplr$", r"_flipud$",
    r"_rot\d+$",
    r"_bright\d+$", r"_brightness\d+$",
    r"_contrast\d+$",
    r"_noise\d+$",
    r"_zoom\d+$",
    r"_shift\d+$",
    r"_shear\d+$",
]

def base_id_from_filename(p: Path) -> str:
    """
    พยายามดึง id ของ "ต้นฉบับ" จากชื่อไฟล์
    เช่น img123_rot10_aug3.png -> img123
    """
    stem = p.stem
    changed = True
    while changed:
        changed = False
        for pat in AUG_SUFFIX_PATTERNS:
            new_stem = re.sub(pat, "", stem)
            if new_stem != stem:
                stem = new_stem
                changed = True
    return stem


def ensure_dirs():
    for split in SPLIT.keys():
        for r in RANKS:
            (OUTPUT_DIR / split / r).mkdir(parents=True, exist_ok=True)


def read_all_images_grouped():
    """
    อ่านรูปจาก:
      SOURCE_DIR/suit/rank/*.jpg
    แล้วจัดกลุ่มเป็น:
      dict[rank][group_id] -> list[Path]
    """
    by_rank_group = {r: defaultdict(list) for r in RANKS}

    for suit in SUITS:
        suit_dir = SOURCE_DIR / suit
        if not suit_dir.exists():
            print(f"⚠️ missing suit folder: {suit_dir}")
            continue

        for rank in RANKS:
            rank_dir = suit_dir / rank
            if not rank_dir.exists():
                continue

            for p in rank_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    base_id = base_id_from_filename(p)
                    # ✅ group_id ผูก suit+rank+base_id เพื่อกันข้าม suit/rank
                    group_id = f"{suit}__{rank}__{base_id}"
                    by_rank_group[rank][group_id].append(p)

    return by_rank_group


def split_groups(group_ids):
    """
    split ที่ระดับ group (ไม่ใช่ระดับไฟล์)
    """
    group_ids = list(group_ids)
    random.shuffle(group_ids)

    n = len(group_ids)
    n_train = int(n * SPLIT["train"])
    n_val = int(n * SPLIT["val"])

    g_train = set(group_ids[:n_train])
    g_val   = set(group_ids[n_train:n_train + n_val])
    g_test  = set(group_ids[n_train + n_val:])

    return g_train, g_val, g_test


def copy_groups(by_rank_group):
    ensure_dirs()

    for r in RANKS:
        groups = by_rank_group.get(r, {})
        group_ids = list(groups.keys())

        g_train, g_val, g_test = split_groups(group_ids)

        # report เป็นจำนวน "กลุ่ม" (สำคัญกว่า) และจำนวนไฟล์
        def count_files(gset):
            return sum(len(groups[gid]) for gid in gset)

        print(
            f"[rank {r}] groups total={len(group_ids)} | "
            f"train={len(g_train)}(files={count_files(g_train)}) "
            f"val={len(g_val)}(files={count_files(g_val)}) "
            f"test={len(g_test)}(files={count_files(g_test)})"
        )

        for split_name, gset in [("train", g_train), ("val", g_val), ("test", g_test)]:
            for gid in gset:
                for p in groups[gid]:
                    # ✅ กันชื่อซ้ำ/ทับกัน: ใส่ suit+base_id+ชื่อเดิม
                    suit, rank, base_id = gid.split("__", 2)
                    safe_name = f"{suit}__{base_id}__{p.name}"
                    dst = OUTPUT_DIR / split_name / r / safe_name
                    shutil.copy2(p, dst)


def main():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    by_rank_group = read_all_images_grouped()

    # report จำนวน group
    for r in RANKS:
        print(f"Found {len(by_rank_group[r])} groups for rank '{r}'")

    copy_groups(by_rank_group)

    print("\n✅ Done! Output structure:")
    print(OUTPUT_DIR.resolve())
    print("Use data_rank/train|val|test/<rank>/... for training.")


if __name__ == "__main__":
    main()