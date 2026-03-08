import os
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


# =========================
# CONFIG
# =========================
CKPT_PATH = r"runs_efficientnetv2\20260225_213713\trial_01\checkpoint_final.pth"
DATA_DIR  = "data_rank"
SPLIT     = "test"   # หรือ "val"
BATCH_SIZE = 64
IMG_SIZE = 224
NUM_WORKERS = 2

OUT_DIR = "eval_outputs"  # โฟลเดอร์เก็บรูป/รายงาน

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# IMPORTANT: ล็อค mapping ให้ตรงกับ ckpt["class_names"]
# =========================
class ImageFolderFixed(datasets.ImageFolder):
    def __init__(self, root, transform, class_to_idx_fixed):
        super().__init__(root, transform=transform)

        missing = [c for c in class_to_idx_fixed.keys() if c not in self.class_to_idx]
        if missing:
            raise ValueError(f"Missing class folders in {root}: {missing}")

        # remap sample targets ให้ตรงกับ class_to_idx_fixed
        new_samples = []
        for path, old_y in self.samples:
            cls_name = self.classes[old_y]          # class name ตาม ImageFolder (alphabetical)
            new_y = class_to_idx_fixed[cls_name]    # index ตาม ckpt
            new_samples.append((path, new_y))

        self.samples = new_samples
        self.targets = [y for _, y in new_samples]
        self.classes = list(class_to_idx_fixed.keys())
        self.class_to_idx = class_to_idx_fixed


# =========================
# PREPROCESS: ตามที่คุณสั่ง (ไม่มี Normalize)
# =========================
def build_eval_transform_no_norm():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),  # ไม่มี Normalize
    ])


@torch.no_grad()
def run_eval(model, loader):
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(y.tolist())
    return y_true, y_pred


# =========================
# SAVE HELPERS
# =========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_text(text: str, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def save_cm_csv(cm: np.ndarray, class_names, path: str):
    ensure_dir(os.path.dirname(path))
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + list(class_names))
        for i, row in enumerate(cm.tolist()):
            w.writerow([class_names[i]] + row)

def plot_and_save_confusion_matrix(cm: np.ndarray, class_names, out_path: str, title: str, normalize: bool):
    ensure_dir(os.path.dirname(out_path))

    cm_plot = cm.astype(np.float64)

    if normalize:
        row_sum = cm_plot.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        cm_plot = cm_plot / row_sum

    fig = plt.figure(figsize=(12, 10), dpi=160)
    ax = fig.gca()

    im = ax.imshow(cm_plot)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    # ตัวเลขในแต่ละช่อง
    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            txt = f"{cm_plot[i, j]:.2f}" if normalize else str(int(cm_plot[i, j]))
            ax.text(j, i, txt, ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(CKPT_PATH)

    split_dir = os.path.join(DATA_DIR, SPLIT)
    if not os.path.exists(split_dir):
        raise FileNotFoundError(split_dir)

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

    model_name = ckpt["model_name"]
    class_names = ckpt["class_names"]
    num_classes = len(class_names)

    print("CKPT model_name:", model_name)
    print("CKPT classes   :", class_names)
    print("DEVICE         :", DEVICE)

    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    class_to_idx_fixed = {c: i for i, c in enumerate(class_names)}

    eval_tf = build_eval_transform_no_norm()
    ds = ImageFolderFixed(split_dir, transform=eval_tf, class_to_idx_fixed=class_to_idx_fixed)

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    y_true, y_pred = run_eval(model, loader)

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")

    report = classification_report(y_true, y_pred, target_names=class_names, digits=6)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n=== EVAL ({SPLIT.upper()}) FROM FINAL CKPT (NO NORMALIZE) ===")
    print("Accuracy:", acc)
    print("Macro F1:", f1m)

    print("\nClassification Report:")
    print(report)

    print("\nConfusion Matrix:")
    print(cm)

    # =========================
    # SAVE OUTPUTS
    # =========================
    tag = SPLIT.lower()
    ensure_dir(OUT_DIR)

    # 1) report txt
    save_text(report, os.path.join(OUT_DIR, f"classification_report_{tag}.txt"))

    # 2) confusion matrix csv
    save_cm_csv(cm, class_names, os.path.join(OUT_DIR, f"confusion_matrix_{tag}.csv"))

    # 3) confusion matrix png (raw)
    plot_and_save_confusion_matrix(
        cm,
        class_names,
        out_path=os.path.join(OUT_DIR, f"confusion_matrix_{tag}.png"),
        title=f"Confusion Matrix ({SPLIT.upper()})",
        normalize=False
    )

    # 4) confusion matrix png (normalized by row)
    plot_and_save_confusion_matrix(
        cm,
        class_names,
        out_path=os.path.join(OUT_DIR, f"confusion_matrix_{tag}_norm.png"),
        title=f"Confusion Matrix Normalized ({SPLIT.upper()})",
        normalize=True
    )

    # 5) summary txt
    summary = (
        f"CKPT: {CKPT_PATH}\n"
        f"SPLIT: {SPLIT}\n"
        f"Accuracy: {acc}\n"
        f"Macro F1: {f1m}\n"
        f"Classes: {class_names}\n"
    )
    save_text(summary, os.path.join(OUT_DIR, f"summary_{tag}.txt"))

    print(f"\nSaved outputs to: {OUT_DIR}/")
    print(f"- classification_report_{tag}.txt")
    print(f"- confusion_matrix_{tag}.csv")
    print(f"- confusion_matrix_{tag}.png")
    print(f"- confusion_matrix_{tag}_norm.png")
    print(f"- summary_{tag}.txt")


if __name__ == "__main__":
    main()