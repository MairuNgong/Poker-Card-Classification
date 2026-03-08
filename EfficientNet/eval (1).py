# =========================
# TEST (SUIT) : 4 classes + Save Confusion Matrix PNG
# =========================
import os
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)

# -------------------------
# CONFIG
# -------------------------
DATA_DIR = "data_suit"                       # <-- แก้ให้ตรงกับของคุณ
TEST_DIR = os.path.join(DATA_DIR, "test")

BATCH_SIZE = 64
IMG_SIZE = 224
NUM_WORKERS = 2

MODEL_NAME = "tf_efficientnetv2_s"           # <-- ต้องตรงกับตอนเทรน suit
SUIT_MODEL_PATH = "suit_classification.pth"  # <-- checkpoint suit ของคุณ

OUT_DIR = "outputs"
CM_PNG_PATH = os.path.join(OUT_DIR, "suit_confusion_matrix.png")
CM_PNG_NORM_PATH = os.path.join(OUT_DIR, "suit_confusion_matrix_norm.png")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------
# PREPROCESS (เหมือนที่คุณใช้: no normalize)
# -------------------------
eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
])


# -------------------------
# CONFUSION MATRIX SAVER (PNG)
# -------------------------
def save_confusion_matrix_png(
    y_true,
    y_pred,
    class_names,
    out_path,
    normalize=False,          # True = % ต่อแถว (recall-based)
    title="Confusion Matrix",
    dpi=250
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_show = cm.astype(float)

    if normalize:
        row_sums = cm_show.sum(axis=1, keepdims=True)
        cm_show = np.divide(cm_show, row_sums, out=np.zeros_like(cm_show), where=row_sums != 0)

    plt.figure(figsize=(7, 7))
    plt.imshow(cm_show)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)

    # ใส่ตัวเลขในช่อง
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalize:
                txt = f"{cm_show[i, j]*100:.1f}%"
            else:
                txt = str(int(cm[i, j]))
            plt.text(j, i, txt, ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()

    return cm


# -------------------------
# DATASET / LOADER
# -------------------------
test_ds = datasets.ImageFolder(TEST_DIR, transform=eval_tf)
test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE == "cuda"),
)

# -------------------------
# LOAD CKPT
# -------------------------
ckpt = torch.load(SUIT_MODEL_PATH, map_location=DEVICE)

# ถ้า ckpt มี model_name / class_names จะใช้จาก ckpt เป็นหลัก
ckpt_model_name = ckpt.get("model_name", MODEL_NAME)
class_names = ckpt.get("class_names", test_ds.classes)
NUM_CLASSES = len(class_names)

# -------------------------
# MODEL
# -------------------------
model = timm.create_model(
    ckpt_model_name,
    pretrained=False,
    num_classes=NUM_CLASSES
).to(DEVICE)

model.load_state_dict(ckpt["model_state"])
model.eval()


# -------------------------
# EVAL
# -------------------------
all_preds, all_labels = [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(y.tolist())

# -------------------------
# METRICS
# -------------------------
acc = accuracy_score(all_labels, all_preds)
macro_f1 = f1_score(all_labels, all_preds, average="macro")
report = classification_report(all_labels, all_preds, target_names=class_names)
cm = confusion_matrix(all_labels, all_preds)

print("\n=== SUIT TEST RESULTS ===")
print("Classes (dataset):", test_ds.classes)
print("Classes (ckpt):   ", class_names)
print("Accuracy:", acc)
print("Macro F1:", macro_f1)

print("\nClassification Report:")
print(report)

print("\nConfusion Matrix:")
print(cm)

# -------------------------
# SAVE CONFUSION MATRIX IMAGES
# -------------------------
_ = save_confusion_matrix_png(
    all_labels, all_preds, class_names,
    out_path=CM_PNG_PATH,
    normalize=False,
    title="Suit Confusion Matrix (Counts)"
)

_ = save_confusion_matrix_png(
    all_labels, all_preds, class_names,
    out_path=CM_PNG_NORM_PATH,
    normalize=True,
    title="Suit Confusion Matrix (Row %)"
)

print("\nSaved Confusion Matrix PNG:")
print(" -", CM_PNG_PATH)
print(" -", CM_PNG_NORM_PATH)