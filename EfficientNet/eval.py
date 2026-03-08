import os
from pathlib import Path
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def save_confusion_matrix_png(y_true, y_pred, class_names, out_path, normalize=False,
                              title="Confusion Matrix", dpi=250):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

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

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm_show[i, j]*100:.1f}%" if normalize else str(int(cm[i, j]))
            plt.text(j, i, txt, ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()

def main():
    SCRIPT_DIR = Path(__file__).resolve().parent
    DATA_DIR = SCRIPT_DIR / "data"
    TEST_DIR = DATA_DIR / "test"
    CKPT_PATH = SCRIPT_DIR / "suit_classification.pth"
    OUT_DIR = SCRIPT_DIR / "outputs"

    BATCH_SIZE = 64
    IMG_SIZE = 224
    NUM_WORKERS = 2  # ใช้ได้แล้ว เพราะอยู่ใน main
    MODEL_NAME = "tf_efficientnetv2_s"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print("SCRIPT_DIR:", SCRIPT_DIR)
    print("DATA_DIR:  ", DATA_DIR)
    print("TEST_DIR:  ", TEST_DIR)
    print("CKPT PATH: ", CKPT_PATH)
    print("OUT_DIR:   ", OUT_DIR)

    if not TEST_DIR.exists():
        raise FileNotFoundError(f"TEST_DIR not found: {TEST_DIR}")
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
    ])

    test_ds = datasets.ImageFolder(str(TEST_DIR), transform=eval_tf)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
    )

    ckpt = torch.load(str(CKPT_PATH), map_location=DEVICE)
    ckpt_model_name = ckpt.get("model_name", MODEL_NAME)
    class_names = ckpt.get("class_names", test_ds.classes)
    num_classes = len(class_names)

    model = timm.create_model(ckpt_model_name, pretrained=False, num_classes=num_classes).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(y.tolist())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    report = classification_report(all_labels, all_preds, target_names=class_names)
    cm = confusion_matrix(all_labels, all_preds)

    print("\n=== SUIT TEST RESULTS ===")
    print("Classes (dataset):", test_ds.classes)
    print("Classes (ckpt):   ", class_names)
    print("Accuracy:", acc)
    print("Macro F1:", macro_f1)
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", cm)

    save_confusion_matrix_png(
        all_labels, all_preds, class_names,
        out_path=OUT_DIR / "suit_confusion_matrix.png",
        normalize=False,
        title="Suit Confusion Matrix (Counts)"
    )
    save_confusion_matrix_png(
        all_labels, all_preds, class_names,
        out_path=OUT_DIR / "suit_confusion_matrix_norm.png",
        normalize=True,
        title="Suit Confusion Matrix (Row %)"
    )

    print("\nSaved to:", OUT_DIR)

if __name__ == "__main__":
    # ถ้าไม่ทำบรรทัดนี้บน Windows แล้ว num_workers>0 มักจะระเบิด
    import multiprocessing as mp
    mp.freeze_support()
    main()