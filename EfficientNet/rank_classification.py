import os
import time
import copy
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import timm

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from datetime import datetime

# =========================
# Reproducibility
# =========================
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # NOTE: deterministic may reduce throughput; turn off if you prefer speed over exact repeatability
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import csv

from timm.data import resolve_data_config, create_transform


class SimpleLogger:
    """เขียนลงไฟล์อย่างเดียว (ไม่ print)"""
    def __init__(self, log_path):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_path = log_path
        self.f = open(log_path, "a", encoding="utf-8")
        # ✅ FIX: datetime.now() ใช้งานได้แล้วเพราะ import ถูก
        self.file(f"\n==== RUN @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ====\n")

    def file(self, msg: str):
        self.f.write(msg + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


class LiveHistory:
    """
    เก็บ history แล้ว:
    - เซฟเป็น metrics.csv
    - เซฟกราฟเป็น .png ทุก epoch (หรือทุกครั้งที่ plot)
    """
    def __init__(self, out_dir, title=""):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        self.title = title
        self.rows = []  # list of dict

        self.csv_path = os.path.join(self.out_dir, "metrics.csv")
        self.png_path = os.path.join(self.out_dir, "plot.png")

        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_acc", "val_f1"])
                w.writeheader()

    def add(self, epoch, train_loss, val_loss, val_acc, val_f1):
        row = {
            "epoch": int(epoch),
            "train_loss": float(train_loss) if train_loss is not None else None,
            "val_loss": float(val_loss) if val_loss is not None else None,
            "val_acc": float(val_acc) if val_acc is not None else None,
            "val_f1": float(val_f1) if val_f1 is not None else None,
        }
        self.rows.append(row)

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_acc", "val_f1"])
            w.writerow(row)

    def plot(self, save=True, show=True):
        if len(self.rows) == 0:
            return

        epochs = [r["epoch"] for r in self.rows]
        train_loss = [r["train_loss"] for r in self.rows]
        val_loss = [r["val_loss"] for r in self.rows]
        val_acc  = [r["val_acc"]  for r in self.rows]
        val_f1   = [r["val_f1"]   for r in self.rows]

        out_dir = os.path.dirname(self.png_path) if hasattr(self, "png_path") else getattr(self, "out_dir", ".")
        os.makedirs(out_dir, exist_ok=True)

        loss_path = os.path.join(out_dir, "plot_loss.png")
        metric_path = os.path.join(out_dir, "plot_metrics.png")

        def beautify_axis(ax):
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.grid(True, which="major", linestyle="-", alpha=0.35)
            ax.grid(True, which="minor", linestyle=":", alpha=0.25)

        fig1 = plt.figure(figsize=(10, 4.5), dpi=120)
        ax1 = fig1.gca()

        ax1.plot(epochs, train_loss, label="train_loss")
        if any(v is not None for v in val_loss):
            ax1.plot(epochs, val_loss, label="val_loss")

        ax1.set_title(f"{self.title} | Loss")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("loss")
        ax1.legend()
        beautify_axis(ax1)
        fig1.tight_layout()

        if save:
            fig1.savefig(loss_path, dpi=200, bbox_inches="tight")
        if show:
            fig1.show()
        plt.close(fig1)

        fig2 = plt.figure(figsize=(10, 4.5), dpi=120)
        ax2 = fig2.gca()

        if any(v is not None for v in val_acc):
            ax2.plot(epochs, val_acc, label="val_acc")
        if any(v is not None for v in val_f1):
            ax2.plot(epochs, val_f1, label="val_f1")

        ax2.set_title(f"{self.title} | Metrics")
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("score")
        ax2.set_ylim(0, 1.0)
        ax2.legend()
        beautify_axis(ax2)
        fig2.tight_layout()

        if save:
            fig2.savefig(metric_path, dpi=200, bbox_inches="tight")
        if show:
            fig2.show()
        plt.close(fig2)

def save_trial_checkpoint(trial_dir, model, cfg, class_names, model_name, img_size,
                          phaseA_best_f1, phaseB_best_f1, final_val_f1, tag="final"):
    ckpt_path = os.path.join(trial_dir, f"checkpoint_{tag}.pth")
    torch.save({
        "model_name": model_name,
        "model_state": model.state_dict(),
        "class_names": class_names,
        "img_size": img_size,
        "cfg": cfg,
        "phaseA_best_f1": float(phaseA_best_f1),
        "phaseB_best_f1": float(phaseB_best_f1),
        "final_val_f1": float(final_val_f1),
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }, ckpt_path)
    return ckpt_path


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_last_blocks_efficientnetv2(model, n_blocks: int):
    freeze_all(model)

    for name, p in model.named_parameters():
        if any(k in name.lower() for k in ["classifier", "head", "fc"]):
            p.requires_grad = True

    if hasattr(model, "blocks"):
        blocks = model.blocks
        total = len(blocks)
        n = max(0, min(n_blocks, total))
        for i in range(total - n, total):
            for p in blocks[i].parameters():
                p.requires_grad = True
        return f"Unfroze last {n}/{total} blocks + head"
    else:
        for p in model.parameters():
            p.requires_grad = True
        return "WARNING: model has no .blocks; unfroze ALL parameters"


@torch.no_grad()
def evaluate(model, loader, device, return_details: bool = False):
    """Evaluate model on a dataloader.

    return_details=True will also return (y_true, y_pred) for reporting.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_preds, all_labels = [], []
    total_loss = 0.0
    n_total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            n_total += x.size(0)

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(y.detach().cpu().tolist())

    avg_loss = total_loss / max(1, n_total)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    if return_details:
        return avg_loss, acc, f1, all_labels, all_preds
    return avg_loss, acc, f1


def clear_console():
    """FIX: ใช้แทน clear_output สำหรับ .py"""
    os.system("cls" if os.name == "nt" else "clear")


def train_one_phase(
    model, loader, device, optimizer, epochs,
    val_loader=None, train_eval_loader=None, history=None, phase_name="", trial_name="",
    patience=5, min_delta=0.0, verbose=True,
    print_every=50,
    logger=None,
    save_every_epoch_plot=True
):
    criterion = nn.CrossEntropyLoss()
    best_state = copy.deepcopy(model.state_dict())
    best_val_f1 = -1.0
    bad_epochs = 0

    total_batches = len(loader)

    def log_file(msg: str):
        if logger is not None:
            logger.file(msg)

    log_file(f"== {trial_name} | {phase_name} | epochs={epochs} ==")

    for ep in range(1, epochs + 1):
        clear_console()
        print(f"[{trial_name}] {phase_name} | epoch {ep}/{epochs}")

        model.train()
        total_loss = 0.0
        n_total = 0
        ep_t0 = time.time()

        for b_idx, (x, y) in enumerate(loader, start=1):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            n_total += x.size(0)

            if (b_idx == 1) or (b_idx % print_every == 0) or (b_idx == total_batches):
                print(f"  batch {b_idx:04d}/{total_batches:04d} | loss={loss.item():.4f}")
                log_file(f"[ep {ep:02d}] batch {b_idx:04d}/{total_batches:04d} | loss={loss.item():.6f}")

        train_loss = total_loss / max(1, n_total)
        ep_sec = time.time() - ep_t0

        train_eval_loss = train_eval_acc = train_eval_f1 = None
        if train_eval_loader is not None:
            train_eval_loss, train_eval_acc, train_eval_f1 = evaluate(model, train_eval_loader, device)

        if val_loader is not None:
            val_loss, val_acc, val_f1 = evaluate(model, val_loader, device)

            improved = False
            if val_f1 > best_val_f1 + min_delta:
                best_val_f1 = val_f1
                best_state = copy.deepcopy(model.state_dict())
                bad_epochs = 0
                improved = True
            else:
                bad_epochs += 1

            if history is not None:
                history.title = f"{trial_name} | {phase_name}"
                history.add(ep, train_loss, val_loss, val_acc, val_f1)
                if save_every_epoch_plot:
                    history.plot(save=True, show=True)

            extra = ""
            if train_eval_loss is not None:
                extra = f"train_eval_loss={train_eval_loss:.4f} acc={train_eval_acc:.4f} f1={train_eval_f1:.4f} | "

            print(
                f"  >>> ep {ep:02d}/{epochs} DONE | train_loss={train_loss:.4f} | "
                f"{extra}"
                f"val_loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f} | "
                f"bad={bad_epochs}/{patience} | {ep_sec:.1f}s" + (" Improve" if improved else "")
            )

            extra_log = ""
            if train_eval_loss is not None:
                extra_log = f" train_eval_loss={train_eval_loss:.6f} train_eval_acc={train_eval_acc:.6f} train_eval_f1={train_eval_f1:.6f}"

            log_file(
                f"[ep {ep:02d}] train_loss={train_loss:.6f}{extra_log} val_loss={val_loss:.6f} "
                f"acc={val_acc:.6f} f1={val_f1:.6f} bad={bad_epochs}/{patience} sec={ep_sec:.2f}"
                + (" IMPROVED" if improved else "")
            )

            if bad_epochs >= patience:
                if verbose:
                    print(f" Early stop at ep {ep} (best val_f1={best_val_f1:.4f})")
                    log_file(f"EARLY_STOP at ep={ep} best_val_f1={best_val_f1:.6f}")
                break
        else:
            if history is not None:
                history.title = f"{trial_name} | {phase_name}"
                history.add(ep, train_loss, None, None, None)
                if save_every_epoch_plot:
                    history.plot(save=True, show=True)

            print(f"  >>> ep {ep:02d}/{epochs} DONE | train_loss={train_loss:.4f} | {ep_sec:.1f}s")
            log_file(f"[ep {ep:02d}] train_loss={train_loss:.6f} sec={ep_sec:.2f}")

    model.load_state_dict(best_state)
    log_file(f"== DONE {trial_name} | {phase_name} | best_val_f1={best_val_f1:.6f} ==")
    return best_val_f1


# ✅ FIX (สำคัญบน Windows ถ้าใช้ num_workers>0): ครอบใน main
if __name__ == "__main__":
    set_seed(42)

    DATA_DIR = "data_rank"  # FIX: use leakage-safe split output
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    VAL_DIR   = os.path.join(DATA_DIR, "val")
    TEST_DIR  = os.path.join(DATA_DIR, "test")

    IMG_SIZE = 224
    BATCH_SIZE = 64
    NUM_CLASSES = 13

    EPOCHS_A = 8
    EPOCHS_B = 30

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE =", DEVICE)
    print("torch:", torch.__version__)
    print("torch cuda:", torch.version.cuda)

    SEED = 42
    seed_everything(SEED)
    print("SEED =", SEED)

    # ✅ FIX: กันพังถ้าไม่มี GPU
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        print("capability:", torch.cuda.get_device_capability(0))
    else:
        print("CPU only")

    GRID = [

        # 1) baseline conservative
        {"lr_a": 1e-3, "wd_a": 1e-4, "lr_b": 1e-4, "wd_b": 1e-4, "unfreeze_blocks": 4},

        # 2) slightly larger backbone LR
        {"lr_a": 1e-3, "wd_a": 1e-4, "lr_b": 8e-5, "wd_b": 1e-4, "unfreeze_blocks": 6},

        # 3) LR_B เบาลง (กัน pretrained พัง/กันแกว่ง)
        {"lr_a": 1e-3, "wd_a": 1e-4, "lr_b": 5e-5, "wd_b": 1e-4, "unfreeze_blocks": 8},

        # 4) stronger head learning
        {"lr_a": 2e-3, "wd_a": 1e-4, "lr_b": 5e-5, "wd_b": 1e-4, "unfreeze_blocks": 8},

        # 5) more regularization
        {"lr_a": 1e-3, "wd_a": 5e-4, "lr_b": 5e-5, "wd_b": 1e-4, "unfreeze_blocks": 8},

        # 6) deeper unfreeze
        {"lr_a": 1e-3, "wd_a": 1e-4, "lr_b": 3e-5, "wd_b": 1e-4, "unfreeze_blocks": 10},

        # 7) aggressive backbone learning
        {"lr_a": 1e-3, "wd_a": 1e-4, "lr_b": 1e-4, "wd_b": 1e-4, "unfreeze_blocks": 12},

        # 8) strong regularization + deep unfreeze
        {"lr_a": 1e-3, "wd_a": 5e-4, "lr_b": 5e-5, "wd_b": 5e-4, "unfreeze_blocks": 12},
    ]


    MODEL_NAME = "tf_efficientnetv2_s"

    best_overall = {"val_f1": -1.0, "cfg": None, "state": None}

    print("\n=== GRID SEARCH START ===")
    t0 = time.time()
    import json

    RUN_DIR = os.path.join("runs_efficientnetv2", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(RUN_DIR, exist_ok=True)

    for i, cfg in enumerate(GRID, start=1):
        trial_id  = f"trial_{i:02d}"
        trial_dir = os.path.join(RUN_DIR, trial_id)
        os.makedirs(trial_dir, exist_ok=True)

        # เก็บ cfg ของ trial นี้
        with open(os.path.join(trial_dir, "cfg.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        logger = SimpleLogger(os.path.join(trial_dir, "log.txt"))
        trial_name = f"Trial {i}/{len(GRID)} | cfg={cfg}"
        logger.file(f"\n--- {trial_name} ---")

        # default เผื่อ exception
        best_f1_a = -1.0
        best_f1_b = -1.0
        val_f1 = -1.0
        class_names = None

        try:
            # สร้างโมเดลใหม่ทุก trial
            model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES).to(DEVICE)

            # transform ให้ตรงกับ pretrained
            # data_cfg = resolve_data_config({}, model=model)
            # train_tf = create_transform(**data_cfg, is_training=True)
            # eval_tf  = create_transform(**data_cfg, is_training=False)

            train_tf = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(IMG_SIZE),
                transforms.ToTensor(),
            ])

            eval_tf = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(IMG_SIZE),
                transforms.ToTensor(),
            ])

            train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
            val_ds   = datasets.ImageFolder(VAL_DIR,   transform=eval_tf)
            test_ds  = datasets.ImageFolder(TEST_DIR,  transform=eval_tf)

            class_names = train_ds.classes
            logger.file(f"Classes: {class_names}")
            assert len(class_names) == NUM_CLASSES, f"Expected {NUM_CLASSES} classes, got {len(class_names)}"

            # (Windows แนะนำเริ่ม num_workers=0 ก่อน)
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
            val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
            test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

            # Extra: evaluate on TRAIN without training-time augmentation
            train_eval_ds = datasets.ImageFolder(TRAIN_DIR, transform=eval_tf)
            train_eval_loader = DataLoader(train_eval_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)


            # ---------------- Phase A ----------------
            freeze_all(model)
            for name, p in model.named_parameters():
                if any(k in name.lower() for k in ["classifier", "head", "fc"]):
                    p.requires_grad = True

            logger.file(f"Phase A trainable params: {count_trainable_params(model)}")

            opt_a = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=cfg["lr_a"], weight_decay=cfg["wd_a"]
            )

            histA = LiveHistory(out_dir=os.path.join(trial_dir, "phaseA"),
                                title=f"{trial_id} | Phase A (head-only)")

            best_f1_a = train_one_phase(
                model, train_loader, DEVICE, opt_a,
                epochs=EPOCHS_A, val_loader=val_loader, train_eval_loader=train_eval_loader,
                history=histA,
                phase_name="Phase A (head-only)",
                trial_name=trial_name,
                patience=3, min_delta=0.001,
                logger=logger, print_every=50, save_every_epoch_plot=True
            )
            logger.file(f"Phase A best val_f1={best_f1_a:.6f}")

            # ---------------- Phase B ----------------
            msg = unfreeze_last_blocks_efficientnetv2(model, cfg["unfreeze_blocks"])
            logger.file(f"Phase B unfreeze: {msg}")
            logger.file(f"Phase B trainable params: {count_trainable_params(model)}")

            opt_b = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=cfg["lr_b"], weight_decay=cfg["wd_b"]
            )

            histB = LiveHistory(out_dir=os.path.join(trial_dir, "phaseB"),
                                title=f"{trial_id} | Phase B (unfreeze {cfg['unfreeze_blocks']} blocks)")

            best_f1_b = train_one_phase(
                model, train_loader, DEVICE, opt_b,
                epochs=EPOCHS_B, val_loader=val_loader, train_eval_loader=train_eval_loader,
                history=histB,
                phase_name=f"Phase B (unfreeze {cfg['unfreeze_blocks']} blocks)",
                trial_name=trial_name,
                patience=5, min_delta=0.001,
                logger=logger, print_every=50, save_every_epoch_plot=True
            )
            logger.file(f"Phase B best val_f1={best_f1_b:.6f}")

            # final val หลัง load best_state ของ Phase B แล้ว
            val_loss, val_acc, val_f1 = evaluate(model, val_loader, DEVICE)
            logger.file(f"Final Val: loss={val_loss:.6f} acc={val_acc:.6f} f1={val_f1:.6f}")

            # ✅ เซฟ checkpoint ของ trial นี้ทันที กัน crash
            ckpt_path = save_trial_checkpoint(
                trial_dir, model, cfg, class_names, MODEL_NAME, IMG_SIZE,
                best_f1_a, best_f1_b, val_f1, tag="final"
            )
            logger.file(f"Saved trial checkpoint: {ckpt_path}")

            # Update best overall
            if val_f1 > best_overall["val_f1"]:
                best_overall["val_f1"] = val_f1
                best_overall["cfg"] = cfg
                best_overall["state"] = copy.deepcopy(model.state_dict())
                logger.file("✅ New BEST trial!")

                # เซฟ best_overall ทันที กันพังกลางทาง
                best_path = os.path.join(RUN_DIR, "best_overall.pth")
                torch.save({
                    "model_name": MODEL_NAME,
                    "model_state": best_overall["state"],
                    "class_names": class_names,
                    "img_size": IMG_SIZE,
                    "best_cfg": best_overall["cfg"],
                    "best_val_f1": float(best_overall["val_f1"]),
                    "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }, best_path)
                logger.file(f"Saved BEST overall: {best_path}")

                with open(os.path.join(RUN_DIR, "best_summary.json"), "w", encoding="utf-8") as f:
                    json.dump({
                        "trial_id": trial_id,
                        "val_f1": float(val_f1),
                        "cfg": cfg,
                        "phaseA_best_f1": float(best_f1_a),
                        "phaseB_best_f1": float(best_f1_b),
                    }, f, indent=2)

        except Exception as e:
            logger.file(f"❌ ERROR in {trial_id}: {repr(e)}")

            # พยายามเซฟ “ฉุกเฉิน” ถ้ามี model อยู่
            try:
                if "model" in locals() and class_names is not None:
                    emergency_path = save_trial_checkpoint(
                        trial_dir, model, cfg, class_names, MODEL_NAME, IMG_SIZE,
                        best_f1_a, best_f1_b, val_f1, tag="emergency"
                    )
                    logger.file(f"✅ Emergency checkpoint saved: {emergency_path}")
            except Exception as e2:
                logger.file(f"⚠️ Emergency save failed: {repr(e2)}")

        finally:
            logger.close()

            # กัน VRAM ค้าง/ช่วยลดโอกาส crash ข้าม trial (optional)
            try:
                del model
            except:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


        print("\n=== GRID SEARCH DONE ===")
        print("Best cfg:", best_overall["cfg"])
        print("Best val_f1:", best_overall["val_f1"])
        print("Time (min):", (time.time() - t0) / 60.0)

        BEST_PATH = "best_efficientnetv2_ab_grid.pth"
        torch.save({
            "model_name": MODEL_NAME,
            "model_state": best_overall["state"],
            "class_names": class_names,
            "img_size": IMG_SIZE,
            "best_cfg": best_overall["cfg"],
        }, BEST_PATH)
        print("Saved:", BEST_PATH)

        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES).to(DEVICE)
        ckpt = torch.load(BEST_PATH, map_location=DEVICE)
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

        print("\n=== TEST RESULTS ===")
        print("Accuracy:", accuracy_score(all_labels, all_preds))
        print("Macro F1:", f1_score(all_labels, all_preds, average="macro"))
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names))
        print("\nConfusion Matrix:")
        print(confusion_matrix(all_labels, all_preds))

        print("Success")