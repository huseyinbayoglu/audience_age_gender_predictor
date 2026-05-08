"""Train a single-task classifier (gender OR age) on the avatar dataset.

Usage:
    python train.py --task gender --backbone efficientnet_b0 --epochs 20
    python train.py --task age    --backbone efficientnet_b0 --epochs 25
"""
import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from dataset import AvatarDataset, CLASSES
from model import build_model

ROOT = Path(__file__).parent
CKPT_DIR = ROOT / "checkpoints"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.1)),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def class_weights_inverse_freq(labels, n_classes: int) -> torch.Tensor:
    counts = Counter(labels)
    freqs = np.array([counts.get(i, 1) for i in range(n_classes)], dtype=np.float64)
    w = freqs.sum() / (n_classes * freqs)
    return torch.tensor(w, dtype=torch.float32)


def make_sampler(labels, n_classes: int) -> WeightedRandomSampler:
    counts = Counter(labels)
    cls_w = {c: 1.0 / counts[c] for c in counts}
    sample_w = np.array([cls_w[y] for y in labels], dtype=np.float64)
    return WeightedRandomSampler(sample_w, num_samples=len(labels), replacement=True)


@torch.no_grad()
def evaluate(model, loader, device, classes):
    model.eval()
    all_y, all_p = [], []
    total_loss, n = 0.0, 0
    crit = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x)
            loss = crit(logits, y)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        all_y.extend(y.cpu().tolist())
        all_p.extend(logits.argmax(1).cpu().tolist())
    acc = float(np.mean(np.array(all_y) == np.array(all_p)))
    macro_f1 = _macro_f1(all_y, all_p, len(classes))
    return total_loss / max(n, 1), acc, macro_f1, all_y, all_p


def _macro_f1(y_true, y_pred, n_classes):
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    f1s = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0); continue
        p = tp / (tp + fp); r = tp / (tp + fn)
        f1s.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
    return float(np.mean(f1s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["gender", "age"], required=True)
    ap.add_argument("--backbone", default="efficientnet_b0")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--sampler", choices=["weighted", "none"], default="weighted",
                    help="WeightedRandomSampler balances classes during training.")
    ap.add_argument("--use-class-weights", action="store_true",
                    help="Apply inverse-freq class weights in CE loss (in addition to/instead of sampler).")
    ap.add_argument("--freeze-epochs", type=int, default=1,
                    help="Freeze backbone for N warmup epochs (head-only).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="Checkpoint output path (.pt)")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    classes = CLASSES[args.task]
    n_classes = len(classes)

    train_tf, eval_tf = build_transforms(args.img_size)
    train_ds = AvatarDataset("train", args.task, transform=train_tf)
    val_ds   = AvatarDataset("val",   args.task, transform=eval_tf)
    test_ds  = AvatarDataset("test",  args.task, transform=eval_tf)
    print(f"Sizes: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    if args.sampler == "weighted":
        sampler = make_sampler(train_ds.labels, n_classes)
        shuffle = False
    else:
        sampler, shuffle = None, True

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              shuffle=shuffle, num_workers=args.workers,
                              pin_memory=pin, persistent_workers=args.workers > 0, drop_last=True)
    val_loader   = DataLoader(val_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=pin,
                              persistent_workers=args.workers > 0)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=pin,
                              persistent_workers=args.workers > 0)

    model = build_model(num_classes=n_classes, backbone=args.backbone, pretrained=True).to(device)

    cls_w = class_weights_inverse_freq(train_ds.labels, n_classes).to(device) \
        if args.use_class_weights else None
    criterion = nn.CrossEntropyLoss(weight=cls_w, label_smoothing=args.label_smoothing)

    # Param groups: separate head from backbone for warmup (head-first).
    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        (head_params if "classifier" in name or name.startswith("fc") or "head" in name
         else backbone_params).append(p)
    optim = torch.optim.AdamW(
        [{"params": backbone_params, "lr": args.lr},
         {"params": head_params,     "lr": args.lr * 3}],
        weight_decay=args.weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    CKPT_DIR.mkdir(exist_ok=True, parents=True)
    out_path = Path(args.out) if args.out else CKPT_DIR / f"{args.task}_{args.backbone}.pt"

    best_metric = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        # Backbone freeze warmup
        freeze = epoch <= args.freeze_epochs
        for p in backbone_params:
            p.requires_grad = not freeze

        model.train()
        running_loss, running_n, running_correct = 0.0, 0, 0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}{' [frozen]' if freeze else ''}")
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
            with torch.no_grad():
                running_loss += loss.item() * x.size(0); running_n += x.size(0)
                running_correct += (logits.argmax(1) == y).sum().item()
            pbar.set_postfix(loss=f"{running_loss/running_n:.3f}",
                             acc=f"{running_correct/running_n:.3f}")

        sched.step()
        train_loss = running_loss / running_n
        train_acc  = running_correct / running_n

        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, device, classes)
        elapsed = time.time() - t0
        print(f"epoch {epoch}: train loss {train_loss:.3f} acc {train_acc:.3f} | "
              f"val loss {val_loss:.3f} acc {val_acc:.3f} macroF1 {val_f1:.3f} | {elapsed:.1f}s")
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc, "val_macro_f1": val_f1})

        # Track best by val macro_f1 (robust to imbalance).
        if val_f1 > best_metric:
            best_metric = val_f1
            torch.save({
                "state_dict": model.state_dict(),
                "task": args.task,
                "backbone": args.backbone,
                "img_size": args.img_size,
                "classes": classes,
                "val_macro_f1": val_f1,
                "val_acc": val_acc,
            }, out_path)
            print(f"  ✔ saved best to {out_path} (val_macro_f1={val_f1:.3f})")

    # Final test eval with best checkpoint
    print("\nLoading best checkpoint for test eval...")
    ckpt = torch.load(out_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test_loss, test_acc, test_f1, y, p = evaluate(model, test_loader, device, classes)
    print(f"TEST  loss {test_loss:.3f} acc {test_acc:.3f} macroF1 {test_f1:.3f}")
    print("\nClassification report (test):")
    print(classification_report(y, p, target_names=classes, digits=3, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y, p, labels=list(range(n_classes))))

    log_path = out_path.with_suffix(".log.json")
    log_path.write_text(json.dumps({
        "args": vars(args), "history": history,
        "test": {"loss": test_loss, "acc": test_acc, "macro_f1": test_f1},
    }, indent=2))
    print(f"Training log -> {log_path}")


if __name__ == "__main__":
    main()
