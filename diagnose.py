"""Pin down whether the bottleneck is data loading or GPU compute.

Runs three independent benchmarks:
  A) Pure dataloader iteration (no model)
  B) Model forward+backward on synthetic GPU tensors (no dataloader)
  C) End-to-end (dataloader + model)

If A throughput is much lower than B → CPU/data pipeline is the bottleneck.
If A and B are similar but C is slow → overlap is poor (workers too few).
If A and B are both high but C is low → unexpected; report numbers.
"""
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import v2 as transforms_v2

from dataset import AvatarDataset, CLASSES
from model import build_model

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_loader(task, img_size, bs, workers, preload, augment, cache_decoded=False):
    if cache_decoded:
        ops = [transforms_v2.RandomHorizontalFlip()] if augment else []
        tf = transforms_v2.Compose(ops + [
            transforms_v2.ToDtype(torch.float32, scale=True),
            transforms_v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        ds = AvatarDataset("train", task, transform=tf,
                           decoded_cache=True, decoded_size=img_size)
    else:
        if augment:
            tf = transforms.Compose([
                transforms.Resize(int(img_size * 1.14)),
                transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                transforms.RandomErasing(p=0.15),
            ])
        else:
            tf = transforms.Compose([
                transforms.Resize(int(img_size * 1.14)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        ds = AvatarDataset("train", task, transform=tf, preload=preload)
    print(f"  dataset size: {len(ds)}, preload={preload}, "
          f"cache_decoded={cache_decoded}, augment={augment}")
    return DataLoader(ds, batch_size=bs, shuffle=True, num_workers=workers,
                      pin_memory=True, persistent_workers=workers > 0,
                      drop_last=True)


def bench_dataloader(loader, max_batches=30):
    t0 = time.time()
    n_imgs = 0
    for i, (x, y) in enumerate(loader):
        n_imgs += x.size(0)
        if i + 1 >= max_batches:
            break
    dt = time.time() - t0
    return n_imgs, dt


def bench_model(device, img_size, bs, n_classes, n_iters=30):
    model = build_model(n_classes).to(device)
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    x = torch.randn(bs, 3, img_size, img_size, device=device)
    y = torch.randint(0, n_classes, (bs,), device=device)
    # warmup
    for _ in range(3):
        optim.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            loss = crit(model(x), y)
        scaler.scale(loss).backward(); scaler.step(optim); scaler.update()
    if device.type == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iters):
        optim.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            loss = crit(model(x), y)
        scaler.scale(loss).backward(); scaler.step(optim); scaler.update()
    if device.type == "cuda": torch.cuda.synchronize()
    return n_iters * bs, time.time() - t0


def bench_e2e(loader, device, n_classes, max_batches=30):
    model = build_model(n_classes).to(device)
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    n_imgs, t0 = 0, time.time()
    for i, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        optim.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            loss = crit(model(x), y)
        scaler.scale(loss).backward(); scaler.step(optim); scaler.update()
        n_imgs += x.size(0)
        if i + 1 >= max_batches: break
    if device.type == "cuda": torch.cuda.synchronize()
    return n_imgs, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="gender")
    ap.add_argument("--img-size", type=int, default=160)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--preload", action="store_true")
    ap.add_argument("--cache-decoded", action="store_true")
    ap.add_argument("--batches", type=int, default=20)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = len(CLASSES[args.task])
    print(f"device={device}  bs={args.batch_size}  img={args.img_size}  workers={args.workers}")

    print("\n[B] Model-only (synthetic GPU tensors, no dataloader)")
    n, dt = bench_model(device, args.img_size, args.batch_size, n_classes, args.batches)
    print(f"   {n} imgs in {dt:.2f}s → {n/dt:.0f} img/s   ({dt/args.batches*1000:.0f} ms/iter)")

    # Build dataset ONCE; swap transforms in-place to avoid duplicating the cache.
    if args.cache_decoded:
        no_aug_tf = transforms_v2.Compose([
            transforms_v2.ToDtype(torch.float32, scale=True),
            transforms_v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        aug_tf = transforms_v2.Compose([
            transforms_v2.RandomHorizontalFlip(),
            transforms_v2.ToDtype(torch.float32, scale=True),
            transforms_v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        ds = AvatarDataset("train", args.task, transform=no_aug_tf,
                           decoded_cache=True, decoded_size=args.img_size)

        def mk(transform):
            ds.transform = transform
            return DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              persistent_workers=args.workers > 0, drop_last=True)

        print("\n[A1] Dataloader-only WITHOUT augmentation")
        n, dt = bench_dataloader(mk(no_aug_tf), args.batches)
        print(f"   {n} imgs in {dt:.2f}s → {n/dt:.0f} img/s")
        print("\n[A2] Dataloader-only WITH augmentation")
        n, dt = bench_dataloader(mk(aug_tf), args.batches)
        print(f"   {n} imgs in {dt:.2f}s → {n/dt:.0f} img/s")
        print("\n[C] End-to-end (dataloader + model)")
        n, dt = bench_e2e(mk(aug_tf), device, n_classes, args.batches)
        print(f"   {n} imgs in {dt:.2f}s → {n/dt:.0f} img/s   ({dt/args.batches*1000:.0f} ms/iter)")
    else:
        print("\n[A1] Dataloader-only WITHOUT augmentation")
        loader = make_loader(args.task, args.img_size, args.batch_size, args.workers,
                             args.preload, augment=False, cache_decoded=False)
        n, dt = bench_dataloader(loader, args.batches)
        print(f"   {n} imgs in {dt:.2f}s → {n/dt:.0f} img/s")
        print("\n[A2] Dataloader-only WITH augmentation")
        loader = make_loader(args.task, args.img_size, args.batch_size, args.workers,
                             args.preload, augment=True, cache_decoded=False)
        n, dt = bench_dataloader(loader, args.batches)
        print(f"   {n} imgs in {dt:.2f}s → {n/dt:.0f} img/s")
        print("\n[C] End-to-end")
        n, dt = bench_e2e(loader, device, n_classes, args.batches)
        print(f"   {n} imgs in {dt:.2f}s → {n/dt:.0f} img/s   ({dt/args.batches*1000:.0f} ms/iter)")

    print("\n--- Verdict ---")
    print("If A2 << B → CPU augmentation is the bottleneck (try lighter aug or GPU aug).")
    print("If A1 << B → JPEG decode/disk is the bottleneck (preload + smaller images).")
    print("If A1 ≈ A2 ≈ B but C is low → improve overlap (more workers, prefetch).")


if __name__ == "__main__":
    main()
