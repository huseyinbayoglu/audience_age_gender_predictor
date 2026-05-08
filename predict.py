"""Inference helper: load a checkpoint and predict on image paths or URLs."""
import argparse
import io
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import build_model

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_model(ckpt_path: str, device=None):
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(num_classes=len(ckpt["classes"]),
                        backbone=ckpt["backbone"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    tf = transforms.Compose([
        transforms.Resize(int(ckpt["img_size"] * 1.14)),
        transforms.CenterCrop(ckpt["img_size"]),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return model, tf, ckpt["classes"], device


def _open(src: str) -> Image.Image:
    if src.startswith(("http://", "https://")):
        import urllib.request
        with urllib.request.urlopen(src, timeout=10) as r:
            return Image.open(io.BytesIO(r.read())).convert("RGB")
    return Image.open(src).convert("RGB")


@torch.no_grad()
def predict(model, tf, classes, device, sources):
    imgs = torch.stack([tf(_open(s)) for s in sources]).to(device)
    logits = model(imgs)
    probs = F.softmax(logits, dim=1).cpu()
    out = []
    for i, s in enumerate(sources):
        idx = int(probs[i].argmax())
        out.append({
            "source": s,
            "pred": classes[idx],
            "confidence": float(probs[i, idx]),
            "probs": {c: float(probs[i, k]) for k, c in enumerate(classes)},
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("inputs", nargs="+", help="image paths or URLs")
    args = ap.parse_args()
    model, tf, classes, device = load_model(args.ckpt)
    for r in predict(model, tf, classes, device, args.inputs):
        print(f"{r['source']}\t{r['pred']}\t{r['confidence']:.3f}")


if __name__ == "__main__":
    main()
