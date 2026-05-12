"""
Apples-to-apples speed benchmark: vLLM teacher vs. our small model.

  python benchmark.py --n 2000 --ckpt checkpoints/gender_efficientnet_b0.pt
  python benchmark.py --skip-vllm    # just measure the small model
  python benchmark.py --skip-small   # just measure vLLM

Reports throughput (img/s) for each, plus their agreement rate.
"""
import argparse
import asyncio
import re
import ssl
import time
from pathlib import Path

import aiohttp
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).parent
DATASET_DIR = ROOT / "dataset"

CLASSIFICATION_PROMPT = (
    "Look at this profile picture. "
    "If there is a human, estimate their gender (male or female) and age range. "
    "Age ranges: 18- (under 18), 18-24, 25-34, 35-44, 45+. "
    "Respond ONLY with a JSON object, nothing else: "
    '{"gender": "male/female/unknown", "age_range": "18-/18-24/25-34/35-44/45+/unknown"}'
    "\nIf no human is visible, respond: "
    '{"gender": "unknown", "age_range": "unknown"}'
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# -------------------- Data --------------------
def sample_rows(n: int, seed: int = 0) -> pd.DataFrame:
    for name in ["labeled_fresh.csv", "followers_dataset.csv",
                 "followers_dataset2.csv", "followers_dataset3.csv"]:
        p = DATASET_DIR / name
        if p.exists():
            df = pd.read_csv(p); break
    else:
        raise FileNotFoundError(f"No source CSV in {DATASET_DIR}")
    df = df.dropna(subset=["avatar"]).drop_duplicates(subset=["id"])
    return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)


_ssl_ctx = ssl.create_default_context()


async def _dl_one(session, sem, url):
    async with sem:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10),
                                   ssl=_ssl_ctx) as r:
                if r.status != 200: return None
                return await r.read()
        except Exception:
            return None


async def download_all(urls):
    sem = asyncio.Semaphore(80)
    conn = aiohttp.TCPConnector(limit=80, limit_per_host=50, ttl_dns_cache=300)
    headers = {"User-Agent": "Mozilla/5.0"}
    async with aiohttp.ClientSession(connector=conn, headers=headers) as s:
        return await asyncio.gather(*[_dl_one(s, sem, u) for u in urls])


def to_pil(data):
    if data is None: return None
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None: return None
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# -------------------- Small model --------------------
def bench_small(ckpt_path: str, images, device, batch_size: int = 512,
                compile_model: bool = True):
    from model import build_model

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    sz = ckpt["img_size"]

    model = build_model(num_classes=len(classes), backbone=ckpt["backbone"],
                        pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval().to(memory_format=torch.channels_last)
    if compile_model:
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"  torch.compile unavailable ({e}); continuing without.")

    tf = transforms.Compose([
        transforms.Resize(int(sz * 1.14)),
        transforms.CenterCrop(sz),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    print("  Preprocessing on CPU...")
    tensors = torch.stack([tf(im) for im in tqdm(images, leave=False)])
    print(f"  Input tensor: {tuple(tensors.shape)} ({tensors.element_size()*tensors.nelement()/1e6:.0f} MB)")

    # Warmup (torch.compile traces on first calls)
    print("  Warmup...")
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.float16,
                                                enabled=device.type == "cuda"):
        for _ in range(3):
            x = tensors[:batch_size].to(device, non_blocking=True
                                        ).to(memory_format=torch.channels_last)
            _ = model(x)
    if device.type == "cuda": torch.cuda.synchronize()

    print(f"  Inference (batch={batch_size})...")
    preds = []
    t0 = time.time()
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.float16,
                                                enabled=device.type == "cuda"):
        for i in range(0, len(tensors), batch_size):
            x = tensors[i:i+batch_size].to(device, non_blocking=True
                                            ).to(memory_format=torch.channels_last)
            preds.append(model(x).argmax(1).cpu())
    if device.type == "cuda": torch.cuda.synchronize()
    dt = time.time() - t0
    preds = torch.cat(preds).tolist()
    return [classes[p] for p in preds], dt


# -------------------- vLLM teacher --------------------
def bench_vllm(images, model_id: str = "Qwen/Qwen2-VL-7B-Instruct"):
    import os
    os.environ.setdefault("VLLM_USE_V1", "0")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    print(f"  Loading {model_id} ...")
    llm = LLM(model=model_id, dtype="float16", gpu_memory_utilization=0.85,
              max_model_len=1024, trust_remote_code=True,
              limit_mm_per_prompt={"image": 1}, enforce_eager=True)

    proc = AutoProcessor.from_pretrained(model_id)
    msg = [{"role": "user", "content": [
        {"type": "image", "image": "placeholder"},
        {"type": "text", "text": CLASSIFICATION_PROMPT}]}]
    prompt = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

    inputs = [{"prompt": prompt, "multi_modal_data": {"image": im}} for im in images]
    sp = SamplingParams(max_tokens=20, temperature=0.0, stop=["\n\n"], min_tokens=5)

    print("  Warmup (8 imgs)...")
    llm.generate(inputs[:8], sampling_params=sp)

    print(f"  Inference ({len(inputs)} imgs)...")
    t0 = time.time()
    outputs = llm.generate(inputs, sampling_params=sp)
    dt = time.time() - t0

    preds = []
    for o in outputs:
        text = o.outputs[0].text.strip().lower()
        m = re.search(r'"gender"\s*:\s*"([a-z]+)"', text)
        g = m.group(1) if m else "unknown"
        preds.append(g if g in {"male", "female", "unknown"} else "unknown")
    return preds, dt


# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--ckpt", default="checkpoints/gender_efficientnet_b0.pt")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--skip-vllm", action="store_true")
    ap.add_argument("--skip-small", action="store_true")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--vllm-model", default="Qwen/Qwen2-VL-7B-Instruct")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    DATASET_DIR.mkdir(exist_ok=True, parents=True)
    print(f"Sampling {args.n} rows...")
    df = sample_rows(args.n, args.seed)
    print(f"Got {len(df)} rows.")

    print(f"\nDownloading {len(df)} images (network — not counted in inference time)...")
    t0 = time.time()
    raw = asyncio.run(download_all(df["avatar"].tolist()))
    print(f"  Download: {time.time()-t0:.1f}s")

    print("Decoding...")
    images = [to_pil(r) for r in raw]
    keep = [i for i, im in enumerate(images) if im is not None]
    images = [images[i] for i in keep]
    df = df.iloc[keep].reset_index(drop=True)
    print(f"  Valid images: {len(images)} / {len(raw)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results = {}
    if not args.skip_small:
        print("\n=== SMALL MODEL ===")
        preds, dt = bench_small(args.ckpt, images, device,
                                batch_size=args.batch_size,
                                compile_model=not args.no_compile)
        thr = len(images) / dt
        ms = 1000 * dt / len(images)
        print(f"  {len(images)} imgs in {dt:.2f}s → {thr:.0f} img/s ({ms:.2f} ms/img)")
        results["small"] = (preds, dt, thr)

    if not args.skip_vllm:
        print("\n=== VLLM TEACHER ===")
        preds, dt = bench_vllm(images, args.vllm_model)
        thr = len(images) / dt
        ms = 1000 * dt / len(images)
        print(f"  {len(images)} imgs in {dt:.2f}s → {thr:.1f} img/s ({ms:.0f} ms/img)")
        results["vllm"] = (preds, dt, thr)

    print("\n=== SUMMARY ===")
    for k, (_, dt, thr) in results.items():
        print(f"  {k:6s}: {dt:7.2f}s  {thr:8.1f} img/s")
    if "small" in results and "vllm" in results:
        ps, _, _ = results["small"]; pv, _, _ = results["vllm"]
        agree = sum(1 for a, b in zip(ps, pv) if a == b)
        print(f"\n  Speedup (vLLM / small): {results['vllm'][1] / results['small'][1]:.1f}×")
        print(f"  Throughput ratio:       {results['small'][2] / results['vllm'][2]:.1f}×")
        print(f"  Agreement:              {agree}/{len(ps)} = {100*agree/len(ps):.1f}%")

    out = df.copy()
    if "small" in results: out["small_pred"] = results["small"][0]
    if "vllm" in results:  out["vllm_pred"]  = results["vllm"][0]
    out_path = DATASET_DIR / "benchmark_results.csv"
    out.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
