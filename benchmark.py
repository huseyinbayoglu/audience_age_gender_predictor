"""
End-to-end speed benchmark: vLLM (Qwen2-VL-7B) vs. our small trained model.

Designed to run in a fresh Colab notebook. The user uploads two files:
  - labeled_fresh.csv   (id + avatar URL + vLLM labels; same schema as in repo)
  - gender_*.pt         (checkpoint produced by train.py)

Then:
  python benchmark.py --csv /content/labeled_fresh.csv \
                      --ckpt /content/gender_efficientnet_b0.pt \
                      --n 2000

Pipeline:
  1. Sample N rows from CSV
  2. Async-download avatars to RAM (timed)
  3. cv2-decode + resize to 224 (same as create_dataset.py) (timed)
  4. Run vLLM teacher inference using the EXACT same setup as create_dataset.py (timed)
  5. Run small-model inference (timed) — fp16, channels_last, torch.compile
  6. Save: benchmark_results.json (all metrics), benchmark_results.csv (per-row preds),
          benchmark_plot.png (throughput + latency bar charts)
"""
import argparse
import asyncio
import gc
import json
import os
import platform
import re
import ssl
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import aiohttp
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).parent

# ---- constants matching create_dataset.py ----
VLLM_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
VLLM_IMG_SIZE = 224
MAX_CONCURRENT_DOWNLOADS = 80
DOWNLOAD_TIMEOUT_SECONDS = 5

CLASSIFICATION_PROMPT = (
    "Look at this profile picture. "
    "If there is a human, estimate their gender (male or female) and age range. "
    "Age ranges: 18- (under 18), 18-24, 25-34, 35-44, 45+. "
    "Respond ONLY with a JSON object, nothing else: "
    '{"gender": "male/female/unknown", "age_range": "18-/18-24/25-34/35-44/45+/unknown"}'
    "\nIf no human is visible, respond: "
    '{"gender": "unknown", "age_range": "unknown"}'
)
VALID_GENDERS = {"male", "female", "unknown"}
VALID_AGE_RANGES = {"18-", "18-24", "25-34", "35-44", "45+", "unknown"}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_ssl_ctx = ssl.create_default_context()


# ==================== Image download (matches create_dataset.py) ====================
def _process_image_bytes(data: bytes) -> Image.Image:
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return Image.new("RGB", (VLLM_IMG_SIZE, VLLM_IMG_SIZE), (0, 0, 0))
    img = cv2.resize(img, (VLLM_IMG_SIZE, VLLM_IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


async def _dl_one(session, sem, uid, url):
    async with sem:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=DOWNLOAD_TIMEOUT_SECONDS),
                                   ssl=_ssl_ctx) as r:
                if r.status != 200:
                    return (uid, None)
                return (uid, await r.read())
        except Exception:
            return (uid, None)


async def download_async(pairs):
    sem = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
    conn = aiohttp.TCPConnector(limit=MAX_CONCURRENT_DOWNLOADS, limit_per_host=50,
                                ttl_dns_cache=300, enable_cleanup_closed=True)
    headers = {"User-Agent": "Mozilla/5.0"}
    async with aiohttp.ClientSession(connector=conn, headers=headers) as s:
        return await asyncio.gather(*[_dl_one(s, sem, u, url) for u, url in pairs])


# ==================== vLLM parser (matches create_dataset.py) ====================
def parse_vlm_response(text: str) -> dict:
    result = {"gender": "unknown", "age_range": "unknown"}
    try:
        m = re.search(r'\{[^}]+\}', text)
        if m:
            parsed = json.loads(m.group())
            g = str(parsed.get("gender", "unknown")).strip().lower()
            a = str(parsed.get("age_range", "unknown")).strip()
            result["gender"] = g if g in VALID_GENDERS else "unknown"
            result["age_range"] = a if a in VALID_AGE_RANGES else "unknown"
            return result
    except Exception:
        pass
    low = text.lower()
    if any(w in low for w in ("female", "woman", "girl")): result["gender"] = "female"
    elif any(w in low for w in ("male", "man", "boy")):    result["gender"] = "male"
    for a in ("18-24", "25-34", "35-44", "45+"):
        if a in text: result["age_range"] = a; break
    else:
        if "18-" in text or "under 18" in low: result["age_range"] = "18-"
    return result


# ==================== vLLM benchmark (mirrors create_dataset.py setup) ====================
def bench_vllm(images, model_id=VLLM_MODEL_ID):
    os.environ.setdefault("VLLM_USE_V1", "0")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    timings = {}

    print(f"  Loading {model_id}...")
    t = time.time()
    llm = LLM(
        model=model_id,
        dtype="float16",
        gpu_memory_utilization=0.85,
        max_model_len=1024,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1},
        enforce_eager=True,
    )
    timings["model_load_s"] = time.time() - t
    print(f"  Loaded in {timings['model_load_s']:.1f}s")

    processor = AutoProcessor.from_pretrained(model_id)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": "placeholder"},
        {"type": "text", "text": CLASSIFICATION_PROMPT},
    ]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    del processor; gc.collect()

    sampling_params = SamplingParams(max_tokens=20, temperature=0.0,
                                     stop=["\n\n"], min_tokens=5)

    inputs = [{"prompt": prompt, "multi_modal_data": {"image": im}} for im in images]

    print("  Warmup (8 imgs)...")
    t = time.time()
    llm.generate(inputs[:8], sampling_params=sampling_params)
    timings["warmup_s"] = time.time() - t

    print(f"  Inference on {len(inputs)} imgs...")
    t = time.time()
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    timings["inference_s"] = time.time() - t

    gender_preds, age_preds, raw_texts = [], [], []
    for o in outputs:
        text = o.outputs[0].text.strip()
        parsed = parse_vlm_response(text)
        gender_preds.append(parsed["gender"])
        age_preds.append(parsed["age_range"])
        raw_texts.append(text)

    # Tear down vLLM to free GPU before small-model bench
    del llm; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()

    return {
        "gender": gender_preds,
        "age_range": age_preds,
        "raw_text": raw_texts,
        "timings": timings,
        "n": len(images),
        "throughput_img_s": len(images) / timings["inference_s"],
        "latency_ms_img": 1000 * timings["inference_s"] / len(images),
    }


# ==================== Small-model benchmark ====================
def bench_small(ckpt_path, images, device, batch_size=512, compile_model=True,
                compile_mode="reduce-overhead", gpu_preload=False):
    from model import build_model

    timings = {}
    t = time.time()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    sz = ckpt["img_size"]
    model = build_model(num_classes=len(classes), backbone=ckpt["backbone"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval().to(memory_format=torch.channels_last)
    if compile_model:
        try:
            model = torch.compile(model, mode=compile_mode)
        except Exception as e:
            print(f"  torch.compile unavailable ({e}); continuing without.")
    timings["model_load_s"] = time.time() - t
    print(f"  Loaded {ckpt['backbone']} (img_size={sz}, classes={classes}) in "
          f"{timings['model_load_s']:.2f}s")

    tf = transforms.Compose([
        transforms.Resize(int(sz * 1.14)),
        transforms.CenterCrop(sz),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    t = time.time()
    tensors = torch.stack([tf(im) for im in images])
    timings["preprocess_s"] = time.time() - t
    print(f"  Preprocessing: {timings['preprocess_s']:.2f}s "
          f"({len(images)/timings['preprocess_s']:.0f} img/s) "
          f"shape={tuple(tensors.shape)}")

    if gpu_preload and device.type == "cuda":
        t = time.time()
        tensors = tensors.to(device, non_blocking=False
                             ).to(memory_format=torch.channels_last)
        torch.cuda.synchronize()
        timings["h2d_preload_s"] = time.time() - t
        mb = tensors.element_size() * tensors.nelement() / 1e6
        print(f"  GPU-preloaded all tensors ({mb:.0f} MB) in "
              f"{timings['h2d_preload_s']:.2f}s — no PCIe transfer during inference")

    print("  Warmup (3× batches)...")
    t = time.time()
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.float16,
                                                enabled=device.type == "cuda"):
        for _ in range(3):
            if tensors.is_cuda:
                x = tensors[:batch_size]
            else:
                x = tensors[:batch_size].to(device, non_blocking=True
                                            ).to(memory_format=torch.channels_last)
            _ = model(x)
    if device.type == "cuda": torch.cuda.synchronize()
    timings["warmup_s"] = time.time() - t

    print(f"  Inference (batch={batch_size}, compile={compile_mode}, "
          f"gpu_preload={gpu_preload})...")
    preds_idx = []
    if device.type == "cuda": torch.cuda.synchronize()
    t = time.time()
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.float16,
                                                enabled=device.type == "cuda"):
        for i in range(0, len(tensors), batch_size):
            if tensors.is_cuda:
                x = tensors[i:i+batch_size]
            else:
                x = tensors[i:i+batch_size].to(device, non_blocking=True
                                                ).to(memory_format=torch.channels_last)
            preds_idx.append(model(x).argmax(1).cpu())
    if device.type == "cuda": torch.cuda.synchronize()
    timings["inference_s"] = time.time() - t

    preds_idx = torch.cat(preds_idx).tolist()
    preds = [classes[p] for p in preds_idx]
    return {
        "task": ckpt.get("task", "gender"),
        "preds": preds,
        "classes": classes,
        "backbone": ckpt["backbone"],
        "img_size": sz,
        "batch_size": batch_size,
        "timings": timings,
        "n": len(images),
        "throughput_img_s": len(images) / timings["inference_s"],
        "latency_ms_img": 1000 * timings["inference_s"] / len(images),
    }


# ==================== Plotting ====================
def make_plot(meta, out_path: Path):
    s = meta.get("small"); v = meta.get("vllm")
    names, throughputs, latencies, colors = [], [], [], []
    if v:
        names.append(f"vLLM\n(Qwen2-VL-7B)")
        throughputs.append(v["throughput_img_s"]); latencies.append(v["latency_ms_img"])
        colors.append("#c0504d")
    if s:
        names.append(f"Small model\n({s['backbone']})")
        throughputs.append(s["throughput_img_s"]); latencies.append(s["latency_ms_img"])
        colors.append("#4f81bd")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].bar(names, throughputs, color=colors)
    axes[0].set_ylabel("Throughput (img/s)")
    axes[0].set_title("Throughput — higher is better")
    axes[0].set_yscale("log")
    axes[0].grid(axis="y", alpha=0.3, which="both")
    for i, v_ in enumerate(throughputs):
        axes[0].text(i, v_, f"{v_:.1f}", ha="center", va="bottom", fontsize=11)

    axes[1].bar(names, latencies, color=colors)
    axes[1].set_ylabel("Latency (ms / image)")
    axes[1].set_title("Per-image latency — lower is better")
    axes[1].set_yscale("log")
    axes[1].grid(axis="y", alpha=0.3, which="both")
    for i, v_ in enumerate(latencies):
        axes[1].text(i, v_, f"{v_:.2f}", ha="center", va="bottom", fontsize=11)

    if s and v:
        speedup = v["latency_ms_img"] / s["latency_ms_img"]
        fig.suptitle(f"Speedup small / vLLM = {speedup:.0f}×   (n={s['n']} images)",
                     fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ==================== Main ====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="labeled_fresh.csv",
                    help="Input CSV with at least columns: id, avatar")
    ap.add_argument("--ckpt", default="gender_efficientnet_b0.pt",
                    help="Path to a checkpoint produced by train.py")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--skip-vllm", action="store_true")
    ap.add_argument("--skip-small", action="store_true")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--compile-mode", default="reduce-overhead",
                    choices=["reduce-overhead", "max-autotune", "default"],
                    help="torch.compile mode. max-autotune is slowest to compile "
                         "but fastest at runtime (Ampere/Hopper).")
    ap.add_argument("--gpu-preload", action="store_true",
                    help="Move all preprocessed tensors to GPU once (skips per-batch "
                         "H2D transfer). Useful on A100/A40 with ample VRAM.")
    ap.add_argument("--vllm-model", default=VLLM_MODEL_ID)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(ROOT))
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "benchmark_results.json"
    csv_path  = out_dir / "benchmark_results.csv"
    plot_path = out_dir / "benchmark_plot.png"

    # ---- Load CSV ----
    csv_p = Path(args.csv)
    if not csv_p.exists():
        raise FileNotFoundError(f"CSV not found: {csv_p}")
    df = pd.read_csv(csv_p)
    if not {"id", "avatar"}.issubset(df.columns):
        raise ValueError(f"CSV must contain 'id' and 'avatar' columns; got {list(df.columns)}")
    df = df.dropna(subset=["avatar"]).drop_duplicates(subset=["id"])
    df = df.sample(n=min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)
    print(f"Sampled {len(df)} rows from {csv_p.name}")

    # ---- Download ----
    print(f"\nDownloading {len(df)} images (concurrency={MAX_CONCURRENT_DOWNLOADS})...")
    t0 = time.time()
    raw = asyncio.run(download_async(list(zip(df["id"].astype(str), df["avatar"]))))
    download_s = time.time() - t0
    n_ok = sum(1 for _, b in raw if b is not None)
    print(f"  Downloaded: {n_ok}/{len(raw)} in {download_s:.1f}s "
          f"({n_ok/download_s:.0f} img/s)")

    # ---- Decode + resize (cv2, matches create_dataset.py) ----
    print("Decoding + resizing (cv2 → 224×224 PIL)...")
    t0 = time.time()
    images, kept_ids = [], []
    for uid, data in raw:
        if data is None:
            continue
        images.append(_process_image_bytes(data))
        kept_ids.append(uid)
    decode_s = time.time() - t0
    print(f"  Decoded: {len(images)} in {decode_s:.1f}s")

    df = df[df["id"].astype(str).isin(set(kept_ids))].copy()
    df["id"] = df["id"].astype(str)
    df = df.set_index("id").loc[kept_ids].reset_index()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}  ({gpu_name})")

    meta = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "hardware": {
            "device": str(device),
            "gpu": gpu_name,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "input": {
            "csv": str(csv_p),
            "n_requested": args.n,
            "n_sampled": len(raw),
            "n_downloaded": n_ok,
            "n_decoded": len(images),
        },
        "data_pipeline": {
            "download_s": download_s,
            "download_throughput_img_s": n_ok / download_s if download_s > 0 else 0,
            "decode_s": decode_s,
            "decode_throughput_img_s": len(images) / decode_s if decode_s > 0 else 0,
        },
    }

    # ---- Small model first (low GPU usage; leaves room before vLLM grabs 85%) ----
    if not args.skip_small:
        print("\n=== SMALL MODEL ===")
        if not Path(args.ckpt).exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
        small = bench_small(args.ckpt, images, device,
                            batch_size=args.batch_size,
                            compile_model=not args.no_compile,
                            compile_mode=args.compile_mode,
                            gpu_preload=args.gpu_preload)
        print(f"  inference:  {small['timings']['inference_s']:.3f}s")
        print(f"  throughput: {small['throughput_img_s']:.1f} img/s")
        print(f"  latency:    {small['latency_ms_img']:.3f} ms/img")
        meta["small"] = {k: v for k, v in small.items() if k not in ("preds",)}
        gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        small = None

    if not args.skip_vllm:
        print("\n=== VLLM TEACHER ===")
        vllm_res = bench_vllm(images, args.vllm_model)
        print(f"  inference:  {vllm_res['timings']['inference_s']:.1f}s")
        print(f"  throughput: {vllm_res['throughput_img_s']:.2f} img/s")
        print(f"  latency:    {vllm_res['latency_ms_img']:.1f} ms/img")
        meta["vllm"] = {k: v for k, v in vllm_res.items()
                        if k not in ("gender", "age_range", "raw_text")}
        meta["vllm"]["model_id"] = args.vllm_model
    else:
        vllm_res = None

    # ---- Summary + agreement ----
    print("\n=== SUMMARY ===")
    if small:
        print(f"  Small  : {small['timings']['inference_s']:9.3f}s   "
              f"{small['throughput_img_s']:9.1f} img/s   "
              f"{small['latency_ms_img']:7.3f} ms/img")
    if vllm_res:
        print(f"  vLLM   : {vllm_res['timings']['inference_s']:9.1f}s   "
              f"{vllm_res['throughput_img_s']:9.2f} img/s   "
              f"{vllm_res['latency_ms_img']:7.1f} ms/img")
    if small and vllm_res:
        sp = vllm_res["latency_ms_img"] / small["latency_ms_img"]
        print(f"  Speedup (small vs vLLM): {sp:.0f}×")
        # Agreement (gender only — small model is gender classifier)
        task = small["task"]
        v_preds = vllm_res["gender"] if task == "gender" else vllm_res["age_range"]
        agree = sum(1 for a, b in zip(small["preds"], v_preds) if a == b)
        meta["agreement"] = {
            "task": task,
            "matches": agree,
            "total": len(v_preds),
            "rate": agree / len(v_preds),
            "speedup_x": sp,
        }
        print(f"  Agreement ({task}): {agree}/{len(v_preds)} = {100*agree/len(v_preds):.1f}%")
        # Per-class breakdown
        cm = Counter(zip(v_preds, small["preds"]))
        meta["agreement"]["confusion_vllm_vs_small"] = {
            f"{t}->{p}": c for (t, p), c in cm.items()
        }

    # ---- Save CSV ----
    out = df.copy()
    if small:
        out["small_gender_pred"] = small["preds"]
    if vllm_res:
        out["vllm_gender_pred"] = vllm_res["gender"]
        out["vllm_age_pred"]    = vllm_res["age_range"]
    out.to_csv(csv_path, index=False)
    print(f"\nPer-row predictions  -> {csv_path}")

    # ---- Save JSON ----
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"All metrics          -> {json_path}")

    # ---- Plot ----
    if small or vllm_res:
        make_plot({"small": meta.get("small"), "vllm": meta.get("vllm")}, plot_path)
        print(f"Plot                 -> {plot_path}")


if __name__ == "__main__":
    main()
