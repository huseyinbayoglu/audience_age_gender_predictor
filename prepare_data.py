"""
Merge raw CSVs, deduplicate, stratified 85/10/5 split, async-download avatars
to dataset/images/{id}.jpg. Re-running is safe: existing images are skipped.
"""
import argparse
import asyncio
import ssl
from pathlib import Path

import aiohttp
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.asyncio import tqdm_asyncio

ROOT = Path(__file__).parent
DATASET_DIR = ROOT / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
SPLIT_DIR = DATASET_DIR / "splits"

GENDER_CLASSES = ["male", "female", "unknown"]
AGE_CLASSES = ["18-", "18-24", "25-34", "35-44", "45+", "unknown"]

MAX_CONCURRENT = 80
TIMEOUT_S = 10
TARGET_SIZE = 256  # store at 256, train transforms will crop to 224


def load_and_clean() -> pd.DataFrame:
    fresh = DATASET_DIR / "labeled_fresh.csv"
    if fresh.exists():
        print(f"Using refreshed dataset: {fresh.name}")
        df = pd.read_csv(fresh)
    else:
        csvs = sorted(DATASET_DIR.glob("followers_dataset*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No followers_dataset*.csv in {DATASET_DIR}")
        df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
    df = df[["id", "avatar", "vllm_gender_prediction", "vllm_age_range_prediction"]]
    df = df.dropna()
    df = df.drop_duplicates(subset=["id"])
    df = df[df["vllm_gender_prediction"].isin(GENDER_CLASSES)]
    df = df[df["vllm_age_range_prediction"].isin(AGE_CLASSES)]
    df["id"] = df["id"].astype(str)
    return df.reset_index(drop=True)


def stratified_split(df: pd.DataFrame, seed: int = 42):
    # Stratify on combined label so both gender & age stay balanced across splits.
    strat = df["vllm_gender_prediction"] + "|" + df["vllm_age_range_prediction"]
    # Some combos are tiny; collapse rare strata (<3 rows) to a single bucket.
    counts = strat.value_counts()
    rare = counts[counts < 3].index
    strat = strat.where(~strat.isin(rare), other="__rare__")

    train_df, temp_df = train_test_split(
        df, test_size=0.15, stratify=strat, random_state=seed
    )
    strat_temp = (
        temp_df["vllm_gender_prediction"] + "|" + temp_df["vllm_age_range_prediction"]
    )
    counts2 = strat_temp.value_counts()
    rare2 = counts2[counts2 < 2].index
    strat_temp = strat_temp.where(~strat_temp.isin(rare2), other="__rare__")
    val_df, test_df = train_test_split(
        temp_df, test_size=1 / 3, stratify=strat_temp, random_state=seed
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


_ssl_ctx = ssl.create_default_context()


async def _download_one(session, sem, uid: str, url: str) -> bool:
    out = IMAGES_DIR / f"{uid}.jpg"
    if out.exists() and out.stat().st_size > 0:
        return True
    async with sem:
        try:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=TIMEOUT_S), ssl=_ssl_ctx
            ) as resp:
                if resp.status != 200:
                    return False
                data = await resp.read()
        except Exception:
            return False
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return False
    h, w = img.shape[:2]
    scale = TARGET_SIZE / min(h, w)
    if scale != 1.0:
        img = cv2.resize(
            img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA
        )
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return True


async def download_all(df: pd.DataFrame):
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    connector = aiohttp.TCPConnector(
        limit=MAX_CONCURRENT, limit_per_host=50, ttl_dns_cache=300, enable_cleanup_closed=True
    )
    headers = {"User-Agent": "Mozilla/5.0"}
    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        tasks = [
            _download_one(session, sem, str(r.id), r.avatar)
            for r in df.itertuples(index=False)
        ]
        results = await tqdm_asyncio.gather(*tasks, desc="Downloading")
    ok = sum(1 for r in results if r)
    print(f"Downloaded/cached: {ok}/{len(results)}")


def filter_to_existing(df: pd.DataFrame) -> pd.DataFrame:
    have = {p.stem for p in IMAGES_DIR.glob("*.jpg")}
    return df[df["id"].isin(have)].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-download", action="store_true", help="skip image download step")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_clean()
    print(f"Cleaned rows: {len(df)}")

    if not args.no_download:
        asyncio.run(download_all(df))

    df = filter_to_existing(df)
    print(f"Rows with images on disk: {len(df)}")

    train_df, val_df, test_df = stratified_split(df, seed=args.seed)
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    train_df.to_csv(SPLIT_DIR / "train.csv", index=False)
    val_df.to_csv(SPLIT_DIR / "val.csv", index=False)
    test_df.to_csv(SPLIT_DIR / "test.csv", index=False)
    print(f"Splits written to {SPLIT_DIR}")

    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"\n[{name}] gender:")
        print(d["vllm_gender_prediction"].value_counts().to_dict())
        print(f"[{name}] age:")
        print(d["vllm_age_range_prediction"].value_counts().to_dict())


if __name__ == "__main__":
    main()
