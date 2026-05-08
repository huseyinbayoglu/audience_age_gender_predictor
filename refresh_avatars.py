"""
Re-fetch fresh avatar URLs for already-labeled IDs.

Strategy:
  1. Merge followers*.csv (has source_username + id) with
     followers_dataset*.csv (has id + vLLM labels) on `id`.
  2. For each unique `source_username`, re-call the TikTok API to fetch
     followers (with fresh, non-expired avatar URLs).
  3. Match new rows to old labeled IDs and write a single
     `dataset/labeled_fresh.csv` with: id, avatar, gender, age, source_username.
  4. prepare_data.py picks this up automatically (LABELED_FRESH overrides the
     old followers_dataset*.csv glob if present).

Reads RAPIDAPI_KEY (and optional RAPIDAPI_HOST) from .env.
"""
import http.client
import json
import os
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).parent
DATASET_DIR = ROOT / "dataset"
OUT_PATH = DATASET_DIR / "labeled_fresh.csv"

API_HOST = "tiktok-scraper7.p.rapidapi.com"


def _load_api_key() -> str:
    """Try Colab userdata first, then .env, then env var."""
    try:
        from google.colab import userdata  # type: ignore
        key = userdata.get("rapid_tiktok")
        if key:
            return key
    except Exception:
        pass
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except Exception:
        pass
    key = os.environ.get("RAPIDAPI_KEY") or os.environ.get("rapid_tiktok")
    if not key:
        raise RuntimeError(
            "API key not found. In Colab: Secrets → add 'rapid_tiktok'. "
            "Locally: set RAPIDAPI_KEY in .env."
        )
    return key


API_KEY = _load_api_key()

GENDER_CLASSES = {"male", "female", "unknown"}
AGE_CLASSES = {"18-", "18-24", "25-34", "35-44", "45+", "unknown"}


def _request(path: str) -> dict:
    conn = http.client.HTTPSConnection(API_HOST, timeout=15)
    conn.request("GET", path, headers={"x-rapidapi-key": API_KEY, "x-rapidapi-host": API_HOST})
    raw = conn.getresponse().read().decode("utf-8")
    conn.close()
    return json.loads(raw)


def get_user_id(username: str) -> str:
    return _request(f"/user/info?unique_id={username}")["data"]["user"]["id"]


def fetch_followers(username: str, target: int = 1500) -> pd.DataFrame:
    uid = get_user_id(username)
    rows, t_cursor, has_more = [], 0, True
    while len(rows) < target and has_more:
        d = _request(f"/user/followers?user_id={uid}&count=200&time={t_cursor}")["data"]
        rows.extend(d["followers"])
        t_cursor = d["time"]
        has_more = d.get("hasMore", False)
        if not d["followers"]:
            break
        time.sleep(0.15)  # gentle pacing
    if not rows:
        return pd.DataFrame(columns=["id", "avatar"])
    df = pd.json_normalize(rows)
    if "id" not in df or "avatar" not in df:
        return pd.DataFrame(columns=["id", "avatar"])
    return df[["id", "avatar"]].dropna().drop_duplicates(subset=["id"])


def load_labeled_with_source() -> pd.DataFrame:
    """Merge raw followers*.csv (id + source_username) with
    followers_dataset*.csv (id + labels) → one frame per id."""
    raw_paths = sorted(DATASET_DIR.glob("followers[0-9]*.csv")) + \
                sorted(DATASET_DIR.glob("followers.csv"))
    raw = pd.concat([pd.read_csv(p) for p in raw_paths], ignore_index=True)
    raw = raw[["source_username", "id"]].dropna().drop_duplicates(subset=["id"])
    raw["id"] = raw["id"].astype(str)

    lab_paths = sorted(DATASET_DIR.glob("followers_dataset*.csv"))
    lab = pd.concat([pd.read_csv(p) for p in lab_paths], ignore_index=True)
    lab = lab[["id", "vllm_gender_prediction", "vllm_age_range_prediction"]].dropna()
    lab = lab[lab["vllm_gender_prediction"].isin(GENDER_CLASSES)]
    lab = lab[lab["vllm_age_range_prediction"].isin(AGE_CLASSES)]
    lab = lab.drop_duplicates(subset=["id"])
    lab["id"] = lab["id"].astype(str)

    merged = lab.merge(raw, on="id", how="inner")
    return merged


def main():
    DATASET_DIR.mkdir(exist_ok=True, parents=True)
    labeled = load_labeled_with_source()
    print(f"Labeled rows with known source_username: {len(labeled)}")
    print("Per source_username counts:")
    print(labeled["source_username"].value_counts().head(20))

    sources = labeled["source_username"].dropna().unique().tolist()
    print(f"\nUnique source_usernames to re-query: {len(sources)}")

    fresh_frames = []
    failed = []
    for name in tqdm(sources, desc="Re-fetching followers"):
        try:
            df = fetch_followers(name, target=1500)
            if not df.empty:
                df["source_username"] = name
                fresh_frames.append(df)
        except Exception as e:
            failed.append((name, str(e)))
            print(f"  ! {name}: {e}")

    if not fresh_frames:
        raise RuntimeError("No fresh data fetched")

    fresh = pd.concat(fresh_frames, ignore_index=True)
    fresh["id"] = fresh["id"].astype(str)
    fresh = fresh.drop_duplicates(subset=["id"])
    print(f"\nFresh rows fetched: {len(fresh)}")

    out = labeled.drop(columns=["source_username"]).merge(
        fresh, on="id", how="inner"
    )
    print(f"Re-matched (labeled ∩ fresh): {len(out)}/{len(labeled)} "
          f"({100*len(out)/max(len(labeled),1):.1f}% recovery)")

    out = out[["id", "avatar", "vllm_gender_prediction",
               "vllm_age_range_prediction", "source_username"]]
    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")
    if failed:
        print(f"\n{len(failed)} usernames failed:")
        for n, e in failed[:10]:
            print(f"  - {n}: {e}")


if __name__ == "__main__":
    main()
