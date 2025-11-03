import pandas as pd
from pathlib import Path

REQUIRED = {
    "provider","instance_type","vcpus","mem_gb",
    "price_per_hour","storage_type","storage_price_per_gb_month"
}
OPTIONAL = {"iops_included","iops_price_per_iops_month"}

def load_pricing(csv_paths):
    frames = []
    for p in csv_paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"Pricing file not found: {p}")
        df = pd.read_csv(p)
        missing = REQUIRED - set(df.columns)
        if missing:
            raise ValueError(f"{p} missing columns: {missing}")
        # add optional cols if absent
        for col in OPTIONAL:
            if col not in df.columns:
                df[col] = 0.0
        frames.append(df)
    pricing = pd.concat(frames, ignore_index=True)

    # types
    pricing["vcpus"] = pricing["vcpus"].astype(int)
    pricing["mem_gb"] = pricing["mem_gb"].astype(float)
    pricing["price_per_hour"] = pricing["price_per_hour"].astype(float)
    pricing["storage_price_per_gb_month"] = pricing["storage_price_per_gb_month"].astype(float)
    pricing["iops_included"] = pricing["iops_included"].astype(float)
    pricing["iops_price_per_iops_month"] = pricing["iops_price_per_iops_month"].astype(float)
    pricing["provider"] = pricing["provider"].str.lower()
    return pricing

def pick_candidates(pricing, vcpus_needed, mem_gb_needed, oversub_factor=1.0):
    v = int((vcpus_needed * oversub_factor + 0.999)//1)
    m = float(mem_gb_needed * oversub_factor)
    candidates = pricing[(pricing["vcpus"] >= v) & (pricing["mem_gb"] >= m)].copy()
    candidates = candidates.sort_values(["price_per_hour","mem_gb","vcpus"])
    return candidates
