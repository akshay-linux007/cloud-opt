# src/pricing.py
from __future__ import annotations
import os
import pandas as pd

COMPUTE_CSV = "data/compute_pricing_regional.csv"
BLOCK_CSV   = "data/block_storage_pricing_regional.csv"
SHARED_CSV  = "data/shared_fs_pricing_regional.csv"

def _norm(df: pd.DataFrame, cols=("provider","region","storage_type","service","tier","instance_type")) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower().str.strip()
    return df

def load_regional_pricing() -> dict[str, pd.DataFrame]:
    missing = [p for p in (COMPUTE_CSV, BLOCK_CSV, SHARED_CSV) if not os.path.exists(p)]
    must = [COMPUTE_CSV, BLOCK_CSV]
    m_must = [p for p in missing if p in must]
    if m_must:
        raise FileNotFoundError(f"Missing pricing files: {m_must}.")
    compute = _norm(pd.read_csv(COMPUTE_CSV))
    block   = _norm(pd.read_csv(BLOCK_CSV))
    shared  = pd.DataFrame()
    if os.path.exists(SHARED_CSV):
        shared = _norm(pd.read_csv(SHARED_CSV))
    return {"compute": compute, "block": block, "shared": shared}

def build_candidates(pricing: dict[str, pd.DataFrame],
                     vcpus_needed: int,
                     mem_gb_needed: float,
                     storage_mode: str,
                     storage_class: str | None) -> pd.DataFrame:
    compute = pricing["compute"].copy()
    block   = pricing["block"].copy()

    compute = compute[(compute["vcpus"] >= vcpus_needed) & (compute["mem_gb"] >= mem_gb_needed)]
    if compute.empty:
        return pd.DataFrame()

    if storage_mode == "replicated":
        if storage_class and storage_class != "any":
            block_f = block[block["storage_type"] == storage_class]
        else:
            block_f = block
        cand = compute.merge(block_f, on=["provider","region"], how="inner")
        for c in ("iops_included","iops_price_per_iops_month"):
            if c not in cand.columns:
                cand[c] = 0.0
        cand = cand.rename(columns={"price_per_gb_month":"storage_price_per_gb_month"})
        cand["storage_type"] = cand["storage_type"]
        return cand[[
            "provider","region","instance_type","vcpus","mem_gb","price_per_hour",
            "storage_type","storage_price_per_gb_month","iops_included","iops_price_per_iops_month"
        ]]

    else:
        shared = pricing.get("shared", pd.DataFrame()).copy()
        if shared.empty:
            cand = compute.copy()
            cand["storage_type"] = "shared"
            cand["storage_price_per_gb_month"] = 0.0
            cand["iops_included"] = 0.0
            cand["iops_price_per_iops_month"] = 0.0
            return cand[[
                "provider","region","instance_type","vcpus","mem_gb","price_per_hour",
                "storage_type","storage_price_per_gb_month","iops_included","iops_price_per_iops_month"
            ]]
        shared = shared.rename(columns={"price_per_gb_month":"storage_price_per_gb_month"})
        cand = compute.merge(shared[["provider","region","service","tier","storage_price_per_gb_month"]],
                             on=["provider","region"], how="inner")
        cand["storage_type"] = cand["service"]
        cand["iops_included"] = 0.0
        cand["iops_price_per_iops_month"] = 0.0
        return cand[[
            "provider","region","instance_type","vcpus","mem_gb","price_per_hour",
            "storage_type","storage_price_per_gb_month","iops_included","iops_price_per_iops_month"
        ]]

def unique_regions(pricing: dict[str, pd.DataFrame], providers: list[str] | None = None) -> list[str]:
    df = pricing["compute"]
    if providers:
        df = df[df["provider"].isin([p.lower() for p in providers])]
    return sorted(df["region"].unique())
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
