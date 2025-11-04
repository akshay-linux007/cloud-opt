
# --- Back-compat wrappers so older app.py keeps working ---
def load_pricing(*_args, **_kwargs):
    """Back-compat: return regional pricing dict just like old loader."""
    return load_regional_pricing()

def pick_candidates(pricing, vcpus_needed, mem_gb_needed,
                    storage_mode="replicated", storage_class=None):
    """Back-compat: delegate to regional candidate builder."""
    return build_candidates(
        pricing=pricing,
        vcpus_needed=vcpus_needed,
        mem_gb_needed=mem_gb_needed,
        storage_mode=storage_mode,
        storage_class=storage_class,
    )

# src/pricing.py
from __future__ import annotations
import os
from typing import Iterable, Optional
import pandas as pd

# --- CSV paths (regional) ---
COMPUTE_CSV = "data/compute_pricing_regional.csv"
BLOCK_CSV   = "data/block_storage_pricing_regional.csv"
SHARED_CSV  = "data/shared_fs_pricing_regional.csv"

# ---------- helpers ----------
def _norm(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower().str.strip()
    return df

def _exists(path: str) -> bool:
    return os.path.exists(path) and os.path.isfile(path)

# ---------- loaders ----------
def load_regional_pricing() -> dict[str, pd.DataFrame]:
    """Load per-region compute + storage pricing and normalize keys."""
    must = [COMPUTE_CSV, BLOCK_CSV]
    missing = [p for p in must if not _exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing pricing files: {missing}")

    compute = pd.read_csv(COMPUTE_CSV)
    block   = pd.read_csv(BLOCK_CSV)
    shared  = pd.read_csv(SHARED_CSV) if _exists(SHARED_CSV) else pd.DataFrame()

    compute = _norm(compute, ["provider", "region", "instance_type"])
    block   = _norm(block,   ["provider", "region", "storage_type"])
    shared  = _norm(shared,  ["provider", "region", "service", "tier"])

    # Validate expected columns exist (fail fast with a helpful message)
    _expect_cols(compute, ["provider","region","instance_type","vcpus","mem_gb","price_per_hour"], "compute_pricing_regional.csv")
    _expect_cols(block,   ["provider","region","storage_type","price_per_gb_month","iops_included","iops_price_per_iops_month"], "block_storage_pricing_regional.csv")
    if not shared.empty:
        _expect_cols(shared, ["provider","region","service","tier","price_per_gb_month","throughput_price_per_mb_s_month"], "shared_fs_pricing_regional.csv")

    return {"compute": compute, "block": block, "shared": shared}

def _expect_cols(df: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")

# ---------- candidate builder ----------
def build_candidates(
    pricing: dict[str, pd.DataFrame],
    vcpus_needed: int,
    mem_gb_needed: float,
    storage_mode: str = "replicated",                    # 'replicated' (block) or 'shared'
    storage_class: Optional[str] = None,                 # e.g., 'ebs-gp3' | 'managed-premium-ssd' | 'pd-ssd' (for block)
    regions: Optional[Iterable[str]] = None,             # optional subset of regions to consider
    shared_service: Optional[str] = None,                # e.g., 'efs' | 'azure-files' | 'filestore'
    shared_tier: Optional[str] = None                    # e.g., 'standard' | 'premium'
) -> pd.DataFrame:
    """
    Return one row per (provider, region, instance_type), merged with appropriate storage pricing.
    No grouping — region stays intact.
    """
    compute = pricing["compute"].copy()
    block   = pricing["block"].copy()
    shared  = pricing["shared"].copy() if "shared" in pricing else pd.DataFrame()

    # Filter by size first (keeps rows per region)
    compute = compute[(compute["vcpus"] >= int(vcpus_needed)) & (compute["mem_gb"] >= float(mem_gb_needed))]
    if compute.empty:
        return pd.DataFrame()

    # Filter by region subset if provided
    if regions:
        rset = {r.lower().strip() for r in regions}
        compute = compute[compute["region"].isin(rset)]
        if compute.empty:
            return pd.DataFrame()

    # Decide storage source
    mode = (storage_mode or "replicated").lower().strip()
    if mode not in ("replicated", "shared"):
        mode = "replicated"

    if mode == "replicated":
        # Optional filter by storage_class for block
        blk = block.copy()
        if storage_class and storage_class.lower().strip() not in ("", "any"):
            sclass = storage_class.lower().strip()
            blk = blk[blk["storage_type"] == sclass]

        # left-join on (provider, region) to preserve region rows in compute
        merged = compute.merge(
            blk,
            how="left",
            on=["provider","region"],
            suffixes=("", "_blk")
        )

        # Rename block storage pricing columns to common names for downstream code
        merged = merged.rename(columns={
            "price_per_gb_month": "storage_price_per_gb_month",
            "iops_included": "iops_included",
            "iops_price_per_iops_month": "iops_price_per_iops_month",
            "storage_type": "storage_type"
        })

        # If a region has no block row (after filtering), drop it — we can’t price storage
        merged = merged.dropna(subset=["storage_price_per_gb_month"]).reset_index(drop=True)

        # Ensure numeric
        for c in ["price_per_hour","storage_price_per_gb_month","iops_included","iops_price_per_iops_month","vcpus","mem_gb"]:
            if c in merged.columns:
                merged[c] = pd.to_numeric(merged[c], errors="coerce")

        return merged[[
            "provider","region","instance_type","vcpus","mem_gb",
            "price_per_hour","storage_type",
            "storage_price_per_gb_month","iops_included","iops_price_per_iops_month"
        ]].copy()

    # shared mode
    if shared.empty:
        # No shared FS pricing table present
        return pd.DataFrame()

    sh = shared.copy()
    if shared_service and shared_service.strip().lower() not in ("", "any"):
        sh = sh[sh["service"] == shared_service.strip().lower()]
    if shared_tier and shared_tier.strip().lower() not in ("", "any"):
        sh = sh[sh["tier"] == shared_tier.strip().lower()]

    merged = compute.merge(
        sh,
        how="left",
        on=["provider","region"],
        suffixes=("", "_shared")
    )

    # If no shared row after filtering, drop (can’t price shared storage for that region)
    merged = merged.dropna(subset=["price_per_gb_month"]).reset_index(drop=True)

    # Map shared FS pricing to the common column names used by the app
    merged = merged.rename(columns={
        "price_per_gb_month": "storage_price_per_gb_month",
        "throughput_price_per_mb_s_month": "throughput_price_per_mb_s_month",
        "service": "shared_service",
        "tier": "shared_tier"
    })
    merged["iops_included"] = 0.0
    merged["iops_price_per_iops_month"] = 0.0
    merged["storage_type"] = merged.get("shared_service", "")

    # Ensure numeric
    for c in ["price_per_hour","storage_price_per_gb_month","throughput_price_per_mb_s_month","vcpus","mem_gb"]:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")

    return merged[[
        "provider","region","instance_type","vcpus","mem_gb",
        "price_per_hour","storage_type",
        "storage_price_per_gb_month","iops_included","iops_price_per_iops_month",
        "shared_service","shared_tier","throughput_price_per_mb_s_month"
    ]].copy()

# ---------- Back-compat wrappers for app.py ----------
def load_pricing(_files=None) -> dict[str, pd.DataFrame]:
    """Backwards-compatible wrapper: ignore _files and use regional CSVs."""
    return load_regional_pricing()

def pick_candidates(
    pricing: dict[str, pd.DataFrame],
    vcpus_needed: int,
    mem_gb_needed: float,
    storage_mode: str = "replicated",
    storage_class: Optional[str] = None,
    regions: Optional[Iterable[str]] = None,
    shared_service: Optional[str] = None,
    shared_tier: Optional[str] = None
) -> pd.DataFrame:
    """Backwards-compatible API used by app.py."""
    return build_candidates(
        pricing=pricing,
        vcpus_needed=vcpus_needed,
        mem_gb_needed=mem_gb_needed,
        storage_mode=storage_mode,
        storage_class=storage_class,
        regions=regions,
        shared_service=shared_service,
        shared_tier=shared_tier
    )
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
