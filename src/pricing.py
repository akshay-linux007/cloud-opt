# src/pricing.py — regional pricing engine with stable wrappers
from __future__ import annotations
import os
import pandas as pd

# ---------------- CSV paths (regional) ----------------
COMPUTE_CSV = "data/compute_pricing_regional.csv"
BLOCK_CSV   = "data/block_storage_pricing_regional.csv"
SHARED_CSV  = "data/shared_fs_pricing_regional.csv"
OBJECT_CSV  = "data/object_storage_pricing_regional.csv"  # NEW

# ---------------- Helpers ----------------
def _norm(
    df: pd.DataFrame,
    lower_cols=("provider","region","instance_type","storage_type","service","tier")
) -> pd.DataFrame:
    """Lower/strip key string columns for safe joins & filters."""
    df = df.copy()
    for c in lower_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower().str.strip()
    return df

def _ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df

# ---------------- Public API ----------------
def load_pricing() -> dict:
    """
    Load all regional pricing tables.
    Returns a dict of dataframes:
      {
        'compute': <df>,
        'block':   <df>,
        'shared':  <df or empty>,
        'object':  <df or empty>
      }
    """
    # Compute & block are required
    must = [COMPUTE_CSV, BLOCK_CSV]
    missing = [p for p in must if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing pricing files: {missing}")

    compute = _norm(pd.read_csv(COMPUTE_CSV))
    block   = _norm(pd.read_csv(BLOCK_CSV))
    shared  = _norm(pd.read_csv(SHARED_CSV)) if os.path.exists(SHARED_CSV) else pd.DataFrame()
    object_ = _norm(pd.read_csv(OBJECT_CSV)) if os.path.exists(OBJECT_CSV) else pd.DataFrame()

    # Minimal schema checks (compute & block)
    for name, df, req_cols in [
        ("compute", compute, ["provider","region","instance_type","vcpus","mem_gb","price_per_hour"]),
        ("block",   block,   ["provider","region","storage_type","price_per_gb_month"]),
    ]:
        missing_cols = [c for c in req_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"{name} CSV missing columns: {missing_cols}")

    # Types
    compute = _ensure_numeric(compute, ["vcpus","mem_gb","price_per_hour"])
    block   = _ensure_numeric(block,   ["price_per_gb_month","iops_included","iops_price_per_iops_month"])

    if not shared.empty:
        shared = _ensure_numeric(shared, ["price_per_gb_month","throughput_price_per_mb_s_month"])
    if not object_.empty:
        object_ = _ensure_numeric(object_, ["price_per_gb_month"])

    return {"compute": compute, "block": block, "shared": shared, "object": object_}

def _best_compute(compute: pd.DataFrame, vcpus_needed: int, mem_gb_needed: float) -> pd.DataFrame:
    """
    Filter to SKUs that satisfy vCPU & RAM, then prefer smallest that fits.
    Keep one best-fitting row per provider+region.
    """
    df = compute[(compute["vcpus"] >= vcpus_needed) & (compute["mem_gb"] >= mem_gb_needed)].copy()
    if df.empty:
        return df
    # tie-breakers: smaller vcpu, then smaller mem, then cheaper price
    df = df.sort_values(["vcpus","mem_gb","price_per_hour"], ascending=[True, True, True])
    # best per provider+region
    df = df.groupby(["provider","region"], as_index=False).first()
    return df

def pick_candidates(
    pricing: dict,
    vcpus_needed: int,
    mem_gb_needed: float,
    storage_mode: str,
    storage_class: str | None = None,
    scope: str = "all",
) -> pd.DataFrame:
    """
    Return a compute candidate per provider+region that fits vCPU/RAM.
    Columns returned (storage* are zero-filled for back-compat; UI computes real storage later):
      provider, region, instance_type, vcpus, mem_gb, price_per_hour,
      storage_price_per_gb_month, iops_included, iops_price_per_iops_month
    """
    compute = pricing.get("compute", pd.DataFrame()).copy()
    block   = pricing.get("block",   pd.DataFrame()).copy()   # kept for schema back-compat check
    # shared/object present in pricing dict but not used here (UI does storage selection)

    if compute.empty or block.empty:
        return pd.DataFrame()

    # Optional provider scoping
    if scope in ("aws","azure","gcp"):
        compute = compute[compute["provider"] == scope]

    base = _best_compute(compute, vcpus_needed, mem_gb_needed)
    if base.empty:
        return pd.DataFrame()

    # Zero-fill storage columns; app.py now picks the best of block/shared/object
    out = base.copy()
    out["storage_price_per_gb_month"]   = 0.0
    out["iops_included"]                = 0.0
    out["iops_price_per_iops_month"]    = 0.0

    # Final column ordering
    ordered = [
        "provider","region","instance_type","vcpus","mem_gb","price_per_hour",
        "storage_price_per_gb_month","iops_included","iops_price_per_iops_month"
    ]
    out = out[[c for c in ordered if c in out.columns]].copy()
    return out
