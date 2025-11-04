# src/pricing.py — regional pricing engine with stable wrappers
import os
import pandas as pd

# CSVs (regional)
COMPUTE_CSV = "data/compute_pricing_regional.csv"
BLOCK_CSV   = "data/block_storage_pricing_regional.csv"
SHARED_CSV  = "data/shared_fs_pricing_regional.csv"

def _norm(df: pd.DataFrame, lower_cols=("provider","region","instance_type","storage_type","service","tier")) -> pd.DataFrame:
    df = df.copy()
    for c in lower_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower().str.strip()
    return df

def load_pricing() -> dict:
    """
    Wrapper used by app.py.
    Returns a dict with three dataframes: compute, block, shared.
    """
    # Strictly require compute + block; shared is optional
    missing = [p for p in (COMPUTE_CSV, BLOCK_CSV) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing pricing files: {missing}")

    compute = _norm(pd.read_csv(COMPUTE_CSV))
    block   = _norm(pd.read_csv(BLOCK_CSV))
    shared  = pd.DataFrame()
    if os.path.exists(SHARED_CSV):
        shared = _norm(pd.read_csv(SHARED_CSV))

    # Minimal schema checks
    for req, cols in [
        ("compute", ["provider","region","instance_type","vcpus","mem_gb","price_per_hour"]),
        ("block",   ["provider","region","storage_type","price_per_gb_month"]),
    ]:
        df = {"compute": compute, "block": block}.get(req, None)
        if df is not None:
            missing_cols = [c for c in cols if c not in df.columns]
            if missing_cols:
                raise ValueError(f"{req} CSV missing columns: {missing_cols}")

    # Make sure numeric cols are numeric
    for df, cols in [
        (compute, ["vcpus","mem_gb","price_per_hour"]),
        (block,   ["price_per_gb_month","iops_included","iops_price_per_iops_month"]),
    ]:
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    if not shared.empty:
        for c in ["price_per_gb_month","throughput_price_per_mb_s_month"]:
            if c in shared.columns:
                shared[c] = pd.to_numeric(shared[c], errors="coerce").fillna(0.0)

    return {"compute": compute, "block": block, "shared": shared}

def _best_compute(compute: pd.DataFrame, vcpus_needed: int, mem_gb_needed: float) -> pd.DataFrame:
    """
    Keep only SKUs that can satisfy the requested vCPU & RAM, then prefer smallest that fits.
    """
    df = compute[(compute["vcpus"] >= vcpus_needed) & (compute["mem_gb"] >= mem_gb_needed)].copy()
    if df.empty:
        return df
    # tie-breakers: smaller vcpu, then smaller mem, then cheaper price
    df = df.sort_values(["vcpus","mem_gb","price_per_hour"], ascending=[True, True, True])
    # keep the 1st per provider+region (best fitting per region)
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
    Wrapper used by app.py.
    Returns dataframe with columns:
      provider, region, instance_type, vcpus, mem_gb, price_per_hour,
      storage_price_per_gb_month, iops_included, iops_price_per_iops_month
    """
    compute = pricing.get("compute", pd.DataFrame()).copy()
    block   = pricing.get("block",   pd.DataFrame()).copy()
    shared  = pricing.get("shared",  pd.DataFrame()).copy()

    if compute.empty or block.empty:
        return pd.DataFrame()

    # Optional provider scope
    if scope in ("aws","azure","gcp"):
        compute = compute[compute["provider"] == scope]
        block   = block[block["provider"]   == scope]
        if not shared.empty:
            shared = shared[shared["provider"] == scope]

    # Find best compute candidate per provider+region
    base = _best_compute(compute, vcpus_needed, mem_gb_needed)
    if base.empty:
        return pd.DataFrame()

    # Attach storage pricing depending on storage_mode
    out = base.copy()

    if (storage_mode or "").lower() == "shared":
        # Use shared FS table (service/tier)
        if shared.empty:
            # if no shared table, fall back to zero storage cost
            out["storage_price_per_gb_month"] = 0.0
            out["iops_included"] = 0.0
            out["iops_price_per_iops_month"] = 0.0
        else:
            s = shared.copy()
            # Filter by chosen "class" if provided (match against 'service' or 'tier')
            if storage_class:
                s = s[(s["service"] == storage_class) | (s.get("tier","") == storage_class)]
                if s.empty:
                    return pd.DataFrame()
            # For shared FS, we map price_per_gb_month from shared CSV
            s = s.rename(columns={"price_per_gb_month": "storage_price_per_gb_month"})
            # Join on provider+region; take min price per region/service
            s = s.groupby(["provider","region"], as_index=False)[["storage_price_per_gb_month"]].min()
            out = out.merge(s, on=["provider","region"], how="left")
            out["storage_price_per_gb_month"] = out["storage_price_per_gb_month"].fillna(0.0)
            out["iops_included"] = 0.0
            out["iops_price_per_iops_month"] = 0.0

    else:
        # Replicated/block storage path
        b = block.copy()
        if storage_class:
            b = b[b["storage_type"] == storage_class]
            if b.empty:
                return pd.DataFrame()
        # Aggregate to the cheapest storage type per provider+region
        agg = b.groupby(["provider","region"], as_index=False).agg({
            "price_per_gb_month": "min",
            "iops_included": "max",
            "iops_price_per_iops_month": "min"
        })
        out = out.merge(
            agg.rename(columns={"price_per_gb_month":"storage_price_per_gb_month"}),
            on=["provider","region"],
            how="left"
        )
        for c in ["storage_price_per_gb_month","iops_included","iops_price_per_iops_month"]:
            out[c] = out[c].fillna(0.0)

    # Final ordering / columns
    ordered_cols = [
        "provider","region","instance_type","vcpus","mem_gb","price_per_hour",
        "storage_price_per_gb_month","iops_included","iops_price_per_iops_month"
    ]
    out = out[[c for c in ordered_cols if c in out.columns]].copy()
    return out
