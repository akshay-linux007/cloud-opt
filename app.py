# app.py — Cloud Cost Calculator (Auto storage selection + Object storage + Scope aware)
import math
import pandas as pd
import streamlit as st

from src.pricing import load_pricing  # pricing.py loads compute/block/shared/object CSVs

# -------- Helpers --------
def parse_size_to_gb(size_str: str) -> float:
    s = (size_str or "").lower().strip()
    if not s:
        return 0.0
    if "tb" in s:
        return float(s.replace("tb", "").strip()) * 1024.0
    if "gb" in s:
        return float(s.replace("gb", "").strip())
    return float(s)

def storage_hourly_cost(gb, price_per_gb_month):
    return (float(gb) * float(price_per_gb_month)) / 730.0

def iops_hourly_cost(iops_needed, iops_included, price_per_iops_month):
    extra = max(0.0, float(iops_needed) - float(iops_included))
    return (extra * float(price_per_iops_month)) / 730.0

def _normalize_cols(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower().str.strip()
    return df

# -------- Page setup --------
st.set_page_config(page_title="Cloud Cost Calculator", layout="wide")
st.title("Cloud Cost & Performance Optimizer — Regional Estimator")
st.caption("Estimates single-run costs across AWS / Azure / GCP using regional CSV pricing. Supports Block, Shared (NFS/SMB), and Object storage with Auto selection.")

# -------- Load pricing --------
try:
    pricing = load_pricing()
except Exception as e:
    st.error(f"Failed to load pricing: {e}")
    st.stop()

compute_df = pricing.get("compute", pd.DataFrame()).copy()
block_df   = pricing.get("block",   pd.DataFrame()).copy()
shared_df  = pricing.get("shared",  pd.DataFrame()).copy()
object_df  = pricing.get("object",  pd.DataFrame()).copy()

for df in (compute_df, block_df, shared_df, object_df):
    if not df.empty:
        _normalize_cols(df, ["provider","region","instance_type","storage_type","service","tier"])

# -------- Sidebar inputs --------
st.sidebar.header("Inputs")

scope = st.sidebar.selectbox("Cloud scope", ["all", "aws", "azure", "gcp"], index=0)

st.sidebar.subheader("Runtime")
days    = st.sidebar.number_input("Days",    min_value=0, value=1, step=1)
hours   = st.sidebar.number_input("Hours",   min_value=0, value=0, step=1)
minutes = st.sidebar.number_input("Minutes", min_value=0, value=0, step=5)
run_hours = (days*86400 + hours*3600 + minutes*60) / 3600.0

# Region options (scoped to provider selection so the list is relevant)
region_source = compute_df.copy()
if scope != "all" and "provider" in region_source.columns:
    region_source = region_source[region_source["provider"] == scope]
region_options = sorted(region_source["region"].dropna().unique()) if "region" in region_source else []

st.sidebar.subheader("Regions")
chosen_regions = st.sidebar.multiselect(
    "Select regions (leave empty = all)",
    region_options, default=[]
) if region_options else []

# Compute sizing
st.sidebar.subheader("Compute sizing")
cpu_mode = st.sidebar.selectbox("CPU sizing mode", ["vcpu", "percent"], index=0)
if cpu_mode == "vcpu":
    vcpus_needed = st.sidebar.number_input("vCPUs needed", min_value=1, value=8, step=1)
    mem_str = st.sidebar.text_input("RAM needed (e.g., 16 GB / 32 GB / 0.5 TB)", "32 GB")
else:
    st.sidebar.info("Enter your current usage and we’ll right-size vCPUs automatically.")
    current_vcpus = st.sidebar.number_input("Current On-Prem vCPUs", min_value=1, value=32)
    avg_cpu_percent = st.sidebar.slider("Avg CPU Utilization (%)", 1, 100, 40)
    target_util_percent = st.sidebar.slider("Target Cloud Utilization (%)", 10, 100, 70)
    vcpus_needed = max(1, math.ceil((current_vcpus * (avg_cpu_percent/100.0)) / (target_util_percent/100.0)))
    st.sidebar.success(f"Estimated vCPUs needed: {vcpus_needed}")
    mem_str = st.sidebar.text_input("RAM needed (e.g., 16 / 32 / 64 / 128 GB)", "32 GB")

mem_gb_needed = parse_size_to_gb(mem_str)

# Storage selectors
st.sidebar.subheader("Storage")
storage_mode = st.sidebar.selectbox(
    "Storage mode",
    ["auto (pick cheapest)", "replicated (block)", "shared (nfs/smb)", "object"],
    index=0
)
storage_size = st.sidebar.text_input("Data size (GB or TB) — for block/shared/object", "100 GB")
storage_gb = parse_size_to_gb(storage_size)

# Build storage class/tier list from all sources (block/shared/object; both service and storage_type)
storage_class_set = set()
if not block_df.empty and "storage_type" in block_df:
    storage_class_set.update(block_df["storage_type"].dropna().unique())
if not shared_df.empty and "service" in shared_df:
    storage_class_set.update(shared_df["service"].dropna().unique())
if not shared_df.empty and "tier" in shared_df:
    storage_class_set.update(shared_df["tier"].dropna().unique())
if not object_df.empty and "service" in object_df:
    storage_class_set.update(object_df["service"].dropna().unique())
if not object_df.empty and "storage_type" in object_df:
    storage_class_set.update(object_df["storage_type"].dropna().unique())

storage_dropdown = ["any"] + sorted({str(x).lower().strip() for x in storage_class_set})

storage_class = st.sidebar.selectbox(
    "Storage class / tier (optional filter)",
    storage_dropdown,
    index=0,
    help="e.g., ebs-gp3, pd-ssd, managed-premium-ssd, efs, filestore, s3-standard, azure-blob-hot, gcs-standard"
)

iops_needed = st.sidebar.number_input("Required storage IOPS (if block)", min_value=0, value=0, step=500)

# Currency & tax
st.sidebar.subheader("Currency / Tax")
apply_tax = st.sidebar.toggle("Apply 18% tax", value=False)
convert_inr = st.sidebar.toggle("Convert USD → INR (x83)", value=False)

go = st.sidebar.button("Calculate")

# -------- Input summary --------
with st.expander("Input summary", expanded=True):
    st.write({
        "scope": scope,
        "selected_regions": chosen_regions or "ALL",
        "cpu_mode": cpu_mode,
        "vcpus_needed": int(vcpus_needed),
        "mem_gb_needed": float(mem_gb_needed),
        "run_hours": float(run_hours),
        "storage_mode": storage_mode,
        "storage_gb": float(storage_gb),
        "storage_class": storage_class,
        "iops_needed": int(iops_needed),
        "tax": apply_tax,
        "convert_inr": convert_inr
    })

if not go:
    st.info("Set inputs and click **Calculate**.")
    st.stop()

# -------- Build base compute candidates --------
if compute_df.empty:
    st.error("Compute pricing CSV is empty.")
    st.stop()

base = compute_df.copy()

# Provider scope (this drives the recommendation to be AWS-only / Azure-only / GCP-only when chosen)
if scope != "all" and "provider" in base.columns:
    base = base[base["provider"] == scope]

# Region filter
if chosen_regions and "region" in base.columns:
    base = base[base["region"].isin(chosen_regions)]

# Size filter
if not {"vcpus","mem_gb"}.issubset(base.columns):
    st.error("compute_pricing_regional.csv must have columns: vcpus, mem_gb")
    st.stop()
base = base[(base["vcpus"] >= int(vcpus_needed)) & (base["mem_gb"] >= float(mem_gb_needed))]

if base.empty:
    st.warning("No matching compute instance for requested size. Add rows to CSVs or relax sizing.")
    st.stop()

# -------- Per-mode builders --------
def build_block_table(base_df: pd.DataFrame) -> pd.DataFrame:
    if block_df.empty:
        return pd.DataFrame()
    df = base_df.merge(block_df, on=["provider","region"], how="left", suffixes=("","_blk"))
    if storage_class != "any" and "storage_type" in df.columns:
        df = df[df["storage_type"] == storage_class]
    df["price_per_gb_month"] = pd.to_numeric(df.get("price_per_gb_month", 0.0), errors="coerce").fillna(0.0)
    df["iops_included"] = pd.to_numeric(df.get("iops_included", 0.0), errors="coerce").fillna(0.0)
    df["iops_price_per_iops_month"] = pd.to_numeric(df.get("iops_price_per_iops_month", 0.0), errors="coerce").fillna(0.0)
    df["storage_hourly"] = df["price_per_gb_month"].apply(lambda p: storage_hourly_cost(storage_gb, p))
    df["iops_hourly"] = df.apply(
        lambda r: iops_hourly_cost(iops_needed, r.get("iops_included",0.0), r.get("iops_price_per_iops_month",0.0)),
        axis=1
    )
    df["storage_choice"] = df.get("storage_type", "block").fillna("block")
    return df

def build_shared_table(base_df: pd.DataFrame) -> pd.DataFrame:
    if shared_df.empty:
        return pd.DataFrame()
    df = base_df.merge(shared_df, on=["provider","region"], how="left", suffixes=("","_sh"))
    # allow filter by either service or tier name
    if storage_class != "any" and ("service" in df.columns or "tier" in df.columns):
        df = df[(df.get("service","") == storage_class) | (df.get("tier","") == storage_class)]
    df["price_per_gb_month"] = pd.to_numeric(df.get("price_per_gb_month", 0.0), errors="coerce").fillna(0.0)
    df["storage_hourly"] = df["price_per_gb_month"].apply(lambda p: storage_hourly_cost(storage_gb, p))
    df["iops_hourly"] = 0.0
    df["storage_choice"] = df.get("service", "shared").fillna("shared")
    return df

def build_object_table(base_df: pd.DataFrame) -> pd.DataFrame:
    if object_df.empty:
        return pd.DataFrame()
    df = base_df.merge(object_df, on=["provider","region"], how="left", suffixes=("","_obj"))
    # allow filter by service or storage_type for object
    if storage_class != "any" and ("service" in df.columns or "storage_type" in df.columns):
        df = df[(df.get("service","") == storage_class) | (df.get("storage_type","") == storage_class)]
    df["price_per_gb_month"] = pd.to_numeric(df.get("price_per_gb_month", 0.0), errors="coerce").fillna(0.0)
    df["storage_hourly"] = df["price_per_gb_month"].apply(lambda p: storage_hourly_cost(storage_gb, p))
    df["iops_hourly"] = 0.0
    df["storage_choice"] = df.get("service", df.get("storage_type", "object")).fillna("object")
    return df

# Choose mode
mode_key = storage_mode.split()[0].lower()  # "auto", "replicated", "shared", "object"

tables = []
if mode_key == "auto":
    tb = build_block_table(base)
    ts = build_shared_table(base)
    to = build_object_table(base)
    tables = [t for t in (tb, ts, to) if not t.empty]
    if not tables:
        st.error("No storage pricing available (block/shared/object CSVs empty).")
        st.stop()
    all_modes = pd.concat(tables, ignore_index=True)
    for col in ("price_per_hour", "storage_hourly", "iops_hourly"):
        all_modes[col] = pd.to_numeric(all_modes.get(col, 0.0), errors="coerce").fillna(0.0)
    all_modes["total_hourly_usd"] = (
        all_modes["price_per_hour"] + all_modes["storage_hourly"] + all_modes["iops_hourly"]
    )
    # pick cheapest storage per (provider, region, instance_type)
    all_modes["row_rank"] = (
        all_modes.groupby(["provider","region","instance_type"])["total_hourly_usd"]
        .rank(method="first")
    )
    candidates = all_modes[all_modes["row_rank"] == 1.0].drop(columns=["row_rank"])
else:
    if mode_key.startswith("replicated"):
        candidates = build_block_table(base)
    elif mode_key.startswith("shared"):
        candidates = build_shared_table(base)
    else:  # object
        candidates = build_object_table(base)

if candidates is None or candidates.empty:
    st.warning("No candidates after applying storage filters/tiers.")
    st.stop()

# -------- currency/tax & totals --------
candidates = candidates.copy()
USD_TO_INR = 83.0
tax_mult   = 1.18 if apply_tax else 1.0
curr_mult  = USD_TO_INR if convert_inr else 1.0
curr_label = "INR" if convert_inr else "USD"

for col in ("price_per_hour","storage_hourly","iops_hourly"):
    candidates[col] = pd.to_numeric(candidates.get(col, 0.0), errors="coerce").fillna(0.0)

candidates["est_cost_run_usd"]  = (
    candidates["price_per_hour"] + candidates["storage_hourly"] + candidates["iops_hourly"]
) * float(run_hours)

candidates["total_hourly_disp"] = (
    candidates["price_per_hour"] + candidates["storage_hourly"] + candidates["iops_hourly"]
) * tax_mult * curr_mult

candidates["est_cost_run_disp"] = candidates["est_cost_run_usd"] * tax_mult * curr_mult

# -------- Recommendation (respects provider scope) --------
best = candidates.nsmallest(1, "total_hourly_disp").iloc[0]

st.subheader("Recommendation")
st.write(
    f"**{best['provider'].upper()} | {best['instance_type']} | {best['region']}**  \n"
    f"vCPU={int(best['vcpus'])}, RAM={best['mem_gb']:.0f} GB"
)
st.write(
    f"Compute/hr: **${float(best['price_per_hour']):.4f}**  •  "
    f"Storage/hr: **${float(best['storage_hourly']):.4f}**  •  "
    f"IOPS/hr: **${float(best['iops_hourly']):.4f}**"
)
st.write(
    f"**Total/hr: {float(best['total_hourly_disp']):.4f} {curr_label}**  •  "
    f"**Run: {float(best['est_cost_run_disp']):,.2f} {curr_label}**"
)
st.caption(f"Chosen storage: **{best.get('storage_choice','n/a')}**")

# -------- Per-region costs (all matching providers/regions) --------
st.markdown("---")
st.subheader("Per-region costs")

show = candidates.copy()
show["total_hr_disp"] = show["total_hourly_disp"]
show["run_cost_disp"] = show["est_cost_run_disp"]

columns = [
    "provider","region","instance_type","vcpus","mem_gb",
    "price_per_hour","storage_hourly","iops_hourly",
    "storage_choice","total_hr_disp","run_cost_disp"
]
columns = [c for c in columns if c in show.columns]

pretty = (
    show[columns]
    .rename(columns={
        "price_per_hour": "compute_hr_usd",
        "storage_hourly": "storage_hr_usd",
        "iops_hourly": "iops_hr_usd",
        "total_hr_disp": f"total_hr_{curr_label.lower()}",
        "run_cost_disp": f"run_cost_{curr_label.lower()}",
    })
    .sort_values(by=f"run_cost_{curr_label.lower()}")
)

st.dataframe(pretty, use_container_width=True, hide_index=True)

st.caption("If prices look identical, confirm your *_regional.csv files have region-specific numbers and the provider scope/regions match your selections.")
