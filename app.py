# app.py — Cloud Cost Calculator (macro regions + auto storage + clean export)
import math
import pandas as pd
import streamlit as st

from src.pricing import load_pricing  # we use pricing tables directly

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

def infer_macro_region(provider: str, region: str) -> str:
    """Map cloud regions to macro regions: US, Europe, India (others ignored)."""
    p = (provider or "").lower()
    r = (region or "").lower()

    # AWS
    if p == "aws":
        if r.startswith("us-"):
            return "US"
        if r.startswith("eu-"):
            return "Europe"
        if r == "ap-south-1":  # Mumbai
            return "India"

    # Azure
    if p == "azure":
        if r in {"eastus", "westus", "westus2", "centralus", "southcentralus", "northcentralus"}:
            return "US"
        if r in {"westeurope", "northeurope"}:
            return "Europe"
        if r in {"centralindia", "southindia", "westindia"}:
            return "India"

    # GCP
    if p == "gcp":
        if r.startswith("us-"):
            return "US"
        if r.startswith("europe-"):
            return "Europe"
        if r in {"asia-south1", "asia-south2"}:  # Mumbai, Delhi
            return "India"

    return ""  # anything else not in requested macro-regions

# -------- Page setup --------
st.set_page_config(page_title="Cloud Cost Calculator", layout="wide")
st.title("Cloud Cost & Performance Optimizer — Regional Estimator")
st.caption("Estimates run cost across AWS / Azure / GCP with regional CSV pricing. Auto-picks the cheapest storage (block/shared/object) when selected.")

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

# -------- Build option lists --------
region_options = sorted(compute_df["region"].dropna().unique()) if "region" in compute_df else []

storage_class_set = set()
if "storage_type" in block_df:
    storage_class_set.update(block_df["storage_type"].dropna().unique())
if "service" in shared_df:
    storage_class_set.update(shared_df["service"].dropna().unique())
if "storage_type" in object_df:
    storage_class_set.update(object_df["storage_type"].dropna().unique())
storage_dropdown = ["any"] + sorted(storage_class_set)

# -------- Sidebar inputs --------
st.sidebar.header("Inputs")

scope = st.sidebar.selectbox("Cloud scope", ["all", "aws", "azure", "gcp"], index=0)

st.sidebar.subheader("Runtime")
days    = st.sidebar.number_input("Days",    min_value=0, value=1, step=1)
hours   = st.sidebar.number_input("Hours",   min_value=0, value=0, step=1)
minutes = st.sidebar.number_input("Minutes", min_value=0, value=0, step=5)
run_hours = (days*86400 + hours*3600 + minutes*60) / 3600.0

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

st.sidebar.subheader("Storage")
storage_mode = st.sidebar.selectbox(
    "Storage mode",
    ["auto (pick cheapest)", "replicated (block)", "shared (nfs/smb)", "object"],
    index=0
)
storage_size = st.sidebar.text_input("Data size (GB or TB) — for block/shared/object", "100 GB")
storage_gb = parse_size_to_gb(storage_size)

storage_class = st.sidebar.selectbox(
    "Storage class / tier (optional filter)",
    storage_dropdown,
    index=0,
    help="e.g., ebs-gp3, pd-ssd, managed-premium-ssd, efs, filestore, s3-standard, azure-blob-hot, gcs-standard"
)

iops_needed = st.sidebar.number_input("Required storage IOPS (if block)", min_value=0, value=0, step=500)

st.sidebar.subheader("Currency / Tax")
apply_tax = st.sidebar.toggle("Apply 18% tax", value=False)
convert_inr = st.sidebar.toggle("Convert USD → INR (x83)", value=False)

go = st.sidebar.button("Calculate")

# -------- Input summary --------
with st.expander("Input summary", expanded=True):
    st.write({
        "scope": scope,
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

# Provider scope
if scope != "all" and "provider" in base.columns:
    base = base[base["provider"] == scope]

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
    df["storage_hourly"] = df["price_per_gb_month"].fillna(0.0).apply(lambda p: storage_hourly_cost(storage_gb, p))
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
    if storage_class != "any" and "service" in df.columns:
        df = df[df["service"] == storage_class]
    df["price_per_gb_month"] = df["price_per_gb_month"].fillna(0.0)
    df["storage_hourly"] = df["price_per_gb_month"].apply(lambda p: storage_hourly_cost(storage_gb, p))
    df["iops_hourly"] = 0.0
    df["storage_choice"] = df.get("service", "shared").fillna("shared")
    return df

def build_object_table(base_df: pd.DataFrame) -> pd.DataFrame:
    if object_df.empty:
        return pd.DataFrame()
    df = base_df.merge(object_df, on=["provider","region"], how="left", suffixes=("","_obj"))
    if storage_class != "any" and "storage_type" in df.columns:
        df = df[df["storage_type"] == storage_class]
    df["price_per_gb_month"] = df["price_per_gb_month"].fillna(0.0)
    df["storage_hourly"] = df["price_per_gb_month"].apply(lambda p: storage_hourly_cost(storage_gb, p))
    df["iops_hourly"] = 0.0
    df["storage_choice"] = df.get("storage_type", "object").fillna("object")
    return df

# Build per selected mode
mode_key = storage_mode.split()[0]  # "auto", "replicated", "shared", "object"

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
    all_modes["storage_hourly"] = all_modes["storage_hourly"].fillna(0.0)
    all_modes["iops_hourly"]    = all_modes["iops_hourly"].fillna(0.0)
    all_modes["total_hourly_usd"] = (
        all_modes["price_per_hour"].astype(float)
        + all_modes["storage_hourly"].astype(float)
        + all_modes["iops_hourly"].astype(float)
    )
    # First: pick CHEAPEST *per provider+region+instance* (dedupe)
    all_modes["row_rank_inner"] = (
        all_modes.groupby(["provider","region","instance_type"])["total_hourly_usd"]
        .rank(method="first")
    )
    all_modes = all_modes[all_modes["row_rank_inner"] == 1.0].drop(columns=["row_rank_inner"])
    candidates = all_modes
else:
    if mode_key.startswith("replicated"):
        candidates = build_block_table(base)
    elif mode_key.startswith("shared"):
        candidates = build_shared_table(base)
    else:
        candidates = build_object_table(base)

if candidates is None or candidates.empty:
    st.warning("No candidates after applying storage filters/tiers.")
    st.stop()

# -------- Currency/tax & totals --------
USD_TO_INR = 83.0
tax_mult   = 1.18 if apply_tax else 1.0
curr_mult  = USD_TO_INR if convert_inr else 1.0
curr_label = "INR" if convert_inr else "USD"

candidates = candidates.copy()
candidates["total_hourly_usd"] = (
    candidates["price_per_hour"].astype(float)
    + candidates["storage_hourly"].astype(float)
    + candidates["iops_hourly"].astype(float)
)
candidates["est_cost_run_usd"]  = candidates["total_hourly_usd"] * float(run_hours)
candidates["total_hourly_disp"] = candidates["total_hourly_usd"] * tax_mult * curr_mult
candidates["est_cost_run_disp"] = candidates["est_cost_run_usd"] * tax_mult * curr_mult

# -------- Macro-region folding: US / Europe / India --------
candidates["macro_region"] = candidates.apply(
    lambda r: infer_macro_region(r.get("provider",""), r.get("region","")),
    axis=1
)
# keep only the requested macro regions
candidates = candidates[candidates["macro_region"].isin(["US","Europe","India"])]

if candidates.empty:
    st.warning("No candidates in macro regions US/Europe/India for the requested size.")
    st.stop()

# Pick ONE cheapest row per (provider, macro_region)
candidates = (
    candidates.sort_values("total_hourly_disp")
    .groupby(["provider","macro_region"], as_index=False)
    .first()
)

# If single-cloud scope, keep only that provider
if scope in ("aws","azure","gcp"):
    candidates = candidates[candidates["provider"] == scope]

if candidates.empty:
    st.warning("No rows after applying cloud scope.")
    st.stop()

# -------- Recommendation (cheapest among displayed rows) --------
best = candidates.nsmallest(1, "total_hourly_disp").iloc[0]

st.subheader("Recommendation")
st.write(
    f"**{best['provider'].upper()} | {best['instance_type']} | {best['macro_region']} ({best['region']})**  \n"
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

# -------- Per-cloud × macro-region table (clean & short) --------
st.markdown("---")
st.subheader("Per-cloud cost across US / Europe / India (best region picked per macro-zone)")

show_cols = [
    "provider","macro_region","region","instance_type","vcpus","mem_gb",
    "price_per_hour","storage_hourly","iops_hourly","storage_choice",
    "total_hourly_disp","est_cost_run_disp"
]
show_cols = [c for c in show_cols if c in candidates.columns]

pretty = (
    candidates[show_cols]
    .rename(columns={
        "price_per_hour": "compute_hr_usd",
        "storage_hourly": "storage_hr_usd",
        "iops_hourly": "iops_hr_usd",
        "total_hourly_disp": f"total_hr_{curr_label.lower()}",
        "est_cost_run_disp": f"run_cost_{curr_label.lower()}",
    })
    .sort_values(by=[ "provider", "macro_region" ])
)

st.dataframe(pretty, use_container_width=True, hide_index=True)

# -------- CSV Export (only displayed rows) --------
csv_bytes = pretty.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download results (CSV)",
    data=csv_bytes,
    file_name="cloud-opt_results.csv",
    mime="text/csv",
)
