# app.py — Cloud Cost Calculator (Regional-aware UI)
import math
from pathlib import Path
import pandas as pd
import streamlit as st

# Your pricing engine (back-compat wrapper names)
from src.pricing import load_pricing, pick_candidates

# ---------------- Helpers ----------------
def parse_size_to_gb(size_str: str) -> float:
    s = (size_str or "").lower().strip()
    if not s:
        return 0.0
    if "tb" in s:
        return float(s.replace("tb", "").strip()) * 1024.0
    if "gb" in s:
        return float(s.replace("gb", "").strip())
    # assume a plain number means GB
    return float(s)

def storage_hourly_cost(gb, price_per_gb_month):
    return (float(gb) * float(price_per_gb_month)) / 730.0  # ~730 hrs/month

def iops_hourly_cost(iops_needed, iops_included, price_per_iops_month):
    extra = max(0.0, float(iops_needed) - float(iops_included))
    return (extra * float(price_per_iops_month)) / 730.0

# ---------------- Page setup ----------------
st.set_page_config(page_title="Cloud Cost Calculator", layout="wide")
st.title("Cloud Cost & Performance Optimizer — Regional Estimator")
st.caption("Estimates single-run costs across AWS / Azure / GCP using regional CSV pricing.")

# ---------------- Load pricing once ----------------
try:
    # Updated pricing.py returns a dict with compute, block, (optional) shared
    pricing = load_pricing()
except Exception as e:
    st.error(f"Failed to load pricing: {e}")
    st.stop()

compute_df = pricing.get("compute", pd.DataFrame()).copy()
block_df   = pricing.get("block",   pd.DataFrame()).copy()
shared_df  = pricing.get("shared",  pd.DataFrame()).copy()

# Build storage class options from loaded CSVs
storage_class_options = set()
if not block_df.empty and "storage_type" in block_df.columns:
    storage_class_options.update(block_df["storage_type"].dropna().astype(str).unique())
if not shared_df.empty and "service" in shared_df.columns:
    storage_class_options.update(shared_df["service"].dropna().astype(str).unique())
storage_class_options = sorted(sc.lower().strip() for sc in storage_class_options)
storage_dropdown = ["any"] + storage_class_options

# ---------------- Sidebar inputs ----------------
st.sidebar.header("Inputs")

# Cloud scope (used to filter both the Regions list and the candidates)
scope = st.sidebar.selectbox("Cloud scope", ["all", "aws", "azure", "gcp"], index=0)

# Runtime
st.sidebar.subheader("Runtime")
days    = st.sidebar.number_input("Days",    min_value=0, value=1, step=1)
hours   = st.sidebar.number_input("Hours",   min_value=0, value=0, step=1)
minutes = st.sidebar.number_input("Minutes", min_value=0, value=0, step=5)
run_hours = (days*86400 + hours*3600 + minutes*60) / 3600.0

# Regions (show only regions for the selected cloud when a single cloud is chosen)
st.sidebar.subheader("Regions")
region_options = []
compute_df_scoped = compute_df.copy()
if scope in ("aws", "azure", "gcp") and "provider" in compute_df_scoped.columns:
    compute_df_scoped = compute_df_scoped[
        compute_df_scoped["provider"].str.lower() == scope
    ]
if not compute_df_scoped.empty and "region" in compute_df_scoped.columns:
    region_options = sorted(compute_df_scoped["region"].dropna().astype(str).unique())

if region_options:
    chosen_regions = st.sidebar.multiselect(
        "Select regions (leave empty = all)",
        region_options,
        default=[],
        help="Pick one or more regions. If none selected, the calculator evaluates all available regions."
    )
else:
    chosen_regions = []
    st.sidebar.info("No regions found in compute CSVs for the current scope.")

# ---- Compute sizing (real percent mode) ----
st.sidebar.subheader("Compute sizing")
cpu_mode = st.sidebar.selectbox("CPU sizing mode", ["vcpu", "percent"], index=0)

if cpu_mode == "vcpu":
    # direct entry
    vcpus_needed = st.sidebar.number_input("vCPUs needed", min_value=1, value=8, step=1)
    mem_str = st.sidebar.text_input("RAM needed (e.g., 16 GB / 32 GB / 0.5 TB)", "32 GB")

else:
    # Percent-driven right-sizing
    st.sidebar.info("Enter your current usage and we’ll right-size vCPUs automatically.")

    current_vcpus = st.sidebar.number_input("Current On-Prem vCPUs", min_value=1, value=32)
    avg_cpu_percent = st.sidebar.slider("Avg CPU Utilization (%)", 1, 100, 40)
    target_util_percent = st.sidebar.slider("Target Cloud Utilization (%)", 10, 100, 70,
                                            help="We’ll size so the cloud instance runs near this utilization.")

    vcpus_needed = max(
        1,
        math.ceil((current_vcpus * (avg_cpu_percent / 100.0)) / (target_util_percent / 100.0))
    )

    st.sidebar.success(f"Estimated vCPUs needed: {vcpus_needed}")

    mem_str = st.sidebar.text_input("RAM needed (e.g., 16 GB / 32 GB / 64 GB / 128 GB)", "32 GB")

mem_gb_needed = parse_size_to_gb(mem_str)

# Storage
st.sidebar.subheader("Storage")
storage_mode = st.sidebar.selectbox("Storage mode", ["replicated", "shared"], index=0)
size_label = "Shared FS size (GB or TB)" if storage_mode == "shared" else "Block storage size (GB or TB)"
storage_size = st.sidebar.text_input(size_label, "100 GB")
storage_gb = parse_size_to_gb(storage_size)

# Storage class dropdown
storage_class = st.sidebar.selectbox(
    "Storage class / service",
    storage_dropdown,
    index=0,
    help="Choose a specific class/service to filter (e.g., ebs-gp3, managed-premium-ssd, pd-ssd, efs, filestore)."
)

iops_needed = st.sidebar.number_input("Required storage IOPS (provisioned)", min_value=0, value=0, step=500)

# Currency & tax
st.sidebar.subheader("Currency / Tax")
apply_tax = st.sidebar.toggle("Apply 18% tax", value=False)
convert_inr = st.sidebar.toggle("Convert USD → INR (x83)", value=False)

go = st.sidebar.button("Calculate")

# ---------------- Input Summary ----------------
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

# ---------------- Compute candidates ----------------
candidates = pick_candidates(
    pricing=pricing,
    vcpus_needed=int(vcpus_needed),
    mem_gb_needed=float(mem_gb_needed),
    storage_mode=storage_mode,
    storage_class=None if storage_class == "any" else storage_class
)

if candidates is None or candidates.empty:
    st.warning("No matching instance found. Add larger SKUs to CSVs or relax filters.")
    st.stop()

# Provider filter from UI (apply BEFORE best pick)
if scope in ("aws", "azure", "gcp"):
    candidates = candidates[candidates["provider"].str.lower() == scope]
    if candidates.empty:
        st.warning(f"No matches for provider '{scope.upper()}' with the current inputs. Try relaxing vCPU/RAM or storage.")
        st.stop()

# Optional region filter from UI
if chosen_regions:
    candidates = candidates[candidates["region"].isin(chosen_regions)]
    if candidates.empty:
        st.warning("No candidates in the selected region(s). Try selecting more regions.")
        st.stop()

# Ensure required columns exist
for col in ("storage_price_per_gb_month", "iops_included", "iops_price_per_iops_month"):
    if col not in candidates.columns:
        candidates[col] = 0.0

# Hourly & run cost (USD base)
candidates = candidates.copy()
candidates["storage_hourly"]    = candidates["storage_price_per_gb_month"].apply(
    lambda p: storage_hourly_cost(storage_gb, p)
)
candidates["iops_hourly"]       = candidates.apply(
    lambda r: iops_hourly_cost(iops_needed, r.get("iops_included", 0.0), r.get("iops_price_per_iops_month", 0.0)),
    axis=1
)
candidates["total_hourly_usd"]  = candidates["price_per_hour"] + candidates["storage_hourly"] + candidates["iops_hourly"]
candidates["est_cost_run_usd"]  = candidates["total_hourly_usd"] * run_hours

# Currency / tax
USD_TO_INR = 83.0
tax_mult   = 1.18 if apply_tax else 1.0
curr_mult  = USD_TO_INR if convert_inr else 1.0
curr_label = "INR" if convert_inr else "USD"

candidates["total_hourly_disp"] = candidates["total_hourly_usd"] * tax_mult * curr_mult
candidates["est_cost_run_disp"] = candidates["est_cost_run_usd"] * tax_mult * curr_mult

# Best pick (scoped by provider if selected)
best = candidates.nsmallest(1, "total_hourly_usd").iloc[0]

# ---------------- Output ----------------
st.subheader("Recommendation")
st.write(
    f"**{best['provider'].upper()} | {best['instance_type']} | {best['region']}**  \n"
    f"vCPU={int(best['vcpus'])}, RAM={best['mem_gb']:.0f} GB"
)
st.write(
    f"Compute/hr: **${best['price_per_hour']:.4f}**  •  "
    f"Storage/hr: **${best['storage_hourly']:.4f}**  •  "
    f"IOPS/hr: **${best['iops_hourly']:.4f}**"
)
st.write(
    f"**Total/hr: {best['total_hourly_disp']:.4f} {curr_label}**  •  "
    f"**Run: {best['est_cost_run_disp']:,.2f} {curr_label}**"
)
st.caption("Computed from regional CSVs in `/data`, using your `src/pricing.py` selection rules.")

# -------- Per-region costs (scoped by provider if chosen; all providers when scope=all) --------
st.markdown("---")
st.subheader("Per-region costs")

region_table = candidates.copy()

# Final display columns in chosen currency
region_table["total_hr_disp"] = region_table["total_hourly_usd"] * tax_mult * curr_mult
region_table["run_cost_disp"] = region_table["est_cost_run_usd"] * tax_mult * curr_mult

show_cols = [
    "provider", "region", "instance_type", "vcpus", "mem_gb",
    "price_per_hour",      # compute/hr (USD)
    "storage_hourly",      # storage/hr (USD)
    "iops_hourly",         # iops/hr (USD)
    "total_hr_disp",       # currency/tax-adjusted
    "run_cost_disp"        # currency/tax-adjusted
]
show_cols = [c for c in show_cols if c in region_table.columns]

pretty = (
    region_table[show_cols]
    .rename(columns={
        "price_per_hour": "compute_hr_usd",
        "storage_hourly": "storage_hr_usd",
        "iops_hourly": "iops_hr_usd",
        "total_hr_disp": f"total_hr_{curr_label.lower()}",
        "run_cost_disp": f"run_cost_{curr_label.lower()}",
        "mem_gb": "mem_gb"
    })
    .sort_values(by=f"run_cost_{curr_label.lower()}")
)

st.dataframe(pretty, use_container_width=True, hide_index=True)

# -------- Footer --------
st.caption("Tip: If per-region prices look identical, verify your *_regional.csv files have different numbers across regions.")
