import math
import pandas as pd
import streamlit as st
from pathlib import Path

# ------- import your existing pricing helpers -------
from src.pricing import load_pricing, pick_candidates

# ------- helpers -------
def parse_size_to_gb(size_str: str) -> float:
    s = (size_str or "").lower().strip()
    if "tb" in s:
        return float(s.replace("tb","").strip()) * 1024.0
    if "gb" in s:
        return float(s.replace("gb","").strip())
    return float(s)  # assume GB for raw number

def storage_hourly_cost(gb, price_per_gb_month):
    return (gb * price_per_gb_month) / 730.0  # ~730 hrs/month

def iops_hourly_cost(iops_needed, iops_included, price_per_iops_month):
    extra = max(0.0, float(iops_needed) - float(iops_included))
    return (extra * price_per_iops_month) / 730.0

# ------- page setup -------
st.set_page_config(page_title="Cloud Cost Calculator (Simple)", layout="centered")
st.title("Cloud Cost Calculator (Simple)")
st.caption("Single-run estimator across AWS / Azure / GCP based on CSV pricing in your repo.")

# ------- sidebar inputs -------
st.sidebar.header("Inputs")

scope = st.sidebar.selectbox("Cloud scope", ["all", "aws", "azure", "gcp"], index=0)

cpu_mode = st.sidebar.selectbox("CPU sizing mode", ["vcpu", "percent"], index=0)
if cpu_mode == "vcpu":
    vcpus_needed = st.sidebar.number_input("vCPUs", min_value=1, value=8, step=1)
    mem_str = st.sidebar.text_input("RAM needed (e.g., 32 GB / 0.5 TB)", "32 GB")
else:
    st.sidebar.info("Percent mode: enter fallback until metrics mode is wired.")
    vcpus_needed = st.sidebar.number_input("Estimated vCPUs (fallback)", min_value=1, value=8, step=1)
    mem_str = st.sidebar.text_input("RAM needed (e.g., 32 GB / 0.5 TB)", "32 GB")

mem_gb_needed = parse_size_to_gb(mem_str)

days = st.sidebar.number_input("Run duration — days", min_value=0, value=1, step=1)
hours = st.sidebar.number_input("Run duration — hours", min_value=0, value=0, step=1)
minutes = st.sidebar.number_input("Run duration — minutes", min_value=0, value=0, step=5)
run_hours = (days*86400 + hours*3600 + minutes*60) / 3600.0

storage_mode = st.sidebar.selectbox("Storage mode", ["replicated", "shared"], index=0)
size_label = "Shared FS size (GB or TB)" if storage_mode == "shared" else "Block storage size (GB or TB)"
storage_size = st.sidebar.text_input(size_label, "100 GB")
storage_gb = parse_size_to_gb(storage_size)

storage_class = st.sidebar.text_input(
    "Storage class filter (any | ebs-gp3 | ebs-io2 | managed-premium-ssd | pd-ssd | pd-balanced)",
    "any"
).lower()

iops_needed = st.sidebar.number_input("Required storage IOPS (provisioned)", min_value=0, value=0, step=500)

apply_tax = st.sidebar.toggle("Apply 18% tax", value=False)
convert_inr = st.sidebar.toggle("Convert USD → INR (x83)", value=False)

go = st.sidebar.button("Calculate")

# ------- load pricing -------
pricing_files = []
if scope in ("aws","all"): pricing_files.append("data/aws_pricing.csv")
if scope in ("azure","all"): pricing_files.append("data/azure_pricing.csv")
if scope in ("gcp","all"): pricing_files.append("data/gcp_pricing.csv")

try:
    pricing = load_pricing(pricing_files)
except Exception as e:
    st.error(f"Failed to load pricing CSVs: {e}")
    st.stop()

if storage_class != "any" and "storage_type" in pricing.columns:
    pricing = pricing[pricing["storage_type"].str.lower() == storage_class]

# ------- initial info -------
with st.expander("Input summary", expanded=True):
    st.write({
        "scope": scope,
        "cpu_mode": cpu_mode,
        "vcpus_needed": int(vcpus_needed),
        "mem_gb_needed": float(mem_gb_needed),
        "run_hours": float(run_hours),
        "storage_mode": storage_mode,
        "storage_gb": float(storage_gb),
        "storage_class_filter": storage_class,
        "iops_needed": int(iops_needed),
        "tax": apply_tax,
        "convert_inr": convert_inr
    })

if not go:
    st.info("Set inputs and click **Calculate**.")
    st.stop()

# ------- compute recommendation -------
candidates = pick_candidates(pricing, vcpus_needed, mem_gb_needed)
if candidates.empty:
    st.warning("No matching instance found. Add larger SKUs to CSVs or relax filters.")
    st.stop()

candidates["storage_hourly"] = candidates["storage_price_per_gb_month"].apply(lambda p: storage_hourly_cost(storage_gb, p))
candidates["iops_hourly"] = candidates.apply(
    lambda r: iops_hourly_cost(iops_needed, r.get("iops_included", 0.0), r.get("iops_price_per_iops_month", 0.0)),
    axis=1
)
candidates["total_hourly"] = candidates["price_per_hour"] + candidates["storage_hourly"] + candidates["iops_hourly"]
candidates["est_cost"] = candidates["total_hourly"] * run_hours

best = candidates.nsmallest(1, "total_hourly").iloc[0]

USD_TO_INR = 83.0
tax_mult = 1.18 if apply_tax else 1.0
curr_mult = USD_TO_INR if convert_inr else 1.0
curr_label = "INR" if convert_inr else "USD"

total_cost = best["est_cost"] * tax_mult * curr_mult

# ------- output (simple) -------
st.subheader("Recommendation")
st.write(f"**{best['provider'].upper()} | {best['instance_type']}** — vCPU={int(best['vcpus'])}, RAM={best['mem_gb']:.0f} GB")
st.write(f"Compute/hr: **${best['price_per_hour']:.4f}**  •  Storage/hr: **${best['storage_hourly']:.4f}**  •  IOPS/hr: **${best['iops_hourly']:.4f}**")
st.write(f"**Total/hr: ${best['total_hourly']:.4f}**  •  **Run: {total_cost:,.2f} {curr_label}**")

st.caption("Powered by CSV pricing in /data and your selection rules in src/pricing.py")
