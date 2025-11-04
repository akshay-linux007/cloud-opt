# app.py
import math
import pandas as pd
import streamlit as st
from pathlib import Path

# ---------------- Existing repo helpers ----------------
# These are expected from your repo
# - src/pricing.py must expose: load_pricing(pricing_files) -> pd.DataFrame
#                               pick_candidates(df, vcpus_needed, mem_gb_needed) -> pd.DataFrame
from src.pricing import load_pricing, pick_candidates

# ---------------- Utilities ----------------
def parse_size_to_gb(size_str: str) -> float:
    """Parse strings like '32 GB', '0.5 TB', or '512' into GB as float."""
    s = (size_str or "").lower().strip()
    if "tb" in s:
        return float(s.replace("tb", "").strip()) * 1024.0
    if "gb" in s:
        return float(s.replace("gb", "").strip())
    # raw number is assumed GB
    return float(s)

def storage_hourly_cost(gb: float, price_per_gb_month: float) -> float:
    # ~730 hours per month
    return (gb * float(price_per_gb_month)) / 730.0

def iops_hourly_cost(iops_needed: float, iops_included: float, price_per_iops_month: float) -> float:
    extra = max(0.0, float(iops_needed) - float(iops_included or 0))
    return (extra * float(price_per_iops_month or 0.0)) / 730.0

def scope_to_providers(scope_value: str):
    if scope_value == "aws":
        return ["aws"]
    if scope_value == "azure":
        return ["azure"]
    if scope_value == "gcp":
        return ["gcp"]
    return ["aws", "azure", "gcp"]

# ---------------- Page setup ----------------
st.set_page_config(page_title="Cloud Cost Calculator", layout="wide")
st.title("Cloud Cost & Performance — Quick Estimator")
st.caption("Estimates a single run across AWS / Azure / GCP using your CSV pricing. Supports multi-region comparison.")

# ---------------- Sidebar: Inputs ----------------
st.sidebar.header("Inputs")

# Scope (providers)
scope = st.sidebar.selectbox("Cloud scope", ["all", "aws", "azure", "gcp"], index=0)

# CPU sizing mode
cpu_mode = st.sidebar.selectbox("CPU sizing mode", ["vcpu", "percent"], index=0)
if cpu_mode == "vcpu":
    vcpus_needed = st.sidebar.number_input("vCPUs", min_value=1, value=8, step=1)
    mem_str = st.sidebar.text_input("RAM needed (e.g., 32 GB / 0.5 TB)", "32 GB")
else:
    st.sidebar.info("Percent mode placeholder — using estimated vCPUs for now.")
    vcpus_needed = st.sidebar.number_input("Estimated vCPUs (fallback)", min_value=1, value=8, step=1)
    mem_str = st.sidebar.text_input("RAM needed (e.g., 32 GB / 0.5 TB)", "32 GB")

# Duration
days = st.sidebar.number_input("Run duration — days", min_value=0, value=1, step=1)
hours = st.sidebar.number_input("Run duration — hours", min_value=0, value=0, step=1)
minutes = st.sidebar.number_input("Run duration — minutes", min_value=0, value=0, step=5)
run_hours = (days * 86400 + hours * 3600 + minutes * 60) / 3600.0

# --------------- Regions (moved here as requested) ---------------
# Load available regions
try:
    regions_df = pd.read_csv("data/regions.csv")
except Exception as e:
    st.sidebar.error(f"Failed to load data/regions.csv — {e}")
    regions_df = pd.DataFrame(columns=["provider", "region", "display_name"])

providers_in_scope = scope_to_providers(scope)

regions_all = (
    regions_df[regions_df["provider"].isin(providers_in_scope)]["region"]
    .dropna().drop_duplicates().sort_values().tolist()
)

# If display names are present, show friendly names to the user
use_display = "display_name" in regions_df.columns and regions_df["display_name"].notna().any()
region_mode = st.sidebar.radio("Regions to price", ["All regions", "Choose region(s)"], index=0)

if region_mode == "Choose region(s)":
    if use_display:
        friendly = (
            regions_df[regions_df["provider"].isin(providers_in_scope)]
            .dropna(subset=["region", "display_name"])
            .drop_duplicates(subset=["region"])
            .sort_values("display_name")
        )
        display_map = dict(zip(friendly["display_name"], friendly["region"]))
        chosen_display = st.sidebar.multiselect("Select regions", list(display_map.keys()),
                                               default=list(display_map.keys())[:3])
        selected_regions = [display_map[d] for d in chosen_display]
    else:
        selected_regions = st.sidebar.multiselect("Select regions", regions_all, default=regions_all[:3])
else:
    selected_regions = regions_all

st.sidebar.caption(f"Regions selected: {len(selected_regions)}")

# Storage
storage_mode = st.sidebar.selectbox("Storage mode", ["replicated", "shared"], index=0)
size_label = "Shared filesystem size (GB or TB)" if storage_mode == "shared" else "Block storage size (GB or TB)"
storage_size = st.sidebar.text_input(size_label, "100 GB")

# Storage class — DROPDOWN (replaces free-text)
# Keep “Any” to avoid over-filtering; you asked to remove the *free-text* “any”, not the option entirely.
storage_class_options = [
    "Any",
    "ebs-gp3", "ebs-io2",              # AWS samples
    "managed-premium-ssd",             # Azure sample
    "pd-ssd", "pd-balanced"            # GCP samples
]
storage_class_choice = st.sidebar.selectbox("Storage class", storage_class_options, index=0)

# IOPS
iops_needed = st.sidebar.number_input("Required storage IOPS (provisioned)", min_value=0, value=0, step=500)

# Taxes / currency
apply_tax = st.sidebar.toggle("Apply 18% tax", value=False)
convert_inr = st.sidebar.toggle("Convert USD → INR (x83)", value=False)

go = st.sidebar.button("Calculate")

# ---------------- Load pricing ----------------
pricing_files = []
if scope in ("aws", "all"): pricing_files.append("data/aws_pricing.csv")
if scope in ("azure", "all"): pricing_files.append("data/azure_pricing.csv")
if scope in ("gcp", "all"): pricing_files.append("data/gcp_pricing.csv")

try:
    pricing = load_pricing(pricing_files)
except Exception as e:
    st.error(f"Failed to load pricing CSVs: {e}")
    st.stop()

# Optional filter by storage class
if storage_class_choice != "Any" and "storage_type" in pricing.columns:
    pricing = pricing[pricing["storage_type"].str.lower() == storage_class_choice.lower()]

# ---------------- Summarize inputs ----------------
mem_gb_needed = parse_size_to_gb(mem_str)
storage_gb = parse_size_to_gb(storage_size)

with st.expander("Input summary", expanded=True):
    st.write({
        "scope": scope,
        "cpu_mode": cpu_mode,
        "vcpus_needed": int(vcpus_needed),
        "mem_gb_needed": float(mem_gb_needed),
        "run_hours": float(run_hours),
        "regions_selected": selected_regions,
        "storage_mode": storage_mode,
        "storage_gb": float(storage_gb),
        "storage_class": storage_class_choice,
        "iops_needed": int(iops_needed),
        "apply_tax_18pct": apply_tax,
        "convert_usd_to_inr_83": convert_inr
    })

if not go:
    st.info("Set inputs and click **Calculate**.")
    st.stop()

# ---------------- Compute per-region results ----------------
if len(selected_regions) == 0:
    st.warning("No regions available in scope. Please choose at least one region.")
    st.stop()

candidates = pick_candidates(pricing, vcpus_needed, mem_gb_needed)
if candidates.empty:
    st.warning("No matching instance found. Add larger SKUs to CSVs or relax filters.")
    st.stop()

# Common numbers
USD_TO_INR = 83.0
tax_mult = 1.18 if apply_tax else 1.0
curr_mult = USD_TO_INR if convert_inr else 1.0
curr_label = "INR" if convert_inr else "USD"

# We’ll compute best candidate (by hourly total) for each region
rows = []
for region in selected_regions:
    # Here we reuse the same candidates for each region (your CSVs can later be extended with region-specific pricing)
    cand = candidates.copy()
    cand["storage_hourly"] = cand["storage_price_per_gb_month"].apply(lambda p: storage_hourly_cost(storage_gb, p))
    cand["iops_hourly"] = cand.apply(
        lambda r: iops_hourly_cost(iops_needed, r.get("iops_included", 0.0), r.get("iops_price_per_iops_month", 0.0)),
        axis=1
    )
    cand["total_hourly"] = cand["price_per_hour"] + cand["storage_hourly"] + cand["iops_hourly"]
    cand["est_cost_usd"] = cand["total_hourly"] * run_hours

    # Best for this region
    best_row = cand.nsmallest(1, "total_hourly").iloc[0]

    rows.append({
        "provider": best_row["provider"].upper(),
        "region": region,
        "instance_type": best_row["instance_type"],
        "vcpus": int(best_row["vcpus"]),
        "mem_gb": float(best_row["mem_gb"]),
        "compute_hr_usd": round(float(best_row["price_per_hour"]), 6),
        "storage_hr_usd": round(float(best_row["storage_hourly"]), 6),
        "iops_hr_usd": round(float(best_row["iops_hourly"]), 6),
        "total_hr_usd": round(float(best_row["total_hourly"]), 6),
        "run_cost": round(float(best_row["est_cost_usd"]) * tax_mult * curr_mult, 2),
        "currency": curr_label
    })

results = pd.DataFrame(rows)

# ---------------- Best overall ----------------
# Choose the lowest total_hr_usd across regions
best_overall_idx = results["total_hr_usd"].astype(float).idxmin()
best_overall = results.loc[best_overall_idx]

st.subheader("Best overall")
st.markdown(
    f"**{best_overall['provider']} | {best_overall['instance_type']} — {best_overall['region']}**  \n"
    f"vCPU={best_overall['vcpus']}, RAM={int(best_overall['mem_gb'])} GB  \n"
    f"Compute/hr: **${best_overall['compute_hr_usd']:.4f}** • Storage/hr: **${best_overall['storage_hr_usd']:.4f}** "
    f"• IOPS/hr: **${best_overall['iops_hr_usd']:.4f}**  \n"
    f"**Total/hr: ${best_overall['total_hr_usd']:.4f}**  •  **Run: {best_overall['run_cost']:,.2f} {best_overall['currency']}**"
)

st.divider()

# ---------------- All regions table ----------------
st.subheader("Per-region recommendation")
st.dataframe(
    results[[
        "provider", "region", "instance_type", "vcpus", "mem_gb",
        "compute_hr_usd", "storage_hr_usd", "iops_hr_usd", "total_hr_usd", "run_cost", "currency"
    ]],
    use_container_width=True
)

st.caption("Pricing is derived from the CSVs in /data and your selection rules in src/pricing.py. "
           "Region-specific price variance can be added by extending the CSVs to include per-region rates.")
