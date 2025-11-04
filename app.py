# app.py — Cloud Cost & Performance Optimizer (Regions: single select or compare-all)

import math
from pathlib import Path
import pandas as pd
import streamlit as st

try:
    import altair as alt
except Exception:
    alt = None

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# ===== import your existing pricing helpers =====
try:
    from src.pricing import load_pricing, pick_candidates  # type: ignore
    HAVE_PRICING_MODULE = True
except Exception:
    HAVE_PRICING_MODULE = False

    def load_pricing(pricing_files):
        frames = []
        for f in pricing_files:
            p = (DATA_DIR / Path(f).name) if not Path(f).exists() else Path(f)
            frames.append(pd.read_csv(p))
        df = pd.concat(frames, ignore_index=True).fillna(0)
        for col, default in [
            ("provider", ""),
            ("instance_type", ""),
            ("vcpus", 0),
            ("mem_gb", 0.0),
            ("price_per_hour", 0.0),
            ("storage_type", ""),
            ("storage_price_per_gb_month", 0.0),
            ("iops_included", 0.0),
            ("iops_price_per_iops_month", 0.0),
        ]:
            if col not in df.columns:
                df[col] = default
        return df

    def pick_candidates(df, vcpus_needed, mem_gb_needed):
        filt = (df["vcpus"].astype(float) >= float(vcpus_needed)) & (
            df["mem_gb"].astype(float) >= float(mem_gb_needed)
        )
        cand = df.loc[filt].copy()
        if cand.empty:
            return cand
        return cand.sort_values(["price_per_hour", "vcpus", "mem_gb"])


# ===== helpers =====
def parse_size_to_gb(size_str: str) -> float:
    if size_str is None:
        return 0.0
    s = str(size_str).lower().strip()
    if not s:
        return 0.0
    try:
        return float(s)  # plain number => GB
    except ValueError:
        pass
    if "tb" in s:
        return float(s.replace("tb", "").strip()) * 1024.0
    if "gb" in s:
        return float(s.replace("gb", "").strip())
    nums = "".join(ch if (ch.isdigit() or ch in ".") else " " for ch in s).split()
    return float(nums[0]) if nums else 0.0


def storage_hourly_cost(gb: float, price_per_gb_month: float) -> float:
    return float(gb) * float(price_per_gb_month) / 730.0  # ~730 hrs/mo


def iops_hourly_cost(iops_needed: float, iops_included: float, price_per_iops_month: float) -> float:
    extra = max(0.0, float(iops_needed) - float(iops_included))
    return extra * float(price_per_iops_month) / 730.0


@st.cache_data(show_spinner=False)
def load_regions_csv() -> pd.DataFrame:
    """
    Expected columns in data/regions.csv:
      provider, region
    Optional:
      display_name, price_multiplier  (e.g., 1.00 for baseline, 1.05 for +5%)
    """
    p = DATA_DIR / "regions.csv"
    if not p.exists():
        # sensible defaults if file is missing
        data = [
            {"provider": "aws", "region": "us-east-1", "display_name": "AWS us-east-1", "price_multiplier": 1.00},
            {"provider": "aws", "region": "ap-south-1", "display_name": "AWS ap-south-1", "price_multiplier": 1.02},
            {"provider": "azure", "region": "eastus", "display_name": "Azure eastus", "price_multiplier": 1.00},
            {"provider": "azure", "region": "centralindia", "display_name": "Azure centralindia", "price_multiplier": 1.03},
            {"provider": "gcp", "region": "us-central1", "display_name": "GCP us-central1", "price_multiplier": 1.00},
            {"provider": "gcp", "region": "asia-south1", "display_name": "GCP asia-south1", "price_multiplier": 1.02},
        ]
        return pd.DataFrame(data)
    df = pd.read_csv(p)
    if "display_name" not in df.columns:
        df["display_name"] = df["provider"].astype(str).str.upper() + " " + df["region"].astype(str)
    if "price_multiplier" not in df.columns:
        df["price_multiplier"] = 1.00
    return df


# ===== Page setup =====
st.set_page_config(page_title="Cloud Cost & Performance Optimizer", layout="wide")
st.markdown(
    """
<div style="display:flex;align-items:center;gap:14px;padding:12px 16px;border-radius:12px;background:linear-gradient(90deg,#0e1117,#111827);border:1px solid #1f2937;">
  <div style="line-height:1.2">
    <div style="font-weight:700;font-size:18px;">Cloud Cost & Performance Optimizer</div>
    <div style="opacity:0.8;font-size:13px;">Multi-cloud estimator · Compare regions or pick one</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
st.write("")

# ===== Sidebar inputs =====
st.sidebar.header("Inputs")

scope = st.sidebar.selectbox("Cloud scope", ["all", "aws", "azure", "gcp"], index=0)

cpu_mode = st.sidebar.selectbox("CPU sizing mode", ["vcpu", "percent"], index=0)
if cpu_mode == "vcpu":
    vcpus_needed = st.sidebar.number_input("vCPUs", min_value=1, value=8, step=1)
    mem_str = st.sidebar.text_input("RAM needed (e.g., 32 GB / 0.5 TB)", "32 GB")
else:
    st.sidebar.info("Percent mode uses fallback values until telemetry is integrated.")
    vcpus_needed = st.sidebar.number_input("Estimated vCPUs (fallback)", min_value=1, value=8, step=1)
    mem_str = st.sidebar.text_input("RAM needed (e.g., 32 GB / 0.5 TB)", "32 GB")

mem_gb_needed = parse_size_to_gb(mem_str)

col_dhm = st.sidebar.columns(3)
with col_dhm[0]:
    days = st.number_input("Days", min_value=0, value=1, step=1)
with col_dhm[1]:
    hours = st.number_input("Hours", min_value=0, value=0, step=1)
with col_dhm[2]:
    minutes = st.number_input("Minutes", min_value=0, value=0, step=5)
run_hours = (days * 86400 + hours * 3600 + minutes * 60) / 3600.0

storage_mode = st.sidebar.selectbox("Storage mode", ["replicated", "shared"], index=0)
size_label = "Shared FS size (GB or TB)" if storage_mode == "shared" else "Block storage size (GB or TB)"
storage_size = st.sidebar.text_input(size_label, "100 GB")
storage_gb = parse_size_to_gb(storage_size)

storage_class = st.sidebar.text_input(
    "Storage class filter (any | ebs-gp3 | ebs-io2 | managed-premium-ssd | pd-ssd | pd-balanced)",
    "any",
).lower()

iops_needed = st.sidebar.number_input("Required storage IOPS (provisioned)", min_value=0, value=0, step=500)

st.sidebar.markdown("---")
apply_tax = st.sidebar.toggle("Apply 18% tax", value=False)
convert_inr = st.sidebar.toggle("Convert USD → INR (x83)", value=False)

# ==== Region selection: Specific region OR Compare all ====
regions_df = load_regions_csv()
if scope != "all":
    regions_df = regions_df[regions_df["provider"].astype(str).str.lower() == scope.lower()].copy()
regions_df = regions_df.reset_index(drop=True)

region_choices = ["Compare all regions"] + regions_df["display_name"].tolist()
region_choice = st.sidebar.selectbox("Region scope", region_choices, index=0)

go = st.sidebar.button("Calculate", use_container_width=True)

# ===== Load pricing =====
pricing_files = []
if scope in ("aws", "all"):
    pricing_files.append("data/aws_pricing.csv")
if scope in ("azure", "all"):
    pricing_files.append("data/azure_pricing.csv")
if scope in ("gcp", "all"):
    pricing_files.append("data/gcp_pricing.csv")

try:
    pricing_df = load_pricing(pricing_files)
except Exception as e:
    st.error(f"Failed to load pricing CSVs: {e}")
    st.stop()

if storage_class != "any" and "storage_type" in pricing_df.columns:
    pricing_df = pricing_df[pricing_df["storage_type"].astype(str).str.lower() == storage_class]

# ===== Input summary =====
with st.expander("Input summary", expanded=True):
    st.write(
        {
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
            "convert_inr": convert_inr,
            "region_choice": region_choice,
        }
    )

if not go:
    st.info("Set inputs in the sidebar and click **Calculate**.")
    st.stop()

# ===== Compute base candidates (no regional multiplier yet) =====
candidates = pick_candidates(pricing_df, vcpus_needed, mem_gb_needed)
if candidates.empty:
    st.warning("❌ No matching instance found. Add larger SKUs to CSVs or relax filters.")
    st.stop()

# Ensure columns exist
for col, default in [
    ("storage_price_per_gb_month", 0.0),
    ("iops_included", 0.0),
    ("iops_price_per_iops_month", 0.0),
]:
    if col not in candidates.columns:
        candidates[col] = default

# Base hourly
candidates = candidates.copy()
candidates["storage_hourly"] = candidates["storage_price_per_gb_month"].apply(
    lambda p: storage_hourly_cost(storage_gb, p)
)
candidates["iops_hourly"] = candidates.apply(
    lambda r: iops_hourly_cost(iops_needed, r.get("iops_included", 0.0), r.get("iops_price_per_iops_month", 0.0)),
    axis=1,
)
candidates["total_hourly_base"] = (
    candidates["price_per_hour"].astype(float)
    + candidates["storage_hourly"].astype(float)
    + candidates["iops_hourly"].astype(float)
)

# Best SKU ignoring region multiplier
best_base = candidates.nsmallest(1, "total_hourly_base").iloc[0]

USD_TO_INR = 83.0
tax_mult = 1.18 if apply_tax else 1.0
curr_mult = USD_TO_INR if convert_inr else 1.0
curr_label = "INR" if convert_inr else "USD"

def apply_region_multiplier(hourly_usd: float, multiplier: float) -> float:
    return float(hourly_usd) * float(multiplier)

# ===== Two modes: Compare-all OR single region =====
left, right = st.columns([0.55, 0.45], vertical_alignment="top")

if region_choice == "Compare all regions":
    with left:
        st.subheader("Recommended Instance (by base hourly)")
        st.markdown(
            f"""
<div style="border:1px solid #1f2937;padding:16px;border-radius:12px;background:#0b1220;">
  <div style="font-weight:700;font-size:18px;margin-bottom:8px;">
    {str(best_base['provider']).upper()} &nbsp;|&nbsp; {best_base['instance_type']}
  </div>
  <div style="opacity:0.9;">vCPU = <b>{int(float(best_base['vcpus']))}</b> &nbsp;&nbsp;•&nbsp;&nbsp; RAM = <b>{float(best_base['mem_gb']):.0f} GB</b></div>
  <div style="margin-top:8px;">Storage Mode: <b>{storage_mode}</b> &nbsp;&nbsp;•&nbsp;&nbsp; Size: <b>{storage_gb:.0f} GB</b> &nbsp;&nbsp;•&nbsp;&nbsp; IOPS: <b>{iops_needed}</b></div>
  <hr style="border-color:#1f2937;margin:12px 0;">
  <div>Compute/hr: <b>${float(best_base['price_per_hour']):.4f}</b> &nbsp;&nbsp;•&nbsp;&nbsp; Storage/hr: <b>${float(best_base['storage_hourly']):.4f}</b> &nbsp;&nbsp;•&nbsp;&nbsp; IOPS/hr: <b>${float(best_base['iops_hourly']):.4f}</b></div>
  <div style="margin-top:6px;">Base Total/hr (before region & tax/currency): <b>${float(best_base['total_hourly_base']):.4f}</b></div>
</div>
""",
            unsafe_allow_html=True,
        )

        with st.expander("Detailed candidate table"):
            show_cols = [
                "provider","instance_type","vcpus","mem_gb",
                "price_per_hour","storage_hourly","iops_hourly","total_hourly_base"
            ]
            st.dataframe(candidates[show_cols].reset_index(drop=True), use_container_width=True)

    with right:
        st.subheader("Multi-Region Comparison")

        if regions_df.empty:
            st.info("No regions available. Add rows to data/regions.csv.")
        else:
            # Use the best_base SKU across all listed regions; apply multiplier per region
            comp = []
            for _, r in regions_df.iterrows():
                mult = float(r.get("price_multiplier", 1.0))
                total_hourly_usd = apply_region_multiplier(float(best_base["total_hourly_base"]), mult)
                total_hourly_conv = total_hourly_usd * tax_mult * curr_mult
                run_total_conv = total_hourly_usd * float(run_hours) * tax_mult * curr_mult
                comp.append(
                    {
                        "provider": str(r["provider"]).lower(),
                        "region": r["region"],
                        "display_name": r.get("display_name", f"{r['provider']} {r['region']}"),
                        "price_multiplier": mult,
                        "total_hourly_conv": total_hourly_conv,
                        "run_total_conv": run_total_conv,
                    }
                )
            reg_costs = pd.DataFrame(comp).sort_values("run_total_conv")

            # Best region
            best_row = reg_costs.iloc[0]
            st.success(
                f"**Best region:** {best_row['display_name']} — "
                f"Run total: **{best_row['run_total_conv']:,.2f} {curr_label}** "
                f"(multiplier ×{best_row['price_multiplier']:.3f})"
            )

            st.dataframe(reg_costs[["display_name","price_multiplier","total_hourly_conv","run_total_conv"]]
                         .rename(columns={"display_name":"Region"})
                         .reset_index(drop=True),
                         use_container_width=True)

            if alt is not None and not reg_costs.empty:
                chart = (
                    alt.Chart(reg_costs)
                    .mark_bar()
                    .encode(
                        x=alt.X("display_name:N", sort="-y", title="Region"),
                        y=alt.Y("run_total_conv:Q", title=f"Run Cost ({curr_label})"),
                        color="provider:N",
                        tooltip=["provider","region","price_multiplier",
                                 alt.Tooltip("run_total_conv:Q", format=",.2f")],
                    )
                    .properties(height=300)
                )
                st.altair_chart(chart, use_container_width=True)

            st.download_button(
                "Download region comparison (CSV)",
                data=reg_costs[["provider","region","display_name","price_multiplier","total_hourly_conv","run_total_conv"]].to_csv(index=False),
                file_name="region_costs.csv",
                mime="text/csv",
                use_container_width=True,
            )

else:
    # Single specific region selected
    row = regions_df.loc[regions_df["display_name"] == region_choice].iloc[0]
    mult = float(row.get("price_multiplier", 1.0))

    # Apply multiplier and currency/tax to the best_base SKU
    total_hourly_usd = apply_region_multiplier(float(best_base["total_hourly_base"]), mult)
    total_hourly_conv = total_hourly_usd * tax_mult * curr_mult
    total_run_conv = total_hourly_usd * float(run_hours) * tax_mult * curr_mult

    st.subheader("Selected Region Recommendation")
    st.markdown(
        f"""
<div style="border:1px solid #1f2937;padding:16px;border-radius:12px;background:#0b1220;">
  <div style="font-weight:700;font-size:18px;margin-bottom:8px;">
    {str(best_base['provider']).upper()} &nbsp;|&nbsp; {best_base['instance_type']} &nbsp;•&nbsp; {row['display_name']}
  </div>
  <div style="opacity:0.9;">vCPU = <b>{int(float(best_base['vcpus']))}</b> &nbsp;&nbsp;•&nbsp;&nbsp; RAM = <b>{float(best_base['mem_gb']):.0f} GB</b></div>
  <div style="margin-top:8px;">Storage Mode: <b>{storage_mode}</b> &nbsp;&nbsp;•&nbsp;&nbsp; Size: <b>{storage_gb:.0f} GB</b> &nbsp;&nbsp;•&nbsp;&nbsp; IOPS: <b>{iops_needed}</b></div>
  <hr style="border-color:#1f2937;margin:12px 0;">
  <div>Region price multiplier: <b>×{mult:.3f}</b></div>
  <div style="margin-top:6px;font-size:16px;">Total/hr: <b>{total_hourly_conv:.4f} {curr_label}</b></div>
  <div style="margin-top:4px;font-size:16px;">Run total: <b>{total_run_conv:,.2f} {curr_label}</b></div>
</div>
""",
        unsafe_allow_html=True,
    )

st.caption("Notes: If your pricing CSVs don't vary by region, costs will only change when a region price multiplier is set in data/regions.csv.")
