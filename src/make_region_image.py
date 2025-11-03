import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

IN_CSV = Path("reports/region_costs.csv")
OUT_PNG = Path("reports/region_summary.png")

def format_currency(v):
    try:
        return f"${v:,.2f}"
    except:
        return str(v)

def main():
    if not IN_CSV.exists():
        print(f"❌ Missing {IN_CSV}. Run:  python src/region_compare.py  first.")
        return

    df = pd.read_csv(IN_CSV)
    if df.empty:
        print("❌ CSV is empty.")
        return

    # Sort by total/hr ascending for display
    df = df.sort_values("total_hr", ascending=True).reset_index(drop=True)

    # Build a label like "AWS:us-east-1"
    df["label"] = df.apply(lambda r: f"{str(r['provider']).upper()}:{r['region']}", axis=1)

    # Winner row (cheapest total/hr)
    winner = df.iloc[0]

    # Prepare figure
    os.makedirs("reports", exist_ok=True)
    plt.figure(figsize=(16, 10), dpi=150)
    gs = GridSpec(3, 2, height_ratios=[1, 2, 2], width_ratios=[2, 1])  # title row, then charts+table

    # ---------------- Title + Input summary box ----------------
    ax_title = plt.subplot(gs[0, :])
    ax_title.axis("off")
    title = "Multi-Region Cost Summary (Best instance per region)"
    summary = (
        f"Currency: {winner.get('currency','USD')} | "
        f"Example winner: {winner['provider'].upper()} {winner['region']} • {winner['instance_type']} "
        f"(vCPU={int(winner['vcpus'])}, Mem={int(winner['mem_gb'])} GB)\n"
        f"Costs: Compute/hr={format_currency(winner['compute_hr'])} • "
        f"Storage/hr={format_currency(winner['storage_hr'])} • IOPS/hr={format_currency(winner['iops_hr'])} • "
        f"Total/hr={format_currency(winner['total_hr'])} • "
        f"Run={format_currency(winner['run_cost_final'])}"
    )
    ax_title.text(0.01, 0.75, title, fontsize=18, fontweight="bold")
    ax_title.text(0.01, 0.30, summary, fontsize=11)

    # ---------------- Bar chart: Best total/hr per region ----------------
    ax_bar = plt.subplot(gs[1, 0])
    labels = df["label"].tolist()
    vals = df["total_hr"].tolist()
    ax_bar.bar(labels, vals)
    ax_bar.set_title("Best Total/hr by Region")
    ax_bar.set_ylabel("USD per hour")
    ax_bar.set_xticklabels(labels, rotation=45, ha="right")
    ax_bar.grid(axis="y", linestyle="--", alpha=0.3)

    # ---------------- Winner breakdown box ----------------
    ax_winner = plt.subplot(gs[1, 1])
    ax_winner.axis("off")
    wb = (
        f"OVERALL WINNER\n\n"
        f"Provider: {winner['provider'].upper()}\n"
        f"Region:   {winner['region']}\n"
        f"Instance: {winner['instance_type']}\n"
        f"vCPU:     {int(winner['vcpus'])}\n"
        f"Memory:   {int(winner['mem_gb'])} GB\n\n"
        f"Compute/hr: {format_currency(winner['compute_hr'])}\n"
        f"Storage/hr: {format_currency(winner['storage_hr'])}\n"
        f"IOPS/hr:    {format_currency(winner['iops_hr'])}\n"
        f"Total/hr:   {format_currency(winner['total_hr'])}\n\n"
        f"Run Cost:   {format_currency(winner['run_cost_final'])} {winner.get('currency','')}"
    )
    ax_winner.text(0.02, 0.98, wb, va="top", fontsize=12, bbox=dict(boxstyle="round", alpha=0.05, edgecolor="0.7"))

    # ---------------- Top 10 table ----------------
    ax_tbl = plt.subplot(gs[2, :])
    ax_tbl.axis("off")
    top = df.head(10)[
        ["provider","region","instance_type","vcpus","mem_gb","total_hr","run_cost_final","currency"]
    ].copy()
    top["provider"] = top["provider"].str.upper()
    top["mem_gb"] = top["mem_gb"].astype(int)
    top["vcpus"] = top["vcpus"].astype(int)
    top["total_hr"] = top["total_hr"].map(lambda x: f"${x:,.4f}")
    # format run cost with currency
    def fmt_run(r):
        cur = r["currency"]
        val = r["run_cost_final"]
        if cur == "INR":
            return f"₹{val:,.0f}"
        return f"${val:,.2f}"
    top["run_cost"] = top.apply(fmt_run, axis=1)
    top = top.drop(columns=["run_cost_final","currency"])
    top.columns = ["Provider","Region","Instance","vCPU","Mem(GB)","Total/hr","Run"]

    table = ax_tbl.table(
        cellText=top.values,
        colLabels=top.columns,
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)

    plt.tight_layout()
    plt.savefig(OUT_PNG, bbox_inches="tight")
    plt.close()
    print(f"Saved image: {OUT_PNG}")

if __name__ == "__main__":
    main()
