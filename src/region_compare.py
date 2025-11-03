import math, os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pricing import load_pricing, pick_candidates

def ask(prompt, default=None):
    s = input(f"{prompt} " + (f"[{default}] " if default is not None else ""))
    return s.strip() if s.strip() != "" else default

def ask_float(prompt, default):
    return float(ask(prompt, default))

def ask_int(prompt, default):
    return int(ask(prompt, default))

def parse_size_to_gb(size_str):
    size_str = (size_str or "").lower().strip()
    if "tb" in size_str:
        return float(size_str.replace("tb","").strip()) * 1024.0
    if "gb" in size_str:
        return float(size_str.replace("gb","").strip())
    # raw numeric -> GB
    return float(size_str)

def storage_hourly_cost(gb, price_per_gb_month):
    return (gb * price_per_gb_month) / 730.0

def iops_hourly_cost(iops_needed, iops_included, price_per_iops_month):
    extra = max(0.0, float(iops_needed) - float(iops_included))
    return (extra * price_per_iops_month) / 730.0

def ensure_reports_dir():
    os.makedirs("reports", exist_ok=True)

def main():
    print("\n=== Multi-Region Cost Compare (AWS + Azure + GCP) ===\n")

    scope = (ask("Cloud scope (aws/azure/gcp/all)", "all") or "all").lower()

    # CPU/RAM input
    cpu_mode = ask("CPU sizing mode (vcpu/percent)", "vcpu")
    if cpu_mode == "vcpu":
        vcpus_needed = ask_int("How many vCPUs?", 8)
        mem_str = ask("RAM needed (e.g., 32 GB / 0.5 TB)", "32 GB")
    else:
        metrics_path = ask("Path to metrics CSV", "data/run_cpu2_mem256_r0_metrics.csv")
        dfm = pd.read_csv(metrics_path)
        cpu_peak = float(dfm["cpu_percent"].max())
        cpu_count = int(dfm["cpu_count"].iloc[0])
        vcpus_needed = max(1, int(math.ceil((cpu_peak/100.0) * cpu_count * 1.2)))
        mem_str = ask("RAM needed (e.g., 32 GB / 0.5 TB)", "32 GB")
    mem_gb_needed = parse_size_to_gb(mem_str)

    # Duration
    days = ask_int("Run duration — days", 1)
    hours = ask_int("Run duration — hours", 0)
    minutes = ask_int("Run duration — minutes", 0)
    duration_sec = days*86400 + hours*3600 + minutes*60
    run_hours = duration_sec/3600.0

    # Storage questions
    storage_mode = (ask("Storage mode (replicated/shared)", "replicated") or "replicated").lower()
    if storage_mode == "shared":
        size_input = ask("Shared filesystem size (GB or TB, e.g., 500 GB / 2 TB)", "1 TB")
    else:
        size_input = ask("Block storage size (GB or TB, e.g., 500 GB / 2 TB)", "100 GB")
    storage_gb = parse_size_to_gb(size_input)

    # Storage class filter (optional)
    storage_class = (ask("Storage class filter (aws: ebs-gp3/ebs-io2 | azure: managed-premium-ssd | gcp: pd-ssd/pd-balanced | any)", "any") or "any").lower()

    # IOPS
    iops_needed = ask_int("Required storage IOPS (provisioned)", 0)

    # Tax + INR
    apply_tax = (ask("Apply tax? (yes/no)", "no") or "no").lower() == "yes"
    inr_convert = (ask("Convert USD -> INR? (yes/no)", "no") or "no").lower() == "yes"
    usd_to_inr = 83

    # Load pricing + regions
    files = []
    if scope in ("aws","all"): files.append("data/aws_pricing.csv")
    if scope in ("azure","all"): files.append("data/azure_pricing.csv")
    if scope in ("gcp","all"): files.append("data/gcp_pricing.csv")
    pricing = load_pricing(files)

    if storage_class != "any" and "storage_type" in pricing.columns:
        pricing = pricing[pricing["storage_type"].str.lower() == storage_class]

    regions = pd.read_csv("data/regions.csv")
    if scope != "all":
        regions = regions[regions["provider"] == scope]

    rows = []
    for prov in ["aws","azure","gcp"]:
        if scope != "all" and prov != scope:
            continue
        prov_regions = regions[regions["provider"] == prov]
        if prov_regions.empty:
            continue

        for _, rgn in prov_regions.iterrows():
            # candidate selection (region-agnostic), then apply region multipliers
            cands = pick_candidates(pricing[pricing["provider"]==prov], vcpus_needed, mem_gb_needed)
            if cands.empty:
                continue

            cm, sm, im = float(rgn["compute_mult"]), float(rgn["storage_mult"]), float(rgn["iops_mult"])

            # per-candidate cost under this region's multipliers
            tmp = cands.copy()
            # compute
            tmp["price_per_hour_r"] = tmp["price_per_hour"] * cm
            # storage
            tmp["storage_hourly_r"] = tmp["storage_price_per_gb_month"].apply(lambda p: storage_hourly_cost(storage_gb, p*sm))
            # iops
            tmp["iops_hourly_r"] = tmp.apply(
                lambda rr: iops_hourly_cost(iops_needed, rr.get("iops_included", 0.0), rr.get("iops_price_per_iops_month", 0.0)*im),
                axis=1
            )
            tmp["total_hourly_r"] = tmp["price_per_hour_r"] + tmp["storage_hourly_r"] + tmp["iops_hourly_r"]
            tmp["est_cost_r"] = tmp["total_hourly_r"] * run_hours

            best = tmp.nsmallest(1, "total_hourly_r").iloc[0]
            rows.append({
                "provider": prov,
                "region": rgn["region"],
                "instance_type": best["instance_type"],
                "vcpus": int(best["vcpus"]),
                "mem_gb": float(best["mem_gb"]),
                "compute_hr": float(best["price_per_hour_r"]),
                "storage_hr": float(best["storage_hourly_r"]),
                "iops_hr": float(best["iops_hourly_r"]),
                "total_hr": float(best["total_hourly_r"]),
                "run_cost_usd": float(best["est_cost_r"])
            })

    if not rows:
        print("\n❌ No matches found. Add bigger SKUs or relax filters.\n")
        return

    out = pd.DataFrame(rows)
    tax = 1.18 if apply_tax else 1.00
    out["run_cost_final"] = out["run_cost_usd"] * (usd_to_inr if inr_convert else 1) * tax
    out["currency"] = "INR" if inr_convert else "USD"

    # Sort by total/hr ascending
    out = out.sort_values(["total_hr", "provider", "region"]).reset_index(drop=True)

    # Save table & charts
    ensure_reports_dir()
    out_path = Path("reports/region_costs.csv")
    out.to_csv(out_path, index=False)

    # Chart: best total/hr across regions (bar)
    plt.figure()
    labels = out.apply(lambda r: f"{r['provider'].upper()}:{r['region']}", axis=1).tolist()
    vals = out["total_hr"].tolist()
    plt.bar(labels, vals)
    plt.xticks(rotation=45, ha="right")
    plt.title("Best Total/hr by Region")
    plt.ylabel("USD per hour")
    plt.tight_layout()
    plt.savefig("reports/best_total_per_region.png")
    plt.close()

    # Print summary
    print("\n=== Multi-Region Comparison (best instance per region) ===")
    for _, r in out.iterrows():
        print(f"{r['provider'].upper():5} | {r['region']:12} | {r['instance_type']:20} | vCPU={r['vcpus']:3d} | Mem={r['mem_gb']:.0f} GB"
              f" | Comp/hr=${r['compute_hr']:.4f} | Stor/hr=${r['storage_hr']:.4f} | IOPS/hr=${r['iops_hr']:.4f}"
              f" | Total/hr=${r['total_hr']:.4f} | Run={r['run_cost_final']:.2f} {r['currency']}")

    print("\nSaved:")
    print(f" - {out_path}")
    print(" - reports/best_total_per_region.png\n")

if __name__ == "__main__":
    main()
