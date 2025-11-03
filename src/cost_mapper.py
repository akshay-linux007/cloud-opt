import argparse, math
import pandas as pd
from pathlib import Path
from pricing import load_pricing, pick_candidates

def read_metrics(path: Path):
    df = pd.read_csv(path)
    cpu_peak = float(df["cpu_percent"].max())
    mem_95 = float(df["mem_used_mb"].quantile(0.95))
    cpu_count = int(df["cpu_count"].iloc[0])
    return cpu_peak, mem_95, cpu_count

def estimate_requirements(cpu_peak_percent, cpu_count, mem_95_mb, margin):
    vcpus_raw = math.ceil((cpu_peak_percent/100.0) * cpu_count)
    vcpus_needed = max(1, int(math.ceil(vcpus_raw * (1.0 + margin))))
    mem_req_gb = (mem_95_mb/1024.0) * (1.0 + margin)
    return vcpus_needed, mem_req_gb

def storage_monthly_to_hourly(storage_gb, price_per_gb_month):
    return (storage_gb * price_per_gb_month) / 730.0  # ~730 hrs/mo

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--aws_pricing", default="data/aws_pricing.csv")
    ap.add_argument("--azure_pricing", default="data/azure_pricing.csv")
    ap.add_argument("--gcp_pricing", default="data/gcp_pricing.csv")
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--duration_sec", type=int, default=60)
    ap.add_argument("--storage_gb", type=float, default=0.0)
    ap.add_argument("--provider", choices=["aws","azure","gcp","all"], default="all")
    args = ap.parse_args()

    cpu_peak, mem_95_mb, cpu_count = read_metrics(Path(args.metrics))
    vcpus_needed, mem_req_gb = estimate_requirements(cpu_peak, cpu_count, mem_95_mb, args.margin)

    csvs = []
    if args.provider in ("aws","all"): csvs.append(args.aws_pricing)
    if args.provider in ("azure","all"): csvs.append(args.azure_pricing)
    if args.provider in ("gcp","all"): csvs.append(args.gcp_pricing)
    pricing = load_pricing(csvs)

    candidates = pick_candidates(pricing, vcpus_needed, mem_req_gb, oversub_factor=1.0)
    if candidates.empty:
        print("No instance matches the requirement. Consider reducing margin or adding more SKUs.")
        return

    storage_hourly = candidates["storage_price_per_gb_month"].apply(
        lambda p: storage_monthly_to_hourly(args.storage_gb, p)
    )
    candidates["total_hourly"] = candidates["price_per_hour"] + storage_hourly
    run_hours = args.duration_sec / 3600.0
    candidates["est_run_cost"] = candidates["total_hourly"] * run_hours

    print(f"Observed peak cpu%: {cpu_peak:.1f}")
    print(f"CPU cores available: {cpu_count}")
    print(f"Estimated vCPUs needed: {vcpus_needed}")
    print(f"Observed mem (95th MB): {mem_95_mb:.2f}")
    print(f"Estimated mem(GB) with margin: {mem_req_gb:.3f}")
    print(f"Requested storage: {args.storage_gb:.1f} GB")

    print("\n--- Best per provider ---")
    for prov in ["aws","azure","gcp"]:
        sub = candidates[candidates["provider"]==prov]
        if not sub.empty:
            best = sub.nsmallest(1, "total_hourly").iloc[0]
            print(f"{prov.upper():5} | {best['instance_type']:20} | vCPU={int(best['vcpus'])} | "
                  f"Mem={best['mem_gb']:.1f} GB | Comp/hr=${best['price_per_hour']:.4f} | "
                  f"Stor/hr~=${storage_monthly_to_hourly(args.storage_gb, best['storage_price_per_gb_month']):.4f} | "
                  f"Total/hr=${best['total_hourly']:.4f} | Run=${best['est_run_cost']:.6f}")

    overall = candidates.nsmallest(1, "total_hourly").iloc[0]
    print("\n=== OVERALL RECOMMENDATION ===")
    print(f"{overall['provider'].upper()} | {overall['instance_type']} | vCPU={int(overall['vcpus'])} | "
          f"Mem={overall['mem_gb']:.1f} GB | Total/hr=${overall['total_hourly']:.4f} | "
          f"Run=${overall['est_run_cost']:.6f}")

if __name__ == "__main__":
    main()
