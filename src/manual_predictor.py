import math, os
import pandas as pd
import matplotlib.pyplot as plt
from pricing import load_pricing, pick_candidates

# ---------- helpers ----------
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
    # raw numeric -> assume GB
    return float(size_str)

def storage_hourly_cost(gb, price_per_gb_month):
    return (gb * price_per_gb_month) / 730.0  # ~730 hours/month

def iops_hourly_cost(iops_needed, iops_included, price_per_iops_month):
    extra = max(0.0, float(iops_needed) - float(iops_included))
    return (extra * price_per_iops_month) / 730.0

# Allowed storage types & tiers (light validation only)
ALLOWED = {
    "aws":   {"ebs-gp3": {"tiers": ["standard"]},
              "ebs-io2": {"tiers": ["standard","provisioned"]}},
    "azure": {"managed-premium-ssd": {"tiers": ["standard","premium"]},
              "managed-standard-ssd": {"tiers": ["standard"]}},
    "gcp":   {"pd-ssd": {"tiers": ["standard"]},
              "pd-balanced": {"tiers": ["standard"]}},
}

def validate_storage(provider, sclass, tier):
    # provider may be 'all'—we only validate when provider is a single cloud;
    # otherwise we accept 'any' or whatever is in CSV rows.
    if provider not in ALLOWED:
        return True, None
    allowed_classes = ALLOWED[provider]
    if sclass == "any":
        return True, None
    if sclass not in allowed_classes:
        return False, f"Storage class '{sclass}' not valid for {provider}. Allowed: {', '.join(allowed_classes.keys())}"
    if tier not in allowed_classes[sclass]["tiers"]:
        return False, f"Tier '{tier}' not valid for {provider}/{sclass}. Allowed: {', '.join(allowed_classes[sclass]['tiers'])}"
    return True, None

def ensure_reports_dir():
    os.makedirs("reports", exist_ok=True)

def plot_best_by_provider(best_rows_df, out_path):
    plt.figure()
    labels = best_rows_df['provider'].str.upper().tolist()
    vals = best_rows_df['total_hourly'].tolist()
    plt.bar(labels, vals)
    plt.title("Best Total/hr by Provider")
    plt.ylabel("USD per hour")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_breakdown(overall_row, out_path):
    plt.figure()
    parts = ["Compute", "Storage", "IOPS"]
    vals = [overall_row["price_per_hour"], overall_row["storage_hourly"], overall_row["iops_hourly"]]
    plt.bar(parts, vals)
    plt.title(f"Cost Breakdown: {overall_row['provider'].upper()} {overall_row['instance_type']}")
    plt.ylabel("USD per hour")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# ---------- main ----------
def main():
    print("\n=== Multi-Cloud Instance & Storage Cost Estimator (with IOPS + Charts) ===\n")

    provider_scope = (ask("Cloud provider scope (aws/azure/gcp/all)", "all") or "all").lower()
    region = ask("Region (us-east-1 / ap-south-1 / eastus / centralindia / us-central1 / asia-south1)", "us-east-1")

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

    # NEW: storage class selection (per cloud). If scope is 'all', let user say 'any' to avoid filtering.
    if provider_scope == "all":
        storage_class = (ask("Storage class (aws: ebs-gp3/ebs-io2, azure: managed-premium-ssd, gcp: pd-ssd/pd-balanced)", "any") or "any").lower()
        # Can't validate tiers strictly when multiple providers; still capture a tier string.
        tier = (ask("Tier (standard/premium/basic)", "standard") or "standard").lower()
    else:
        # single provider: strict validation
        default_class = list(ALLOWED.get(provider_scope, {"any":{}}).keys())[0] if provider_scope in ALLOWED else "any"
        storage_class = (ask(f"Storage class for {provider_scope} (choices: {', '.join(ALLOWED.get(provider_scope, {}).keys())} or 'any')", default_class) or default_class).lower()
        tier = (ask("Tier (standard/premium/basic)", "standard") or "standard").lower()
        ok, msg = validate_storage(provider_scope, storage_class, tier)
        if not ok:
            print(f"\n❌ {msg}\n")
            return

    throughput = ask_float("Throughput needed (MB/s)", 0.0)
    iops_needed = ask_int("Required storage IOPS (provisioned)", 0)

    apply_tax = (ask("Apply tax? (yes/no)", "no") or "no").lower() == "yes"
    inr_convert = (ask("Convert USD -> INR? (yes/no)", "no") or "no").lower() == "yes"

    # Load pricing
    files = []
    if provider_scope in ("aws","all"): files.append("data/aws_pricing.csv")
    if provider_scope in ("azure","all"): files.append("data/azure_pricing.csv")
    if provider_scope in ("gcp","all"): files.append("data/gcp_pricing.csv")
    pricing = load_pricing(files)

    # Filter by storage class if a specific class was requested
    if storage_class != "any" and "storage_type" in pricing.columns:
        pricing = pricing[pricing["storage_type"].str.lower() == storage_class]

    candidates = pick_candidates(pricing, vcpus_needed, mem_gb_needed)
    if candidates.empty:
        print("\n❌ No matching instance found. Add larger SKUs to CSVs or relax filters.\n")
        return

    # Cost math
    candidates["storage_hourly"] = candidates["storage_price_per_gb_month"].apply(lambda p: storage_hourly_cost(storage_gb, p))
    candidates["iops_hourly"] = candidates.apply(
        lambda r: iops_hourly_cost(iops_needed, r.get("iops_included", 0.0), r.get("iops_price_per_iops_month", 0.0)),
        axis=1
    )
    candidates["total_hourly"] = candidates["price_per_hour"] + candidates["storage_hourly"] + candidates["iops_hourly"]
    candidates["est_cost"] = candidates["total_hourly"] * run_hours

    # best per provider + overall
    best_rows = []
    for prov in ["aws","azure","gcp"]:
        sub = candidates[candidates["provider"]==prov]
        if not sub.empty:
            best_rows.append(sub.nsmallest(1, "total_hourly").iloc[0])
    best_df = pd.DataFrame(best_rows)
    overall = candidates.nsmallest(1, "total_hourly").iloc[0]

    usd_rate = 83 if inr_convert else 1
    tax = 1.18 if apply_tax else 1.00

    print("\n--- Input Summary ---")
    print(f"Scope: {provider_scope.upper()} | Region: {region} | CPU mode: {cpu_mode}")
    print(f"vCPUs={vcpus_needed}, RAM={mem_gb_needed:.1f} GB | Duration={run_hours:.2f} hrs")
    print(f"Storage={storage_gb:.0f} GB ({storage_mode}, class={storage_class}, tier={tier}, thr={throughput} MB/s), IOPS={iops_needed}")

    print("\n--- Best per provider ---")
    if not best_df.empty:
        for _, r in best_df.iterrows():
            run_cost = r["est_cost"] * usd_rate * tax
            print(f"{r['provider'].upper():5} | {r['instance_type']:20} | vCPU={int(r['vcpus'])} | Mem={r['mem_gb']:.1f} GB"
                  f" | Comp/hr=${r['price_per_hour']:.4f} | Stor/hr=${r['storage_hourly']:.4f} | IOPS/hr=${r['iops_hourly']:.4f}"
                  f" | Total/hr=${r['total_hourly']:.4f} | Run={run_cost:.2f} {'INR' if inr_convert else 'USD'}")

    total_cost = overall["est_cost"] * usd_rate * tax
    print("\n=== OVERALL RECOMMENDATION ===")
    print(f"{overall['provider'].upper()} | {overall['instance_type']} | vCPU={int(overall['vcpus'])} | Mem={overall['mem_gb']:.1f} GB")
    print(f"Compute/hr=${overall['price_per_hour']:.4f} | Storage/hr=${overall['storage_hourly']:.4f} | IOPS/hr=${overall['iops_hourly']:.4f}")
    print(f"Total/hr=${overall['total_hourly']:.4f} | Run={total_cost:.2f} {'INR' if inr_convert else 'USD'}\n")

    # Charts
    ensure_reports_dir()
    if not best_df.empty:
        plot_best_by_provider(best_df, "reports/best_by_provider.png")
    plot_breakdown(overall, "reports/breakdown_overall.png")
    print("Saved charts:")
    if not best_df.empty:
        print(" - reports/best_by_provider.png")
    print(" - reports/breakdown_overall.png")

if __name__ == "__main__":
    main()
