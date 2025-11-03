#!/usr/bin/env python3
"""
Interactive predictor (updated).

Prompts for:
 - metrics CSV (optional) OR manual CPU/memory inputs
 - mandatory storage tier and storage size (accepts GB or TB, e.g. "100", "100GB", "1TB", "0.5TB")

Outputs:
 - printed JSON recommendation (instance + storage + costs)
 - saves a summary PNG to reports/interactive_summary_<ts>.png and attempts to open it (Mac)
"""
import os, json, time, re
import pandas as pd, numpy as np, joblib
import matplotlib.pyplot as plt

PRICING_CSV = "data/aws_pricing.csv"
STORAGE_CSV = "data/storage_pricing.csv"
MODEL_PKL = "models/recommender_rf.pkl"
OUT_DIR = "reports"

def parse_storage_size_to_gb(s):
    """Accept strings like '100', '100GB', '1TB', '0.5TB' and return size in GB (float)."""
    if isinstance(s, (int, float)):
        return float(s)
    s = s.strip().upper()
    # accept plain number -> GB
    m = re.match(r'^([0-9]*\.?[0-9]+)\s*(TB|GB)?$', s)
    if not m:
        raise ValueError("Cannot parse storage size. Use format like '100', '100GB', '1TB', '0.5TB'.")
    val = float(m.group(1))
    unit = m.group(2)
    if unit == 'TB':
        return val * 1024.0
    else:  # GB or None
        return val

def choose_instance(vcpus_needed, mem_req_gb, pricing_df):
    candidates = pricing_df[(pricing_df['vCPU'] >= vcpus_needed) & (pricing_df['RAM_GB'] >= mem_req_gb)].copy()
    if candidates.empty:
        candidates = pricing_df.copy()
    return candidates.sort_values('cost_per_hour').iloc[0]

def render_summary_png(out_path, info):
    plt.figure(figsize=(10,5))
    plt.axis('off')
    txt = []
    txt.append("CLOUD COST & PERFORMANCE SUGGESTION")
    txt.append("")
    txt.append("Input / Observed")
    txt.append(f"  Peak CPU % : {info['cpu_peak_percent']:.1f}%")
    txt.append(f"  CPU count  : {info['cpu_count']}")
    txt.append(f"  Mem (95%)  : {info['mem_95_mb']:.0f} MB")
    txt.append(f"  Duration   : {info['duration_s']} s")
    txt.append("")
    txt.append("Storage")
    txt.append(f"  Tier       : {info['storage_tier']}")
    txt.append(f"  Size       : {info['storage_gb']:.2f} GB")
    txt.append(f"  Storage $/GB-month : ${info['storage_price_per_gb_month']:.4f}")
    txt.append("")
    txt.append("Recommendation")
    txt.append(f"  Instance   : {info['instance_type']}")
    txt.append(f"  vCPU / RAM : {info['vCPU']} vCPU / {info['RAM_GB']} GB")
    txt.append(f"  Cost/hr (instance) : ${info['cost_per_hour']:.4f}")
    txt.append(f"  Cost/hr (storage)  : ${info['storage_cost_per_hour']:.6f}")
    txt.append(f"  Total cost (run)   : ${info['est_cost_for_run_total']:.6f}")
    if info.get('note'):
        txt.append("")
        txt.append("Note: " + info['note'])
    plt.text(0.01, 0.99, "\n".join(txt), va='top', ha='left', fontsize=11, family='monospace')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path

def safe_float(s, name):
    try:
        v = float(s)
        return v
    except:
        raise ValueError(f"Invalid number for {name}: {s}")

def prompt_manual():
    while True:
        try:
            cpu_peak = safe_float(input("Enter peak CPU % (0-100): ").strip(), "cpu_peak")
            if not (0 <= cpu_peak <= 100):
                print("Please enter a percentage between 0 and 100.")
                continue
            mem95 = safe_float(input("Enter 95th percentile memory used (MB): ").strip(), "mem95_mb")
            cpu_count = int(safe_float(input("Enter number of logical CPUs observed (e.g. 4 or 8): ").strip(), "cpu_count"))
            duration_s = int(safe_float(input("Enter duration in seconds for cost estimate (default 60): ").strip() or "60", "duration_s"))
            return cpu_peak, mem95, cpu_count, duration_s
        except ValueError as e:
            print(e)
            print("Let's try again.\n")

def prompt_file():
    while True:
        path = input("Enter path to metrics CSV (example: data/run1_metrics.csv): ").strip()
        if not os.path.exists(path):
            print("File not found:", path)
            try_again = input("Try again? (y/n): ").strip().lower()
            if try_again != 'y':
                return None
            else:
                continue
        try:
            df = pd.read_csv(path)
            required = {'cpu_percent','mem_used_mb','cpu_count'}
            if not required.issubset(set(df.columns)):
                print("CSV missing required columns. Required:", required)
                return None
            cpu_peak = float(df['cpu_percent'].max())
            mem95 = float(df['mem_used_mb'].quantile(0.95))
            cpu_count = int(df['cpu_count'].median())
            duration_s = int(len(df))
            return cpu_peak, mem95, cpu_count, duration_s
        except Exception as e:
            print("Failed to read CSV:", e)
            return None

def prompt_storage(pricing_df):
    print("\nStorage pricing tiers available:")
    for tier in pricing_df['storage_tier'].tolist():
        print(" -", tier)
    while True:
        tier = input("Enter storage tier from above (default gp3): ").strip() or "gp3"
        if tier not in pricing_df['storage_tier'].values:
            print("Tier not found. Pick one from the list.")
            continue
        size_raw = input("Enter storage size (accepts GB or TB, e.g. 100, 100GB, 1TB): ").strip() or "100"
        try:
            size_gb = parse_storage_size_to_gb(size_raw)
            price_row = pricing_df[pricing_df['storage_tier'] == tier].iloc[0]
            price_per_gb_month = float(price_row['price_per_gb_month_usd'])
            return tier, size_gb, price_per_gb_month
        except ValueError as e:
            print(e)
            print("Try again.")

def main():
    print("=== Interactive Cloud Cost & Performance Predictor (with Storage GB/TB support) ===")
    print("Type 'file' to load a metrics CSV, or press Enter to input manual values.")
    choice = input("Your choice [file/manual]: ").strip().lower()
    vals = None
    if choice == 'file' or choice == 'f':
        vals = prompt_file()
        if vals is None:
            print("Falling back to manual input.")
            vals = prompt_manual()
    else:
        vals = prompt_manual()

    cpu_peak, mem95, cpu_count, duration_s = vals

    # load storage pricing CSV
    if not os.path.exists(STORAGE_CSV):
        print("Storage pricing CSV not found:", STORAGE_CSV)
        print("Please ensure file exists or edit data/storage_pricing.csv")
        return
    storage_df = pd.read_csv(STORAGE_CSV)

    storage_tier, storage_gb, storage_price_per_gb_month = prompt_storage(storage_df)

    # compute requirements (use margin)
    margin = 0.20
    peak_core_usage = (cpu_peak / 100.0) * cpu_count
    vcpus_needed = int(np.ceil(peak_core_usage * (1.0 + margin)))
    mem_req_gb = (mem95 / 1024.0) * (1.0 + margin)

    # read instance pricing
    if not os.path.exists(PRICING_CSV):
        print("Instance pricing CSV not found:", PRICING_CSV)
        return
    pricing = pd.read_csv(PRICING_CSV)

    chosen = None
    model_note = None
    if os.path.exists(MODEL_PKL):
        try:
            clf = joblib.load(MODEL_PKL)
            feat_df = pd.DataFrame([{
                'cpu_peak_percent': cpu_peak,
                'mem_95_mb': mem95,
                'cpu_workers': cpu_count,
                'duration_s': duration_s
            }])
            pred = clf.predict(feat_df)[0]
            row = pricing[pricing['instance_type'] == pred]
            if not row.empty:
                chosen_row = row.iloc[0]
                chosen = {
                    'instance_type': chosen_row['instance_type'],
                    'vCPU': int(chosen_row['vCPU']),
                    'RAM_GB': float(chosen_row['RAM_GB']),
                    'cost_per_hour': float(chosen_row['cost_per_hour'])
                }
                model_note = "Selected by trained model"
        except Exception as e:
            model_note = f"Model error: {e}"

    if chosen is None:
        chosen_row = choose_instance(vcpus_needed, mem_req_gb, pricing)
        chosen = {
            'instance_type': chosen_row['instance_type'],
            'vCPU': int(chosen_row['vCPU']),
            'RAM_GB': float(chosen_row['RAM_GB']),
            'cost_per_hour': float(chosen_row['cost_per_hour'])
        }
        model_note = model_note or "Selected by rule-based fallback"

    # storage cost: price_per_gb_month -> per hour = / (30*24)
    storage_cost_per_hour = (storage_price_per_gb_month * storage_gb) / (30.0 * 24.0)
    inst_cost_for_run = chosen['cost_per_hour'] * (duration_s / 3600.0)
    storage_cost_for_run = storage_cost_per_hour * (duration_s / 3600.0)
    total_run_cost = inst_cost_for_run + storage_cost_for_run

    info = {
        'cpu_peak_percent': cpu_peak,
        'mem_95_mb': mem95,
        'cpu_count': cpu_count,
        'duration_s': duration_s,
        'vcpus_needed': vcpus_needed,
        'mem_req_gb': round(mem_req_gb,3),
        'instance_type': chosen['instance_type'],
        'vCPU': chosen['vCPU'],
        'RAM_GB': chosen['RAM_GB'],
        'cost_per_hour': chosen['cost_per_hour'],
        'storage_tier': storage_tier,
        'storage_gb': storage_gb,
        'storage_price_per_gb_month': storage_price_per_gb_month,
        'storage_cost_per_hour': round(storage_cost_per_hour,8),
        'est_cost_for_run_instance': round(inst_cost_for_run,8),
        'est_cost_for_run_storage': round(storage_cost_for_run,8),
        'est_cost_for_run_total': round(total_run_cost,8),
        'note': model_note
    }

    print("\nRecommendation:")
    print(json.dumps(info, indent=2))

    ts = int(time.time())
    out_png = os.path.join(OUT_DIR, f"interactive_summary_{ts}.png")
    render_summary_png(out_png, info)
    print("Saved summary image to:", out_png)
    try:
        os.system(f'open "{out_png}"')
    except:
        pass

if __name__ == '__main__':
    main()
