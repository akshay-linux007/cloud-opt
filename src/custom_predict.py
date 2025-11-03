#!/usr/bin/env python3
"""
custom_predict.py

Usage examples:
  # 1) Use an existing metrics CSV (produces prediction + cost + PNG summary)
  python src/custom_predict.py --metrics data/run1_metrics.csv --out reports/custom_summary_run1.png

  # 2) Supply raw numbers (single prediction)
  python src/custom_predict.py --cpu_peak 35.0 --mem95_mb 8000 --cpu_count 8 --duration_s 60 --out reports/custom_summary_manual.png

Outputs:
 - printed text recommendation + cost
 - image summary saved to --out (PNG)
 - optionally prints JSON if --json flag used
"""
import os, argparse, json
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# helper to read pricing and pick best candidate
def choose_instance(vcpus_needed, mem_req_gb, pricing_df):
    candidates = pricing_df[(pricing_df['vCPU'] >= vcpus_needed) & (pricing_df['RAM_GB'] >= mem_req_gb)].copy()
    if candidates.empty:
        # if none match, return the cheapest (we'll still show note)
        candidates = pricing_df.copy()
    candidate = candidates.sort_values('cost_per_hour').iloc[0]
    return candidate

def render_summary_png(out_path, info):
    # small card style image
    plt.figure(figsize=(8,4))
    plt.axis('off')
    txt = []
    txt.append(f"Cloud Cost & Performance Suggestion")
    txt.append("")
    txt.append(f"Observed / Input")
    txt.append(f"  Peak CPU % : {info['cpu_peak_percent']:.1f}%")
    txt.append(f"  CPU count  : {info['cpu_count']}")
    txt.append(f"  Mem (95%)  : {info['mem_95_mb']:.0f} MB ({info['mem_req_gb']:.2f} GB w/ margin)")
    txt.append(f"  Duration   : {info['duration_s']} s")
    txt.append("")
    txt.append(f"Recommendation")
    txt.append(f"  Instance   : {info['instance_type']}")
    txt.append(f"  vCPU / RAM : {info['vCPU']} vCPU / {info['RAM_GB']} GB")
    txt.append(f"  Cost/hr    : ${info['cost_per_hour']:.4f}")
    txt.append(f"  Cost (run) : ${info['est_cost_for_run']:.6f}")
    txt.append("")
    if info.get('note'):
        txt.append("Note: " + info['note'])
    # draw text box
    plt.text(0.01, 0.99, "\n".join(txt), va='top', ha='left', fontsize=12, family='monospace')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', help="path to metrics CSV (optional). If provided, metrics are read from this file and --cpu_peak/--mem95 etc are ignored.")
    parser.add_argument('--cpu_peak', type=float, help="peak cpu % (0-100)")
    parser.add_argument('--mem95_mb', type=float, help="95th percentile memory used (MB)")
    parser.add_argument('--cpu_count', type=int, help="number of logical CPUs observed (e.g. 8)")
    parser.add_argument('--duration_s', type=int, default=60, help="duration in seconds used for cost estimation")
    parser.add_argument('--margin', type=float, default=0.2, help="safety margin fraction to add to vcpu and memory")
    parser.add_argument('--pricing', default='data/aws_pricing.csv', help="pricing CSV path")
    parser.add_argument('--model', default='models/recommender_rf.pkl', help="trained model (optional). If missing, script uses rule-based selection.")
    parser.add_argument('--out', default='reports/custom_summary.png', help='output PNG path (summary card)')
    parser.add_argument('--json', action='store_true', help='also print JSON result')
    args = parser.parse_args()

    # read pricing
    pricing = pd.read_csv(args.pricing)

    # compute features either from CSV or from manual inputs
    if args.metrics:
        if not os.path.exists(args.metrics):
            raise SystemExit("Metrics CSV not found: " + args.metrics)
        df = pd.read_csv(args.metrics)
        cpu_peak = float(df['cpu_percent'].max())
        mem95 = float(df['mem_used_mb'].quantile(0.95))
        cpu_count = int(df['cpu_count'].median())
        duration = args.duration_s or len(df)
    else:
        if args.cpu_peak is None or args.mem95_mb is None or args.cpu_count is None:
            raise SystemExit("When --metrics is not provided, supply --cpu_peak --mem95_mb --cpu_count")
        cpu_peak = float(args.cpu_peak)
        mem95 = float(args.mem95_mb)
        cpu_count = int(args.cpu_count)
        duration = args.duration_s

    # estimate cores/vcpus & mem with margin (same logic as cost_mapper)
    peak_core_usage = (cpu_peak / 100.0) * cpu_count
    vcpus_needed = int(np.ceil(peak_core_usage * (1.0 + args.margin)))
    mem_req_gb = (mem95 / 1024.0) * (1.0 + args.margin)

    # choose instance: prefer model prediction if available
    chosen = None
    model_note = None
    if os.path.exists(args.model):
        try:
            clf = joblib.load(args.model)
            feat_df = pd.DataFrame([{
                'cpu_peak_percent': cpu_peak,
                'mem_95_mb': mem95,
                'cpu_workers': cpu_count,
                'duration_s': duration
            }])
            pred = clf.predict(feat_df)[0]
            # find in pricing
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
            model_note = f"Model load/predict failed: {e}"

    # fallback: rule-based
    if chosen is None:
        chosen_row = choose_instance(vcpus_needed, mem_req_gb, pricing)
        chosen = {
            'instance_type': chosen_row['instance_type'],
            'vCPU': int(chosen_row['vCPU']),
            'RAM_GB': float(chosen_row['RAM_GB']),
            'cost_per_hour': float(chosen_row['cost_per_hour'])
        }
        model_note = model_note or "Selected by rule-based fallback"

    est_cost = chosen['cost_per_hour'] * (duration / 3600.0)

    info = {
        'cpu_peak_percent': cpu_peak,
        'mem_95_mb': mem95,
        'cpu_count': cpu_count,
        'duration_s': duration,
        'vcpus_needed': vcpus_needed,
        'mem_req_gb': round(mem_req_gb,3),
        'instance_type': chosen['instance_type'],
        'vCPU': chosen['vCPU'],
        'RAM_GB': chosen['RAM_GB'],
        'cost_per_hour': chosen['cost_per_hour'],
        'est_cost_for_run': round(est_cost,8),
        'note': model_note
    }

    # print results
    print("Recommendation:")
    print(json.dumps(info, indent=2))
    if args.json:
        print(json.dumps(info))

    # render PNG summary
    out_png = render_summary_png(args.out, info)
    print("Saved summary image to:", out_png)

if __name__ == '__main__':
    main()
