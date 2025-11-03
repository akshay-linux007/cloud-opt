#!/usr/bin/env python3
import subprocess, time, argparse, os, sys
import pandas as pd, math

parser = argparse.ArgumentParser()
parser.add_argument('--out', default='data/dataset.csv')
parser.add_argument('--duration', type=int, default=20)   # safe short runs
parser.add_argument('--repeat', type=int, default=1)
args = parser.parse_args()

# parameter grid (keep safe for Mac)
cpu_vals = [1, 2, 4]         # number of CPU workers in generator
mem_vals = [64, 128, 256, 512]  # MB allocated
disk_vals = [0]              # avoid heavy disk writes on laptop

rows = []
for r in range(args.repeat):
    for cpu in cpu_vals:
        for mem in mem_vals:
            for disk in disk_vals:
                run_name = f"run_cpu{cpu}_mem{mem}_r{r}"
                metrics_path = os.path.abspath(f"data/{run_name}_metrics.csv")
                # run orchestrator
                cmd = [sys.executable, os.path.join('src','orchestrator.py'),
                       '--out_prefix', f"data/{run_name}", '--duration', str(args.duration),
                       '--cpu', str(cpu), '--mem_mb', str(mem), '--disk_mb', str(disk)]
                print("Running:", ' '.join(cmd))
                subprocess.check_call(cmd)
                # read metrics and compute features
                df = pd.read_csv(metrics_path)
                cpu_peak = df['cpu_percent'].max()
                cpu_count = int(df['cpu_count'].median())
                mem_95 = df['mem_used_mb'].quantile(0.95)
                # estimate vcpus required (same logic as cost_mapper)
                peak_core_usage = (cpu_peak / 100.0) * cpu_count
                vcpus_needed = math.ceil(peak_core_usage * 1.2)
                mem_req_gb = (mem_95 / 1024.0) * 1.2
                # find recommended instance using pricing file
                pricing = pd.read_csv('data/aws_pricing.csv')
                candidates = pricing[(pricing['vCPU'] >= vcpus_needed) & (pricing['RAM_GB'] >= mem_req_gb)].copy()
                if candidates.empty:
                    candidates = pricing.copy()
                candidate = candidates.sort_values('cost_per_hour').iloc[0]
                est_cost = candidate['cost_per_hour'] * (args.duration/3600.0)
                rows.append({
                    'run': run_name,
                    'cpu_workers': cpu,
                    'mem_alloc_mb': mem,
                    'duration_s': args.duration,
                    'cpu_peak_percent': cpu_peak,
                    'cpu_count': cpu_count,
                    'vcpus_needed': vcpus_needed,
                    'mem_95_mb': round(mem_95,2),
                    'mem_req_gb': round(mem_req_gb,3),
                    'recommended_instance': candidate['instance_type'],
                    'cost_per_hour': candidate['cost_per_hour'],
                    'est_cost_for_run': round(est_cost,8)
                })
                time.sleep(0.5)

# write dataset
out_df = pd.DataFrame(rows)
out_df.to_csv(args.out, index=False)
print("Dataset saved to", args.out)
