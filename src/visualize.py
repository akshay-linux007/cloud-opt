#!/usr/bin/env python3
import pandas as pd, matplotlib.pyplot as plt, argparse, os

parser = argparse.ArgumentParser()
parser.add_argument('--metrics', required=True)
parser.add_argument('--out_prefix', default='reports/plot')
args = parser.parse_args()

df = pd.read_csv(args.metrics, parse_dates=['timestamp'])

# Plot CPU %
plt.figure(figsize=(8,3))
plt.plot(df['timestamp'], df['cpu_percent'], marker='o', label='CPU %')
plt.xlabel('Time')
plt.ylabel('CPU %')
plt.title('CPU usage over time')
plt.xticks(rotation=30)
plt.legend()
plt.tight_layout()
plt.savefig(f"{args.out_prefix}_cpu.png")
plt.close()

# Plot Memory (GB)
plt.figure(figsize=(8,3))
plt.plot(df['timestamp'], df['mem_used_mb']/1024.0, marker='o', color='orange', label='Memory (GB)')
plt.xlabel('Time')
plt.ylabel('Memory (GB)')
plt.title('Memory usage over time')
plt.xticks(rotation=30)
plt.legend()
plt.tight_layout()
plt.savefig(f"{args.out_prefix}_mem.png")
plt.close()

print("Saved plots:", f"{args.out_prefix}_cpu.png and {args.out_prefix}_mem.png")
