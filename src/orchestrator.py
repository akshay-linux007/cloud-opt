#!/usr/bin/env python3
import subprocess, time, argparse, os, sys

parser = argparse.ArgumentParser()
parser.add_argument('--out_prefix', default='data/run1')
parser.add_argument('--duration', type=int, default=30)
parser.add_argument('--cpu', type=int, default=1)
parser.add_argument('--mem_mb', type=int, default=128)
parser.add_argument('--disk_mb', type=int, default=0)
args = parser.parse_args()

metrics_path = os.path.abspath(args.out_prefix + '_metrics.csv')
os.makedirs(os.path.dirname(metrics_path) or '.', exist_ok=True)

monitor_py = os.path.join(os.path.dirname(__file__), 'monitor.py')
load_py = os.path.join(os.path.dirname(__file__), 'py_load_generator.py')

monitor_cmd = [sys.executable, monitor_py, '--out', metrics_path, '--interval', '1', '--duration', str(args.duration)]
load_cmd = [sys.executable, load_py, '--cpu', str(args.cpu), '--mem_mb', str(args.mem_mb), '--disk_mb', str(args.disk_mb), '--duration', str(args.duration)]

print("Starting monitor:", ' '.join(monitor_cmd))
mproc = subprocess.Popen(monitor_cmd)
time.sleep(0.6)
print("Starting load generator:", ' '.join(load_cmd))
lproc = subprocess.Popen(load_cmd)
lproc.wait()
time.sleep(0.5)
try:
    mproc.terminate()
except:
    pass
print("Orchestration completed. Metrics saved to:", metrics_path)
