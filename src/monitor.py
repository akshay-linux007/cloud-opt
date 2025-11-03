#!/usr/bin/env python3
import psutil, csv, time, argparse, datetime, os

parser = argparse.ArgumentParser()
parser.add_argument('--out', required=True)
parser.add_argument('--interval', type=float, default=1.0)
parser.add_argument('--duration', type=float, default=60.0)
args = parser.parse_args()

out = args.out
os.makedirs(os.path.dirname(out) or '.', exist_ok=True)

fields = ['timestamp','cpu_percent','cpu_count','mem_total_mb','mem_used_mb','mem_percent','disk_read_mb_s','disk_write_mb_s']
with open(out, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(fields)
    start = time.time()
    prev_disk = psutil.disk_io_counters()
    while time.time() - start < args.duration:
        t = datetime.datetime.utcnow().isoformat()
        cpu = psutil.cpu_percent(interval=None)
        cpu_count = psutil.cpu_count(logical=True)
        mem = psutil.virtual_memory()
        curr_disk = psutil.disk_io_counters()
        read_bytes = (curr_disk.read_bytes - prev_disk.read_bytes) / max(args.interval,1e-6)
        write_bytes = (curr_disk.write_bytes - prev_disk.write_bytes) / max(args.interval,1e-6)
        prev_disk = curr_disk
        row = [t, cpu, cpu_count, mem.total/1024/1024, mem.used/1024/1024, mem.percent, read_bytes/1024/1024, write_bytes/1024/1024]
        writer.writerow(row)
        f.flush()
        time.sleep(args.interval)
print("Monitoring finished. Saved:", out)
