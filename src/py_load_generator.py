#!/usr/bin/env python3
import argparse, time, multiprocessing, os, sys

parser = argparse.ArgumentParser()
parser.add_argument('--cpu', type=int, default=1)
parser.add_argument('--mem_mb', type=int, default=128)
parser.add_argument('--disk_mb', type=int, default=0)
parser.add_argument('--duration', type=int, default=30)
parser.add_argument('--disk_file', default='tmp_io.bin')
args = parser.parse_args()

def cpu_load(flag):
    while not flag.value:
        x = 0
        for _ in range(10000):
            x += 1

def mem_alloc(mb):
    a = bytearray(mb * 1024 * 1024)
    for i in range(0, len(a), 4096):
        a[i] = 1
    return a

def disk_write(path, mb, duration):
    with open(path, 'wb') as f:
        block = b'0' * 1024 * 1024
        start = time.time()
        written = 0
        while time.time() - start < duration and written < mb:
            f.write(block)
            f.flush()
            try:
                os.fsync(f.fileno())
            except:
                pass
            written += 1

if __name__ == '__main__':
    stop = multiprocessing.Value('b', False)
    procs = []
    for _ in range(max(1, args.cpu)):
        p = multiprocessing.Process(target=cpu_load, args=(stop,))
        p.start()
        procs.append(p)
    mem_obj = None
    if args.mem_mb > 0:
        mem_obj = mem_alloc(args.mem_mb)
    if args.disk_mb > 0:
        dw = multiprocessing.Process(target=disk_write, args=(args.disk_file, args.disk_mb, args.duration))
        dw.start()
        procs.append(dw)
    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        pass
    finally:
        stop.value = True
        for p in procs:
            try:
                p.terminate()
            except:
                pass
        print("Load finished")
