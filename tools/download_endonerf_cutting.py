"""Resumable curl downloader for the EndoNeRF 'cutting_tissues_twice' clip.

Reads tools/endonerf_cutting/fileids.tsv (id<TAB>relpath), downloads only the files the
binocular loader needs (images/, depth/, masks/, poses_bounds.npy) into the EndoGaussian
layout data/endonerf/cutting/. Uses the plain uc?export=download endpoint via curl (less
rate-limited than gdown), validates PNG/NPY magic bytes, and skips anything already valid so
it can be re-run to resume. Login-node only (compute nodes have no internet).
"""
import os, subprocess, time

DST = "data/endonerf/cutting"
KEEP = ("images/", "depth/", "masks/")   # plus poses_bounds.npy
jobs = []
for line in open("tools/endonerf_cutting/fileids.tsv"):
    fid, rel = line.rstrip("\n").split("\t")
    if rel == "poses_bounds.npy" or rel.startswith(KEEP):
        jobs.append((fid, os.path.join(DST, rel)))

for _, p in jobs:
    os.makedirs(os.path.dirname(p), exist_ok=True)

def valid(p):
    if not os.path.exists(p) or os.path.getsize(p) < 100:
        return False
    m = open(p, "rb").read(8)
    if p.endswith(".png"):
        return m.startswith(b"\x89PNG")
    if p.endswith(".npy"):
        return m.startswith(b"\x93NUMPY")
    return True

total = len(jobs)
done = sum(valid(p) for _, p in jobs)
print(f"total {total}; already valid {done}", flush=True)

for fid, path in jobs:
    if valid(path):
        continue
    for attempt in range(6):
        try:
            subprocess.run(
                ["curl", "-sSL", f"https://drive.google.com/uc?export=download&id={fid}", "-o", path],
                timeout=60)
        except Exception:
            pass
        if valid(path):
            break
        if os.path.exists(path):
            try: os.remove(path)
            except OSError: pass
        time.sleep(4 * (attempt + 1))
    if valid(path):
        done += 1
        if done % 25 == 0:
            print(f"{done}/{total} ok", flush=True)
        time.sleep(0.3)
    else:
        print(f"FAILED {path}", flush=True)

print(f"FINISHED {sum(valid(p) for _, p in jobs)}/{total} valid", flush=True)
