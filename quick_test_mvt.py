#!/usr/bin/env python3
"""
run_starlet_build_test.py

Runs the equivalent of:

    starlet build --input /Users/tarlan/Downloads/NE_countries.parquet --outdir datasets/NE_countries --zoom 20

but with live log streaming and simple stage timing.
"""

from __future__ import annotations

import subprocess
import sys
import time

INPUT = "/Users/tarlan/Downloads/NE_countries.parquet"
OUTDIR = "datasets/NE_countries"
ZOOM = "20"

cmd = [
    "starlet",
    "build",
    "--input", INPUT,
    "--outdir", OUTDIR,
    "--zoom", ZOOM,
]

print("Running:")
print(" ".join(cmd))
print()

start = time.time()
proc = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)

saw_mvt_start = False
saw_nonempty = False

try:
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")

        if "Starting MVT generation pipeline" in line:
            saw_mvt_start = True
            print(f"\n[marker] MVT stage started at {time.time() - start:.2f}s\n")

        if "Computing nonempty tiles" in line:
            saw_nonempty = True
            print(f"\n[marker] Reached old nonempty-tile scan at {time.time() - start:.2f}s\n")
            print("[marker] If it stays here, you are still running the old slow path.\n")

    rc = proc.wait()
    elapsed = time.time() - start

    print()
    print(f"Exit code: {rc}")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Saw MVT start: {saw_mvt_start}")
    print(f"Saw 'Computing nonempty tiles': {saw_nonempty}")

    sys.exit(rc)

except KeyboardInterrupt:
    print("\nInterrupted. Terminating child process...")
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    raise