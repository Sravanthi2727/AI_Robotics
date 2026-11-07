#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_parallel_remediation.py
Orchestrates parallel execution of scenario remediation across 3 worker pools,
compiles deliverables, and verifies quality gate across all videos.
"""

from __future__ import print_function
import os
import sys
import subprocess
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.path.join(REPO_ROOT, "env", "bin", "python")

# Ensure required asset symlinks in /tmp/home
try:
    models_dir = os.path.join(REPO_ROOT, "models")
    maps_dir = os.path.join(REPO_ROOT, "maps")
    tmp_navrep = "/tmp/home/navrep"
    tmp_models = "/tmp/home/navrep/models"
    tmp_maps = "/tmp/home/maps"
    os.makedirs(tmp_navrep, exist_ok=True)
    os.makedirs("/tmp/matplotlib", exist_ok=True)
    if not os.path.exists(tmp_models):
        os.symlink(models_dir, tmp_models)
    if not os.path.exists(tmp_maps):
        os.symlink(maps_dir, tmp_maps)
except Exception:
    pass

SCENARIOS = [
    "navreptrain_train",
    "navreptrain_test",
    "irosasl1",
    "irosasl2",
    "irosasl3",
    "irosasl_office_j1",
    "irosasl_office_j2",
    "irosasl_office_j3",
    "irosunity_scene_map1",
    "irosunity_scene_map2",
    "irosunity_scene_map3",
]


def run_scenario(scenario):
    cmd = [
        PYTHON,
        "-u",
        os.path.join(REPO_ROOT, "scripts_custom", "remediate_and_record.py"),
        "--scenario", scenario
    ]
    env = dict(os.environ)
    env["HOME"] = "/tmp/home"
    env["MPLCONFIGDIR"] = "/tmp/matplotlib"
    log_path = os.path.join(REPO_ROOT, "results", "worker_{}.log".format(scenario))
    log_f = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
    return proc, log_f


def main():
    print("="*75)
    print("STARTING PARALLEL REMEDIATION ACROSS ALL 11 SCENARIOS (3 CONCURRENT WORKERS)")
    print("="*75)

    queue = list(SCENARIOS)
    running = {}
    max_workers = 3

    t0 = time.time()
    while queue or running:
        # Start new workers up to max_workers
        while queue and len(running) < max_workers:
            sc = queue.pop(0)
            print("[LAUNCH] Starting worker for scenario: {}".format(sc))
            proc, log_f = run_scenario(sc)
            running[sc] = (proc, log_f)

        # Check running workers
        done_scs = []
        for sc, (proc, log_f) in list(running.items()):
            ret = proc.poll()
            if ret is not None:
                log_f.close()
                if ret == 0:
                    print("[DONE] Scenario {} completed successfully!".format(sc))
                else:
                    print("[ERROR] Scenario {} failed with code {} (see results/worker_{}.log)".format(sc, ret, sc))
                done_scs.append(sc)

        for sc in done_scs:
            del running[sc]

        time.sleep(2)

    total_time = time.time() - t0
    print("\n" + "="*75)
    print("ALL 11 SCENARIOS COMPLETED in {:.1f}s ({:.1f} min)!".format(total_time, total_time / 60.0))
    print("="*75)

    # Compile deliverables
    print("\n>>> Compiling metrics_existing.csv and VIDEO_INDEX.md...")
    env = dict(os.environ)
    env["HOME"] = "/tmp/home"
    env["MPLCONFIGDIR"] = "/tmp/matplotlib"
    subprocess.check_call([
        PYTHON,
        os.path.join(REPO_ROOT, "scripts_custom", "remediate_and_record.py"),
        "--compile-only"
    ], env=env)

    # Run verification suite
    print("\n>>> Running automated verification suite...")
    ret = subprocess.call([
        PYTHON,
        os.path.join(REPO_ROOT, "scripts_custom", "verify_videos.py")
    ], env=env)

    return ret


if __name__ == "__main__":
    sys.exit(main())

