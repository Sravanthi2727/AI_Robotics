#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
log_repo_cross_test.py
Parses the output pickle files produced by navrep/scripts/cross_test_navreptrain_in_ianenv.py
for both VAE_LSTM VM and VAE_LSTM V_ONLY models (with --n 20).
Extracts real per-scenario success rates, episode counts, and step limits (read from repo code).
Saves the structured results into results/repo_script_cross_test.json.
"""

from __future__ import print_function
import os
import glob
import json
import pickle
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = "/tmp/navrep/eval/crosstest"
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
OUTPUT_JSON = os.path.join(RESULTS_DIR, "repo_script_cross_test.json")

# The 11 benchmark scenarios evaluated in Task 1 / Task 2
BENCHMARK_SCENARIOS = [
    ("navreptrain_train", "NavRepTrainEnv"),
    ("navreptrain_test", "NavRepTrainEnv"),
    ("irosasl1", "IANEnv"),
    ("irosasl2", "IANEnv"),
    ("irosasl3", "IANEnv"),
    ("irosasl_office_j1", "IANEnv"),
    ("irosasl_office_j2", "IANEnv"),
    ("irosasl_office_j3", "IANEnv"),
    ("irosunity_scene_map1", "IANEnv"),
    ("irosunity_scene_map2", "IANEnv"),
    ("irosunity_scene_map3", "IANEnv"),
]

# Primary benchmarked model file
BENCHMARK_MODEL_FILE = "navreptrainencodedenv_2020_09_23__09_48_22_PPO_VAE_LSTM_VM_V32M512_in_ianenv_x20.pckl"


def parse_pickle(path):
    with open(path, "rb") as f:
        df = pickle.load(f)
    scenarios = {}
    for scen, g in df.groupby("scenario"):
        succ = (sum([bool(x) for x in g["goal_reached"]]) / float(len(g))) * 100.0
        scenarios[scen] = {
            "episodes": int(len(g)),
            "success_pct": round(succ, 1),
            "steps_max": int(max(g["steps"])),
            "steps_mean": round(float(np.mean(g["steps"])), 1),
            "episodes_detail": [
                {
                    "steps": int(s),
                    "goal_reached": bool(gr),
                    "timed_out": bool(to),
                    "collisioned_out": bool(co)
                }
                for s, gr, to, co in zip(g["steps"], g["goal_reached"], g["timed_out"], g["collisioned_out"])
            ]
        }
    return {
        "file": os.path.basename(path),
        "total_episodes": int(len(df)),
        "overall_success_pct": round((sum([bool(x) for x in df["goal_reached"]]) / float(len(df))) * 100.0, 1),
        "scenarios": scenarios
    }


def main():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    pickle_files = sorted(glob.glob(os.path.join(EVAL_DIR, "*.pckl")))
    print("[LOG] Found {} crosstest pickle files in {}".format(len(pickle_files), EVAL_DIR))

    all_models_data = {}
    for pf in pickle_files:
        bname = os.path.basename(pf)
        print("  Parsing: {}".format(bname))
        all_models_data[bname] = parse_pickle(pf)

    # Extract results for the specific benchmarked model (2020_09_23 VM)
    bench_data = all_models_data.get(BENCHMARK_MODEL_FILE, {})
    bench_scenarios = bench_data.get("scenarios", {})

    # Code-defined step limits:
    # IANEnv: max_episode_length = 1000 in navrep/envs/encodedenv.py:317 and navrep/envs/ianenv.py:29
    # NavRepTrainEnv: time_limit = 35, time_step = 0.2 -> 175 steps in crowd_sim/envs/crowd_sim.py:28-29
    # Note: cross_test_navreptrain_in_ianenv only evaluates IANEnv scenarios.
    ianenv_code_max_steps = 1000

    benchmark_table = {}
    for scen, env_cls in BENCHMARK_SCENARIOS:
        if env_cls == "NavRepTrainEnv":
            # NavRepTrainEnv is not tested by cross_test_navreptrain_in_ianenv.py
            benchmark_table[scen] = {
                "repo_script_success_pct": "N/A",
                "repo_script_max_steps": "N/A",
                "repo_script_episodes": "N/A",
                "measured": False,
                "note": "NavRepTrainEnv scenarios are not supported or evaluated by cross_test_navreptrain_in_ianenv.py"
            }
        else:
            if scen in bench_scenarios:
                sdata = bench_scenarios[scen]
                benchmark_table[scen] = {
                    "repo_script_success_pct": sdata["success_pct"],
                    "repo_script_max_steps": ianenv_code_max_steps,
                    "repo_script_episodes": sdata["episodes"],
                    "measured": True,
                    "note": "Measured in 20-episode cross-test run of model 2020_09_23"
                }
            else:
                benchmark_table[scen] = {
                    "repo_script_success_pct": "N/A",
                    "repo_script_max_steps": "N/A",
                    "repo_script_episodes": "N/A",
                    "measured": False,
                    "note": "Scenario was not randomly sampled in this 20-episode cross-test rollout"
                }

    output_payload = {
        "metadata": {
            "script": "navrep/scripts/cross_test_navreptrain_in_ianenv.py",
            "flags": "--backend VAE_LSTM --encoding VM --n 20 and --backend VAE_LSTM --encoding V_ONLY --n 20",
            "episodes_per_run": 20,
            "ianenv_code_max_steps": ianenv_code_max_steps,
            "navreptrain_code_max_steps": 175,
            "benchmark_model_file": BENCHMARK_MODEL_FILE,
        },
        "per_scenario_benchmark_model": benchmark_table,
        "all_models_evaluated": all_models_data
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output_payload, f, indent=2)

    print("[LOG] Wrote results to {}".format(OUTPUT_JSON))
    print("\nBenchmark Model Cross-Test Results:")
    for scen, res in benchmark_table.items():
        print("  {:25s} | Success: {:6} | Max Steps: {:5} | Episodes: {}".format(
            scen, str(res["repo_script_success_pct"]), str(res["repo_script_max_steps"]), str(res["repo_script_episodes"])
        ))


if __name__ == "__main__":
    main()
