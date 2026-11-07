#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rerecord_fixes.py
Re-records the 5 success scenarios (04, 05, 07, 10, 11) with goal caching before auto-reset
so final frame HUD and sidecar final distance is <= 1.0m.
Also re-records 03_irosasl1_failure with seed 115 (different from seed 23).
Then recompiles metrics_existing.csv and VIDEO_INDEX.md.
"""

from __future__ import print_function
import os
import sys

# Ensure importlib_resources for imageio on Python 3.6
import importlib
try:
    import importlib_resources
    sys.modules['importlib.resources'] = importlib_resources
    importlib.resources = importlib_resources
except ImportError:
    pass

from scripts_custom.remediate_and_record import (
    load_policy_and_encoder,
    record_verified_video,
    compile_all_deliverables
)

RE_RECORD_TASKS = [
    # (prefix, scenario_name, role, seed, max_steps)
    ("03", "irosasl1", "failure", 115, 500),
    ("04", "irosasl2", "success", 26, 500),
    ("05", "irosasl3", "success", 56, 500),
    ("07", "irosasl_office_j2", "success", 47, 500),
    ("10", "irosunity_scene_map2", "success", 58, 500),
    ("11", "irosunity_scene_map3", "success", 34, 500),
]


def main():
    policy, encoder, model_filename = load_policy_and_encoder()

    for prefix, scenario, role, seed, max_steps in RE_RECORD_TASKS:
        print("\n" + "="*70)
        print(">>> RE-RECORDING: {} {} (role={}, seed={})".format(prefix, scenario, role, seed))
        print("="*70)
        sidecar_data, outc, steps, dur = record_verified_video(
            prefix=prefix,
            scenario_name=scenario,
            role=role,
            seed=seed,
            policy=policy,
            encoder=encoder,
            max_steps=max_steps
        )
        print("RESULT: {} -> outcome={}, steps={}, final_dist={}m, dur={}s".format(
            sidecar_data["file"], outc, steps, sidecar_data["final_dist"], dur
        ))
        if role == "success":
            assert outc == "GOAL_REACHED", "Failed to reach goal for {}".format(scenario)
            assert sidecar_data["final_dist"] <= 1.0, "final_dist {} > 1.0m for {}".format(
                sidecar_data["final_dist"], scenario
            )
        elif role == "failure":
            assert outc in ["COLLISION", "TIMEOUT"], "Unexpected outcome {} for {}".format(outc, scenario)
            assert steps >= 30, "Steps {} < 30".format(steps)
            assert dur >= 7.9, "Duration {} < 8.0s".format(dur)

    print("\n" + "="*70)
    print(">>> COMPILING DELIVERABLES")
    print("="*70)
    compile_all_deliverables(model_filename)
    print("Deliverables recompiled successfully!")


if __name__ == "__main__":
    main()
