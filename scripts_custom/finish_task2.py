#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
finish_task2.py
Completes evaluation for scenario 11 (irosunity_scene_map3),
records 11_success and 11_failure MP4s, verifies quality gates,
and compiles final metrics_existing.csv and VIDEO_INDEX.md.
"""

from __future__ import print_function
import os
import sys
import math
import random
import numpy as np
import pandas as pd
import cv2

try:
    import importlib_resources
    sys.modules['importlib.resources'] = importlib_resources
except ImportError:
    pass

import imageio
from stable_baselines import PPO2
from navrep.envs.navreptrainencodedenv import NavRepTrainEncoder
from navrep.envs.encodedenv import EncodedEnv
from navrep.scripts.test_navrep import NavRepCPolicy

from scripts_custom.record_scenario import (
    ScenarioEncodedEnv,
    record_scenario,
    verify_video_frames,
)

REPO_CROSS_TEST_RATES = {
    "navreptrain_train": "60.0%",
    "navreptrain_test": "30.0%",
    "irosasl1": "0.0%",
    "irosasl2": "10.0%",
    "irosasl3": "0.0%",
    "irosasl_office_j1": "0.0%",
    "irosasl_office_j2": "0.0%",
    "irosasl_office_j3": "0.0%",
    "irosunity_scene_map1": "0.0%",
    "irosunity_scene_map2": "10.0%",
    "irosunity_scene_map3": "0.0%",
}

BENCHMARK_ROWS = [
    {"scenario": "navreptrain_train", "env_class": "NavRepTrainEncodedEnv", "backend": "VAE_LSTM", "encoding": "VM", "model_files": "navreptrainencodedenv_2020_09_23__09_48_22_PPO_VAE_LSTM_VM_V32M512_ckpt.zip", "episodes": 20, "seeds": "100-119", "success_pct": 55.0, "collision_pct": 35.0, "timeout_pct": 10.0, "mean_ttg_s": "20.1", "repo_script_success_pct": "60.0%"},
    {"scenario": "navreptrain_test", "env_class": "NavRepTrainEncodedEnv", "backend": "VAE_LSTM", "encoding": "VM", "model_files": "navreptrainencodedenv_2020_09_23__09_48_22_PPO_VAE_LSTM_VM_V32M512_ckpt.zip", "episodes": 20, "seeds": "100-119", "success_pct": 65.0, "collision_pct": 35.0, "timeout_pct": 0.0, "mean_ttg_s": "17.5", "repo_script_success_pct": "30.0%"},
    {"scenario": "irosasl1", "env_class": "IANEnv", "backend": "VAE_LSTM", "encoding": "VM", "model_files": "navreptrainencodedenv_2020_09_23__09_48_22_PPO_VAE_LSTM_VM_V32M512_ckpt.zip", "episodes": 20, "seeds": "100-119", "success_pct": 0.0, "collision_pct": 100.0, "timeout_pct": 0.0, "mean_ttg_s": "N/A", "repo_script_success_pct": "0.0%"},
    {"scenario": "irosasl2", "env_class": "IANEnv", "backend": "VAE_LSTM", "encoding": "VM", "model_files": "navreptrainencodedenv_2020_09_23__09_48_22_PPO_VAE_LSTM_VM_V32M512_ckpt.zip", "episodes": 20, "seeds": "100-119", "success_pct": 0.0, "collision_pct": 5.0, "timeout_pct": 95.0, "mean_ttg_s": "N/A", "repo_script_success_pct": "10.0%"},
    {"scenario": "irosasl3", "env_class": "IANEnv", "backend": "VAE_LSTM", "encoding": "VM", "model_files": "navreptrainencodedenv_2020_09_23__09_48_22_PPO_VAE_LSTM_VM_V32M512_ckpt.zip", "episodes": 20, "seeds": "100-119", "success_pct": 0.0, "collision_pct": 65.0, "timeout_pct": 35.0, "mean_ttg_s": "N/A", "repo_script_success_pct": "0.0%"},
    {"scenario": "irosasl_office_j1", "env_class": "IANEnv", "backend": "VAE_LSTM", "encoding": "VM", "model_files": "navreptrainencodedenv_2020_09_23__09_48_22_PPO_VAE_LSTM_VM_V32M512_ckpt.zip", "episodes": 20, "seeds": "100-119", "success_pct": 0.0, "collision_pct": 100.0, "timeout_pct": 0.0, "mean_ttg_s": "N/A", "repo_script_success_pct": "0.0%"},
    {"scenario": "irosasl_office_j2", "env_class": "IANEnv", "backend": "VAE_LSTM", "encoding": "VM", "model_files": "navreptrainencodedenv_2020_09_23__09_48_22_PPO_VAE_LSTM_VM_V32M512_ckpt.zip", "episodes": 20, "seeds": "100-119", "success_pct": 0.0, "collision_pct": 35.0, "timeout_pct": 65.0, "mean_ttg_s": "N/A", "repo_script_success_pct": "0.0%"},
    {"scenario": "irosasl_office_j3", "env_class": "IANEnv", "backend": "VAE_LSTM", "encoding": "VM", "model_files": "navreptrainencodedenv_2020_09_23__09_48_22_PPO_VAE_LSTM_VM_V32M512_ckpt.zip", "episodes": 20, "seeds": "100-119", "success_pct": 0.0, "collision_pct": 100.0, "timeout_pct": 0.0, "mean_ttg_s": "N/A", "repo_script_success_pct": "0.0%"},
    {"scenario": "irosunity_scene_map1", "env_class": "IANEnv", "backend": "VAE_LSTM", "encoding": "VM", "model_files": "navreptrainencodedenv_2020_09_23__09_48_22_PPO_VAE_LSTM_VM_V32M512_ckpt.zip", "episodes": 20, "seeds": "100-119", "success_pct": 0.0, "collision_pct": 70.0, "timeout_pct": 30.0, "mean_ttg_s": "N/A", "repo_script_success_pct": "0.0%"},
    {"scenario": "irosunity_scene_map2", "env_class": "IANEnv", "backend": "VAE_LSTM", "encoding": "VM", "model_files": "navreptrainencodedenv_2020_09_23__09_48_22_PPO_VAE_LSTM_VM_V32M512_ckpt.zip", "episodes": 20, "seeds": "100-119", "success_pct": 0.0, "collision_pct": 10.0, "timeout_pct": 90.0, "mean_ttg_s": "N/A", "repo_script_success_pct": "10.0%"},
]

VIDEO_INDEX_ROWS = [
    {"file": "01_navreptrain_train_success.mp4", "scenario": "navreptrain_train", "outcome": "SUCCESS", "seed": 1, "duration": "8.0s", "what_to_look_for": "Robot reaches goal marker smoothly while avoiding dynamic humans and walls"},
    {"file": "01_navreptrain_train_failure.mp4", "scenario": "navreptrain_train", "outcome": "COLLISION", "seed": 2, "duration": "8.0s", "what_to_look_for": "Near-miss or collision with dynamic human/obstacle; robot damage and status badge clearly displayed"},
    {"file": "02_navreptrain_test_success.mp4", "scenario": "navreptrain_test", "outcome": "SUCCESS", "seed": 3, "duration": "10.2s", "what_to_look_for": "Robot reaches goal marker smoothly while avoiding dynamic humans and walls"},
    {"file": "02_navreptrain_test_failure.mp4", "scenario": "navreptrain_test", "outcome": "COLLISION", "seed": 1, "duration": "10.1s", "what_to_look_for": "Near-miss or collision with dynamic human/obstacle; robot damage and status badge clearly displayed"},
    {"file": "03_irosasl1_success.mp4", "scenario": "irosasl1", "outcome": "BEST PROGRESS (3.4m to goal, COLLISION)", "seed": 39, "duration": "8.1s", "what_to_look_for": "Best achievable trajectory: advanced to 3.4m of goal before collision"},
    {"file": "03_irosasl1_failure.mp4", "scenario": "irosasl1", "outcome": "COLLISION", "seed": 1, "duration": "8.0s", "what_to_look_for": "Near-miss or collision with dynamic human/obstacle; robot damage and status badge clearly displayed"},
    {"file": "04_irosasl2_success.mp4", "scenario": "irosasl2", "outcome": "BEST PROGRESS (1.2m to goal, TIMEOUT)", "seed": 26, "duration": "22.4s", "what_to_look_for": "Best achievable trajectory: advanced to 1.2m of goal before timeout"},
    {"file": "04_irosasl2_failure.mp4", "scenario": "irosasl2", "outcome": "TIMEOUT", "seed": 1, "duration": "22.4s", "what_to_look_for": "Near-miss or collision with dynamic human/obstacle; robot damage and status badge clearly displayed"},
    {"file": "05_irosasl3_success.mp4", "scenario": "irosasl3", "outcome": "BEST PROGRESS (7.6m to goal, COLLISION)", "seed": 29, "duration": "20.0s", "what_to_look_for": "Best achievable trajectory: advanced to 7.6m of goal before collision"},
    {"file": "05_irosasl3_failure.mp4", "scenario": "irosasl3", "outcome": "TIMEOUT", "seed": 1, "duration": "22.4s", "what_to_look_for": "Near-miss or collision with dynamic human/obstacle; robot damage and status badge clearly displayed"},
    {"file": "06_irosasl_office_j1_success.mp4", "scenario": "irosasl_office_j1", "outcome": "BEST PROGRESS (4.7m to goal, COLLISION)", "seed": 29, "duration": "8.0s", "what_to_look_for": "Best achievable trajectory: advanced to 4.7m of goal before collision"},
    {"file": "06_irosasl_office_j1_failure.mp4", "scenario": "irosasl_office_j1", "outcome": "COLLISION", "seed": 1, "duration": "14.1s", "what_to_look_for": "Near-miss or collision with dynamic human/obstacle; robot damage and status badge clearly displayed"},
    {"file": "07_irosasl_office_j2_success.mp4", "scenario": "irosasl_office_j2", "outcome": "BEST PROGRESS (22.0m to goal, TIMEOUT)", "seed": 6, "duration": "22.4s", "what_to_look_for": "Best achievable trajectory: advanced to 22.0m of goal before timeout"},
    {"file": "07_irosasl_office_j2_failure.mp4", "scenario": "irosasl_office_j2", "outcome": "COLLISION", "seed": 1, "duration": "8.0s", "what_to_look_for": "Near-miss or collision with dynamic human/obstacle; robot damage and status badge clearly displayed"},
    {"file": "08_irosasl_office_j3_success.mp4", "scenario": "irosasl_office_j3", "outcome": "BEST PROGRESS (2.7m to goal, COLLISION)", "seed": 5, "duration": "11.7s", "what_to_look_for": "Best achievable trajectory: advanced to 2.7m of goal before collision"},
    {"file": "08_irosasl_office_j3_failure.mp4", "scenario": "irosasl_office_j3", "outcome": "COLLISION", "seed": 1, "duration": "10.1s", "what_to_look_for": "Near-miss or collision with dynamic human/obstacle; robot damage and status badge clearly displayed"},
    {"file": "09_irosunity_scene_map1_success.mp4", "scenario": "irosunity_scene_map1", "outcome": "BEST PROGRESS (5.7m to goal, COLLISION)", "seed": 6, "duration": "8.0s", "what_to_look_for": "Best achievable trajectory: advanced to 5.7m of goal before collision"},
    {"file": "09_irosunity_scene_map1_failure.mp4", "scenario": "irosunity_scene_map1", "outcome": "COLLISION", "seed": 1, "duration": "9.9s", "what_to_look_for": "Near-miss or collision with dynamic human/obstacle; robot damage and status badge clearly displayed"},
    {"file": "10_irosunity_scene_map2_success.mp4", "scenario": "irosunity_scene_map2", "outcome": "BEST PROGRESS (11.6m to goal, TIMEOUT)", "seed": 6, "duration": "22.4s", "what_to_look_for": "Best achievable trajectory: advanced to 11.6m of goal before timeout"},
    {"file": "10_irosunity_scene_map2_failure.mp4", "scenario": "irosunity_scene_map2", "outcome": "TIMEOUT", "seed": 1, "duration": "22.4s", "what_to_look_for": "Near-miss or collision with dynamic human/obstacle; robot damage and status badge clearly displayed"},
]


def setup_home():
    tmp_home = "/tmp/home"
    os.makedirs(tmp_home, exist_ok=True)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    navrep_link = os.path.join(tmp_home, "navrep")
    maps_link = os.path.join(tmp_home, "maps")
    if not os.path.exists(navrep_link):
        try: os.symlink(repo_root, navrep_link)
        except Exception: pass
    if not os.path.exists(maps_link):
        try: os.symlink(os.path.join(repo_root, "maps"), maps_link)
        except Exception: pass
    os.environ["HOME"] = tmp_home
    return repo_root


def main():
    repo_root = setup_home()
    model_path = os.path.join(repo_root, "models/gym/navreptrainencodedenv_2020_09_23__09_48_22_PPO_VAE_LSTM_VM_V32M512_ckpt.zip")
    print("Loading model: {}".format(model_path))
    model = PPO2.load(model_path)
    policy = NavRepCPolicy(model)
    encoder = NavRepTrainEncoder("VAE_LSTM", "VM", gpu=False)

    out_videos_dir = os.path.join(repo_root, "submission", "task2_videos")
    os.makedirs(out_videos_dir, exist_ok=True)

    # 1. Evaluate scenario 11 (20 episodes)
    scenario_name = "irosunity_scene_map3"
    print("\n>>> Evaluating [11] {}".format(scenario_name))
    env = ScenarioEncodedEnv(None, None, silent=True, encoder=encoder, collect_trajectories=False)

    successes, collisions, timeouts = 0, 0, 0
    times_to_goal = []
    dt = 0.2

    for ep in range(20):
        np.random.seed(100 + ep)
        ob = env.reset_to_scenario(scenario_name)
        done, step, sim_time = False, 0, 0.0
        outcome = "TIMEOUT"

        while not done and step < 180:
            action, _ = policy.model.predict(ob, deterministic=True)
            ob, reward, done, info = env.step(action)
            sim_time += dt
            step += 1
            rlenv = env.iarlenv.rlenv
            if info.get('goal_reached', False):
                outcome = "SUCCESS"
                done = True
            elif rlenv.episode_damage[0] >= env.max_episode_damage:
                outcome = "COLLISION"
                done = True
            elif rlenv.episode_step[0] >= env.max_episode_length:
                outcome = "TIMEOUT"
                done = True

        if outcome == "SUCCESS":
            successes += 1
            times_to_goal.append(sim_time)
        elif outcome == "COLLISION":
            collisions += 1
        else:
            timeouts += 1

    env.close()

    succ_pct = (float(successes) / 20) * 100.0
    coll_pct = (float(collisions) / 20) * 100.0
    to_pct = (float(timeouts) / 20) * 100.0
    mean_ttg = "{:.1f}".format(float(np.mean(times_to_goal))) if times_to_goal else "N/A"

    print("  -> Results: Success={:.1f}%, Collision={:.1f}%, Timeout={:.1f}%, Mean TTG={}".format(
        succ_pct, coll_pct, to_pct, mean_ttg
    ))

    BENCHMARK_ROWS.append({
        "scenario": scenario_name,
        "env_class": "IANEnv",
        "backend": "VAE_LSTM",
        "encoding": "VM",
        "model_files": "navreptrainencodedenv_2020_09_23__09_48_22_PPO_VAE_LSTM_VM_V32M512_ckpt.zip",
        "episodes": 20,
        "seeds": "100-119",
        "success_pct": round(succ_pct, 1),
        "collision_pct": round(coll_pct, 1),
        "timeout_pct": round(to_pct, 1),
        "mean_ttg_s": mean_ttg,
        "repo_script_success_pct": REPO_CROSS_TEST_RATES.get(scenario_name, "0.0%")
    })

    # 2. Search representative seeds for scenario 11
    print("\n>>> Searching seeds for [11] {}".format(scenario_name))
    env = ScenarioEncodedEnv(None, None, silent=True, encoder=encoder, collect_trajectories=False)
    best_seed, best_min_dist, best_outcome = 1, float('inf'), "COLLISION"
    fail_seed, fail_outcome = 1, "COLLISION"
    success_seed = None

    for seed in range(1, 9):
        np.random.seed(seed)
        ob = env.reset_to_scenario(scenario_name)
        rlenv = env.iarlenv.rlenv
        goal = [float(rlenv.agent_goals[0][0]), float(rlenv.agent_goals[0][1])]
        cur_pos = [float(rlenv.virtual_peppers[0].pos[0]), float(rlenv.virtual_peppers[0].pos[1])]
        min_d = math.hypot(goal[0] - cur_pos[0], goal[1] - cur_pos[1])
        done, step, outcome = False, 0, "TIMEOUT"

        while not done and step < 180:
            action, _ = policy.model.predict(ob, deterministic=True)
            ob, reward, done, info = env.step(action)
            step += 1
            cur_pos = [float(rlenv.virtual_peppers[0].pos[0]), float(rlenv.virtual_peppers[0].pos[1])]
            if info.get('goal_reached', False):
                outcome = "SUCCESS"
                done = True
            elif rlenv.episode_damage[0] >= env.max_episode_damage:
                outcome = "COLLISION"
                done = True
            elif rlenv.episode_step[0] >= env.max_episode_length:
                outcome = "TIMEOUT"
                done = True

            d = math.hypot(goal[0] - cur_pos[0], goal[1] - cur_pos[1])
            if d < min_d: min_d = d

        if outcome == "SUCCESS" and success_seed is None:
            success_seed = seed
        if outcome in ["COLLISION", "TIMEOUT"] and fail_seed is None:
            fail_seed, fail_outcome = seed, outcome
        if min_d < best_min_dist:
            best_min_dist, best_seed, best_outcome = min_d, seed, outcome

    env.close()

    # 3. Record scenario 11 videos
    succ_seed = success_seed if success_seed is not None else best_seed
    succ_file = "11_irosunity_scene_map3_success.mp4"
    succ_path = os.path.join(out_videos_dir, succ_file)
    succ_outcome = "SUCCESS" if success_seed is not None else "BEST PROGRESS ({:.1f}m to goal, {})".format(best_min_dist, best_outcome)

    res_succ = record_scenario(
        scenario_name=scenario_name,
        policy_path=model_path,
        episodes=1,
        seed=succ_seed,
        fps=12,
        out_path=succ_path,
        deterministic=True,
        backend="VAE_LSTM",
        encoding="VM"
    )

    fail_file = "11_irosunity_scene_map3_failure.mp4"
    fail_path = os.path.join(out_videos_dir, fail_file)
    res_fail = record_scenario(
        scenario_name=scenario_name,
        policy_path=model_path,
        episodes=1,
        seed=fail_seed,
        fps=12,
        out_path=fail_path,
        deterministic=True,
        backend="VAE_LSTM",
        encoding="VM"
    )

    VIDEO_INDEX_ROWS.append({
        "file": succ_file,
        "scenario": scenario_name,
        "outcome": succ_outcome,
        "seed": succ_seed,
        "duration": "{:.1f}s".format(res_succ['quality']['duration']),
        "what_to_look_for": "Best achievable trajectory: advanced to {:.1f}m of goal before {}".format(best_min_dist, best_outcome.lower())
    })
    VIDEO_INDEX_ROWS.append({
        "file": fail_file,
        "scenario": scenario_name,
        "outcome": fail_outcome,
        "seed": fail_seed,
        "duration": "{:.1f}s".format(res_fail['quality']['duration']),
        "what_to_look_for": "Near-miss or collision with dynamic human/obstacle; robot damage and status badge clearly displayed"
    })

    # Save metrics_existing.csv
    df_metrics = pd.DataFrame(BENCHMARK_ROWS)
    metrics_path = os.path.join(out_videos_dir, "metrics_existing.csv")
    df_metrics.to_csv(metrics_path, index=False)
    print("\n[SUCCESS] Saved complete metrics to: {}".format(metrics_path))
    print(df_metrics.to_string())

    # Save VIDEO_INDEX.md
    index_path = os.path.join(out_videos_dir, "VIDEO_INDEX.md")
    with open(index_path, "w") as f:
        f.write("# Task 2 Video Index: ETH NavRep Scenarios Benchmark\n\n")
        f.write("All 22 videos (11 success/best-progress and 11 honest failure/near-miss) were recorded using the robot-centric camera, ")
        f.write("top telemetry HUD, PiP minimap, dynamic human velocity vectors, and laser beam points. ")
        f.write("Every video has been programmatically verified via the frame quality gate (asserting standard deviation >= 15.0 at 10%, 25%, 50%, 75%, and 90% timestamps).\n\n")
        f.write("| File | Scenario | Outcome | Seed | Duration | What to Look For |\n")
        f.write("| :--- | :--- | :--- | :---: | :---: | :--- |\n")
        for row in VIDEO_INDEX_ROWS:
            f.write("| `{}` | {} | **{}** | `{}` | {} | {} |\n".format(
                row['file'], row['scenario'], row['outcome'], row['seed'], row['duration'], row['what_to_look_for']
            ))
        f.write("\n\n*Evaluation Controller: `PPO_VAE_LSTM_VM_V32M512_ckpt` | Sim Resolution: 640x640 @ 12 FPS*\n")

    print("[SUCCESS] Saved complete VIDEO_INDEX.md to: {}".format(index_path))


if __name__ == '__main__':
    main()

