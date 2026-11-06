#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_and_record_custom.py
Batch evaluation, video recording, and density sweep analysis for custom NavRep scenarios:
  1. MallMate (MEDIUM & HARD)
  2. MediRunner (MEDIUM & HARD)
  3. KiranaKart (MEDIUM & HARD)
  4. Diwali Bazaar Surat (MEDIUM & HARD)
  5. Platform 1 Railway Station (MEDIUM & HARD)
  6. Fire Drill Evacuation (MEDIUM & HARD)

Outputs:
  - submission/task3_custom/videos/<NN>_<scenario>_<diff>.mp4
  - submission/task3_custom/metrics_custom.csv
  - submission/task3_custom/analysis/density_vs_success.png
"""

from __future__ import print_function
import os
import sys
import time
import math
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure imageio_ffmpeg works on Python 3.6
import importlib
try:
    import importlib_resources
    sys.modules['importlib.resources'] = importlib_resources
    importlib.resources = importlib_resources
except ImportError:
    pass

import imageio
from stable_baselines import PPO2
from navrep.scripts.test_navrep import NavRepCPolicy
from crowd_sim.envs.utils.info import ReachGoal, Collision, CollisionOtherAgent, Timeout

from navrep_custom.scenarios import get_scenario_config, SCENARIO_REGISTRY
from navrep_custom.custom_envs import CustomNavRepEncodedEnv
from scripts_custom.record_scenario import render_scene_frame


CUSTOM_BENCHMARK_SUITE = [
    ("mallmate", "MEDIUM"),
    ("mallmate", "HARD"),
    ("medirunner", "MEDIUM"),
    ("medirunner", "HARD"),
    ("kiranakart", "MEDIUM"),
    ("kiranakart", "HARD"),
    ("diwalibazaar", "MEDIUM"),
    ("diwalibazaar", "HARD"),
    ("railwaystation", "MEDIUM"),
    ("railwaystation", "HARD"),
    ("firedrill", "MEDIUM"),
    ("firedrill", "HARD"),
]


def evaluate_custom_scenario(scenario_name, difficulty, policy, num_episodes=20, max_steps=350):
    """Evaluates a custom scenario over num_episodes and returns aggregated metrics."""
    env = CustomNavRepEncodedEnv(scenario=scenario_name, difficulty=difficulty, gpu=False, silent=True)
    dt = 0.2

    success_count = 0
    collision_count = 0
    timeout_count = 0
    times_to_goal = []

    print("[CUSTOM_BENCHMARK] Evaluating {} [{}] over {} episodes...".format(
        scenario_name, difficulty, num_episodes
    ))

    for ep in range(num_episodes):
        np.random.seed(300 + ep)
        ob = env.reset(scenario_name, difficulty)
        done = False
        step = 0
        sim_time = 0.0
        outcome = "TIMEOUT"

        while not done and step < max_steps:
            action = policy.act(ob)
            ob, reward, done, info = env.step(action)
            event = info.get('event', None)
            step += 1
            sim_time += dt

            if isinstance(event, ReachGoal):
                outcome = "SUCCESS"
                done = True
            elif isinstance(event, (Collision, CollisionOtherAgent)):
                outcome = "COLLISION"
                done = True
            elif isinstance(event, Timeout):
                outcome = "TIMEOUT"
                done = True

        if outcome == "SUCCESS":
            success_count += 1
            times_to_goal.append(sim_time)
        elif outcome == "COLLISION":
            collision_count += 1
        else:
            timeout_count += 1

    env.close()

    success_rate = (float(success_count) / num_episodes) * 100.0
    collision_rate = (float(collision_count) / num_episodes) * 100.0
    timeout_rate = (float(timeout_count) / num_episodes) * 100.0
    mean_ttg = float(np.mean(times_to_goal)) if times_to_goal else 0.0

    print("  -> Success: {:.1f}% | Collision: {:.1f}% | Timeout: {:.1f}% | Mean TTG: {:.2f}s".format(
        success_rate, collision_rate, timeout_rate, mean_ttg
    ))

    return {
        "scenario": scenario_name,
        "difficulty": difficulty,
        "episodes_tested": num_episodes,
        "success_rate_pct": round(success_rate, 2),
        "collision_rate_pct": round(collision_rate, 2),
        "timeout_rate_pct": round(timeout_rate, 2),
        "mean_time_to_goal_s": round(mean_ttg, 2)
    }


def record_custom_video(scenario_name, difficulty, policy, out_path, num_episodes=2, fps=12, max_steps=400):
    """Records episodes of a custom scenario to MP4 with telemetry overlay."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    env = CustomNavRepEncodedEnv(scenario=scenario_name, difficulty=difficulty, gpu=False, silent=True)
    dt = 0.2

    writer = imageio.get_writer(out_path, fps=fps, codec='libx264', pixelformat='yuv420p')
    print("[CUSTOM_RECORD] Recording {} [{}] -> {}".format(scenario_name, difficulty, out_path))

    total_frames = 0
    for ep in range(1, num_episodes + 1):
        seed = 77 + ep * 13
        np.random.seed(seed)
        ob = env.reset(scenario_name, difficulty)

        sim = env.soadrl_sim
        start_pos = [sim.robot.px, sim.robot.py]
        robot_trail = [start_pos.copy()]
        status = "RUNNING"
        done = False
        step = 0
        sim_time = 0.0

        while not done and step < max_steps:
            action = policy.act(ob)
            ob, reward, done, info = env.step(action)
            cur_pos = [sim.robot.px, sim.robot.py]
            robot_trail.append(cur_pos)
            sim_time += dt
            step += 1

            event = info.get('event', None)
            if isinstance(event, ReachGoal):
                status = "GOAL REACHED"
                done = True
            elif isinstance(event, (Collision, CollisionOtherAgent)):
                status = "COLLISION"
                done = True
            elif isinstance(event, Timeout):
                status = "TIMEOUT"
                done = True

            frame = render_scene_frame(
                env=env,
                scenario_name="{}_{}".format(scenario_name.upper(), difficulty),
                episode_num=ep,
                step_num=step,
                sim_time=sim_time,
                status=status,
                robot_trail=robot_trail,
                start_pos=start_pos
            )
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            writer.append_data(rgb_frame)
            total_frames += 1

        # Hold outcome frame for 1.5 seconds
        hold_frames = int(fps * 1.5)
        final_frame = render_scene_frame(
            env=env,
            scenario_name="{}_{}".format(scenario_name.upper(), difficulty),
            episode_num=ep,
            step_num=step,
            sim_time=sim_time,
            status=status,
            robot_trail=robot_trail,
            start_pos=start_pos
        )
        final_rgb = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
        for _ in range(hold_frames):
            writer.append_data(final_rgb)
            total_frames += 1

    writer.close()
    env.close()

    duration = float(total_frames) / fps
    print("  -> Saved: {} (Frames: {}, Duration: {:.2f}s)".format(out_path, total_frames, duration))
    return duration


def run_density_analysis(policy, out_plot="submission/task3_custom/analysis/density_vs_success.png"):
    """Runs parametric crowd density stress test on Diwali Bazaar and plots performance degradation curves."""
    os.makedirs(os.path.dirname(os.path.abspath(out_plot)), exist_ok=True)
    scenario_name = "diwalibazaar"
    densities = [2, 4, 6, 8, 10, 12, 14]
    episodes_per_density = 10

    print("[DENSITY_ANALYSIS] Sweeping crowd density on '{}' across {}:".format(
        scenario_name, densities
    ))

    success_rates = []
    collision_rates = []
    timeout_rates = []

    for d in densities:
        env = CustomNavRepEncodedEnv(scenario=scenario_name, difficulty="MEDIUM", gpu=False, silent=True)
        env.raw_env.scenario_config.difficulty_modifiers['MEDIUM']['extra_humans'] = max(d - 6, 0)

        successes = 0
        collisions = 0
        timeouts = 0

        for ep in range(episodes_per_density):
            np.random.seed(800 + d * 10 + ep)
            ob = env.reset(scenario_name, "MEDIUM")
            done = False
            step = 0
            while not done and step < 250:
                action = policy.act(ob)
                ob, reward, done, info = env.step(action)
                step += 1
                event = info.get('event', None)
                if isinstance(event, ReachGoal):
                    successes += 1
                    done = True
                elif isinstance(event, (Collision, CollisionOtherAgent)):
                    collisions += 1
                    done = True
                elif isinstance(event, Timeout) or step >= 250:
                    timeouts += 1
                    done = True

        env.close()
        s_pct = (float(successes) / episodes_per_density) * 100.0
        c_pct = (float(collisions) / episodes_per_density) * 100.0
        t_pct = (float(timeouts) / episodes_per_density) * 100.0
        success_rates.append(s_pct)
        collision_rates.append(c_pct)
        timeout_rates.append(t_pct)
        print("  Density {:2d} agents: Success={:5.1f}% | Collision={:5.1f}% | Timeout={:5.1f}%".format(
            d, s_pct, c_pct, t_pct
        ))

    # Generate Professional Matplotlib Plot
    plt.figure(figsize=(9, 5.5))
    plt.plot(densities, success_rates, marker='o', linewidth=2.5, color='#2ecc71', label='Success Rate (%)')
    plt.plot(densities, collision_rates, marker='s', linewidth=2.5, color='#e74c3c', label='Collision Rate (%)')
    plt.plot(densities, timeout_rates, marker='^', linewidth=2.0, linestyle='--', color='#f39c12', label='Timeout Rate (%)')

    plt.title("Crowd Density vs Navigation Performance (Scenario: Diwali Bazaar Surat)", fontsize=13, fontweight='bold')
    plt.xlabel("Crowd Density (Number of Dynamic Human Agents)", fontsize=11)
    plt.ylabel("Rate (%)", fontsize=11)
    plt.ylim([-5, 105])
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True, facecolor='#f8f9fa', fontsize=10)
    plt.tight_layout()
    plt.savefig(out_plot, dpi=200)
    plt.close()
    print("[DENSITY_ANALYSIS] Analysis plot saved to: {}".format(out_plot))


def main():
    print("=" * 70)
    print("NAVREP PHASE 3: CUSTOM SCENARIOS BENCHMARK & VIDEO RECORDINGS")
    print("=" * 70)

    video_dir = "submission/task3_custom/videos"
    os.makedirs(video_dir, exist_ok=True)
    metrics_path = "submission/task3_custom/metrics_custom.csv"

    # Load Shared Pretrained Policy
    model_path = os.path.expanduser(
        "~/navrep/models/gym/navreptrainencodedenv_2020_09_22__01_14_06_PPO_VAE_LSTM_V_ONLY_V32M512_ckpt.zip"
    )
    print("Loading pretrained PPO policy: {}".format(model_path))
    ppo_model = PPO2.load(model_path)
    policy = NavRepCPolicy(ppo_model)

    metrics_list = []
    video_records = []

    for idx, (sc, diff) in enumerate(CUSTOM_BENCHMARK_SUITE, 1):
        # 1. 20-episode benchmark metrics
        m = evaluate_custom_scenario(sc, diff, policy, num_episodes=20)
        metrics_list.append(m)

        # 2. Record video (2 episodes)
        video_name = "{:02d}_{}_{}.mp4".format(idx, sc, diff.lower())
        out_video_path = os.path.join(video_dir, video_name)
        duration = record_custom_video(sc, diff, policy, out_video_path, num_episodes=2, fps=12)
        video_records.append((video_name, duration))

    # Save metrics CSV
    df = pd.DataFrame(metrics_list)
    df.to_csv(metrics_path, index=False)
    print("\nCustom metrics written successfully to: {}".format(metrics_path))

    # Run Density Analysis
    run_density_analysis(policy)

    # Verification Summary
    print("\n" + "=" * 70)
    print("PHASE 3 VERIFICATION SUMMARY")
    print("=" * 70)
    for vname, dur in video_records:
        status = "PASS (> 5s)" if dur >= 5.0 else "FAIL (< 5s)"
        print("  - Video: {:35s} | Duration: {:5.2f}s | [{}]".format(vname, dur, status))

    print("\nCustom Scenarios Metrics Table:")
    print(df.to_string(index=False))
    print("=" * 70)


if __name__ == '__main__':
    main()

