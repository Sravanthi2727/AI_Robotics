#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_custom.py
Command-line interface to execute, benchmark, analyze, and record custom NavRep scenarios.

Features:
  - Run any registered scenario ('mallmate', 'medirunner', 'kiranakart', 'diwalibazaar', 'railwaystation', 'firedrill')
  - Difficulty ladder support: EASY, MEDIUM, HARD
  - Video recording with telemetry HUD overlay, trajectory trail, and goal markers
  - Crowd density sweep analysis: plots success/collision rates vs crowd density
  - 2D map preview export to PNG
"""

from __future__ import print_function
import os
import sys
import argparse
import time
import math
import numpy as np
import pandas as pd
import cv2

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
from scripts_custom.record_scenario import render_scene_frame, transform_world_to_canvas


def record_custom_scenario(scenario_name="mallmate",
                           difficulty="MEDIUM",
                           policy_path=None,
                           episodes=2,
                           seed=42,
                           fps=12,
                           out_path=None,
                           max_steps=400):
    """Runs episodes of a custom scenario and records an MP4 video."""
    np.random.seed(seed)
    
    if policy_path is None or not os.path.exists(policy_path):
        policy_path = os.path.expanduser(
            "~/navrep/models/gym/navreptrainencodedenv_2020_09_22__01_14_06_PPO_VAE_LSTM_V_ONLY_V32M512_ckpt.zip"
        )
    print("[RUN_CUSTOM] Loading policy: {}".format(policy_path))
    ppo_model = PPO2.load(policy_path)
    policy = NavRepCPolicy(ppo_model)

    env = CustomNavRepEncodedEnv(scenario=scenario_name, difficulty=difficulty, gpu=False)
    
    if out_path is None:
        out_path = "submission/task3_custom/videos/{}_{}.mp4".format(scenario_name, difficulty.lower())
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    writer = imageio.get_writer(out_path, fps=fps, codec='libx264', pixelformat='yuv420p')
    print("[RUN_CUSTOM] Recording {} [Diff: {}] -> {}".format(scenario_name, difficulty, out_path))
    dt = 0.2
    total_frames = 0

    for ep in range(1, episodes + 1):
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
                scenario_name="{}_{}".format(scenario_name.upper(), difficulty.upper()),
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

        if not done and step >= max_steps:
            status = "TIMEOUT"

        # Hold outcome frame for 1.5 seconds
        hold_frames = int(fps * 1.5)
        final_frame = render_scene_frame(
            env=env,
            scenario_name="{}_{}".format(scenario_name.upper(), difficulty.upper()),
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

        print("  Episode {}: Status={} Steps={} SimTime={:.1f}s".format(ep, status, step, sim_time))

    writer.close()
    env.close()
    duration = float(total_frames) / fps
    print("[RUN_CUSTOM] Video saved: {} (Duration: {:.2f}s)".format(out_path, duration))
    return out_path, duration


def export_map_previews(out_dir="submission/task3_custom/maps"):
    """Renders top-down 2D architectural map previews for all custom scenarios."""
    os.makedirs(out_dir, exist_ok=True)
    print("[MAP_EXPORT] Exporting top-down 2D map previews...")

    for sc_id, factory in SCENARIO_REGISTRY.items():
        cfg = factory()
        bounds = cfg.world_bounds
        canvas_w, canvas_h = 720, 720
        frame = np.full((canvas_h, canvas_w, 3), (24, 28, 34), dtype=np.uint8)

        # Draw grid
        for gx in range(0, canvas_w, 40):
            cv2.line(frame, (gx, 0), (gx, canvas_h), (34, 40, 50), 1)
        for gy in range(0, canvas_h, 40):
            cv2.line(frame, (0, gy), (canvas_w, gy), (34, 40, 50), 1)

        # Draw obstacles
        for poly in cfg.static_obstacles:
            pts = []
            for vx, vy in poly:
                px, py, _ = transform_world_to_canvas(vx, vy, [bounds[0], bounds[1]], [bounds[2], bounds[3]], canvas_w, canvas_h, margin=60)
                pts.append([px, py])
            if len(pts) >= 3:
                pts_arr = np.array(pts, dtype=np.int32)
                cv2.fillPoly(frame, [pts_arr], (60, 68, 82))
                cv2.polylines(frame, [pts_arr], True, (100, 115, 140), 2)

        # Start Marker (Green S)
        spx, spy, scale = transform_world_to_canvas(cfg.robot_start[0], cfg.robot_start[1], [bounds[0], bounds[1]], [bounds[2], bounds[3]], canvas_w, canvas_h, margin=60)
        cv2.circle(frame, (spx, spy), 14, (46, 204, 113), -1)
        cv2.putText(frame, "S", (spx - 6, spy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Goal Marker (Gold G)
        gpx, gpy, _ = transform_world_to_canvas(cfg.robot_goal[0], cfg.robot_goal[1], [bounds[0], bounds[1]], [bounds[2], bounds[3]], canvas_w, canvas_h, margin=60)
        cv2.circle(frame, (gpx, gpy), 16, (0, 215, 255), 2)
        cv2.circle(frame, (gpx, gpy), 8, (0, 215, 255), -1)
        cv2.putText(frame, "GOAL", (gpx + 14, gpy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2)

        # Guidance trajectory vector
        cv2.line(frame, (spx, spy), (gpx, gpy), (90, 90, 90), 1, cv2.LINE_AA)

        # Humans
        for h in cfg.base_humans:
            hpx, hpy, _ = transform_world_to_canvas(h['pos'][0], h['pos'][1], [bounds[0], bounds[1]], [bounds[2], bounds[3]], canvas_w, canvas_h, margin=60)
            cv2.circle(frame, (hpx, hpy), 10, (40, 90, 255), -1)  # Coral
            cv2.circle(frame, (hpx, hpy), 10, (255, 255, 255), 1)

        # Title Card
        cv2.rectangle(frame, (0, 0), (canvas_w, 56), (15, 18, 22), -1)
        cv2.line(frame, (0, 56), (canvas_w, 56), (50, 60, 75), 1)
        cv2.putText(frame, cfg.name, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        out_img_path = os.path.join(out_dir, "{}.png".format(sc_id))
        cv2.imwrite(out_img_path, frame)
        print("  - Saved map preview: {}".format(out_img_path))


def run_density_sweep_analysis(scenario_name="diwalibazaar",
                               densities=[2, 4, 6, 8, 10, 12, 14],
                               episodes_per_density=10,
                               out_plot="submission/task3_custom/analysis/density_vs_success.png"):
    """Sweeps human crowd density and plots Success / Collision degradation curves."""
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(out_plot)), exist_ok=True)
    print("[SWEEP] Starting crowd density sweep on '{}' across densities: {}".format(
        scenario_name, densities
    ))

    policy_path = os.path.expanduser(
        "~/navrep/models/gym/navreptrainencodedenv_2020_09_22__01_14_06_PPO_VAE_LSTM_V_ONLY_V32M512_ckpt.zip"
    )
    ppo_model = PPO2.load(policy_path)
    policy = NavRepCPolicy(ppo_model)

    success_rates = []
    collision_rates = []
    timeout_rates = []

    for d in densities:
        # Override extra humans to match density
        env = CustomNavRepEncodedEnv(scenario=scenario_name, difficulty="MEDIUM", gpu=False)
        # Modify scenario difficulty configuration dynamically
        env.raw_env.scenario_config.difficulty_modifiers['MEDIUM']['extra_humans'] = max(d - len(env.raw_env.scenario_config.base_humans), 0)

        successes = 0
        collisions = 0
        timeouts = 0

        for ep in range(episodes_per_density):
            np.random.seed(500 + d * 10 + ep)
            ob = env.reset(scenario_name, "MEDIUM")
            done = False
            step = 0
            while not done and step < 300:
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
                elif isinstance(event, Timeout) or step >= 300:
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

    # Plot results
    plt.figure(figsize=(9, 5.5))
    plt.plot(densities, success_rates, marker='o', linewidth=2.5, color='#2ecc71', label='Success Rate (%)')
    plt.plot(densities, collision_rates, marker='s', linewidth=2.5, color='#e74c3c', label='Collision Rate (%)')
    plt.plot(densities, timeout_rates, marker='^', linewidth=2.0, linestyle='--', color='#f39c12', label='Timeout Rate (%)')

    plt.title("Crowd Density vs Navigation Performance (Scenario: {})".format(scenario_name.upper()), fontsize=13, fontweight='bold')
    plt.xlabel("Crowd Density (Number of Dynamic Human Agents)", fontsize=11)
    plt.ylabel("Rate (%)", fontsize=11)
    plt.ylim([-5, 105])
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True, facecolor='#f8f9fa', fontsize=10)
    plt.tight_layout()
    plt.savefig(out_plot, dpi=200)
    plt.close()
    print("[SWEEP] Density sweep plot saved successfully to: {}".format(out_plot))


def main():
    parser = argparse.ArgumentParser(description="NavRep Custom Scenarios Runner & Suite")
    parser.add_argument("--scenario", type=str, default="mallmate", help="Custom scenario name")
    parser.add_argument("--difficulty", type=str, default="MEDIUM", choices=["EASY", "MEDIUM", "HARD"], help="Difficulty")
    parser.add_argument("--policy", type=str, default=None, help="Policy zip path")
    parser.add_argument("--episodes", type=int, default=2, help="Number of episodes")
    parser.add_argument("--fps", type=int, default=12, help="Recording FPS")
    parser.add_argument("--out", type=str, default=None, help="Output MP4 path")
    parser.add_argument("--export-maps", action="store_true", help="Generate 2D map preview PNGs")
    parser.add_argument("--sweep-density", action="store_true", help="Run crowd density sweep analysis")
    args = parser.parse_args()

    if args.export_maps:
        export_map_previews()

    if args.sweep_density:
        run_density_sweep_analysis(scenario_name=args.scenario)

    if not args.export_maps and not args.sweep_density:
        record_custom_scenario(
            scenario_name=args.scenario,
            difficulty=args.difficulty,
            policy_path=args.policy,
            episodes=args.episodes,
            fps=args.fps,
            out_path=args.out
        )


if __name__ == '__main__':
    main()

