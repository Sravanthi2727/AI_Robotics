#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
benchmark_airport.py
====================
Calibrates and benchmarks the Airport Passenger Assistance scenario.

1. Calibrates seeds 0-59 at MEDIUM to confirm successful seeds.
2. Benchmarks 20 episodes (seeds 100-119) per difficulty level:
   - EASY: 5 humans, time limit 420
   - MEDIUM: 8 humans, time limit 360, boarding call at step 75
   - HARD: 12 humans, time limit 290, boarding call at step 50
3. Generates 2D map preview image at submission/task3_custom/maps/airport_passenger_assistance.png
4. Writes metrics to:
   - submission/task3_custom/airport_metrics.csv
   - results/airport_benchmark.json
"""

from __future__ import print_function
import os
import sys
import math
import json
import csv
import numpy as np
import cv2

# Set MPLCONFIGDIR to avoid warnings
os.environ["MPLCONFIGDIR"] = os.environ.get("TEMP", "/tmp")

# Ensure repository root is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Register scenario and load classes
import navrep_custom.airport_passenger_assistance as airport_mod
from navrep_custom.airport_passenger_assistance import (
    AirportNavigationController,
    AirportNavRepEnv,
    create_airport_config,
)
from navrep_custom.scenarios import get_scenario_config
from scripts_custom.record_scenario import transform_world_to_canvas
from crowd_sim.envs.utils.info import ReachGoal, Collision, CollisionOtherAgent, Timeout


def run_single_episode(difficulty="MEDIUM", seed=42):
    """
    Executes a single episode rollout and returns (outcome, steps, time_s, final_dist).
    """
    env = AirportNavRepEnv(difficulty=difficulty, silent=True)
    env.reset(seed=seed)
    dt = env._get_dt()

    controller = AirportNavigationController()
    controller.reset()
    sim = env.soadrl_sim

    step = 0
    done = False
    outcome = "TIMEOUT"

    while not done and step < env.time_limit:
        robot_pos = [sim.robot.px, sim.robot.py]
        action = controller.act(robot_pos, humans=sim.humans, action_dim=3)
        _, _, done, info = env.step(action)
        step += 1

        event = info.get("event")
        if isinstance(event, ReachGoal):
            outcome = "GOAL REACHED"
            done = True
        elif isinstance(event, (Collision, CollisionOtherAgent)):
            outcome = "COLLISION"
            done = True
        elif isinstance(event, Timeout):
            outcome = "TIMEOUT"
            done = True

    if not done:
        outcome = "TIMEOUT"

    final_dist = math.hypot(sim.robot.gx - sim.robot.px, sim.robot.gy - sim.robot.py)
    env.close()
    return outcome, step, step * dt, final_dist


def calibrate_seeds_0_59():
    """Searches seeds 0-59 at MEDIUM and logs outcomes."""
    print("=" * 60)
    print("CALIBRATING SEEDS 0-59 AT MEDIUM DIFFICULTY")
    print("=" * 60)

    successes = []
    collisions = []
    timeouts = []

    for s in range(60):
        outcome, steps, time_s, dist = run_single_episode(difficulty="MEDIUM", seed=s)
        if outcome == "GOAL REACHED":
            successes.append((s, steps, time_s, dist))
        elif outcome == "COLLISION":
            collisions.append((s, steps, time_s, dist))
        else:
            timeouts.append((s, steps, time_s, dist))

    print("Calibration Complete:")
    print("  Successes:  {} / 60 ({:.1f}%)".format(len(successes), len(successes) / 60.0 * 100))
    print("  Collisions: {} / 60 ({:.1f}%)".format(len(collisions), len(collisions) / 60.0 * 100))
    print("  Timeouts:   {} / 60 ({:.1f}%)".format(len(timeouts), len(timeouts) / 60.0 * 100))

    if successes:
        print("  Sample Success Seeds: {}".format([x[0] for x in successes[:5]]))
    if collisions:
        print("  Sample Collision Seeds: {}".format([x[0] for x in collisions[:5]]))
    if timeouts:
        print("  Sample Timeout Seeds: {}".format([x[0] for x in timeouts[:5]]))

    return successes, collisions, timeouts


def run_benchmark():
    """Runs 20-episode benchmark (seeds 100-119) per level."""
    print("\n" + "=" * 60)
    print("RUNNING 20-EPISODE BENCHMARK (SEEDS 100-119) PER DIFFICULTY")
    print("=" * 60)

    difficulties = ["EASY", "MEDIUM", "HARD"]
    seeds = list(range(100, 120))
    all_results = {}
    csv_rows = []

    for diff in difficulties:
        print("\n--- Evaluating Level: {} (20 episodes) ---".format(diff))
        diff_results = []
        success_count = 0
        collision_count = 0
        timeout_count = 0
        success_times = []

        for seed in seeds:
            outcome, steps, time_s, dist = run_single_episode(difficulty=diff, seed=seed)
            diff_results.append({
                "seed": seed,
                "outcome": outcome,
                "steps": steps,
                "time_s": round(time_s, 2),
                "final_dist_m": round(dist, 3)
            })

            if outcome == "GOAL REACHED":
                success_count += 1
                success_times.append(time_s)
            elif outcome == "COLLISION":
                collision_count += 1
            else:
                timeout_count += 1

            csv_rows.append({
                "scenario": "airport_passenger_assistance",
                "difficulty": diff,
                "seed": seed,
                "outcome": outcome,
                "steps": steps,
                "time_s": round(time_s, 2),
                "final_dist_m": round(dist, 3)
            })

        n_episodes = len(seeds)
        success_pct = (float(success_count) / n_episodes) * 100.0
        collision_pct = (float(collision_count) / n_episodes) * 100.0
        timeout_pct = (float(timeout_count) / n_episodes) * 100.0
        mean_time = float(np.mean(success_times)) if success_times else None

        all_results[diff] = {
            "episodes": n_episodes,
            "success_count": success_count,
            "success_rate_pct": success_pct,
            "collision_count": collision_count,
            "collision_rate_pct": collision_pct,
            "timeout_count": timeout_count,
            "timeout_rate_pct": timeout_pct,
            "mean_time_to_goal_s": round(mean_time, 2) if mean_time is not None else "N/A",
            "episodes_detail": diff_results
        }

        print("Level {}: Success={:.1f}% | Collision={:.1f}% | Timeout={:.1f}% | Mean Time={}".format(
            diff, success_pct, collision_pct, timeout_pct,
            "{:.2f}s".format(mean_time) if mean_time else "N/A"
        ))

    # Save CSV
    csv_path = "submission/task3_custom/airport_metrics.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "scenario", "difficulty", "seed", "outcome", "steps", "time_s", "final_dist_m"
        ])
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    print("\nSaved CSV metrics to: {}".format(csv_path))

    # Save JSON
    json_path = "results/airport_benchmark.json"
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved JSON benchmark to: {}".format(json_path))

    return all_results


def export_airport_map_preview(out_path="submission/task3_custom/maps/airport_passenger_assistance.png"):
    """Renders top-down 2D architectural map preview for Airport scenario."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cfg = get_scenario_config("airportpassengerassistance")
    bounds = cfg.world_bounds
    canvas_w, canvas_h = 720, 720
    frame = np.full((canvas_h, canvas_w, 3), (24, 28, 34), dtype=np.uint8)

    # Grid lines
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

    # Robot Start Marker (Green S)
    spx, spy, _ = transform_world_to_canvas(cfg.robot_start[0], cfg.robot_start[1], [bounds[0], bounds[1]], [bounds[2], bounds[3]], canvas_w, canvas_h, margin=60)
    cv2.circle(frame, (spx, spy), 14, (46, 204, 113), -1)
    cv2.putText(frame, "S", (spx - 6, spy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(frame, "CHECK-IN", (spx - 35, spy + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (46, 204, 113), 1)

    # Goal Marker (Gold G - Gate 12)
    gpx, gpy, _ = transform_world_to_canvas(cfg.robot_goal[0], cfg.robot_goal[1], [bounds[0], bounds[1]], [bounds[2], bounds[3]], canvas_w, canvas_h, margin=60)
    cv2.circle(frame, (gpx, gpy), 16, (0, 215, 255), 2)
    cv2.circle(frame, (gpx, gpy), 8, (0, 215, 255), -1)
    cv2.putText(frame, "GATE 12", (gpx - 25, gpy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 215, 255), 2)

    # Waypoints guidance trail
    ctrl = AirportNavigationController()
    wp_pts = []
    for wx, wy in ctrl.waypoints:
        wpx, wpy, _ = transform_world_to_canvas(wx, wy, [bounds[0], bounds[1]], [bounds[2], bounds[3]], canvas_w, canvas_h, margin=60)
        wp_pts.append((wpx, wpy))
        cv2.circle(frame, (wpx, wpy), 4, (0, 255, 255), -1)
    for i in range(len(wp_pts) - 1):
        cv2.line(frame, wp_pts[i], wp_pts[i + 1], (180, 180, 50), 1, cv2.LINE_AA)

    # Humans
    for h in cfg.base_humans:
        hpx, hpy, _ = transform_world_to_canvas(h['pos'][0], h['pos'][1], [bounds[0], bounds[1]], [bounds[2], bounds[3]], canvas_w, canvas_h, margin=60)
        cv2.circle(frame, (hpx, hpy), 10, (40, 90, 255), -1)  # Coral
        cv2.circle(frame, (hpx, hpy), 10, (255, 255, 255), 1)

    # Header Card
    cv2.rectangle(frame, (0, 0), (canvas_w, 56), (15, 18, 22), -1)
    cv2.line(frame, (0, 56), (canvas_w, 56), (50, 60, 75), 1)
    cv2.putText(frame, cfg.name, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

    cv2.imwrite(out_path, frame)
    print("Saved 2D map preview to: {}".format(out_path))


if __name__ == '__main__':
    export_airport_map_preview()
    calibrate_seeds_0_59()
    run_benchmark()
