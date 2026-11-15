#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
record_hospital_demo.py
=======================
Records a MediRunner Hospital Delivery navigation demonstration video.

Uses HospitalNavigationController (a hand-crafted waypoint + emergency-gurney
yielding controller) to navigate from the Central Pharmacy (-10.5, 0.0) to the
ICU Nurse Station (+10.5, 0.0) in the custom ``medirunner`` scenario.

Usage::

    # EASY smoke test (default):
    python scripts_custom/record_hospital_demo.py

    # Specific difficulty:
    python scripts_custom/record_hospital_demo.py --difficulty MEDIUM

    # Full argument set:
    python scripts_custom/record_hospital_demo.py \\
        --difficulty HARD --episodes 1 --max-steps 350 \\
        --output-dir submission/task3_custom/videos

Outputs:
    submission/task3_custom/videos/hospital_easy.mp4   (EASY)
    submission/task3_custom/videos/hospital_medium.mp4 (MEDIUM)
    submission/task3_custom/videos/hospital_hard.mp4   (HARD)

Python 3.6 compatible (no f-strings, no walrus operator).
"""

from __future__ import print_function, division

import os
import sys
import argparse

import numpy as np
import cv2
import imageio

from crowd_sim.envs.utils.info import ReachGoal, Collision, CollisionOtherAgent, Timeout

from navrep_custom.scenarios import get_scenario_config
from navrep_custom.custom_envs import CustomNavRepEnv
from navrep_custom.hospital_delivery import (
    HospitalNavigationController,
    sync_and_rasterize_hospital_sim,
)
from scripts_custom.record_scenario import render_scene_frame, verify_video_frames


# ---------------------------------------------------------------------------
# Output filename helper
# ---------------------------------------------------------------------------

_DIFFICULTY_TO_FILENAME = {
    "EASY":   "hospital_easy.mp4",
    "MEDIUM": "hospital_medium.mp4",
    "HARD":   "hospital_hard.mp4",
}


def _output_path(difficulty, output_dir):
    filename = _DIFFICULTY_TO_FILENAME.get(
        difficulty.upper(),
        "hospital_{}.mp4".format(difficulty.lower())
    )
    return os.path.join(output_dir, filename)


# ---------------------------------------------------------------------------
# Main recording function
# ---------------------------------------------------------------------------

def record_hospital_demo(difficulty="EASY", episodes=1, fps=12,
                         output_dir="submission/task3_custom/videos",
                         max_steps=350):
    """Record one or more MediRunner episodes and save a video."""

    np.random.seed(42)

    out_path = _output_path(difficulty, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Scenario config (static obstacles, world bounds, etc.)
    cfg = get_scenario_config("medirunner")

    # Build environment (non-encoded; same pattern as record_hotel_demo.py)
    env = CustomNavRepEnv(
        scenario="medirunner",
        difficulty=difficulty,
        silent=True,
    )

    # Controller is stateful; a single instance is reused across episodes.
    controller = HospitalNavigationController(
        static_obstacles=cfg.static_obstacles,
        world_bounds=cfg.world_bounds,
    )

    dt = env._get_dt()
    total_frames = 0

    writer = imageio.get_writer(
        out_path,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
    )

    scenario_label = "MEDIRUNNER HOSPITAL - {}".format(difficulty.upper())

    try:
        for ep in range(1, episodes + 1):
            # ------------------------------------------------------------------
            # Episode reset
            # ------------------------------------------------------------------
            env.reset("medirunner", difficulty)

            # Synchronise occupancy grid and protect spawn zones
            sync_and_rasterize_hospital_sim(env, cfg)

            # Reset controller state machine for new episode
            controller.reset()

            sim = env.soadrl_sim
            start_pos = [sim.robot.px, sim.robot.py]
            robot_trail = [start_pos[:]]

            status = "RUNNING"
            sim_time = 0.0
            step = 0
            done = False

            print("[HOSPITAL DEMO] Episode {}/{} started (difficulty={})".format(
                ep, episodes, difficulty))

            # ------------------------------------------------------------------
            # Simulation loop
            # ------------------------------------------------------------------
            while not done and step < max_steps:
                robot_pos = [sim.robot.px, sim.robot.py]
                humans = sim.humans
                human_meta = env.human_meta   # list of dicts with 'role', etc.

                # Controller returns [vx, vy, 0.0] for action_dim=3
                action = controller.act(
                    robot_pos,
                    humans=humans,
                    action_dim=3,
                    human_meta=human_meta,
                )

                _, _, done, info = env.step(action)

                step += 1
                sim_time += dt

                robot_trail.append([sim.robot.px, sim.robot.py])

                event = info.get("event")

                if isinstance(event, ReachGoal):
                    status = "GOAL REACHED"
                    done = True
                elif isinstance(event, (Collision, CollisionOtherAgent)):
                    status = "COLLISION"
                    done = True
                elif isinstance(event, Timeout):
                    status = "TIMEOUT"
                    done = True

                # Render and append frame
                frame = render_scene_frame(
                    env=env,
                    scenario_name=scenario_label,
                    episode_num=ep,
                    step_num=step,
                    sim_time=sim_time,
                    status=status,
                    robot_trail=robot_trail,
                    start_pos=start_pos,
                    policy_name="HOSPITAL NAVIGATION",
                )

                writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                total_frames += 1

            if not done:
                status = "TIMEOUT"

            # ------------------------------------------------------------------
            # Hold final frame for 2 seconds
            # ------------------------------------------------------------------
            final_frame = render_scene_frame(
                env=env,
                scenario_name=scenario_label,
                episode_num=ep,
                step_num=step,
                sim_time=sim_time,
                status=status,
                robot_trail=robot_trail,
                start_pos=start_pos,
                policy_name="HOSPITAL NAVIGATION",
            )
            final_rgb = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
            for _ in range(int(fps * 2)):
                writer.append_data(final_rgb)
                total_frames += 1

            print("[HOSPITAL DEMO] Episode {} result: {} | steps={} | sim_time={:.1f}s".format(
                ep, status, step, sim_time))
            print("[HOSPITAL DEMO] Final robot position: ({:.2f}, {:.2f})".format(
                sim.robot.px, sim.robot.py))

    finally:
        writer.close()
        env.close()

    duration = float(total_frames) / fps

    print("[HOSPITAL DEMO] Video saved: {}".format(out_path))
    print("[HOSPITAL DEMO] Duration: {:.2f}s  |  Frames: {}".format(duration, total_frames))

    verify_video_frames(out_path)
    print("[HOSPITAL DEMO] Video quality check passed!")

    return out_path, status, step, sim_time


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Record MediRunner Hospital Delivery navigation demo."
    )
    parser.add_argument(
        "--difficulty",
        default="EASY",
        choices=["EASY", "MEDIUM", "HARD"],
        help="Scenario difficulty level (default: EASY)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to record (default: 1)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=350,
        dest="max_steps",
        help="Maximum simulation steps per episode (default: 350)",
    )
    parser.add_argument(
        "--output-dir",
        default="submission/task3_custom/videos",
        dest="output_dir",
        help="Output directory for video files",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=12,
        help="Frames per second for output video (default: 12)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    record_hospital_demo(
        difficulty=args.difficulty,
        episodes=args.episodes,
        fps=args.fps,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
    )
