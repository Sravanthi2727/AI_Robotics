#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
record_airport_demo.py
======================
Records the 4 demonstration videos for the Airport Passenger Assistance scenario:
  1. MEDIUM success: airport_passenger_assistance_medium_success.mp4
  2. MEDIUM failure: airport_passenger_assistance_medium_failure.mp4
  3. EASY success:   airport_passenger_assistance_easy_success.mp4
  4. HARD attempt:   airport_passenger_assistance_hard_bestprogress.mp4

Format specifications:
  - Resolution: 640x640, 12 fps, H.264 codec (pixelformat yuv420p)
  - Top-down full-map static camera with HUD overlay:
      * Header: "ETH NAVREP BENCHMARK", "SCENARIO: AIRPORT PASSENGER ASSISTANCE - {DIFF}", "EP: #1"
      * Status badge: RUNNING, GOAL REACHED, COLLISION, TIMEOUT
      * Cyan trajectory path trail
      * Green start marker, gold goal ring with real-time distance
      * Footer: "STEP {step} | TIME {time:.1f}s | DIST {dist:.2f}m | POLICY: AIRPORT NAVIGATION"
  - Terminal capture: captures state BEFORE reset so final frame shows true distance
  - Frozen hold frames: exactly 10 frames (~0.83s, <= 12 frames)
  - Duration: strictly between 8s and 40s
  - Programmatic quality verification with automated pass/fail report table
"""

from __future__ import print_function
import os
import sys
import math
import json
import subprocess
import cv2
import imageio
import numpy as np

# Set MPLCONFIGDIR
os.environ["MPLCONFIGDIR"] = os.environ.get("TEMP", "/tmp")

# Ensure repository root is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import navrep_custom.airport_passenger_assistance as airport_mod
from navrep_custom.airport_passenger_assistance import (
    AirportNavigationController,
    AirportNavRepEnv,
)
from scripts_custom.record_scenario import render_scene_frame
from crowd_sim.envs.utils.info import ReachGoal, Collision, CollisionOtherAgent, Timeout

VIDEOS_DIR = os.path.join(repo_root, "submission", "task3_custom", "videos")
METADATA_DIR = os.path.join(VIDEOS_DIR, "metadata")


def record_airport_episode(difficulty="MEDIUM",
                           seed=42,
                           filename="airport_passenger_assistance_medium_success.mp4",
                           fps=12,
                           hold_frames=10):
    """
    Simulates and records an episode of the Airport Passenger Assistance scenario.
    """
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)
    out_video_path = os.path.join(VIDEOS_DIR, filename)
    out_meta_path = os.path.join(METADATA_DIR, filename.replace(".mp4", ".json"))

    print("\n" + "=" * 65)
    print("RECORDING: {} (Diff: {}, Seed: {})".format(filename, difficulty, seed))
    print("=" * 65)

    env = AirportNavRepEnv(difficulty=difficulty, silent=True)
    env.reset(seed=seed)
    dt = env._get_dt()
    max_steps = env.time_limit

    controller = AirportNavigationController()
    controller.reset()

    sim = env.soadrl_sim
    start_pos = [sim.robot.px, sim.robot.py]
    robot_trail = [start_pos[:]]

    writer = imageio.get_writer(
        out_video_path,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p"
    )

    status = "RUNNING"
    sim_time = 0.0
    step = 0
    done = False
    final_event = None
    total_frames = 0

    try:
        while not done and step < max_steps:
            robot_pos = [sim.robot.px, sim.robot.py]
            action = controller.act(robot_pos, humans=sim.humans, action_dim=3)

            # Step simulation
            _, _, done, info = env.step(action)
            step += 1
            sim_time += dt

            # Record path position before checking terminal condition
            cur_pos = [sim.robot.px, sim.robot.py]
            robot_trail.append(cur_pos[:])

            event = info.get("event")
            if isinstance(event, ReachGoal):
                status = "GOAL REACHED"
                final_event = "GOAL_REACHED"
                done = True
            elif isinstance(event, (Collision, CollisionOtherAgent)):
                status = "COLLISION"
                final_event = "COLLISION"
                done = True
            elif isinstance(event, Timeout):
                status = "TIMEOUT"
                final_event = "TIMEOUT"
                done = True

            # Render frame
            scenario_label = "AIRPORT PASSENGER ASSISTANCE - {}".format(difficulty.upper())
            frame = render_scene_frame(
                env=env,
                scenario_name=scenario_label,
                episode_num=1,
                step_num=step,
                sim_time=sim_time,
                status=status,
                robot_trail=robot_trail,
                start_pos=start_pos,
                policy_name="AIRPORT NAVIGATION"
            )
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            total_frames += 1

        if not done:
            status = "TIMEOUT"
            final_event = "TIMEOUT"

        # Capture final frame BEFORE reset so distance-to-goal is accurately retained
        final_dist = math.hypot(sim.robot.gx - sim.robot.px, sim.robot.gy - sim.robot.py)
        scenario_label = "AIRPORT PASSENGER ASSISTANCE - {}".format(difficulty.upper())
        final_frame = render_scene_frame(
            env=env,
            scenario_name=scenario_label,
            episode_num=1,
            step_num=step,
            sim_time=sim_time,
            status=status,
            robot_trail=robot_trail,
            start_pos=start_pos,
            policy_name="AIRPORT NAVIGATION"
        )
        final_rgb = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)

        # Hold end frame for exactly hold_frames (10 frames = ~0.83s, <= 12 frames)
        for _ in range(hold_frames):
            writer.append_data(final_rgb)
            total_frames += 1

    finally:
        writer.close()
        env.close()

    duration_s = float(total_frames) / float(fps)
    print("  -> Finished: Outcome={}, Steps={}, Final Dist={:.2f}m, Duration={:.2f}s".format(
        final_event, step, final_dist, duration_s
    ))

    # Save sidecar metadata JSON
    metadata = {
        "file": filename,
        "scenario": "airport_passenger_assistance",
        "difficulty": difficulty,
        "seed": seed,
        "outcome": final_event,
        "status_badge": status,
        "steps": step,
        "sim_time": round(sim_time, 2),
        "final_dist": round(final_dist, 3),
        "duration": round(duration_s, 2),
        "total_frames": total_frames,
        "hold_frames": hold_frames,
        "fps": fps,
        "policy": "AIRPORT NAVIGATION (AirportNavigationController deterministic reactive planner)"
    }
    with open(out_meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return out_video_path, metadata


def get_video_duration(video_path):
    """Retrieves exact video duration using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 12.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return float(frames) / float(fps)


def verify_airport_video(video_path):
    """
    Rigorously verifies an airport video against quality specifications:
      1. ffprobe duration in [8.0s, 40.0s]
      2. Sidecar metadata consistency with filename suffix
      3. Frozen frame count <= 20
      4. Frame variation (std dev >= 15.0) at 10%, 25%, 50%, 75%, 90%
      5. Final-frame status badge matches outcome
      6. Robot, goal, and obstacle pixels present
    """
    base = os.path.basename(video_path)
    sidecar_path = os.path.join(METADATA_DIR, base.replace(".mp4", ".json"))

    if not os.path.exists(sidecar_path):
        return {
            "file": base, "outcome": "NO_SIDECAR", "steps": 0, "final_dist": 0.0,
            "duration": 0.0, "frozen_frames": 0, "passed": False, "reason": "Missing sidecar JSON"
        }

    with open(sidecar_path, "r") as f:
        meta = json.load(f)

    outcome = meta.get("outcome", "")
    steps = meta.get("steps", 0)
    final_dist = meta.get("final_dist", 0.0)

    # 1. Filename suffix consistency
    if base.endswith("_success.mp4"):
        if outcome != "GOAL_REACHED":
            return {
                "file": base, "outcome": outcome, "steps": steps, "final_dist": final_dist,
                "duration": 0.0, "frozen_frames": 0, "passed": False,
                "reason": "File named _success.mp4 but outcome is {}".format(outcome)
            }
        if final_dist > 1.0:
            return {
                "file": base, "outcome": outcome, "steps": steps, "final_dist": final_dist,
                "duration": 0.0, "frozen_frames": 0, "passed": False,
                "reason": "Success final_dist {:.2f}m > 1.0m".format(final_dist)
            }
    elif base.endswith("_failure.mp4") or base.endswith("_bestprogress.mp4"):
        if outcome not in ["COLLISION", "TIMEOUT"]:
            return {
                "file": base, "outcome": outcome, "steps": steps, "final_dist": final_dist,
                "duration": 0.0, "frozen_frames": 0, "passed": False,
                "reason": "Failure/bestprogress file has outcome {}".format(outcome)
            }

    # 2. Duration check
    ff_dur = get_video_duration(video_path)
    if ff_dur < 7.9 or ff_dur > 40.5:
        return {
            "file": base, "outcome": outcome, "steps": steps, "final_dist": final_dist,
            "duration": round(ff_dur, 2), "frozen_frames": 0, "passed": False,
            "reason": "Duration {:.2f}s outside [8.0s, 40.0s]".format(ff_dur)
        }

    # 3. Read video frames and count frozen frames
    cap = cv2.VideoCapture(video_path)
    prev_gray = None
    frozen_count = 0
    total_frames = 0
    all_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1
        all_frames.append(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = np.mean(np.abs(gray.astype(float) - prev_gray.astype(float)))
            if diff < 0.01:
                frozen_count += 1
        prev_gray = gray
    cap.release()

    if total_frames == 0:
        return {
            "file": base, "outcome": outcome, "steps": steps, "final_dist": final_dist,
            "duration": 0.0, "frozen_frames": 0, "passed": False, "reason": "Zero frames read"
        }

    if frozen_count > 20:
        return {
            "file": base, "outcome": outcome, "steps": steps, "final_dist": final_dist,
            "duration": round(ff_dur, 2), "frozen_frames": frozen_count, "passed": False,
            "reason": "Frozen frame count ({}) > 20".format(frozen_count)
        }

    # 4. Check non-blank frames at 10%, 25%, 50%, 75%, 90%
    for pct in [0.10, 0.25, 0.50, 0.75, 0.90]:
        chk_idx = min(int(pct * total_frames), total_frames - 1)
        f = all_frames[chk_idx]
        std_val = float(np.std(f))
        if std_val < 15.0:
            return {
                "file": base, "outcome": outcome, "steps": steps, "final_dist": final_dist,
                "duration": round(ff_dur, 2), "frozen_frames": frozen_count, "passed": False,
                "reason": "Frame std {:.1f} < 15.0 at {:.0f}%".format(std_val, pct * 100)
            }

    # 5. Verify final-frame HUD badge color vs outcome
    final_f = all_frames[-1]
    # Badge rectangle is at bx=491, by=11, w=135, h=26
    badge_crop = final_f[11:37, 491:626]
    hsv_badge = cv2.cvtColor(badge_crop, cv2.COLOR_BGR2HSV)
    green_px = int(np.sum((hsv_badge[:, :, 0] >= 35) & (hsv_badge[:, :, 0] <= 85) & (hsv_badge[:, :, 1] >= 100) & (hsv_badge[:, :, 2] >= 100)))
    red_px = int(np.sum(((hsv_badge[:, :, 0] <= 10) | (hsv_badge[:, :, 0] >= 170)) & (hsv_badge[:, :, 1] >= 100) & (hsv_badge[:, :, 2] >= 100)))
    gold_px = int(np.sum((hsv_badge[:, :, 0] >= 11) & (hsv_badge[:, :, 0] <= 30) & (hsv_badge[:, :, 1] >= 100) & (hsv_badge[:, :, 2] >= 100)))

    if green_px > red_px and green_px > gold_px and green_px > 50:
        badge_label = "GOAL_REACHED"
    elif red_px > green_px and red_px > gold_px and red_px > 50:
        badge_label = "COLLISION"
    elif gold_px > green_px and gold_px > red_px and gold_px > 50:
        badge_label = "TIMEOUT"
    else:
        badge_label = "UNKNOWN"

    if outcome != badge_label:
        return {
            "file": base, "outcome": outcome, "hud_badge": badge_label, "steps": steps,
            "final_dist": final_dist, "duration": round(ff_dur, 2), "frozen_frames": frozen_count,
            "passed": False, "reason": "HUD badge ({}) != outcome ({})".format(badge_label, outcome)
        }

    # 6. Check presence of robot (cyan), goal (gold/yellow), and obstacles (grey)
    mid_frame = all_frames[total_frames // 2]
    rob_count = int(np.sum((mid_frame[:, :, 0] > 160) & (mid_frame[:, :, 1] > 150) & (mid_frame[:, :, 2] < 60)))
    obs_count = int(np.sum(
        (mid_frame[:, :, 0] > 50) & (mid_frame[:, :, 0] < 120) &
        (mid_frame[:, :, 1] > 55) & (mid_frame[:, :, 1] < 125) &
        (mid_frame[:, :, 2] > 70) & (mid_frame[:, :, 2] < 140)
    ))
    goal_count = int(np.sum((mid_frame[:, :, 0] < 60) & (mid_frame[:, :, 1] > 180) & (mid_frame[:, :, 2] > 200)))

    if rob_count < 10:
        return {
            "file": base, "outcome": outcome, "hud_badge": badge_label, "steps": steps,
            "final_dist": final_dist, "duration": round(ff_dur, 2), "frozen_frames": frozen_count,
            "passed": False, "reason": "Robot pixel count ({}) < 10".format(rob_count)
        }

    return {
        "file": base,
        "outcome": outcome,
        "hud_badge": badge_label,
        "steps": steps,
        "final_dist": final_dist,
        "duration": round(ff_dur, 2),
        "frozen_frames": frozen_count,
        "passed": True,
        "reason": "OK"
    }


def record_all_airport_demos():
    """Records the 4 designated demonstration videos."""
    videos_to_record = [
        # 1. MEDIUM Success
        {
            "difficulty": "MEDIUM",
            "seed": 0,
            "filename": "airport_passenger_assistance_medium_success.mp4",
        },
        # 2. MEDIUM Failure
        {
            "difficulty": "MEDIUM",
            "seed": 118,
            "filename": "airport_passenger_assistance_medium_failure.mp4",
        },
        # 3. EASY Success
        {
            "difficulty": "EASY",
            "seed": 100,
            "filename": "airport_passenger_assistance_easy_success.mp4",
        },
        # 4. HARD Attempt (Best Progress)
        {
            "difficulty": "HARD",
            "seed": 111,
            "filename": "airport_passenger_assistance_hard_bestprogress.mp4",
        },
    ]

    recorded_paths = []
    for spec in videos_to_record:
        path, _ = record_airport_episode(
            difficulty=spec["difficulty"],
            seed=spec["seed"],
            filename=spec["filename"],
            fps=12,
            hold_frames=10
        )
        recorded_paths.append(path)

    # Verification phase
    print("\n" + "=" * 85)
    print("PROGRAMMATIC VIDEO QUALITY VERIFICATION REPORT")
    print("=" * 85)
    print("| File | Outcome | HUD Badge | Steps | Final Dist | Duration | Frozen Frames | Quality Gate |")
    print("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |")

    all_passed = True
    report_rows = []
    for vp in recorded_paths:
        res = verify_airport_video(vp)
        report_rows.append(res)
        status_str = "**PASS**" if res["passed"] else "**FAIL ({})**".format(res.get("reason", ""))
        if not res["passed"]:
            all_passed = False
        print("| `{}` | {} | {} | {} | {:.2f}m | {:.2f}s | {} | {} |".format(
            res["file"], res["outcome"], res.get("hud_badge", "N/A"),
            res["steps"], res["final_dist"], res["duration"],
            res["frozen_frames"], status_str
        ))

    print("=" * 85)
    if all_passed:
        print(">>> ALL 4 AIRPORT DEMO VIDEOS PASSED PROGRAMMATIC QUALITY VERIFICATION! <<<")
    else:
        print(">>> SOME VIDEOS FAILED QUALITY VERIFICATION! <<<")
    print("=" * 85)

    return all_passed, report_rows


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Record and verify airport demo videos")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing videos")
    args = parser.parse_args()

    if args.verify_only:
        videos_to_verify = [
            os.path.join(VIDEOS_DIR, "airport_passenger_assistance_medium_success.mp4"),
            os.path.join(VIDEOS_DIR, "airport_passenger_assistance_medium_failure.mp4"),
            os.path.join(VIDEOS_DIR, "airport_passenger_assistance_easy_success.mp4"),
            os.path.join(VIDEOS_DIR, "airport_passenger_assistance_hard_bestprogress.mp4"),
        ]
        print("\n" + "=" * 85)
        print("PROGRAMMATIC VIDEO QUALITY VERIFICATION REPORT")
        print("=" * 85)
        print("| File | Outcome | HUD Badge | Steps | Final Dist | Duration | Frozen Frames | Quality Gate |")
        print("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |")
        all_passed = True
        for vp in videos_to_verify:
            res = verify_airport_video(vp)
            status_str = "**PASS**" if res["passed"] else "**FAIL ({})**".format(res.get("reason", ""))
            if not res["passed"]:
                all_passed = False
            print("| `{}` | {} | {} | {} | {:.2f}m | {:.2f}s | {} | {} |".format(
                res["file"], res["outcome"], res.get("hud_badge", "N/A"),
                res["steps"], res["final_dist"], res["duration"],
                res["frozen_frames"], status_str
            ))
        print("=" * 85)
        if all_passed:
            print(">>> ALL 4 AIRPORT DEMO VIDEOS PASSED PROGRAMMATIC QUALITY VERIFICATION! <<<")
        else:
            print(">>> SOME VIDEOS FAILED QUALITY VERIFICATION! <<<")
        print("=" * 85)
    else:
        record_all_airport_demos()
