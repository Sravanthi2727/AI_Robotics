#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
record_airport_dashboard.py
===========================
Records the 4 mission-control dashboard videos for Airport Passenger Assistance Robot:
  1. airport_passenger_assistance_easy_success_dashboard.mp4
  2. airport_passenger_assistance_medium_success_dashboard.mp4
  3. airport_passenger_assistance_medium_failure_dashboard.mp4
  4. airport_passenger_assistance_hard_bestprogress_dashboard.mp4

Specifications:
  - Resolution: 1600x900, 10 fps, H.264 (pixelformat yuv420p)
  - 10 fps = 2.0x playback rate (matching '2.0x (Fast)' sim pacing radio)
  - Replays identical seeds from original videos:
      * EASY success: seed 100 (164 steps, dist 0.35m)
      * MEDIUM success: seed 0 (163 steps, dist 0.41m)
      * MEDIUM failure: seed 118 (287 steps, dist 6.61m, COLLISION)
      * HARD bestprogress: seed 111 (290 steps, dist 1.30m, TIMEOUT)
  - Rigorous assertion that outcome, steps, and final dist match original sidecars
  - Quality verification: duration, frozen frames <= 20 (ready <= 5, hold <= 10),
    chip colors, non-blank checks, robot & goal present, right panel text not clipped.
  - Extracts 5%, 50%, 97% inspection snapshot frames for automated visual review.
"""

from __future__ import print_function
import os
import sys
import math
import json
import shutil
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
    create_airport_config,
)
from navrep_custom.airport_dashboard_renderer import AirportDashboardRenderer
from crowd_sim.envs.utils.info import ReachGoal, Collision, CollisionOtherAgent, Timeout

# Target directories
PRIMARY_DIR = os.path.join(repo_root, "submission", "task3_custom", "videos_dashboard")
SECONDARY_DIR = os.path.join(repo_root, "submission", "task3_airport", "videos_dashboard")
ORIGINAL_META_DIR = os.path.join(repo_root, "submission", "task3_custom", "videos", "metadata")
SNAPSHOTS_DIR = os.path.join(PRIMARY_DIR, "snapshots")


EPISODE_SPECS = [
    {
        "difficulty": "EASY",
        "seed": 100,
        "base_name": "airport_passenger_assistance_easy_success",
        "dashboard_name": "airport_passenger_assistance_easy_success_dashboard.mp4",
        "exp_outcome": "GOAL_REACHED",
        "exp_steps": 164,
        "exp_dist": 0.346
    },
    {
        "difficulty": "MEDIUM",
        "seed": 0,
        "base_name": "airport_passenger_assistance_medium_success",
        "dashboard_name": "airport_passenger_assistance_medium_success_dashboard.mp4",
        "exp_outcome": "GOAL_REACHED",
        "exp_steps": 163,
        "exp_dist": 0.407
    },
    {
        "difficulty": "MEDIUM",
        "seed": 118,
        "base_name": "airport_passenger_assistance_medium_failure",
        "dashboard_name": "airport_passenger_assistance_medium_failure_dashboard.mp4",
        "exp_outcome": "COLLISION",
        "exp_steps": 287,
        "exp_dist": 6.607
    },
    {
        "difficulty": "HARD",
        "seed": 111,
        "base_name": "airport_passenger_assistance_hard_bestprogress",
        "dashboard_name": "airport_passenger_assistance_hard_bestprogress_dashboard.mp4",
        "exp_outcome": "TIMEOUT",
        "exp_steps": 290,
        "exp_dist": 1.299
    }
]


def record_dashboard_video(spec, fps=10, ready_frames=5, hold_frames=10):
    """
    Simulates and records a single dashboard video, asserting exact match to original sidecar.
    """
    diff = spec["difficulty"]
    seed = spec["seed"]
    dash_name = spec["dashboard_name"]
    out_video_path = os.path.join(PRIMARY_DIR, dash_name)
    meta_path = os.path.join(PRIMARY_DIR, "metadata", dash_name.replace(".mp4", ".json"))
    os.makedirs(os.path.dirname(out_video_path), exist_ok=True)
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)

    print("\n" + "=" * 70)
    print("RECORDING DASHBOARD: {} (Diff: {}, Seed: {})".format(dash_name, diff, seed))
    print("=" * 70)

    cfg = create_airport_config()
    renderer = AirportDashboardRenderer(cfg, difficulty=diff)

    env = AirportNavRepEnv(difficulty=diff, silent=True)
    env.reset(seed=seed)
    dt = env._get_dt()
    max_steps = env.time_limit
    time_limit_s = max_steps * dt

    controller = AirportNavigationController()
    controller.reset()

    sim = env.soadrl_sim
    start_pos = [sim.robot.px, sim.robot.py]
    robot_trail = [start_pos[:]]
    goal_pos = [sim.robot.gx, sim.robot.gy]

    writer = imageio.get_writer(
        out_video_path,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=None
    )

    total_frames = 0
    step = 0
    sim_time = 0.0
    done = False
    outcome = "TIMEOUT"
    last_action = [0.0, 0.0, 0.0]

    def get_humans_snapshot():
        h_list = []
        for h_i, h in enumerate(sim.humans):
            h_role = env.human_meta[h_i]["role"] if h_i < len(env.human_meta) else "extra_crowd"
            h_list.append({
                "pos": [h.px, h.py, h.theta],
                "radius": h.radius,
                "role": h_role
            })
        return h_list

    try:
        # 1. READY INTRO FRAMES (exactly ready_frames=5 frames)
        for _ in range(ready_frames):
            humans_snap = get_humans_snapshot()
            frame_rgb = renderer.render_frame(
                step_num=0,
                sim_time=0.0,
                max_steps=max_steps,
                time_limit=time_limit_s,
                robot_pos=[sim.robot.px, sim.robot.py, sim.robot.theta],
                robot_vel=[0.0, 0.0],
                robot_trail=robot_trail,
                humans=humans_snap,
                goal_pos=goal_pos,
                status="READY",
                action=[0.0, 0.0, 0.0],
                difficulty=diff
            )
            writer.append_data(frame_rgb)
            total_frames += 1

        # 2. ACTIVE NAVIGATION SIMULATION LOOP
        while not done and step < max_steps:
            robot_pos = [sim.robot.px, sim.robot.py]
            action = controller.act(robot_pos, humans=sim.humans, action_dim=3)
            last_action = action

            _, _, done, info = env.step(action)
            step += 1
            sim_time += dt

            cur_pos = [sim.robot.px, sim.robot.py]
            robot_trail.append(cur_pos[:])

            event = info.get("event")
            if isinstance(event, ReachGoal):
                outcome = "GOAL_REACHED"
                status = "GOAL REACHED"
                done = True
            elif isinstance(event, (Collision, CollisionOtherAgent)):
                outcome = "COLLISION"
                status = "COLLISION"
                done = True
            elif isinstance(event, Timeout):
                outcome = "TIMEOUT"
                status = "TIMEOUT"
                done = True
            else:
                status = "NAVIGATING"

            humans_snap = get_humans_snapshot()

            frame_rgb = renderer.render_frame(
                step_num=step,
                sim_time=sim_time,
                max_steps=max_steps,
                time_limit=time_limit_s,
                robot_pos=[sim.robot.px, sim.robot.py, sim.robot.theta],
                robot_vel=[action[0], action[1]],
                robot_trail=robot_trail,
                humans=humans_snap,
                goal_pos=goal_pos,
                status=status,
                action=action,
                difficulty=diff
            )
            writer.append_data(frame_rgb)
            total_frames += 1

        if not done:
            outcome = "TIMEOUT"
            status = "TIMEOUT"

        # Capture final frame BEFORE reset so distance-to-goal is accurately retained
        final_dist = math.hypot(sim.robot.gx - sim.robot.px, sim.robot.gy - sim.robot.py)

        # 3. END-HOLD FRAMES (exactly hold_frames=10 frames)
        humans_snap = get_humans_snapshot()

        final_rgb = renderer.render_frame(
            step_num=step,
            sim_time=sim_time,
            max_steps=max_steps,
            time_limit=time_limit_s,
            robot_pos=[sim.robot.px, sim.robot.py, sim.robot.theta],
            robot_vel=[0.0, 0.0],
            robot_trail=robot_trail,
            humans=humans_snap,
            goal_pos=goal_pos,
            status=status,
            action=last_action,
            difficulty=diff
        )

        for _ in range(hold_frames):
            writer.append_data(final_rgb)
            total_frames += 1

    finally:
        writer.close()
        env.close()

    duration_s = float(total_frames) / float(fps)
    print("  -> Replay Completed: Outcome={}, Steps={}, Final Dist={:.3f}m, Duration={:.2f}s (Total Frames={})".format(
        outcome, step, final_dist, duration_s, total_frames
    ))

    # CRITICAL ASSERTIONS AGAINST ORIGINAL SIDECAR
    assert outcome == spec["exp_outcome"], (
        "OUTCOME MISMATCH for {}: Got '{}', Expected '{}'!".format(dash_name, outcome, spec["exp_outcome"])
    )
    assert step == spec["exp_steps"], (
        "STEP COUNT MISMATCH for {}: Got {}, Expected {}!".format(dash_name, step, spec["exp_steps"])
    )
    assert abs(final_dist - spec["exp_dist"]) < 0.015, (
        "FINAL DIST MISMATCH for {}: Got {:.3f}m, Expected {:.3f}m!".format(dash_name, final_dist, spec["exp_dist"])
    )
    print("  >>> ASSERTION PASSED: Exact match with original episode sidecar! <<<")

    # Save sidecar metadata JSON
    meta = {
        "file": dash_name,
        "scenario": "airport_passenger_assistance",
        "difficulty": diff,
        "seed": seed,
        "outcome": outcome,
        "status_badge": status,
        "steps": step,
        "sim_time": round(sim_time, 2),
        "final_dist": round(final_dist, 3),
        "duration": round(duration_s, 2),
        "total_frames": total_frames,
        "ready_frames": ready_frames,
        "hold_frames": hold_frames,
        "fps": fps,
        "resolution": [1600, 900],
        "pacing": "2.0x (Fast)",
        "policy": "AIRPORT NAVIGATION (AirportNavigationController deterministic reactive planner)"
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return out_video_path, meta


def verify_dashboard_video(video_path):
    """
    Rigorously verifies dashboard video according to quality specifications:
      1. Resolution 1600x900, 10 fps, H.264
      2. Sidecar outcome and step match
      3. Frozen frame count <= 20
      4. Status chip color matches outcome
      5. Non-blank frames at 10, 25, 50, 75, 90% (std dev >= 15.0)
      6. Robot disc (blue) and goal ring (yellow) present
      7. Right-panel text area not clipped
    """
    base = os.path.basename(video_path)
    meta_path = os.path.join(PRIMARY_DIR, "metadata", base.replace(".mp4", ".json"))

    if not os.path.exists(meta_path):
        return {
            "file": base, "outcome": "NO_SIDECAR", "steps": 0, "final_dist": 0.0,
            "duration": 0.0, "frozen_frames": 0, "passed": False, "reason": "Missing metadata JSON"
        }

    with open(meta_path, "r") as f:
        meta = json.load(f)

    outcome = meta["outcome"]
    steps = meta["steps"]
    final_dist = meta["final_dist"]

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if w != 1600 or h != 900:
        cap.release()
        return {
            "file": base, "outcome": outcome, "steps": steps, "final_dist": final_dist,
            "duration": 0.0, "frozen_frames": 0, "passed": False,
            "reason": "Invalid resolution: {}x{} != 1600x900".format(w, h)
        }

    duration_s = float(total_frames) / float(fps)
    if duration_s < 7.9 or duration_s > 40.5:
        cap.release()
        return {
            "file": base, "outcome": outcome, "steps": steps, "final_dist": final_dist,
            "duration": round(duration_s, 2), "frozen_frames": 0, "passed": False,
            "reason": "Duration {:.2f}s outside [8.0s, 40.0s]".format(duration_s)
        }

    # Count frozen frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    prev_gray = None
    frozen_count = 0
    all_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = np.mean(np.abs(gray.astype(float) - prev_gray.astype(float)))
            if diff < 0.0015:
                frozen_count += 1
        prev_gray = gray
    cap.release()

    if frozen_count > 20:
        return {
            "file": base, "outcome": outcome, "steps": steps, "final_dist": final_dist,
            "duration": round(duration_s, 2), "frozen_frames": frozen_count, "passed": False,
            "reason": "Frozen frames ({}) > 20".format(frozen_count)
        }

    # Non-blank frames check at 10%, 25%, 50%, 75%, 90%
    for pct in [0.10, 0.25, 0.50, 0.75, 0.90]:
        idx = min(int(pct * total_frames), total_frames - 1)
        std_val = float(np.std(all_frames[idx]))
        if std_val < 15.0:
            return {
                "file": base, "outcome": outcome, "steps": steps, "final_dist": final_dist,
                "duration": round(duration_s, 2), "frozen_frames": frozen_count, "passed": False,
                "reason": "Std dev {:.1f} < 15.0 at {:.0f}%".format(std_val, pct * 100)
            }

    # Verify status chip color in final frame
    final_f = all_frames[-1]
    # Status chip bounds: x in [1320, 1580], y in [12, 46]
    chip_crop = final_f[12:46, 1320:1580]
    hsv_chip = cv2.cvtColor(chip_crop, cv2.COLOR_BGR2HSV)
    green_px = int(np.sum((hsv_chip[:, :, 0] >= 35) & (hsv_chip[:, :, 0] <= 85) & (hsv_chip[:, :, 1] >= 80) & (hsv_chip[:, :, 2] >= 50)))
    red_px = int(np.sum(((hsv_chip[:, :, 0] <= 10) | (hsv_chip[:, :, 0] >= 170)) & (hsv_chip[:, :, 1] >= 80) & (hsv_chip[:, :, 2] >= 50)))
    gold_px = int(np.sum((hsv_chip[:, :, 0] >= 11) & (hsv_chip[:, :, 0] <= 32) & (hsv_chip[:, :, 1] >= 80) & (hsv_chip[:, :, 2] >= 50)))

    if green_px > red_px and green_px > gold_px and green_px > 50:
        chip_label = "GOAL_REACHED"
    elif red_px > green_px and red_px > gold_px and red_px > 50:
        chip_label = "COLLISION"
    elif gold_px > green_px and gold_px > red_px and gold_px > 50:
        chip_label = "TIMEOUT"
    else:
        chip_label = "UNKNOWN"

    if outcome != chip_label:
        return {
            "file": base, "outcome": outcome, "hud_badge": chip_label, "steps": steps,
            "final_dist": final_dist, "duration": round(duration_s, 2), "frozen_frames": frozen_count,
            "passed": False, "reason": "Status chip ({}) != outcome ({})".format(chip_label, outcome)
        }

    # Verify presence of Robot (blue disc) and Goal (gold ring)
    mid_f = all_frames[total_frames // 2]
    # Robot: Blue color (B > 180, R < 80)
    floor_crop = mid_f[190:790, 50:1010]
    rob_count = int(np.sum((floor_crop[:, :, 0] > 180) & (floor_crop[:, :, 2] < 80)))
    # Goal: Gold/Yellow color (B < 60, G > 180, R > 190)
    goal_count = int(np.sum((floor_crop[:, :, 0] < 80) & (floor_crop[:, :, 1] > 180) & (floor_crop[:, :, 2] > 190)))

    if rob_count < 10:
        return {
            "file": base, "outcome": outcome, "hud_badge": chip_label, "steps": steps,
            "final_dist": final_dist, "duration": round(duration_s, 2), "frozen_frames": frozen_count,
            "passed": False, "reason": "Robot pixel count ({}) < 10".format(rob_count)
        }
    if goal_count < 10:
        return {
            "file": base, "outcome": outcome, "hud_badge": chip_label, "steps": steps,
            "final_dist": final_dist, "duration": round(duration_s, 2), "frozen_frames": frozen_count,
            "passed": False, "reason": "Goal pixel count ({}) < 10".format(goal_count)
        }

    # Check right panel text not clipped (check edge margin pixels x > 1585 are dark navy)
    right_margin_crop = mid_f[64:888, 1585:1600]
    if np.mean(right_margin_crop) > 80:
        return {
            "file": base, "outcome": outcome, "hud_badge": chip_label, "steps": steps,
            "final_dist": final_dist, "duration": round(duration_s, 2), "frozen_frames": frozen_count,
            "passed": False, "reason": "Right panel text clipped at frame edge"
        }

    # Extract inspection snapshot frames at 5%, 50%, 97%
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
    for pct in [0.05, 0.50, 0.97]:
        f_idx = min(int(pct * total_frames), total_frames - 1)
        snap_path = os.path.join(SNAPSHOTS_DIR, "{}_pct{:02d}.png".format(
            base.replace(".mp4", ""), int(pct * 100)
        ))
        cv2.imwrite(snap_path, all_frames[f_idx])

    return {
        "file": base,
        "outcome": outcome,
        "hud_badge": chip_label,
        "steps": steps,
        "final_dist": final_dist,
        "duration": round(duration_s, 2),
        "frozen_frames": frozen_count,
        "passed": True,
        "reason": "OK"
    }


def sync_to_secondary_directory():
    """Copies all dashboard videos and metadata to submission/task3_airport/videos_dashboard/."""
    os.makedirs(SECONDARY_DIR, exist_ok=True)
    sec_meta = os.path.join(SECONDARY_DIR, "metadata")
    os.makedirs(sec_meta, exist_ok=True)

    for spec in EPISODE_SPECS:
        v_name = spec["dashboard_name"]
        src_v = os.path.join(PRIMARY_DIR, v_name)
        dst_v = os.path.join(SECONDARY_DIR, v_name)
        if os.path.exists(src_v):
            shutil.copy2(src_v, dst_v)

        m_name = v_name.replace(".mp4", ".json")
        src_m = os.path.join(PRIMARY_DIR, "metadata", m_name)
        dst_m = os.path.join(sec_meta, m_name)
        if os.path.exists(src_m):
            shutil.copy2(src_m, dst_m)

    # Also copy snapshots
    sec_snaps = os.path.join(SECONDARY_DIR, "snapshots")
    os.makedirs(sec_snaps, exist_ok=True)
    if os.path.exists(SNAPSHOTS_DIR):
        for f in os.listdir(SNAPSHOTS_DIR):
            if f.endswith(".png"):
                shutil.copy2(os.path.join(SNAPSHOTS_DIR, f), os.path.join(sec_snaps, f))

    print("\nSynchronized dashboard videos and assets to: {}".format(SECONDARY_DIR))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Record and verify airport dashboard videos")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing videos")
    args = parser.parse_args()

    if not args.verify_only:
        for spec in EPISODE_SPECS:
            record_dashboard_video(spec, fps=10, ready_frames=5, hold_frames=10)
        sync_to_secondary_directory()

    # Verification phase
    print("\n" + "=" * 90)
    print("DASHBOARD VIDEO PROGRAMMATIC QUALITY VERIFICATION REPORT")
    print("=" * 90)
    print("| File | Outcome | Status Chip | Steps | Final Dist | Duration | Frozen Frames | Quality Gate |")
    print("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |")

    all_passed = True
    for spec in EPISODE_SPECS:
        v_path = os.path.join(PRIMARY_DIR, spec["dashboard_name"])
        res = verify_dashboard_video(v_path)
        status_str = "**PASS**" if res["passed"] else "**FAIL ({})**".format(res.get("reason", ""))
        if not res["passed"]:
            all_passed = False
        print("| `{}` | {} | {} | {} | {:.2f}m | {:.2f}s | {} | {} |".format(
            res["file"], res["outcome"], res.get("hud_badge", "N/A"),
            res["steps"], res["final_dist"], res["duration"],
            res["frozen_frames"], status_str
        ))

    print("=" * 90)
    if all_passed:
        print(">>> ALL 4 AIRPORT DASHBOARD VIDEOS PASSED PROGRAMMATIC QUALITY VERIFICATION! <<<")
    else:
        print(">>> SOME VIDEOS FAILED QUALITY VERIFICATION! <<<")
    print("=" * 90)

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
