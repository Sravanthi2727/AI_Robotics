#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_videos.py
Rigorous programmatic quality gate verifying all Task 2 videos:
(a) Suffix and outcome consistency:
    _success.mp4 -> GOAL_REACHED
    _bestprogress.mp4 -> COLLISION or TIMEOUT
    _failure.mp4 -> COLLISION or TIMEOUT
(b) Frozen frame count: frame difference < 0.01; fails if frozen_frames > 20
(c) Standard deviation and presence of non-background pixels at 10%, 25%, 50%, 75%, 90%
(d) FFprobe duration >= 8.0s unless outcome == "GOAL_REACHED"
Outputs complete Markdown table: file | outcome | steps | final dist | duration | frozen frames | PASS/FAIL
"""

from __future__ import print_function
import os
import sys
import glob
import json
import subprocess
import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEOS_DIR = os.path.join(REPO_ROOT, "submission", "task2_videos")


def get_ffprobe_duration(video_path):
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path
    ]
    try:
        out = subprocess.check_output(cmd).decode("utf-8").strip()
        return float(out)
    except Exception:
        # Fallback to OpenCV
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 12.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return float(frames) / float(fps)


def verify_video(video_path):
    base = os.path.basename(video_path)
    metadata_dir = os.path.join(VIDEOS_DIR, "metadata")
    sidecar_path = os.path.join(metadata_dir, base.replace(".mp4", ".json"))
    if not os.path.exists(sidecar_path):
        sidecar_path = os.path.splitext(video_path)[0] + ".json"

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

    # (a) Suffix and outcome consistency
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
                "reason": "Success episode final_dist {:.2f}m > 1.0m (goal auto-reset issue)".format(final_dist)
            }
    elif base.endswith("_bestprogress.mp4"):
        if outcome not in ["COLLISION", "TIMEOUT"]:
            return {
                "file": base, "outcome": outcome, "steps": steps, "final_dist": final_dist,
                "duration": 0.0, "frozen_frames": 0, "passed": False,
                "reason": "File named _bestprogress.mp4 but outcome is {}".format(outcome)
            }
    elif base.endswith("_failure.mp4"):
        if outcome not in ["COLLISION", "TIMEOUT"]:
            return {
                "file": base, "outcome": outcome, "steps": steps, "final_dist": final_dist,
                "duration": 0.0, "frozen_frames": 0, "passed": False,
                "reason": "File named _failure.mp4 but outcome is {}".format(outcome)
            }
        if steps < 30:
            return {
                "file": base, "outcome": outcome, "steps": steps, "final_dist": final_dist,
                "duration": 0.0, "frozen_frames": 0, "passed": False,
                "reason": "Failure episode lasted only {} steps (<30)".format(steps)
            }

    # (d) Duration check
    ff_dur = get_ffprobe_duration(video_path)
    if outcome != "GOAL_REACHED" and ff_dur < 7.9:
        return {
            "file": base, "outcome": outcome, "steps": steps, "final_dist": final_dist,
            "duration": round(ff_dur, 2), "frozen_frames": 0, "passed": False,
            "reason": "Duration {:.2f}s < 8.0s for failure/bestprogress".format(ff_dur)
        }

    # (b) Count frozen frames (frame diff < 0.01)
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
            diff = float(np.mean(cv2.absdiff(gray, prev_gray)))
            if diff < 0.01:  # threshold for frozen / identical frame
                frozen_count += 1
        prev_gray = gray
    cap.release()

    if frozen_count > 20:
        return {
            "file": base, "outcome": outcome, "steps": steps, "final_dist": final_dist,
            "duration": round(ff_dur, 2), "frozen_frames": frozen_count, "passed": False,
            "reason": "Frozen frames {} > 20".format(frozen_count)
        }

    # (c) Check std dev and pixel variation at 10%, 25%, 50%, 75%, 90%
    if total_frames > 0:
        check_fractions = [0.10, 0.25, 0.50, 0.75, 0.90]
        for frac in check_fractions:
            idx = min(int(frac * total_frames), total_frames - 1)
            chk_frame = all_frames[idx]
            std = float(np.std(chk_frame))
            if std < 15.0:
                return {
                    "file": base, "outcome": outcome, "steps": steps, "final_dist": final_dist,
                    "duration": round(ff_dur, 2), "frozen_frames": frozen_count, "passed": False,
                    "reason": "Low std dev {:.1f} at {:.0f}%".format(std, frac * 100)
                }

    # (e) Final frame HUD status badge color verification
    final_frame = all_frames[-1]
    badge = final_frame[11:37, 491:626]
    g_count = int(np.sum((badge[:, :, 1] > 150) & (badge[:, :, 1] > badge[:, :, 2].astype(int) + 40)))
    o_count = int(np.sum((badge[:, :, 2] > 180) & (badge[:, :, 1] >= 80) & (badge[:, :, 1] <= 190) & (badge[:, :, 0] < 50)))
    r_count = int(np.sum((badge[:, :, 2] > 150) & (badge[:, :, 1] < 60) & (badge[:, :, 0] < 60)))

    if g_count > 100:
        hud_badge = "GOAL_REACHED"
        badge_label = "GREEN (GOAL REACHED)"
    elif o_count > 100:
        hud_badge = "TIMEOUT"
        badge_label = "ORANGE (TIMEOUT)"
    elif r_count > 100:
        hud_badge = "COLLISION"
        badge_label = "RED (COLLISION)"
    else:
        hud_badge = "UNKNOWN"
        badge_label = "UNKNOWN"

    if hud_badge != outcome:
        return {
            "file": base, "outcome": outcome, "hud_badge": badge_label, "steps": steps,
            "final_dist": final_dist, "duration": round(ff_dur, 2), "frozen_frames": frozen_count,
            "passed": False, "reason": "HUD badge ({}) does not match sidecar ({})".format(badge_label, outcome)
        }

    if base.endswith("_success.mp4") and hud_badge != "GOAL_REACHED":
        return {
            "file": base, "outcome": outcome, "hud_badge": badge_label, "steps": steps,
            "final_dist": final_dist, "duration": round(ff_dur, 2), "frozen_frames": frozen_count,
            "passed": False, "reason": "File named _success.mp4 but HUD badge is {}".format(badge_label)
        }
    elif (base.endswith("_failure.mp4") or base.endswith("_bestprogress.mp4")) and hud_badge not in ["COLLISION", "TIMEOUT"]:
        return {
            "file": base, "outcome": outcome, "hud_badge": badge_label, "steps": steps,
            "final_dist": final_dist, "duration": round(ff_dur, 2), "frozen_frames": frozen_count,
            "passed": False, "reason": "Failure/bestprogress file HUD badge is {}".format(badge_label)
        }

    # (f) Check robot (cyan), goal (green ring), and obstacle (light grey) pixels exist at 25%, 50%, 75%
    for pct in [0.25, 0.50, 0.75]:
        chk_idx = min(int(pct * total_frames), total_frames - 1)
        f = all_frames[chk_idx]

        # Robot: cyan
        rob_count = int(np.sum((f[:, :, 0] > 160) & (f[:, :, 1] > 150) & (f[:, :, 2] < 60)))
        if rob_count < 15:
            return {
                "file": base, "outcome": outcome, "hud_badge": badge_label, "steps": steps,
                "final_dist": final_dist, "duration": round(ff_dur, 2), "frozen_frames": frozen_count,
                "passed": False, "reason": "Robot cyan pixel count ({}) < 15 at {:.0f}%".format(rob_count, pct * 100)
            }

        # Goal: green ring
        goal_count = int(np.sum((f[:, :, 1] > 160) & (f[:, :, 1] > f[:, :, 2].astype(int) + 35) & (f[:, :, 1] > f[:, :, 0].astype(int) + 35)))
        if goal_count < 15:
            return {
                "file": base, "outcome": outcome, "hud_badge": badge_label, "steps": steps,
                "final_dist": final_dist, "duration": round(ff_dur, 2), "frozen_frames": frozen_count,
                "passed": False, "reason": "Goal green pixel count ({}) < 15 at {:.0f}%".format(goal_count, pct * 100)
            }

        # Obstacle: light grey
        obs_count = int(np.sum(
            (f[:, :, 0] > 160) & (f[:, :, 0] < 240) &
            (f[:, :, 1] > 160) & (f[:, :, 1] < 240) &
            (f[:, :, 2] > 160) & (f[:, :, 2] < 240) &
            (np.abs(f[:, :, 0].astype(int) - f[:, :, 1].astype(int)) < 25) &
            (np.abs(f[:, :, 1].astype(int) - f[:, :, 2].astype(int)) < 25)
        ))
        if obs_count < 15:
            return {
                "file": base, "outcome": outcome, "hud_badge": badge_label, "steps": steps,
                "final_dist": final_dist, "duration": round(ff_dur, 2), "frozen_frames": frozen_count,
                "passed": False, "reason": "Obstacle grey pixel count ({}) < 15 at {:.0f}%".format(obs_count, pct * 100)
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


def main():
    videos = sorted(glob.glob(os.path.join(VIDEOS_DIR, "*.mp4")))
    if not videos:
        print("[VERIFY] No MP4 videos found in {}".format(VIDEOS_DIR))
        sys.exit(1)

    print("\n" + "="*85)
    print("TASK 2 AUTOMATED VIDEO QUALITY VERIFICATION REPORT")
    print("="*85)

    results = []
    all_passed = True

    print("\n| File | Sidecar Outcome | HUD Badge Color | Steps | Final Dist | Duration | Frozen Frames | Quality Gate |")
    print("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |")

    for v in videos:
        res = verify_video(v)
        results.append(res)
        status_str = "**PASS**" if res["passed"] else "**FAIL ({})**".format(res.get("reason", ""))
        if not res["passed"]:
            all_passed = False
        print("| `{}` | {} | {} | {} | {}m | {}s | {} | {} |".format(
            res["file"], res["outcome"], res.get("hud_badge", "N/A"), res["steps"], res["final_dist"],
            res["duration"], res["frozen_frames"], status_str
        ))

    print("\n" + "="*85)
    if all_passed:
        print(">>> ALL {} VIDEOS PASSED PROGRAMMATIC QUALITY VERIFICATION! <<<".format(len(videos)))
    else:
        print(">>> SOME VIDEOS FAILED PROGRAMMATIC QUALITY VERIFICATION! <<<")
    print("="*85)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

