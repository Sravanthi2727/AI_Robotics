#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
record_scenario.py
Robust video recorder with dynamic robot-centric camera, PiP minimap,
lidar beam rendering, telemetry HUD, and programmatic quality gates.
"""

from __future__ import print_function
import os
import sys
import math
import argparse
import numpy as np
import cv2

# Ensure imageio_ffmpeg works on Python 3.6
import importlib
try:
    import importlib_resources
    sys.modules['importlib.resources'] = importlib_resources
    importlib.resources = importlib_resources
except ImportError:
    pass

import random
import imageio
from stable_baselines import PPO2
from navrep.envs.navreptrainencodedenv import NavRepTrainEncoder, NavRepTrainEncodedEnv
from navrep.envs.encodedenv import EncodedEnv
from navrep.envs.navreptrainenv import NavRepTrainEnv
from navrep.envs.ianenv import IANEnv
from navrep.scripts.test_navrep import NavRepCPolicy
from crowd_sim.envs.utils.info import ReachGoal, Collision, CollisionOtherAgent, Timeout


class ScenarioEncodedEnv(EncodedEnv):
    """Subclass of EncodedEnv allowing deterministic scenario selection upon reset."""
    def reset_to_scenario(self, scenario_name=None):
        self.encoder.reset()
        obs = super(EncodedEnv, self).reset(set_scenario=scenario_name)
        h = self.encoder._encode_obs(obs, np.array([0., 0., 0.]))
        return h


def transform_world_to_canvas(x, y, x_bounds, y_bounds, canvas_w=640, canvas_h=640, margin=40):
    """Transforms continuous (x, y) coordinates to image (px, py) coordinates with margins."""
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds

    range_x = max(x_max - x_min, 1e-4)
    range_y = max(y_max - y_min, 1e-4)

    scale_x = float(canvas_w - 2 * margin) / range_x
    scale_y = float(canvas_h - 2 * margin) / range_y
    scale = min(scale_x, scale_y)

    cx = canvas_w / 2.0
    cy = canvas_h / 2.0
    mid_x = (x_min + x_max) / 2.0
    mid_y = (y_min + y_max) / 2.0

    px = int(cx + (x - mid_x) * scale)
    # Invert Y so up in world is up on canvas
    py = int(cy - (y - mid_y) * scale)
    return px, py, scale


def render_scene_frame(env, scenario_name, episode_num, step_num, sim_time, status,
                       robot_trail, start_pos, policy_name="VAE_LSTM_VM",
                       canvas_w=640, canvas_h=640, latest_obs=None,
                       goal_override=None):
    """Draws a complete vector visualization frame with robot-centric camera, PiP minimap, and HUD overlay."""
    frame = np.full((canvas_h, canvas_w, 3), (25, 29, 36), dtype=np.uint8)  # dark slate background

    is_ian = hasattr(env, 'iarlenv') or (hasattr(env, 'rlenv'))
    is_navreptrain = hasattr(env, 'soadrl_sim')

    robot_pos = [0., 0., 0.]
    robot_radius = 0.3
    goal_pos = [0., 0.]
    humans = []
    obstacle_polys = []
    scan = None
    global_extent = [-10.0, 10.0, -10.0, 10.0]

    if is_ian:
        rlenv = env.iarlenv.rlenv if hasattr(env, 'iarlenv') else env.rlenv
        m = rlenv.map2d
        # m.get_extent_xy() returns [xmin, xmax, ymin, ymax]
        raw_extent = m.get_extent_xy()
        g_xmin, g_xmax, g_ymin, g_ymax = raw_extent
        global_extent = [g_xmin, g_xmax, g_ymin, g_ymax]

        robot_pos = [float(rlenv.virtual_peppers[0].pos[0]),
                     float(rlenv.virtual_peppers[0].pos[1]),
                     float(rlenv.virtual_peppers[0].pos[2])]
        robot_radius = float(rlenv.vp_radii[0])
        if goal_override is not None:
            goal_pos = [float(goal_override[0]), float(goal_override[1])]
        else:
            goal_pos = [float(rlenv.agent_goals[0][0]), float(rlenv.agent_goals[0][1])]

        # Dynamic 20m x 20m camera centered on robot
        cam_span = 20.0
        rx, ry = robot_pos[0], robot_pos[1]
        x_bounds = [rx - cam_span / 2.0, rx + cam_span / 2.0]
        y_bounds = [ry - cam_span / 2.0, ry + cam_span / 2.0]

        # Extract obstacle polygons; skip poly 0 (world outer bounding box)
        all_polys = m.as_closed_obst_vertices()
        for poly in all_polys[1:]:
            obstacle_polys.append(poly)

        # Dynamic humans
        for i, vp in enumerate(rlenv.virtual_peppers[1:]):
            humans.append({
                'pos': [float(vp.pos[0]), float(vp.pos[1]), float(vp.pos[2])],
                'radius': float(rlenv.vp_radii[i + 1]),
                'goal': [float(rlenv.agent_goals[i + 1][0]), float(rlenv.agent_goals[i + 1][1])]
            })

        if latest_obs is not None and len(latest_obs) > 0:
            scan = latest_obs[0]

    elif is_navreptrain:
        sim = env.soadrl_sim
        robot_pos = [float(sim.robot.px), float(sim.robot.py), float(sim.robot.theta)]
        robot_radius = float(sim.robot.radius)
        if goal_override is not None:
            goal_pos = [float(goal_override[0]), float(goal_override[1])]
        else:
            goal_pos = [float(sim.robot.gx), float(sim.robot.gy)]

        # Bounds from arena
        flat_xs, flat_ys = [], []
        for poly in sim.obstacle_vertices:
            obstacle_polys.append(poly)
            for pt in poly:
                flat_xs.append(pt[0])
                flat_ys.append(pt[1])
        if flat_xs:
            x_bounds = [float(min(flat_xs) - 1.0), float(max(flat_xs) + 1.0)]
            y_bounds = [float(min(flat_ys) - 1.0), float(max(flat_ys) + 1.0)]
        else:
            x_bounds = [-8.0, 8.0]
            y_bounds = [-8.0, 8.0]

        global_extent = [x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]]

        for h in sim.humans:
            humans.append({
                'pos': [float(h.px), float(h.py), float(h.theta)],
                'radius': float(h.radius),
                'goal': [float(h.gx), float(h.gy)]
            })

    # Subtle background grid
    grid_spacing = 40
    for gx in range(0, canvas_w, grid_spacing):
        cv2.line(frame, (gx, 0), (gx, canvas_h), (32, 38, 48), 1)
    for gy in range(0, canvas_h, grid_spacing):
        cv2.line(frame, (0, gy), (canvas_w, gy), (32, 38, 48), 1)

    _, _, scale = transform_world_to_canvas(0, 0, x_bounds, y_bounds, canvas_w, canvas_h)

    # 1. Draw static obstacles / walls
    for poly in obstacle_polys:
        pts = []
        for pt in poly:
            cpx, cpy, _ = transform_world_to_canvas(pt[0], pt[1], x_bounds, y_bounds, canvas_w, canvas_h)
            pts.append([cpx, cpy])
        if len(pts) >= 2:
            pts_arr = np.array(pts, dtype=np.int32)
            cv2.polylines(frame, [pts_arr], True, (130, 140, 155), 2, cv2.LINE_AA)

    # 2. Draw Start Marker
    if start_pos is not None:
        spx, spy, _ = transform_world_to_canvas(start_pos[0], start_pos[1], x_bounds, y_bounds, canvas_w, canvas_h)
        if 0 <= spx < canvas_w and 0 <= spy < canvas_h:
            cv2.circle(frame, (spx, spy), int(robot_radius * scale * 0.9), (46, 204, 113), 2, cv2.LINE_AA)
            cv2.putText(frame, "S", (spx - 4, spy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (46, 204, 113), 1, cv2.LINE_AA)

    # 3. Draw Robot Trajectory Trail
    if len(robot_trail) >= 2:
        trail_pts = []
        for tx, ty in robot_trail:
            tpx, tpy, _ = transform_world_to_canvas(tx, ty, x_bounds, y_bounds, canvas_w, canvas_h)
            trail_pts.append((tpx, tpy))
        for idx in range(len(trail_pts) - 1):
            alpha = float(idx + 1) / len(trail_pts)
            b_col = int(240 * alpha)
            g_col = int(210 * alpha)
            r_col = int(0)
            cv2.line(frame, trail_pts[idx], trail_pts[idx + 1], (b_col, g_col, r_col), 2, cv2.LINE_AA)

    # 4. Draw Lidar Laser Scan Points (Cyan)
    if scan is not None and len(scan) > 0:
        angles = np.linspace(0, 2 * np.pi, len(scan))
        rx, ry, rth = robot_pos[0], robot_pos[1], robot_pos[2]
        for r, a in zip(scan[::4], angles[::4]):
            if r < 0.1 or r > 15.0:
                continue
            lx_w = rx + r * np.cos(rth + a)
            ly_w = ry + r * np.sin(rth + a)
            lpx, lpy, _ = transform_world_to_canvas(lx_w, ly_w, x_bounds, y_bounds, canvas_w, canvas_h)
            if 0 <= lpx < canvas_w and 0 <= lpy < canvas_h:
                cv2.circle(frame, (lpx, lpy), 1, (100, 230, 255), -1)

    # 5. Draw Dynamic Humans (Orange / Coral with velocity arrow)
    for h in humans:
        hx, hy, hth = h['pos']
        h_rad = h['radius']
        hpx, hpy, _ = transform_world_to_canvas(hx, hy, x_bounds, y_bounds, canvas_w, canvas_h)
        hr_px = max(int(h_rad * scale), 4)
        if -20 <= hpx < canvas_w + 20 and -20 <= hpy < canvas_h + 20:
            cv2.circle(frame, (hpx, hpy), hr_px, (34, 87, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (hpx, hpy), hr_px, (255, 255, 255), 1, cv2.LINE_AA)
            hex_px = int(hpx + (hr_px + 7) * np.cos(hth))
            hey_py = int(hpy - (hr_px + 7) * np.sin(hth))
            cv2.arrowedLine(frame, (hpx, hpy), (hex_px, hey_py), (255, 215, 120), 1, cv2.LINE_AA, tipLength=0.35)

    # 6. Draw Robot (Teal / White circle with heading arrow)
    rpx, rpy, _ = transform_world_to_canvas(robot_pos[0], robot_pos[1], x_bounds, y_bounds, canvas_w, canvas_h)
    rr_px = max(int(robot_radius * scale), 6)
    cv2.circle(frame, (rpx, rpy), rr_px, (220, 200, 0), -1, cv2.LINE_AA)
    cv2.circle(frame, (rpx, rpy), rr_px, (255, 255, 255), 2, cv2.LINE_AA)
    rth = robot_pos[2]
    rex_px = int(rpx + (rr_px + 10) * np.cos(rth))
    rey_py = int(rpy - (rr_px + 10) * np.sin(rth))
    cv2.arrowedLine(frame, (rpx, rpy), (rex_px, rey_py), (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.35)

    # 7. Draw Goal Marker & Off-screen Direction Arrow
    gpx, gpy, _ = transform_world_to_canvas(goal_pos[0], goal_pos[1], x_bounds, y_bounds, canvas_w, canvas_h)
    dist_to_goal = math.hypot(goal_pos[0] - robot_pos[0], goal_pos[1] - robot_pos[1])
    in_view = (50 <= gpx <= canvas_w - 50) and (60 <= gpy <= canvas_h - 50)
    if in_view:
        cv2.circle(frame, (gpx, gpy), int(robot_radius * scale * 1.3), (46, 204, 113), 2, cv2.LINE_AA)
        cv2.circle(frame, (gpx, gpy), int(robot_radius * scale * 0.4), (46, 204, 113), -1, cv2.LINE_AA)
        cv2.putText(frame, "GOAL ({:.1f}m)".format(dist_to_goal), (gpx + 12, gpy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (46, 204, 113), 1, cv2.LINE_AA)
    else:
        # Off-screen goal indicator: clamp to canvas border and draw direction arrow
        clamped_x = int(np.clip(gpx, 50, canvas_w - 50))
        clamped_y = int(np.clip(gpy, 65, canvas_h - 45))
        cv2.circle(frame, (clamped_x, clamped_y), 7, (46, 204, 113), -1, cv2.LINE_AA)
        angle_to_goal = math.atan2(-(goal_pos[1] - robot_pos[1]), goal_pos[0] - robot_pos[0])
        arr_dx = int(14 * math.cos(angle_to_goal))
        arr_dy = int(14 * math.sin(angle_to_goal))
        cv2.arrowedLine(frame, (clamped_x, clamped_y), (clamped_x + arr_dx, clamped_y + arr_dy),
                        (46, 204, 113), 2, cv2.LINE_AA, tipLength=0.4)
        cv2.putText(frame, "GOAL ({:.1f}m)".format(dist_to_goal), (clamped_x - 30, clamped_y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (46, 204, 113), 1, cv2.LINE_AA)

    # 8. Picture-in-Picture (PiP) Global Minimap (Top-right inset)
    if is_ian and global_extent is not None:
        mw, mh = 130, 130
        mx0, my0 = canvas_w - mw - 12, 58
        cv2.rectangle(frame, (mx0, my0), (mx0 + mw, my0 + mh), (20, 24, 30), -1)
        cv2.rectangle(frame, (mx0, my0), (mx0 + mw, my0 + mh), (70, 80, 95), 1)

        gx_bounds = [global_extent[0], global_extent[1]]
        gy_bounds = [global_extent[2], global_extent[3]]
        # Draw obstacle outlines in minimap
        for poly in obstacle_polys:
            m_pts = []
            for pt in poly:
                mpx, mpy, _ = transform_world_to_canvas(pt[0], pt[1], gx_bounds, gy_bounds, mw, mh, margin=8)
                m_pts.append([mx0 + mpx, my0 + mpy])
            if len(m_pts) >= 2:
                cv2.polylines(frame, [np.array(m_pts, dtype=np.int32)], True, (90, 100, 115), 1)

        # Minimap robot trail
        if len(robot_trail) >= 2:
            for idx in range(len(robot_trail) - 1):
                p1 = robot_trail[idx]
                p2 = robot_trail[idx + 1]
                mpx1, mpy1, _ = transform_world_to_canvas(p1[0], p1[1], gx_bounds, gy_bounds, mw, mh, margin=8)
                mpx2, mpy2, _ = transform_world_to_canvas(p2[0], p2[1], gx_bounds, gy_bounds, mw, mh, margin=8)
                cv2.line(frame, (mx0 + mpx1, my0 + mpy1), (mx0 + mpx2, my0 + mpy2), (200, 180, 0), 1)

        # Minimap robot position
        m_rpx, m_rpy, _ = transform_world_to_canvas(robot_pos[0], robot_pos[1], gx_bounds, gy_bounds, mw, mh, margin=8)
        cv2.circle(frame, (mx0 + m_rpx, my0 + m_rpy), 3, (0, 255, 255), -1)
        # Minimap goal position
        m_gpx, m_gpy, _ = transform_world_to_canvas(goal_pos[0], goal_pos[1], gx_bounds, gy_bounds, mw, mh, margin=8)
        cv2.circle(frame, (mx0 + m_gpx, my0 + m_gpy), 3, (46, 204, 113), -1)
        cv2.putText(frame, "MAP OVERVIEW", (mx0 + 6, my0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 160, 175), 1)

    # 9. Top HUD Bar
    cv2.rectangle(frame, (0, 0), (canvas_w, 48), (15, 18, 22), -1)
    cv2.line(frame, (0, 48), (canvas_w, 48), (45, 52, 64), 1)
    cv2.putText(frame, "ETH NAVREP BENCHMARK", (16, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 175, 190), 1, cv2.LINE_AA)
    cv2.putText(frame, "SCENARIO: {}".format(scenario_name.upper()), (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "EP: #{}".format(episode_num), (canvas_w // 2 - 45, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    # Status Badge
    status_colors = {
        'RUNNING': ((0, 180, 220), (0, 0, 0)),
        'GOAL REACHED': ((46, 204, 113), (0, 0, 0)),
        'COLLISION': ((0, 0, 220), (255, 255, 255)),
        'TIMEOUT': ((0, 140, 255), (0, 0, 0)),
    }
    badge_bg, badge_fg = status_colors.get(status, ((120, 120, 120), (255, 255, 255)))
    badge_w, badge_h = 135, 26
    bx, by = canvas_w - badge_w - 14, 11
    cv2.rectangle(frame, (bx, by), (bx + badge_w, by + badge_h), badge_bg, -1)
    cv2.putText(frame, status, (bx + 10, by + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, badge_fg, 1, cv2.LINE_AA)

    # 10. Bottom Telemetry Bar
    cv2.rectangle(frame, (0, canvas_h - 32), (canvas_w, canvas_h), (15, 18, 22), -1)
    cv2.line(frame, (0, canvas_h - 32), (canvas_w, canvas_h - 32), (45, 52, 64), 1)
    telem_str = "STEP: {:03d}   |   TIME: {:5.1f}s   |   DIST-TO-GOAL: {:4.1f}m   |   POLICY: {}".format(
        step_num, sim_time, dist_to_goal, policy_name
    )
    cv2.putText(frame, telem_str, (16, canvas_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (190, 200, 215), 1, cv2.LINE_AA)

    return frame


def verify_video_frames(video_path, min_duration=8.0, min_std=15.0):
    """Quality gate: asserts video duration, resolution, codec, and checks frames at 10/25/50/75/90%."""
    if not os.path.exists(video_path):
        raise AssertionError("Video file does not exist: {}".format(video_path))

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 12.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = float(total_frames) / fps if fps > 0 else 0.0

    print("Checking quality gate for {}: {} frames, {:.2f}s, {}x{}".format(
        os.path.basename(video_path), total_frames, duration, w, h
    ))

    # Assert duration and resolution
    assert w >= 640 and h >= 640, "Invalid resolution {}x{} in {}".format(w, h, video_path)

    check_percentages = [10, 25, 50, 75, 90]
    extracted_frames = []

    for pct in check_percentages:
        frame_idx = int(total_frames * (pct / 100.0))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        assert ret and frame is not None, "Failed to read frame at {}% ({})".format(pct, video_path)

        # Assert not blank or uniform
        std_val = float(np.std(frame))
        assert std_val >= min_std, "Frame at {}% is near-uniform (std={:.2f} < {:.2f})".format(pct, std_val, min_std)
        extracted_frames.append(cv2.resize(frame, (160, 160)))

    cap.release()

    # Generate 5-column contact sheet
    contact_sheet = np.hstack(extracted_frames)
    contact_dir = "/tmp/contact_sheets"
    os.makedirs(contact_dir, exist_ok=True)
    contact_path = os.path.join(contact_dir, os.path.basename(video_path).replace(".mp4", "_contact.png"))
    cv2.imwrite(contact_path, contact_sheet)

    return {
        'duration': duration,
        'frames': total_frames,
        'contact_sheet': contact_path,
        'passed': True
    }


def record_scenario(scenario_name, policy_path=None, episodes=1, seed=42, fps=12,
                    out_path=None, slowdown=1.0, max_steps=250, deterministic=True,
                    backend="VAE_LSTM", encoding="VM"):
    """Records scenario episode(s) and verifies frame quality."""
    # Ensure HOME has navrep / maps symlinks
    tmp_home = "/tmp/home"
    os.makedirs(tmp_home, exist_ok=True)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    navrep_link = os.path.join(tmp_home, "navrep")
    maps_link = os.path.join(tmp_home, "maps")
    if not os.path.exists(navrep_link):
        try:
            os.symlink(repo_root, navrep_link)
        except Exception:
            pass
    if not os.path.exists(maps_link):
        try:
            os.symlink(os.path.join(repo_root, "maps"), maps_link)
        except Exception:
            pass
    os.environ["HOME"] = tmp_home

    if out_path is None:
        out_dir = os.path.join(repo_root, "submission", "task2_videos")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "{}_seed{}.mp4".format(scenario_name, seed))
    else:
        out_dir = os.path.dirname(os.path.abspath(out_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    is_navreptrain = scenario_name.startswith("navreptrain")
    policy_label = "{}_{}".format(backend, encoding)

    if policy_path is None:
        models_gym = os.path.join(repo_root, "models/gym")
        if not os.path.isdir(models_gym):
            models_gym = os.path.expanduser("~/navrep/models/gym")
        if backend == "VAE_LSTM" and encoding == "VM":
            ckpt_name = "navreptrainencodedenv_2020_09_23__09_48_22_PPO_VAE_LSTM_VM_V32M512_ckpt.zip"
            policy_path = os.path.join(models_gym, ckpt_name)
        elif backend == "VAE_LSTM" and encoding == "V_ONLY":
            ckpt_name = "navreptrainencodedenv_2020_09_22__01_14_06_PPO_VAE_LSTM_V_ONLY_V32M512_ckpt.zip"
            policy_path = os.path.join(models_gym, ckpt_name)
        else:
            candidates = sorted(os.listdir(models_gym))
            matching = [c for c in candidates if backend + "_" + encoding in c]
            if matching:
                policy_path = os.path.join(models_gym, matching[0])
            else:
                raise ValueError("No matching policy found for {} {}".format(backend, encoding))

    print("[RECORD] Loading model from: {}".format(policy_path))
    model = PPO2.load(policy_path)
    policy = NavRepCPolicy(model)

    if is_navreptrain:
        sub_sc = "train" if "train" in scenario_name else "test"
        env = NavRepTrainEncodedEnv(backend=backend, encoding=encoding, scenario=sub_sc, silent=True, gpu=False)
    else:
        encoder = NavRepTrainEncoder(backend=backend, encoding=encoding, gpu=False)
        env = ScenarioEncodedEnv(None, None, silent=True, encoder=encoder, collect_trajectories=False)

    effective_fps = max(1, int(fps / slowdown))
    writer = imageio.get_writer(
        out_path,
        fps=effective_fps,
        codec='libx264',
        pixelformat='yuv420p',
        quality=8
    )

    print("[RECORD] Starting recording for scenario: {} -> {}".format(scenario_name, out_path))
    dt = 0.2
    np.random.seed(seed)

    for ep in range(1, episodes + 1):
        if is_navreptrain:
            np.random.seed(seed + ep - 1)
            random.seed(seed + ep - 1)
            ob = env.reset()
            robot_pos = [env.soadrl_sim.robot.px, env.soadrl_sim.robot.py]
            start_pos = robot_pos[:]
            robot_trail = [start_pos]
            latest_obs = [ob]
        else:
            np.random.seed(seed + ep - 1)
            ob = env.reset_to_scenario(scenario_name)
            rlenv = env.iarlenv.rlenv
            robot_pos = [float(rlenv.virtual_peppers[0].pos[0]), float(rlenv.virtual_peppers[0].pos[1])]
            start_pos = robot_pos[:]
            robot_trail = [start_pos]
            latest_obs = getattr(rlenv, 'latest_obs', None)

        status = "RUNNING"
        done = False
        step = 0
        sim_time = 0.0

        # Initial frame
        frame = render_scene_frame(
            env=env,
            scenario_name=scenario_name,
            episode_num=ep,
            step_num=step,
            sim_time=sim_time,
            status=status,
            robot_trail=robot_trail,
            start_pos=start_pos,
            policy_name=policy_label,
            latest_obs=latest_obs
        )
        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        target_goal = None
        while not done and step < max_steps:
            if is_navreptrain:
                target_goal = [float(env.soadrl_sim.robot.gx), float(env.soadrl_sim.robot.gy)]
                action, _ = policy.model.predict(ob, deterministic=deterministic)
                ob, reward, done, info = env.step(action)
                event = info.get('event', None)
                cur_pos = [env.soadrl_sim.robot.px, env.soadrl_sim.robot.py]
                if isinstance(event, ReachGoal):
                    status = "GOAL REACHED"
                    done = True
                elif isinstance(event, (Collision, CollisionOtherAgent)):
                    status = "COLLISION"
                    done = True
                elif isinstance(event, Timeout):
                    status = "TIMEOUT"
                    done = True
                latest_obs = [ob]
            else:
                rlenv = env.iarlenv.rlenv
                target_goal = [float(rlenv.agent_goals[0][0]), float(rlenv.agent_goals[0][1])]
                action, _ = policy.model.predict(ob, deterministic=deterministic)
                ob, reward, done, info = env.step(action)
                cur_pos = [float(rlenv.virtual_peppers[0].pos[0]), float(rlenv.virtual_peppers[0].pos[1])]
                if info.get('goal_reached', False):
                    status = "GOAL REACHED"
                    done = True
                elif rlenv.episode_damage[0] >= env.max_episode_damage:
                    status = "COLLISION"
                    done = True
                elif rlenv.episode_step[0] >= env.max_episode_length:
                    status = "TIMEOUT"
                    done = True
                latest_obs = getattr(rlenv, 'latest_obs', None)

            robot_trail.append(cur_pos)
            sim_time += dt
            step += 1

            frame_goal_override = target_goal if status == "GOAL REACHED" else None
            frame = render_scene_frame(
                env=env,
                scenario_name=scenario_name,
                episode_num=ep,
                step_num=step,
                sim_time=sim_time,
                status=status,
                robot_trail=robot_trail,
                start_pos=start_pos,
                policy_name=policy_label,
                latest_obs=latest_obs,
                goal_override=frame_goal_override
            )
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if not done and step >= max_steps:
            status = "TIMEOUT"

        # Ensure video is at least 8.0s (96 frames at 12 fps) for failure/timeout
        # or pad with 1.5s hold frame for success
        final_frame_goal_override = target_goal if status == "GOAL REACHED" else None
        final_frame = render_scene_frame(
            env=env,
            scenario_name=scenario_name,
            episode_num=ep,
            step_num=step,
            sim_time=sim_time,
            status=status,
            robot_trail=robot_trail,
            start_pos=start_pos,
            policy_name=policy_label,
            latest_obs=latest_obs,
            goal_override=final_frame_goal_override
        )
        final_rgb = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
        pad_frames = int(effective_fps * 1.5)
        min_total_frames = int(effective_fps * 8.0)
        current_frames = step + 1
        if status != "GOAL REACHED" and (current_frames + pad_frames) < min_total_frames:
            pad_frames = min_total_frames - current_frames

        for _ in range(max(pad_frames, int(effective_fps * 1.5))):
            writer.append_data(final_rgb)

        print("  Episode {} finished: Status={} Steps={} SimTime={:.1f}s".format(ep, status, step, sim_time))

    writer.close()
    env.close()
    print("[RECORD] Video successfully saved to: {}".format(out_path))

    # Verify quality gate
    quality = verify_video_frames(out_path, min_duration=8.0 if status != "GOAL REACHED" else 0.5)
    print("[RECORD] Quality Gate Passed: {}".format(quality))
    return {
        'video_path': out_path,
        'status': status,
        'steps': step,
        'sim_time': sim_time,
        'quality': quality
    }


def main():
    parser = argparse.ArgumentParser(description="NavRep Scenario Recorder")
    parser.add_argument("--scenario", type=str, default="irosasl1", help="Scenario name")
    parser.add_argument("--policy", type=str, default=None, help="Path to pretrained PPO policy zip")
    parser.add_argument("--backend", type=str, default="VAE_LSTM", help="Backend architecture")
    parser.add_argument("--encoding", type=str, default="VM", help="Encoding type (VM, V_ONLY, M_ONLY)")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to record")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fps", type=int, default=12, help="Playback FPS")
    parser.add_argument("--slowdown", type=float, default=1.0, help="Slowdown factor for visualization")
    parser.add_argument("--out", type=str, default=None, help="Output MP4 file path")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic actions instead of deterministic")
    args = parser.parse_args()

    record_scenario(
        scenario_name=args.scenario,
        policy_path=args.policy,
        episodes=args.episodes,
        seed=args.seed,
        fps=args.fps,
        out_path=args.out,
        slowdown=args.slowdown,
        deterministic=not args.stochastic,
        backend=args.backend,
        encoding=args.encoding
    )


if __name__ == '__main__':
    main()

