#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
record_scenario.py
Reusable, non-invasive scenario recorder and visualizer for ETH NavRep environments.
Produces high-quality H.264 MP4 videos with HUD telemetry overlays, trajectory trails,
and goal markers.
"""

from __future__ import print_function
import os
import sys
import argparse
import time
import math
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


def transform_world_to_canvas(x, y, x_bounds, y_bounds, canvas_w=640, canvas_h=640, margin=50):
    """Transforms continuous (x, y) coordinates to image (px, py) coordinates with margins."""
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    
    range_x = max(x_max - x_min, 1e-4)
    range_y = max(y_max - y_min, 1e-4)
    
    scale_x = (canvas_w - 2 * margin) / range_x
    scale_y = (canvas_h - 2 * margin) / range_y
    scale = min(scale_x, scale_y)
    
    cx = (canvas_w) / 2.0
    cy = (canvas_h) / 2.0
    mid_x = (x_min + x_max) / 2.0
    mid_y = (y_min + y_max) / 2.0
    
    px = int(cx + (x - mid_x) * scale)
    # Invert Y so up in world is up on canvas
    py = int(cy - (y - mid_y) * scale)
    return px, py, scale


def render_scene_frame(env, scenario_name, episode_num, step_num, sim_time, status,
                       robot_trail, start_pos, policy_name="VAE_LSTM",
                       canvas_w=640, canvas_h=640):
    """Draws a complete vector visualization frame with HUD overlay."""
    frame = np.full((canvas_h, canvas_w, 3), (26, 30, 36), dtype=np.uint8)  # dark slate background
    
    # Draw subtle background grid
    grid_spacing = 40
    for gx in range(0, canvas_w, grid_spacing):
        cv2.line(frame, (gx, 0), (gx, canvas_h), (34, 40, 49), 1)
    for gy in range(0, canvas_h, grid_spacing):
        cv2.line(frame, (0, gy), (canvas_w, gy), (34, 40, 49), 1)

    is_ian = hasattr(env, 'iarlenv') or (hasattr(env, 'rlenv'))
    is_navreptrain = hasattr(env, 'soadrl_sim')

    robot_pos = [0., 0., 0.]
    robot_radius = 0.3
    robot_vel = [0., 0.]
    goal_pos = [0., 0.]
    humans = []
    x_bounds = [-10.0, 10.0]
    y_bounds = [-10.0, 10.0]
    obstacle_polys = []

    if is_ian:
        rlenv = env.iarlenv.rlenv if hasattr(env, 'iarlenv') else env.rlenv
        m = rlenv.map2d
        # Get map bounding extent
        x_min, y_min, x_max, y_max = m.get_extent_xy()
        x_bounds = [x_min, x_max]
        y_bounds = [y_min, y_max]
        
        # Obstacles
        for poly in m.as_closed_obst_vertices():
            obstacle_polys.append(poly)
            
        # Robot
        robot_pos = rlenv.virtual_peppers[0].pos
        robot_radius = rlenv.vp_radii[0]
        goal_pos = rlenv.agent_goals[0]
        
        # Humans
        for i, vp in enumerate(rlenv.virtual_peppers[1:]):
            humans.append({
                'pos': vp.pos,
                'radius': rlenv.vp_radii[i+1],
                'goal': rlenv.agent_goals[i+1]
            })

    elif is_navreptrain:
        sim = env.soadrl_sim
        flat_xs = []
        flat_ys = []
        for poly in sim.obstacle_vertices:
            obstacle_polys.append(poly)
            for pt in poly:
                flat_xs.append(pt[0])
                flat_ys.append(pt[1])
        if flat_xs:
            x_bounds = [float(min(flat_xs) - 1.0), float(max(flat_xs) + 1.0)]
            y_bounds = [float(min(flat_ys) - 1.0), float(max(flat_ys) + 1.0)]
        else:
            x_bounds = [-10.0, 10.0]
            y_bounds = [-10.0, 10.0]
        
        robot_pos = [sim.robot.px, sim.robot.py, sim.robot.theta]
        robot_radius = sim.robot.radius
        robot_vel = [sim.robot.vx, sim.robot.vy]
        goal_pos = [sim.robot.gx, sim.robot.gy]
        
        for h in sim.humans:
            humans.append({
                'pos': [h.px, h.py, h.theta],
                'radius': h.radius,
                'goal': [h.gx, h.gy]
            })

    _, _, scale = transform_world_to_canvas(0, 0, x_bounds, y_bounds, canvas_w, canvas_h)

    # 1. Draw static obstacles / walls
    for poly in obstacle_polys:
        pts = []
        for vx, vy in poly:
            cpx, cpy, _ = transform_world_to_canvas(vx, vy, x_bounds, y_bounds, canvas_w, canvas_h)
            pts.append([cpx, cpy])
        if len(pts) >= 3:
            pts_arr = np.array(pts, dtype=np.int32)
            cv2.fillPoly(frame, [pts_arr], (50, 56, 68))
            cv2.polylines(frame, [pts_arr], True, (90, 100, 120), 2)

    # 2. Draw robot start marker
    if start_pos is not None:
        spx, spy, _ = transform_world_to_canvas(start_pos[0], start_pos[1], x_bounds, y_bounds, canvas_w, canvas_h)
        cv2.circle(frame, (spx, spy), int(robot_radius * scale * 0.8), (46, 204, 113), 2)
        cv2.putText(frame, "S", (spx - 5, spy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (46, 204, 113), 1, cv2.LINE_AA)

    # 3. Draw goal marker
    gpx, gpy, _ = transform_world_to_canvas(goal_pos[0], goal_pos[1], x_bounds, y_bounds, canvas_w, canvas_h)
    cv2.circle(frame, (gpx, gpy), int(robot_radius * scale * 1.3), (255, 215, 0), 2)
    cv2.circle(frame, (gpx, gpy), int(robot_radius * scale * 0.5), (255, 215, 0), -1)
    cv2.line(frame, (gpx - 10, gpy), (gpx + 10, gpy), (255, 215, 0), 1)
    cv2.line(frame, (gpx, gpy - 10), (gpx, gpy + 10), (255, 215, 0), 1)
    cv2.putText(frame, "GOAL", (gpx + 12, gpy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 215, 0), 1, cv2.LINE_AA)

    # 4. Draw robot trajectory trail
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

    # 5. Draw target guidance ray
    rpx, rpy, _ = transform_world_to_canvas(robot_pos[0], robot_pos[1], x_bounds, y_bounds, canvas_w, canvas_h)
    cv2.line(frame, (rpx, rpy), (gpx, gpy), (80, 80, 80), 1, cv2.LINE_AA)

    # 6. Draw dynamic humans
    for h in humans:
        hx, hy, hth = h['pos']
        h_rad = h['radius']
        hpx, hpy, _ = transform_world_to_canvas(hx, hy, x_bounds, y_bounds, canvas_w, canvas_h)
        hr_px = max(int(h_rad * scale), 4)
        
        # Human body (Orange / Coral)
        cv2.circle(frame, (hpx, hpy), hr_px, (34, 87, 255), -1, cv2.LINE_AA)  # BGR: coral red
        cv2.circle(frame, (hpx, hpy), hr_px, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Heading direction
        hex_px = int(hpx + (hr_px + 7) * np.cos(hth))
        hey_py = int(hpy - (hr_px + 7) * np.sin(hth))
        cv2.arrowedLine(frame, (hpx, hpy), (hex_px, hey_py), (255, 255, 255), 1, tipLength=0.35)

    # 7. Draw Robot (Bright Teal / White)
    rr_px = max(int(robot_radius * scale), 6)
    cv2.circle(frame, (rpx, rpy), rr_px, (220, 200, 0), -1, cv2.LINE_AA)  # BGR: bright teal
    cv2.circle(frame, (rpx, rpy), rr_px, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Robot heading arrow
    rth = robot_pos[2]
    rex_px = int(rpx + (rr_px + 9) * np.cos(rth))
    rey_py = int(rpy - (rr_px + 9) * np.sin(rth))
    cv2.arrowedLine(frame, (rpx, rpy), (rex_px, rey_py), (0, 255, 255), 2, tipLength=0.35)

    # 8. Render HUD Banners
    # Top HUD Bar
    cv2.rectangle(frame, (0, 0), (canvas_w, 48), (15, 18, 22), -1)
    cv2.line(frame, (0, 48), (canvas_w, 48), (45, 52, 64), 1)

    cv2.putText(frame, "ETH NAVREP BENCHMARK", (16, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 175, 190), 1, cv2.LINE_AA)
    cv2.putText(frame, "SCENARIO: {}".format(scenario_name), (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    
    cv2.putText(frame, "EP: #{}".format(episode_num), (canvas_w // 2 - 40, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Status Pill Badge
    status_colors = {
        'RUNNING': ((0, 180, 220), (0, 0, 0)),        # Cyan
        'GOAL REACHED': ((46, 204, 113), (0, 0, 0)), # Green
        'COLLISION': ((0, 0, 220), (255, 255, 255)),  # Red
        'TIMEOUT': ((0, 140, 255), (0, 0, 0)),        # Orange
    }
    badge_bg, badge_fg = status_colors.get(status, ((120, 120, 120), (255, 255, 255)))
    badge_w, badge_h = 130, 26
    bx, by = canvas_w - badge_w - 14, 11
    cv2.rectangle(frame, (bx, by), (bx + badge_w, by + badge_h), badge_bg, -1)
    cv2.rectangle(frame, (bx, by), (bx + badge_w, by + badge_h), (255, 255, 255), 1)
    
    text_size = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
    tx = bx + (badge_w - text_size[0]) // 2
    ty = by + (badge_h + text_size[1]) // 2
    cv2.putText(frame, status, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, badge_fg, 1, cv2.LINE_AA)

    # Bottom Telemetry Bar
    cv2.rectangle(frame, (0, canvas_h - 40), (canvas_w, canvas_h), (15, 18, 22), -1)
    cv2.line(frame, (0, canvas_h - 40), (canvas_w, canvas_h - 40), (45, 52, 64), 1)

    dist_to_goal = math.sqrt((robot_pos[0] - goal_pos[0])**2 + (robot_pos[1] - goal_pos[1])**2)
    speed = math.sqrt(robot_vel[0]**2 + robot_vel[1]**2) if is_navreptrain else 0.0

    hud_str = "Step: {:04d} | Sim Time: {:5.1f}s | Dist to Goal: {:4.2f}m | Policy: {}".format(
        step_num, sim_time, dist_to_goal, policy_name
    )
    cv2.putText(frame, hud_str, (16, canvas_h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 210, 225), 1, cv2.LINE_AA)

    return frame


def record_scenario(scenario_name, policy_path=None, episodes=3, seed=42, fps=15,
                    out_path=None, slowdown=1.0, max_steps=600):
    """Executes episodes in the requested scenario and encodes an MP4 video."""
    np.random.seed(seed)
    
    # 1. Determine environment type
    is_navreptrain = scenario_name.startswith("navreptrain")
    
    if is_navreptrain:
        sub_scenario = scenario_name.replace("navreptrain_", "")
        env = NavRepTrainEnv(scenario=sub_scenario, silent=True)
        # Random action policy or PPO policy
        policy = None
        if policy_path and os.path.exists(policy_path):
            policy = PPO2.load(policy_path)
    else:
        encoder = NavRepTrainEncoder('VAE_LSTM', 'V_ONLY', gpu=False)
        env = ScenarioEncodedEnv(None, None, silent=True, encoder=encoder, collect_trajectories=True)
        if policy_path is None or not os.path.exists(policy_path):
            policy_path = os.path.expanduser(
                "~/navrep/models/gym/navreptrainencodedenv_2020_09_22__01_14_06_PPO_VAE_LSTM_V_ONLY_V32M512_ckpt.zip"
            )
        ppo_model = PPO2.load(policy_path)
        policy = NavRepCPolicy(ppo_model)

    if out_path is None:
        out_path = "submission/task2_videos/scenario_{}.mp4".format(scenario_name)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    effective_fps = max(int(fps / slowdown), 1)
    writer = imageio.get_writer(out_path, fps=effective_fps, codec='libx264', pixelformat='yuv420p')

    print("[RECORD] Starting recording for scenario: {} -> {}".format(scenario_name, out_path))
    dt = 0.2

    for ep in range(1, episodes + 1):
        if is_navreptrain:
            ob = env.reset()
            robot_pos = [env.soadrl_sim.robot.px, env.soadrl_sim.robot.py]
        else:
            ob = env.reset_to_scenario(scenario_name)
            robot_pos = env.iarlenv.rlenv.virtual_peppers[0].pos[:2].copy()

        start_pos = robot_pos.copy()
        robot_trail = [start_pos]
        status = "RUNNING"
        done = False
        step = 0
        sim_time = 0.0

        while not done and step < max_steps:
            if is_navreptrain:
                if policy:
                    action, _ = policy.predict(ob, deterministic=True)
                else:
                    action = np.clip(np.random.randn(3) * 0.3 + [0.5, 0.0, 0.0], -1, 1)
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
            else:
                action = policy.act(ob)
                ob, reward, done, info = env.step(action)
                cur_pos = env.iarlenv.rlenv.virtual_peppers[0].pos[:2].copy()
                if info.get('goal_reached', False):
                    status = "GOAL REACHED"
                    done = True
                elif env.iarlenv.rlenv.episode_damage[0] >= env.max_episode_damage:
                    status = "COLLISION"
                    done = True
                elif env.iarlenv.rlenv.episode_step[0] >= env.max_episode_length:
                    status = "TIMEOUT"
                    done = True

            robot_trail.append(cur_pos)
            sim_time += dt
            step += 1

            frame = render_scene_frame(
                env=env,
                scenario_name=scenario_name,
                episode_num=ep,
                step_num=step,
                sim_time=sim_time,
                status=status,
                robot_trail=robot_trail,
                start_pos=start_pos
            )
            # Write frame (convert BGR from OpenCV to RGB for imageio)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            writer.append_data(rgb_frame)

        if not done and step >= max_steps:
            status = "TIMEOUT"

        # Hold the final outcome frame for ~1.5 seconds (20-30 frames)
        final_frame = render_scene_frame(
            env=env,
            scenario_name=scenario_name,
            episode_num=ep,
            step_num=step,
            sim_time=sim_time,
            status=status,
            robot_trail=robot_trail,
            start_pos=start_pos
        )
        final_rgb = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
        for _ in range(int(effective_fps * 1.5)):
            writer.append_data(final_rgb)

        print("  Episode {} finished: Status={} Steps={} SimTime={:.1f}s".format(ep, status, step, sim_time))

    writer.close()
    env.close()
    print("[RECORD] Video successfully saved to: {}".format(out_path))
    return out_path


def main():
    parser = argparse.ArgumentParser(description="NavRep Scenario Recorder")
    parser.add_argument("--scenario", type=str, default="irosasl1", help="Scenario name")
    parser.add_argument("--policy", type=str, default=None, help="Path to pretrained PPO policy zip")
    parser.add_argument("--episodes", type=int, default=2, help="Number of episodes to record")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fps", type=int, default=15, help="Playback FPS")
    parser.add_argument("--slowdown", type=float, default=1.0, help="Slowdown factor for visualization")
    parser.add_argument("--out", type=str, default=None, help="Output MP4 file path")
    args = parser.parse_args()

    record_scenario(
        scenario_name=args.scenario,
        policy_path=args.policy,
        episodes=args.episodes,
        seed=args.seed,
        fps=args.fps,
        out_path=args.out,
        slowdown=args.slowdown
    )


if __name__ == '__main__':
    main()

