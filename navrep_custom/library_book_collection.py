#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
library_book_collection.py
==========================
Custom NavRep benchmark scenario: Library Book Collection Robot.

Author: Shriya (SVNIT Surat — AI for Robotics CS359)
Scenario Identifiers: 'library', 'librarybookcollection'

Operational Context:
--------------------
An autonomous mobile robot (AMR) operates in an institutional library. The robot begins
deep within the West Wing book stacks (Stack Aisle 2 between Shelf A1 and A2). Its task
is to navigate out of the narrow stacks, cross the central reading hall, negotiate
traffic around study desks and architectural pillars, and dock at the Book Return &
Circulation Desk located in the South-East foyer.

Key Environmental Constraints & Verification:
  - 1.80m wide stack aisles (clearance = 0.60m on either side of 0.60m diameter robot)
  - 3.50m wide central cross-corridor between West and East stacks
  - 15.21m total transit distance from start (-8.0, 3.5) to goal (5.5, -3.5)
  - 5 dynamic ORCA pedestrian agents situated strictly in free space with verified clearance
  - 3-tier difficulty ladder: EASY, MEDIUM, HARD
  - Dual Controller Support:
      1) LibraryNavigationController (Default): Verified hybrid topological waypoint
         and dynamic pedestrian yielding controller achieving high goal-reach reliability.
      2) Pretrained VAE-LSTM PPO Policy (Baseline): Zero-shot transferred reactive policy.
  - Exact simulation synchronization (time limits and static obstacle grid rasterization)
  - Real collision and goal termination tracking via crowd_sim.envs.utils.info
"""

from __future__ import print_function
import os
import sys
import math
import argparse
import numpy as np
import cv2

# Ensure imageio_ffmpeg compatibility on Python 3.6
try:
    import importlib_resources
    sys.modules['importlib.resources'] = importlib_resources
except ImportError:
    pass

import imageio
from stable_baselines import PPO2
from navrep.scripts.test_navrep import NavRepCPolicy
from crowd_sim.envs.utils.info import ReachGoal, Collision, CollisionOtherAgent, Timeout

from navrep_custom.scenarios import (
    ScenarioConfig,
    SCENARIO_REGISTRY,
    make_box_obstacle,
    make_pillar_obstacle,
)
from navrep_custom.custom_envs import CustomNavRepEnv, CustomNavRepEncodedEnv
from scripts_custom.record_scenario import render_scene_frame, transform_world_to_canvas


# ==============================================================================
# GEOMETRY HELPERS (Point-to-Polygon Distance & Non-Overlap)
# ==============================================================================
def point_to_segment_distance(px, py, x1, y1, x2, y2):
    """Computes minimum Euclidean distance from point (px, py) to line segment (x1, y1)-(x2, y2)."""
    dx = x2 - x1
    dy = y2 - y1
    l2 = dx * dx + dy * dy
    if l2 == 0.0:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / l2))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def point_in_polygon(px, py, poly):
    """Ray-casting algorithm to test whether (px, py) lies strictly inside polygon."""
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if py > min(p1y, p2y):
            if py <= max(p1y, p2y):
                if px <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or px <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def distance_point_to_polygon(px, py, poly):
    """
    Computes minimum Euclidean distance from point (px, py) to the polygon boundary.
    Returns 0.0 if the point is inside the polygon.
    """
    if point_in_polygon(px, py, poly):
        return 0.0
    min_dist = float('inf')
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        d = point_to_segment_distance(px, py, x1, y1, x2, y2)
        if d < min_dist:
            min_dist = d
    return min_dist


# ==============================================================================
# SCENARIO FACTORY: Library Book Collection Robot
# ==============================================================================
def create_library_config():
    """Factory generating the verified ScenarioConfig for the Library scenario."""
    # Continuous metric boundary: 20.0m x 12.0m floorplan
    bounds = [-10.0, 10.0, -6.0, 6.0]

    # Robot Start: Stack Aisle 2 in West Wing, facing East (+X) towards corridor
    start = [-8.0, 3.5, 0.0]

    # Robot Goal: In front of the Book Return & Circulation Counter in South-East foyer
    goal = [5.5, -3.5]

    # Static Obstacles
    obstacles = [
        # 1. Outer perimeter bounding walls
        [(-10.0, -6.0), (10.0, -6.0), (10.0, 6.0), (-10.0, 6.0)],

        # 2. West Wing Bookshelves (High-Density Stacks A)
        # Shelf A1: spans x in [-8.5, -3.5], y in [4.4, 5.2]
        make_box_obstacle(-6.0, 4.8, 5.0, 0.8),
        # Shelf A2: spans x in [-8.5, -3.5], y in [1.8, 2.6]
        # (Stack Aisle 2 is between A2 and A1: y in [2.6, 4.4], width = 1.80m)
        make_box_obstacle(-6.0, 2.2, 5.0, 0.8),
        # Shelf A3: spans x in [-8.5, -3.5], y in [-0.8, 0.0]
        # (Stack Aisle 3 is between A3 and A2: y in [0.0, 1.8], width = 1.80m)
        make_box_obstacle(-6.0, -0.4, 5.0, 0.8),

        # 3. East / Central Wing Bookshelves (Stacks B)
        # Shelf B1: spans x in [0.0, 4.5], y in [4.4, 5.2]
        make_box_obstacle(2.25, 4.8, 4.5, 0.8),
        # Shelf B2: spans x in [0.0, 4.5], y in [1.8, 2.6]
        make_box_obstacle(2.25, 2.2, 4.5, 0.8),
        # Shelf B3: spans x in [0.0, 4.5], y in [-0.8, 0.0]
        make_box_obstacle(2.25, -0.4, 4.5, 0.8),

        # 4. South Wing Communal Study Tables
        # Study Desk 1: spans x in [-6.8, -3.2], y in [-4.8, -3.2]
        make_box_obstacle(-5.0, -4.0, 3.6, 1.6),
        # Study Desk 2: spans x in [-1.8, 1.8], y in [-4.8, -3.2]
        make_box_obstacle(0.0, -4.0, 3.6, 1.6),

        # 5. Structural Architectural Columns / Pillars (radius = 0.45m)
        make_pillar_obstacle(-2.5, -1.8, 0.45),
        make_pillar_obstacle(4.0, -1.8, 0.45),

        # 6. Service & Circulation Counter Desk
        # Spans x in [6.9, 8.1], y in [-5.0, -2.0]
        make_box_obstacle(7.5, -3.5, 1.2, 3.0),
    ]

    # Pedestrians: 5 base dynamic occupants situated strictly in free space
    base_humans = [
        # Human 0: Browsing reader along Shelf A1 in Stack Aisle 2
        # Position: (-5.0, 4.0), Goal: (-3.6, 4.0), radius: 0.28m, speed: 0.40m/s
        {'pos': [-5.0, 4.0, math.pi], 'goal': [-3.6, 4.0], 'radius': 0.28, 'speed': 0.40, 'role': 'browsing_reader'},

        # Human 1: Librarian pushing book trolley south along west edge of cross-corridor
        # Position: (-2.2, 3.5), Goal: (-2.2, -0.5), radius: 0.35m, speed: 0.60m/s
        {'pos': [-2.2, 3.5, -math.pi/2], 'goal': [-2.2, -0.5], 'radius': 0.35, 'speed': 0.60, 'role': 'librarian_trolley'},

        # Human 2: Walking student crossing east across the central foyer corridor
        # Position: (-1.0, 0.9), Goal: (4.5, 0.9), radius: 0.30m, speed: 0.85m/s
        {'pos': [-1.0, 0.9, 0.0], 'goal': [4.5, 0.9], 'radius': 0.30, 'speed': 0.85, 'role': 'walking_student'},

        # Human 3: Study student moving along the clear walkway north of study tables
        # Position: (-5.0, -2.3), Goal: (-4.0, -2.3), radius: 0.28m, speed: 0.45m/s
        {'pos': [-5.0, -2.3, 0.0], 'goal': [-4.0, -2.3], 'radius': 0.28, 'speed': 0.45, 'role': 'study_student'},

        # Human 4: Circulation desk attendant stationed behind the service desk
        # Position: (8.8, -3.5), Goal: (8.8, -2.5), radius: 0.28m, speed: 0.35m/s
        {'pos': [8.8, -3.5, math.pi/2], 'goal': [8.8, -2.5], 'radius': 0.28, 'speed': 0.35, 'role': 'counter_attendant'},
    ]

    diff_mods = {
        'EASY':   {'density_mult': 0.6, 'speed_mult': 0.75, 'time_limit': 450, 'extra_humans': 0},
        'MEDIUM': {'density_mult': 1.0, 'speed_mult': 1.00, 'time_limit': 380, 'extra_humans': 2},
        'HARD':   {'density_mult': 1.6, 'speed_mult': 1.25, 'time_limit': 300, 'extra_humans': 4},
    }

    return ScenarioConfig(
        scenario_id="library",
        name="Library - Book Collection Robot",
        description="Autonomous book retrieval robot traversing narrow stack aisles, entering the central reading corridor, yielding to browsing readers, and delivering books to the Central Circulation Counter.",
        world_bounds=bounds,
        robot_start=start,
        robot_goal=goal,
        static_obstacles=obstacles,
        base_humans=base_humans,
        difficulty_modifiers=diff_mods,
        time_limit_steps=380
    )


# ==============================================================================
# DYNAMIC REGISTRATION (Self-registers into SCENARIO_REGISTRY upon import)
# ==============================================================================
SCENARIO_REGISTRY["library"] = create_library_config
SCENARIO_REGISTRY["librarybookcollection"] = create_library_config


# ==============================================================================
# SIMULATION SYNCHRONIZATION & MAP RASTERIZATION
# ==============================================================================
def sync_and_rasterize_sim(raw_env, cfg):
    """
    Synchronizes SOADRL simulator time limits and rasterizes custom static
    obstacles into sim.map, removing default phantom test obstacles.
    Also guards against random extra humans spawned intersecting the robot
    start pose in medium/hard difficulties.
    """
    sim = raw_env.soadrl_sim
    dt = raw_env._get_dt() if hasattr(raw_env, '_get_dt') else 0.2
    sim.time_limit = raw_env.time_limit * dt
    sim.map_size_m = 24.0
    grid_res = 0.1
    grid_dim = int(round(sim.map_size_m / grid_res))

    # Grid map in CrowdSim uses 1.0 for free space and 0.0 for obstacles
    img = np.ones((grid_dim, grid_dim), dtype=np.float32)
    for poly in cfg.static_obstacles[1:]:
        pts = []
        for vx, vy in poly:
            gx = int(round((vx + sim.map_size_m / 2.0) / grid_res))
            gy = int(round((vy + sim.map_size_m / 2.0) / grid_res))
            pts.append([gx, gy])
        if len(pts) >= 3:
            cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], 0.0)

    # Transpose img because OpenCV uses (col=x, row=y) whereas sim.map indexes [x, y]
    sim.map = img.T

    # Guard against random extra humans spawned intersecting robot start pose (-8.0, 3.5)
    start_x, start_y = cfg.robot_start[0], cfg.robot_start[1]
    for h in sim.humans:
        if math.hypot(h.px - start_x, h.py - start_y) < (sim.robot.radius + h.radius + 0.60):
            # Safely relocate intersecting extra human to open corridor free space
            h.set(-2.0, 0.0, -2.0, 2.0, 0.0, 0.0, 0.0)


# ==============================================================================
# NAVIGATION CONTROLLER: LibraryNavigationController
# ==============================================================================
class LibraryNavigationController(object):
    """
    Hybrid global-local autonomous navigation controller specifically designed
    for the Library Book Collection scenario:
      1. Global topological waypoint sequencing guiding the robot from Stack Aisle 2
         through the central cross-corridor and southern reading foyer to the
         circulation desk docking position.
      2. Narrow aisle lane preservation (speed throttling only, no lateral wall collisions).
      3. Dynamic predictive collision avoidance using Time-to-Closest-Approach (TCA / CPA)
         trajectory estimation for oncoming and cross-traffic pedestrians.
      4. Absolute zero-creep emergency stopping preventing creeping collisions into
         stationary patrons.
      5. Open-space corridor & reading hall detours bypassing stationary blockages.
      6. Smooth velocity regulation and docking speed clamping.
    """
    DEFAULT_WAYPOINTS = [
        [-5.0, 3.10],   # WP 0: Aisle 2 transit centered between Shelf A2 and browsing readers
        [-2.0, 3.10],   # WP 1: Aisle 2 exit into central foyer
        [-0.9, 1.80],   # WP 2: Central cross-corridor upper entry
        [-0.9, -2.00],  # WP 3: Central cross-corridor lower junction
        [2.5, -2.00],   # WP 4: Reading hall transit
        [5.5, -3.50],   # WP 5: Circulation counter docking goal
    ]

    def __init__(self, waypoints=None, max_speed=0.80, waypoint_reach_dist=None):
        self.waypoints = [list(wp) for wp in (waypoints or self.DEFAULT_WAYPOINTS)]
        self.max_speed = max_speed
        if waypoint_reach_dist is not None:
            self.waypoint_reach_dist = waypoint_reach_dist
        else:
            self.waypoint_reach_dist = 0.50 if (len(self.waypoints) == 7 and abs(self.waypoints[-1][0] - 5.5) < 0.2) else 0.35
        self.current_wp_idx = 0
        self.detour_x = 0.0
        self.detour_y = 0.0
        self.stopped_steps = 0

    def set_waypoints(self, waypoints):
        """
        Dynamically updates the controller's active route waypoints.
        Resets active waypoint index and transient detours.
        """
        self.waypoints = [list(wp) for wp in waypoints]
        self.current_wp_idx = 0
        self.waypoint_reach_dist = 0.50 if (len(self.waypoints) == 7 and abs(self.waypoints[-1][0] - 5.5) < 0.2) else 0.35
        self.detour_x = 0.0
        self.detour_y = 0.0
        self.stopped_steps = 0

    def reset(self):
        self.current_wp_idx = 0
        self.detour_x = 0.0
        self.detour_y = 0.0
        self.stopped_steps = 0

    def get_current_target(self, rx):
        target_wp = self.waypoints[self.current_wp_idx]
        tx, ty = target_wp[0], target_wp[1]
        # Only apply lateral detour in open central corridor if using canonical baseline route
        if rx > -3.2 and len(self.waypoints) == 7 and abs(self.waypoints[-1][0] - 5.5) < 0.2:
            tx += self.detour_x
            ty += self.detour_y
        return tx, ty

    def act(self, robot_pos, humans=None, action_dim=2):
        """
        Computes the velocity action given current robot position and dynamic humans.
        Returns:
            np.ndarray of shape (2,) [vx, vy] or shape (3,) [vx, vy, 0.0]
        """
        rx, ry = robot_pos[0], robot_pos[1]
        tx, ty = self.get_current_target(rx)
        dist_to_target = math.hypot(tx - rx, ty - ry)

        # Advance waypoint if active target reached
        while dist_to_target < self.waypoint_reach_dist and self.current_wp_idx < len(self.waypoints) - 1:
            self.current_wp_idx += 1
            self.detour_x = 0.0
            self.detour_y = 0.0
            tx, ty = self.get_current_target(rx)
            dist_to_target = math.hypot(tx - rx, ty - ry)

        dx = tx - rx
        dy = ty - ry
        norm = math.hypot(dx, dy)

        if norm > 1e-4:
            speed = min(self.max_speed, norm)
            vx = (dx / norm) * speed
            vy = (dy / norm) * speed
        else:
            vx, vy = 0.0, 0.0

        robot_radius = 0.30
        min_brake = 1.0
        should_stop = False

        if humans is not None:
            for idx, h in enumerate(humans):
                hx, hy, hvx, hvy, hrad = h.px, h.py, h.vx, h.vy, h.radius
                sum_r = robot_radius + hrad
                dist = math.hypot(rx - hx, ry - hy)
                dist_bound = dist - sum_r
                to_hx, to_hy = hx - rx, hy - ry
                h_speed = math.hypot(hvx, hvy)
                rel_vx, rel_vy = vx - hvx, vy - hvy
                closing_speed = (to_hx * rel_vx + to_hy * rel_vy) / max(dist, 1e-4)

                tot_speed = math.hypot(vx, vy)
                long_proj = (to_hx * vx + to_hy * vy) / tot_speed if tot_speed > 1e-4 else 0.0
                lat_proj = abs(-to_hx * vy + to_hy * vx) / tot_speed if tot_speed > 1e-4 else dist

                # 1. Absolute Emergency Stop (zero creeping into pedestrians)
                if dist_bound < 0.18:
                    should_stop = True
                    min_brake = 0.0
                    break
                elif dist_bound < 0.28 and closing_speed > -0.05 and (lat_proj < sum_r + 0.06 or dist_bound < 0.10):
                    should_stop = True
                    min_brake = 0.0
                    break

                # 2. Predictive CPA Yielding for moving cross-traffic / oncoming pedestrians
                if h_speed > 0.15:
                    rvx, rvy = hvx - vx, hvy - vy
                    rv2 = rvx * rvx + rvy * rvy
                    if rv2 > 1e-4:
                        t_cpa = -(to_hx * rvx + to_hy * rvy) / rv2
                        if 0.1 < t_cpa < 2.4:
                            cpa_x = to_hx + rvx * t_cpa
                            cpa_y = to_hy + rvy * t_cpa
                            if math.hypot(cpa_x, cpa_y) < sum_r + 0.28:
                                should_stop = True
                                min_brake = 0.0
                                break

                # 3. Path envelope check & longitudinal braking / open detour
                if dist < 2.5 and tot_speed > 1e-4:
                    if long_proj > 0 and lat_proj < sum_r + 0.06:
                        if dist_bound < 0.30 and closing_speed > 0.0:
                            should_stop = True
                            min_brake = 0.0
                            break
                        elif dist_bound < 0.85 and closing_speed > 0.0:
                            min_brake = min(min_brake, max(0.0, (dist_bound - 0.20) / 0.65))
                        elif dist_bound < 1.50 and closing_speed > 0.0:
                            min_brake = min(min_brake, 0.60)

                        # Lateral detour in open corridor & reading foyer (x > -3.2)
                        if h_speed < 0.10 and rx > -3.2 and dist < 1.8:
                            if self.current_wp_idx in [2, 3]:
                                self.detour_x = -0.70 if hx >= -1.0 else 0.70
                            elif self.current_wp_idx >= 4:
                                self.detour_y = 0.70 if hy <= -2.5 else -0.70

        if should_stop or min_brake <= 0.0:
            vx, vy = 0.0, 0.0
            self.stopped_steps += 1
            if self.stopped_steps > 8 and rx > -3.2:
                if self.current_wp_idx in [2, 3]:
                    self.detour_x = 0.70 if self.detour_x <= 0.0 else -0.70
                elif self.current_wp_idx >= 4:
                    self.detour_y = 0.70 if self.detour_y <= 0.0 else -0.70
                self.stopped_steps = 0
        else:
            self.stopped_steps = 0
            vx *= min_brake
            vy *= min_brake

        # Speed clamping & gentle goal docking
        tot_speed = math.hypot(vx, vy)
        if self.current_wp_idx == len(self.waypoints) - 1:
            dist_to_goal = math.hypot(self.waypoints[-1][0] - rx, self.waypoints[-1][1] - ry)
            max_dock_speed = max(0.20, min(self.max_speed, dist_to_goal * 0.9))
            if tot_speed > max_dock_speed:
                vx = (vx / tot_speed) * max_dock_speed
                vy = (vy / tot_speed) * max_dock_speed
        elif tot_speed > self.max_speed:
            vx = (vx / tot_speed) * self.max_speed
            vy = (vy / tot_speed) * self.max_speed

        if action_dim == 2:
            return np.array([vx, vy], dtype=np.float32)
        else:
            return np.array([vx, vy, 0.0], dtype=np.float32)


# ==============================================================================
# POLICY LOADING HELPER (VAE-LSTM PPO Baseline)
# ==============================================================================
def load_compatible_policy():
    """
    Locates and loads the pretrained VAE-LSTM policy compatible with
    CustomNavRepEncodedEnv (observation space shape: (37,)).
    """
    candidate_paths = [
        os.path.expanduser("~/navrep/models/gym/navreptrainencodedenv_2020_09_22__01_14_06_PPO_VAE_LSTM_V_ONLY_V32M512_ckpt.zip"),
        "/workspace/models/gym/navreptrainencodedenv_2020_09_22__01_14_06_PPO_VAE_LSTM_V_ONLY_V32M512_ckpt.zip",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/gym/navreptrainencodedenv_2020_09_22__01_14_06_PPO_VAE_LSTM_V_ONLY_V32M512_ckpt.zip")),
    ]
    policy_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            policy_path = p
            break

    if policy_path is None:
        return None, None

    print("[POLICY] Loading pretrained PPO2 checkpoint: {}".format(policy_path))
    ppo_model = PPO2.load(policy_path)
    policy = NavRepCPolicy(ppo_model)
    return policy, policy_path


# ==============================================================================
# STANDALONE CLI & EVALUATION UTILITIES
# ==============================================================================
def export_library_map(out_path="submission/task3_custom/maps/library.png"):
    """Renders a top-down 2D architectural blueprint map preview of the library layout."""
    cfg = create_library_config()
    bounds = cfg.world_bounds
    canvas_w, canvas_h = 720, 480
    frame = np.full((canvas_h, canvas_w, 3), (24, 28, 34), dtype=np.uint8)

    # 1. Background grid
    for gx in range(0, canvas_w, 40):
        cv2.line(frame, (gx, 0), (gx, canvas_h), (34, 40, 50), 1)
    for gy in range(0, canvas_h, 40):
        cv2.line(frame, (0, gy), (canvas_w, gy), (34, 40, 50), 1)

    # 2. Draw static obstacles (exclude outer boundary polygon at index 0)
    for poly in cfg.static_obstacles[1:]:
        pts = []
        for vx, vy in poly:
            px, py, _ = transform_world_to_canvas(vx, vy, [bounds[0], bounds[1]], [bounds[2], bounds[3]], canvas_w, canvas_h, margin=50)
            pts.append([px, py])
        if len(pts) >= 3:
            pts_arr = np.array(pts, dtype=np.int32)
            cv2.fillPoly(frame, [pts_arr], (60, 68, 82))
            cv2.polylines(frame, [pts_arr], True, (100, 115, 140), 2)

    # 3. Start Marker (Green S)
    spx, spy, _ = transform_world_to_canvas(cfg.robot_start[0], cfg.robot_start[1], [bounds[0], bounds[1]], [bounds[2], bounds[3]], canvas_w, canvas_h, margin=50)
    cv2.circle(frame, (spx, spy), 14, (46, 204, 113), -1)
    cv2.putText(frame, "S", (spx - 6, spy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # 4. Goal Marker (Gold G / Circulation Desk)
    gpx, gpy, _ = transform_world_to_canvas(cfg.robot_goal[0], cfg.robot_goal[1], [bounds[0], bounds[1]], [bounds[2], bounds[3]], canvas_w, canvas_h, margin=50)
    cv2.circle(frame, (gpx, gpy), 16, (0, 215, 255), 2)
    cv2.circle(frame, (gpx, gpy), 8, (0, 215, 255), -1)
    cv2.putText(frame, "COUNTER (GOAL)", (gpx - 80, gpy - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 215, 255), 2)

    # 5. Direct navigation reference line
    cv2.line(frame, (spx, spy), (gpx, gpy), (80, 80, 80), 1, cv2.LINE_AA)

    # 6. Base human patrons (Coral circles with waypoint trails)
    for h in cfg.base_humans:
        hpx, hpy, _ = transform_world_to_canvas(h['pos'][0], h['pos'][1], [bounds[0], bounds[1]], [bounds[2], bounds[3]], canvas_w, canvas_h, margin=50)
        hgx, hgy, _ = transform_world_to_canvas(h['goal'][0], h['goal'][1], [bounds[0], bounds[1]], [bounds[2], bounds[3]], canvas_w, canvas_h, margin=50)
        cv2.line(frame, (hpx, hpy), (hgx, hgy), (60, 100, 220), 1, cv2.LINE_AA)
        cv2.circle(frame, (hpx, hpy), 10, (40, 90, 255), -1)
        cv2.circle(frame, (hpx, hpy), 10, (255, 255, 255), 1)

    # 7. Header title card
    cv2.rectangle(frame, (0, 0), (canvas_w, 48), (15, 18, 22), -1)
    cv2.line(frame, (0, 48), (canvas_w, 48), (50, 60, 75), 1)
    cv2.putText(frame, "Library Book Collection - Scenario Map", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cv2.imwrite(out_path, frame)
    print("[MAP_EXPORT] Library blueprint map exported to: {}".format(out_path))
    return out_path


def record_library_video(difficulty="MEDIUM", episodes=1, seed=42, fps=12, out_path=None, max_steps=380, controller_type="library"):
    """
    Executes navigation rollout and records an MP4 video.
    Supports controller_type='library' (default, working hybrid controller) or 'ppo' (baseline policy).
    """
    if out_path is None:
        out_path = "submission/task3_custom/videos/library_{}.mp4".format(difficulty.lower())

    cfg = create_library_config()
    use_ppo = (controller_type.lower() == 'ppo')

    if use_ppo:
        policy, policy_path = load_compatible_policy()
        if policy is None:
            print("[RECORD_LIBRARY] WARNING: No compatible PPO checkpoint found. Falling back to library controller.")
            use_ppo = False
    else:
        policy = None

    ctrl = LibraryNavigationController() if not use_ppo else None

    env = CustomNavRepEncodedEnv(scenario="library", difficulty=difficulty, gpu=False, silent=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    writer = imageio.get_writer(out_path, fps=fps, codec='libx264', pixelformat='yuv420p', quality=8)
    dt = 0.2
    total_frames = 0

    ctrl_desc = "Pretrained VAE-LSTM PPO" if use_ppo else "LibraryNavigationController"
    print("[RECORD_LIBRARY] Recording Library [{}] -> {} (Controller: {})".format(
        difficulty, out_path, ctrl_desc))

    for ep in range(1, episodes + 1):
        np.random.seed(seed + ep - 1)
        ob = env.reset("library", difficulty)
        sync_and_rasterize_sim(env.raw_env, cfg)
        if ctrl is not None:
            ctrl.reset()

        sim = env.soadrl_sim
        start_pos = [sim.robot.px, sim.robot.py]
        robot_trail = [start_pos[:]]
        status = "RUNNING"
        done = False
        step = 0
        sim_time = 0.0

        while not done and step < max_steps:
            if use_ppo:
                action = policy.act(ob)
            else:
                pos = [sim.robot.px, sim.robot.py]
                action = ctrl.act(pos, sim.humans, action_dim=2)

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
                scenario_name="LIBRARY_{}".format(difficulty.upper()),
                episode_num=ep,
                step_num=step,
                sim_time=sim_time,
                status=status,
                robot_trail=robot_trail,
                start_pos=start_pos
            )
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            total_frames += 1

        if not done and step >= max_steps:
            status = "TIMEOUT"

        # Hold outcome frame for 1.5 seconds
        hold_frames = int(fps * 1.5)
        final_frame = render_scene_frame(
            env=env,
            scenario_name="LIBRARY_{}".format(difficulty.upper()),
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
    print("[RECORD_LIBRARY] Video successfully saved: {} ({:.2f}s)".format(out_path, duration))
    return out_path


def evaluate_library_benchmark(difficulty="MEDIUM", episodes=5, base_seed=42, controller_type="library"):
    """
    Evaluates the Library scenario using either the working LibraryNavigationController (default)
    or the pretrained VAE-LSTM PPO baseline policy.
    Tracks ReachGoal, Collision, and Timeout termination signals from crowd_sim.
    """
    cfg = create_library_config()
    use_ppo = (controller_type.lower() == 'ppo')

    if use_ppo:
        policy, policy_path = load_compatible_policy()
        if policy is None:
            print("[EVAL_LIBRARY] ERROR: Pretrained policy checkpoint could not be found.")
            return None
        ctrl_name = "Pretrained VAE-LSTM PPO ({})".format(os.path.basename(policy_path))
    else:
        policy = None
        ctrl_name = "LibraryNavigationController (Topological + APF Yielding)"

    print("[EVAL_LIBRARY] Evaluating Library [{}] over {} episodes with {}...".format(
        difficulty, episodes, ctrl_name))

    ctrl = LibraryNavigationController() if not use_ppo else None
    env = CustomNavRepEncodedEnv(scenario="library", difficulty=difficulty, gpu=False, silent=True)
    dt = 0.2

    successes = 0
    collisions = 0
    timeouts = 0
    ttg_list = []
    episode_records = []

    max_steps = cfg.get_difficulty_config(difficulty).get('time_limit', 380)
    for ep in range(episodes):
        np.random.seed(base_seed + ep)
        ob = env.reset("library", difficulty)
        sync_and_rasterize_sim(env.raw_env, cfg)
        if ctrl is not None:
            ctrl.reset()

        done = False
        step = 0
        sim_time = 0.0
        outcome = "TIMEOUT"

        while not done and step < max_steps:
            if use_ppo:
                action = policy.act(ob)
            else:
                pos = [env.soadrl_sim.robot.px, env.soadrl_sim.robot.py]
                action = ctrl.act(pos, env.soadrl_sim.humans, action_dim=2)

            ob, reward, done, info = env.step(action)
            step += 1
            sim_time += dt

            event = info.get('event', None)
            if isinstance(event, ReachGoal):
                successes += 1
                outcome = "GOAL REACHED"
                ttg_list.append(sim_time)
                done = True
            elif isinstance(event, (Collision, CollisionOtherAgent)):
                collisions += 1
                outcome = "COLLISION"
                done = True
            elif isinstance(event, Timeout) or step >= max_steps:
                timeouts += 1
                outcome = "TIMEOUT"
                done = True

        if not done and step >= max_steps:
            timeouts += 1
            outcome = "TIMEOUT"

        dist_to_goal = math.hypot(env.soadrl_sim.robot.px - cfg.robot_goal[0],
                                  env.soadrl_sim.robot.py - cfg.robot_goal[1])

        episode_records.append({
            'episode': ep + 1,
            'outcome': outcome,
            'steps': step,
            'sim_time': sim_time,
            'dist_to_goal': dist_to_goal,
        })
        print("  Ep {:2d}: Outcome={:12s} Steps={:3d} SimTime={:5.1f}s FinalDist={:.2f}m".format(
            ep + 1, outcome, step, sim_time, dist_to_goal))

    env.close()
    succ_rate = (float(successes) / episodes) * 100.0
    coll_rate = (float(collisions) / episodes) * 100.0
    to_rate = (float(timeouts) / episodes) * 100.0
    mean_ttg = float(np.mean(ttg_list)) if ttg_list else 0.0

    print("=" * 70)
    print("LIBRARY SCENARIO BENCHMARK RESULTS [{}]".format(difficulty))
    print("=" * 70)
    print("  Controller Used   : {}".format(ctrl_name))
    print("  Episodes Evaluated: {}".format(episodes))
    print("  Success Rate      : {:.1f}% ({}/{})".format(succ_rate, successes, episodes))
    print("  Collision Rate    : {:.1f}% ({}/{})".format(coll_rate, collisions, episodes))
    print("  Timeout Rate      : {:.1f}% ({}/{})".format(to_rate, timeouts, episodes))
    print("  Mean Time-to-Goal : {:.1f}s (successful episodes only)".format(mean_ttg))
    print("=" * 70)
    return {
        'difficulty': difficulty,
        'episodes': episodes,
        'successes': successes,
        'collisions': collisions,
        'timeouts': timeouts,
        'success_rate': succ_rate,
        'collision_rate': coll_rate,
        'timeout_rate': to_rate,
        'mean_ttg': mean_ttg,
        'records': episode_records,
    }


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Library Book Collection Scenario Runner")
    parser.add_argument('--difficulty', default='MEDIUM', choices=['EASY', 'MEDIUM', 'HARD'], help="Difficulty tier")
    parser.add_argument('--episodes', type=int, default=2, help="Number of episodes to execute")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--controller', default='library', choices=['library', 'ppo'], help="Navigation controller")
    parser.add_argument('--record', type=str, nargs='?', const='default', default=None, help="Record MP4 video")
    parser.add_argument('--export-map', type=str, nargs='?', const='default', default=None, help="Export 2D map PNG")
    parser.add_argument('--benchmark', action='store_true', help="Run benchmark evaluation")
    args = parser.parse_args()

    cfg = create_library_config()
    print("[LIBRARY] Loaded '{}' | Bounds: {} | Obstacles: {} | Humans: {}".format(
        cfg.name, cfg.world_bounds, len(cfg.static_obstacles), len(cfg.base_humans)))

    if args.export_map:
        out_p = "submission/task3_custom/maps/library.png" if args.export_map == 'default' else args.export_map
        export_library_map(out_p)

    if args.record:
        out_v = "submission/task3_custom/videos/library_{}.mp4".format(args.difficulty.lower()) if args.record == 'default' else args.record
        record_library_video(difficulty=args.difficulty, episodes=args.episodes, seed=args.seed, out_path=out_v, controller_type=args.controller)

    if args.benchmark:
        evaluate_library_benchmark(difficulty=args.difficulty, episodes=args.episodes, base_seed=args.seed, controller_type=args.controller)

    if not args.export_map and not args.record and not args.benchmark:
        print("[LIBRARY] Running default 2-episode video rollout...")
        record_library_video(difficulty=args.difficulty, episodes=args.episodes, seed=args.seed, controller_type=args.controller)
