#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
airport_passenger_assistance.py
===============================
Task 3 Custom Scenario: Airport Passenger Assistance

Story:
  An assistance robot guides an elderly passenger carrying heavy luggage from
  the check-in hall entrance to boarding Gate 12 through a busy airport terminal
  hall with check-in desks, information kiosks, security lane blocks, seating
  lounges, and crossing pedestrians.

Features:
  - Terminal hall map (24m x 15m) with double outer wall boundary.
  - 10 rectangular static obstacles: check-in counter rows, info kiosk, security
    lane divider, structural pillars, waiting lounge seating blocks, duty free counter,
    and Gate 12 service podium.
  - Heterogeneous human agents with realistic airport roles:
      * Slow luggage-trolley passengers (larger radius 0.45m, slow 0.40 m/s)
      * Hurried passengers rushing to flights (radius 0.30m, fast 0.95 m/s)
      * Family group walking together (parent r=0.32m, child r=0.25m, 0.55 m/s)
      * Airport ground staff crossing the concourse (radius 0.30m, 0.60 m/s)
  - Monotonic difficulty scaling:
      * EASY: 5 humans, time limit 420 steps (84.0s)
      * MEDIUM: 8 humans, time limit 360 steps (72.0s), boarding call event at T=75
      * HARD: 12 humans, time limit 290 steps (58.0s), boarding call event at T=50
  - Reactive intelligent controller (AirportNavigationController) with time-to-closest-approach
    (TCA) collision prediction, detour maneuvering, and final docking slowdown.
  - AirportNavRepEnv subclass of CustomNavRepEnv handling time-limit synchronization
    and dynamic boarding call events.
  - Self-registers in SCENARIO_REGISTRY under 'airportpassengerassistance' and 'airport'.
"""

from __future__ import print_function
import math
import numpy as np

from navrep_custom.scenarios import (
    ScenarioConfig,
    SCENARIO_REGISTRY,
    make_box_obstacle,
)
from navrep_custom.custom_envs import CustomNavRepEnv


# ==============================================================================
# GEOMETRIC UTILITY FUNCTIONS
# ==============================================================================

def point_to_segment_distance(px, py, x1, y1, x2, y2):
    """Euclidean distance from point (px, py) to line segment (x1, y1)-(x2, y2)."""
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-9:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def point_in_polygon(px, py, polygon):
    """Ray casting algorithm to determine if (px, py) is strictly inside polygon."""
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if py > min(p1y, p2y):
            if py <= max(p1y, p2y):
                if px <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or px <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def distance_point_to_polygon(px, py, polygon):
    """Shortest distance from point (px, py) to any edge of polygon."""
    min_dist = float('inf')
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        d = point_to_segment_distance(px, py, x1, y1, x2, y2)
        if d < min_dist:
            min_dist = d
    return min_dist


# ==============================================================================
# SCENARIO CONFIGURATION FACTORY
# ==============================================================================

def create_airport_config():
    """
    Constructs the Airport Passenger Assistance scenario configuration.
    
    Map dimensions: 24m x 15m (x: [-12.0, 12.0], y: [-7.5, 7.5]).
    Start: Check-in Entrance at [-10.2, -2.5, 0.0 rad] (heading East).
    Goal: Gate 12 Boarding Entrance at [9.8, 2.8] (~20.69m Euclidean separation).
    """
    bounds = [-12.0, 12.0, -7.5, 7.5]
    start = [-10.2, -2.5, 0.0]
    goal = [9.8, 2.8]

    # 1. Boundary and Static Obstacles
    obstacles = [
        # Outer boundary wall (double outer wall structure with CustomNavRepEnv border)
        [(-12.0, -7.5), (12.0, -7.5), (12.0, 7.5), (-12.0, 7.5)],

        # Check-in counter rows (North & South of entry aisle)
        make_box_obstacle(-7.5, 3.8, 1.8, 2.6),   # Counter Row A
        make_box_obstacle(-7.5, -4.8, 1.8, 2.2),  # Counter Row B

        # Information kiosk and Security Inspection Island
        make_box_obstacle(-3.5, 4.2, 1.5, 1.5),   # FIDS Info Kiosk
        make_box_obstacle(-3.5, 0.0, 1.6, 2.4),   # Security Checkpoint Block

        # Central Concourse Structural Pillars
        make_box_obstacle(0.5, 2.8, 1.2, 1.2),    # North Pillar
        make_box_obstacle(0.5, -2.8, 1.2, 1.2),   # South Pillar

        # Waiting Lounge Seating blocks
        make_box_obstacle(5.0, 4.5, 3.0, 1.2),    # North Seating Bank
        make_box_obstacle(5.0, -1.8, 3.0, 1.2),   # South Seating Bank

        # Concessions & Gate Service Podium
        make_box_obstacle(7.5, -4.8, 2.2, 2.0),   # Duty-Free Shop Island
        make_box_obstacle(8.5, 5.2, 1.6, 1.2),    # Gate 12 Service Desk
    ]

    # 2. Dynamic Human Agents
    base_humans = [
        # 1. Slow passenger pushing large luggage trolley
        {
            'pos': [2.0, -0.5, math.pi],
            'goal': [-6.0, -0.5],
            'radius': 0.45,
            'speed': 0.40,
            'role': 'passenger_luggage_trolley'
        },
        # 2. Hurried business traveler rushing to gate
        {
            'pos': [6.0, 0.8, math.pi],
            'goal': [-7.0, 0.8],
            'radius': 0.30,
            'speed': 0.95,
            'role': 'passenger_hurried'
        },
        # 3. Family group parent
        {
            'pos': [-1.0, 3.0, 0.0],
            'goal': [4.0, 3.0],
            'radius': 0.32,
            'speed': 0.55,
            'role': 'family_parent'
        },
        # 4. Family group child
        {
            'pos': [-1.0, 3.6, 0.0],
            'goal': [4.0, 3.6],
            'radius': 0.25,
            'speed': 0.55,
            'role': 'family_child'
        },
        # 5. Airport ground staff crossing terminal concourse
        {
            'pos': [2.5, 3.2, -math.pi / 2],
            'goal': [2.5, -3.2],
            'radius': 0.30,
            'speed': 0.60,
            'role': 'staff_crossing'
        },
    ]

    # 3. Difficulty Modifiers
    diff_mods = {
        'EASY': {
            'density_mult': 0.70,
            'speed_mult': 0.85,
            'time_limit': 420,  # 84.0 seconds at dt=0.2s
            'extra_humans': 0   # Total: 5 humans
        },
        'MEDIUM': {
            'density_mult': 1.0,
            'speed_mult': 1.0,
            'time_limit': 360,  # 72.0 seconds at dt=0.2s
            'extra_humans': 3   # Total: 8 humans
        },
        'HARD': {
            'density_mult': 1.35,
            'speed_mult': 1.15,
            'time_limit': 290,  # 58.0 seconds at dt=0.2s
            'extra_humans': 7   # Total: 12 humans
        }
    }

    return ScenarioConfig(
        scenario_id="airportpassengerassistance",
        name="Airport Passenger Assistance - Terminal Concourse to Gate 12",
        description=(
            "An assistance robot guides an elderly passenger carrying heavy luggage "
            "from the check-in hall entrance to boarding Gate 12 through a busy terminal hall."
        ),
        world_bounds=bounds,
        robot_start=start,
        robot_goal=goal,
        static_obstacles=obstacles,
        base_humans=base_humans,
        difficulty_modifiers=diff_mods,
        time_limit_steps=360
    )


# Self-register scenario in registry
SCENARIO_REGISTRY["airportpassengerassistance"] = create_airport_config
SCENARIO_REGISTRY["airport"] = create_airport_config


# ==============================================================================
# REACTIVE NAVIGATION CONTROLLER
# ==============================================================================

class AirportNavigationController(object):
    """
    Intelligent reactive navigation controller for Airport Passenger Assistance.
    
    Navigates through defined terminal waypoints, avoiding static obstacles,
    predicting collisions with dynamic pedestrians using Time-to-Closest-Approach (TCA),
    yielding to luggage trolleys and hurried passengers, taking dynamic detours when blocked,
    and smoothly decelerating when arriving at Gate 12.
    """

    def __init__(self, max_speed=0.65):
        self.max_speed = max_speed
        self.waypoints = [
            [-10.2, -2.5],  # WP0: Check-in entrance departure
            [-6.0, -2.5],   # WP1: Clear check-in counter aisle
            [-2.0, -2.5],   # WP2: Security lane passage
            [0.5, 0.0],     # WP3: Central terminal concourse junction
            [3.0, 1.2],     # WP4: North corridor past seating banks
            [6.5, 1.8],     # WP5: Gate 12 approach concourse
            [9.8, 2.8]      # WP6: Gate 12 boarding podium
        ]
        self.current_wp_idx = 0
        self.stopped_steps = 0
        self.detour_y = 0.0

    def reset(self):
        """Resets controller state for a new episode."""
        self.current_wp_idx = 0
        self.stopped_steps = 0
        self.detour_y = 0.0

    def act(self, robot_pos, humans=None, action_dim=3):
        """
        Computes velocity action [vx, vy, (w)] given current robot pose and pedestrian state.
        
        Args:
            robot_pos: [rx, ry] current position
            humans: list of Human objects from simulator
            action_dim: 2 for [vx, vy], 3 for [vx, vy, w]
            
        Returns:
            np.ndarray of shape (action_dim,)
        """
        rx, ry = robot_pos[0], robot_pos[1]
        tx, ty = self.waypoints[self.current_wp_idx]
        ty += self.detour_y

        distance = math.hypot(tx - rx, ty - ry)

        # Advance waypoint if within acceptance threshold
        while distance < 0.50 and self.current_wp_idx < len(self.waypoints) - 1:
            self.current_wp_idx += 1
            self.detour_y = 0.0
            tx, ty = self.waypoints[self.current_wp_idx]
            distance = math.hypot(tx - rx, ty - ry)

        # Baseline velocity vector towards current waypoint
        if distance > 0.001:
            speed = min(self.max_speed, distance)
            vx = (tx - rx) / distance * speed
            vy = (ty - ry) / distance * speed
        else:
            vx, vy = 0.0, 0.0

        stopped = False

        # Collision avoidance against dynamic pedestrians
        if humans is not None:
            for h in humans:
                dx = h.px - rx
                dy = h.py - ry
                dist = math.hypot(dx, dy)

                # Static safety buffer (robot 0.30 + human radius + 0.28m clearance)
                safe_dist = 0.30 + h.radius + 0.28

                if dist < safe_dist:
                    stopped = True
                    break

                # Dynamic Time-to-Closest-Approach (TCA) collision check
                rel_vx = vx - h.vx
                rel_vy = vy - h.vy
                rel_speed_sq = rel_vx * rel_vx + rel_vy * rel_vy

                if rel_speed_sq > 0.001:
                    tca = -(dx * rel_vx + dy * rel_vy) / rel_speed_sq
                    if 0.0 < tca < 1.6:
                        fut_x = dx + rel_vx * tca
                        fut_y = dy + rel_vy * tca
                        if math.hypot(fut_x, fut_y) < safe_dist:
                            stopped = True
                            break

        if stopped:
            vx, vy = 0.0, 0.0
            self.stopped_steps += 1
            # If blocked for 6 steps (~1.2s), attempt lateral detour
            if self.stopped_steps >= 6:
                self.detour_y = 0.70 if self.detour_y <= 0 else -0.70
                self.stopped_steps = 0
        else:
            self.stopped_steps = 0

        # Smooth deceleration approaching Gate 12 docking ring
        if self.current_wp_idx == len(self.waypoints) - 1:
            dock_speed = max(0.20, min(self.max_speed, distance * 0.75))
            total_speed = math.hypot(vx, vy)
            if total_speed > dock_speed:
                vx = vx / total_speed * dock_speed
                vy = vy / total_speed * dock_speed

        if action_dim == 2:
            return np.array([vx, vy], dtype=np.float32)
        return np.array([vx, vy, 0.0], dtype=np.float32)


# ==============================================================================
# AIRPORT ENVIRONMENT SUBCLASS
# ==============================================================================

class AirportNavRepEnv(CustomNavRepEnv):
    """
    Specialized environment for the Airport Passenger Assistance scenario.
    
    Automatically synchronizes CrowdSim time limits and injects the
    dynamic 'Boarding Call' event at step T for MEDIUM and HARD levels.
    """

    def __init__(self, difficulty="MEDIUM", silent=True):
        super(AirportNavRepEnv, self).__init__(
            scenario="airportpassengerassistance",
            difficulty=difficulty,
            silent=silent
        )

    def reset(self, scenario_name=None, difficulty=None, seed=None):
        if seed is not None:
            self.soadrl_sim.case_capacity['val'] = 0
            self.soadrl_sim.case_counter['test'] = seed
            np.random.seed(seed)

        obs = super(AirportNavRepEnv, self).reset(
            scenario_name or "airportpassengerassistance",
            difficulty or self.current_difficulty
        )
        # Synchronize soadrl_sim time limit with environment time limit
        self.soadrl_sim.time_limit = self.time_limit * self._get_dt()

        # Enforce minimum spawn clearance >= 2.0m from robot start position
        sx, sy = self.scenario_config.robot_start[0], self.scenario_config.robot_start[1]
        safe_slots = [
            [-6.0, 1.0], [0.0, 0.0], [2.0, 1.5], [2.0, -1.5],
            [4.0, 0.0], [6.0, 0.0], [-1.0, 0.0], [1.0, -1.0]
        ]
        slot_idx = 0
        for human in self.soadrl_sim.humans:
            dist = math.hypot(human.px - sx, human.py - sy)
            if dist < 2.0:
                slot = safe_slots[slot_idx % len(safe_slots)]
                human.px, human.py = float(slot[0]), float(slot[1])
                slot_idx += 1

        return obs

    def step(self, action):
        # Trigger dynamic "Boarding Call" event for levels above EASY
        sim = self.soadrl_sim
        diff = self.current_difficulty

        if diff == "MEDIUM" and self.steps_since_reset == 75:
            # Flight 204 boarding call: concourse pedestrians move towards Gate 12
            for h_idx in [2, 3]:
                if h_idx < len(sim.humans):
                    sim.humans[h_idx].gx = 11.2
                    sim.humans[h_idx].gy = 2.8
        elif diff == "HARD" and self.steps_since_reset == 50:
            # Hard boarding surge: multiple passenger agents head to Gate 12
            for h_idx in [1, 2, 3, 4]:
                if h_idx < len(sim.humans):
                    sim.humans[h_idx].gx = 11.2
                    sim.humans[h_idx].gy = 2.8

        return super(AirportNavRepEnv, self).step(action)
