#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scenarios.py
Registry and specifications for custom robotics navigation scenarios:
  1. MallMate: Shopping-Mall Assistance Robot (Atrium, kiosks, family group, darting child, sale hotspot)
  2. MediRunner: Hospital Medicine-Delivery Robot (Corridors, fast nurses, slow walker patients, emergency gurney)
  3. KiranaKart: Warehouse AGV (Grid of shelf aisles, cross-aisle pickers, blind-corner crossings, rush periods)
  4. DiwaliBazaar: Festival Market Rush (SVNIT Surat street market, high density counter-flow, bargaining clusters)
  5. RailwayPlatform: Platform 1 Station Rush (Platform pillars, train arrival door passenger surge)
  6. FireDrill: Evacuation Counter-Flow (High-urgency emergency exit flow vs robot ingress)

Includes Difficulty Ladder (EASY / MEDIUM / HARD) and parametric crowd scaling.
"""

from __future__ import print_function
import math
import numpy as np


class ScenarioConfig(object):
    """Holds configuration parameters and geometry for a navigation scenario."""
    def __init__(self,
                 scenario_id,
                 name,
                 description,
                 world_bounds,
                 robot_start,
                 robot_goal,
                 static_obstacles,
                 base_humans,
                 difficulty_modifiers=None,
                 special_behaviors=None,
                 time_limit_steps=400):
        self.scenario_id = scenario_id
        self.name = name
        self.description = description
        self.world_bounds = world_bounds  # [x_min, x_max, y_min, y_max]
        self.robot_start = robot_start    # [x, y, theta]
        self.robot_goal = robot_goal      # [x, y]
        self.static_obstacles = static_obstacles  # list of polygons [[(x,y), ...], ...]
        self.base_humans = base_humans    # list of dicts: {'pos', 'goal', 'radius', 'speed', 'role'}
        self.difficulty_modifiers = difficulty_modifiers or {}
        self.special_behaviors = special_behaviors or {}
        self.time_limit_steps = time_limit_steps

    def get_difficulty_config(self, difficulty="MEDIUM"):
        """Returns adjusted human list, speed multiplier, and time limit for given difficulty."""
        diff = difficulty.upper()
        mod = self.difficulty_modifiers.get(diff, {
            'density_mult': 1.0,
            'speed_mult': 1.0,
            'time_limit': self.time_limit_steps
        })
        return mod


def make_box_obstacle(cx, cy, width, height):
    """Generates a rectangular polygon obstacle centered at (cx, cy)."""
    hw = width / 2.0
    hh = height / 2.0
    return [
        (cx - hw, cy - hh),
        (cx + hw, cy - hh),
        (cx + hw, cy + hh),
        (cx - hw, cy + hh)
    ]


def make_pillar_obstacle(cx, cy, radius, sides=8):
    """Generates a polygonal pillar approximating a circular column."""
    angles = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    return [(cx + radius * np.cos(a), cy + radius * np.sin(a)) for a in angles]


# ==============================================================================
# SCENARIO 1: MallMate - Shopping-Mall Assistance Robot
# ==============================================================================
def create_mallmate_config():
    # 20m x 20m Central Atrium
    bounds = [-10.0, 10.0, -10.0, 10.0]
    # Family entrance -> Food court opposite
    start = [-8.0, 0.0, 0.0]
    goal = [8.0, 0.0]
    
    obstacles = [
        # Outer Atrium Walls
        [(-10.0, -10.0), (10.0, -10.0), (10.0, 10.0), (-10.0, 10.0)],
        # Central Pillars
        make_pillar_obstacle(-3.0, 3.0, 0.6),
        make_pillar_obstacle(-3.0, -3.0, 0.6),
        make_pillar_obstacle(3.0, 3.0, 0.6),
        make_pillar_obstacle(3.0, -3.0, 0.6),
        # Central Information Kiosk
        make_box_obstacle(0.0, 0.0, 2.0, 2.0),
        # Resting Benches
        make_box_obstacle(-2.0, 6.0, 2.5, 0.8),
        make_box_obstacle(2.0, -6.0, 2.5, 0.8),
        # Sale Event Booth (Crowded shop corner)
        make_box_obstacle(5.0, 5.0, 3.0, 1.5),
    ]

    # Base Human Agents
    base_humans = [
        # Window Shopper 1 (Slow walker)
        {'pos': [-4.0, 5.0, -math.pi/2], 'goal': [-4.0, -5.0], 'radius': 0.3, 'speed': 0.45, 'role': 'shopper_slow'},
        # Window Shopper 2
        {'pos': [4.0, -5.0, math.pi/2], 'goal': [4.0, 5.0], 'radius': 0.3, 'speed': 0.50, 'role': 'shopper_slow'},
        # Rushed walker (Late for movie)
        {'pos': [0.0, -8.0, math.pi/2], 'goal': [0.0, 8.0], 'radius': 0.3, 'speed': 1.40, 'role': 'shopper_fast'},
        # Family Group: Parent 1 + Child (darting agent)
        {'pos': [1.0, 3.0, -math.pi/2], 'goal': [1.0, -4.0], 'radius': 0.35, 'speed': 0.70, 'role': 'family_parent'},
        {'pos': [1.8, 3.2, -math.pi/2], 'goal': [1.5, -4.0], 'radius': 0.25, 'speed': 0.95, 'role': 'family_child_erratic'},
        # Sale Hotspot Bargainers (stationary/slow clustering)
        {'pos': [6.0, 4.0, 0.0], 'goal': [5.5, 3.5], 'radius': 0.3, 'speed': 0.20, 'role': 'sale_cluster'},
        {'pos': [4.5, 6.2, -math.pi/4], 'goal': [4.8, 5.8], 'radius': 0.3, 'speed': 0.20, 'role': 'sale_cluster'},
    ]

    diff_mods = {
        'EASY': {'density_mult': 0.6, 'speed_mult': 0.8, 'time_limit': 500, 'extra_humans': 0},
        'MEDIUM': {'density_mult': 1.0, 'speed_mult': 1.0, 'time_limit': 400, 'extra_humans': 2},
        'HARD': {'density_mult': 1.6, 'speed_mult': 1.3, 'time_limit': 300, 'extra_humans': 5},
    }

    return ScenarioConfig(
        scenario_id="mallmate",
        name="MallMate - Shopping-Mall Assistance Robot",
        description="Navigation through a busy shopping mall atrium with pillars, kiosks, shopper clusters, and an unpredictable darting child agent.",
        world_bounds=bounds,
        robot_start=start,
        robot_goal=goal,
        static_obstacles=obstacles,
        base_humans=base_humans,
        difficulty_modifiers=diff_mods,
        time_limit_steps=400
    )


# ==============================================================================
# SCENARIO 2: MediRunner - Hospital Medicine-Delivery Robot
# ==============================================================================
def create_medirunner_config():
    # Long hospital corridor wing: 24m x 8m
    bounds = [-12.0, 12.0, -4.0, 4.0]
    # Pharmacy (West Wing) -> ICU Nurse Station (East Wing)
    start = [-10.5, 0.0, 0.0]
    goal = [10.5, 0.0]

    obstacles = [
        # Corridor Outer Boundaries
        [(-12.0, -4.0), (12.0, -4.0), (12.0, 4.0), (-12.0, 4.0)],
        # North Corridor Wall Segments with Room Recesses
        make_box_obstacle(-6.0, 3.0, 4.0, 2.0),
        make_box_obstacle(2.0, 3.0, 4.0, 2.0),
        # South Corridor Wall Segments
        make_box_obstacle(-2.0, -3.0, 4.0, 2.0),
        make_box_obstacle(6.0, -3.0, 4.0, 2.0),
        # Mid-hallway Triage Desk / Hand sanitizer column
        make_box_obstacle(0.0, 1.2, 1.5, 0.8),
    ]

    base_humans = [
        # Purposeful Nurse 1 (Fast counter-flow)
        {'pos': [9.0, 0.5, math.pi], 'goal': [-9.0, 0.5], 'radius': 0.3, 'speed': 1.25, 'role': 'nurse_fast'},
        # Purposeful Nurse 2
        {'pos': [-7.0, -0.6, 0.0], 'goal': [7.0, -0.6], 'radius': 0.3, 'speed': 1.15, 'role': 'nurse_fast'},
        # Frail Patient with Walker (Extremely slow)
        {'pos': [3.0, -0.2, math.pi], 'goal': [-8.0, -0.2], 'radius': 0.45, 'speed': 0.28, 'role': 'patient_walker'},
        # Visiting Relative (Wandering)
        {'pos': [-3.0, 0.8, 0.0], 'goal': [4.0, 0.8], 'radius': 0.3, 'speed': 0.60, 'role': 'visitor'},
        # Emergency Gurney (Heavy, large radius, high priority speed)
        {'pos': [7.0, -1.0, math.pi], 'goal': [-11.0, -1.0], 'radius': 0.65, 'speed': 1.65, 'role': 'emergency_gurney'},
    ]

    diff_mods = {
        'EASY': {'density_mult': 0.6, 'speed_mult': 0.7, 'time_limit': 450, 'extra_humans': 0},
        'MEDIUM': {'density_mult': 1.0, 'speed_mult': 1.0, 'time_limit': 350, 'extra_humans': 2},
        'HARD': {'density_mult': 1.5, 'speed_mult': 1.35, 'time_limit': 240, 'extra_humans': 4},
    }

    return ScenarioConfig(
        scenario_id="medirunner",
        name="MediRunner - Hospital Medicine Delivery",
        description="Time-critical medicine transit along narrow hospital corridors, yielding to fast nurses, accommodating slow walker patients, and evading emergency gurneys.",
        world_bounds=bounds,
        robot_start=start,
        robot_goal=goal,
        static_obstacles=obstacles,
        base_humans=base_humans,
        difficulty_modifiers=diff_mods,
        time_limit_steps=350
    )


# ==============================================================================
# SCENARIO 3: KiranaKart - Warehouse Fulfillment AGV
# ==============================================================================
def create_kiranakart_config():
    # 22m x 16m Fulfillment Logistics Warehouse
    bounds = [-11.0, 11.0, -8.0, 8.0]
    # Inbound Unloading Dock (South-West) -> Outbound Packing Station (North-East)
    start = [-9.0, -6.5, math.pi/4]
    goal = [9.0, 6.5]

    obstacles = [
        # Warehouse Boundary
        [(-11.0, -8.0), (11.0, -8.0), (11.0, 8.0), (-11.0, 8.0)],
        # Storage Shelf Aisle Grid (Rows of rectangular high-bay racks)
        make_box_obstacle(-6.0, -3.5, 4.0, 1.4),
        make_box_obstacle(0.0, -3.5, 4.0, 1.4),
        make_box_obstacle(6.0, -3.5, 4.0, 1.4),

        make_box_obstacle(-6.0, 0.0, 4.0, 1.4),
        make_box_obstacle(0.0, 0.0, 4.0, 1.4),
        make_box_obstacle(6.0, 0.0, 4.0, 1.4),

        make_box_obstacle(-6.0, 3.5, 4.0, 1.4),
        make_box_obstacle(0.0, 3.5, 4.0, 1.4),
        make_box_obstacle(6.0, 3.5, 4.0, 1.4),
    ]

    base_humans = [
        # Human Picker 1 (Moving between Aisle 1 and Cross-Aisle)
        {'pos': [-3.0, -3.5, math.pi/2], 'goal': [-3.0, 3.5], 'radius': 0.32, 'speed': 0.85, 'role': 'picker'},
        # Human Picker 2 (Orthogonal cross-aisle transition)
        {'pos': [3.0, 3.5, -math.pi/2], 'goal': [3.0, -3.5], 'radius': 0.32, 'speed': 0.85, 'role': 'picker'},
        # Forklift / Fast Cross AGV (Blind-corner horizontal crossing)
        {'pos': [-8.5, 0.0, 0.0], 'goal': [8.5, 0.0], 'radius': 0.55, 'speed': 1.50, 'role': 'forklift_agv'},
        # Pallet jack operator
        {'pos': [0.0, 6.0, -math.pi/2], 'goal': [0.0, -6.0], 'radius': 0.40, 'speed': 0.70, 'role': 'pallet_operator'},
        # Aisle Inspector
        {'pos': [8.5, -2.0, math.pi/2], 'goal': [8.5, 4.0], 'radius': 0.30, 'speed': 0.50, 'role': 'inspector'},
    ]

    diff_mods = {
        'EASY': {'density_mult': 0.6, 'speed_mult': 0.8, 'time_limit': 450, 'extra_humans': 0},
        'MEDIUM': {'density_mult': 1.0, 'speed_mult': 1.0, 'time_limit': 380, 'extra_humans': 2},
        'HARD': {'density_mult': 1.6, 'speed_mult': 1.3, 'time_limit': 280, 'extra_humans': 5},
    }

    return ScenarioConfig(
        scenario_id="kiranakart",
        name="KiranaKart - Warehouse Fulfillment AGV",
        description="Automated Guided Vehicle navigating orthogonal high-bay shelf racks, blind intersections, cross-aisle pickers, and fast forklifts.",
        world_bounds=bounds,
        robot_start=start,
        robot_goal=goal,
        static_obstacles=obstacles,
        base_humans=base_humans,
        difficulty_modifiers=diff_mods,
        time_limit_steps=380
    )


# ==============================================================================
# SCENARIO 4: Diwali Bazaar Surat - Festival Market Rush
# ==============================================================================
def create_diwalibazaar_config():
    # Long, dense Indian festival market street: 24m x 10m (SVNIT Surat local flavour)
    bounds = [-12.0, 12.0, -5.0, 5.0]
    # Western Market Entrance -> Eastern Sweets & Lanterns Square
    start = [-10.5, 0.0, 0.0]
    goal = [10.5, 0.0]

    obstacles = [
        # Street Perimeter Walls
        [(-12.0, -5.0), (12.0, -5.0), (12.0, 5.0), (-12.0, 5.0)],
        # North Side Market Stalls (Rangoli, Diya, and Sweets Stalls)
        make_box_obstacle(-8.0, 3.6, 2.8, 1.6),
        make_box_obstacle(-3.0, 3.6, 3.2, 1.6),
        make_box_obstacle(2.5, 3.6, 3.0, 1.6),
        make_box_obstacle(7.5, 3.6, 3.2, 1.6),
        # South Side Stalls (Textiles, Jewelry, Firecracker vendors)
        make_box_obstacle(-6.0, -3.6, 3.0, 1.6),
        make_box_obstacle(-0.5, -3.6, 3.5, 1.6),
        make_box_obstacle(5.0, -3.6, 3.0, 1.6),
        # Mid-Street Temporary Display Kiosks & Light Poles
        make_pillar_obstacle(-2.0, 0.8, 0.35),
        make_pillar_obstacle(3.0, -0.8, 0.35),
    ]

    base_humans = [
        # Dense Bidirectional Flow (Counter-flow shoppers)
        {'pos': [10.0, 0.8, math.pi], 'goal': [-10.0, 0.8], 'radius': 0.30, 'speed': 0.90, 'role': 'pedestrian_counter'},
        {'pos': [8.0, -0.6, math.pi], 'goal': [-9.0, -0.6], 'radius': 0.30, 'speed': 0.80, 'role': 'pedestrian_counter'},
        {'pos': [-7.0, -0.8, 0.0], 'goal': [9.0, -0.8], 'radius': 0.30, 'speed': 0.85, 'role': 'pedestrian_forward'},
        {'pos': [-5.0, 0.6, 0.0], 'goal': [8.0, 0.6], 'radius': 0.30, 'speed': 0.75, 'role': 'pedestrian_forward'},
        # Stationary Bargaining Clusters (Temporary Human Obstacles at Stalls)
        {'pos': [-3.0, 2.2, -math.pi/2], 'goal': [-3.0, 2.4], 'radius': 0.32, 'speed': 0.10, 'role': 'stall_bargainer'},
        {'pos': [-2.4, 2.3, math.pi], 'goal': [-2.3, 2.2], 'radius': 0.32, 'speed': 0.10, 'role': 'stall_bargainer'},
        {'pos': [5.0, -2.2, math.pi/2], 'goal': [5.0, -2.4], 'radius': 0.32, 'speed': 0.10, 'role': 'stall_bargainer'},
        {'pos': [5.6, -2.1, 0.0], 'goal': [5.5, -2.2], 'radius': 0.32, 'speed': 0.10, 'role': 'stall_bargainer'},
        # Sweet vendor with cart
        {'pos': [1.0, 0.0, math.pi], 'goal': [-9.0, 0.0], 'radius': 0.45, 'speed': 0.55, 'role': 'cart_vendor'},
    ]

    diff_mods = {
        'EASY': {'density_mult': 0.5, 'speed_mult': 0.75, 'time_limit': 450, 'extra_humans': 0},
        'MEDIUM': {'density_mult': 1.0, 'speed_mult': 1.0, 'time_limit': 380, 'extra_humans': 3},
        'HARD': {'density_mult': 1.7, 'speed_mult': 1.25, 'time_limit': 280, 'extra_humans': 7},
    }

    return ScenarioConfig(
        scenario_id="diwalibazaar",
        name="Diwali Bazaar Surat - Festival Market Rush",
        description="High-density festive Indian market street with continuous counter-flowing crowds, roadside stall kiosks, and stationary bargaining clusters.",
        world_bounds=bounds,
        robot_start=start,
        robot_goal=goal,
        static_obstacles=obstacles,
        base_humans=base_humans,
        difficulty_modifiers=diff_mods,
        time_limit_steps=380
    )


# ==============================================================================
# SCENARIO 5: Platform 1 - Railway Station Rush
# ==============================================================================
def create_railwaystation_config():
    # Long train passenger platform: 26m x 10m
    bounds = [-13.0, 13.0, -5.0, 5.0]
    # Platform Entrance (Luggage drop) -> Exit Escalator
    start = [-11.0, 2.0, 0.0]
    goal = [11.0, 2.0]

    obstacles = [
        # Platform Boundaries (North: Waiting lounge wall, South: Platform edge track)
        [(-13.0, -5.0), (13.0, -5.0), (13.0, 5.0), (-13.0, 5.0)],
        # Central Roof Support Columns
        make_pillar_obstacle(-8.0, 0.0, 0.5),
        make_pillar_obstacle(-2.5, 0.0, 0.5),
        make_pillar_obstacle(3.0, 0.0, 0.5),
        make_pillar_obstacle(8.5, 0.0, 0.5),
        # Platform Info Display & Snack Kiosks
        make_box_obstacle(-5.0, 3.2, 2.2, 1.2),
        make_box_obstacle(5.5, 3.2, 2.2, 1.2),
        make_box_obstacle(0.0, -3.2, 3.0, 1.2),
    ]

    base_humans = [
        # Train Arrival Passengers (Flowing from train doors at y=-4.5 toward station exit at y=+3.5)
        {'pos': [-7.0, -4.0, math.pi/2], 'goal': [-4.0, 4.0], 'radius': 0.32, 'speed': 1.10, 'role': 'surge_commuter'},
        {'pos': [-6.0, -4.2, math.pi/2], 'goal': [-3.0, 4.0], 'radius': 0.30, 'speed': 1.20, 'role': 'surge_commuter'},
        {'pos': [1.0, -4.0, math.pi/2], 'goal': [4.0, 4.0], 'radius': 0.32, 'speed': 1.15, 'role': 'surge_commuter'},
        {'pos': [2.2, -4.2, math.pi/2], 'goal': [5.0, 4.0], 'radius': 0.30, 'speed': 1.30, 'role': 'surge_commuter'},
        # Commuter walking with rolling luggage (larger effective footprint)
        {'pos': [6.0, 1.5, math.pi], 'goal': [-10.0, 1.5], 'radius': 0.50, 'speed': 0.85, 'role': 'luggage_traveler'},
        # Station staff agent
        {'pos': [-1.0, 1.0, 0.0], 'goal': [7.0, 1.0], 'radius': 0.30, 'speed': 0.95, 'role': 'station_staff'},
    ]

    diff_mods = {
        'EASY': {'density_mult': 0.6, 'speed_mult': 0.8, 'time_limit': 450, 'extra_humans': 0},
        'MEDIUM': {'density_mult': 1.0, 'speed_mult': 1.0, 'time_limit': 380, 'extra_humans': 3},
        'HARD': {'density_mult': 1.8, 'speed_mult': 1.3, 'time_limit': 280, 'extra_humans': 6},
    }

    return ScenarioConfig(
        scenario_id="railwaystation",
        name="Platform 1 - Railway Station Rush",
        description="Train arrival passenger surge flowing across a long platform with pillars and luggage travelers heading for station exits.",
        world_bounds=bounds,
        robot_start=start,
        robot_goal=goal,
        static_obstacles=obstacles,
        base_humans=base_humans,
        difficulty_modifiers=diff_mods,
        time_limit_steps=380
    )


# ==============================================================================
# SCENARIO 6: Fire Drill - Evacuation Counter-Flow (Stretch Scenario)
# ==============================================================================
def create_firedrill_config():
    # Building main corridor / exit hall: 22m x 10m
    bounds = [-11.0, 11.0, -5.0, 5.0]
    # Robot ingress (delivering first-aid kit inside) -> Deep inside building
    start = [-9.5, 0.0, 0.0]
    goal = [9.5, 0.0]

    obstacles = [
        # Hallway Boundary
        [(-11.0, -5.0), (11.0, -5.0), (11.0, 5.0), (-11.0, 5.0)],
        # Office partitions and fire doors creating bottlenecks
        make_box_obstacle(-3.0, 3.0, 1.2, 3.0),
        make_box_obstacle(-3.0, -3.0, 1.2, 3.0),
        make_box_obstacle(4.0, 2.5, 1.2, 3.0),
        make_box_obstacle(4.0, -2.5, 1.2, 3.0),
    ]

    # Every human is evacuating in pure counter-flow towards the exit (x=-10.0)
    base_humans = [
        {'pos': [8.0, 0.5, math.pi], 'goal': [-10.5, 0.5], 'radius': 0.32, 'speed': 1.35, 'role': 'evacuee'},
        {'pos': [7.0, -0.6, math.pi], 'goal': [-10.5, -0.6], 'radius': 0.32, 'speed': 1.45, 'role': 'evacuee'},
        {'pos': [3.0, 0.2, math.pi], 'goal': [-10.5, 0.0], 'radius': 0.32, 'speed': 1.40, 'role': 'evacuee'},
        {'pos': [1.0, -1.0, math.pi], 'goal': [-10.5, -0.8], 'radius': 0.30, 'speed': 1.50, 'role': 'evacuee'},
        {'pos': [-1.0, 0.8, math.pi], 'goal': [-10.5, 0.5], 'radius': 0.32, 'speed': 1.30, 'role': 'evacuee'},
        {'pos': [5.5, 1.0, math.pi], 'goal': [-10.5, 0.8], 'radius': 0.30, 'speed': 1.40, 'role': 'evacuee'},
    ]

    diff_mods = {
        'EASY': {'density_mult': 0.6, 'speed_mult': 0.8, 'time_limit': 400, 'extra_humans': 0},
        'MEDIUM': {'density_mult': 1.0, 'speed_mult': 1.0, 'time_limit': 320, 'extra_humans': 2},
        'HARD': {'density_mult': 1.7, 'speed_mult': 1.3, 'time_limit': 240, 'extra_humans': 5},
    }

    return ScenarioConfig(
        scenario_id="firedrill",
        name="Fire Drill - Evacuation Counter-Flow",
        description="Emergency evacuation stress test: complete counter-flow of rushing occupants towards building exit while robot navigates inwards through structural bottlenecks.",
        world_bounds=bounds,
        robot_start=start,
        robot_goal=goal,
        static_obstacles=obstacles,
        base_humans=base_humans,
        difficulty_modifiers=diff_mods,
        time_limit_steps=320
    )

# ==============================================================================
# SCENARIO 7: Hotel Assist - Room Service Delivery Robot
# ==============================================================================
def create_hotelroomservice_config():
    # Hotel floor: 24m x 16m
    bounds = [-12.0, 12.0, -8.0, 8.0]

    # Kitchen service station -> Guest Room 204
    start = [-10.0, 0.0, 0.0]
    goal = [10.0, 0.0]

    obstacles = [
        # Outer hotel boundary
        [(-12.0, -8.0), (12.0, -8.0),
         (12.0, 8.0), (-12.0, 8.0)],

        # Reception desk
        make_box_obstacle(-8.0, 4.8, 3.0, 1.0),

        # Lobby seating
        make_box_obstacle(-5.0, -5.5, 3.0, 1.2),
        make_box_obstacle(-1.0, -5.5, 2.5, 1.2),

        # North-side guest room partitions
        make_box_obstacle(-5.0, 4.0, 2.0, 1.0),
        make_box_obstacle(0.0, 4.0, 2.0, 1.0),
        make_box_obstacle(5.0, 4.0, 2.0, 1.0),

        # South-side guest room partitions
        make_box_obstacle(-3.0, -4.0, 2.0, 1.0),
        make_box_obstacle(2.0, -4.0, 2.0, 1.0),
        make_box_obstacle(7.0, -4.0, 2.0, 1.0),

        # Housekeeping storage area
        make_box_obstacle(9.0, 6.0, 2.0, 1.5),
    ]

    base_humans = [
        # Hotel guest walking along the corridor
        {'pos': [-3.0, 1.0, 0.0],
         'goal': [4.0, 1.0],
         'radius': 0.30, 'speed': 0.65,
         'role': 'pedestrian_guest'},

        # Guest crossing the corridor
        {'pos': [3.0, -2.0, math.pi/2],
         'goal': [3.0, 2.0],
         'radius': 0.30, 'speed': 0.55,
         'role': 'pedestrian_guest'},

        # Housekeeping staff with trolley
        {'pos': [-1.0, -1.5, 0.0],
         'goal': [6.0, -1.5],
         'radius': 0.42, 'speed': 0.45,
         'role': 'pedestrian_housekeeping'},

        # Guest moving in the opposite direction
        {'pos': [8.0, 1.5, math.pi],
         'goal': [-6.0, 1.5],
         'radius': 0.30, 'speed': 0.75,
         'role': 'pedestrian_guest'},
    ]

    diff_mods = {
        'EASY': {
            'density_mult': 0.6,
            'speed_mult': 0.8,
            'time_limit': 450,
            'extra_humans': 0
        },
        'MEDIUM': {
            'density_mult': 1.0,
            'speed_mult': 1.0,
            'time_limit': 380,
            'extra_humans': 2
        },
        'HARD': {
            'density_mult': 1.5,
            'speed_mult': 1.25,
            'time_limit': 280,
            'extra_humans': 4
        }
    }

    return ScenarioConfig(
        scenario_id="hotelroomservice",
        name="Hotel Assist - Room Service Delivery Robot",
        description="A hotel service robot delivering room-service orders through guest corridors while navigating around hotel guests and housekeeping staff.",
        world_bounds=bounds,
        robot_start=start,
        robot_goal=goal,
        static_obstacles=obstacles,
        base_humans=base_humans,
        difficulty_modifiers=diff_mods,
        time_limit_steps=380
    )

# Registry of all custom scenarios
SCENARIO_REGISTRY = {
    "mallmate": create_mallmate_config,
    "medirunner": create_medirunner_config,
    "kiranakart": create_kiranakart_config,
    "diwalibazaar": create_diwalibazaar_config,
    "railwaystation": create_railwaystation_config,
    "firedrill": create_firedrill_config,
    "hotelroomservice": create_hotelroomservice_config,
}


def get_scenario_config(scenario_id):
    """Retrieves an instantiated scenario config by id."""
    key = scenario_id.lower().replace("-", "").replace("_", "")
    for reg_key, factory in SCENARIO_REGISTRY.items():
        if key == reg_key:
            return factory()
    raise KeyError("Unknown scenario: {}. Available: {}".format(
        scenario_id, list(SCENARIO_REGISTRY.keys())
    ))

