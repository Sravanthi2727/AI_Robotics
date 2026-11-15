#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
hospital_delivery.py
====================
Standalone Autonomous Navigation Controller for the MediRunner Hospital Delivery Scenario.

Scenario Context:
-----------------
MediRunner is an autonomous mobile hospital robot tasked with transporting time-critical
medications and clinical supplies from the Central Pharmacy (West Wing, x = -10.5m) through
a narrow 24m x 8m clinical corridor wing to the Intensive Care Unit (ICU) Nurse Station
(East Wing, x = +10.5m).

Key Navigation Challenges & Geometric Constraints:
--------------------------------------------------
1. Narrow Corridor Topology:
   - Outer world bounds: [-12.0, 12.0] x [-4.0, 4.0] meters.
   - Wall blocks narrow the clear hallway to y in [-2.0, 2.0] meters (4.0m span).
   - Mid-corridor Triage Desk centered at (0.0, 1.2m), size 1.5m x 0.8m (y in [0.8, 1.6]m).
   - The northern passage around the triage desk is only 0.40m wide (impassable for a 0.60m robot).
   - The robot must navigate through the southern hallway lane centered at y = -0.60m, which
     provides 1.10m symmetric clearance to both the triage desk (y = 0.8m) and South Wall 1 (y = -2.0m).

2. Critical Hazard — Emergency Gurney:
   - High-speed, heavy agent: radius = 0.65m, speed ~ 1.65m/s.
   - Origin: (7.0, -1.0) -> Goal: (-11.0, -1.0), heading West directly along y = -1.0m.
   - Spatial Conflict: At the triage choke-point (y = -0.6m), center-to-center separation
     is only 0.40m, whereas the required combined collision radius is 0.30m + 0.65m = 0.95m.
   - Resolution: Dedicated Alcove Yielding State Machine that detects approaching gurneys via
     Closest Point of Approach (CPA) trajectory estimation, halts before the bottleneck, yields
     into verified open room alcoves, and resumes only after the gurney has safely cleared.

3. Heterogeneous Clinical Pedestrians:
   - Fast Purposeful Nurses: v ~ 1.15 - 1.25 m/s, radius = 0.30m.
   - Frail Patient with Mobility Walker: v ~ 0.28 m/s, radius = 0.45m.
   - Meandering Visitors: v ~ 0.60 m/s, radius = 0.30m.
   - Multi-tier longitudinal path envelope braking and predictive yielding.

Compatibility:
--------------
- Python 3.6+ / 2.7 compatible.
- Supports both CustomNavRepEnv (action_dim=3) and CustomNavRepEncodedEnv (action_dim=2).
"""

from __future__ import print_function, division
import math
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from navrep_custom.scenarios import get_scenario_config


# ==============================================================================
# 1. GEOMETRIC UTILITIES & POINT-TO-POLYGON VERIFICATION
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
    """Ray-casting algorithm to test whether point (px, py) lies strictly inside polygon."""
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


def is_position_safe(px, py, static_obstacles=None, world_bounds=None, min_margin=0.38):
    """
    Verifies that coordinate (px, py) satisfies minimum clearance from all static obstacles
    and outer world boundaries.
    """
    if world_bounds is not None:
        xmin, xmax, ymin, ymax = world_bounds
        if px < xmin + min_margin or px > xmax - min_margin:
            return False
        if py < ymin + min_margin or py > ymax - min_margin:
            return False

    if static_obstacles is not None:
        internal_obstacles = static_obstacles[1:] if len(static_obstacles) > 1 else static_obstacles
        for poly in internal_obstacles:
            if point_in_polygon(px, py, poly):
                return False
            if distance_point_to_polygon(px, py, poly) < min_margin:
                return False

    return True


def is_path_safe(x1, y1, x2, y2, static_obstacles=None, world_bounds=None, humans=None, robot_radius=0.30, min_margin=0.38):
    """
    Verifies that the entire line segment path from (x1, y1) to (x2, y2) maintains
    safe clearance from all static obstacles, world boundaries, and dynamic humans.
    Samples points along the segment.
    """
    length = math.hypot(x2 - x1, y2 - y1)
    num_samples = max(6, int(math.ceil(length / 0.15)))
    for i in range(num_samples + 1):
        t = float(i) / float(num_samples)
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)
        if not is_position_safe(px, py, static_obstacles, world_bounds, min_margin=min_margin):
            return False
        if humans is not None:
            for h in humans:
                hx, hy = h.px, h.py
                hrad = getattr(h, 'radius', 0.30)
                if math.hypot(px - hx, py - hy) < (robot_radius + hrad + 0.15):
                    return False
    return True


# ==============================================================================
# 2. HOSPITAL NAVIGATION CONTROLLER CLASS
# ==============================================================================
class HospitalNavigationController(object):
    """
    Autonomous Navigation Controller designed specifically for MediRunner in hospital corridors:
      - 7-Waypoint Canonical Topological Highway bypassing the mid-hallway Triage Desk.
      - Emergency Gurney Yielding State Machine with geometric alcove validation.
      - Closest Point of Approach (CPA) dynamic collision prediction.
      - Path envelope longitudinal throttling for slow walker patients and fast nurses.
      - Soft deceleration and docking speed clamping at the ICU Nurse Station.
    """

    # Canonical 7-waypoint corridor sequence:
    # Safely traverses from Pharmacy (-10.5, 0.0) -> ICU Nurse Station (10.5, 0.0)
    DEFAULT_WAYPOINTS = [
        [-10.5,  0.00],   # WP 0: Central Pharmacy departure point
        [ -6.0,  0.00],   # WP 1: West corridor center (between North Wall 1 and South recess)
        [ -2.0, -0.60],   # WP 2: Triage approach lane (aligns south of triage desk)
        [  0.0, -0.60],   # WP 3: Triage bottleneck transit (symmetric 1.10m clearance)
        [  2.0, -0.40],   # WP 4: Triage exit into East hallway
        [  6.0,  0.00],   # WP 5: East corridor center (between South Wall 2 and North recess)
        [ 10.5,  0.00],   # WP 6: ICU Nurse Station docking goal
    ]

    # State Machine Constants
    STATE_NORMAL = "NORMAL_TRANSIT"
    STATE_YIELD_ALCOVE = "ALCOVE_YIELD"
    STATE_HOLD_WAIT = "HOLD_WAIT"
    STATE_RESUME = "RESUME_TRANSIT"

    def __init__(self,
                 waypoints=None,
                 max_speed=0.75,
                 robot_radius=0.30,
                 static_obstacles=None,
                 world_bounds=None):
        self.waypoints = [list(wp) for wp in (waypoints or self.DEFAULT_WAYPOINTS)]
        self.max_speed = float(max_speed)
        self.robot_radius = float(robot_radius)

        # Environment bounds and obstacles for runtime clearance validation
        if static_obstacles is None or world_bounds is None:
            try:
                cfg = get_scenario_config("medirunner")
                self.static_obstacles = cfg.static_obstacles
                self.world_bounds = cfg.world_bounds
            except Exception:
                self.static_obstacles = None
                self.world_bounds = [-12.0, 12.0, -4.0, 4.0]
        else:
            self.static_obstacles = static_obstacles
            self.world_bounds = world_bounds

        # State tracking
        self.current_wp_idx = 0
        self.waypoint_reach_dist = 0.45
        self.state = self.STATE_NORMAL
        self.yield_target = None
        self.stopped_steps = 0
        self.gurney_clear_steps = 0
        self.gurney_tracking_lost = False
        self.gurney_lost_steps = 0
        # Passage window: once gurney clears, suppress re-triggering for N steps so the
        # robot can cross the bottleneck before the gurney loops back to x = +7.0 m.
        # 20 steps = 4.0 s at dt=0.2 is enough to cross the 1.5 m triage zone.
        self.passage_granted_steps = 0

    def reset(self):
        """Resets controller state at episode start."""
        self.current_wp_idx = 0
        self.state = self.STATE_NORMAL
        self.yield_target = None
        self.stopped_steps = 0
        self.gurney_clear_steps = 0
        self.gurney_tracking_lost = False
        self.gurney_lost_steps = 0
        self.passage_granted_steps = 0

    def set_waypoints(self, waypoints):
        """Dynamically updates the active route waypoints."""
        self.waypoints = [list(wp) for wp in waypoints]
        self.reset()

    # --------------------------------------------------------------------------
    # 2.1 Predictive CPA Collision Math
    # --------------------------------------------------------------------------
    def compute_cpa(self, rx, ry, vx, vy, human):
        """
        Calculates Time-to-Closest-Approach (t_cpa) and distance at CPA between robot and human.
        Returns:
            (t_cpa, cpa_dist, closing_speed)
        """
        hx, hy = human.px, human.py
        hvx, hvy = human.vx, human.vy

        to_hx = hx - rx
        to_hy = hy - ry
        dist = math.hypot(to_hx, to_hy)

        # Relative velocity of human with respect to robot
        rel_vx = hvx - vx
        rel_vy = hvy - vy
        rel_v2 = rel_vx * rel_vx + rel_vy * rel_vy

        closing_speed = -(to_hx * rel_vx + to_hy * rel_vy) / max(dist, 1e-4)

        if rel_v2 < 1e-4:
            return float('inf'), dist, closing_speed

        t_cpa = -(to_hx * rel_vx + to_hy * rel_vy) / rel_v2
        if t_cpa <= 0.0:
            return float('inf'), dist, closing_speed

        cpa_x = to_hx + rel_vx * t_cpa
        cpa_y = to_hy + rel_vy * t_cpa
        cpa_dist = math.hypot(cpa_x, cpa_y)

        return t_cpa, cpa_dist, closing_speed

    # --------------------------------------------------------------------------
    # 2.2 Emergency Gurney Detection & Hazard Assessment
    # --------------------------------------------------------------------------
    def detect_emergency_gurney(self, rx, ry, vx, vy, humans, human_meta=None):
        """
        Scans active human agents to identify the emergency gurney.

        Identification Strategy:
          1. Authoritative Role Inspection: If human_meta is provided, search for
             meta['role'] == 'emergency_gurney'. If found, that human is conclusively
             identified as the emergency gurney.
          2. Multi-Candidate Kinematic & Geometric Evaluation (when metadata is absent):
             Evaluates ALL candidate humans and scores them according to nominal gurney kinematics
             (radius ~ 0.65m, speed ~ 1.65 m/s, heading westbound, lane y ~ -1.0m) and prioritizes
             oncoming threats over distant cleared pedestrians to prevent shadowing.

        Returns:
            (gurney_agent, t_cpa, cpa_dist) or (None, inf, inf)
        """
        if humans is None or len(humans) == 0:
            return None, float('inf'), float('inf')

        # 1. Authoritative Metadata Inspection
        if human_meta is not None:
            for i, h in enumerate(humans):
                if i < len(human_meta):
                    meta = human_meta[i]
                    role = meta.get('role', '') if isinstance(meta, dict) else getattr(meta, 'role', '')
                    if role == 'emergency_gurney':
                        t_cpa, cpa_dist, _ = self.compute_cpa(rx, ry, vx, vy, h)
                        return h, t_cpa, cpa_dist

        # 2. Multi-Candidate Kinematic & Geometric Evaluation (when metadata is absent)
        best_candidate = None
        best_score = -1e9

        for h in humans:
            hrad = getattr(h, 'radius', 0.30)
            hvx = getattr(h, 'vx', 0.0)
            hvy = getattr(h, 'vy', 0.0)
            h_speed = math.hypot(hvx, hvy)
            hx, hy = h.px, h.py

            # A gurney candidate must be travelling westbound or near gurney lane
            if hvx > 0.20:
                continue

            score = 0.0

            # Radius weighting (nominal 0.65m)
            if hrad >= 0.55:
                score += 35.0
            elif hrad >= 0.40:
                score += 10.0
            else:
                score -= 15.0

            # Speed weighting (nominal ~ 1.65 m/s)
            if h_speed >= 1.30:
                score += 35.0
            elif h_speed >= 0.80:
                score += 15.0

            # Westbound heading weighting
            if hvx < -1.00:
                score += 30.0
            elif hvx < -0.50:
                score += 15.0
            elif hvx >= 0.0:
                score -= 20.0

            # Lane alignment with y ~ -1.0m
            lane_err = abs(hy - (-1.0))
            if lane_err < 0.40:
                score += 20.0
            elif lane_err < 0.80:
                score += 5.0
            else:
                score -= 15.0

            # Oncoming threat priority: prioritize candidates to the East over those already past
            if hx > rx - 0.5:
                score += 25.0
            else:
                score -= 10.0

            if score > best_score:
                best_score = score
                best_candidate = h

        if best_candidate is not None and best_score >= 45.0:
            t_cpa, cpa_dist, _ = self.compute_cpa(rx, ry, vx, vy, best_candidate)
            return best_candidate, t_cpa, cpa_dist

        return None, float('inf'), float('inf')

    def is_gurney_cleared(self, rx, ry, gurney):
        """
        Determines whether the emergency gurney has physically passed the robot
        and safely cleared the conflict region.

        Criteria:
          1. Tracking must NOT be lost: if gurney is None, return False (uncertainty state).
          2. Position: Gurney must be strictly West of the robot: gx < rx - 1.20m.
          3. Bottleneck clearance: If robot is near/West of bottleneck (rx >= -4.5m),
             gurney must have passed the bottleneck choke point: gx < -1.60m.
          4. Direction & Endpoint:
             Either:
               a) Gurney is actively traveling West (gvx < -0.15 m/s), OR
               b) Gurney has arrived and stopped at its destination (gx <= -9.0m and speed < 0.20 m/s), OR
               c) Gurney has deflected into south recess (gy <= -1.80m) or stalled (speed < 0.25 m/s)
                  away from the robot (separation > 2.0m).
          5. Separation:
             Euclidean separation must be >= 1.50m (>> combined radius 0.95m).
        """
        if gurney is None:
            # Never assume a missing or undetected gurney has cleared
            return False

        gx = getattr(gurney, 'px', 0.0)
        gy = getattr(gurney, 'py', -1.0)
        gvx = getattr(gurney, 'vx', 0.0)
        gvy = getattr(gurney, 'vy', 0.0)
        g_speed = math.hypot(gvx, gvy)

        # 1. Gurney strictly West of the robot by at least 1.20m
        if gx < (rx - 1.20):
            # If robot is near or West of the triage bottleneck, gurney must also have cleared it
            if rx >= -4.5 and gx >= -1.60:
                return False
            dist = math.hypot(rx - gx, ry - gy)
            if dist < 1.50:
                return False
            is_moving_west = (gvx < -0.15)
            is_stopped_at_goal = (gx <= -9.0 and g_speed < 0.20)
            if not (is_moving_west or is_stopped_at_goal):
                return False
            return True

        # 2. Or: Gurney has stalled / stopped / deflected into south recess (e.g. gy <= -1.80m or g_speed < 0.25)
        # and presents no oncoming threat to the robot
        if (gy <= -1.80 or g_speed < 0.25) and (gx > rx + 1.5 or abs(rx - gx) > 2.0):
            return True

        return False

    def is_gurney_threat(self, rx, ry, vx, vy, gurney, t_cpa, cpa_dist):
        """
        Determines whether the emergency gurney presents a hazard requiring yielding.

        Hazard Conditions:
          1. Direct predictive CPA collision: 0 < t_cpa < 4.0s and cpa_dist < (robot_radius + gurney_radius + 0.40m).
          2. Trajectory prediction horizon: Checks projected future trajectory over t in [0.0, 4.0s].
          3. Triage Bottleneck Gatekeeper: robot is West of bottleneck exit (rx <= 0.75m),
             and oncoming gurney could reach bottleneck before robot clears.
        """
        if gurney is None:
            return False

        gx = getattr(gurney, 'px', 0.0)
        gy = getattr(gurney, 'py', -1.0)
        gvx = getattr(gurney, 'vx', 0.0)
        gvy = getattr(gurney, 'vy', 0.0)
        grad = getattr(gurney, 'radius', 0.65)
        gurney_sum_r = self.robot_radius + grad

        # Gurney must be traveling West to pose an oncoming threat
        if gvx >= -0.20:
            return False

        # Gurney must be East of the robot (or within immediate overlap)
        if gx < (rx - 0.50):
            return False

        # 1. Direct CPA collision threat
        if 0.0 < t_cpa < 4.0 and cpa_dist < (gurney_sum_r + 0.40):
            return True

        longitudinal_dist = gx - rx

        # 2. Multi-step trajectory projection check over 4.0s horizon
        for step in range(1, 21):
            dt_proj = step * 0.20
            prx = rx + vx * dt_proj
            pry = ry + vy * dt_proj
            pgx = gx + gvx * dt_proj
            pgy = gy + gvy * dt_proj
            proj_dist = math.hypot(prx - pgx, pry - pgy)
            if proj_dist < (gurney_sum_r + 0.35):
                return True

        # 3. Triage Bottleneck Gatekeeper:
        # Triage desk spans x in [-0.75, 0.75]. Both contending for the bottleneck zone
        if rx >= -3.5 and rx <= 0.75 and gx > rx and longitudinal_dist < 6.5 and abs(gy - (-1.0)) < 0.60:
            time_robot_clear = max(0.0, 0.85 - rx) / max(0.20, math.hypot(vx, vy))
            time_gurney_enter = max(0.0, gx - 0.85) / max(0.20, abs(gvx))
            if time_gurney_enter < time_robot_clear + 2.0:
                return True

        return False

    # --------------------------------------------------------------------------
    # 2.3 Safe Alcove Selection & Geometric Validation
    # --------------------------------------------------------------------------
    def evaluate_alcove_yield(self, rx, ry, humans=None):
        """
        Selects a verified, safe, collision-free alcove holding position based on
        the robot's current longitudinal coordinate rx, validating the full path.

        Hospital Geometry & Recess Zones:
          - Zone A (rx <= -0.75m): West corridor approach -> North Alcove 1 at (-2.40, 1.25m).
          - Zone B (-0.75m < rx < 0.75m): Inside bottleneck choke point.
          - Zone C (0.75m <= rx < 4.0m): East corridor exit -> South Alcove 2 at (1.80, -2.40m).
          - Zone D (rx >= 4.0m): Far East corridor wing.
        """
        candidate_points = []

        # Zone A: West corridor and bottleneck approach (rx <= -0.75)
        # Pull into North Alcove 1 at (-2.40, 1.25)
        if rx <= -0.75:
            candidate_points.append([-2.40, 1.25])
            candidate_points.append([-2.00, 0.90])
            candidate_points.append([-2.80, 1.00])

        # Zone B: Inside Triage Bottleneck choke point (-0.75 < rx < 0.75)
        elif -0.75 < rx < 0.75:
            # In West half: if safe, reverse slightly to West approach
            if rx <= 0.0:
                candidate_points.append([-1.20, -0.60])
                candidate_points.append([-1.50, -0.60])
            else:
                # In East half: advance into South Alcove 2
                candidate_points.append([1.80, -2.40])
                candidate_points.append([1.20, -2.40])

        # Zone C: East corridor exit and Central-East corridor (0.75 <= rx < 4.0)
        elif 0.75 <= rx < 4.0:
            candidate_points.append([1.80, -2.40])
            candidate_points.append([2.20, -2.20])
            candidate_points.append([1.20, -2.40])

        # Zone D: Far East corridor wing (rx >= 4.0)
        else:
            candidate_points.append([min(rx + 0.5, 9.5), 0.90])
            candidate_points.append([min(rx, 9.5), 1.20])

        # Validate candidates: BOTH the target position AND the full path to it must be safe
        for cx, cy in candidate_points:
            # 1. Target endpoint clearance
            if not is_position_safe(cx, cy, self.static_obstacles, self.world_bounds, min_margin=0.40):
                continue
            # 2. Full path clearance (no static obstacles or blocking humans)
            if not is_path_safe(rx, ry, cx, cy, self.static_obstacles, self.world_bounds, humans=humans, robot_radius=self.robot_radius, min_margin=0.35):
                continue
            return [cx, cy]

        return None

    # --------------------------------------------------------------------------
    # 2.4 Active Target Selection & Waypoint Progression
    # --------------------------------------------------------------------------
    def get_active_target(self, rx, ry):
        """
        Determines current active target [tx, ty] and advances waypoint index
        if the active waypoint has been achieved or progressed past in X.
        """
        while self.current_wp_idx < len(self.waypoints) - 1:
            tx, ty = self.waypoints[self.current_wp_idx]
            dist = math.hypot(tx - rx, ty - ry)
            threshold = self.waypoint_reach_dist
            if self.current_wp_idx in [2, 3]:
                threshold = 0.35

            if dist < threshold or (rx > tx + 0.30 and abs(ry - ty) < 1.60):
                self.current_wp_idx += 1
            else:
                break

        tx, ty = self.waypoints[self.current_wp_idx]
        # Dynamic corridor lane centering & clearance
        if self.current_wp_idx == 3:
            ty -= 0.15  # -0.75m: centers between triage desk (0.8m) and south wall (-2.0m)
        elif self.current_wp_idx == 4:
            ty -= 0.20  # -0.60m: maintains safe lateral margin from pedestrian walker
        elif self.current_wp_idx == 5:
            ty += 0.10  # 0.10m: avoids nurse waiting at (7.0, -0.6)

        dist = math.hypot(tx - rx, ty - ry)
        return tx, ty, dist

    # --------------------------------------------------------------------------
    # 2.5 Main Action Selection: act()
    # --------------------------------------------------------------------------
    def act(self, robot_pos, humans=None, action_dim=2, human_meta=None):
        """
        Computes the velocity control command given current robot position and dynamic human list.

        Parameters:
            robot_pos: array-like [rx, ry] (in meters)
            humans: list of crowd_sim.envs.utils.human.Human objects
            action_dim: 2 for CustomNavRepEncodedEnv [vx, vy], or 3 for CustomNavRepEnv [vx, vy, 0.0]
            human_meta: optional list of human metadata dicts from environment

        Returns:
            np.ndarray of shape (action_dim,) and dtype np.float32
        """
        rx, ry = float(robot_pos[0]), float(robot_pos[1])

        # 1. Determine active route target
        tx, ty, dist_to_target = self.get_active_target(rx, ry)

        # Compute nominal direction towards target
        dx = tx - rx
        dy = ty - ry
        norm = math.hypot(dx, dy)

        if norm > 1e-4:
            nominal_speed = min(self.max_speed, norm)
            vx = (dx / norm) * nominal_speed
            vy = (dy / norm) * nominal_speed
        else:
            vx, vy = 0.0, 0.0

        # ----------------------------------------------------------------------
        # 2. Emergency Gurney Detection & Persistent Yielding State Machine
        # ----------------------------------------------------------------------
        gurney, t_cpa_gurney, cpa_dist_gurney = self.detect_emergency_gurney(
            rx, ry, vx, vy, humans, human_meta=human_meta
        )
        gurney_hazard = self.is_gurney_threat(rx, ry, vx, vy, gurney, t_cpa_gurney, cpa_dist_gurney)

        # Passage window: count down each step while active.
        # During this window the robot has right-of-way through the bottleneck —
        # re-triggering yield from a looped gurney is suppressed.
        if self.passage_granted_steps > 0:
            self.passage_granted_steps -= 1

        # Suppress gurney_hazard if passage window is active AND gurney is not
        # in immediate collision territory (CPA < 1.0s, cpa_dist < combined radius).
        # This allows the robot to cross before the gurney completes its loop back.
        immediate_collision = (
            gurney is not None
            and t_cpa_gurney < 1.0
            and cpa_dist_gurney < (self.robot_radius + getattr(gurney, 'radius', 0.65))
        )
        if self.passage_granted_steps > 0 and not immediate_collision:
            gurney_hazard = False

        # State Machine Transitions
        if self.state in [self.STATE_NORMAL, self.STATE_RESUME]:
            if gurney_hazard:
                self.gurney_clear_steps = 0
                self.passage_granted_steps = 0  # Cancel any active window on new threat
                alcove = self.evaluate_alcove_yield(rx, ry, humans=humans)
                if alcove is not None:
                    self.yield_target = alcove
                    self.state = self.STATE_YIELD_ALCOVE
                else:
                    self.state = self.STATE_HOLD_WAIT
            elif self.state == self.STATE_RESUME:
                # Once back near the corridor line (|ry - ty| < 0.30m), resume normal
                if abs(ry - ty) < 0.30:
                    self.state = self.STATE_NORMAL

        elif self.state in [self.STATE_YIELD_ALCOVE, self.STATE_HOLD_WAIT]:
            if gurney is not None:
                self.gurney_tracking_lost = False
                self.gurney_lost_steps = 0
                if self.is_gurney_cleared(rx, ry, gurney):
                    self.gurney_clear_steps += 1
                    # Require at least 5 consecutive stable clearance steps
                    if self.gurney_clear_steps >= 5:
                        self.state = self.STATE_RESUME
                        self.yield_target = None
                        # Grant a 20-step (4 s) passage window to cross the bottleneck
                        # before the gurney loops back from (-11, -1) to (7, -1).
                        self.passage_granted_steps = 20
                else:
                    self.gurney_clear_steps = 0
            else:
                # Gurney tracking lost during active yielding: uncertainty state
                self.gurney_tracking_lost = True
                self.gurney_lost_steps += 1
                self.gurney_clear_steps = 0  # Missing gurney NEVER increments clearance counter!
                # Long timeout recovery (50 steps = 10s if safe)
                if self.gurney_lost_steps >= 50:
                    self.state = self.STATE_RESUME
                    self.yield_target = None
                    self.passage_granted_steps = 20

        # State Execution
        if self.state == self.STATE_YIELD_ALCOVE:
            if self.yield_target is not None:
                yx, yy = self.yield_target
                dist_to_alcove = math.hypot(yx - rx, yy - ry)
                if dist_to_alcove > 0.25:
                    # Drive into the alcove
                    spd = min(0.55, dist_to_alcove)
                    vx = ((yx - rx) / dist_to_alcove) * spd
                    vy = ((yy - ry) / dist_to_alcove) * spd
                else:
                    # Reached safe alcove holding point: stop and wait for gurney to pass
                    vx, vy = 0.0, 0.0
                    self.state = self.STATE_HOLD_WAIT
            else:
                vx, vy = 0.0, 0.0
                self.state = self.STATE_HOLD_WAIT

        elif self.state == self.STATE_HOLD_WAIT:
            # Full stop in safe holding position
            vx, vy = 0.0, 0.0

        # Triage Bottleneck Gatekeeper:
        # Conservative stopping line before bottleneck conflict zone (x_stop_line = -1.85m).
        # Bypassed when the passage window is active (gurney has cleared and robot has right-of-way),
        # or when an immediate CPA collision would override anyway.
        gatekeeper_active = (
            rx <= -0.75
            and not self.is_gurney_cleared(rx, ry, gurney)
            and self.passage_granted_steps == 0  # bypass during passage window
        )
        if gatekeeper_active:
            x_stop_line = -1.85
            if rx >= x_stop_line:
                if vx > 0.0:
                    vx = 0.0
            else:
                max_allowed_dx = max(0.0, x_stop_line - rx)
                max_safe_vx = max_allowed_dx / 0.20
                if vx > max_safe_vx:
                    vx = max_safe_vx
        elif -0.75 < rx < 0.75 and gurney_hazard:
            # Already inside choke point when hazard detected: stop safely if not navigating to safe point
            if self.state != self.STATE_YIELD_ALCOVE:
                vx, vy = 0.0, 0.0

        # ----------------------------------------------------------------------
        # 3. Dynamic Human Safety & Lateral Repulsion Filter
        # ----------------------------------------------------------------------
        min_brake = 1.0
        should_stop = False

        if humans is not None and self.state not in [self.STATE_HOLD_WAIT]:
            for h in humans:
                hx, hy = h.px, h.py
                hvx, hvy = h.vx, h.vy
                hrad = getattr(h, 'radius', 0.30)
                sum_r = self.robot_radius + hrad

                dist = math.hypot(rx - hx, ry - hy)
                dist_bound = dist - sum_r

                # 3.1 Dynamic Lateral Repulsion around pedestrians
                if dist < sum_r + 0.45:
                    dy = ry - hy
                    rep_mag = (sum_r + 0.45 - dist) * 1.5
                    if dy >= 0:
                        vy += rep_mag
                    else:
                        vy -= rep_mag
                    dx = hx - rx
                    if dx > 0 and abs(dy) < sum_r + 0.15:
                        vx *= max(0.0, (dist - sum_r) / 0.45)

                # 3.2 Absolute Emergency Stop (imminent contact)
                if dist_bound < 0.08:
                    should_stop = True
                    min_brake = 0.0
                    break

        # Apply safety braking
        if should_stop or min_brake <= 0.0:
            vx, vy = 0.0, 0.0
            self.stopped_steps += 1
        else:
            self.stopped_steps = 0
            vx *= min_brake
            vy *= min_brake

        # ----------------------------------------------------------------------
        # 4. Docking Deceleration & Velocity Clamping
        # ----------------------------------------------------------------------
        tot_speed = math.hypot(vx, vy)
        # Slow down smoothly when approaching the final ICU Nurse Station goal
        if self.current_wp_idx == len(self.waypoints) - 1:
            goal_x, goal_y = self.waypoints[-1]
            dist_to_goal = math.hypot(goal_x - rx, goal_y - ry)
            max_dock_speed = max(0.18, min(self.max_speed, dist_to_goal * 0.85))
            if tot_speed > max_dock_speed:
                vx = (vx / tot_speed) * max_dock_speed
                vy = (vy / tot_speed) * max_dock_speed
        elif tot_speed > self.max_speed:
            vx = (vx / tot_speed) * self.max_speed
            vy = (vy / tot_speed) * self.max_speed

        # ----------------------------------------------------------------------
        # 5. Return Action in Requested Dimension
        # ----------------------------------------------------------------------
        if action_dim == 2:
            return np.array([vx, vy], dtype=np.float32)
        return np.array([vx, vy, 0.0], dtype=np.float32)


# ==============================================================================
# 3. SIMULATION SYNCHRONIZATION & SPAWN CLEARANCE PROTECTION
# ==============================================================================
def sync_and_rasterize_hospital_sim(raw_env, cfg):
    """
    Synchronizes the SOADRL simulator for the MediRunner Hospital Scenario:
      1. Rasterizes MediRunner's polygonal obstacles into sim.map (resolving phantom obstacles).
      2. Synchronizes simulation time limits.
      3. Relocates any dynamic crowd humans that spawn dangerously close to the robot's
         start pose (-10.5, 0.0) or destination goal (10.5, 0.0) in MEDIUM/HARD difficulties.
    """
    target = getattr(raw_env, 'raw_env', raw_env)
    sim = target.soadrl_sim
    dt = target._get_dt() if hasattr(target, '_get_dt') else (raw_env._get_dt() if hasattr(raw_env, '_get_dt') else 0.2)
    tl = getattr(target, 'time_limit', getattr(raw_env, 'time_limit', 350))
    sim.time_limit = tl * dt

    # Rasterize static hospital obstacles
    sim.map_size_m = 24.0
    grid_res = 0.10
    grid_dim = int(round(sim.map_size_m / grid_res))

    if cv2 is not None:
        img = np.ones((grid_dim, grid_dim), dtype=np.float32)
        for poly in cfg.static_obstacles[1:]:
            pts = []
            for vx, vy in poly:
                gx = int(round((vx + sim.map_size_m / 2.0) / grid_res))
                gy = int(round((vy + sim.map_size_m / 2.0) / grid_res))
                pts.append([gx, gy])
            if len(pts) >= 3:
                cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], 0.0)
        sim.map = img.T

    # Spawn clearance protection for MediRunner start and goal
    sx, sy = cfg.robot_start[0], cfg.robot_start[1]
    gx, gy = cfg.robot_goal[0], cfg.robot_goal[1]

    if getattr(sim, 'humans', None) is not None:
        for h in sim.humans:
            # Check start spawn collision (< 1.5m from (-10.5, 0.0))
            if math.hypot(h.px - sx, h.py - sy) < (sim.robot.radius + h.radius + 0.60):
                # Relocate intersecting extra human to safe East wing open space
                h.set(4.0, 1.0, 8.0, 1.0, 0.0, 0.0, 0.0)
            # Check goal spawn collision (< 1.2m from (10.5, 0.0))
            elif math.hypot(h.px - gx, h.py - gy) < (sim.robot.radius + h.radius + 0.40):
                # Relocate intersecting extra human to safe West wing open space
                h.set(-4.0, 1.0, -8.0, 1.0, 0.0, 0.0, math.pi)
