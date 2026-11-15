#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_airport_scenario.py
========================
Isolated unit test suite for Airport Passenger Assistance scenario:
  - Verifies dynamic registration in SCENARIO_REGISTRY under 'airportpassengerassistance' and 'airport'
  - Validates world bounds, robot start/goal coordinates, and path separation
  - Validates start/goal differ strictly from hotel and baseline scenarios
  - Checks geometric non-overlap and clearance margins (start clearance >= 1.0 m, goal clearance >= 1.0 m)
  - Rigorously checks that all humans spawn at least 2.0 m away from the robot start position
  - Validates that a continuous free collision-free path exists from start to goal
  - Verifies that a stationary robot experiences zero collisions in the first 20 steps
  - Validates monotonic difficulty scaling across EASY, MEDIUM, and HARD
  - Verifies that AirportNavigationController successfully guides robot to ReachGoal
"""

import math
import unittest
import numpy as np

# Importing airport_passenger_assistance registers scenario into SCENARIO_REGISTRY
import navrep_custom.airport_passenger_assistance as airport_mod
from navrep_custom.airport_passenger_assistance import (
    point_to_segment_distance,
    point_in_polygon,
    distance_point_to_polygon,
    AirportNavigationController,
    AirportNavRepEnv,
)
from navrep_custom.scenarios import SCENARIO_REGISTRY, get_scenario_config
from crowd_sim.envs.utils.info import ReachGoal, Collision, CollisionOtherAgent, Timeout


class TestAirportScenario(unittest.TestCase):
    """Rigorous unit test suite for the Airport Passenger Assistance scenario."""

    def test_airport_registration(self):
        """Verify scenario self-registers under 'airportpassengerassistance' and 'airport'."""
        self.assertIn("airportpassengerassistance", SCENARIO_REGISTRY)
        self.assertIn("airport", SCENARIO_REGISTRY)

        cfg = get_scenario_config("airportpassengerassistance")
        self.assertEqual(cfg.scenario_id, "airportpassengerassistance")
        self.assertIn("Airport", cfg.name)
        self.assertTrue(len(cfg.description) > 20)

    def test_airport_world_bounds_and_waypoints(self):
        """Verify world bounds and start-to-goal Euclidean separation distance."""
        cfg = get_scenario_config("airportpassengerassistance")
        bounds = cfg.world_bounds
        sx, sy, sth = cfg.robot_start
        gx, gy = cfg.robot_goal

        # Check bounds
        self.assertTrue(bounds[0] <= sx <= bounds[1], "Start X out of bounds")
        self.assertTrue(bounds[2] <= sy <= bounds[3], "Start Y out of bounds")
        self.assertTrue(bounds[0] <= gx <= bounds[1], "Goal X out of bounds")
        self.assertTrue(bounds[2] <= gy <= bounds[3], "Goal Y out of bounds")

        # Start-to-goal separation
        dist = math.hypot(sx - gx, sy - gy)
        self.assertGreaterEqual(dist, 18.0, "Start-to-Goal distance ({:.2f}m) < 18.0m".format(dist))

    def test_start_and_goal_differ_from_hotel_and_baseline(self):
        """Verify start and goal coordinates differ from hotel and baseline scenarios."""
        cfg = get_scenario_config("airportpassengerassistance")
        sx, sy, _ = cfg.robot_start
        gx, gy = cfg.robot_goal

        # Hotel coordinates: start [-10.0, 0.0], goal [10.0, 0.0]
        hotel_start = [-10.0, 0.0]
        hotel_goal = [10.0, 0.0]
        self.assertGreater(math.hypot(sx - hotel_start[0], sy - hotel_start[1]), 1.0)
        self.assertGreater(math.hypot(gx - hotel_goal[0], gy - hotel_goal[1]), 1.0)

        # Baseline coordinates: start typically [-7.0, 0.0] or [0, -4]
        baseline_start = [-7.0, 0.0]
        baseline_goal = [7.0, 0.0]
        self.assertGreater(math.hypot(sx - baseline_start[0], sy - baseline_start[1]), 1.0)
        self.assertGreater(math.hypot(gx - baseline_goal[0], gy - baseline_goal[1]), 1.0)

    def test_start_and_goal_not_inside_obstacles_or_walls(self):
        """Verify start and goal do NOT lie inside any obstacle or outer perimeter."""
        cfg = get_scenario_config("airportpassengerassistance")
        internal_obstacles = cfg.static_obstacles[1:]

        sx, sy = cfg.robot_start[0], cfg.robot_start[1]
        for idx, poly in enumerate(internal_obstacles, start=1):
            self.assertFalse(
                point_in_polygon(sx, sy, poly),
                "Robot start ({}, {}) is inside obstacle {}".format(sx, sy, idx)
            )

        gx, gy = cfg.robot_goal[0], cfg.robot_goal[1]
        for idx, poly in enumerate(internal_obstacles, start=1):
            self.assertFalse(
                point_in_polygon(gx, gy, poly),
                "Robot goal ({}, {}) is inside obstacle {}".format(gx, gy, idx)
            )

    def test_start_and_goal_clearance_margins(self):
        """
        Verify start clearance >= 1.0 m and goal clearance >= 1.0 m
        with respect to all internal obstacles and outer walls.
        """
        cfg = get_scenario_config("airportpassengerassistance")
        internal_obstacles = cfg.static_obstacles[1:]
        bounds = cfg.world_bounds
        required_clearance = 1.0

        # Start position clearance to internal obstacles
        sx, sy = cfg.robot_start[0], cfg.robot_start[1]
        for idx, poly in enumerate(internal_obstacles, start=1):
            d = distance_point_to_polygon(sx, sy, poly)
            self.assertGreaterEqual(
                d, required_clearance,
                "Robot start too close to obstacle {}: dist={:.2f}m < {:.2f}m".format(idx, d, required_clearance)
            )

        # Start clearance to outer walls
        wall_clearance_sx = min(abs(sx - bounds[0]), abs(bounds[1] - sx))
        wall_clearance_sy = min(abs(sy - bounds[2]), abs(bounds[3] - sy))
        self.assertGreaterEqual(wall_clearance_sx, required_clearance)
        self.assertGreaterEqual(wall_clearance_sy, required_clearance)

        # Goal position clearance to internal obstacles
        gx, gy = cfg.robot_goal[0], cfg.robot_goal[1]
        for idx, poly in enumerate(internal_obstacles, start=1):
            d = distance_point_to_polygon(gx, gy, poly)
            self.assertGreaterEqual(
                d, required_clearance,
                "Robot goal too close to obstacle {}: dist={:.2f}m < {:.2f}m".format(idx, d, required_clearance)
            )

        # Goal clearance to outer walls
        wall_clearance_gx = min(abs(gx - bounds[0]), abs(bounds[1] - gx))
        wall_clearance_gy = min(abs(gy - bounds[2]), abs(bounds[3] - gy))
        self.assertGreaterEqual(wall_clearance_gx, required_clearance)
        self.assertGreaterEqual(wall_clearance_gy, required_clearance)

    def test_humans_spawn_distance_from_robot(self):
        """Verify that all dynamic human agents spawn >= 2.0 m away from the robot."""
        cfg = get_scenario_config("airportpassengerassistance")
        sx, sy = cfg.robot_start[0], cfg.robot_start[1]

        # Check base humans
        for idx, h_spec in enumerate(cfg.base_humans):
            h_x, h_y, _ = h_spec['pos']
            dist = math.hypot(h_x - sx, h_y - sy)
            self.assertGreaterEqual(
                dist, 2.0,
                "Base human {} ({:.2f}, {:.2f}) spawned {:.2f}m from robot (< 2.0m)".format(idx, h_x, h_y, dist)
            )

        # Check humans in simulated environment across multiple seeds
        for diff in ["EASY", "MEDIUM", "HARD"]:
            for seed in [42, 100, 101]:
                env = AirportNavRepEnv(difficulty=diff, silent=True)
                env.reset(seed=seed)
                for idx, human in enumerate(env.soadrl_sim.humans):
                    dist = math.hypot(human.px - sx, human.py - sy)
                    self.assertGreaterEqual(
                        dist, 2.0,
                        "Human {} in {} seed {} spawned {:.2f}m from robot (< 2.0m)".format(idx, diff, seed, dist)
                    )
                env.close()

    def test_free_path_exists_from_start_to_goal(self):
        """Verify that a collision-free path exists through waypoints from start to goal."""
        cfg = get_scenario_config("airportpassengerassistance")
        internal_obstacles = cfg.static_obstacles[1:]
        ctrl = AirportNavigationController()
        waypoints = ctrl.waypoints

        robot_radius = 0.30
        min_clearance = robot_radius + 0.15  # At least 0.45m clearance along waypoints

        # Interpolate points along waypoint segments and verify clearance
        for i in range(len(waypoints) - 1):
            w1 = waypoints[i]
            w2 = waypoints[i + 1]
            seg_dist = math.hypot(w2[0] - w1[0], w2[1] - w1[1])
            num_samples = max(5, int(seg_dist / 0.2))

            for s in range(num_samples + 1):
                t = float(s) / num_samples
                px = w1[0] + t * (w2[0] - w1[0])
                py = w1[1] + t * (w2[1] - w1[1])

                for o_idx, poly in enumerate(internal_obstacles):
                    self.assertFalse(
                        point_in_polygon(px, py, poly),
                        "Waypoint segment point ({:.2f}, {:.2f}) inside obstacle {}".format(px, py, o_idx)
                    )
                    d = distance_point_to_polygon(px, py, poly)
                    self.assertGreaterEqual(
                        d, min_clearance,
                        "Waypoint segment point ({:.2f}, {:.2f}) too close to obstacle {}: {:.2f}m".format(px, py, o_idx, d)
                    )

    def test_no_collision_first_20_steps_stationary_robot(self):
        """Verify that a stationary robot suffers zero collisions during the first 20 steps."""
        for diff in ["EASY", "MEDIUM", "HARD"]:
            for seed in [0, 42, 100]:
                env = AirportNavRepEnv(difficulty=diff, silent=True)
                env.reset(seed=seed)
                for step in range(20):
                    action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                    _, _, done, info = env.step(action)
                    event = info.get("event")
                    self.assertFalse(
                        isinstance(event, (Collision, CollisionOtherAgent)),
                        "Collision occurred at step {} while stationary in {} (seed {})".format(step, diff, seed)
                    )
                    self.assertFalse(done)
                env.close()

    def test_difficulty_scaling(self):
        """Verify difficulty scaling scales human count and sets fair time limits."""
        cfg = get_scenario_config("airportpassengerassistance")
        easy = cfg.get_difficulty_config("EASY")
        med = cfg.get_difficulty_config("MEDIUM")
        hard = cfg.get_difficulty_config("HARD")

        base_count = len(cfg.base_humans)
        self.assertEqual(base_count + easy['extra_humans'], 5)
        self.assertEqual(base_count + med['extra_humans'], 8)
        self.assertEqual(base_count + hard['extra_humans'], 12)

        self.assertGreater(easy['time_limit'], med['time_limit'])
        self.assertGreater(med['time_limit'], hard['time_limit'])

    def test_goal_reached_with_controller(self):
        """Verify that AirportNavigationController guides the robot to ReachGoal on seed 42."""
        env = AirportNavRepEnv(difficulty="MEDIUM", silent=True)
        env.reset(seed=42)
        ctrl = AirportNavigationController()
        ctrl.reset()

        sim = env.soadrl_sim
        done = False
        step = 0
        event = None

        while not done and step < env.time_limit:
            action = ctrl.act([sim.robot.px, sim.robot.py], humans=sim.humans, action_dim=3)
            _, _, done, info = env.step(action)
            step += 1
            ev = info.get("event")
            if isinstance(ev, (ReachGoal, Collision, CollisionOtherAgent, Timeout)):
                done = True
                event = ev

        self.assertIsInstance(event, ReachGoal, "Controller should reach goal on MEDIUM seed 42")
        env.close()


if __name__ == '__main__':
    unittest.main()
