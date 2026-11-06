#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_custom_scenarios.py
Automated test suite for custom NavRep scenarios and environments:
  - Verifies instantiation of all custom scenarios in SCENARIO_REGISTRY
  - Validates that robot start and goal poses are within world boundaries and free space
  - Validates difficulty ladders (EASY, MEDIUM, HARD) and parametric crowd scaling
  - Executes short rollouts (10 steps) in CustomNavRepEnv and CustomNavRepEncodedEnv
  - Asserts observations, step transitions, and info dict compliance
"""

import math
import unittest
import numpy as np

from navrep_custom.scenarios import SCENARIO_REGISTRY, get_scenario_config
from navrep_custom.custom_envs import CustomNavRepEnv, CustomNavRepEncodedEnv
from crowd_sim.envs.utils.info import ReachGoal, Collision, CollisionOtherAgent, Timeout


class TestCustomScenarios(unittest.TestCase):
    def test_scenario_registry_completeness(self):
        """Verify all 6 custom scenarios are registered."""
        expected = ["mallmate", "medirunner", "kiranakart", "diwalibazaar", "railwaystation", "firedrill"]
        for sc in expected:
            self.assertIn(sc, SCENARIO_REGISTRY, "Scenario {} missing from registry".format(sc))
            cfg = get_scenario_config(sc)
            self.assertIsNotNone(cfg.name)
            self.assertTrue(len(cfg.description) > 10)

    def test_start_and_goal_validity(self):
        """Verify robot start and goal are within world bounds and have obstacle clearance."""
        for sc_id, factory in SCENARIO_REGISTRY.items():
            cfg = factory()
            bounds = cfg.world_bounds  # [x_min, x_max, y_min, y_max]
            sx, sy, sth = cfg.robot_start
            gx, gy = cfg.robot_goal

            # Check bounds
            self.assertTrue(bounds[0] <= sx <= bounds[1], "{}: Start X ({}) out of bounds".format(sc_id, sx))
            self.assertTrue(bounds[2] <= sy <= bounds[3], "{}: Start Y ({}) out of bounds".format(sc_id, sy))
            self.assertTrue(bounds[0] <= gx <= bounds[1], "{}: Goal X ({}) out of bounds".format(sc_id, gx))
            self.assertTrue(bounds[2] <= gy <= bounds[3], "{}: Goal Y ({}) out of bounds".format(sc_id, gy))

            # Check start-to-goal separation
            start_goal_dist = math.sqrt((sx - gx)**2 + (sy - gy)**2)
            self.assertTrue(start_goal_dist >= 8.0, "{}: Start-Goal distance ({:.1f}m) too small".format(sc_id, start_goal_dist))

            # Check that start does not intersect obstacle bounding centers directly
            for poly in cfg.static_obstacles:
                cx = np.mean([p[0] for p in poly])
                cy = np.mean([p[1] for p in poly])
                d_start = math.sqrt((sx - cx)**2 + (sy - cy)**2)
                d_goal = math.sqrt((gx - cx)**2 + (gy - cy)**2)
                self.assertTrue(d_start > 0.4, "{}: Start too close to obstacle centroid ({}, {})".format(sc_id, cx, cy))
                self.assertTrue(d_goal > 0.4, "{}: Goal too close to obstacle centroid ({}, {})".format(sc_id, cx, cy))

    def test_difficulty_ladder_scaling(self):
        """Verify EASY, MEDIUM, and HARD difficulty modifiers adjust density and speeds."""
        for sc_id, factory in SCENARIO_REGISTRY.items():
            cfg = factory()
            easy = cfg.get_difficulty_config("EASY")
            med = cfg.get_difficulty_config("MEDIUM")
            hard = cfg.get_difficulty_config("HARD")

            self.assertTrue(easy['density_mult'] <= med['density_mult'] <= hard['density_mult'])
            self.assertTrue(easy['speed_mult'] <= med['speed_mult'] <= hard['speed_mult'])
            self.assertTrue(easy['time_limit'] >= med['time_limit'] >= hard['time_limit'])

    def test_custom_env_rollout(self):
        """Verify CustomNavRepEnv executes 10 steps without errors."""
        for sc_id in ["mallmate", "medirunner", "kiranakart", "diwalibazaar", "railwaystation", "firedrill"]:
            env = CustomNavRepEnv(scenario=sc_id, difficulty="MEDIUM", silent=True)
            obs = env.reset()

            self.assertEqual(len(obs), 2, "Observation must be tuple (scan, robotstate)")
            scan, rstate = obs
            self.assertEqual(scan.shape, (1080,), "Scan shape must be (1080,)")
            self.assertEqual(rstate.shape, (5,), "Robotstate shape must be (5,)")

            for _ in range(10):
                action = np.array([0.5, 0.0, 0.0], dtype=np.float32)
                obs, reward, done, info = env.step(action)
                self.assertIn('event', info)

            env.close()

    def test_custom_encoded_env_rollout(self):
        """Verify CustomNavRepEncodedEnv produces 37-dim observations for PPO policy."""
        env = CustomNavRepEncodedEnv(scenario="mallmate", difficulty="MEDIUM", gpu=False, silent=True)
        obs = env.reset()
        self.assertEqual(obs.shape, (37,), "Encoded observation must be (37,)")

        for _ in range(10):
            action = np.array([0.5, 0.0], dtype=np.float32)
            obs, reward, done, info = env.step(action)
            self.assertEqual(obs.shape, (37,))

        env.close()


if __name__ == '__main__':
    unittest.main()
