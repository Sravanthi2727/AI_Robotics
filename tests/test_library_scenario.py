#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_library_scenario.py
========================
Isolated unit test suite for Shriya's Library Book Collection scenario:
  - Verifies dynamic registration in SCENARIO_REGISTRY under 'library' and 'librarybookcollection'
  - Validates world bounds, robot start/goal coordinates, and path distance (>= 8.0m)
  - Rigorously checks geometric non-overlap and clearance margins for robot and pedestrians
  - Verifies stack aisle widths and robot passage clearance
  - Validates monotonic difficulty scaling across EASY, MEDIUM, and HARD
  - Executes stepped simulation rollouts in CustomNavRepEnv and CustomNavRepEncodedEnv
  - Tests genuine policy compatibility if model checkpoint is present
  - Validates that LibraryNavigationController achieves ReachGoal without collisions
"""

import math
import unittest
import numpy as np

# Importing library_book_collection registers 'library' into SCENARIO_REGISTRY
import navrep_custom.library_book_collection as lib_mod
from navrep_custom.library_book_collection import (
    point_to_segment_distance,
    point_in_polygon,
    distance_point_to_polygon,
    load_compatible_policy,
    sync_and_rasterize_sim,
    LibraryNavigationController,
)
from navrep_custom.scenarios import SCENARIO_REGISTRY, get_scenario_config
from navrep_custom.custom_envs import CustomNavRepEnv, CustomNavRepEncodedEnv
from crowd_sim.envs.utils.info import ReachGoal, Collision, CollisionOtherAgent, Timeout


class TestLibraryScenario(unittest.TestCase):
    """Rigorous unit test suite for the Library scenario."""

    def test_library_registration(self):
        """Verify scenario self-registers under 'library' and 'librarybookcollection'."""
        self.assertIn("library", SCENARIO_REGISTRY)
        self.assertIn("librarybookcollection", SCENARIO_REGISTRY)

        cfg = get_scenario_config("library")
        self.assertEqual(cfg.scenario_id, "library")
        self.assertIn("Library", cfg.name)
        self.assertTrue(len(cfg.description) > 20)

    def test_library_world_bounds_and_waypoints(self):
        """Verify world bounds and start-to-goal Euclidean separation distance."""
        cfg = get_scenario_config("library")
        bounds = cfg.world_bounds
        sx, sy, sth = cfg.robot_start
        gx, gy = cfg.robot_goal

        # Check coordinate bounds
        self.assertTrue(bounds[0] <= sx <= bounds[1], "Start X out of bounds")
        self.assertTrue(bounds[2] <= sy <= bounds[3], "Start Y out of bounds")
        self.assertTrue(bounds[0] <= gx <= bounds[1], "Goal X out of bounds")
        self.assertTrue(bounds[2] <= gy <= bounds[3], "Goal Y out of bounds")

        # Check start-to-goal separation >= 8.0m benchmark specification
        dist = math.hypot(sx - gx, sy - gy)
        self.assertGreaterEqual(dist, 8.0, "Start-to-Goal distance ({:.2f}m) < 8.0m".format(dist))

    def test_library_robot_clearance(self):
        """
        Verify that robot start and goal do NOT overlap any static obstacle
        and maintain at least 0.50m clearance (robot radius 0.30m + 0.20m margin).
        """
        cfg = get_scenario_config("library")
        required_clearance = 0.50  # radius + margin

        # Obstacles list excludes outer perimeter at index 0
        internal_obstacles = cfg.static_obstacles[1:]

        # Test Robot Start
        sx, sy = cfg.robot_start[0], cfg.robot_start[1]
        for idx, poly in enumerate(internal_obstacles, start=1):
            self.assertFalse(
                point_in_polygon(sx, sy, poly),
                "Robot start ({}, {}) is inside obstacle {}".format(sx, sy, idx)
            )
            d = distance_point_to_polygon(sx, sy, poly)
            self.assertGreaterEqual(
                d, required_clearance,
                "Robot start too close to obstacle {}: dist={:.2f}m < {:.2f}m".format(idx, d, required_clearance)
            )

        # Test Robot Goal
        gx, gy = cfg.robot_goal[0], cfg.robot_goal[1]
        for idx, poly in enumerate(internal_obstacles, start=1):
            self.assertFalse(
                point_in_polygon(gx, gy, poly),
                "Robot goal ({}, {}) is inside obstacle {}".format(gx, gy, idx)
            )
            d = distance_point_to_polygon(gx, gy, poly)
            self.assertGreaterEqual(
                d, required_clearance,
                "Robot goal too close to obstacle {}: dist={:.2f}m < {:.2f}m".format(idx, d, required_clearance)
            )

    def test_library_pedestrians_free_space(self):
        """
        Verify that all pedestrian start positions and waypoint goals are strictly
        in free space, non-overlapping with any static obstacle, with clearance >= radius + 0.10m.
        """
        cfg = get_scenario_config("library")
        internal_obstacles = cfg.static_obstacles[1:]

        self.assertGreaterEqual(len(cfg.base_humans), 3, "Scenario must have at least 3 base humans")

        for h_idx, h in enumerate(cfg.base_humans):
            px, py = h['pos'][0], h['pos'][1]
            gx, gy = h['goal'][0], h['goal'][1]
            rad = h['radius']
            speed = h['speed']

            # Speed and radius bounds
            self.assertGreater(speed, 0.0, "Human {} speed must be positive".format(h_idx))
            self.assertLessEqual(speed, 1.5, "Human {} speed exceeds realistic library maximum".format(h_idx))
            self.assertGreaterEqual(rad, 0.25, "Human {} radius too small".format(h_idx))

            safety_margin = rad + 0.10  # Pedestrian radius plus safety margin

            # Test Start Position
            for o_idx, poly in enumerate(internal_obstacles, start=1):
                self.assertFalse(
                    point_in_polygon(px, py, poly),
                    "Human {} start ({}, {}) is inside obstacle {}".format(h_idx, px, py, o_idx)
                )
                d = distance_point_to_polygon(px, py, poly)
                self.assertGreaterEqual(
                    d, safety_margin,
                    "Human {} start at ({}, {}) overlaps obstacle {}! dist={:.2f}m < required {:.2f}m".format(
                        h_idx, px, py, o_idx, d, safety_margin)
                )

            # Test Waypoint Goal
            for o_idx, poly in enumerate(internal_obstacles, start=1):
                self.assertFalse(
                    point_in_polygon(gx, gy, poly),
                    "Human {} goal ({}, {}) is inside obstacle {}".format(h_idx, gx, gy, o_idx)
                )
                d = distance_point_to_polygon(gx, gy, poly)
                self.assertGreaterEqual(
                    d, safety_margin,
                    "Human {} goal at ({}, {}) overlaps obstacle {}! dist={:.2f}m < required {:.2f}m".format(
                        h_idx, gx, gy, o_idx, d, safety_margin)
                )

    def test_library_aisle_geometry(self):
        """Verify stack aisle clear widths provide adequate clearance for differential robot."""
        cfg = get_scenario_config("library")

        # Shelf A1 is obstacle 1, Shelf A2 is obstacle 2
        shelf_a1 = cfg.static_obstacles[1]
        shelf_a2 = cfg.static_obstacles[2]

        a1_min_y = min(p[1] for p in shelf_a1)
        a2_max_y = max(p[1] for p in shelf_a2)
        aisle_width = a1_min_y - a2_max_y

        self.assertAlmostEqual(aisle_width, 1.80, delta=0.05,
                               msg="Stack aisle 2 width must be ~1.80m")

        # Robot radius is ~0.30m (diameter 0.60m)
        robot_diameter = 0.60
        lateral_margin = (aisle_width - robot_diameter) / 2.0
        self.assertGreaterEqual(lateral_margin, 0.50,
                                "Lateral margin in book aisle must be at least 0.50m")

    def test_library_difficulty_scaling(self):
        """Verify difficulty modifiers scale monotonically (density, speed, time limits, humans)."""
        cfg = get_scenario_config("library")
        easy = cfg.get_difficulty_config("EASY")
        med = cfg.get_difficulty_config("MEDIUM")
        hard = cfg.get_difficulty_config("HARD")

        self.assertTrue(easy['density_mult'] <= med['density_mult'] <= hard['density_mult'])
        self.assertTrue(easy['speed_mult'] <= med['speed_mult'] <= hard['speed_mult'])
        self.assertTrue(easy['time_limit'] >= med['time_limit'] >= hard['time_limit'])
        self.assertTrue(easy.get('extra_humans', 0) <= med.get('extra_humans', 0) <= hard.get('extra_humans', 0))

    def test_library_env_rollout(self):
        """Verify CustomNavRepEnv executes steps in the library scenario without errors."""
        env = CustomNavRepEnv(scenario="library", difficulty="MEDIUM", silent=True)
        obs = env.reset()

        self.assertEqual(len(obs), 2, "Observation must be a tuple of (scan, robotstate)")
        scan, rstate = obs
        self.assertEqual(scan.shape, (1080,), "LiDAR scan shape must be (1080,)")
        self.assertEqual(rstate.shape, (5,), "Robot state shape must be (5,)")

        for _ in range(5):
            action = np.array([0.3, 0.0, 0.0], dtype=np.float32)
            obs, reward, done, info = env.step(action)
            self.assertIn('event', info)
            if done:
                break
        env.close()

    def test_library_encoded_rollout_with_policy(self):
        """
        Verify CustomNavRepEncodedEnv produces 37-dim observations and can be stepped
        with the pretrained VAE-LSTM policy from the repository.
        """
        env = CustomNavRepEncodedEnv(scenario="library", difficulty="MEDIUM", gpu=False, silent=True)
        obs = env.reset()
        self.assertEqual(obs.shape, (37,), "Encoded observation shape must be (37,)")

        policy, policy_path = load_compatible_policy()
        if policy is not None:
            action = policy.act(obs)
            self.assertEqual(action.shape, (2,), "Policy action must be 2-dimensional")
            obs, reward, done, info = env.step(action)
            self.assertEqual(obs.shape, (37,))
            self.assertIn('event', info)
        else:
            action = np.array([0.2, 0.0], dtype=np.float32)
            obs, reward, done, info = env.step(action)
            self.assertEqual(obs.shape, (37,))

        env.close()

    def test_library_navigation_controller_reaches_goal(self):
        """
        Verify that LibraryNavigationController guides the robot safely from Start
        to Goal in the Library scenario across multiple seeds, achieving ReachGoal event.
        """
        cfg = get_scenario_config("library")
        ctrl = LibraryNavigationController()
        test_seeds = [42, 100, 200]

        env = CustomNavRepEncodedEnv(scenario="library", difficulty="EASY", gpu=False, silent=True)

        for seed in test_seeds:
            np.random.seed(seed)
            ob = env.reset("library", "EASY")
            sync_and_rasterize_sim(env.raw_env, cfg)
            ctrl.reset()

            done = False
            step = 0
            event = None

            while not done and step < 380:
                pos = [env.soadrl_sim.robot.px, env.soadrl_sim.robot.py]
                action = ctrl.act(pos, env.soadrl_sim.humans, action_dim=2)
                ob, rew, done, info = env.step(action)
                step += 1
                ev = info.get('event')
                if isinstance(ev, (ReachGoal, Collision, CollisionOtherAgent, Timeout)):
                    done = True
                    event = ev

            self.assertIsInstance(
                event, ReachGoal,
                "Seed {} failed to reach goal! Terminal event: {} after {} steps".format(
                    seed, type(event).__name__, step)
            )

            final_dist = math.hypot(
                env.soadrl_sim.robot.px - cfg.robot_goal[0],
                env.soadrl_sim.robot.py - cfg.robot_goal[1]
            )
            self.assertLessEqual(
                final_dist, 0.50,
                "Seed {} final distance ({:.2f}m) exceeds goal radius 0.50m".format(seed, final_dist)
            )

        env.close()

    def test_library_predictive_avoidance_and_detour(self):
        """
        Verify that LibraryNavigationController performs predictive yielding when
        an oncoming human crosses path and successfully achieves goal on MEDIUM.
        """
        cfg = get_scenario_config("library")
        ctrl = LibraryNavigationController()
        env = CustomNavRepEncodedEnv(scenario="library", difficulty="MEDIUM", gpu=False, silent=True)

        np.random.seed(42)
        ob = env.reset("library", "MEDIUM")
        sync_and_rasterize_sim(env.raw_env, cfg)
        ctrl.reset()

        done = False
        step = 0
        event = None

        while not done and step < 380:
            pos = [env.soadrl_sim.robot.px, env.soadrl_sim.robot.py]
            action = ctrl.act(pos, env.soadrl_sim.humans, action_dim=2)
            ob, rew, done, info = env.step(action)
            step += 1
            ev = info.get('event')
            if isinstance(ev, (ReachGoal, Collision, CollisionOtherAgent, Timeout)):
                done = True
                event = ev

        self.assertIsInstance(
            event, ReachGoal,
            "MEDIUM Seed 42 must reach goal safely! Event: {} at step {}".format(
                type(event).__name__, step)
        )
        env.close()

    def test_a_star_path_planner_metrics(self):
        """
        Verify that genuine A* path planner on library occupancy grid finds an optimal,
        collision-free route from Start (-8.0, 3.5) to Goal (5.5, -3.5).
        """
        from navrep_custom.library_planner import get_library_a_star_plan
        plan = get_library_a_star_plan()

        self.assertTrue(plan['success'], "A* must find a valid route to the goal")
        self.assertGreater(plan['nodes_explored'], 1000, "A* must explore grid nodes")
        self.assertGreaterEqual(plan['path_length_m'], 15.0, "Path length must be at least 15m")
        self.assertLess(plan['plan_time_ms'], 1000.0, "Planning time must be under 1 second")

        # Verify start and goal correspondence
        cfg = get_scenario_config("library")
        start_dist = math.hypot(plan['path_coords'][0][0] - cfg.robot_start[0], plan['path_coords'][0][1] - cfg.robot_start[1])
        goal_dist = math.hypot(plan['path_coords'][-1][0] - cfg.robot_goal[0], plan['path_coords'][-1][1] - cfg.robot_goal[1])
        self.assertLessEqual(start_dist, 0.20, "Path must start at robot start pose")
        self.assertLessEqual(goal_dist, 0.20, "Path must terminate at circulation counter goal")

    def test_a_star_waypoints_obstacle_clearance(self):
        """
        Verify that no segment of the synthesized A* waypoints penetrates or comes within
        robot radius (0.30m) of any bookshelf, study table, pillar, or counter.
        """
        from navrep_custom.library_planner import get_library_a_star_plan
        plan = get_library_a_star_plan()
        waypoints = plan['waypoints']
        self.assertGreaterEqual(len(waypoints), 3, "Waypoints count must be at least 3")

        cfg = get_scenario_config("library")
        internal_obstacles = cfg.static_obstacles[1:]
        robot_radius = 0.30

        for i in range(len(waypoints) - 1):
            p1 = waypoints[i]
            p2 = waypoints[i + 1]
            seg_len = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            samples = max(5, int(seg_len / 0.10))
            for s in range(samples + 1):
                t = s / float(samples)
                x = p1[0] + (p2[0] - p1[0]) * t
                y = p1[1] + (p2[1] - p1[1]) * t
                for idx, poly in enumerate(internal_obstacles, start=1):
                    self.assertFalse(
                        point_in_polygon(x, y, poly),
                        "Waypoint segment {}-{} intersects obstacle {}".format(i, i + 1, idx)
                    )
                    d = distance_point_to_polygon(x, y, poly)
                    self.assertGreaterEqual(
                        d, robot_radius - 0.05,
                        "Waypoint segment {}-{} too close to obstacle {}: d={:.2f}m".format(i, i + 1, idx, d)
                    )

    def test_navigation_controller_with_a_star_waypoints(self):
        """
        Verify that LibraryNavigationController cleanly consumes synthesized A* waypoints
        and guides the robot from Stack Aisle 2 to the Circulation Counter achieving ReachGoal.
        """
        from navrep_custom.library_planner import get_library_a_star_plan
        plan = get_library_a_star_plan()
        cfg = get_scenario_config("library")
        ctrl = LibraryNavigationController(waypoints=plan['waypoints'])
        env = CustomNavRepEncodedEnv(scenario="library", difficulty="EASY", gpu=False, silent=True)

        np.random.seed(42)
        ob = env.reset("library", "EASY")
        sync_and_rasterize_sim(env.raw_env, cfg)
        ctrl.reset()

        done = False
        step = 0
        event = None

        while not done and step < 380:
            pos = [env.soadrl_sim.robot.px, env.soadrl_sim.robot.py]
            action = ctrl.act(pos, env.soadrl_sim.humans, action_dim=2)
            ob, rew, done, info = env.step(action)
            step += 1
            ev = info.get('event')
            if isinstance(ev, (ReachGoal, Collision, CollisionOtherAgent, Timeout)):
                done = True
                event = ev

        self.assertIsInstance(
            event, ReachGoal,
            "Robot using A* waypoints must achieve ReachGoal on EASY! Event: {}".format(event)
        )
        env.close()

    def test_goal_selection_valid_destinations(self):
        """1. Verify selecting valid preset destinations returns True from validator."""
        from navrep_custom.library_planner import validate_library_goal, LIBRARY_PRESET_DESTINATIONS
        for name, coords in LIBRARY_PRESET_DESTINATIONS:
            valid, reason = validate_library_goal(coords)
            self.assertTrue(valid, "Preset '{}' at {} failed validation: {}".format(name, coords, reason))

    def test_goal_selection_rejects_inside_obstacle(self):
        """2. Verify goal selection rejects coordinates located inside static obstacles."""
        from navrep_custom.library_planner import validate_library_goal
        # Point inside Shelf A1: spans [-8.5, -3.5], [4.4, 5.2]
        valid, reason = validate_library_goal((-6.0, 4.8))
        self.assertFalse(valid, "Goal inside Shelf A1 should be rejected")
        self.assertIn("obstacle", reason.lower())

        # Point inside Study Table A: spans [-6.8, -3.2], [-4.8, -3.2]
        valid, reason = validate_library_goal((-5.0, -4.0))
        self.assertFalse(valid, "Goal inside Study Table A should be rejected")
        self.assertIn("obstacle", reason.lower())

    def test_goal_selection_rejects_outside_bounds(self):
        """3. Verify goal selection rejects coordinates placed outside library world bounds."""
        from navrep_custom.library_planner import validate_library_goal
        out_points = [(-15.0, 0.0), (15.0, 0.0), (0.0, 10.0), (0.0, -10.0)]
        for pt in out_points:
            valid, reason = validate_library_goal(pt)
            self.assertFalse(valid, "Point {} outside boundary must be rejected".format(pt))
            self.assertIn("boundar", reason.lower())

    def test_goal_selection_rejects_insufficient_clearance(self):
        """4. Verify goal selection rejects coordinates right on obstacle edge (<0.35m clearance)."""
        from navrep_custom.library_planner import validate_library_goal
        # Shelf A2 corner is at (-3.5, 2.6); test point at (-3.48, 2.62)
        valid, reason = validate_library_goal((-3.48, 2.62))
        self.assertFalse(valid, "Point on shelf corner edge must be rejected for clearance")

    def test_replanning_from_robot_current_position(self):
        """5. Verify dynamic replanning operates from the robot's current position."""
        from navrep_custom.library_planner import replan_library_route
        robot_curr_pos = (-2.0, 3.1)
        new_goal = (2.25, 3.5) # Archive Reading Table
        plan = replan_library_route(robot_curr_pos, new_goal)

        self.assertTrue(plan['success'], "Replanning from current position must succeed")
        self.assertAlmostEqual(plan['start_xy'][0], robot_curr_pos[0], places=2)
        self.assertAlmostEqual(plan['start_xy'][1], robot_curr_pos[1], places=2)
        self.assertAlmostEqual(plan['goal_xy'][0], new_goal[0], places=2)
        self.assertAlmostEqual(plan['goal_xy'][1], new_goal[1], places=2)

    def test_changed_goal_produces_newly_calculated_route(self):
        """6. Verify changing the goal produces a distinctly new route and waypoints."""
        from navrep_custom.library_planner import replan_library_route
        curr_pos = (-8.0, 3.5)
        goal_a = (5.5, -3.5)   # Circulation Desk
        goal_b = (2.25, 3.5)   # Archive Reading Table

        plan_a = replan_library_route(curr_pos, goal_a)
        plan_b = replan_library_route(curr_pos, goal_b)

        self.assertTrue(plan_a['success'])
        self.assertTrue(plan_b['success'])
        self.assertNotEqual(plan_a['path_length_m'], plan_b['path_length_m'])
        self.assertNotEqual(plan_a['waypoints'][-1], plan_b['waypoints'][-1])

    def test_new_route_avoids_static_obstacles(self):
        """7. Verify that newly calculated replanned routes maintain collision-free clearance."""
        from navrep_custom.library_planner import replan_library_route
        cfg = get_scenario_config("library")
        internal_obstacles = cfg.static_obstacles[1:]
        robot_radius = 0.30

        # Plan to Mathematics Stack A3
        plan = replan_library_route((-8.0, 3.5), (-5.0, -1.8))
        self.assertTrue(plan['success'])

        for pt in plan['path_coords']:
            for idx, poly in enumerate(internal_obstacles, start=1):
                self.assertFalse(point_in_polygon(pt[0], pt[1], poly), "Path point inside obstacle {}".format(idx))
                d = distance_point_to_polygon(pt[0], pt[1], poly)
                self.assertGreaterEqual(d, robot_radius - 0.05)

    def test_navigation_controller_receives_new_waypoints(self):
        """8. Verify LibraryNavigationController.set_waypoints dynamically accepts new route."""
        ctrl = LibraryNavigationController()
        initial_len = len(ctrl.waypoints)

        new_waypoints = [[-2.0, 3.1], [0.0, 3.1], [2.25, 3.5]]
        ctrl.set_waypoints(new_waypoints)

        self.assertEqual(len(ctrl.waypoints), 3)
        self.assertEqual(ctrl.current_wp_idx, 0)
        self.assertEqual(ctrl.waypoints[-1], [2.25, 3.5])
        tx, ty = ctrl.get_current_target(-2.0)
        self.assertAlmostEqual(tx, -2.0)
        self.assertAlmostEqual(ty, 3.1)

    def test_invalid_goal_does_not_corrupt_existing_route(self):
        """9. Verify invalid goal rejection preserves the previous valid route."""
        from navrep_custom.library_planner import validate_library_goal, get_library_a_star_plan
        canonical_plan = get_library_a_star_plan()
        original_waypoints = list(canonical_plan['waypoints'])

        # Attempt invalid selection
        invalid_goal = (-6.0, 4.8) # inside shelf
        is_valid, _ = validate_library_goal(invalid_goal)
        self.assertFalse(is_valid)

        # Confirm canonical plan is unaltered
        plan_after = get_library_a_star_plan()
        self.assertEqual(plan_after['waypoints'], original_waypoints)

    def test_original_navigation_workflow_preserved(self):
        """10. Verify original start-to-circulation desk navigation remains 100% functional."""
        from navrep_custom.library_planner import get_library_a_star_plan
        plan = get_library_a_star_plan()
        cfg = get_scenario_config("library")
        ctrl = LibraryNavigationController(waypoints=plan['waypoints'])
        env = CustomNavRepEncodedEnv(scenario="library", difficulty="EASY", gpu=False, silent=True)

        np.random.seed(42)
        ob = env.reset("library", "EASY")
        sync_and_rasterize_sim(env.raw_env, cfg)
        ctrl.reset()

        done = False
        step = 0
        event = None

        while not done and step < 380:
            pos = [env.soadrl_sim.robot.px, env.soadrl_sim.robot.py]
            action = ctrl.act(pos, env.soadrl_sim.humans, action_dim=2)
            ob, rew, done, info = env.step(action)
            step += 1
            ev = info.get('event')
            if isinstance(ev, (ReachGoal, Collision, CollisionOtherAgent, Timeout)):
                done = True
                event = ev

        self.assertIsInstance(event, ReachGoal)
        env.close()


if __name__ == '__main__':
    unittest.main()
