#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_hospital_controller.py
===========================
Focused unit test suite for HospitalNavigationController:
  - Validates waypoint progression and geometric clearance against static obstacles.
  - Validates action shapes: (2,) for encoded env, (3,) for raw env (float32).
  - Validates emergency gurney detection, CPA estimation, and alcove yielding state machine.
  - Validates speed regulation and docking deceleration.
  - Validates pedestrian safety braking.
"""

from __future__ import print_function, division
import math
import unittest
import numpy as np

from navrep_custom.hospital_delivery import (
    HospitalNavigationController,
    is_position_safe,
    distance_point_to_polygon,
    point_in_polygon
)
from navrep_custom.scenarios import get_scenario_config


class MockHuman(object):
    """Lightweight mock object providing verified attributes of crowd_sim Human."""
    def __init__(self, px, py, vx, vy, radius=0.30):
        self.px = float(px)
        self.py = float(py)
        self.vx = float(vx)
        self.vy = float(vy)
        self.radius = float(radius)


class TestHospitalNavigationController(unittest.TestCase):

    def setUp(self):
        self.cfg = get_scenario_config("medirunner")
        self.ctrl = HospitalNavigationController(
            static_obstacles=self.cfg.static_obstacles,
            world_bounds=self.cfg.world_bounds
        )

    def test_initialization_and_waypoints(self):
        """Verify default 7 waypoints, start/goal alignment, and initial state."""
        self.assertEqual(len(self.ctrl.waypoints), 7, "Must have exactly 7 default waypoints")
        self.assertEqual(self.ctrl.waypoints[0], [-10.5, 0.0], "WP 0 must match start")
        self.assertEqual(self.ctrl.waypoints[-1], [10.5, 0.0], "WP 6 must match goal")
        self.assertEqual(self.ctrl.state, HospitalNavigationController.STATE_NORMAL)

    def test_geometric_clearance_of_waypoints(self):
        """Verify every waypoint has >= 0.40m clearance from static obstacles and bounds."""
        for idx, (wx, wy) in enumerate(self.ctrl.waypoints):
            safe = is_position_safe(
                wx, wy,
                static_obstacles=self.cfg.static_obstacles,
                world_bounds=self.cfg.world_bounds,
                min_margin=0.40
            )
            self.assertTrue(safe, "Waypoint #{} ({:.2f}, {:.2f}) violates static obstacle safety margin".format(idx, wx, wy))

    def test_action_shapes_and_types(self):
        """Verify action output shapes and float32 dtype for both 2D and 3D modes."""
        pos = [-10.5, 0.0]

        # 2D mode (CustomNavRepEncodedEnv)
        act2 = self.ctrl.act(pos, humans=[], action_dim=2)
        self.assertIsInstance(act2, np.ndarray)
        self.assertEqual(act2.shape, (2,), "action_dim=2 must return shape (2,)")
        self.assertEqual(act2.dtype, np.float32, "Action dtype must be float32")
        self.assertTrue(act2[0] > 0.0, "Robot at start should produce positive vx towards goal")

        # 3D mode (CustomNavRepEnv)
        act3 = self.ctrl.act(pos, humans=[], action_dim=3)
        self.assertIsInstance(act3, np.ndarray)
        self.assertEqual(act3.shape, (3,), "action_dim=3 must return shape (3,)")
        self.assertEqual(act3.dtype, np.float32, "Action dtype must be float32")
        self.assertEqual(act3[2], 0.0, "vtheta should be 0.0")

    def test_waypoint_progression(self):
        """Verify controller advances waypoint index as the robot reaches targets."""
        self.assertEqual(self.ctrl.current_wp_idx, 0)
        # Robot moves close to WP 0
        self.ctrl.act([-10.45, 0.0], humans=[], action_dim=2)
        self.assertGreaterEqual(self.ctrl.current_wp_idx, 1, "Should advance past WP 0 when reached")

    def test_docking_speed_deceleration(self):
        """Verify velocity throttles smoothly as the robot nears the final goal."""
        self.ctrl.current_wp_idx = len(self.ctrl.waypoints) - 1
        # Close to goal at (10.5, 0.0)
        close_pos = [10.35, 0.0]
        act = self.ctrl.act(close_pos, humans=[], action_dim=2)
        speed = math.hypot(act[0], act[1])
        self.assertLess(speed, self.ctrl.max_speed, "Speed near goal must decelerate below max_speed")
        self.assertGreater(speed, 0.0, "Speed near goal should remain non-zero until docked")

    def test_emergency_gurney_detection(self):
        """Verify controller accurately identifies approaching emergency gurney."""
        # Robot at West side: (-2.0, -0.6)
        # Gurney at (3.0, -1.0) rushing West at -1.65 m/s, radius = 0.65m
        gurney = MockHuman(px=3.0, py=-1.0, vx=-1.65, vy=0.0, radius=0.65)
        regular_ped = MockHuman(px=-1.0, py=0.8, vx=0.6, vy=0.0, radius=0.30)

        detected_gurney, t_cpa, cpa_dist = self.ctrl.detect_emergency_gurney(
            rx=-2.0, ry=-0.6, vx=0.6, vy=0.0, humans=[regular_ped, gurney]
        )
        self.assertIsNotNone(detected_gurney)
        self.assertEqual(detected_gurney.radius, 0.65)
        self.assertGreater(t_cpa, 0.0)
        self.assertLess(t_cpa, 5.0)

    def test_emergency_gurney_alcove_yielding_state_machine(self):
        """Verify yielding state transitions when emergency gurney approaches and clears."""
        # Robot at West approach (-2.0, -0.6)
        gurney = MockHuman(px=2.5, py=-1.0, vx=-1.65, vy=0.0, radius=0.65)

        # 1. Approach step: controller must trigger alcove yield
        act = self.ctrl.act([-2.0, -0.6], humans=[gurney], action_dim=2)
        self.assertIn(
            self.ctrl.state,
            [HospitalNavigationController.STATE_YIELD_ALCOVE, HospitalNavigationController.STATE_HOLD_WAIT]
        )
        if self.ctrl.yield_target is not None:
            # Yield target must be safe from static obstacles
            yx, yy = self.ctrl.yield_target
            safe = is_position_safe(yx, yy, self.cfg.static_obstacles, self.cfg.world_bounds, min_margin=0.38)
            self.assertTrue(safe, "Selected alcove point ({:.2f}, {:.2f}) must be safe".format(yx, yy))
            # Distance from gurney lane y = -1.0 must exceed combined radii (0.95m)
            self.assertGreater(abs(yy - (-1.0)), 0.95, "Alcove must be clear of gurney lane y=-1.0")

        # 2. Once gurney clears to the West of the robot (px < rx - 1.2m)
        cleared_gurney = MockHuman(px=-5.0, py=-1.0, vx=-1.65, vy=0.0, radius=0.65)
        for _ in range(6):
            self.ctrl.act([-2.0, -0.6], humans=[cleared_gurney], action_dim=2)

        self.assertIn(
            self.ctrl.state,
            [HospitalNavigationController.STATE_RESUME, HospitalNavigationController.STATE_NORMAL],
            "State must transition back to RESUME or NORMAL once gurney has cleared"
        )

    def test_pedestrian_emergency_stop(self):
        """Verify absolute emergency stop when a pedestrian is dangerously close."""
        # Robot at (0.0, -0.6), pedestrian directly in front at (0.20, -0.6)
        ped = MockHuman(px=0.20, py=-0.6, vx=0.0, vy=0.0, radius=0.30)
        act = self.ctrl.act([0.0, -0.6], humans=[ped], action_dim=2)
        self.assertEqual(act[0], 0.0, "vx must be 0.0 when pedestrian is dangerously close")
        self.assertEqual(act[1], 0.0, "vy must be 0.0 when pedestrian is dangerously close")

    def test_robot_remains_in_alcove_while_gurney_approaching(self):
        """Verify robot remains stopped in alcove while emergency gurney is approaching from East."""
        # Set controller into HOLD_WAIT state inside North Alcove 1 (-2.40, 1.25)
        self.ctrl.state = HospitalNavigationController.STATE_HOLD_WAIT
        self.ctrl.yield_target = [-2.40, 1.25]
        self.ctrl.gurney_clear_steps = 0

        # Gurney is at x = 2.0, y = -1.0, traveling West at -1.65 m/s
        # Even though lateral separation is 2.25m, gurney is approaching (gx > rx)
        gurney = MockHuman(px=2.0, py=-1.0, vx=-1.65, vy=0.0, radius=0.65)

        for step in range(10):
            # Advance gurney position westward each step (simulating 0.2s dt)
            gurney.px = 2.0 - step * 0.33
            act = self.ctrl.act([-2.40, 1.25], humans=[gurney], action_dim=2)

            self.assertEqual(
                self.ctrl.state,
                HospitalNavigationController.STATE_HOLD_WAIT,
                "Robot must remain in STATE_HOLD_WAIT while gurney approaches (step {})".format(step)
            )
            self.assertEqual(act[0], 0.0, "vx must be 0.0 while holding in alcove")
            self.assertEqual(act[1], 0.0, "vy must be 0.0 while holding in alcove")
            self.assertEqual(self.ctrl.gurney_clear_steps, 0, "Clearance counter must stay at 0 while gurney has not passed")

    def test_robot_does_not_resume_immediately_when_gurney_first_passes(self):
        """Verify robot does not resume immediately on first pass, requiring stable clearance."""
        self.ctrl.state = HospitalNavigationController.STATE_HOLD_WAIT
        self.ctrl.yield_target = [-2.40, 1.25]
        self.ctrl.gurney_clear_steps = 0

        # Gurney just passes rx - 1.20m = -3.60m (e.g. px = -3.80m)
        for step in range(1, 5):
            gurney = MockHuman(px=-3.60 - step * 0.25, py=-1.0, vx=-1.65, vy=0.0, radius=0.65)
            act = self.ctrl.act([-2.40, 1.25], humans=[gurney], action_dim=2)

            self.assertEqual(self.ctrl.gurney_clear_steps, step)
            self.assertEqual(
                self.ctrl.state,
                HospitalNavigationController.STATE_HOLD_WAIT,
                "State must not transition to RESUME before 5 stable steps (at step {})".format(step)
            )
            self.assertEqual(act[0], 0.0, "Robot must remain stopped before 5 stable clearance steps")
            self.assertEqual(act[1], 0.0, "Robot must remain stopped before 5 stable clearance steps")

    def test_clearance_counter_resets_if_threat_reappears(self):
        """Verify clearance counter resets to 0 if a threat reappears before clearance threshold."""
        self.ctrl.state = HospitalNavigationController.STATE_HOLD_WAIT
        self.ctrl.yield_target = [-2.40, 1.25]
        self.ctrl.gurney_clear_steps = 0

        # Gurney is clearing for 3 steps
        for step in range(1, 4):
            gurney = MockHuman(px=-3.80 - step * 0.20, py=-1.0, vx=-1.65, vy=0.0, radius=0.65)
            self.ctrl.act([-2.40, 1.25], humans=[gurney], action_dim=2)
            self.assertEqual(self.ctrl.gurney_clear_steps, step)

        # Threat reappears on step 4 (e.g., gurney loops or another gurney approaches from East)
        threat_gurney = MockHuman(px=4.0, py=-1.0, vx=-1.65, vy=0.0, radius=0.65)
        act = self.ctrl.act([-2.40, 1.25], humans=[threat_gurney], action_dim=2)

        self.assertEqual(self.ctrl.gurney_clear_steps, 0, "Clearance counter must reset to 0 when threat reappears")
        self.assertEqual(self.ctrl.state, HospitalNavigationController.STATE_HOLD_WAIT)
        self.assertEqual(act[0], 0.0)
        self.assertEqual(act[1], 0.0)

    def test_triage_bottleneck_gatekeeper_blocks_entry(self):
        """Verify gatekeeper stops robot from entering triage bottleneck when gurney approaches."""
        # Robot is approaching the bottleneck from the West: rx = -1.00m, ry = -0.60m
        self.ctrl.current_wp_idx = 2  # targeting WP 2/3
        gurney = MockHuman(px=3.5, py=-1.0, vx=-1.65, vy=0.0, radius=0.65)

        act = self.ctrl.act([-1.00, -0.60], humans=[gurney], action_dim=2)

        # Controller must detect hazard and trigger alcove yield
        self.assertIn(
            self.ctrl.state,
            [HospitalNavigationController.STATE_YIELD_ALCOVE, HospitalNavigationController.STATE_HOLD_WAIT]
        )
        # Forward velocity into the bottleneck must be inhibited (not driving East into choke point)
        self.assertLessEqual(act[0], 0.0, "Gatekeeper must not command forward motion eastward into bottleneck")

        # Even if robot is right at the bottleneck threshold x = -0.75m
        act_at_threshold = self.ctrl.act([-0.75, -0.60], humans=[gurney], action_dim=2)
        self.assertLessEqual(act_at_threshold[0], 0.0, "Robot at threshold must not cross into x > -0.75m")

    def test_resumes_safely_after_gurney_clears(self):
        """Verify robot transitions to RESUME and moves towards corridor after gurney clears for 5 steps."""
        self.ctrl.state = HospitalNavigationController.STATE_HOLD_WAIT
        self.ctrl.yield_target = [-2.40, 1.25]
        self.ctrl.current_wp_idx = 2  # target WP 2 at (-2.0, -0.60)
        self.ctrl.gurney_clear_steps = 0

        # Gurney stably clear for 5 steps (moving West, px < -3.60m)
        for step in range(5):
            cleared_gurney = MockHuman(px=-4.0 - step * 0.30, py=-1.0, vx=-1.65, vy=0.0, radius=0.65)
            act = self.ctrl.act([-2.40, 1.25], humans=[cleared_gurney], action_dim=2)

        self.assertGreaterEqual(self.ctrl.gurney_clear_steps, 5)
        self.assertEqual(
            self.ctrl.state,
            HospitalNavigationController.STATE_RESUME,
            "State must transition to STATE_RESUME after 5 stable clearance steps"
        )
        self.assertIsNone(self.ctrl.yield_target, "yield_target must be cleared on resume")
        # In RESUME from alcove (y = 1.25), robot should steer South towards corridor (WP 2 at y = -0.60)
        self.assertLess(act[1], 0.0, "vy must direct robot southwards back into corridor")

    def test_gurney_identification_via_human_meta(self):
        """Verify gurney is identified accurately via human_meta role attribute."""
        # Non-standard radius/velocity human that would NOT match heuristic alone
        sneaky_gurney = MockHuman(px=3.0, py=-1.0, vx=-0.50, vy=0.0, radius=0.35)
        regular_nurse = MockHuman(px=-1.0, py=0.5, vx=0.80, vy=0.0, radius=0.30)
        humans = [regular_nurse, sneaky_gurney]

        meta = [
            {'role': 'nurse_fast'},
            {'role': 'emergency_gurney'}
        ]

        detected_gurney, t_cpa, cpa_dist = self.ctrl.detect_emergency_gurney(
            rx=-2.0, ry=-0.60, vx=0.50, vy=0.0,
            humans=humans,
            human_meta=meta
        )
        self.assertIsNotNone(detected_gurney, "Gurney must be detected via human_meta role")
        self.assertIs(detected_gurney, sneaky_gurney, "Detected agent must match emergency_gurney meta")

        # Act with human_meta should trigger yielding
        act = self.ctrl.act([-2.0, -0.60], humans=humans, action_dim=2, human_meta=meta)
        self.assertIn(
            self.ctrl.state,
            [HospitalNavigationController.STATE_YIELD_ALCOVE, HospitalNavigationController.STATE_HOLD_WAIT]
        )

        # Fallback heuristic: when human_meta is None, nominal gurney is still detected
        nominal_gurney = MockHuman(px=3.0, py=-1.0, vx=-1.65, vy=0.0, radius=0.65)
        detected_fallback, _, _ = self.ctrl.detect_emergency_gurney(
            rx=-2.0, ry=-0.60, vx=0.50, vy=0.0,
            humans=[nominal_gurney],
            human_meta=None
        )
        self.assertIsNotNone(detected_fallback, "Gurney must be detected via fallback heuristics when human_meta is None")


class ClosedLoopSim(object):
    """
    Deterministic discrete-time 2D simulation harness for MediRunner hospital scenario.
    Simulation time step dt = 0.2s.

    Verified action-to-velocity conversion:
      In CustomNavRepEncodedEnv, the controller action [vx, vy] is directly the robot's
      commanded velocity in meters/second (clamped to max_speed).
      Kinematic update:
        rx_{t+1} = rx_t + vx * dt
        ry_{t+1} = ry_t + vy * dt
        hx_{t+1} = hx_t + hvx * dt
        hy_{t+1} = hy_t + hvy * dt
    """
    def __init__(self, controller, robot_start=None, humans=None, human_meta=None, dt=0.2):
        self.controller = controller
        self.robot_pos = list(robot_start if robot_start is not None else [-10.5, 0.0])
        self.humans = humans if humans is not None else []
        self.human_meta = human_meta
        self.dt = float(dt)
        self.time = 0.0
        self.step_count = 0
        self.min_gurney_dist = float('inf')
        self.collision_detected = False
        self.state_history = [self.controller.state]
        self.pos_history = [list(self.robot_pos)]

    def step(self):
        act = self.controller.act(
            self.robot_pos,
            humans=self.humans,
            action_dim=2,
            human_meta=self.human_meta
        )
        vx = float(act[0])
        vy = float(act[1])

        # Kinematic update for robot
        self.robot_pos[0] += vx * self.dt
        self.robot_pos[1] += vy * self.dt

        # Kinematic update for humans
        for h in self.humans:
            h.px += getattr(h, 'vx', 0.0) * self.dt
            h.py += getattr(h, 'vy', 0.0) * self.dt

        # Evaluate separation and collisions with emergency gurney
        for i, h in enumerate(self.humans):
            is_g = False
            if self.human_meta and i < len(self.human_meta):
                if self.human_meta[i].get('role') == 'emergency_gurney':
                    is_g = True
            elif getattr(h, 'radius', 0.30) >= 0.55:
                is_g = True

            if is_g:
                dist = math.hypot(self.robot_pos[0] - h.px, self.robot_pos[1] - h.py)
                if dist < self.min_gurney_dist:
                    self.min_gurney_dist = dist
                sum_r = self.controller.robot_radius + getattr(h, 'radius', 0.65)
                if dist < sum_r:
                    self.collision_detected = True

        self.time += self.dt
        self.step_count += 1
        self.state_history.append(self.controller.state)
        self.pos_history.append(list(self.robot_pos))
        return act


class TestClosedLoopHospitalController(unittest.TestCase):
    """
    Genuine closed-loop multi-step deterministic unit tests for MediRunner hospital controller:
      - Validates the complete state machine lifecycle:
        NORMAL_TRANSIT -> ALCOVE_YIELD -> HOLD_WAIT -> RESUME_TRANSIT -> NORMAL_TRANSIT.
      - Validates zero collisions and conservative separation margins.
      - Validates temporary disappearance handling (uncertainty state).
      - Validates gurney resets, bottleneck boundaries, and blocked alcove paths.
    """

    def setUp(self):
        self.cfg = get_scenario_config("medirunner")
        self.ctrl = HospitalNavigationController(
            static_obstacles=self.cfg.static_obstacles,
            world_bounds=self.cfg.world_bounds
        )

    def test_closed_loop_lifecycle_no_collision(self):
        """Test full closed-loop lifecycle: NORMAL -> YIELD -> HOLD -> RESUME -> NORMAL with zero collisions."""
        self.ctrl.current_wp_idx = 1
        robot_start = [-3.5, 0.0]
        gurney = MockHuman(px=5.0, py=-1.0, vx=-1.65, vy=0.0, radius=0.65)
        meta = [{'role': 'emergency_gurney'}]

        sim = ClosedLoopSim(self.ctrl, robot_start=robot_start, humans=[gurney], human_meta=meta)

        for step in range(80):
            sim.step()
            if gurney.px < -11.0:
                gurney.vx = 0.0  # Stopped at destination

        self.assertFalse(sim.collision_detected, "No collision must occur during full closed-loop run")
        self.assertGreater(sim.min_gurney_dist, 0.95, "Min gurney separation must exceed combined radius (0.95m)")
        distinct_states = set(sim.state_history)
        self.assertIn(HospitalNavigationController.STATE_NORMAL, distinct_states)
        self.assertIn(HospitalNavigationController.STATE_YIELD_ALCOVE, distinct_states)
        self.assertIn(HospitalNavigationController.STATE_HOLD_WAIT, distinct_states)
        self.assertIn(HospitalNavigationController.STATE_RESUME, distinct_states)
        # Verify robot progressed past bottleneck
        self.assertGreater(sim.robot_pos[0], 0.0, "Robot must successfully transit through bottleneck")

    def test_closed_loop_gurney_temporarily_disappears(self):
        """Verify robot remains stopped in uncertainty state when gurney tracking is lost during yielding."""
        self.ctrl.state = HospitalNavigationController.STATE_HOLD_WAIT
        self.ctrl.yield_target = [-2.40, 1.25]
        gurney = MockHuman(px=2.0, py=-1.0, vx=-1.65, vy=0.0, radius=0.65)
        meta = [{'role': 'emergency_gurney'}]

        sim = ClosedLoopSim(self.ctrl, robot_start=[-2.40, 1.25], humans=[gurney], human_meta=meta)

        # Step 0: Gurney approaching
        sim.step()
        self.assertEqual(self.ctrl.state, HospitalNavigationController.STATE_HOLD_WAIT)

        # Steps 1-6: Gurney disappears from tracking (humans=[])
        sim.humans = []
        sim.human_meta = []
        for _ in range(6):
            act = sim.step()
            self.assertEqual(self.ctrl.state, HospitalNavigationController.STATE_HOLD_WAIT,
                             "Robot must remain in HOLD_WAIT when gurney disappears")
            self.assertEqual(act[0], 0.0)
            self.assertEqual(act[1], 0.0)
            self.assertEqual(self.ctrl.gurney_clear_steps, 0,
                             "Clearance counter must NOT increment when gurney is missing")

        # Step 7: Gurney reappears already safely to the West at x = -4.5m
        cleared_gurney = MockHuman(px=-4.5, py=-1.0, vx=-1.65, vy=0.0, radius=0.65)
        sim.humans = [cleared_gurney]
        sim.human_meta = [{'role': 'emergency_gurney'}]

        for _ in range(6):
            sim.step()

        self.assertEqual(self.ctrl.state, HospitalNavigationController.STATE_RESUME,
                         "Robot must resume after gurney reliably clears")

    def test_closed_loop_metadata_identifies_gurney_among_pedestrians(self):
        """Verify gurney role metadata correctly targets emergency gurney in heterogeneous crowd."""
        nurse1 = MockHuman(px=2.0, py=0.5, vx=-1.25, vy=0.0, radius=0.30)
        walker = MockHuman(px=-1.0, py=-0.2, vx=-0.28, vy=0.0, radius=0.45)
        visitor = MockHuman(px=0.0, py=0.8, vx=0.60, vy=0.0, radius=0.30)
        gurney = MockHuman(px=5.0, py=-1.0, vx=-1.65, vy=0.0, radius=0.65)
        humans = [nurse1, walker, visitor, gurney]
        meta = [
            {'role': 'nurse_fast'},
            {'role': 'patient_walker'},
            {'role': 'visitor'},
            {'role': 'emergency_gurney'}
        ]

        self.ctrl.current_wp_idx = 1
        sim = ClosedLoopSim(self.ctrl, robot_start=[-3.5, 0.0], humans=humans, human_meta=meta)

        # Step simulation and verify gurney is identified
        detected, _, _ = self.ctrl.detect_emergency_gurney(-3.5, 0.0, 0.5, 0.0, humans, human_meta=meta)
        self.assertIs(detected, gurney)

        sim.step()
        self.assertIn(self.ctrl.state, [HospitalNavigationController.STATE_YIELD_ALCOVE, HospitalNavigationController.STATE_HOLD_WAIT])

    def test_closed_loop_gurney_resets_while_waiting(self):
        """Verify clearance counter resets to 0 and robot remains stopped if gurney unexpectedly loops."""
        self.ctrl.state = HospitalNavigationController.STATE_HOLD_WAIT
        self.ctrl.yield_target = [-2.40, 1.25]
        gurney = MockHuman(px=-4.0, py=-1.0, vx=-1.65, vy=0.0, radius=0.65)
        meta = [{'role': 'emergency_gurney'}]

        sim = ClosedLoopSim(self.ctrl, robot_start=[-2.40, 1.25], humans=[gurney], human_meta=meta)

        # Stably clear for 3 steps
        for _ in range(3):
            sim.step()
        self.assertEqual(self.ctrl.gurney_clear_steps, 3)

        # Unexpected reset: gurney suddenly respawns at x = 7.0m rushing West
        gurney.px = 7.0
        gurney.vx = -1.65
        sim.step()

        self.assertEqual(self.ctrl.gurney_clear_steps, 0, "Clearance counter must reset when gurney resets to East")
        self.assertEqual(self.ctrl.state, HospitalNavigationController.STATE_HOLD_WAIT)

    def test_closed_loop_robot_already_at_neg_0_6(self):
        """Verify controller does not advance into oncoming gurney when already at x = -0.60m."""
        self.ctrl.current_wp_idx = 3
        robot_start = [-0.60, -0.60]
        gurney = MockHuman(px=5.0, py=-1.0, vx=-1.65, vy=0.0, radius=0.65)
        meta = [{'role': 'emergency_gurney'}]

        sim = ClosedLoopSim(self.ctrl, robot_start=robot_start, humans=[gurney], human_meta=meta)
        act = sim.step()

        # Must not command forward motion eastward deeper into bottleneck towards gurney
        self.assertLessEqual(act[0], 0.0, "vx must be <= 0.0 when oncoming gurney threatens bottleneck")
        # Must not select distant large reverse waypoint [-6.0, 0.6]
        if self.ctrl.yield_target is not None:
            self.assertGreater(self.ctrl.yield_target[0], -3.0, "Yield target must not be a distant reverse point")

    def test_closed_loop_robot_already_at_neg_0_59(self):
        """Verify controller safely stops or yields when robot is at x = -0.59m."""
        self.ctrl.current_wp_idx = 3
        robot_start = [-0.59, -0.60]
        gurney = MockHuman(px=5.0, py=-1.0, vx=-1.65, vy=0.0, radius=0.65)
        meta = [{'role': 'emergency_gurney'}]

        sim = ClosedLoopSim(self.ctrl, robot_start=robot_start, humans=[gurney], human_meta=meta)
        act = sim.step()

        self.assertLessEqual(act[0], 0.0, "Robot at x = -0.59m must not advance eastward into bottleneck")

    def test_closed_loop_insufficient_stopping_distance(self):
        """Verify gatekeeper stops robot before the bottleneck conflict zone (x <= -1.85m)."""
        self.ctrl.current_wp_idx = 2
        robot_start = [-2.10, -0.60]
        gurney = MockHuman(px=3.5, py=-1.0, vx=-1.65, vy=0.0, radius=0.65)
        meta = [{'role': 'emergency_gurney'}]

        sim = ClosedLoopSim(self.ctrl, robot_start=robot_start, humans=[gurney], human_meta=meta)

        for _ in range(10):
            sim.step()

        # Robot must remain West of the safe hold boundary (-1.85m)
        self.assertLessEqual(sim.robot_pos[0], -1.80, "Robot must stop before bottleneck conflict zone")

    def test_closed_loop_alcove_path_blocked(self):
        """Verify controller rejects North Alcove when path is blocked by a pedestrian, safely stopping."""
        # Blocking human directly between corridor and North Alcove
        blocker = MockHuman(px=-2.20, py=0.40, vx=0.0, vy=0.0, radius=0.35)
        gurney = MockHuman(px=4.0, py=-1.0, vx=-1.65, vy=0.0, radius=0.65)
        humans = [blocker, gurney]
        meta = [{'role': 'patient_walker'}, {'role': 'emergency_gurney'}]

        self.ctrl.current_wp_idx = 1
        sim = ClosedLoopSim(self.ctrl, robot_start=[-2.20, -0.60], humans=humans, human_meta=meta)

        act = sim.step()
        # Controller must detect blocked alcove path, reject alcove, and hold safely in place
        self.assertEqual(self.ctrl.state, HospitalNavigationController.STATE_HOLD_WAIT)
        self.assertEqual(act[0], 0.0)
        self.assertEqual(act[1], 0.0)

    def test_closed_loop_pedestrian_large_radius_not_mistaken_for_gurney(self):
        """Verify pedestrian with radius 0.60m is not falsely identified as gurney when real gurney is oncoming."""
        # Non-gurney pedestrian with large radius (e.g. bariatric patient) moving slowly
        large_ped = MockHuman(px=-5.0, py=0.0, vx=-0.30, vy=0.0, radius=0.60)
        # Real emergency gurney approaching from East
        real_gurney = MockHuman(px=3.5, py=-1.0, vx=-1.65, vy=0.0, radius=0.65)
        humans = [large_ped, real_gurney]

        # Call without metadata to test candidate scoring
        detected, _, _ = self.ctrl.detect_emergency_gurney(
            rx=-2.40, ry=1.25, vx=0.0, vy=0.0, humans=humans, human_meta=None
        )
        self.assertIs(detected, real_gurney, "Scoring must select the real oncoming gurney, not the west pedestrian")

    def test_closed_loop_gurney_stopped_at_goal_resumes(self):
        """Verify robot safely resumes when gurney has completed its route and stopped at goal outside conflict zone."""
        self.ctrl.state = HospitalNavigationController.STATE_HOLD_WAIT
        self.ctrl.yield_target = [-2.40, 1.25]
        # Gurney arrived at goal x = -10.5m, stopped (vx = 0.0)
        stopped_gurney = MockHuman(px=-10.5, py=-1.0, vx=0.0, vy=0.0, radius=0.65)
        meta = [{'role': 'emergency_gurney'}]

        sim = ClosedLoopSim(self.ctrl, robot_start=[-2.40, 1.25], humans=[stopped_gurney], human_meta=meta)

        for _ in range(6):
            sim.step()

        self.assertEqual(self.ctrl.state, HospitalNavigationController.STATE_RESUME,
                         "Robot must resume after gurney is verified stopped at goal outside conflict zone")


    def test_passage_window_granted_on_clearance(self):
        """Verify passage_granted_steps == 20 immediately after gurney clears (5 consecutive clear steps)."""
        self.ctrl.state = HospitalNavigationController.STATE_HOLD_WAIT
        self.ctrl.yield_target = [-2.40, 1.25]
        # Gurney already past: gx < rx - 1.20, gx < -1.60, moving west
        cleared_gurney = MockHuman(px=-5.0, py=-1.0, vx=-1.15, vy=0.0, radius=0.65)
        meta = [{'role': 'emergency_gurney'}]
        sim = ClosedLoopSim(self.ctrl, robot_start=[-2.40, 1.25],
                            humans=[cleared_gurney], human_meta=meta)
        for _ in range(6):
            sim.step()
        self.assertEqual(self.ctrl.state, HospitalNavigationController.STATE_RESUME,
                         "Controller must transition to RESUME after 5 clear steps")
        # Passage window should have been set (may already be counting down: 20 or 19)
        self.assertGreater(self.ctrl.passage_granted_steps, 0,
                           "passage_granted_steps must be > 0 after clearance")

    def test_passage_window_allows_crossing_after_gurney_loops(self):
        """Verify robot can cross bottleneck during passage window even when gurney teleports back to x=7."""
        self.ctrl.state = HospitalNavigationController.STATE_HOLD_WAIT
        self.ctrl.yield_target = [-2.40, 1.25]
        # First: grant clearance with gurney far west
        cleared_gurney = MockHuman(px=-5.0, py=-1.0, vx=-1.15, vy=0.0, radius=0.65)
        meta = [{'role': 'emergency_gurney'}]
        sim = ClosedLoopSim(self.ctrl, robot_start=[-2.40, 1.25],
                            humans=[cleared_gurney], human_meta=meta)
        for _ in range(6):
            sim.step()
        self.assertEqual(self.ctrl.state, HospitalNavigationController.STATE_RESUME)

        # Now simulate gurney teleporting back to x=7 (looped).
        # During passage window, controller must NOT re-trigger gurney_hazard.
        looped_gurney = MockHuman(px=7.0, py=-1.0, vx=-1.15, vy=0.0, radius=0.65)
        sim.humans = [looped_gurney]
        sim.human_meta = [{'role': 'emergency_gurney'}]

        # On the first step after the loop, passage window should still be active
        self.assertGreater(self.ctrl.passage_granted_steps, 0,
                           "Passage window must still be active right after loop")
        act = sim.step()
        # Robot must NOT have re-entered yield states; vx should be positive (moving east)
        self.assertNotEqual(self.ctrl.state, HospitalNavigationController.STATE_YIELD_ALCOVE,
                            "Robot must NOT re-trigger YIELD during passage window")
        self.assertNotEqual(self.ctrl.state, HospitalNavigationController.STATE_HOLD_WAIT,
                            "Robot must NOT re-trigger HOLD_WAIT during passage window")
        self.assertGreater(act[0], 0.0,
                           "Robot must move east (vx > 0) during passage window despite looped gurney")

    def test_passage_window_expires_and_yields_again(self):
        """Verify robot re-yields after passage window expires and gurney is approaching again."""
        # Manually set passage window to 1 (almost expired)
        self.ctrl.passage_granted_steps = 1
        self.ctrl.state = HospitalNavigationController.STATE_RESUME
        self.ctrl.current_wp_idx = 2  # approaching bottleneck

        # Gurney approaching from east, well within threat range
        oncoming = MockHuman(px=4.0, py=-1.0, vx=-1.15, vy=0.0, radius=0.65)
        meta = [{'role': 'emergency_gurney'}]
        # Robot inside bottleneck approach zone
        robot_pos = [-2.40, 1.25]
        # Step once — passage_granted_steps drops to 0
        self.ctrl.act(robot_pos, humans=[oncoming], action_dim=2, human_meta=meta)
        # Now window is expired; next step should re-trigger gurney_hazard
        self.ctrl.act(robot_pos, humans=[oncoming], action_dim=2, human_meta=meta)
        self.assertIn(self.ctrl.state,
                      [HospitalNavigationController.STATE_YIELD_ALCOVE,
                       HospitalNavigationController.STATE_HOLD_WAIT],
                      "Controller must re-yield after passage window expires with oncoming gurney")


if __name__ == '__main__':
    unittest.main()
