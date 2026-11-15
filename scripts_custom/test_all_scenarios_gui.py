#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_all_scenarios_gui.py
Automated end-to-end verification script for LibraryDemoApp:
Tests Library, Hotel Room Service, and MediRunner Hospital scenarios,
dynamic switching, A* path planning, canvas rendering, presets, and goal validation.
"""
from __future__ import print_function
import sys
import os
import unittest
import tkinter as tk

# Ensure paths
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/external')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from navrep_custom.library_gui import LibraryDemoApp
from navrep_custom.library_planner import (
    LIBRARY_PRESET_DESTINATIONS,
    HOTEL_PRESET_DESTINATIONS,
    HOSPITAL_PRESET_DESTINATIONS,
    validate_library_goal,
    validate_hotel_goal,
    validate_hospital_goal,
)


class TestAllScenariosGUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Requires DISPLAY=:99
        cls.root = tk.Tk()
        cls.app = LibraryDemoApp(cls.root, default_scenario="library")

    @classmethod
    def tearDownClass(cls):
        try:
            cls.root.destroy()
        except Exception:
            pass

    def test_01_library_initialization(self):
        """Verify initial state is library with correct bounds and presets."""
        self.assertEqual(self.app.active_scenario, "library")
        self.assertEqual(self.app.original_goal, [5.5, -3.5])
        self.assertIsNotNone(self.app.planner_data)
        self.assertGreater(len(self.app.active_a_star_waypoints), 5)
        # Test rendering
        self.app._render_current_frame()
        self.assertIsNotNone(self.app._last_bgr_img)
        self.assertEqual(self.app._last_bgr_img.shape[:2], (self.app.canvas_height, self.app.canvas_width))

    def test_02_switch_to_hotel(self):
        """Verify switching to hotel room service updates bounds and canvas."""
        self.app.switch_scenario("hotelroomservice")
        self.assertEqual(self.app.active_scenario, "hotelroomservice")
        self.assertEqual(self.app.original_goal, [10.0, 0.0])
        self.assertIsNotNone(self.app.planner_data)
        self.assertGreater(len(self.app.active_a_star_waypoints), 5)
        self.app._render_current_frame()
        self.assertIsNotNone(self.app._last_bgr_img)

    def test_03_switch_to_hospital(self):
        """Verify switching to hospital delivery updates bounds and canvas."""
        self.app.switch_scenario("medirunner")
        self.assertIn(self.app.active_scenario, ["medirunner", "hospital"])
        self.assertEqual(self.app.original_goal, [10.5, 0.0])
        self.assertIsNotNone(self.app.planner_data)
        self.assertGreater(len(self.app.active_a_star_waypoints), 5)
        self.app._render_current_frame()
        self.assertIsNotNone(self.app._last_bgr_img)

    def test_04_switch_back_to_library(self):
        """Verify switching back to library restores library geometry."""
        self.app.switch_scenario("library")
        self.assertEqual(self.app.active_scenario, "library")
        self.assertEqual(self.app.original_goal, [5.5, -3.5])
        self.assertIsNotNone(self.app.planner_data)
        self.app._render_current_frame()
        self.assertIsNotNone(self.app._last_bgr_img)

    def test_05_preset_selection_and_replan_each_scenario(self):
        """Verify preset selection and replanning works correctly across all three scenarios."""
        # 1. Library: choose Reference Table 1
        self.app.switch_scenario("library")
        preset_name = LIBRARY_PRESET_DESTINATIONS[1][0]
        self.app.on_preset_destination_selected(preset_name)
        self.assertIsNotNone(self.app.selected_goal)
        self.app.on_replan_pressed()
        self.assertGreater(len(self.app.active_a_star_waypoints), 3)

        # 2. Hotel: choose Suite 202
        self.app.switch_scenario("hotelroomservice")
        preset_name = HOTEL_PRESET_DESTINATIONS[1][0]
        self.app.on_preset_destination_selected(preset_name)
        self.assertIsNotNone(self.app.selected_goal)
        self.app.on_replan_pressed()
        self.assertGreater(len(self.app.active_a_star_waypoints), 3)

        # 3. Hospital: choose Room 102
        self.app.switch_scenario("medirunner")
        preset_name = HOSPITAL_PRESET_DESTINATIONS[2][0]
        self.app.on_preset_destination_selected(preset_name)
        self.assertIsNotNone(self.app.selected_goal)
        self.app.on_replan_pressed()
        self.assertGreater(len(self.app.active_a_star_waypoints), 3)

    def test_06_controller_act_each_scenario(self):
        """Verify active controller produces valid actions across all 3 scenarios."""
        scenarios = ["library", "hotelroomservice", "medirunner"]
        for scen in scenarios:
            self.app.switch_scenario(scen)
            robot = self.app.env.soadrl_sim.robot
            humans = self.app.env.soadrl_sim.humans
            action = self.app.ctrl.act([robot.px, robot.py], humans, action_dim=2)
            self.assertEqual(len(action), 2)
            self.assertFalse(any(map(lambda v: v is None or v != v, action)))  # no NaN


if __name__ == "__main__":
    unittest.main()
