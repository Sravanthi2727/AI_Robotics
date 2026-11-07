#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
diagnose_ian_rendering.py
Step 1: Diagnostics for IANEnv rendering bugs and camera bounds comparison.
"""
from __future__ import print_function
import os
import sys
import numpy as np
import cv2

from navrep.envs.ianenv import IANEnv

def diagnose_scenario(sc):
    print("=" * 60)
    print("DIAGNOSING SCENARIO: {}".format(sc))
    print("=" * 60)
    env = IANEnv(silent=True)
    env.reset(sc)
    rlenv = env.iarlenv.rlenv
    m = rlenv.map2d

    # 1. Map properties
    extent = m.get_extent_xy() # [xmin, xmax, ymin, ymax]
    shape = m.occupancy().shape
    res = m.resolution()
    print("Map resolution: {:.4f} m/cell | Grid shape: {}".format(res, shape))
    print("Raw get_extent_xy():", extent)

    # Old recorder bug:
    # x_min, y_min, x_max, y_max = m.get_extent_xy()
    # x_bounds = [x_min, x_max]
    # y_bounds = [y_min, y_max]
    old_xmin, old_ymin, old_xmax, old_ymax = extent
    old_x_bounds = [old_xmin, old_xmax]
    old_y_bounds = [old_ymin, old_ymax]
    print("\n--- BUGGY RECORDER BOUNDS ---")
    print("old_x_bounds:", old_x_bounds, "span:", old_x_bounds[1] - old_x_bounds[0])
    print("old_y_bounds:", old_y_bounds, "span:", old_y_bounds[1] - old_y_bounds[0])

    # True bounds:
    true_xmin, true_xmax, true_ymin, true_ymax = extent
    true_x_bounds = [true_xmin, true_xmax]
    true_y_bounds = [true_ymin, true_ymax]
    print("\n--- CORRECT WORLD BOUNDS ---")
    print("true_x_bounds:", true_x_bounds, "span:", true_x_bounds[1] - true_x_bounds[0])
    print("true_y_bounds:", true_y_bounds, "span:", true_y_bounds[1] - true_y_bounds[0])

    # Robot & Goal poses
    robot_pos = rlenv.virtual_peppers[0].pos
    goal_pos = rlenv.agent_goals[0]
    print("\n--- AGENT & GOAL POSES ---")
    print("Robot pose [x, y, theta]:", robot_pos)
    print("Goal pose [x, y]:", goal_pos)
    print("Number of humans:", len(rlenv.virtual_peppers) - 1)
    if len(rlenv.virtual_peppers) > 1:
        print("Human 0 pose:", rlenv.virtual_peppers[1].pos)

    # Robot-centric window (e.g. 20m x 20m)
    rx, ry = robot_pos[0], robot_pos[1]
    cam_span = 20.0
    cam_x_bounds = [rx - cam_span/2.0, rx + cam_span/2.0]
    cam_y_bounds = [ry - cam_span/2.0, ry + cam_span/2.0]
    print("\n--- PROPOSED 20m ROBOT-CENTRIC CAMERA BOUNDS ---")
    print("cam_x_bounds:", cam_x_bounds)
    print("cam_y_bounds:", cam_y_bounds)

    env.close()

if __name__ == '__main__':
    for s in ['irosasl1', 'irosasl_office_j1', 'irosunity_scene_map1']:
        diagnose_scenario(s)

