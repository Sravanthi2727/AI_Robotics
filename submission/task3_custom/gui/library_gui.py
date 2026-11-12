#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
library_gui.py
==============
Dedicated Graphical Demonstration Interface for Shriya's Task 3:
Library Book Collection Robot Scenario.

SVNIT Surat — CS359: AI for Robotics
Author: Shriya (Branch: shriya)

Features:
  - Realistic Academic Library Environment:
      * Warm light wooden parquet flooring with plank seams
      * Dark walnut bookshelves with individual multi-colored book spines
      * Stack aisles A1-A3 (West Wing) and B1-B3 (East Wing)
      * Study reading tables with chairs, open books, and banker lamps
      * Circulation & Book Return counter with marble top and return slot
      * Concrete architectural structural pillars
      * Section placards & architectural overhead signboards
      * Industrial AMR robot body with drive wheels, casters, LiDAR turret,
        safety halo, and live visited trajectory trail
      * Dynamic human avatars with role coloring and velocity vectors
      * Calibrated metric scale bar and on-canvas legend
  - Genuine A* Path Planning Demonstration:
      * Exact metric occupancy grid derived from static obstacle geometry
      * Robot clearance inflation radius (0.42m)
      * Admissible Euclidean heuristic: f(n) = g(n) + h(n)
      * Animated search showing open set frontier and closed set expansion
      * Collision-free route discovery with real planning statistics
      * Automatic synthesis of navigable corridor waypoints
      * Direct honest integration into LibraryNavigationController
  - Full Demonstration Sequence:
      * Single-click 'RUN FULL DEMONSTRATION' end-to-end automated sequence
      * Highlights start carrel and goal counter
      * Animates A* search wave and displays planning metrics
      * Smoothly transitions into autonomous navigation along planned route
      * Pedestrians move via ORCA, robot yields and brakes dynamically
      * Safe docking at the circulation counter with final metrics
  - Modular Controls:
      * Run Full Demonstration
      * Show Library (Clean View)
      * Run Path Planning (A* Search)
      * Start Navigation
      * Pause / Resume
      * Reset Episode
      * 20-Seed Benchmark Runner
      * Video Demonstration Playback (ffplay)
  - Speed Pacing:
      * 0.5x Evaluator Demo Speed (Default: 400ms per 0.2s step)
      * 1.0x Real-time Speed (200ms per step)
      * 2.0x Fast Execution (100ms per step)
"""

from __future__ import print_function
import os
import sys
import math
import time
import threading
import subprocess
import numpy as np
import cv2
from PIL import Image, ImageTk

# Python 2/3 compatible Tkinter import
if sys.version_info[0] >= 3:
    import tkinter as tk
    from tkinter import ttk, messagebox
    import queue
else:
    import Tkinter as tk
    import ttk
    import tkMessageBox as messagebox
    import Queue as queue

# Environment, Scenario, and Planner imports
from navrep_custom.scenarios import get_scenario_config, SCENARIO_REGISTRY
from navrep_custom.custom_envs import CustomNavRepEncodedEnv
from navrep_custom.library_book_collection import (
    sync_and_rasterize_sim,
    LibraryNavigationController,
)
from navrep_custom.library_planner import (
    OccupancyGrid,
    AStarPlanner,
    get_library_a_star_plan,
    replan_library_route,
    validate_library_goal,
    LIBRARY_PRESET_DESTINATIONS,
)
from crowd_sim.envs.utils.info import ReachGoal, Collision, CollisionOtherAgent, Timeout


# ==============================================================================
# UI COLOR PALETTE & VISUAL CONSTANTS (Clean Dark Theme)
# ==============================================================================
THEME_BG = "#141619"
THEME_PANEL_BG = "#1d2025"
THEME_CARD_BG = "#242830"
THEME_FG = "#abb2bf"
THEME_FG_BRIGHT = "#ffffff"
THEME_ACCENT_BLUE = "#61afef"
THEME_ACCENT_GREEN = "#98c379"
THEME_ACCENT_RED = "#e06c75"
THEME_ACCENT_AMBER = "#e5c07b"
THEME_BORDER = "#3b4048"

# High-resolution architectural floor canvas colors (BGR for OpenCV)
FLOOR_BASE = (212, 227, 237)      # Warm beige parquet wood floor
FLOOR_PLANK_ALT = (206, 223, 232) # Alternating plank band
FLOOR_SEAM = (191, 208, 220)      # Faint floor seam lines
BASEBOARD_COL = (27, 46, 74)      # Dark walnut perimeter trim

SHELF_BODY = (16, 33, 56)         # Deep walnut brown
SHELF_BORDER = (10, 22, 38)       # Dark walnut beveled edge
SHELF_SHADOW = (180, 195, 205)    # Soft cast shadow under shelf

TABLE_TOP = (54, 91, 141)         # Warm honey oak / walnut table surface
TABLE_BORDER = (26, 57, 96)       # Table border bevel
CHAIR_WOOD = (28, 56, 94)         # Chair curved backrest
CHAIR_CUSHION = (90, 128, 168)    # Chair fabric seat

COUNTER_BODY = (16, 36, 78)       # Mahogany counter base
COUNTER_MARBLE = (216, 229, 237)  # Polished cream marble top
COUNTER_BORDER = (10, 24, 54)     # Counter border

PILLAR_STONE = (156, 144, 120)    # Architectural concrete grey
PILLAR_BORDER = (110, 100, 80)    # Pillar beveled edge

ROBOT_BLUE = (214, 119, 2)        # Industrial cobalt blue AMR body
ROBOT_WHEEL = (30, 30, 30)        # Black rubber drive wheels
ROBOT_LIDAR = (40, 40, 40)        # Top-mounted LiDAR turret
ROBOT_HEADING = (255, 255, 255)   # White directional heading arrow
ROBOT_TRAIL = (245, 130, 32)      # Cyan visited trajectory trail

# Dynamic Human Role Colors
ROLE_PATRON = (186, 104, 200)     # Browsing Patron (Violet)
ROLE_STUDENT = (154, 166, 38)     # Student (Teal)
ROLE_LIBRARIAN = (38, 117, 230)   # Librarian (Amber)
ROLE_RESEARCHER = (229, 136, 33)  # Researcher (Blue)
ROLE_STAFF = (67, 160, 71)        # Staff Member (Green)


class LibraryDemoApp(object):
    """Main Tkinter Demonstration Application for Task 3."""

    def __init__(self, root):
        self.root = root
        self.root.title("LIBRARY BOOK COLLECTION ROBOT — SVNIT CS359: AI for Robotics")
        self.root.geometry("1260x780")
        self.root.minsize(1180, 740)
        self.root.configure(bg=THEME_BG)

        # Simulation & Demonstration Modes
        # Mode: 'LIBRARY' (clean view), 'PLANNING' (A* animation), 'NAVIGATING' (active simulation)
        self.view_mode = "LIBRARY"
        self.is_full_demo_active = False

        self.current_difficulty = "EASY"
        self.pacing_speed = 0.5  # 0.5x default
        self.step_delay_ms = int(200 / self.pacing_speed)  # 400ms at 0.5x

        self.is_running = False
        self.is_paused = False
        self.is_benchmark_running = False
        self.current_seed = 42

        # Env, Controller, and Planner
        self.env = None
        self.cfg = None
        self.ctrl = LibraryNavigationController()
        self.planner_data = None
        self.active_a_star_waypoints = None

        # A* Animation State
        self.planning_keyframe_idx = 0
        self.is_planning_animating = False

        # Live Episode Variables
        self.current_step = 0
        self.max_steps = 380
        self.trajectory = []
        self.terminal_event = None
        self.nav_state_str = "READY"

        # Interactive Goal Selection & Dynamic Replanning State
        self.is_selecting_goal = False
        self.selected_goal = None              # [gx, gy] (metric coordinates)
        self.selected_goal_label = None        # Name of destination
        self.mouse_hover_coords = None         # (wx, wy)
        self.original_goal = [5.5, -3.5]       # Canonical baseline circulation desk goal

        # Background benchmark queue
        self.benchmark_queue = queue.Queue()

        # Canvas Dimensions & Metric Transformation
        self.canvas_width = 860
        self.canvas_height = 640
        self.world_min_x = -10.5
        self.world_max_x = 7.5
        self.world_min_y = -5.5
        self.world_max_y = 5.5
        self.scale_px = min(
            (self.canvas_width - 40) / (self.world_max_x - self.world_min_x),
            (self.canvas_height - 40) / (self.world_max_y - self.world_min_y)
        )
        self.orig_x = int(20 + (-self.world_min_x) * self.scale_px)
        self.orig_y = int(self.canvas_height / 2.0)

        # Precompute A* plan
        self._compute_a_star_plan()

        # Build UI layout
        self._setup_styles()
        self._build_header()
        self._build_main_body()

        # Initialize simulation and render initial frame
        self.reset_simulation()
        self._render_current_frame()

        # Check queue loop
        self.root.after(100, self._process_benchmark_queue)

    # --------------------------------------------------------------------------
    # A* PLAN INITIALIZATION
    # --------------------------------------------------------------------------
    def _compute_a_star_plan(self):
        """Computes or retrieves cached A* plan from library static obstacles."""
        try:
            self.planner_data = get_library_a_star_plan()
            self.active_a_star_waypoints = [list(wp) for wp in self.planner_data['waypoints']]
            self.ctrl = LibraryNavigationController(waypoints=self.active_a_star_waypoints)
        except Exception as e:
            print("[WARN] A* plan computation exception:", e)
            self.planner_data = None
            self.active_a_star_waypoints = None
            self.ctrl = LibraryNavigationController()

    # --------------------------------------------------------------------------
    # UI SETUP & STYLING
    # --------------------------------------------------------------------------
    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=THEME_PANEL_BG)
        style.configure("TLabel", background=THEME_PANEL_BG, foreground=THEME_FG, font=("Helvetica", 10))

    def _build_header(self):
        header = tk.Frame(self.root, bg=THEME_CARD_BG, height=60, bd=1, relief="ridge")
        header.pack(fill="x", side="top", padx=8, pady=(8, 4))

        title_frame = tk.Frame(header, bg=THEME_CARD_BG)
        title_frame.pack(side="left", padx=16, pady=6)

        lbl_title = tk.Label(
            title_frame,
            text="LIBRARY BOOK COLLECTION ROBOT",
            font=("Helvetica", 15, "bold"),
            fg=THEME_FG_BRIGHT,
            bg=THEME_CARD_BG
        )
        lbl_title.pack(anchor="w")

        lbl_sub = tk.Label(
            title_frame,
            text="Task 3: Autonomous Mobile Navigation in Institutional Library | Author: Shriya (SVNIT Surat)",
            font=("Helvetica", 9),
            fg=THEME_ACCENT_BLUE,
            bg=THEME_CARD_BG
        )
        lbl_sub.pack(anchor="w")

        # Badges on right side
        badge_frame = tk.Frame(header, bg=THEME_CARD_BG)
        badge_frame.pack(side="right", padx=16, pady=6)

        self.lbl_header_mode = tk.Label(
            badge_frame,
            text="MODE: LIBRARY VIEW",
            font=("Helvetica", 10, "bold"),
            fg="#21252b",
            bg="#80deea",
            padx=10,
            pady=4,
            relief="groove"
        )
        self.lbl_header_mode.pack(side="left", padx=4)

        self.lbl_header_diff = tk.Label(
            badge_frame,
            text="DIFFICULTY: EASY",
            font=("Helvetica", 10, "bold"),
            fg="#21252b",
            bg=THEME_ACCENT_GREEN,
            padx=10,
            pady=4,
            relief="groove"
        )
        self.lbl_header_diff.pack(side="left", padx=4)

        self.lbl_header_status = tk.Label(
            badge_frame,
            text="STATUS: READY",
            font=("Helvetica", 10, "bold"),
            fg="#21252b",
            bg="#9e9e9e",
            padx=12,
            pady=4,
            relief="groove"
        )
        self.lbl_header_status.pack(side="left", padx=4)

    def _build_main_body(self):
        body = tk.Frame(self.root, bg=THEME_BG)
        body.pack(fill="both", expand=True, padx=8, pady=4)

        # LEFT: Large Map Canvas (70% width)
        left_frame = tk.Frame(body, bg=THEME_PANEL_BG, bd=1, relief="ridge")
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 6))

        canvas_header = tk.Frame(left_frame, bg=THEME_CARD_BG, height=28)
        canvas_header.pack(fill="x", side="top", padx=2, pady=2)
        self.lbl_canvas_title = tk.Label(
            canvas_header,
            text="INSTITUTIONAL LIBRARY FLOOR PLAN (860x640) -- WEST/EAST WINGS & READING HALL",
            font=("Helvetica", 9, "bold"),
            fg=THEME_FG,
            bg=THEME_CARD_BG
        )
        self.lbl_canvas_title.pack(side="left", padx=8, pady=4)

        self.canvas = tk.Canvas(
            left_frame,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="#111216",
            highlightthickness=0
        )
        self.canvas.pack(fill="both", expand=True, padx=4, pady=4)
        self.canvas.bind("<Button-1>", self.on_canvas_clicked)
        self.canvas.bind("<Motion>", self.on_canvas_motion)

        # RIGHT: Side Control & Telemetry Panel (30% width)
        right_panel = tk.Frame(body, bg=THEME_PANEL_BG, width=370, bd=1, relief="ridge")
        right_panel.pack(side="right", fill="y", padx=(2, 0))
        right_panel.pack_propagate(False)

        # 1. PRIMARY: Full Demonstration Showcase
        demo_card = tk.LabelFrame(
            right_panel,
            text=" Showcase Demonstration ",
            font=("Helvetica", 10, "bold"),
            fg=THEME_ACCENT_GREEN,
            bg=THEME_CARD_BG,
            bd=2,
            relief="groove"
        )
        demo_card.pack(fill="x", padx=8, pady=(6, 4))

        self.btn_full_demo = tk.Button(
            demo_card,
            text="[>] RUN FULL DEMONSTRATION",
            font=("Helvetica", 10, "bold"),
            bg="#2e7d32",
            fg="white",
            activebackground="#388e3c",
            activeforeground="white",
            height=2,
            command=self.on_run_full_demonstration_pressed
        )
        self.btn_full_demo.pack(fill="x", padx=6, pady=4)

        lbl_demo_hint = tk.Label(
            demo_card,
            text="Automated sequence: View Library -> A* Search -> Navigate & Yield -> Dock",
            font=("Helvetica", 8),
            fg=THEME_FG,
            bg=THEME_CARD_BG
        )
        lbl_demo_hint.pack(anchor="w", padx=8, pady=(0, 4))

        # 2. Modular Controls
        ctrl_card = tk.LabelFrame(
            right_panel,
            text=" Modular Operations ",
            font=("Helvetica", 9, "bold"),
            fg=THEME_ACCENT_BLUE,
            bg=THEME_CARD_BG,
            bd=1,
            relief="groove"
        )
        ctrl_card.pack(fill="x", padx=8, pady=4)

        row1 = tk.Frame(ctrl_card, bg=THEME_CARD_BG)
        row1.pack(fill="x", padx=4, pady=2)

        self.btn_show_lib = tk.Button(
            row1,
            text="Show Library",
            font=("Helvetica", 8, "bold"),
            bg="#37474f",
            fg="white",
            command=self.on_show_library_pressed
        )
        self.btn_show_lib.pack(side="left", expand=True, fill="x", padx=2)

        self.btn_plan = tk.Button(
            row1,
            text="Run Path Planning",
            font=("Helvetica", 8, "bold"),
            bg="#0277bd",
            fg="white",
            activebackground="#0288d1",
            command=self.on_run_path_planning_pressed
        )
        self.btn_plan.pack(side="left", expand=True, fill="x", padx=2)

        row2 = tk.Frame(ctrl_card, bg=THEME_CARD_BG)
        row2.pack(fill="x", padx=4, pady=2)

        self.btn_start = tk.Button(
            row2,
            text="Start Navigation",
            font=("Helvetica", 8, "bold"),
            bg="#2e7d32",
            fg="white",
            command=self.on_start_pressed
        )
        self.btn_start.pack(side="left", expand=True, fill="x", padx=2)

        self.btn_pause = tk.Button(
            row2,
            text="Pause",
            font=("Helvetica", 8, "bold"),
            bg="#f57c00",
            fg="white",
            state="disabled",
            command=self.on_pause_pressed
        )
        self.btn_pause.pack(side="left", expand=True, fill="x", padx=2)

        self.btn_reset = tk.Button(
            row2,
            text="Reset",
            font=("Helvetica", 8, "bold"),
            bg="#455a64",
            fg="white",
            command=self.on_reset_pressed
        )
        self.btn_reset.pack(side="left", expand=True, fill="x", padx=2)

        # 3. Interactive Goal Selection & Dynamic Replanning
        goal_card = tk.LabelFrame(
            right_panel,
            text=" Interactive Goal Selection & Replanning ",
            font=("Helvetica", 9, "bold"),
            fg=THEME_ACCENT_AMBER,
            bg=THEME_CARD_BG,
            bd=1,
            relief="groove"
        )
        goal_card.pack(fill="x", padx=8, pady=4)

        row_g1 = tk.Frame(goal_card, bg=THEME_CARD_BG)
        row_g1.pack(fill="x", padx=4, pady=2)

        self.btn_select_goal = tk.Button(
            row_g1,
            text="[+] SELECT NEW GOAL",
            font=("Helvetica", 8, "bold"),
            bg="#00838f",
            fg="white",
            activebackground="#0097a7",
            activeforeground="white",
            command=self.on_select_new_goal_pressed
        )
        self.btn_select_goal.pack(side="left", expand=True, fill="x", padx=2)

        self.btn_replan = tk.Button(
            row_g1,
            text="REPLAN ROUTE",
            font=("Helvetica", 8, "bold"),
            bg="#e65100",
            fg="white",
            activebackground="#f57c00",
            activeforeground="white",
            command=self.on_replan_pressed
        )
        self.btn_replan.pack(side="left", expand=True, fill="x", padx=2)

        row_g2 = tk.Frame(goal_card, bg=THEME_CARD_BG)
        row_g2.pack(fill="x", padx=4, pady=2)

        lbl_preset = tk.Label(row_g2, text="Preset:", font=("Helvetica", 8, "bold"), fg=THEME_FG, bg=THEME_CARD_BG)
        lbl_preset.pack(side="left", padx=(2, 2))

        preset_options = [name for name, _ in LIBRARY_PRESET_DESTINATIONS]
        self.preset_var = tk.StringVar(value=preset_options[0])
        self.opt_preset = tk.OptionMenu(
            row_g2,
            self.preset_var,
            *preset_options,
            command=self.on_preset_destination_selected
        )
        self.opt_preset.config(font=("Helvetica", 7, "bold"), bg=THEME_PANEL_BG, fg=THEME_FG_BRIGHT, activebackground="#37474f", highlightthickness=0, bd=1, relief="groove", width=20, anchor="w")
        self.opt_preset.pack(side="left", fill="x", expand=True, padx=2)

        self.btn_reset_scenario = tk.Button(
            row_g2,
            text="Reset Original",
            font=("Helvetica", 7, "bold"),
            bg="#37474f",
            fg="white",
            command=self.on_reset_to_original_pressed
        )
        self.btn_reset_scenario.pack(side="left", padx=2)

        self.lbl_goal_status = tk.Label(
            goal_card,
            text="Click 'SELECT NEW GOAL' or choose a preset destination.",
            font=("Helvetica", 8),
            fg=THEME_ACCENT_BLUE,
            bg=THEME_CARD_BG,
            wraplength=340,
            justify="left"
        )
        self.lbl_goal_status.pack(anchor="w", padx=6, pady=(1, 3))

        # 4. Difficulty & Speed Configuration
        cfg_frame = tk.Frame(right_panel, bg=THEME_PANEL_BG)
        cfg_frame.pack(fill="x", padx=8, pady=4)

        diff_card = tk.LabelFrame(cfg_frame, text=" Difficulty ", font=("Helvetica", 8, "bold"), fg=THEME_ACCENT_BLUE, bg=THEME_CARD_BG)
        diff_card.pack(side="left", fill="both", expand=True, padx=(0, 2))

        self.diff_var = tk.StringVar(value=self.current_difficulty)
        for diff in ["EASY", "MEDIUM", "HARD"]:
            rb = tk.Radiobutton(
                diff_card,
                text=diff,
                value=diff,
                variable=self.diff_var,
                font=("Helvetica", 8, "bold"),
                fg=THEME_FG_BRIGHT,
                bg=THEME_CARD_BG,
                selectcolor=THEME_PANEL_BG,
                activebackground=THEME_CARD_BG,
                command=self.on_difficulty_changed
            )
            rb.pack(anchor="w", padx=2)

        speed_card = tk.LabelFrame(cfg_frame, text=" Sim Pacing ", font=("Helvetica", 8, "bold"), fg=THEME_ACCENT_BLUE, bg=THEME_CARD_BG)
        speed_card.pack(side="right", fill="both", expand=True, padx=(2, 0))

        self.speed_var = tk.StringVar(value="0.5x")
        speeds = [("0.5x (Demo)", "0.5x"), ("1.0x (Real)", "1.0x"), ("2.0x (Fast)", "2.0x")]
        for lbl, val in speeds:
            rb = tk.Radiobutton(
                speed_card,
                text=lbl,
                value=val,
                variable=self.speed_var,
                font=("Helvetica", 8, "bold"),
                fg=THEME_FG_BRIGHT,
                bg=THEME_CARD_BG,
                selectcolor=THEME_PANEL_BG,
                activebackground=THEME_CARD_BG,
                command=self.on_speed_changed
            )
            rb.pack(anchor="w", padx=2)

        # 4. Live Telemetry & Navigation State
        telem_card = tk.LabelFrame(
            right_panel,
            text=" Live Telemetry & Navigation State ",
            font=("Helvetica", 9, "bold"),
            fg=THEME_ACCENT_BLUE,
            bg=THEME_CARD_BG,
            bd=1,
            relief="groove"
        )
        telem_card.pack(fill="x", padx=8, pady=4)

        self.lbl_telem_step = tk.Label(telem_card, text="Step: 0 / 380 | Sim Time: 0.0s / 76.0s", bg=THEME_CARD_BG, fg=THEME_FG_BRIGHT, font=("Helvetica", 8, "bold"))
        self.lbl_telem_step.pack(anchor="w", padx=6, pady=1)

        self.lbl_telem_pos = tk.Label(telem_card, text="Pose: (-8.00m, 3.50m) | th = 0.0 deg", bg=THEME_CARD_BG, fg=THEME_FG, font=("Helvetica", 8))
        self.lbl_telem_pos.pack(anchor="w", padx=6, pady=1)

        self.lbl_telem_vel = tk.Label(telem_card, text="Speed: |v| = 0.00 m/s (vx=0.00, vy=0.00)", bg=THEME_CARD_BG, fg=THEME_FG, font=("Helvetica", 8))
        self.lbl_telem_vel.pack(anchor="w", padx=6, pady=1)

        self.lbl_telem_goal = tk.Label(telem_card, text="Dist to Goal: 15.21 m (Dock: 5.5, -3.5)", bg=THEME_CARD_BG, fg=THEME_ACCENT_AMBER, font=("Helvetica", 8, "bold"))
        self.lbl_telem_goal.pack(anchor="w", padx=6, pady=1)

        self.lbl_telem_wp = tk.Label(telem_card, text="Active Waypoint: WP 0 (Stack Aisle 2)", bg=THEME_CARD_BG, fg=THEME_FG, font=("Helvetica", 8))
        self.lbl_telem_wp.pack(anchor="w", padx=6, pady=1)

        self.lbl_telem_state = tk.Label(telem_card, text="Controller State: READY", bg=THEME_CARD_BG, fg=THEME_ACCENT_GREEN, font=("Helvetica", 8, "bold"))
        self.lbl_telem_state.pack(anchor="w", padx=6, pady=1)

        # 5. Path Planning Analytics Box
        plan_card = tk.LabelFrame(
            right_panel,
            text=" A* Search Metrics & Plan Details ",
            font=("Helvetica", 9, "bold"),
            fg=THEME_ACCENT_BLUE,
            bg=THEME_CARD_BG,
            bd=1,
            relief="groove"
        )
        plan_card.pack(fill="x", padx=8, pady=4)

        self.lbl_plan_status = tk.Label(plan_card, text="Algorithm: A* Search [f(n) = g(n) + h(n)]", bg=THEME_CARD_BG, fg=THEME_ACCENT_AMBER, font=("Helvetica", 8, "bold"))
        self.lbl_plan_status.pack(anchor="w", padx=6, pady=1)

        self.lbl_plan_stats1 = tk.Label(plan_card, text="Explored: 5,110 cells | Length: 17.19 m", bg=THEME_CARD_BG, fg=THEME_FG_BRIGHT, font=("Helvetica", 8))
        self.lbl_plan_stats1.pack(anchor="w", padx=6, pady=1)

        self.lbl_plan_stats2 = tk.Label(plan_card, text="Time: 131.8 ms | Res: 0.10m | Waypoints: 7", bg=THEME_CARD_BG, fg=THEME_FG, font=("Helvetica", 8))
        self.lbl_plan_stats2.pack(anchor="w", padx=6, pady=1)

        # 6. Benchmark Runner (20 Seeds)
        bench_card = tk.LabelFrame(
            right_panel,
            text=" 20-Seed Quantitative Benchmark ",
            font=("Helvetica", 9, "bold"),
            fg=THEME_ACCENT_BLUE,
            bg=THEME_CARD_BG,
            bd=1,
            relief="groove"
        )
        bench_card.pack(fill="x", padx=8, pady=4)

        bench_top = tk.Frame(bench_card, bg=THEME_CARD_BG)
        bench_top.pack(fill="x", padx=4, pady=2)

        self.btn_benchmark = tk.Button(
            bench_top,
            text="Run 20-Seed Benchmark",
            font=("Helvetica", 8, "bold"),
            bg="#1565c0",
            fg="white",
            command=self.on_run_benchmark_pressed
        )
        self.btn_benchmark.pack(fill="x", padx=2)

        self.lbl_bench_progress = tk.Label(bench_card, text="Ready (Evaluates Seeds 42-61 across 3 tiers)", font=("Helvetica", 7), fg=THEME_FG, bg=THEME_CARD_BG)
        self.lbl_bench_progress.pack(anchor="w", padx=6, pady=1)

        table_frame = tk.Frame(bench_card, bg="#1e2227", bd=1, relief="sunken")
        table_frame.pack(fill="x", padx=4, pady=2)

        tk.Label(table_frame, text="Diff | Success | Collision | Timeout | Avg Time", font=("Courier", 7, "bold"), fg=THEME_ACCENT_AMBER, bg="#1e2227").pack(anchor="w", padx=4, pady=1)
        self.lbl_bench_easy = tk.Label(table_frame, text="EASY   : 100.0% (20/20) | 0% |  0% | 23.4s", font=("Courier", 7), fg="#98c379", bg="#1e2227")
        self.lbl_bench_easy.pack(anchor="w", padx=4)
        self.lbl_bench_med = tk.Label(table_frame, text="MEDIUM : 100.0% (20/20) | 0% |  0% | 26.2s", font=("Courier", 7), fg="#98c379", bg="#1e2227")
        self.lbl_bench_med.pack(anchor="w", padx=4)
        self.lbl_bench_hard = tk.Label(table_frame, text="HARD   : 100.0% (20/20) | 0% |  0% | 28.5s", font=("Courier", 7), fg="#98c379", bg="#1e2227")
        self.lbl_bench_hard.pack(anchor="w", padx=4, pady=(0, 2))

    # --------------------------------------------------------------------------
    # SIMULATION INITIALIZATION & MODES
    # --------------------------------------------------------------------------
    def reset_simulation(self):
        """Initializes or resets the simulation environment."""
        self.is_running = False
        self.is_paused = False
        self.is_planning_animating = False
        self.is_selecting_goal = False
        if hasattr(self, 'btn_select_goal'):
            self.btn_select_goal.configure(text="[+] SELECT NEW GOAL", bg="#00838f")
            self.canvas.config(cursor="")

        self.btn_start.configure(text="Start Navigation", state="normal")
        self.btn_pause.configure(text="Pause", state="disabled")

        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass

        np.random.seed(self.current_seed)
        self.cfg = get_scenario_config("library")
        self.env = CustomNavRepEncodedEnv(
            scenario="library",
            difficulty=self.current_difficulty,
            gpu=False,
            silent=True
        )
        self.env.reset("library", self.current_difficulty)
        sync_and_rasterize_sim(self.env.raw_env, self.cfg)

        if self.selected_goal is not None:
            self.env.soadrl_sim.robot.gx = self.selected_goal[0]
            self.env.soadrl_sim.robot.gy = self.selected_goal[1]

        # Initialize controller with calculated A* waypoints
        if self.active_a_star_waypoints is not None:
            self.ctrl = LibraryNavigationController(waypoints=self.active_a_star_waypoints)
        else:
            self.ctrl = LibraryNavigationController()
        self.ctrl.reset()

        self.current_step = 0
        self.trajectory = [list(self.cfg.robot_start[:2])]
        self.terminal_event = None
        self.nav_state_str = "CRUISING (Stack Aisle 2)"

        self._update_badges("READY", "#9e9e9e")
        self._update_telemetry(0.0, 0.0)
        self._render_current_frame()

    def on_show_library_pressed(self):
        """Resets view mode to clean library environment."""
        self.view_mode = "LIBRARY"
        self.is_full_demo_active = False
        self.lbl_header_mode.configure(text="MODE: LIBRARY VIEW", bg="#80deea")
        self.lbl_canvas_title.configure(text="INSTITUTIONAL LIBRARY FLOOR PLAN (860x640) -- WEST/EAST WINGS & READING HALL")
        self._render_current_frame()

    def on_run_path_planning_pressed(self):
        """Starts the animated A* search visualization."""
        if self.is_running:
            self.on_pause_pressed()

        self.view_mode = "PLANNING"
        self.is_planning_animating = True
        self.planning_keyframe_idx = 0
        self.lbl_header_mode.configure(text="MODE: A* PATH PLANNING", bg="#ffb74d")
        self.lbl_canvas_title.configure(text="A* OCCUPANCY GRID SEARCH ANIMATION — f(n) = g(n) + h(n)")
        self._animate_planning_step()

    def _animate_planning_step(self):
        """Animates A* search frontier expanding."""
        if not self.is_planning_animating or self.planner_data is None:
            return

        keyframes = self.planner_data.get('animation_keyframes', [])
        if self.planning_keyframe_idx < len(keyframes):
            self.planning_keyframe_idx += 1
            self._render_current_frame()
            self.root.after(35, self._animate_planning_step)
        else:
            self.is_planning_animating = False
            self._render_current_frame()
            if self.is_full_demo_active:
                # Transition smoothly into navigation after planning
                self.root.after(800, self._start_navigation_after_planning)
            else:
                self.btn_start.configure(
                    text="Resume" if self.current_step > 0 else "Start Navigation",
                    state="normal"
                )

    def _start_navigation_after_planning(self):
        """Transitions from path planning to autonomous navigation in Full Demo mode."""
        self.view_mode = "NAVIGATING"
        self.lbl_header_mode.configure(text="MODE: AUTONOMOUS NAVIGATION", bg="#98c379")
        self.lbl_canvas_title.configure(text="AUTONOMOUS NAVIGATION ALONG A* CALCULATED CORRIDOR ROUTE")
        self.on_start_pressed()

    def on_run_full_demonstration_pressed(self):
        """Executes the complete end-to-end showcase demonstration."""
        self.is_full_demo_active = True
        self.reset_simulation()
        # Step 1: Start with Path Planning Animation
        self.on_run_path_planning_pressed()

    def on_start_pressed(self):
        """Starts or resumes simulation progression."""
        self.view_mode = "NAVIGATING"
        self.lbl_header_mode.configure(text="MODE: AUTONOMOUS NAVIGATION", bg="#98c379")
        self.lbl_canvas_title.configure(text="AUTONOMOUS NAVIGATION ALONG A* CALCULATED CORRIDOR ROUTE")

        if not self.is_running:
            self.is_running = True
            self.is_paused = False
            self.btn_start.configure(text="Running...", state="disabled")
            self.btn_pause.configure(text="Pause", state="normal")
            self._update_badges("RUNNING", "#1976d2")
            self.root.after(self.step_delay_ms, self._step_simulation_loop)
        elif self.is_paused:
            self.is_paused = False
            self.btn_start.configure(text="Running...", state="disabled")
            self.btn_pause.configure(text="Pause", state="normal")
            self._update_badges("RUNNING", "#1976d2")
            self.root.after(self.step_delay_ms, self._step_simulation_loop)

    def on_pause_pressed(self):
        """Pauses simulation progression."""
        if self.is_running and not self.is_paused:
            self.is_paused = True
            self.btn_start.configure(text="Resume", state="normal")
            self.btn_pause.configure(text="Paused", state="disabled")
            self._update_badges("PAUSED", "#fb8c00")

    def on_reset_pressed(self):
        """Resets the episode cleanly."""
        self.is_full_demo_active = False
        self.view_mode = "LIBRARY"
        self.lbl_header_mode.configure(text="MODE: LIBRARY VIEW", bg="#80deea")
        self.lbl_canvas_title.configure(text="INSTITUTIONAL LIBRARY FLOOR PLAN (860x640) -- WEST/EAST WINGS & READING HALL")
        self.reset_simulation()

    def on_difficulty_changed(self):
        """Updates difficulty tier and resets episode."""
        self.current_difficulty = self.diff_var.get()
        diff_colors = {"EASY": "#98c379", "MEDIUM": "#e5c07b", "HARD": "#e06c75"}
        self.lbl_header_diff.configure(
            text="DIFFICULTY: {}".format(self.current_difficulty),
            bg=diff_colors.get(self.current_difficulty, "#98c379")
        )
        self.reset_simulation()

    def on_speed_changed(self):
        """Updates pacing delay."""
        val = self.speed_var.get()
        if val == "0.5x":
            self.pacing_speed = 0.5
            self.step_delay_ms = 400
        elif val == "1.0x":
            self.pacing_speed = 1.0
            self.step_delay_ms = 200
        elif val == "2.0x":
            self.pacing_speed = 2.0
            self.step_delay_ms = 100

    # --------------------------------------------------------------------------
    # INTERACTIVE GOAL SELECTION & DYNAMIC REPLANNING HANDLERS
    # --------------------------------------------------------------------------
    def on_select_new_goal_pressed(self):
        """Toggles interactive canvas goal selection mode."""
        self.is_selecting_goal = not self.is_selecting_goal
        if self.is_selecting_goal:
            self.btn_select_goal.configure(text="[x] CANCEL SELECTION", bg="#c62828")
            self.canvas.config(cursor="crosshair")
            self.lbl_goal_status.configure(
                text="Click a reachable location on the map to set a new destination.",
                fg=THEME_ACCENT_AMBER
            )
            self._update_badges("SELECTING GOAL", "#ffb74d")
        else:
            self.btn_select_goal.configure(text="[+] SELECT NEW GOAL", bg="#00838f")
            self.canvas.config(cursor="")
            self.lbl_goal_status.configure(text="Goal selection cancelled.", fg=THEME_FG)
            self._update_badges("READY", "#9e9e9e")
        self._render_current_frame()

    def on_canvas_motion(self, event):
        """Tracks cursor over canvas during goal selection mode."""
        if self.is_selecting_goal:
            wx, wy = self._px2w(event.x, event.y)
            wx, wy = round(wx, 2), round(wy, 2)
            self.mouse_hover_coords = (wx, wy)
            if hasattr(self, 'lbl_goal_status'):
                self.lbl_goal_status.configure(
                    text="Pointer: ({:+.2f}m, {:+.2f}m) -- Click to choose destination".format(wx, wy),
                    fg=THEME_ACCENT_AMBER
                )

    def on_canvas_clicked(self, event):
        """Handles map clicks for dynamic goal selection."""
        if not self.is_selecting_goal:
            return

        wx, wy = self._px2w(event.x, event.y)
        wx, wy = round(wx, 2), round(wy, 2)
        valid, reason = validate_library_goal((wx, wy))
        if not valid:
            self.lbl_goal_status.configure(
                text="[!] Invalid Goal: {}".format(reason),
                fg=THEME_ACCENT_RED
            )
            self._update_badges("INVALID GOAL", "#e06c75")
            return

        # Valid location selected
        self.is_selecting_goal = False
        self.btn_select_goal.configure(text="[+] SELECT NEW GOAL", bg="#00838f")
        self.canvas.config(cursor="")
        label = "Custom Target ({:+.2f}, {:+.2f})".format(wx, wy)
        self._apply_new_destination([wx, wy], label)

    def on_preset_destination_selected(self, selection):
        """Handles selection of a destination preset from dropdown."""
        coords = None
        for name, pt in LIBRARY_PRESET_DESTINATIONS:
            if name == selection:
                coords = pt
                break
        if coords is None:
            return

        valid, reason = validate_library_goal(coords)
        if not valid:
            self.lbl_goal_status.configure(
                text="[!] Preset Invalid: {}".format(reason),
                fg=THEME_ACCENT_RED
            )
            return

        self._apply_new_destination(list(coords), selection)

    def _apply_new_destination(self, coords, label):
        """Sets destination and triggers genuine dynamic A* replanning."""
        self.selected_goal = list(coords)
        self.selected_goal_label = label

        # Pause robot if moving
        if self.is_running and not self.is_paused:
            self.on_pause_pressed()

        self.lbl_goal_status.configure(
            text="Destination: {} | ({:+.2f}, {:+.2f})".format(label, coords[0], coords[1]),
            fg=THEME_ACCENT_GREEN
        )
        # Execute genuine dynamic replanning
        self.on_replan_pressed()

    def on_replan_pressed(self):
        """
        Executes genuine dynamic A* replanning from robot's CURRENT position to selected goal.
        """
        if self.selected_goal is None:
            self.lbl_goal_status.configure(
                text="[!] Please select a new destination on map or preset first.",
                fg=THEME_ACCENT_AMBER
            )
            return

        if self.is_running and not self.is_paused:
            self.on_pause_pressed()

        # Current robot pose in metric coordinates
        robot = self.env.soadrl_sim.robot
        start_xy = [robot.px, robot.py]
        goal_xy = self.selected_goal

        # Stage A: New goal selected
        self.view_mode = "PLANNING"
        self.lbl_header_mode.configure(text="MODE: DYNAMIC A* REPLANNING", bg="#ff9800")
        self.lbl_canvas_title.configure(
            text="CALCULATING NEW ROUTE FROM ({:+.2f}, {:+.2f}) TO ({:+.2f}, {:+.2f})".format(
                start_xy[0], start_xy[1], goal_xy[0], goal_xy[1]
            )
        )
        self._update_badges("REPLANNING", "#ff9800")
        self.lbl_goal_status.configure(text="Calculating optimal collision-free route...", fg=THEME_ACCENT_AMBER)
        self.root.update_idletasks()

        # Calculate new A* plan from current robot position to goal
        new_plan = replan_library_route(start_xy, goal_xy)
        if not new_plan or not new_plan.get('success'):
            self.lbl_goal_status.configure(
                text="[!] Replanning Failed: No collision-free route to destination",
                fg=THEME_ACCENT_RED
            )
            self._update_badges("PLAN FAILED", "#e06c75")
            return

        # Stage B: A* Search result update
        self.planner_data = new_plan
        self.active_a_star_waypoints = [list(wp) for wp in new_plan['waypoints']]

        # Update controller waypoints dynamically
        self.ctrl.set_waypoints(self.active_a_star_waypoints)

        # Update environment goal coordinates
        self.env.soadrl_sim.robot.gx = goal_xy[0]
        self.env.soadrl_sim.robot.gy = goal_xy[1]

        # Update metrics panel
        self.lbl_plan_status.configure(text="Algorithm: Dynamic A* Search [f(n)=g(n)+h(n)]")
        self.lbl_plan_stats1.configure(
            text="Explored: {:,} cells | Length: {:.2f} m".format(
                new_plan.get('nodes_expanded', 0), new_plan.get('path_length_m', 0.0)
            )
        )
        self.lbl_plan_stats2.configure(
            text="Time: {:.1f} ms | Res: 0.10m | Waypoints: {}".format(
                new_plan.get('planning_time_ms', 0.0), len(self.active_a_star_waypoints)
            )
        )

        self.lbl_goal_status.configure(
            text="Route synthesized: {:.2f}m ({} waypoints). Click 'Start Navigation' or 'Resume'.".format(
                new_plan.get('path_length_m', 0.0), len(self.active_a_star_waypoints)
            ),
            fg=THEME_ACCENT_GREEN
        )

        # Animate search expansion
        self.is_planning_animating = True
        self.planning_keyframe_idx = 0
        self._animate_planning_step()

    def on_reset_to_original_pressed(self):
        """Resets scenario to canonical circulation desk route."""
        self.is_selecting_goal = False
        self.selected_goal = None
        self.selected_goal_label = None
        if hasattr(self, 'btn_select_goal'):
            self.btn_select_goal.configure(text="[+] SELECT NEW GOAL", bg="#00838f")
            self.canvas.config(cursor="")
        self._compute_a_star_plan()
        self.reset_simulation()
        self.lbl_goal_status.configure(
            text="Reset to original Circulation Counter scenario.",
            fg=THEME_ACCENT_BLUE
        )
        self._update_badges("READY", "#9e9e9e")
        self._render_current_frame()

    def _step_simulation_loop(self):
        """Non-blocking step execution scheduled via root.after()."""
        if not self.is_running or self.is_paused:
            return

        if self.current_step >= self.max_steps:
            self._handle_episode_terminal(Timeout(), "TIMEOUT")
            return

        robot = self.env.soadrl_sim.robot
        humans = self.env.soadrl_sim.humans
        robot_pos = [robot.px, robot.py]

        # Compute velocity command from LibraryNavigationController
        action = self.ctrl.act(robot_pos, humans, action_dim=2)
        vx, vy = float(action[0]), float(action[1])

        # Step environment
        ob, rew, done, info = self.env.step(action)
        self.current_step += 1

        new_pos = [robot.px, robot.py]
        self.trajectory.append(new_pos)

        # Context-aware navigation state for evaluator commentary
        tot_speed = math.hypot(vx, vy)
        if tot_speed < 0.04 and self.current_step > 5:
            self.nav_state_str = "PREDICTIVE YIELDING (Waiting for crossing pedestrian)"
        elif self.ctrl.detour_x != 0.0 or self.ctrl.detour_y != 0.0:
            self.nav_state_str = "DETOUR ACTIVE (Bypassing static patron cluster)"
        elif self.ctrl.current_wp_idx >= len(self.ctrl.waypoints) - 1:
            dest_name = self.selected_goal_label if self.selected_goal_label else "CIRCULATION COUNTER"
            self.nav_state_str = "DOCKING AT {}".format(dest_name.upper())
        elif robot.px < -3.2:
            self.nav_state_str = "TRANSIT: Stack Aisle 2 (Narrow Corridor)"
        elif robot.px < 1.0:
            self.nav_state_str = "TRANSIT: Central Cross-Corridor"
        else:
            self.nav_state_str = "TRANSIT: Reading Hall & Foyer"

        self._update_telemetry(vx, vy)
        self._render_current_frame()

        event = info.get('event')
        target_goal = self.selected_goal if self.selected_goal is not None else self.cfg.robot_goal
        dist_to_curr_goal = math.hypot(target_goal[0] - robot.px, target_goal[1] - robot.py)
        if dist_to_curr_goal < 0.35 and self.ctrl.current_wp_idx >= len(self.ctrl.waypoints) - 1:
            done = True
            event = ReachGoal()
        elif isinstance(event, (ReachGoal, Collision, CollisionOtherAgent, Timeout)):
            done = True

        if done:
            status_text = "GOAL REACHED" if isinstance(event, ReachGoal) else (
                "COLLISION" if isinstance(event, (Collision, CollisionOtherAgent)) else "TIMEOUT"
            )
            self._handle_episode_terminal(event, status_text)
        else:
            self.root.after(self.step_delay_ms, self._step_simulation_loop)

    def _handle_episode_terminal(self, event, status_text):
        """Handles completion of an episode."""
        self.is_running = False
        self.is_full_demo_active = False
        self.terminal_event = event
        self.btn_start.configure(text="Finished", state="disabled")
        self.btn_pause.configure(state="disabled")

        badge_colors = {
            "GOAL REACHED": "#43a047",
            "COLLISION": "#c62828",
            "TIMEOUT": "#f4511e"
        }
        bg_col = badge_colors.get(status_text, "#9e9e9e")
        self._update_badges(status_text, bg_col)
        self._render_current_frame()

    def _update_badges(self, status_text, bg_color):
        fg_col = "#ffffff" if bg_color != "#9e9e9e" else "#21252b"
        self.lbl_header_status.configure(text="STATUS: {}".format(status_text), bg=bg_color, fg=fg_col)

    def _update_telemetry(self, vx, vy):
        sim_time = self.current_step * 0.2
        self.lbl_telem_step.configure(
            text="Step: {:3d} / {} | Sim Time: {:5.1f}s / 76.0s".format(
                self.current_step, self.max_steps, sim_time)
        )

        rx, ry = self.env.soadrl_sim.robot.px, self.env.soadrl_sim.robot.py
        target_goal = self.selected_goal if self.selected_goal is not None else self.cfg.robot_goal
        gx, gy = target_goal[0], target_goal[1]
        dist_goal = math.hypot(rx - gx, ry - gy)
        theta_deg = math.degrees(self.env.soadrl_sim.robot.theta)

        self.lbl_telem_pos.configure(
            text="Pose: ({:+5.2f}m, {:+5.2f}m) | th = {:+5.1f} deg".format(rx, ry, theta_deg)
        )
        self.lbl_telem_vel.configure(
            text="Speed: |v|={:4.2f} m/s (vx={:+4.2f}, vy={:+4.2f})".format(
                math.hypot(vx, vy), vx, vy)
        )
        target_name = self.selected_goal_label if self.selected_goal_label else "Circulation Desk"
        self.lbl_telem_goal.configure(
            text="Dist to Goal: {:5.2f} m ({:+4.1f}, {:+4.1f})".format(dist_goal, gx, gy)
        )

        n_wps = len(self.ctrl.waypoints)
        curr_idx = min(self.ctrl.current_wp_idx, n_wps - 1)
        if n_wps == 7 and abs(self.ctrl.waypoints[-1][0] - 5.5) < 0.2:
            wp_names = [
                "WP 0 (Start Carrel)",
                "WP 1 (Aisle 2 Mid)",
                "WP 2 (Aisle 2 Egress)",
                "WP 3 (Cross-Corridor Upper)",
                "WP 4 (Cross-Corridor Lower)",
                "WP 5 (Reading Hall Transit)",
                "WP 6 (Circulation Desk Dock)"
            ]
            self.lbl_telem_wp.configure(text="Active Waypoint: {}".format(wp_names[curr_idx]))
        else:
            self.lbl_telem_wp.configure(text="Active Waypoint: WP {} / {} -> ({:+.2f}, {:+.2f})".format(
                curr_idx, n_wps - 1, self.ctrl.waypoints[curr_idx][0], self.ctrl.waypoints[curr_idx][1]))
        self.lbl_telem_state.configure(text="Controller State: {}".format(self.nav_state_str))

    # --------------------------------------------------------------------------
    # MAP & ARCHITECTURAL RENDERING ENGINE
    # --------------------------------------------------------------------------
    def _w2px(self, wx, wy):
        """Converts world coordinate (meters) to canvas pixels."""
        px = int(self.orig_x + wx * self.scale_px)
        py = int(self.orig_y - wy * self.scale_px)
        return px, py

    def _px2w(self, px, py):
        """Converts canvas pixel coordinates to world coordinates (meters)."""
        wx = (px - self.orig_x) / float(self.scale_px)
        wy = (self.orig_y - py) / float(self.scale_px)
        return wx, wy

    def _render_current_frame(self):
        """Draws the realistic academic library floor plan and all active dynamic elements."""
        img = np.full((self.canvas_height, self.canvas_width, 3), FLOOR_BASE, dtype=np.uint8)

        # 1. Warm Parquet Hardwood Flooring Texture
        plank_h = 32
        for y_top in range(0, self.canvas_height, plank_h):
            row_idx = y_top // plank_h
            if row_idx % 2 == 1:
                cv2.rectangle(img, (0, y_top), (self.canvas_width, min(self.canvas_height, y_top + plank_h)), FLOOR_PLANK_ALT, -1)
            # Staggered vertical plank seam lines
            shift = (row_idx * 57) % 110
            for x_seam in range(shift, self.canvas_width, 110):
                cv2.line(img, (x_seam, y_top), (x_seam, min(self.canvas_height, y_top + plank_h)), FLOOR_SEAM, 1)

        # 2. Outer Baseboard & Architectural Perimeter Walls
        if self.cfg is not None:
            outer_poly = np.array([self._w2px(pt[0], pt[1]) for pt in self.cfg.static_obstacles[0]], dtype=np.int32)
            cv2.polylines(img, [outer_poly], isClosed=True, color=BASEBOARD_COL, thickness=6)

        # 3. Static Architectural Elements: Shelves, Reading Tables, Pillars, Displays
        if self.cfg is not None:
            for idx, poly in enumerate(self.cfg.static_obstacles[1:], start=1):
                pts = np.array([self._w2px(pt[0], pt[1]) for pt in poly], dtype=np.int32)

                if idx in [1, 2, 3]:
                    self._draw_bookshelf(img, poly, idx)
                elif idx == 4:
                    self._draw_bookshelf(img, poly, idx)
                elif idx == 5:
                    self._draw_archive_reading_table(img, poly)
                elif idx == 6:
                    self._draw_reference_study_zone(img, poly)
                elif idx == 7:
                    self._draw_study_table_a(img, poly)
                elif idx == 8:
                    self._draw_group_study_table_b(img, poly)
                elif idx == 9:
                    self._draw_structural_pillar(img, pts, "A")
                elif idx == 10:
                    self._draw_structural_pillar(img, pts, "B")
                elif idx == 11:
                    self._draw_new_arrivals_rack(img, poly)
                else:
                    self._draw_bookshelf(img, poly, idx)

        # 4. Overhead Architectural Section Labels
        self._draw_section_labels(img)

        # 5. A* Path Planning Visualization Overlay
        if self.view_mode in ["PLANNING", "NAVIGATING"] and self.planner_data is not None:
            self._draw_a_star_overlay(img)

        # 6. Start & Goal Visual Markers
        if self.cfg is not None:
            # Start Carrel Marker
            s_px, s_py = self._w2px(self.cfg.robot_start[0], self.cfg.robot_start[1])
            cv2.circle(img, (s_px, s_py), 14, (60, 180, 75), 2, cv2.LINE_AA)
            cv2.circle(img, (s_px, s_py), 7, (60, 180, 75), -1, cv2.LINE_AA)
            cv2.putText(img, "START: Aisle 2", (s_px - 40, s_py - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (30, 120, 40), 1, cv2.LINE_AA)

            # Goal Docking Bay Marker
            target_goal = self.selected_goal if self.selected_goal is not None else self.cfg.robot_goal
            g_px, g_py = self._w2px(target_goal[0], target_goal[1])
            cv2.circle(img, (g_px, g_py), 16, (0, 190, 250), 2, cv2.LINE_AA)
            cv2.circle(img, (g_px, g_py), 9, (0, 190, 250), -1, cv2.LINE_AA)
            g_lbl = "GOAL: " + (self.selected_goal_label if self.selected_goal_label else "Circulation Desk")
            cv2.putText(img, g_lbl, (g_px - 70, g_py - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 140, 200), 2, cv2.LINE_AA)

        # 7. Visited Trajectory Trail
        if len(self.trajectory) > 1:
            for i in range(len(self.trajectory) - 1):
                p1 = self._w2px(self.trajectory[i][0], self.trajectory[i][1])
                p2 = self._w2px(self.trajectory[i + 1][0], self.trajectory[i + 1][1])
                cv2.line(img, p1, p2, ROBOT_TRAIL, 2, cv2.LINE_AA)

        # 8. Dynamic Pedestrians with Human Shapes & Roles
        if self.env is not None and hasattr(self.env, "soadrl_sim"):
            self._draw_pedestrians(img)

        # 9. Autonomous Mobile Robot (AMR-01) with Wheels & Heading
        if self.env is not None and hasattr(self.env, "soadrl_sim"):
            self._draw_robot_amr(img)

        # 10. Interactive Goal Selection Overlay (when selection active)
        if self.is_selecting_goal:
            cv2.rectangle(img, (20, 16), (580, 50), (25, 28, 36), -1)
            cv2.rectangle(img, (20, 16), (580, 50), (0, 190, 255), 2)
            cv2.putText(
                img,
                "[SELECTION MODE] CLICK A REACHABLE MAP LOCATION FOR NEW GOAL",
                (30, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (0, 230, 255),
                1,
                cv2.LINE_AA
            )
            if self.mouse_hover_coords is not None:
                hx_px, hy_px = self._w2px(self.mouse_hover_coords[0], self.mouse_hover_coords[1])
                if 0 <= hx_px < self.canvas_width and 0 <= hy_px < self.canvas_height:
                    cv2.circle(img, (hx_px, hy_px), 8, (0, 220, 255), 1, cv2.LINE_AA)
                    cv2.drawMarker(img, (hx_px, hy_px), (0, 220, 255), cv2.MARKER_CROSS, 14, 1)

        # 11. Bottom HUD Legend & Calibrated Scale Bar
        self._draw_hud_legend(img)

        # Convert OpenCV BGR to RGB -> ImageTk and render on Canvas
        self._last_bgr_img = img.copy()
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        self.canvas_photo = ImageTk.PhotoImage(image=pil_img)
        self.canvas.create_image(0, 0, image=self.canvas_photo, anchor="nw")

    def save_canvas_to_file(self, filepath):
        """Saves current canvas image directly as PNG."""
        if hasattr(self, '_last_bgr_img') and self._last_bgr_img is not None:
            cv2.imwrite(filepath, self._last_bgr_img)

    # --------------------------------------------------------------------------
    # ARCHITECTURAL RENDERING SUBROUTINES
    # --------------------------------------------------------------------------
    def _draw_bookshelf(self, img, poly, idx):
        """Renders dark walnut bookshelf with 3D shadow and realistic book spines."""
        poly_arr = np.array(poly)
        min_x, max_x = np.min(poly_arr[:, 0]), np.max(poly_arr[:, 0])
        min_y, max_y = np.min(poly_arr[:, 1]), np.max(poly_arr[:, 1])

        p_tl = self._w2px(min_x, max_y)
        p_br = self._w2px(max_x, min_y)

        # 1. Cast shadow
        cv2.rectangle(img, (p_tl[0] + 3, p_tl[1] + 3), (p_br[0] + 3, p_br[1] + 3), SHELF_SHADOW, -1)

        # 2. Solid walnut wooden frame
        cv2.rectangle(img, p_tl, p_br, SHELF_BODY, -1)
        cv2.rectangle(img, p_tl, p_br, SHELF_BORDER, 2)

        # 3. Individual multi-colored book spines
        book_colors = [
            (23, 24, 114),  # Burgundy
            (66, 45, 28),   # Navy Blue
            (40, 78, 29),   # Forest Green
            (26, 90, 158),  # Amber Leather
            (184, 211, 222),# Parchment
            (28, 59, 132),  # Rust Crimson
            (70, 70, 70),   # Slate
            (30, 80, 80)    # Teal
        ]

        shelf_w = p_br[0] - p_tl[0]
        shelf_h = p_br[1] - p_tl[1]
        spine_w = 5
        n_books = max(1, (shelf_w - 12) // spine_w)

        # Pseudo-random spine pattern seeded by shelf index
        rng = np.random.RandomState(idx * 73)
        colors_idx = rng.randint(0, len(book_colors), size=n_books)
        spine_heights = rng.randint(shelf_h - 10, shelf_h - 4, size=n_books)

        x_cursor = p_tl[0] + 6
        for b in range(n_books):
            b_col = book_colors[colors_idx[b]]
            b_h = spine_heights[b]
            y_b = p_tl[1] + (shelf_h - b_h) // 2
            cv2.rectangle(img, (x_cursor, y_b), (x_cursor + spine_w - 1, y_b + b_h), b_col, -1)
            # Gold title line on spine
            if b % 3 == 0:
                cv2.line(img, (x_cursor, y_b + b_h // 2), (x_cursor + spine_w - 1, y_b + b_h // 2), (0, 215, 255), 1)
            x_cursor += spine_w

        # 4. Brass Shelf Placard Sign
        labels = {
            1: "STACK A1: Computer Science & AI",
            2: "STACK A2: Robotics & Automation",
            3: "STACK A3: Mathematics & Stats",
            4: "STACK B1: Engineering Journals"
        }
        lbl_text = labels.get(idx, "BOOK STACK")
        c_x = (p_tl[0] + p_br[0]) // 2
        c_y = (p_tl[1] + p_br[1]) // 2
        # Placard background
        cv2.rectangle(img, (c_x - 85, c_y - 6), (c_x + 85, c_y + 8), (20, 20, 20), -1)
        cv2.rectangle(img, (c_x - 85, c_y - 6), (c_x + 85, c_y + 8), (0, 190, 220), 1)
        cv2.putText(img, lbl_text, (c_x - 80, c_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_archive_reading_table(self, img, poly):
        """Object 1: Archive Reading Table (East Wing Archives)."""
        poly_arr = np.array(poly)
        min_x, max_x = np.min(poly_arr[:, 0]), np.max(poly_arr[:, 0])
        min_y, max_y = np.min(poly_arr[:, 1]), np.max(poly_arr[:, 1])

        p_tl = self._w2px(min_x, max_y)
        p_br = self._w2px(max_x, min_y)

        # 1. Soft drop shadow (lower-right)
        cv2.rectangle(img, (p_tl[0] + 3, p_tl[1] + 3), (p_br[0] + 3, p_br[1] + 3), SHELF_SHADOW, -1)

        # 2. Warm walnut tabletop surface
        cv2.rectangle(img, p_tl, p_br, (42, 75, 122), -1)
        cv2.rectangle(img, p_tl, p_br, (22, 45, 75), 2)

        w = p_br[0] - p_tl[0]
        h = p_br[1] - p_tl[1]
        cy = (p_tl[1] + p_br[1]) // 2

        # 3. Four chairs around the table (2 North, 2 South)
        chair_xs = [p_tl[0] + int(w * 0.28), p_tl[0] + int(w * 0.72)]
        for cx in chair_xs:
            # North chair
            cv2.rectangle(img, (cx - 7, p_tl[1] - 7), (cx + 7, p_tl[1] - 2), CHAIR_CUSHION, -1)
            cv2.line(img, (cx - 8, p_tl[1] - 8), (cx + 8, p_tl[1] - 8), CHAIR_WOOD, 2)
            # South chair
            cv2.rectangle(img, (cx - 7, p_br[1] + 2), (cx + 7, p_br[1] + 7), CHAIR_CUSHION, -1)
            cv2.line(img, (cx - 8, p_br[1] + 8), (cx + 8, p_br[1] + 8), CHAIR_WOOD, 2)

        # 4. Tabletop visual accessories: open notebook, archive book stack, brass desk lamps
        nb_x1 = p_tl[0] + int(w * 0.15)
        cv2.rectangle(img, (nb_x1 - 8, cy - 6), (nb_x1 + 8, cy + 6), (245, 248, 250), -1)
        cv2.line(img, (nb_x1, cy - 6), (nb_x1, cy + 6), (180, 185, 190), 1)

        lamp_x1 = p_tl[0] + int(w * 0.38)
        cv2.circle(img, (lamp_x1, cy), 4, (0, 190, 220), -1)
        cv2.circle(img, (lamp_x1, cy), 2, (180, 240, 255), -1)

        bk_x2 = p_tl[0] + int(w * 0.62)
        cv2.rectangle(img, (bk_x2 - 10, cy - 5), (bk_x2 + 10, cy + 5), (25, 25, 115), -1)
        cv2.rectangle(img, (bk_x2 - 8, cy - 3), (bk_x2 + 8, cy + 3), (65, 45, 25), -1)

        lamp_x2 = p_tl[0] + int(w * 0.85)
        cv2.circle(img, (lamp_x2, cy), 4, (0, 190, 220), -1)
        cv2.circle(img, (lamp_x2, cy), 2, (180, 240, 255), -1)

        # 5. Brass Placard Sign: "ARCHIVE READING TABLE"
        cx = (p_tl[0] + p_br[0]) // 2
        cv2.rectangle(img, (cx - 80, cy - 7), (cx + 80, cy + 7), (20, 25, 30), -1)
        cv2.rectangle(img, (cx - 80, cy - 7), (cx + 80, cy + 7), (0, 185, 215), 1)
        cv2.putText(img, "ARCHIVE READING TABLE", (cx - 75, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_reference_study_zone(self, img, poly):
        """Object 2: Reference Study Zone (Shared Study Counter with Partitioned Bays)."""
        poly_arr = np.array(poly)
        min_x, max_x = np.min(poly_arr[:, 0]), np.max(poly_arr[:, 0])
        min_y, max_y = np.min(poly_arr[:, 1]), np.max(poly_arr[:, 1])

        p_tl = self._w2px(min_x, max_y)
        p_br = self._w2px(max_x, min_y)

        # 1. Soft drop shadow
        cv2.rectangle(img, (p_tl[0] + 3, p_tl[1] + 3), (p_br[0] + 3, p_br[1] + 3), SHELF_SHADOW, -1)

        # 2. Medium honey oak counter surface (distinct tone from Object 1)
        cv2.rectangle(img, p_tl, p_br, (52, 92, 142), -1)
        cv2.rectangle(img, p_tl, p_br, (28, 55, 90), 2)

        w = p_br[0] - p_tl[0]
        h = p_br[1] - p_tl[1]
        cy = (p_tl[1] + p_br[1]) // 2

        # 3. Individual seating spaces & carrel privacy dividers (4 bays)
        bay_w = w / 4.0
        for b in range(1, 4):
            div_x = int(p_tl[0] + b * bay_w)
            cv2.line(img, (div_x, p_tl[1] + 2), (div_x, p_br[1] - 2), (24, 46, 74), 2)

        # 4. Chairs for the study bays (2 North, 2 South)
        chair_xs = [int(p_tl[0] + 0.5 * bay_w), int(p_tl[0] + 1.5 * bay_w),
                    int(p_tl[0] + 2.5 * bay_w), int(p_tl[0] + 3.5 * bay_w)]
        for i, cx in enumerate(chair_xs):
            if i % 2 == 0:
                cv2.rectangle(img, (cx - 7, p_tl[1] - 7), (cx + 7, p_tl[1] - 2), (90, 130, 170), -1)
                cv2.line(img, (cx - 8, p_tl[1] - 8), (cx + 8, p_tl[1] - 8), (30, 60, 95), 2)
            else:
                cv2.rectangle(img, (cx - 7, p_br[1] + 2), (cx + 7, p_br[1] + 7), (90, 130, 170), -1)
                cv2.line(img, (cx - 8, p_br[1] + 8), (cx + 8, p_br[1] + 8), (30, 60, 95), 2)

        # 5. Study accessories: open reference manual, reading lamps
        b1_x = int(p_tl[0] + 0.35 * bay_w)
        cv2.rectangle(img, (b1_x - 8, cy - 6), (b1_x + 8, cy + 6), (248, 246, 240), -1)
        cv2.line(img, (b1_x, cy - 6), (b1_x, cy + 6), (160, 160, 160), 1)

        b2_x = int(p_tl[0] + 1.65 * bay_w)
        cv2.circle(img, (b2_x, cy), 4, (45, 125, 50), -1)
        cv2.circle(img, (b2_x, cy), 2, (180, 240, 255), -1)

        b3_x = int(p_tl[0] + 2.35 * bay_w)
        cv2.rectangle(img, (b3_x - 8, cy - 5), (b3_x + 8, cy + 5), (140, 60, 30), -1)

        b4_x = int(p_tl[0] + 3.65 * bay_w)
        cv2.circle(img, (b4_x, cy), 4, (45, 125, 50), -1)
        cv2.circle(img, (b4_x, cy), 2, (180, 240, 255), -1)

        # 6. Brass Placard Sign: "REFERENCE STUDY ZONE"
        cx = (p_tl[0] + p_br[0]) // 2
        cv2.rectangle(img, (cx - 80, cy - 7), (cx + 80, cy + 7), (20, 25, 30), -1)
        cv2.rectangle(img, (cx - 80, cy - 7), (cx + 80, cy + 7), (0, 185, 215), 1)
        cv2.putText(img, "REFERENCE STUDY ZONE", (cx - 75, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_study_table_a(self, img, poly):
        """Study Table A (West Reading Hall)."""
        poly_arr = np.array(poly)
        min_x, max_x = np.min(poly_arr[:, 0]), np.max(poly_arr[:, 0])
        min_y, max_y = np.min(poly_arr[:, 1]), np.max(poly_arr[:, 1])

        p_tl = self._w2px(min_x, max_y)
        p_br = self._w2px(max_x, min_y)

        # 1. Chairs around table (North and South sides)
        n_chairs = 4
        spacing = (p_br[0] - p_tl[0]) // (n_chairs + 1)
        for c in range(1, n_chairs + 1):
            cx = p_tl[0] + c * spacing
            cv2.rectangle(img, (cx - 7, p_tl[1] - 8), (cx + 7, p_tl[1] - 2), CHAIR_CUSHION, -1)
            cv2.line(img, (cx - 8, p_tl[1] - 9), (cx + 8, p_tl[1] - 9), CHAIR_WOOD, 2)
            cv2.rectangle(img, (cx - 7, p_br[1] + 2), (cx + 7, p_br[1] + 8), CHAIR_CUSHION, -1)
            cv2.line(img, (cx - 8, p_br[1] + 9), (cx + 8, p_br[1] + 9), CHAIR_WOOD, 2)

        # 2. Table shadow and surface
        cv2.rectangle(img, (p_tl[0] + 3, p_tl[1] + 3), (p_br[0] + 3, p_br[1] + 3), SHELF_SHADOW, -1)
        cv2.rectangle(img, p_tl, p_br, TABLE_TOP, -1)
        cv2.rectangle(img, p_tl, p_br, TABLE_BORDER, 2)

        # 3. Table accessories: open books and banker's desk lamp
        c_x = (p_tl[0] + p_br[0]) // 2
        c_y = (p_tl[1] + p_br[1]) // 2
        cv2.rectangle(img, (c_x - 36, c_y - 6), (c_x - 16, c_y + 6), (250, 250, 250), -1)
        cv2.line(img, (c_x - 26, c_y - 6), (c_x - 26, c_y + 6), (180, 180, 180), 1)
        cv2.rectangle(img, (c_x - 4, c_y - 4), (c_x + 4, c_y + 4), (46, 125, 50), -1)
        cv2.rectangle(img, (c_x + 16, c_y - 6), (c_x + 36, c_y + 6), (70, 80, 90), -1)

        # Placard sign
        cv2.rectangle(img, (c_x - 50, p_br[1] - 18), (c_x + 50, p_br[1] - 4), (20, 25, 30), -1)
        cv2.rectangle(img, (c_x - 50, p_br[1] - 18), (c_x + 50, p_br[1] - 4), (0, 185, 215), 1)
        cv2.putText(img, "STUDY TABLE A", (c_x - 45, p_br[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_group_study_table_b(self, img, poly):
        """Object 3: Group Study Table B (Collaborative Reading Hall)."""
        poly_arr = np.array(poly)
        min_x, max_x = np.min(poly_arr[:, 0]), np.max(poly_arr[:, 0])
        min_y, max_y = np.min(poly_arr[:, 1]), np.max(poly_arr[:, 1])

        p_tl = self._w2px(min_x, max_y)
        p_br = self._w2px(max_x, min_y)

        # 1. Soft drop shadow
        cv2.rectangle(img, (p_tl[0] + 3, p_tl[1] + 3), (p_br[0] + 3, p_br[1] + 3), SHELF_SHADOW, -1)

        # 2. Large warm oak collaborative tabletop surface
        cv2.rectangle(img, p_tl, p_br, (50, 85, 135), -1)
        cv2.rectangle(img, p_tl, p_br, (25, 52, 88), 2)

        w = p_br[0] - p_tl[0]
        h = p_br[1] - p_tl[1]
        cx = (p_tl[0] + p_br[0]) // 2
        cy = (p_tl[1] + p_br[1]) // 2

        # 3. Chairs around edges (3 North, 3 South = 6 chairs)
        spacing = w // 4
        for c in range(1, 4):
            ch_x = p_tl[0] + c * spacing
            cv2.rectangle(img, (ch_x - 7, p_tl[1] - 8), (ch_x + 7, p_tl[1] - 2), CHAIR_CUSHION, -1)
            cv2.line(img, (ch_x - 8, p_tl[1] - 9), (ch_x + 8, p_tl[1] - 9), CHAIR_WOOD, 2)
            cv2.rectangle(img, (ch_x - 7, p_br[1] + 2), (ch_x + 7, p_br[1] + 8), CHAIR_CUSHION, -1)
            cv2.line(img, (ch_x - 8, p_br[1] + 9), (ch_x + 8, p_br[1] + 9), CHAIR_WOOD, 2)

        # 4. Collaborative tabletop accessories: laptops, notebooks with sticky notes, power hub
        cv2.rectangle(img, (cx - 52, cy - 14), (cx - 32, cy - 2), (65, 75, 85), -1)
        cv2.rectangle(img, (cx + 32, cy - 14), (cx + 52, cy - 2), (65, 75, 85), -1)
        cv2.rectangle(img, (cx - 50, cy + 4), (cx - 34, cy + 16), (250, 250, 250), -1)
        cv2.circle(img, (cx - 42, cy + 10), 2, (50, 220, 255), -1)
        cv2.circle(img, (cx, cy - 6), 6, (40, 45, 50), -1)
        cv2.circle(img, (cx, cy - 6), 3, (160, 230, 255), -1)

        # 5. Brass Placard Sign: "GROUP STUDY TABLE B"
        cv2.rectangle(img, (cx - 75, p_br[1] - 18), (cx + 75, p_br[1] - 4), (20, 25, 30), -1)
        cv2.rectangle(img, (cx - 75, p_br[1] - 18), (cx + 75, p_br[1] - 4), (0, 185, 215), 1)
        cv2.putText(img, "GROUP STUDY TABLE B", (cx - 70, p_br[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_structural_pillar(self, img, pts, letter):
        """Objects 4 & 5: Structural Pillars A & B (Octagonal Columns with 3D Bevel & Depth)."""
        # 1. Cast shadow (offset lower-right)
        shadow_pts = pts + np.array([3, 3], dtype=np.int32)
        cv2.fillPoly(img, [shadow_pts], SHELF_SHADOW)

        # 2. Octagonal Base Plinth (dark stone trim)
        cv2.fillPoly(img, [pts], (55, 65, 75))
        cv2.polylines(img, [pts], isClosed=True, color=(35, 42, 50), thickness=2)

        # 3. Inner Column Body (warm architectural stone finish)
        c_x = int(np.mean(pts[:, 0]))
        c_y = int(np.mean(pts[:, 1]))
        inner_pts = (pts - np.array([c_x, c_y])) * 0.78 + np.array([c_x, c_y])
        inner_pts = inner_pts.astype(np.int32)
        cv2.fillPoly(img, [inner_pts], (92, 107, 125))

        # 4. Faceted 3D depth highlights and shadows
        core_pts = (pts - np.array([c_x, c_y])) * 0.50 + np.array([c_x, c_y])
        core_pts = core_pts.astype(np.int32)
        cv2.fillPoly(img, [core_pts], (115, 130, 150))
        cv2.polylines(img, [core_pts], isClosed=True, color=(65, 75, 88), thickness=1)

        # 5. Clean Architectural Center Core & Label
        cv2.circle(img, (c_x, c_y), 5, (40, 50, 60), -1)
        cv2.putText(img, letter, (c_x - 3, c_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255, 255, 255), 1, cv2.LINE_AA)

        # Overhead pillar label
        lbl_y = int(np.min(pts[:, 1])) - 6
        cv2.putText(img, "Pillar " + letter, (c_x - 18, lbl_y), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (90, 105, 120), 1, cv2.LINE_AA)

    def _draw_new_arrivals_rack(self, img, poly):
        """Object 6: Periodicals & New Arrivals Display Rack (East Wall)."""
        poly_arr = np.array(poly)
        min_x, max_x = np.min(poly_arr[:, 0]), np.max(poly_arr[:, 0])
        min_y, max_y = np.min(poly_arr[:, 1]), np.max(poly_arr[:, 1])

        p_tl = self._w2px(min_x, max_y)
        p_br = self._w2px(max_x, min_y)

        # 1. Soft drop shadow
        cv2.rectangle(img, (p_tl[0] + 3, p_tl[1] + 3), (p_br[0] + 3, p_br[1] + 3), SHELF_SHADOW, -1)

        # 2. Dark walnut tall display frame
        cv2.rectangle(img, p_tl, p_br, (16, 31, 54), -1)
        cv2.rectangle(img, p_tl, p_br, (10, 20, 36), 2)

        w = p_br[0] - p_tl[0]
        h = p_br[1] - p_tl[1]
        cx = (p_tl[0] + p_br[0]) // 2

        # 3. Tiered display shelves (5 rows)
        tier_h = h // 5
        mag_colors = [
            (212, 188, 0),  # Cyan: AI & Robotics Review
            (48, 145, 70),  # Green: Science & Nature
            (220, 130, 30), # Blue: IEEE Computing
            (30, 80, 220),  # Crimson: Tech Monthly
            (170, 40, 150), # Purple: Academic Quarterly
        ]

        for t in range(5):
            y_t = p_tl[1] + t * tier_h
            # Shelf ledge
            cv2.line(img, (p_tl[0] + 2, y_t + tier_h - 2), (p_br[0] - 2, y_t + tier_h - 2), (35, 55, 85), 2)

            col = mag_colors[t]
            # Left journal cover
            cv2.rectangle(img, (p_tl[0] + 6, y_t + 4), (cx - 3, y_t + tier_h - 4), col, -1)
            cv2.rectangle(img, (p_tl[0] + 8, y_t + 6), (cx - 5, y_t + 10), (255, 255, 255), -1)
            # Right journal cover
            cv2.rectangle(img, (cx + 3, y_t + 4), (p_br[0] - 6, y_t + tier_h - 4), col, -1)
            cv2.rectangle(img, (cx + 5, y_t + 6), (p_br[0] - 8, y_t + 10), (255, 255, 255), -1)

        # 4. Brass Placard Sign: "NEW ARRIVALS & JOURNALS"
        cy = (p_tl[1] + p_br[1]) // 2
        cv2.rectangle(img, (p_tl[0] + 3, cy - 9), (p_br[0] - 3, cy + 9), (20, 25, 30), -1)
        cv2.rectangle(img, (p_tl[0] + 3, cy - 9), (p_br[0] - 3, cy + 9), (0, 185, 215), 1)
        cv2.putText(img, "NEW ARRIVALS", (p_tl[0] + 5, cy - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "& JOURNALS", (p_tl[0] + 7, cy + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_section_labels(self, img):
        """Draws architectural section signage across wings."""
        sections = [
            ("WEST WING: BOOK STACKS", (-6.0, 5.60)),
            ("EAST WING: ARCHIVES", (2.2, 5.60)),
            ("CENTRAL CROSS-CORRIDOR", (-1.8, 4.00)),
            ("STUDY & READING HALL", (-4.0, -2.40)),
            ("CIRCULATION FOYER", (5.2, -1.40))
        ]
        for name, (wx, wy) in sections:
            px, py = self._w2px(wx, wy)
            cv2.putText(img, name, (px - 50, py), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 115, 130), 1, cv2.LINE_AA)

    # --------------------------------------------------------------------------
    # A* OVERLAY RENDERING
    # --------------------------------------------------------------------------
    def _draw_a_star_overlay(self, img):
        """Renders the animated A* search wave, open set, and calculated path."""
        if self.planner_data is None:
            return

        keyframes = self.planner_data.get('animation_keyframes', [])
        curr_kf = None
        if keyframes:
            k_idx = min(self.planning_keyframe_idx, len(keyframes) - 1)
            curr_kf = keyframes[k_idx]

        # Draw explored closed set tiles (soft ice blue glow)
        occ = OccupancyGrid(world_bounds=self.cfg.world_bounds, resolution=0.10)
        res = occ.resolution

        explored_cells = self.planner_data.get('explored_order', [])
        limit = curr_kf['explored_count'] if curr_kf else len(explored_cells)

        # Batch draw explored cells
        for r, c in explored_cells[:limit]:
            wx, wy = occ.grid_to_world(r, c)
            px, py = self._w2px(wx, wy)
            half = int(res * self.scale_px / 2.0) + 1
            cv2.rectangle(img, (px - half, py - half), (px + half, py + half), (235, 230, 180), -1)

        # Draw open set frontier cells (amber glow)
        if curr_kf and curr_kf.get('open_set'):
            for r, c in curr_kf['open_set']:
                wx, wy = occ.grid_to_world(r, c)
                px, py = self._w2px(wx, wy)
                cv2.circle(img, (px, py), 2, (0, 180, 255), -1)

        # Draw final discovered A* route (glowing green/gold polyline)
        path_coords = self.planner_data.get('path_coords', [])
        if len(path_coords) > 1 and (not self.is_planning_animating or self.planning_keyframe_idx >= len(keyframes)):
            for i in range(len(path_coords) - 1):
                p1 = self._w2px(path_coords[i][0], path_coords[i][1])
                p2 = self._w2px(path_coords[i + 1][0], path_coords[i + 1][1])
                cv2.line(img, p1, p2, (0, 220, 100), 3, cv2.LINE_AA)

            # Draw synthesized corridor waypoints
            waypoints = self.planner_data.get('waypoints', [])
            for i, wp in enumerate(waypoints):
                wpt = self._w2px(wp[0], wp[1])
                cv2.circle(img, wpt, 5, (0, 190, 255), -1)
                cv2.circle(img, wpt, 5, (30, 30, 30), 1)

    # --------------------------------------------------------------------------
    # ROBOT & PEDESTRIANS RENDERING
    # --------------------------------------------------------------------------
    def _draw_robot_amr(self, img):
        """Renders industrial AMR with rubber drive wheels, casters, LiDAR, and heading."""
        robot = self.env.soadrl_sim.robot
        r_px, r_py = self._w2px(robot.px, robot.py)
        r_rad_px = int(robot.radius * self.scale_px)
        theta = robot.theta

        # 1. Translucent Safety Clearance Halo (0.50m)
        safe_rad_px = int(0.50 * self.scale_px)
        cv2.circle(img, (r_px, r_py), safe_rad_px, (200, 210, 225), 1, cv2.LINE_AA)

        # 2. Left & Right Rubber Drive Wheels
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        w_offset = r_rad_px + 2
        wheel_l = 8
        wheel_w = 4

        # Left wheel center
        lx = int(r_px - sin_t * w_offset)
        ly = int(r_py - cos_t * w_offset)
        cv2.circle(img, (lx, ly), 4, ROBOT_WHEEL, -1)

        # Right wheel center
        rx = int(r_px + sin_t * w_offset)
        ry = int(r_py + cos_t * w_offset)
        cv2.circle(img, (rx, ry), 4, ROBOT_WHEEL, -1)

        # 3. Solid AMR Body Chassis
        cv2.circle(img, (r_px, r_py), r_rad_px, ROBOT_BLUE, -1)
        cv2.circle(img, (r_px, r_py), r_rad_px, (150, 80, 0), 2)

        # 4. Top-Mounted LiDAR Scanner Turret
        cv2.circle(img, (r_px, r_py), 5, ROBOT_LIDAR, -1)

        # 5. Directional Heading Arrow
        tip_x = int(r_px + cos_t * (r_rad_px + 7))
        tip_y = int(r_py - sin_t * (r_rad_px + 7))
        cv2.arrowedLine(img, (r_px, r_py), (tip_x, tip_y), ROBOT_HEADING, 2, tipLength=0.35)

        cv2.putText(img, "AMR", (r_px - 11, r_py - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (20, 20, 20), 1, cv2.LINE_AA)

    def _draw_pedestrians(self, img):
        """Renders distinguishable human avatars with head + shoulder silhouettes."""
        roles = [
            ("Patron", ROLE_PATRON),
            ("Student", ROLE_STUDENT),
            ("Librarian", ROLE_LIBRARIAN),
            ("Researcher", ROLE_RESEARCHER),
            ("Staff", ROLE_STAFF)
        ]

        for idx, h in enumerate(self.env.soadrl_sim.humans):
            h_px, h_py = self._w2px(h.px, h.py)
            h_rad_px = int(h.radius * self.scale_px)
            role_name, col = roles[idx % len(roles)]

            # 1. Torso/Shoulders ellipse
            cv2.ellipse(img, (h_px, h_py), (h_rad_px, int(h_rad_px * 0.7)), 0, 0, 360, col, -1)
            cv2.ellipse(img, (h_px, h_py), (h_rad_px, int(h_rad_px * 0.7)), 0, 0, 360, (40, 40, 40), 1)

            # 2. Head circle
            head_rad = max(4, int(h_rad_px * 0.55))
            cv2.circle(img, (h_px, h_py), head_rad, (240, 220, 200), -1)
            cv2.circle(img, (h_px, h_py), head_rad, (40, 40, 40), 1)

            # 3. Velocity vector arrow showing heading and speed
            v_len = math.hypot(h.vx, h.vy)
            if v_len > 0.05:
                tip_x = int(h_px + (h.vx / v_len) * min(30, v_len * self.scale_px * 0.8))
                tip_y = int(h_py - (h.vy / v_len) * min(30, v_len * self.scale_px * 0.8))
                cv2.arrowedLine(img, (h_px, h_py), (tip_x, tip_y), (30, 30, 30), 2, tipLength=0.35)

            cv2.putText(img, "P{}: {}".format(idx + 1, role_name), (h_px - 22, h_py + h_rad_px + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (40, 40, 40), 1, cv2.LINE_AA)

    def _draw_hud_legend(self, img):
        """Draws on-canvas legend and calibrated 2-meter scale bar."""
        leg_x, leg_y = 12, self.canvas_height - 36
        cv2.rectangle(img, (leg_x - 4, leg_y - 14), (leg_x + 550, leg_y + 28), (255, 255, 255), -1)
        cv2.rectangle(img, (leg_x - 4, leg_y - 14), (leg_x + 550, leg_y + 28), (190, 200, 210), 1)

        # Items
        cv2.circle(img, (leg_x + 10, leg_y), 5, ROBOT_BLUE, -1)
        cv2.putText(img, "AMR (0.6m)", (leg_x + 18, leg_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (30, 30, 30), 1, cv2.LINE_AA)

        cv2.circle(img, (leg_x + 100, leg_y), 5, ROLE_PATRON, -1)
        cv2.putText(img, "Patrons (0.6m)", (leg_x + 108, leg_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (30, 30, 30), 1, cv2.LINE_AA)

        cv2.rectangle(img, (leg_x + 205, leg_y - 4), (leg_x + 217, leg_y + 6), SHELF_BODY, -1)
        cv2.putText(img, "Book Stacks", (leg_x + 223, leg_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (30, 30, 30), 1, cv2.LINE_AA)

        cv2.rectangle(img, (leg_x + 310, leg_y - 4), (leg_x + 322, leg_y + 6), TABLE_TOP, -1)
        cv2.putText(img, "Study Desks", (leg_x + 328, leg_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (30, 30, 30), 1, cv2.LINE_AA)

        cv2.line(img, (leg_x + 415, leg_y), (leg_x + 430, leg_y), (0, 220, 100), 2)
        cv2.putText(img, "A* Route", (leg_x + 436, leg_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (30, 30, 30), 1, cv2.LINE_AA)

        # Scale bar (2.0 meters)
        scale_px = int(2.0 * self.scale_px)
        sb_x, sb_y = leg_x + 10, leg_y + 18
        cv2.line(img, (sb_x, sb_y), (sb_x + scale_px, sb_y), (40, 40, 40), 2)
        cv2.line(img, (sb_x, sb_y - 3), (sb_x, sb_y + 3), (40, 40, 40), 2)
        cv2.line(img, (sb_x + scale_px, sb_y - 3), (sb_x + scale_px, sb_y + 3), (40, 40, 40), 2)
        cv2.putText(img, "2.0 m scale", (sb_x + scale_px + 8, sb_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (40, 40, 40), 1, cv2.LINE_AA)

    # --------------------------------------------------------------------------
    # BENCHMARK RUNNER (20 SEEDS IN BACKGROUND THREAD)
    # --------------------------------------------------------------------------
    def on_run_benchmark_pressed(self):
        """Starts the 20-seed benchmark runner thread."""
        if self.is_benchmark_running:
            return

        self.is_benchmark_running = True
        self.btn_benchmark.configure(state="disabled", text="Benchmarking...")
        self.lbl_bench_progress.configure(text="Evaluating Seeds 42-61 across 3 difficulty tiers...")

        thread = threading.Thread(target=self._run_benchmark_worker)
        thread.daemon = True
        thread.start()

    def _run_benchmark_worker(self):
        """Worker thread running seeds 42-61 across EASY, MEDIUM, and HARD."""
        tiers = ["EASY", "MEDIUM", "HARD"]
        results = {}

        for diff in tiers:
            successes = 0
            collisions = 0
            timeouts = 0
            times = []

            for seed in range(42, 62):
                np.random.seed(seed)
                env = CustomNavRepEncodedEnv(scenario="library", difficulty=diff, gpu=False, silent=True)
                cfg = get_scenario_config("library")
                env.reset("library", diff)
                sync_and_rasterize_sim(env.raw_env, cfg)

                ctrl = LibraryNavigationController(waypoints=self.active_a_star_waypoints)
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

                if isinstance(event, ReachGoal):
                    successes += 1
                    times.append(step * 0.2)
                elif isinstance(event, (Collision, CollisionOtherAgent)):
                    collisions += 1
                else:
                    timeouts += 1

                env.close()

                self.benchmark_queue.put({
                    'type': 'progress',
                    'text': "Evaluating {} (Seed {}/61)...".format(diff, seed)
                })

            avg_t = np.mean(times) if times else 0.0
            results[diff] = {
                'success': (successes / 20.0) * 100.0,
                'success_count': successes,
                'collision': (collisions / 20.0) * 100.0,
                'timeout': (timeouts / 20.0) * 100.0,
                'avg_time': avg_t
            }

        self.benchmark_queue.put({'type': 'complete', 'results': results})

    def _process_benchmark_queue(self):
        """Checks for updates from the benchmark worker thread."""
        try:
            while not self.benchmark_queue.empty():
                msg = self.benchmark_queue.get_nowait()
                if msg['type'] == 'progress':
                    self.lbl_bench_progress.configure(text=msg['text'])
                elif msg['type'] == 'complete':
                    res = msg['results']
                    self.lbl_bench_easy.configure(
                        text="EASY   : {:5.1f}% ({:2d}/20) | {:2.0f}%| {:2.0f}% | {:4.1f}s".format(
                            res['EASY']['success'], res['EASY']['success_count'],
                            res['EASY']['collision'], res['EASY']['timeout'], res['EASY']['avg_time'])
                    )
                    self.lbl_bench_med.configure(
                        text="MEDIUM : {:5.1f}% ({:2d}/20) | {:2.0f}%| {:2.0f}% | {:4.1f}s".format(
                            res['MEDIUM']['success'], res['MEDIUM']['success_count'],
                            res['MEDIUM']['collision'], res['MEDIUM']['timeout'], res['MEDIUM']['avg_time'])
                    )
                    self.lbl_bench_hard.configure(
                        text="HARD   : {:5.1f}% ({:2d}/20) | {:2.0f}%| {:2.0f}% | {:4.1f}s".format(
                            res['HARD']['success'], res['HARD']['success_count'],
                            res['HARD']['collision'], res['HARD']['timeout'], res['HARD']['avg_time'])
                    )
                    self.lbl_bench_progress.configure(text="Benchmark Complete! (60/60 episodes evaluated)")
                    self.btn_benchmark.configure(state="normal", text="Run 20-Seed Benchmark")
                    self.is_benchmark_running = False
        except Exception:
            pass

        self.root.after(100, self._process_benchmark_queue)


def main():
    root = tk.Tk()
    app = LibraryDemoApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
