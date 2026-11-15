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
from navrep_custom.hotel_room_service import HotelNavigationController
from navrep_custom.hospital_delivery import (
    HospitalNavigationController,
    sync_and_rasterize_hospital_sim,
)
from navrep_custom.library_planner import (
    OccupancyGrid,
    AStarPlanner,
    get_library_a_star_plan,
    replan_library_route,
    validate_library_goal,
    LIBRARY_PRESET_DESTINATIONS,
    get_hotel_a_star_plan,
    replan_hotel_route,
    validate_hotel_goal,
    HOTEL_PRESET_DESTINATIONS,
    get_hospital_a_star_plan,
    replan_hospital_route,
    validate_hospital_goal,
    HOSPITAL_PRESET_DESTINATIONS,
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

# Hotel Scenario Luxury Color Palette & Metadata
HOTEL_SUITE_METADATA = {
    4: ("Suite 201", "Deluxe King", True),
    5: ("Suite 202", "Executive Suite", True),
    6: ("Suite 203", "Presidential Suite", True),
    7: ("Suite 101", "Deluxe Twin", False),
    8: ("Suite 102", "Superior King", False),
    9: ("Suite 103", "Garden Villa Suite", False),
}
HOTEL_FLOOR_GROUT = (215, 220, 225)
HOTEL_CARPET_BASE = (38, 28, 128)
HOTEL_BRASS_TRIM = (45, 175, 215)
HOTEL_CARPET_PATTERN = (55, 45, 160)
HOTEL_WALL_WOOD = (25, 45, 75)
HOTEL_WALL_BORDER = (15, 28, 48)
HOTEL_DOOR_MAHOGANY = (18, 30, 60)
HOTEL_DOOR_BRASS = (50, 190, 230)
HOTEL_PLAQUE_BG = (15, 18, 24)
HOTEL_PLAQUE_GOLD = (50, 200, 240)
HOTEL_RECEPTION_TOP = (45, 75, 115)
HOTEL_RECEPTION_BELL = (50, 205, 245)
HOTEL_SOFA_SAPPHIRE = (115, 65, 30)
HOTEL_SOFA_BORDER = (80, 42, 18)
HOTEL_PLANT_POT = (35, 70, 110)
HOTEL_PLANT_GREEN = (45, 140, 60)
HOTEL_COFFEE_TABLE = (25, 42, 70)
HOTEL_TABLE_GLASS = (160, 200, 215)
HOTEL_MONITOR_BEZEL = (30, 30, 30)
HOTEL_MONITOR_SCREEN = (180, 170, 80)
HOTEL_TOWEL_WHITE = (245, 245, 250)
HOTEL_TOWEL_TEAL = (180, 160, 40)
HOTEL_HAMPER_COL = (45, 80, 120)
ROLE_GUEST = (220, 140, 45)
ROLE_HOUSEKEEPING = (45, 150, 80)
ROLE_LINEN_CART = (90, 100, 110)
ROLE_SUITCASE = (40, 50, 130)
HOTEL_ROBOT_SILVER = (205, 210, 215)
HOTEL_ROBOT_CLOCHE = (60, 210, 245)
HOTEL_ROBOT_HANDLE = (20, 140, 190)
HOTEL_ROBOT_AMBER = (0, 190, 255)

# Hospital Scenario Clinical Medical Palette & Metadata
HOSPITAL_FLOOR_BASE = (248, 250, 252)      # Antiseptic clinical porcelain / vinyl tiles
HOSPITAL_FLOOR_SEAM = (220, 228, 235)      # Tile grout seams
HOSPITAL_WALL_DARK = (45, 55, 65)          # Architectural corridor boundary walls
HOSPITAL_WALL_BORDER = (25, 35, 45)        # Wall border
HOSPITAL_BASEBOARD = (38, 50, 56)          # Baseboard
HOSPITAL_CROSS_RED = (40, 40, 225)         # Medical Red Cross
HOSPITAL_CROSS_CYAN = (200, 150, 0)        # Medical Cyan accent
HOSPITAL_TRIAGE_DESK = (165, 180, 195)     # Triage counter laminate
HOSPITAL_TRIAGE_TOP = (195, 210, 225)      # Triage counter surface
HOSPITAL_TRIAGE_BORDER = (120, 135, 150)
HOSPITAL_BED_FRAME = (130, 140, 150)
HOSPITAL_BED_LINEN = (255, 255, 255)
HOSPITAL_BED_PILLOW = (225, 238, 245)
HOSPITAL_BED_STRIPE = (210, 170, 70)       # Cyan medical blanket stripe
HOSPITAL_ECG_SCREEN = (25, 30, 35)
HOSPITAL_ECG_GREEN = (60, 230, 100)
HOSPITAL_ALCOVE_HAZARD = (40, 200, 240)    # Hazard refuge line
HOSPITAL_GURNEY_FRAME = (85, 90, 95)
HOSPITAL_GURNEY_PAD = (215, 140, 50)       # Emergency trauma mattress
HOSPITAL_GURNEY_FLASH = (20, 30, 240)      # High-intensity emergency beacon
ROLE_NURSE = (175, 145, 0)                 # Purposeful Nurse (Teal/Cyan scrubs in BGR)
ROLE_PATIENT_WALKER = (220, 180, 130)      # Frail Patient with walker
ROLE_VISITOR = (160, 120, 140)             # Visitor
ROLE_GURNEY = (30, 30, 220)                # Emergency Gurney
HOSPITAL_ROBOT_WHITE = (250, 252, 255)     # Clean clinical AMR body
HOSPITAL_ROBOT_ACCENT = (210, 150, 0)      # Clinical cyan trim



class LibraryDemoApp(object):
    """Main Tkinter Demonstration Application for Task 3."""

    def __init__(self, root, default_scenario="library"):
        if default_scenario == "hospital":
            default_scenario = "medirunner"
        self.active_scenario = default_scenario
        self.root = root
        title_str = "AUTONOMOUS MOBILE ROBOT NAVIGATION — SVNIT CS359: AI for Robotics"
        if self.active_scenario == "hotelroomservice":
            title_str = "GRAND HOTEL ROOM SERVICE ROBOT — SVNIT CS359: AI for Robotics"
        elif self.active_scenario in ["medirunner", "hospital"]:
            title_str = "MEDIRUNNER HOSPITAL MEDICINE DELIVERY ROBOT — SVNIT CS359: AI for Robotics"
        self.root.title(title_str)
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
        if self.active_scenario == "hotelroomservice":
            self.world_min_x = -12.5
            self.world_max_x = 12.5
            self.world_min_y = -8.5
            self.world_max_y = 8.5
            self.scale_px = min(
                (self.canvas_width - 40) / (self.world_max_x - self.world_min_x),
                (self.canvas_height - 40) / (self.world_max_y - self.world_min_y)
            )
            self.orig_x = int(20 + (-self.world_min_x) * self.scale_px)
            self.orig_y = int(self.canvas_height / 2.0)
            self.original_goal = [10.0, 0.0]
        elif self.active_scenario in ["medirunner", "hospital"]:
            self.world_min_x = -12.5
            self.world_max_x = 12.5
            self.world_min_y = -4.5
            self.world_max_y = 4.5
            self.scale_px = min(
                (self.canvas_width - 40) / (self.world_max_x - self.world_min_x),
                (self.canvas_height - 40) / (self.world_max_y - self.world_min_y)
            )
            self.orig_x = int(20 + (-self.world_min_x) * self.scale_px)
            self.orig_y = int(self.canvas_height / 2.0)
            self.original_goal = [10.5, 0.0]
        else:
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
            self.original_goal = [5.5, -3.5]

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
        """Computes or retrieves cached A* plan from static obstacles."""
        try:
            if self.active_scenario == "hotelroomservice":
                self.planner_data = get_hotel_a_star_plan()
                self.active_a_star_waypoints = [list(wp) for wp in self.planner_data['waypoints']]
                self.ctrl = HotelNavigationController(waypoints=self.active_a_star_waypoints)
            elif self.active_scenario in ["medirunner", "hospital"]:
                self.planner_data = get_hospital_a_star_plan()
                self.active_a_star_waypoints = [list(wp) for wp in self.planner_data['waypoints']]
                cfg = self.cfg if self.cfg is not None else get_scenario_config("medirunner")
                self.ctrl = HospitalNavigationController(
                    static_obstacles=cfg.static_obstacles,
                    world_bounds=cfg.world_bounds,
                    waypoints=self.active_a_star_waypoints
                )
            else:
                self.planner_data = get_library_a_star_plan()
                self.active_a_star_waypoints = [list(wp) for wp in self.planner_data['waypoints']]
                self.ctrl = LibraryNavigationController(waypoints=self.active_a_star_waypoints)
        except Exception as e:
            print("[WARN] A* plan computation exception:", e)
            self.planner_data = None
            self.active_a_star_waypoints = None
            if self.active_scenario == "hotelroomservice":
                self.ctrl = HotelNavigationController()
            elif self.active_scenario in ["medirunner", "hospital"]:
                self.ctrl = HospitalNavigationController()
            else:
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

        initial_title = "LIBRARY BOOK COLLECTION ROBOT"
        initial_sub = "Task 3: Autonomous Mobile Navigation in Institutional Library | Author: Shriya (SVNIT Surat)"
        initial_mode = "MODE: LIBRARY VIEW"
        if self.active_scenario == "hotelroomservice":
            initial_title = "GRAND HOTEL ROOM SERVICE ROBOT"
            initial_sub = "Task 3: Autonomous Room Service Delivery | Luxury Hotel Guest Corridors"
            initial_mode = "MODE: HOTEL VIEW"
        elif self.active_scenario in ["medirunner", "hospital"]:
            initial_title = "MEDIRUNNER HOSPITAL MEDICINE DELIVERY ROBOT"
            initial_sub = "Task 3: Autonomous Clinical Corridor Navigation | Emergency Gurney & Alcove Yielding"
            initial_mode = "MODE: HOSPITAL VIEW"

        self.lbl_title = tk.Label(
            title_frame,
            text=initial_title,
            font=("Helvetica", 15, "bold"),
            fg=THEME_FG_BRIGHT,
            bg=THEME_CARD_BG
        )
        self.lbl_title.pack(anchor="w")

        self.lbl_subtitle = tk.Label(
            title_frame,
            text=initial_sub,
            font=("Helvetica", 9),
            fg=THEME_ACCENT_BLUE,
            bg=THEME_CARD_BG
        )
        self.lbl_subtitle.pack(anchor="w")

        # Badges on right side
        badge_frame = tk.Frame(header, bg=THEME_CARD_BG)
        badge_frame.pack(side="right", padx=16, pady=6)

        self.lbl_header_mode = tk.Label(
            badge_frame,
            text=initial_mode,
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

        canvas_title_text = "INSTITUTIONAL LIBRARY FLOOR PLAN (860x640) -- WEST/EAST WINGS & READING HALL"
        if self.active_scenario == "hotelroomservice":
            canvas_title_text = "GRAND HOTEL FLOOR PLAN (860x640) -- GUEST CORRIDORS & SUITES"
        elif self.active_scenario in ["medirunner", "hospital"]:
            canvas_title_text = "HOSPITAL WING FLOOR PLAN (860x640) -- PHARMACY, CORRIDOR, TRIAGE & ICU"

        self.lbl_canvas_title = tk.Label(
            canvas_header,
            text=canvas_title_text,
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

        # 0. Active Scenario Selector
        scen_card = tk.LabelFrame(
            right_panel,
            text=" Active Robot Scenario ",
            font=("Helvetica", 9, "bold"),
            fg=THEME_ACCENT_BLUE,
            bg=THEME_CARD_BG,
            bd=1,
            relief="groove"
        )
        scen_card.pack(fill="x", padx=8, pady=(4, 2))

        self.btn_scen_lib = tk.Button(
            scen_card,
            text="[1] Institutional Library Book Collection",
            font=("Helvetica", 8, "bold"),
            bg="#0277bd" if self.active_scenario == "library" else "#263238",
            fg="white",
            relief="sunken" if self.active_scenario == "library" else "raised",
            command=lambda: self.switch_scenario("library")
        )
        self.btn_scen_lib.pack(fill="x", padx=4, pady=2)

        self.btn_scen_hotel = tk.Button(
            scen_card,
            text="[2] Grand Hotel Room Service Delivery",
            font=("Helvetica", 8, "bold"),
            bg="#0277bd" if self.active_scenario == "hotelroomservice" else "#263238",
            fg="white",
            relief="sunken" if self.active_scenario == "hotelroomservice" else "raised",
            command=lambda: self.switch_scenario("hotelroomservice")
        )
        self.btn_scen_hotel.pack(fill="x", padx=4, pady=2)

        self.btn_scen_hospital = tk.Button(
            scen_card,
            text="[3] MediRunner Hospital Medicine Delivery",
            font=("Helvetica", 8, "bold"),
            bg="#0277bd" if self.active_scenario in ["medirunner", "hospital"] else "#263238",
            fg="white",
            relief="sunken" if self.active_scenario in ["medirunner", "hospital"] else "raised",
            command=lambda: self.switch_scenario("medirunner")
        )
        self.btn_scen_hospital.pack(fill="x", padx=4, pady=2)

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

        if self.active_scenario == "hotelroomservice":
            active_presets = HOTEL_PRESET_DESTINATIONS
        elif self.active_scenario in ["medirunner", "hospital"]:
            active_presets = HOSPITAL_PRESET_DESTINATIONS
        else:
            active_presets = LIBRARY_PRESET_DESTINATIONS
        preset_options = [name for name, _ in active_presets]
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
    def switch_scenario(self, scenario_name):
        """Switches between Library, Hotel Room Service, and MediRunner Hospital scenarios."""
        if scenario_name == "hospital":
            scenario_name = "medirunner"
        if scenario_name == self.active_scenario:
            return

        self.active_scenario = scenario_name
        if hasattr(self, 'btn_scen_lib'):
            self.btn_scen_lib.configure(
                bg="#0277bd" if self.active_scenario == "library" else "#263238",
                relief="sunken" if self.active_scenario == "library" else "raised"
            )
        if hasattr(self, 'btn_scen_hotel'):
            self.btn_scen_hotel.configure(
                bg="#0277bd" if self.active_scenario == "hotelroomservice" else "#263238",
                relief="sunken" if self.active_scenario == "hotelroomservice" else "raised"
            )
        if hasattr(self, 'btn_scen_hospital'):
            self.btn_scen_hospital.configure(
                bg="#0277bd" if self.active_scenario in ["medirunner", "hospital"] else "#263238",
                relief="sunken" if self.active_scenario in ["medirunner", "hospital"] else "raised"
            )

        if self.active_scenario == "hotelroomservice":
            self.world_min_x = -12.5
            self.world_max_x = 12.5
            self.world_min_y = -8.5
            self.world_max_y = 8.5
            self.scale_px = min(
                (self.canvas_width - 40) / (self.world_max_x - self.world_min_x),
                (self.canvas_height - 40) / (self.world_max_y - self.world_min_y)
            )
            self.orig_x = int(20 + (-self.world_min_x) * self.scale_px)
            self.orig_y = int(self.canvas_height / 2.0)
            self.original_goal = [10.0, 0.0]
            self.selected_goal = None
            self.selected_goal_label = None
            self.lbl_title.configure(text="GRAND HOTEL ROOM SERVICE ROBOT")
            self.lbl_subtitle.configure(text="Task 3: Autonomous Room Service Delivery | Luxury Hotel Guest Corridors")
            self.lbl_canvas_title.configure(text="GRAND HOTEL FLOOR PLAN (860x640) -- GUEST CORRIDORS & SUITES")
            active_presets = HOTEL_PRESET_DESTINATIONS
        elif self.active_scenario in ["medirunner", "hospital"]:
            self.world_min_x = -12.5
            self.world_max_x = 12.5
            self.world_min_y = -4.5
            self.world_max_y = 4.5
            self.scale_px = min(
                (self.canvas_width - 40) / (self.world_max_x - self.world_min_x),
                (self.canvas_height - 40) / (self.world_max_y - self.world_min_y)
            )
            self.orig_x = int(20 + (-self.world_min_x) * self.scale_px)
            self.orig_y = int(self.canvas_height / 2.0)
            self.original_goal = [10.5, 0.0]
            self.selected_goal = None
            self.selected_goal_label = None
            self.lbl_title.configure(text="MEDIRUNNER HOSPITAL MEDICINE DELIVERY ROBOT")
            self.lbl_subtitle.configure(text="Task 3: Autonomous Clinical Corridor Navigation | Emergency Gurney & Alcove Yielding")
            self.lbl_canvas_title.configure(text="HOSPITAL WING FLOOR PLAN (860x640) -- PHARMACY, CORRIDOR, TRIAGE & ICU")
            active_presets = HOSPITAL_PRESET_DESTINATIONS
        else:
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
            self.original_goal = [5.5, -3.5]
            self.selected_goal = None
            self.selected_goal_label = None
            self.lbl_title.configure(text="LIBRARY BOOK COLLECTION ROBOT")
            self.lbl_subtitle.configure(text="Task 3: Autonomous Mobile Navigation in Institutional Library | Author: Shriya (SVNIT Surat)")
            self.lbl_canvas_title.configure(text="INSTITUTIONAL LIBRARY FLOOR PLAN (860x640) -- WEST/EAST WINGS & READING HALL")
            active_presets = LIBRARY_PRESET_DESTINATIONS

        if hasattr(self, 'btn_show_env'):
            if self.active_scenario == "hotelroomservice":
                self.btn_show_env.configure(text="Show Hotel")
            elif self.active_scenario in ["medirunner", "hospital"]:
                self.btn_show_env.configure(text="Show Hospital")
            else:
                self.btn_show_env.configure(text="Show Library")
        if hasattr(self, 'lbl_seq'):
            if self.active_scenario == "hotelroomservice":
                self.lbl_seq.configure(text="sequence: View Hotel -> A* Search -> Navigate & Yield")
            elif self.active_scenario in ["medirunner", "hospital"]:
                self.lbl_seq.configure(text="sequence: View Hospital -> A* Search -> Navigate & Yield")
            else:
                self.lbl_seq.configure(text="sequence: View Library -> A* Search -> Navigate & Yield")

        self._rebuild_preset_menu(active_presets)
        self._compute_a_star_plan()
        self.on_reset_pressed()

    def _rebuild_preset_menu(self, preset_list):
        """Dynamically updates OptionMenu items for active scenario."""
        menu = self.opt_preset["menu"]
        menu.delete(0, "end")
        for name, _ in preset_list:
            menu.add_command(label=name, command=lambda val=name: self.preset_var.set(val) or self.on_preset_destination_selected(val))
        self.preset_var.set(preset_list[0][0])

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
        if self.active_scenario == "hotelroomservice":
            self.cfg = get_scenario_config("hotelroomservice")
            self.env = CustomNavRepEncodedEnv(
                scenario="hotelroomservice",
                difficulty=self.current_difficulty,
                gpu=False,
                silent=True
            )
            self.env.reset("hotelroomservice", self.current_difficulty)
            if self.selected_goal is not None:
                self.env.soadrl_sim.robot.gx = self.selected_goal[0]
                self.env.soadrl_sim.robot.gy = self.selected_goal[1]
            if self.active_a_star_waypoints is not None:
                self.ctrl = HotelNavigationController(waypoints=self.active_a_star_waypoints)
            else:
                self.ctrl = HotelNavigationController()
            self.ctrl.reset()
            self.max_steps = 380
        elif self.active_scenario in ["medirunner", "hospital"]:
            self.cfg = get_scenario_config("medirunner")
            self.env = CustomNavRepEncodedEnv(
                scenario="medirunner",
                difficulty=self.current_difficulty,
                gpu=False,
                silent=True
            )
            self.env.reset("medirunner", self.current_difficulty)
            sync_and_rasterize_hospital_sim(self.env, self.cfg)
            if self.selected_goal is not None:
                self.env.soadrl_sim.robot.gx = self.selected_goal[0]
                self.env.soadrl_sim.robot.gy = self.selected_goal[1]
            if self.active_a_star_waypoints is not None:
                self.ctrl = HospitalNavigationController(
                    static_obstacles=self.cfg.static_obstacles,
                    world_bounds=self.cfg.world_bounds,
                    waypoints=self.active_a_star_waypoints
                )
            else:
                self.ctrl = HospitalNavigationController(
                    static_obstacles=self.cfg.static_obstacles,
                    world_bounds=self.cfg.world_bounds
                )
            self.ctrl.reset()
            diff_cfg = self.cfg.get_difficulty_config(self.current_difficulty)
            self.max_steps = diff_cfg.get('time_limit', 350)
        else:
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
            if self.active_a_star_waypoints is not None:
                self.ctrl = LibraryNavigationController(waypoints=self.active_a_star_waypoints)
            else:
                self.ctrl = LibraryNavigationController()
            self.ctrl.reset()
            diff_cfg = self.cfg.get_difficulty_config(self.current_difficulty)
            self.max_steps = diff_cfg.get('time_limit', 380)

        self.current_step = 0
        self.trajectory = [list(self.cfg.robot_start[:2])]
        self.terminal_event = None
        if self.active_scenario in ["medirunner", "hospital"]:
            self.nav_state_str = "CRUISING (Central Pharmacy Wing)"
        elif self.active_scenario == "hotelroomservice":
            self.nav_state_str = "CRUISING (Kitchen Dispatch)"
        else:
            self.nav_state_str = "CRUISING (Stack Aisle 2)"

        self._update_badges("READY", "#9e9e9e")
        self._update_telemetry(0.0, 0.0)
        self._render_current_frame()

    def on_show_library_pressed(self):
        """Resets view mode to clean environment view."""
        self.view_mode = "LIBRARY"
        self.is_full_demo_active = False
        lbl = "MODE: LIBRARY VIEW"
        if self.active_scenario == "hotelroomservice":
            lbl = "MODE: HOTEL VIEW"
        elif self.active_scenario in ["medirunner", "hospital"]:
            lbl = "MODE: HOSPITAL VIEW"
        self.lbl_header_mode.configure(text=lbl, bg="#80deea")
        if self.active_scenario == "hotelroomservice":
            self.lbl_canvas_title.configure(text="GRAND HOTEL FLOOR PLAN (860x640) -- GUEST CORRIDORS & SUITES")
        elif self.active_scenario in ["medirunner", "hospital"]:
            self.lbl_canvas_title.configure(text="HOSPITAL WING FLOOR PLAN (860x640) -- PHARMACY, CORRIDOR, TRIAGE & ICU")
        else:
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
        if self.active_scenario == "hotelroomservice":
            self.lbl_header_mode.configure(text="MODE: HOTEL VIEW", bg="#80deea")
            self.lbl_canvas_title.configure(text="GRAND HOTEL FLOOR PLAN (860x640) -- GUEST CORRIDORS & SUITES")
        elif self.active_scenario in ["medirunner", "hospital"]:
            self.lbl_header_mode.configure(text="MODE: HOSPITAL VIEW", bg="#80deea")
            self.lbl_canvas_title.configure(text="HOSPITAL WING FLOOR PLAN (860x640) -- PHARMACY, CORRIDOR, TRIAGE & ICU")
        else:
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
        if self.active_scenario == "hotelroomservice":
            valid, reason = validate_hotel_goal((wx, wy))
        elif self.active_scenario in ["medirunner", "hospital"]:
            valid, reason = validate_hospital_goal((wx, wy))
        else:
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
        if self.active_scenario == "hotelroomservice":
            active_presets = HOTEL_PRESET_DESTINATIONS
        elif self.active_scenario in ["medirunner", "hospital"]:
            active_presets = HOSPITAL_PRESET_DESTINATIONS
        else:
            active_presets = LIBRARY_PRESET_DESTINATIONS
        for name, pt in active_presets:
            if name == selection:
                coords = pt
                break
        if coords is None:
            return

        if self.active_scenario == "hotelroomservice":
            valid, reason = validate_hotel_goal(coords)
        elif self.active_scenario in ["medirunner", "hospital"]:
            valid, reason = validate_hospital_goal(coords)
        else:
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
        if self.active_scenario == "hotelroomservice":
            new_plan = replan_hotel_route(start_xy, goal_xy)
        elif self.active_scenario in ["medirunner", "hospital"]:
            new_plan = replan_hospital_route(start_xy, goal_xy)
        else:
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
        orig_name = "Circulation Counter"
        if self.active_scenario == "hotelroomservice":
            orig_name = "Suite 204"
        elif self.active_scenario in ["medirunner", "hospital"]:
            orig_name = "ICU Nurse Station"
        self.lbl_goal_status.configure(
            text="Reset to original {} scenario.".format(orig_name),
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

        # Compute velocity command from active controller
        action = self.ctrl.act(robot_pos, humans, action_dim=2)
        vx, vy = float(action[0]), float(action[1])

        # Step environment
        ob, rew, done, info = self.env.step(action)
        self.current_step += 1

        new_pos = [robot.px, robot.py]
        self.trajectory.append(new_pos)

        # Context-aware navigation state for evaluator commentary
        tot_speed = math.hypot(vx, vy)
        has_detour = getattr(self.ctrl, 'detour_y', 0.0) != 0.0 or getattr(self.ctrl, 'detour_x', 0.0) != 0.0
        if self.active_scenario in ["medirunner", "hospital"]:
            c_state = getattr(self.ctrl, 'state', 'NORMAL')
            if c_state == getattr(self.ctrl, 'STATE_YIELDING', 'ALCOVE_YIELDING'):
                self.nav_state_str = "ALCOVE YIELDING (Pulling into safe alcove)"
            elif c_state == getattr(self.ctrl, 'STATE_WAITING', 'WAITING_IN_ALCOVE'):
                self.nav_state_str = "WAITING IN ALCOVE (Letting Emergency Gurney Pass)"
            elif c_state == getattr(self.ctrl, 'STATE_RESUMING', 'RESUMING_TRANSIT'):
                self.nav_state_str = "RESUMING TRANSIT (Gurney Safely Cleared)"
            elif tot_speed < 0.04 and self.current_step > 5:
                self.nav_state_str = "PREDICTIVE YIELDING (Pedestrian Proximity)"
            elif self.ctrl.current_wp_idx >= len(self.ctrl.waypoints) - 1:
                default_dest = "ICU NURSE STATION"
                dest_name = self.selected_goal_label if self.selected_goal_label else default_dest
                self.nav_state_str = "DOCKING AT {}".format(dest_name.upper())
            elif robot.px < -6.0:
                self.nav_state_str = "TRANSIT: Central Pharmacy Wing"
            elif -3.0 <= robot.px <= 1.0:
                self.nav_state_str = "TRANSIT: Triage Choke-Point (South Lane y=-0.6m)"
            elif robot.px > 6.0:
                self.nav_state_str = "TRANSIT: ICU Wing Approach"
            else:
                self.nav_state_str = "TRANSIT: Clinical Hallway"
        elif tot_speed < 0.04 and self.current_step > 5:
            self.nav_state_str = "PREDICTIVE YIELDING (Waiting for crossing pedestrian)"
        elif has_detour:
            self.nav_state_str = "DETOUR ACTIVE (Bypassing pedestrian obstacle)"
        elif self.ctrl.current_wp_idx >= len(self.ctrl.waypoints) - 1:
            default_dest = "SUITE 204" if self.active_scenario == "hotelroomservice" else "CIRCULATION COUNTER"
            dest_name = self.selected_goal_label if self.selected_goal_label else default_dest
            self.nav_state_str = "DOCKING AT {}".format(dest_name.upper())
        elif self.active_scenario == "hotelroomservice":
            self.nav_state_str = "TRANSIT: Guest Room Corridor"
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
        """Draws the realistic architectural floor plan and all active dynamic elements."""
        if self.active_scenario == "hotelroomservice":
            self._render_hotel_frame()
            return
        elif self.active_scenario in ["medirunner", "hospital"]:
            self._render_hospital_frame()
            return

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
    # --------------------------------------------------------------------------
    # HOTEL SCENARIO RENDERING PIPELINE (Completely separate from Library)
    # --------------------------------------------------------------------------
    def _render_hotel_frame(self):
        """Draws the luxury Grand Hotel corridor floor plan and dynamic elements."""
        img = np.full((self.canvas_height, self.canvas_width, 3), (242, 245, 248), dtype=np.uint8)

        # 1. Italian marble floor grout lines
        for gx in range(0, self.canvas_width, 36):
            cv2.line(img, (gx, 0), (gx, self.canvas_height), HOTEL_FLOOR_GROUT, 1)
        for gy in range(0, self.canvas_height, 36):
            cv2.line(img, (0, gy), (self.canvas_width, gy), HOTEL_FLOOR_GROUT, 1)

        # 2. Central plush runner carpet
        c_tl = self._w2px(-12.0, 2.7)
        c_br = self._w2px(12.0, -2.7)
        cv2.rectangle(img, (0, c_tl[1]), (self.canvas_width, c_br[1]), HOTEL_CARPET_BASE, -1)
        cv2.line(img, (0, c_tl[1]), (self.canvas_width, c_tl[1]), HOTEL_BRASS_TRIM, 4)
        cv2.line(img, (0, c_br[1]), (self.canvas_width, c_br[1]), HOTEL_BRASS_TRIM, 4)
        cv2.line(img, (0, c_tl[1] + 6), (self.canvas_width, c_tl[1] + 6), (70, 210, 245), 1)
        cv2.line(img, (0, c_br[1] - 6), (self.canvas_width, c_br[1] - 6), (70, 210, 245), 1)

        mid_y = int((c_tl[1] + c_br[1]) / 2.0)
        for dx in range(30, self.canvas_width, 60):
            d_pts = np.array([[dx, mid_y - 8], [dx + 12, mid_y], [dx, mid_y + 8], [dx - 12, mid_y]], dtype=np.int32)
            cv2.polylines(img, [d_pts], isClosed=True, color=HOTEL_CARPET_PATTERN, thickness=1)

        # 3. Outer architectural perimeter walls
        if self.cfg is not None:
            outer_poly = np.array([self._w2px(pt[0], pt[1]) for pt in self.cfg.static_obstacles[0]], dtype=np.int32)
            cv2.polylines(img, [outer_poly], isClosed=True, color=(20, 35, 60), thickness=6)

        # 4. Obstacles
        if self.cfg is not None:
            for idx, poly in enumerate(self.cfg.static_obstacles[1:], start=1):
                pts = np.array([self._w2px(pt[0], pt[1]) for pt in poly], dtype=np.int32)
                x_min, x_max = np.min(pts[:, 0]), np.max(pts[:, 0])
                y_min, y_max = np.min(pts[:, 1]), np.max(pts[:, 1])

                if idx == 1:
                    # Reception Desk
                    cv2.fillPoly(img, [pts + np.array([4, 4], dtype=np.int32)], (180, 190, 200))
                    cv2.fillPoly(img, [pts], HOTEL_RECEPTION_TOP)
                    cv2.polylines(img, [pts], isClosed=True, color=HOTEL_BRASS_TRIM, thickness=2)
                    b_px, b_py = self._w2px(-7.2, 4.8)
                    cv2.circle(img, (b_px, b_py), 6, HOTEL_RECEPTION_BELL, -1)
                    cv2.circle(img, (b_px, b_py), 7, (20, 120, 180), 1)
                    cv2.line(img, (b_px, b_py - 6), (b_px, b_py - 9), (20, 120, 180), 2)
                    m_px, m_py = self._w2px(-8.7, 4.8)
                    cv2.rectangle(img, (m_px - 8, m_py - 5), (m_px + 8, m_py + 5), HOTEL_MONITOR_BEZEL, -1)
                    cv2.rectangle(img, (m_px - 6, m_py - 4), (m_px + 6, m_py + 4), HOTEL_MONITOR_SCREEN, -1)
                    bl_px, bl_py = self._w2px(-7.9, 4.8)
                    cv2.rectangle(img, (bl_px - 6, bl_py - 4), (bl_px + 6, bl_py + 4), (40, 70, 120), -1)
                    c_x = int(np.mean(pts[:, 0]))
                    cv2.putText(img, "CONCIERGE & RECEPTION", (c_x - 72, y_min - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (30, 45, 80), 1, cv2.LINE_AA)

                elif idx in [2, 3]:
                    # Lobby Lounge Sofas
                    is_left = (idx == 2)
                    cv2.fillPoly(img, [pts + np.array([4, 4], dtype=np.int32)], (180, 190, 200))
                    cv2.fillPoly(img, [pts], HOTEL_SOFA_SAPPHIRE)
                    cv2.polylines(img, [pts], isClosed=True, color=HOTEL_SOFA_BORDER, thickness=2)
                    mid_x = int((x_min + x_max) / 2)
                    cv2.line(img, (mid_x, y_min), (mid_x, y_max), HOTEL_SOFA_BORDER, 1)
                    if is_left:
                        p_px, p_py = self._w2px(-7.0, -5.5)
                        cv2.circle(img, (p_px, p_py), 8, HOTEL_PLANT_POT, -1)
                        cv2.circle(img, (p_px, p_py), 6, HOTEL_PLANT_GREEN, -1)
                    else:
                        t_px, t_py = self._w2px(-2.85, -5.5)
                        cv2.rectangle(img, (t_px - 8, t_py - 10), (t_px + 8, t_py + 10), HOTEL_COFFEE_TABLE, -1)
                        cv2.rectangle(img, (t_px - 6, t_py - 8), (t_px + 6, t_py + 8), HOTEL_TABLE_GLASS, -1)
                        cv2.circle(img, (t_px, t_py), 3, (180, 80, 220), -1)
                        p_px, p_py = self._w2px(0.8, -5.5)
                        cv2.circle(img, (p_px, p_py), 8, HOTEL_PLANT_POT, -1)
                        cv2.circle(img, (p_px, p_py), 6, HOTEL_PLANT_GREEN, -1)
                        cv2.putText(img, "EXECUTIVE LOUNGE", (t_px - 50, y_max + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (30, 45, 80), 1, cv2.LINE_AA)

                elif idx in HOTEL_SUITE_METADATA:
                    suite_no, suite_type, is_north = HOTEL_SUITE_METADATA[idx]
                    cv2.fillPoly(img, [pts + np.array([3, 3], dtype=np.int32)], (180, 190, 200))
                    cv2.fillPoly(img, [pts], HOTEL_WALL_WOOD)
                    cv2.polylines(img, [pts], isClosed=True, color=HOTEL_WALL_BORDER, thickness=2)
                    door_w = 20
                    door_cx = int((x_min + x_max) / 2)
                    door_y1 = y_max - 8 if is_north else y_min
                    door_y2 = y_max if is_north else y_min + 8
                    cv2.rectangle(img, (door_cx - door_w // 2, door_y1), (door_cx + door_w // 2, door_y2), HOTEL_DOOR_MAHOGANY, -1)
                    cv2.polylines(img, [np.array([
                        [door_cx - door_w // 2, door_y1], [door_cx + door_w // 2, door_y1],
                        [door_cx + door_w // 2, door_y2], [door_cx - door_w // 2, door_y2]
                    ], dtype=np.int32)], isClosed=True, color=HOTEL_DOOR_BRASS, thickness=1)
                    plaque_w, plaque_h = 56, 14
                    plaque_x = door_cx - plaque_w // 2
                    plaque_y = door_y1 - 10 if is_north else door_y2 + 2
                    cv2.rectangle(img, (plaque_x, plaque_y), (plaque_x + plaque_w, plaque_y + plaque_h), HOTEL_PLAQUE_BG, -1)
                    cv2.rectangle(img, (plaque_x, plaque_y), (plaque_x + plaque_w, plaque_y + plaque_h), HOTEL_PLAQUE_GOLD, 1)
                    cv2.putText(img, suite_no, (plaque_x + 4, plaque_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.28, HOTEL_PLAQUE_GOLD, 1, cv2.LINE_AA)

                elif idx == 10:
                    # Housekeeping Depot
                    cv2.fillPoly(img, [pts + np.array([3, 3], dtype=np.int32)], (180, 190, 200))
                    cv2.fillPoly(img, [pts], (30, 48, 70))
                    cv2.polylines(img, [pts], isClosed=True, color=(50, 80, 110), thickness=2)
                    shelf_y = int((y_min + y_max) / 2)
                    cv2.line(img, (x_min + 6, shelf_y), (x_max - 6, shelf_y), (100, 130, 160), 2)
                    cv2.rectangle(img, (x_min + 10, shelf_y - 8), (x_min + 24, shelf_y - 2), HOTEL_TOWEL_WHITE, -1)
                    cv2.rectangle(img, (x_min + 28, shelf_y - 8), (x_min + 42, shelf_y - 2), HOTEL_TOWEL_TEAL, -1)
                    cv2.rectangle(img, (x_max - 28, shelf_y + 4), (x_max - 10, y_max - 4), HOTEL_HAMPER_COL, -1)
                    c_x = int(np.mean(pts[:, 0]))
                    cv2.putText(img, "HOUSEKEEPING & LINEN", (c_x - 55, y_min - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (30, 60, 90), 1, cv2.LINE_AA)

        # 5. A* Path Planning Visualization Overlay
        if self.view_mode in ["PLANNING", "NAVIGATING"] and self.planner_data is not None:
            self._draw_a_star_overlay(img)

        # 6. Start & Goal Markers
        if self.cfg is not None:
            s_px, s_py = self._w2px(self.cfg.robot_start[0], self.cfg.robot_start[1])
            cv2.rectangle(img, (s_px - 14, s_py - 18), (s_px + 14, s_py + 18), (170, 175, 180), -1)
            cv2.line(img, (s_px, s_py - 18), (s_px, s_py + 18), (110, 115, 120), 2)
            cv2.circle(img, (s_px, s_py), 14, (60, 180, 75), 2, cv2.LINE_AA)
            cv2.circle(img, (s_px, s_py), 6, (60, 180, 75), -1, cv2.LINE_AA)
            cv2.putText(img, "START: Pantry", (s_px - 45, s_py - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 255, 230), 1, cv2.LINE_AA)

            target_goal = self.selected_goal if self.selected_goal is not None else self.cfg.robot_goal
            g_px, g_py = self._w2px(target_goal[0], target_goal[1])
            cv2.rectangle(img, (g_px - 12, g_py - 22), (g_px + 12, g_py + 22), HOTEL_DOOR_MAHOGANY, -1)
            cv2.rectangle(img, (g_px - 12, g_py - 22), (g_px + 12, g_py + 22), HOTEL_DOOR_BRASS, 1)
            cv2.circle(img, (g_px, g_py), 16, (0, 190, 250), 2, cv2.LINE_AA)
            cv2.circle(img, (g_px, g_py), 8, HOTEL_ROBOT_CLOCHE, -1, cv2.LINE_AA)
            cv2.circle(img, (g_px, g_py - 8), 2, HOTEL_ROBOT_HANDLE, -1, cv2.LINE_AA)
            g_lbl = "GOAL: " + (self.selected_goal_label if self.selected_goal_label else "Suite 204")
            cv2.putText(img, g_lbl, (g_px - 55, g_py - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (70, 220, 255), 2, cv2.LINE_AA)

        # 7. Visited Trajectory Trail
        if len(self.trajectory) > 1:
            for i in range(len(self.trajectory) - 1):
                p1 = self._w2px(self.trajectory[i][0], self.trajectory[i][1])
                p2 = self._w2px(self.trajectory[i + 1][0], self.trajectory[i + 1][1])
                cv2.line(img, p1, p2, (30, 180, 245), 2, cv2.LINE_AA)

        # 8. Dynamic Pedestrians with Hotel Roles
        if self.env is not None and hasattr(self.env, "soadrl_sim"):
            self._draw_hotel_pedestrians(img)

        # 9. Hotel Room Service AMR Avatar
        if self.env is not None and hasattr(self.env, "soadrl_sim"):
            self._draw_hotel_robot(img)

        # 10. Interactive Goal Selection Overlay
        if self.is_selecting_goal:
            cv2.rectangle(img, (20, 16), (580, 50), (25, 28, 36), -1)
            cv2.rectangle(img, (20, 16), (580, 50), (0, 190, 255), 2)
            cv2.putText(
                img,
                "[HOTEL SELECTION MODE] CLICK A REACHABLE CORRIDOR LOCATION FOR ROOM SERVICE",
                (30, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                (0, 230, 255),
                1,
                cv2.LINE_AA
            )
            if self.mouse_hover_coords is not None:
                hx_px, hy_px = self._w2px(self.mouse_hover_coords[0], self.mouse_hover_coords[1])
                if 0 <= hx_px < self.canvas_width and 0 <= hy_px < self.canvas_height:
                    cv2.circle(img, (hx_px, hy_px), 8, (0, 220, 255), 1, cv2.LINE_AA)
                    cv2.drawMarker(img, (hx_px, hy_px), (0, 220, 255), cv2.MARKER_CROSS, 14, 1)

        # 11. HUD Legend
        self._draw_hotel_hud_legend(img)

        # Render on Canvas
        self._last_bgr_img = img.copy()
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        self.canvas_photo = ImageTk.PhotoImage(image=pil_img)
        self.canvas.create_image(0, 0, image=self.canvas_photo, anchor="nw")

    def _draw_hotel_robot(self, img):
        if self.env is None or not hasattr(self.env, "soadrl_sim"):
            return
        robot = self.env.soadrl_sim.robot
        rx, ry = self._w2px(robot.px, robot.py)
        rr = max(10, int(robot.radius * self.scale_px))
        theta = robot.theta

        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        nx = -sin_t
        ny = cos_t
        w1_x = int(rx + nx * (rr + 2))
        w1_y = int(ry - ny * (rr + 2))
        w2_x = int(rx - nx * (rr + 2))
        w2_y = int(ry + ny * (rr + 2))
        cv2.circle(img, (w1_x, w1_y), 3, (30, 30, 30), -1)
        cv2.circle(img, (w2_x, w2_y), 3, (30, 30, 30), -1)

        cv2.circle(img, (rx, ry), rr, HOTEL_ROBOT_SILVER, -1)
        cv2.circle(img, (rx, ry), rr, (140, 145, 150), 2)

        cloche_r = max(5, int(rr * 0.55))
        cv2.circle(img, (rx, ry), cloche_r, HOTEL_ROBOT_CLOCHE, -1)
        cv2.circle(img, (rx, ry), cloche_r, (160, 165, 170), 1)
        cv2.circle(img, (rx, ry), 2, HOTEL_ROBOT_HANDLE, -1)

        head_x = int(rx + cos_t * (rr - 2))
        head_y = int(ry - sin_t * (rr - 2))
        cv2.circle(img, (head_x, head_y), 3, HOTEL_ROBOT_AMBER, -1)

        ray_x = int(rx + cos_t * (rr + 12))
        ray_y = int(ry - sin_t * (rr + 12))
        cv2.line(img, (rx, ry), (ray_x, ray_y), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "AMR: Service", (rx - 36, ry - rr - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_hotel_pedestrians(self, img):
        if self.env is None or not hasattr(self.env, "soadrl_sim"):
            return
        sim = self.env.soadrl_sim
        for i, h in enumerate(sim.humans):
            hx, hy = self._w2px(h.px, h.py)
            hr = max(6, int(h.radius * self.scale_px))
            speed = math.hypot(h.vx, h.vy)

            role = 'pedestrian_guest'
            if self.cfg and hasattr(self.cfg, 'base_humans') and i < len(self.cfg.base_humans):
                role = self.cfg.base_humans[i].get('role', 'pedestrian_guest')

            if role == 'pedestrian_housekeeping':
                cv2.circle(img, (hx, hy), hr, ROLE_HOUSEKEEPING, -1)
                cv2.circle(img, (hx, hy), hr, (20, 90, 50), 1)
                if speed > 0.05:
                    dir_x = h.vx / speed
                    dir_y = -h.vy / speed
                else:
                    dir_x, dir_y = 1.0, 0.0
                cart_cx = int(hx + dir_x * (hr + 12))
                cart_cy = int(hy + dir_y * (hr + 12))
                cart_w, cart_h = 16, 14
                cv2.rectangle(img, (cart_cx - cart_w // 2, cart_cy - cart_h // 2),
                              (cart_cx + cart_w // 2, cart_cy + cart_h // 2), ROLE_LINEN_CART, -1)
                cv2.rectangle(img, (cart_cx - 6, cart_cy - 4), (cart_cx - 1, cart_cy + 4), HOTEL_TOWEL_WHITE, -1)
                cv2.putText(img, "Housekeeping", (hx - 36, hy - hr - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(img, (hx, hy), hr, ROLE_GUEST, -1)
                cv2.circle(img, (hx, hy), hr, (20, 60, 100), 1)
                if speed > 0.05:
                    dir_x = -h.vx / speed
                    dir_y = h.vy / speed
                else:
                    dir_x, dir_y = -1.0, 0.0
                sc_cx = int(hx + dir_x * (hr + 10))
                sc_cy = int(hy + dir_y * (hr + 10))
                cv2.rectangle(img, (sc_cx - 5, sc_cy - 4), (sc_cx + 5, sc_cy + 4), ROLE_SUITCASE, -1)
                cv2.line(img, (sc_cx, sc_cy), (hx, hy), (50, 50, 50), 1)
                cv2.putText(img, "Guest", (hx - 14, hy - hr - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255, 255, 255), 1, cv2.LINE_AA)

            if speed > 0.05:
                v_end_x = int(hx + h.vx * self.scale_px * 0.4)
                v_end_y = int(hy - h.vy * self.scale_px * 0.4)
                cv2.arrowedLine(img, (hx, hy), (v_end_x, v_end_y), (255, 255, 255), 1, tipLength=0.3)

    def _draw_hotel_hud_legend(self, img):
        cv2.rectangle(img, (20, self.canvas_height - 38), (560, self.canvas_height - 12), (25, 28, 34), -1)
        cv2.rectangle(img, (20, self.canvas_height - 38), (560, self.canvas_height - 12), (50, 55, 65), 1)
        cv2.circle(img, (32, self.canvas_height - 25), 6, HOTEL_ROBOT_SILVER, -1)
        cv2.circle(img, (32, self.canvas_height - 25), 3, HOTEL_ROBOT_CLOCHE, -1)
        cv2.putText(img, "AMR Cloche (0.6m)", (44, self.canvas_height - 21), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220, 225, 230), 1)
        cv2.circle(img, (155, self.canvas_height - 25), 6, ROLE_GUEST, -1)
        cv2.putText(img, "Guest + Luggage", (167, self.canvas_height - 21), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220, 225, 230), 1)
        cv2.circle(img, (275, self.canvas_height - 25), 6, ROLE_HOUSEKEEPING, -1)
        cv2.putText(img, "Housekeeping + Cart", (287, self.canvas_height - 21), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220, 225, 230), 1)
        cv2.rectangle(img, (415, self.canvas_height - 29), (427, self.canvas_height - 21), HOTEL_WALL_WOOD, -1)
        cv2.putText(img, "Suites", (433, self.canvas_height - 21), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220, 225, 230), 1)
        cv2.line(img, (485, self.canvas_height - 25), (505, self.canvas_height - 25), (40, 210, 80), 2)
        cv2.putText(img, "A* Route", (512, self.canvas_height - 21), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220, 225, 230), 1)

    def _render_hospital_frame(self):
        """Draws the realistic clinical hospital floor plan and dynamic medical entities."""
        img = np.full((self.canvas_height, self.canvas_width, 3), HOSPITAL_FLOOR_BASE, dtype=np.uint8)

        # 1. Antiseptic porcelain tile floor grout lines
        tile_sz = 34
        for gx in range(0, self.canvas_width, tile_sz):
            cv2.line(img, (gx, 0), (gx, self.canvas_height), HOSPITAL_FLOOR_SEAM, 1)
        for gy in range(0, self.canvas_height, tile_sz):
            cv2.line(img, (0, gy), (self.canvas_width, gy), HOSPITAL_FLOOR_SEAM, 1)

        # 2. Central clinical transit hallway subtle guide zone (y in [-2.0, 2.0])
        h_tl = self._w2px(-12.0, 2.0)
        h_br = self._w2px(12.0, -2.0)
        cv2.rectangle(img, (0, h_tl[1]), (self.canvas_width, h_br[1]), (242, 246, 249), -1)
        # Corridor boundary dashed guidelines
        for dx in range(10, self.canvas_width, 40):
            cv2.line(img, (dx, h_tl[1]), (dx + 20, h_tl[1]), (205, 218, 228), 2)
            cv2.line(img, (dx, h_br[1]), (dx + 20, h_br[1]), (205, 218, 228), 2)

        # 3. Outer boundary walls
        if self.cfg is not None:
            outer_poly = np.array([self._w2px(pt[0], pt[1]) for pt in self.cfg.static_obstacles[0]], dtype=np.int32)
            cv2.polylines(img, [outer_poly], isClosed=True, color=(20, 30, 42), thickness=6)

        # 4. Static obstacles: Corridor Walls & Central Triage Desk
        if self.cfg is not None:
            for idx, poly in enumerate(self.cfg.static_obstacles[1:], start=1):
                pts = np.array([self._w2px(pt[0], pt[1]) for pt in poly], dtype=np.int32)
                x_min, x_max = np.min(pts[:, 0]), np.max(pts[:, 0])
                y_min, y_max = np.min(pts[:, 1]), np.max(pts[:, 1])

                if idx in [1, 2, 3, 4]:
                    # Corridor partition walls (North 1, North 2, South 1, South 2)
                    cv2.fillPoly(img, [pts + np.array([3, 3], dtype=np.int32)], (200, 210, 220))
                    cv2.fillPoly(img, [pts], (42, 54, 68))
                    cv2.polylines(img, [pts], isClosed=True, color=(22, 32, 44), thickness=2)
                    cv2.line(img, (x_min + 4, y_min + 4), (x_max - 4, y_min + 4), (70, 85, 105), 1)

                    # Wall segment placards
                    c_x = int(np.mean(pts[:, 0]))
                    c_y = int(np.mean(pts[:, 1]))
                    if idx == 1:
                        cv2.putText(img, "NORTH WALL 1", (c_x - 38, c_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (160, 175, 190), 1, cv2.LINE_AA)
                    elif idx == 2:
                        cv2.putText(img, "NORTH WALL 2", (c_x - 38, c_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (160, 175, 190), 1, cv2.LINE_AA)
                    elif idx == 3:
                        cv2.putText(img, "SOUTH WALL 1", (c_x - 38, c_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (160, 175, 190), 1, cv2.LINE_AA)
                    elif idx == 4:
                        cv2.putText(img, "SOUTH WALL 2", (c_x - 38, c_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (160, 175, 190), 1, cv2.LINE_AA)

                elif idx == 5:
                    # Central Triage Desk (Obstacle 5: [-0.75, 0.75] x [0.8, 1.6])
                    cv2.fillPoly(img, [pts + np.array([4, 4], dtype=np.int32)], (190, 202, 212))
                    cv2.fillPoly(img, [pts], (165, 180, 195))
                    cv2.polylines(img, [pts], isClosed=True, color=(90, 108, 126), thickness=2)
                    # Countertop bevel
                    cv2.rectangle(img, (x_min + 4, y_min + 4), (x_max - 4, y_max - 4), (195, 210, 225), -1)
                    cv2.rectangle(img, (x_min + 4, y_min + 4), (x_max - 4, y_max - 4), (140, 155, 172), 1)

                    # Dual diagnostic telemetry monitors
                    m1_cx = int(x_min + (x_max - x_min) * 0.30)
                    m2_cx = int(x_min + (x_max - x_min) * 0.70)
                    m_cy = int((y_min + y_max) / 2)
                    for mc in [m1_cx, m2_cx]:
                        cv2.rectangle(img, (mc - 10, m_cy - 7), (mc + 10, m_cy + 7), (25, 30, 35), -1)
                        cv2.rectangle(img, (mc - 8, m_cy - 5), (mc + 8, m_cy + 5), (15, 25, 20), -1)
                        # ECG green pulse on screen
                        cv2.line(img, (mc - 7, m_cy), (mc - 3, m_cy), (40, 220, 80), 1)
                        cv2.line(img, (mc - 3, m_cy), (mc - 1, m_cy - 4), (40, 220, 80), 1)
                        cv2.line(img, (mc - 1, m_cy - 4), (mc + 1, m_cy + 4), (40, 220, 80), 1)
                        cv2.line(img, (mc + 1, m_cy + 4), (mc + 3, m_cy), (40, 220, 80), 1)
                        cv2.line(img, (mc + 3, m_cy), (mc + 7, m_cy), (40, 220, 80), 1)

                    c_x = int(np.mean(pts[:, 0]))
                    cv2.putText(img, "TRIAGE & DIAGNOSTICS", (c_x - 68, y_min - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (30, 45, 60), 1, cv2.LINE_AA)
                    cv2.putText(img, "BOTTLENECK CHOKE-POINT", (c_x - 65, y_max + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (20, 100, 200), 1, cv2.LINE_AA)

        # 5. Four Recessed Patient Room Alcoves & Yield Refuges
        # North Alcove 1: [-4.0, 0.0] x [1.2, 3.8] (Primary Yield Refuge)
        self._draw_hospital_room_alcove(img, -2.4, 2.7, "ROOM 101: POST-OP RECOVERY", is_yield_zone=True, is_north=True)
        # North Alcove 2: [4.0, 11.5] x [1.2, 3.8]
        self._draw_hospital_room_alcove(img, 6.0, 2.7, "ROOM 102: CARDIOLOGY WING", is_yield_zone=False, is_north=True)
        # South Alcove 1: [-11.5, -4.0] x [-3.8, -1.2]
        self._draw_hospital_room_alcove(img, -7.5, -2.7, "ROOM 103: PEDIATRICS WING", is_yield_zone=False, is_north=False)
        # South Alcove 2: [0.0, 4.0] x [-3.8, -1.2] (Secondary Yield Refuge)
        self._draw_hospital_room_alcove(img, 2.0, -2.7, "ROOM 104: TRAUMA CARE", is_yield_zone=True, is_north=False)

        # 6. West Wing Central Pharmacy Station (x = -10.5, y = 0.0)
        p_px, p_py = self._w2px(-10.5, 0.0)
        cv2.rectangle(img, (p_px - 22, p_py - 30), (p_px + 22, p_py + 30), (220, 230, 238), -1)
        cv2.rectangle(img, (p_px - 22, p_py - 30), (p_px + 22, p_py + 30), (40, 120, 180), 2)
        # Medical Red Cross on Pharmacy Counter
        cv2.circle(img, (p_px, p_py - 8), 12, (255, 255, 255), -1)
        cv2.circle(img, (p_px, p_py - 8), 12, (30, 40, 220), 1)
        cv2.rectangle(img, (p_px - 3, p_py - 16), (p_px + 3, p_py), (30, 40, 220), -1)
        cv2.rectangle(img, (p_px - 8, p_py - 11), (p_px + 8, p_py - 5), (30, 40, 220), -1)
        cv2.putText(img, "PHARMACY", (p_px - 25, p_py + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (20, 40, 70), 1, cv2.LINE_AA)
        cv2.putText(img, "DISPATCH", (p_px - 23, p_py + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (20, 40, 70), 1, cv2.LINE_AA)

        # 7. East Wing Intensive Care Unit (ICU) Nurse Station (x = 10.5, y = 0.0)
        icu_px, icu_py = self._w2px(10.5, 0.0)
        cv2.rectangle(img, (icu_px - 24, icu_py - 32), (icu_px + 24, icu_py + 32), (218, 230, 240), -1)
        cv2.rectangle(img, (icu_px - 24, icu_py - 32), (icu_px + 24, icu_py + 32), (25, 95, 175), 2)
        cv2.circle(img, (icu_px, icu_py - 8), 12, (255, 255, 255), -1)
        cv2.circle(img, (icu_px, icu_py - 8), 12, (20, 150, 40), 1)
        cv2.rectangle(img, (icu_px - 3, icu_py - 16), (icu_px + 3, icu_py), (20, 150, 40), -1)
        cv2.rectangle(img, (icu_px - 8, icu_py - 11), (icu_px + 8, icu_py - 5), (20, 150, 40), -1)
        cv2.putText(img, "ICU NURSE", (icu_px - 25, icu_py + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (15, 35, 65), 1, cv2.LINE_AA)
        cv2.putText(img, "STATION", (icu_px - 21, icu_py + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (15, 35, 65), 1, cv2.LINE_AA)

        # 8. Overhead Architectural Section Signage
        sections = [
            ("WEST WING: PHARMACY DISPATCH", (-10.5, 3.65)),
            ("CENTRAL CORRIDOR: 4.0m CLINICAL HALLWAY", (-2.0, 3.65)),
            ("EAST WING: INTENSIVE CARE UNIT", (7.0, 3.65)),
        ]
        for name, (wx, wy) in sections:
            px, py = self._w2px(wx, wy)
            cv2.putText(img, name, (px - 60, py), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 115, 130), 1, cv2.LINE_AA)

        # Bottleneck South Bypass lane indicator: y = -0.60m
        bp_px, bp_py = self._w2px(0.0, -0.60)
        cv2.circle(img, (bp_px, bp_py), 4, (40, 180, 80), -1)
        cv2.putText(img, "Nav Lane: y = -0.60m (Clearance: 1.1m)", (bp_px - 95, bp_py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.27, (30, 130, 60), 1, cv2.LINE_AA)

        # 9. A* Path Planning Visualization Overlay
        if self.view_mode in ["PLANNING", "NAVIGATING"] and self.planner_data is not None:
            self._draw_a_star_overlay(img)

        # 10. Start & Goal Visual Markers
        if self.cfg is not None:
            s_px, s_py = self._w2px(self.cfg.robot_start[0], self.cfg.robot_start[1])
            cv2.circle(img, (s_px, s_py), 16, (60, 180, 75), 2, cv2.LINE_AA)
            cv2.circle(img, (s_px, s_py), 8, (60, 180, 75), -1, cv2.LINE_AA)
            cv2.putText(img, "START: Pharmacy", (s_px - 48, s_py - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (30, 120, 40), 1, cv2.LINE_AA)

            target_goal = self.selected_goal if self.selected_goal is not None else self.cfg.robot_goal
            g_px, g_py = self._w2px(target_goal[0], target_goal[1])
            cv2.circle(img, (g_px, g_py), 18, (0, 190, 250), 2, cv2.LINE_AA)
            cv2.circle(img, (g_px, g_py), 10, (0, 190, 250), -1, cv2.LINE_AA)
            g_lbl = "GOAL: " + (self.selected_goal_label if self.selected_goal_label else "ICU Nurse Station")
            cv2.putText(img, g_lbl, (g_px - 65, g_py - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 140, 200), 2, cv2.LINE_AA)

        # 11. Visited Trajectory Trail
        if len(self.trajectory) > 1:
            for i in range(len(self.trajectory) - 1):
                p1 = self._w2px(self.trajectory[i][0], self.trajectory[i][1])
                p2 = self._w2px(self.trajectory[i + 1][0], self.trajectory[i + 1][1])
                cv2.line(img, p1, p2, (230, 160, 40), 2, cv2.LINE_AA)

        # 12. Dynamic Pedestrians with Hospital Clinical Roles
        if self.env is not None and hasattr(self.env, "soadrl_sim"):
            self._draw_hospital_pedestrians(img)

        # 13. MediRunner Autonomous Mobile Robot Avatar
        if self.env is not None and hasattr(self.env, "soadrl_sim"):
            self._draw_hospital_robot(img)

        # 14. Interactive Goal Selection Overlay
        if self.is_selecting_goal:
            cv2.rectangle(img, (20, 16), (620, 50), (25, 28, 36), -1)
            cv2.rectangle(img, (20, 16), (620, 50), (0, 190, 255), 2)
            cv2.putText(
                img,
                "[HOSPITAL MODE] CLICK REACHABLE CORRIDOR / ALCOVE FOR MEDICINE DELIVERY",
                (30, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.33,
                (0, 230, 255),
                1,
                cv2.LINE_AA
            )
            if self.mouse_hover_coords is not None:
                hx_px, hy_px = self._w2px(self.mouse_hover_coords[0], self.mouse_hover_coords[1])
                if 0 <= hx_px < self.canvas_width and 0 <= hy_px < self.canvas_height:
                    cv2.circle(img, (hx_px, hy_px), 8, (0, 220, 255), 1, cv2.LINE_AA)
                    cv2.drawMarker(img, (hx_px, hy_px), (0, 220, 255), cv2.MARKER_CROSS, 14, 1)

        # 15. Hospital HUD Legend
        self._draw_hospital_hud_legend(img)

        # Render on Canvas
        self._last_bgr_img = img.copy()
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        self.canvas_photo = ImageTk.PhotoImage(image=pil_img)
        self.canvas.create_image(0, 0, image=self.canvas_photo, anchor="nw")

    def _draw_hospital_room_alcove(self, img, cx, cy, label, is_yield_zone=False, is_north=True):
        """Draws a recessed patient room alcove with bed, telemetry, and yield zone markings."""
        p_cx, p_cy = self._w2px(cx, cy)
        bw, bh = 48, 30
        p_tl = (p_cx - bw // 2, p_cy - bh // 2)
        p_br = (p_cx + bw // 2, p_cy + bh // 2)

        # 1. Yield zone safety hatching
        if is_yield_zone:
            yz_pad = 12
            yz_tl = (p_tl[0] - yz_pad, p_tl[1] - yz_pad)
            yz_br = (p_br[0] + yz_pad, p_br[1] + yz_pad)
            cv2.rectangle(img, yz_tl, yz_br, (230, 245, 235), -1)
            cv2.rectangle(img, yz_tl, yz_br, (60, 180, 100), 1)
            cv2.putText(img, "[+] ALCOVE YIELD REFUGE", (yz_tl[0] + 4, yz_tl[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (20, 120, 50), 1, cv2.LINE_AA)

        # 2. Hospital Bed
        cv2.rectangle(img, (p_tl[0] + 3, p_tl[1] + 3), (p_br[0] + 3, p_br[1] + 3), (205, 215, 222), -1)
        cv2.rectangle(img, p_tl, p_br, (135, 145, 155), -1)
        cv2.rectangle(img, p_tl, p_br, (80, 90, 100), 1)

        # Mattress & Linen
        cv2.rectangle(img, (p_tl[0] + 4, p_tl[1] + 3), (p_br[0] - 4, p_br[1] - 3), (255, 255, 255), -1)
        # Pillow
        if is_north:
            cv2.rectangle(img, (p_tl[0] + 6, p_tl[1] + 4), (p_tl[0] + 16, p_br[1] - 4), (220, 235, 245), -1)
            cv2.rectangle(img, (p_tl[0] + 20, p_tl[1] + 3), (p_br[0] - 4, p_br[1] - 3), (210, 170, 70), -1)
        else:
            cv2.rectangle(img, (p_br[0] - 16, p_tl[1] + 4), (p_br[0] - 6, p_br[1] - 4), (220, 235, 245), -1)
            cv2.rectangle(img, (p_tl[0] + 4, p_tl[1] + 3), (p_br[0] - 20, p_br[1] - 3), (210, 170, 70), -1)

        # 3. Bedside vitals monitor
        vm_x = p_br[0] + 6
        vm_y = p_cy - 6
        cv2.rectangle(img, (vm_x - 6, vm_y - 6), (vm_x + 6, vm_y + 6), (25, 30, 35), -1)
        cv2.line(img, (vm_x - 4, vm_y), (vm_x + 4, vm_y), (40, 220, 80), 1)

        # 4. Room Placard
        lbl_y = p_tl[1] - 4 if is_north else p_br[1] + 12
        cv2.putText(img, label, (p_cx - 45, lbl_y), cv2.FONT_HERSHEY_SIMPLEX, 0.27, (40, 60, 80), 1, cv2.LINE_AA)

    def _draw_hospital_robot(self, img):
        """Draws the MediRunner AMR with clean white livery, medical cross, and safety halo."""
        if self.env is None or not hasattr(self.env, "soadrl_sim"):
            return
        robot = self.env.soadrl_sim.robot
        rx, ry = self._w2px(robot.px, robot.py)
        rr = max(10, int(robot.radius * self.scale_px))
        theta = robot.theta

        # Drive wheels
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        nx = -sin_t
        ny = cos_t
        w1_x = int(rx + nx * (rr + 2))
        w1_y = int(ry - ny * (rr + 2))
        w2_x = int(rx - nx * (rr + 2))
        w2_y = int(ry + ny * (rr + 2))
        cv2.circle(img, (w1_x, w1_y), 3, (25, 25, 25), -1)
        cv2.circle(img, (w2_x, w2_y), 3, (25, 25, 25), -1)

        # Dynamic Safety Halo depending on controller state
        c_state = getattr(self.ctrl, 'state', 'NORMAL')
        halo_color = (60, 220, 80) # Default cruising green
        if c_state in ["ALCOVE_YIELDING", "WAITING_IN_ALCOVE"]:
            halo_color = (0, 190, 255) # Warning Amber in alcove yield
        elif getattr(self.ctrl, 'stopped_steps', 0) > 0:
            halo_color = (30, 30, 240) # Emergency Stop Red
        cv2.circle(img, (rx, ry), rr + 6, halo_color, 1, cv2.LINE_AA)

        # Robot chassis: clinical pearl white
        cv2.circle(img, (rx, ry), rr, (250, 252, 255), -1)
        cv2.circle(img, (rx, ry), rr, (200, 150, 0), 2) # Cyan bumper

        # Medical Red Cross on top deck
        cr_w = max(2, rr // 4)
        cr_len = max(5, int(rr * 0.60))
        cv2.rectangle(img, (rx - cr_w, ry - cr_len), (rx + cr_w, ry + cr_len), (35, 35, 225), -1)
        cv2.rectangle(img, (rx - cr_len, ry - cr_w), (rx + cr_len, ry + cr_w), (35, 35, 225), -1)

        # Digital medicine vault temperature indicator (+4°C LED)
        cv2.circle(img, (rx - 4, ry - rr + 4), 2, (40, 220, 60), -1)

        # Top LiDAR turret
        head_x = int(rx + cos_t * (rr - 2))
        head_y = int(ry - sin_t * (rr - 2))
        cv2.circle(img, (head_x, head_y), 3, (35, 45, 55), -1)

        # Heading orientation vector
        ray_x = int(rx + cos_t * (rr + 12))
        ray_y = int(ry - sin_t * (rr + 12))
        cv2.line(img, (rx, ry), (ray_x, ray_y), (255, 255, 255), 2, cv2.LINE_AA)

        # State label
        lbl_str = "AMR: MediRunner"
        if c_state in ["ALCOVE_YIELDING", "WAITING_IN_ALCOVE"]:
            lbl_str = "AMR: Yielding in Alcove"
        cv2.putText(img, lbl_str, (rx - 44, ry - rr - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (20, 40, 60), 1, cv2.LINE_AA)

    def _draw_hospital_pedestrians(self, img):
        """Draws dynamic clinical agents: Emergency Gurney, Nurses, Walker Patients, Visitors."""
        if self.env is None or not hasattr(self.env, "soadrl_sim"):
            return
        sim = self.env.soadrl_sim
        for i, h in enumerate(sim.humans):
            hx, hy = self._w2px(h.px, h.py)
            hr = max(6, int(h.radius * self.scale_px))
            speed = math.hypot(h.vx, h.vy)

            role = 'visitor'
            if self.cfg and hasattr(self.cfg, 'base_humans') and i < len(self.cfg.base_humans):
                role = self.cfg.base_humans[i].get('role', 'visitor')

            if role == 'emergency_gurney':
                # Critical Hazard: High-speed Emergency Gurney (r=0.65m, v=1.65m/s)
                # 1. Pulsating emergency red/amber hazard beacon halo
                cv2.circle(img, (hx, hy), hr + 8, (0, 160, 255), 1, cv2.LINE_AA)
                cv2.circle(img, (hx, hy), hr + 12, (20, 30, 240), 1, cv2.LINE_AA)

                # 2. Gurney metallic stretcher frame
                gw = max(24, int(hr * 1.8))
                gh = max(14, int(hr * 1.1))
                cv2.rectangle(img, (hx - gw // 2 + 2, hy - gh // 2 + 2), (hx + gw // 2 + 2, hy + gh // 2 + 2), (180, 190, 200), -1)
                cv2.rectangle(img, (hx - gw // 2, hy - gh // 2), (hx + gw // 2, hy + gh // 2), (75, 80, 85), -1)
                cv2.rectangle(img, (hx - gw // 2, hy - gh // 2), (hx + gw // 2, hy + gh // 2), (40, 45, 50), 2)

                # 3. Patient lying on emergency mattress
                cv2.rectangle(img, (hx - gw // 2 + 3, hy - gh // 2 + 3), (hx + gw // 2 - 3, hy + gh // 2 - 3), (215, 140, 50), -1)
                cv2.rectangle(img, (hx - gw // 2 + 6, hy - gh // 2 + 3), (hx + gw // 2 - 10, hy + gh // 2 - 3), (250, 252, 255), -1)
                cv2.circle(img, (hx + gw // 2 - 7, hy), 4, (220, 230, 240), -1)

                # 4. Flashing beacon light on top
                cv2.circle(img, (hx, hy), 4, (20, 30, 245), -1)
                cv2.circle(img, (hx, hy), 2, (255, 255, 255), -1)

                cv2.putText(img, "EMERGENCY GURNEY (1.65 m/s)", (hx - 68, hy - hr - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (20, 30, 240), 1, cv2.LINE_AA)
                cv2.putText(img, "[RIGHT OF WAY]", (hx - 36, hy - hr - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 140, 230), 1, cv2.LINE_AA)

            elif role == 'nurse_fast':
                # Purposeful Nurse (Fast counterflow 1.25 m/s)
                cv2.circle(img, (hx, hy), hr, (175, 145, 0), -1)
                cv2.circle(img, (hx, hy), hr, (120, 100, 0), 1)
                cv2.circle(img, (hx, hy - 2), max(2, hr // 3), (255, 255, 255), -1)
                cv2.ellipse(img, (hx, hy + 2), (hr - 2, hr // 2), 0, 0, 180, (240, 240, 240), 1)
                cv2.putText(img, "Nurse (1.2m/s)", (hx - 32, hy - hr - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.29, (140, 110, 0), 1, cv2.LINE_AA)

            elif role == 'patient_walker':
                # Frail Patient with Mobility Walker (r=0.45m, v=0.28m/s)
                cv2.circle(img, (hx, hy), hr, (220, 180, 130), -1)
                cv2.circle(img, (hx, hy), hr, (150, 110, 70), 1)
                wk_w, wk_h = 14, 12
                wk_cx = hx - 12 if h.vx < 0 else hx + 12
                cv2.rectangle(img, (wk_cx - wk_w // 2, hy - wk_h // 2), (wk_cx + wk_w // 2, hy + wk_h // 2), (160, 170, 180), 2)
                cv2.circle(img, (wk_cx - wk_w // 2, hy - wk_h // 2), 2, (30, 30, 30), -1)
                cv2.circle(img, (wk_cx + wk_w // 2, hy - wk_h // 2), 2, (30, 30, 30), -1)
                cv2.circle(img, (wk_cx - wk_w // 2, hy + wk_h // 2), 2, (30, 30, 30), -1)
                cv2.circle(img, (wk_cx + wk_w // 2, hy + wk_h // 2), 2, (30, 30, 30), -1)
                cv2.putText(img, "Patient + Walker", (hx - 38, hy - hr - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (140, 100, 50), 1, cv2.LINE_AA)

            else:
                # Visitor
                cv2.circle(img, (hx, hy), hr, (160, 120, 140), -1)
                cv2.circle(img, (hx, hy), hr, (100, 70, 90), 1)
                cv2.putText(img, "Visitor", (hx - 16, hy - hr - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (80, 50, 70), 1, cv2.LINE_AA)

            # Velocity vector arrow
            if speed > 0.05:
                v_end_x = int(hx + h.vx * self.scale_px * 0.4)
                v_end_y = int(hy - h.vy * self.scale_px * 0.4)
                cv2.arrowedLine(img, (hx, hy), (v_end_x, v_end_y), (255, 255, 255), 1, tipLength=0.3)

    def _draw_hospital_hud_legend(self, img):
        """Draws the bottom floating HUD legend for the Hospital scenario."""
        cv2.rectangle(img, (20, self.canvas_height - 38), (620, self.canvas_height - 12), (25, 28, 34), -1)
        cv2.rectangle(img, (20, self.canvas_height - 38), (620, self.canvas_height - 12), (50, 55, 65), 1)
        cv2.circle(img, (32, self.canvas_height - 25), 6, (250, 252, 255), -1)
        cv2.circle(img, (32, self.canvas_height - 25), 6, (200, 150, 0), 1)
        cv2.putText(img, "MediRunner (AMR)", (44, self.canvas_height - 21), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (220, 225, 230), 1)
        cv2.rectangle(img, (155, self.canvas_height - 30), (170, self.canvas_height - 20), (20, 30, 240), -1)
        cv2.putText(img, "Emergency Gurney (1.65m/s)", (176, self.canvas_height - 21), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (220, 225, 230), 1)
        cv2.circle(img, (340, self.canvas_height - 25), 6, (175, 145, 0), -1)
        cv2.putText(img, "Nurse", (350, self.canvas_height - 21), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (220, 225, 230), 1)
        cv2.circle(img, (400, self.canvas_height - 25), 6, (220, 180, 130), -1)
        cv2.putText(img, "Walker Patient", (410, self.canvas_height - 21), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (220, 225, 230), 1)
        cv2.rectangle(img, (505, self.canvas_height - 29), (517, self.canvas_height - 21), (60, 180, 100), 1)
        cv2.putText(img, "Alcove Refuge", (523, self.canvas_height - 21), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (220, 225, 230), 1)

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
        active_scen = "hotelroomservice" if self.active_scenario == "hotelroomservice" else ("medirunner" if self.active_scenario in ["medirunner", "hospital"] else "library")

        for diff in tiers:
            successes = 0
            collisions = 0
            timeouts = 0
            times = []

            for seed in range(42, 62):
                np.random.seed(seed)
                env = CustomNavRepEncodedEnv(scenario=active_scen, difficulty=diff, gpu=False, silent=True)
                cfg = get_scenario_config(active_scen)
                env.reset(active_scen, diff)
                if active_scen == "medirunner":
                    sync_and_rasterize_hospital_sim(env, cfg)
                    ctrl = HospitalNavigationController(
                        static_obstacles=cfg.static_obstacles,
                        world_bounds=cfg.world_bounds,
                        waypoints=self.active_a_star_waypoints
                    )
                elif active_scen == "hotelroomservice":
                    ctrl = HotelNavigationController(waypoints=self.active_a_star_waypoints)
                else:
                    sync_and_rasterize_sim(env.raw_env, cfg)
                    ctrl = LibraryNavigationController(waypoints=self.active_a_star_waypoints)

                done = False
                step = 0
                event = None
                diff_limit = cfg.get_difficulty_config(diff).get('time_limit', 380)

                while not done and step < diff_limit:
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
                    'text': "Evaluating {} [{}] (Seed {}/61)...".format(active_scen, diff, seed)
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
    import argparse
    parser = argparse.ArgumentParser(description="SVNIT CS359 Multi-Scenario Autonomous Robot Navigation GUI")
    parser.add_argument("--scenario", type=str, default="library", choices=["library", "hotelroomservice", "medirunner", "hospital"],
                        help="Scenario to launch: 'library', 'hotelroomservice', or 'medirunner' / 'hospital'")
    args, _ = parser.parse_known_args()

    scen = args.scenario
    if scen == "hospital":
        scen = "medirunner"

    root = tk.Tk()
    app = LibraryDemoApp(root, default_scenario=scen)
    root.mainloop()


if __name__ == "__main__":
    main()
