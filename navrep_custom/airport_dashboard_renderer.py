#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
airport_dashboard_renderer.py
=============================
Custom high-fidelity dashboard renderer for Airport Passenger Assistance Robot.
Replicates the 1600x900 modern robotics mission-control UI layout:
  - Top header bar with title, subtitle, and dynamic status chips
  - Left panel: High-resolution architectural airport terminal floor plan (24x15m)
    with double outer wall, zone labels, textured static obstacles with label strips,
    dynamic humans with role-based colors/labels and velocity arrows, robot disc,
    heading arrow, cyan path trail, start and goal indicators.
  - Right panel: Modular operations, goal selection, difficulty & sim pacing radios,
    live telemetry & navigation state, navigation policy details (real PPO/controller specs),
    and 20-seed quantitative benchmark results for the active difficulty level.

Optimized with static-layer caching for fast execution on CPU.
"""

from __future__ import print_function
import os
import sys
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Ensure repository root is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


# ==============================================================================
# COLOR PALETTE SPECIFICATIONS
# ==============================================================================
DARK_BG = (11, 15, 21)          # Canvas dark charcoal
HEADER_BG = (10, 14, 20)        # Top bar
BORDER_COLOR = (28, 38, 54)     # Section borders
PANEL_BG = (12, 17, 24)         # Right panel dark navy
BOX_BG = (15, 22, 32)           # Titled box background
BOX_BORDER = (32, 44, 62)       # Titled box border

# Floor Plan Colors
FLOOR_BG = (235, 230, 218)      # Light beige marble floor
FLOOR_GRID = (220, 214, 200)    # Floor grid lines
WALL_OUTER = (61, 39, 29)       # Dark brown outer boundary wall
WALL_INNER = (92, 58, 30)       # Inner wall stroke
ZONE_TEXT = (142, 130, 113)     # Subtle architectural zone labels

# Button and Accent Colors
ACCENT_CYAN = (56, 182, 255)
GREEN_SUCCESS = (46, 204, 113)
RED_COLLISION = (231, 76, 60)
ORANGE_TIMEOUT = (243, 156, 18)
AMBER_WARN = (255, 183, 3)
BLUE_NAV = (30, 136, 229)
GREY_TEXT = (160, 175, 190)
WHITE_TEXT = (255, 255, 255)


# ==============================================================================
# BENCHMARK REFERENCE DATA (20 Episodes per level, Seeds 100-119)
# ==============================================================================
BENCHMARK_STATS = {
    'EASY': {
        'success_rate': 100.0,
        'success_count': 20,
        'collision_rate': 0.0,
        'collision_count': 0,
        'timeout_rate': 0.0,
        'timeout_count': 0,
        'mean_time': '32.76'
    },
    'MEDIUM': {
        'success_rate': 50.0,
        'success_count': 10,
        'collision_rate': 20.0,
        'collision_count': 4,
        'timeout_rate': 30.0,
        'timeout_count': 6,
        'mean_time': '33.46'
    },
    'HARD': {
        'success_rate': 15.0,
        'success_count': 3,
        'collision_rate': 50.0,
        'collision_count': 10,
        'timeout_rate': 35.0,
        'timeout_count': 7,
        'mean_time': '32.87'
    }
}


class AirportDashboardRenderer(object):
    """
    Renders 1600x900 mission-control dashboard frames for Airport Passenger Assistance.
    """

    def __init__(self, scenario_config, difficulty="MEDIUM"):
        self.cfg = scenario_config
        self.difficulty = difficulty.upper()
        self.canvas_w = 1600
        self.canvas_h = 900

        # Load Fonts
        font_paths = [
            "C:/Windows/Fonts/consola.ttf",
            "C:/Windows/Fonts/cour.ttf",
            "C:/Windows/Fonts/lucon.ttf",
            "C:/Windows/Fonts/arial.ttf"
        ]
        chosen_font = None
        chosen_bold = None
        for p in font_paths:
            if os.path.exists(p):
                chosen_font = p
                bold_p = p.replace("consola.ttf", "consolab.ttf").replace("arial.ttf", "arialbd.ttf")
                chosen_bold = bold_p if os.path.exists(bold_p) else p
                break

        if chosen_font is not None:
            self.font_title = ImageFont.truetype(chosen_bold, 20)
            self.font_sub = ImageFont.truetype(chosen_font, 13)
            self.font_chip = ImageFont.truetype(chosen_bold, 13)
            self.font_hdr = ImageFont.truetype(chosen_bold, 14)
            self.font_btn = ImageFont.truetype(chosen_bold, 13)
            self.font_btn_lg = ImageFont.truetype(chosen_bold, 15)
            self.font_body = ImageFont.truetype(chosen_font, 13)
            self.font_body_bold = ImageFont.truetype(chosen_bold, 13)
            self.font_small = ImageFont.truetype(chosen_font, 11)
            self.font_zone = ImageFont.truetype(chosen_bold, 13)
            self.font_obs = ImageFont.truetype(chosen_bold, 11)
        else:
            self.font_title = ImageFont.load_default()
            self.font_sub = ImageFont.load_default()
            self.font_chip = ImageFont.load_default()
            self.font_hdr = ImageFont.load_default()
            self.font_btn = ImageFont.load_default()
            self.font_btn_lg = ImageFont.load_default()
            self.font_body = ImageFont.load_default()
            self.font_body_bold = ImageFont.load_default()
            self.font_small = ImageFont.load_default()
            self.font_zone = ImageFont.load_default()
            self.font_obs = ImageFont.load_default()

        # Floor plan coordinate system
        # World: x in [-12, 12] (24m), y in [-7.5, 7.5] (15m)
        self.scale = 40.0  # 40 pixels per meter
        self.cx = 530      # World origin x on canvas
        self.cy = 490      # World origin y on canvas

        # Pre-render static background
        self.static_bg = self._create_static_background()

    def world_to_canvas(self, wx, wy):
        """Transforms continuous world coordinate (m) to floor plan canvas pixel."""
        px = int(round(self.cx + wx * self.scale))
        py = int(round(self.cy - wy * self.scale))
        return px, py

    def _create_static_background(self):
        """Constructs and caches the static background layer."""
        img = Image.new("RGB", (self.canvas_w, self.canvas_h), DARK_BG)
        draw = ImageDraw.Draw(img)

        # ----------------------------------------------------------------------
        # 1. TOP HEADER BAR (y: 0 to 58)
        # ----------------------------------------------------------------------
        draw.rectangle([(0, 0), (self.canvas_w, 58)], fill=HEADER_BG)
        draw.line([(0, 58), (self.canvas_w, 58)], fill=BORDER_COLOR, width=2)

        # Title and Subtitle
        draw.text((20, 10), "AIRPORT PASSENGER ASSISTANCE ROBOT", font=self.font_title, fill=WHITE_TEXT)
        subtitle = "Task 3: Autonomous Mobile Navigation in an International Airport Terminal | Author: Ritu (SVNIT Surat)"
        draw.text((20, 36), subtitle, font=self.font_sub, fill=ACCENT_CYAN)

        # ----------------------------------------------------------------------
        # 2. LEFT PANEL: FLOOR PLAN CONTAINER
        # ----------------------------------------------------------------------
        panel_left_x0, panel_left_y0 = 16, 64
        panel_left_x1, panel_left_y1 = 1045, 888

        # Header Strip
        draw.rectangle([(panel_left_x0, panel_left_y0), (panel_left_x1, panel_left_y0 + 28)], fill=(20, 27, 36))
        draw.rectangle([(panel_left_x0, panel_left_y0), (panel_left_x1, panel_left_y1)], outline=BORDER_COLOR, width=2)
        fl_title = "AIRPORT TERMINAL FLOOR PLAN (24.0x15.0 m) -- CHECK-IN HALL, SECURITY & GATE AREA"
        draw.text((26, panel_left_y0 + 6), fl_title, font=self.font_hdr, fill=(175, 195, 215))

        # Floor Canvas Rectangle
        # World bounds [-12, 12] x [-7.5, 7.5] -> [50, 1010] x [190, 790]
        fl_x0, fl_y0 = 50, 190
        fl_x1, fl_y1 = 1010, 790

        # Light beige floor fill
        draw.rectangle([(fl_x0, fl_y0), (fl_x1, fl_y1)], fill=FLOOR_BG)

        # Subtle 1m floor tile grid lines
        for wx in np.arange(-12.0, 12.1, 1.0):
            gx, _ = self.world_to_canvas(wx, 0.0)
            if fl_x0 <= gx <= fl_x1:
                draw.line([(gx, fl_y0), (gx, fl_y1)], fill=FLOOR_GRID, width=1)
        for wy in np.arange(-7.5, 7.6, 1.0):
            _, gy = self.world_to_canvas(0.0, wy)
            if fl_y0 <= gy <= fl_y1:
                draw.line([(fl_x0, gy), (fl_x1, gy)], fill=FLOOR_GRID, width=1)

        # Double outer walls
        # Outer boundary (thick dark brown)
        draw.rectangle([(fl_x0 - 4, fl_y0 - 4), (fl_x1 + 4, fl_y1 + 4)], outline=WALL_OUTER, width=6)
        # Inner perimeter stroke
        draw.rectangle([(fl_x0, fl_y0), (fl_x1, fl_y1)], outline=WALL_INNER, width=2)

        # Architectural Zone Labels
        zones = [
            ("CHECK-IN ENTRANCE & HALL", -8.0, 6.5),
            ("SECURITY SCREENING", -3.5, 6.5),
            ("CENTRAL CONCOURSE", 0.5, 6.5),
            ("DEPARTURE LOUNGE", 5.0, 6.5),
            ("GATE 12 BOARDING", 9.6, 6.5),
            ("JETBRIDGE ACCESS", 9.2, -6.5),
            ("DUTY-FREE AISLE", 4.5, -6.5),
            ("CHECK-IN BAGGAGE DROP", -8.0, -6.5),
        ]
        for z_name, zwx, zwy in zones:
            zpx, zpy = self.world_to_canvas(zwx, zwy)
            # Center text approximately
            bbox = self.font_zone.getbbox(z_name)
            w = bbox[2] - bbox[0]
            draw.text((zpx - w // 2, zpy - 7), z_name, font=self.font_zone, fill=ZONE_TEXT)

        # ----------------------------------------------------------------------
        # 3. STATIC OBSTACLES (Furniture, Desks, Kiosks, Pillars, Seating)
        # ----------------------------------------------------------------------
        # Obstacle styling specification
        obstacle_specs = [
            # Check-in A: (-7.5, 3.8, 1.8, 2.6)
            {"cx": -7.5, "cy": 3.8, "w": 1.8, "h": 2.6, "type": "counter",
             "label": "CHECK-IN A", "color": (124, 83, 53), "border": (78, 49, 27)},
            # Check-in B: (-7.5, -4.8, 1.8, 2.2)
            {"cx": -7.5, "cy": -4.8, "w": 1.8, "h": 2.2, "type": "counter",
             "label": "CHECK-IN B", "color": (124, 83, 53), "border": (78, 49, 27)},
            # FIDS Info Kiosk: (-3.5, 4.2, 1.5, 1.5)
            {"cx": -3.5, "cy": 4.2, "w": 1.5, "h": 1.5, "type": "kiosk",
             "label": "FIDS KIOSK", "color": (46, 64, 87), "border": (26, 37, 51)},
            # Security Screening: (-3.5, 0.0, 1.6, 2.4)
            {"cx": -3.5, "cy": 0.0, "w": 1.6, "h": 2.4, "type": "security",
             "label": "SECURITY", "color": (61, 74, 89), "border": (37, 45, 55)},
            # Pillar North: (0.5, 2.8, 1.2, 1.2)
            {"cx": 0.5, "cy": 2.8, "w": 1.2, "h": 1.2, "type": "pillar",
             "letter": "A", "label": "Pillar A", "color": (90, 98, 106), "border": (45, 50, 56)},
            # Pillar South: (0.5, -2.8, 1.2, 1.2)
            {"cx": 0.5, "cy": -2.8, "w": 1.2, "h": 1.2, "type": "pillar",
             "letter": "B", "label": "Pillar B", "color": (90, 98, 106), "border": (45, 50, 56)},
            # Seating North: (5.0, 4.5, 3.0, 1.2)
            {"cx": 5.0, "cy": 4.5, "w": 3.0, "h": 1.2, "type": "seating",
             "label": "LOUNGE NORTH", "color": (139, 90, 43), "border": (89, 58, 28)},
            # Seating South: (5.0, -1.8, 3.0, 1.2)
            {"cx": 5.0, "cy": -1.8, "w": 3.0, "h": 1.2, "type": "seating",
             "label": "LOUNGE SOUTH", "color": (139, 90, 43), "border": (89, 58, 28)},
            # Duty-Free Retail: (7.5, -4.8, 2.2, 2.0)
            {"cx": 7.5, "cy": -4.8, "w": 2.2, "h": 2.0, "type": "retail",
             "label": "DUTY FREE", "color": (125, 90, 68), "border": (77, 55, 42)},
            # Gate 12 Service Desk: (8.5, 5.2, 1.6, 1.2)
            {"cx": 8.5, "cy": 5.2, "w": 1.6, "h": 1.2, "type": "counter",
             "label": "GATE 12", "color": (96, 72, 56), "border": (58, 43, 33)},
        ]

        for obs in obstacle_specs:
            ocx, ocy = obs["cx"], obs["cy"]
            ow, oh = obs["w"], obs["h"]

            # Compute pixel box
            p_left, p_top = self.world_to_canvas(ocx - ow / 2.0, ocy + oh / 2.0)
            p_right, p_bottom = self.world_to_canvas(ocx + ow / 2.0, ocy - oh / 2.0)

            if obs["type"] == "pillar":
                # Draw octagonal / circular pillar disc with letter
                prx = (p_right - p_left) // 2
                pcx = (p_left + p_right) // 2
                pcy = (p_top + p_bottom) // 2

                # Shadow
                draw.ellipse([(pcx - prx - 2, pcy - prx - 2), (pcx + prx + 2, pcy + prx + 2)], fill=(40, 44, 50))
                # Pillar Body
                draw.ellipse([(pcx - prx, pcy - prx), (pcx + prx, pcy + prx)], fill=obs["color"], outline=obs["border"], width=3)
                # Pillar Letter
                draw.text((pcx - 5, pcy - 8), obs["letter"], font=self.font_hdr, fill=WHITE_TEXT)
                # Label below
                lbl = obs["label"]
                lbbox = self.font_obs.getbbox(lbl)
                lw = lbbox[2] - lbbox[0]
                draw.text((pcx - lw // 2, pcy + prx + 3), lbl, font=self.font_obs, fill=(80, 75, 70))
            else:
                # Rectangular block with shadow
                draw.rectangle([(p_left + 3, p_top + 3), (p_right + 3, p_bottom + 3)], fill=(180, 172, 160))
                draw.rectangle([(p_left, p_top), (p_right, p_bottom)], fill=obs["color"], outline=obs["border"], width=3)

                # Inner striped pattern for visual richness (like books/desks in reference)
                for stripe_x in range(p_left + 8, p_right - 8, 14):
                    draw.line([(stripe_x, p_top + 4), (stripe_x, p_bottom - 4)], fill=(obs["color"][0] - 20, obs["color"][1] - 20, obs["color"][2] - 20), width=3)

                # Center dark label strip
                strip_h = 18
                mid_y = (p_top + p_bottom) // 2
                draw.rectangle([(p_left + 4, mid_y - strip_h // 2), (p_right - 4, mid_y + strip_h // 2)], fill=(15, 18, 22))
                draw.rectangle([(p_left + 4, mid_y - strip_h // 2), (p_right - 4, mid_y + strip_h // 2)], outline=obs["border"], width=1)

                lbl = obs["label"]
                lbbox = self.font_obs.getbbox(lbl)
                lw = lbbox[2] - lbbox[0]
                lx = max(p_left + 6, (p_left + p_right - lw) // 2)
                draw.text((lx, mid_y - 6), lbl, font=self.font_obs, fill=(255, 230, 110))

        # Static Start and Goal Annotations
        sx, sy = self.cfg.robot_start[0], self.cfg.robot_start[1]
        spx, spy = self.world_to_canvas(sx, sy)
        # Start Green Marker
        draw.ellipse([(spx - 12, spy - 12), (spx + 12, spy + 12)], outline=GREEN_SUCCESS, width=3)
        draw.text((spx - 45, spy - 28), "START: Check-in Entrance", font=self.font_btn, fill=(35, 155, 86))

        # Goal Gold Marker
        gx, gy = self.cfg.robot_goal[0], self.cfg.robot_goal[1]
        gpx, gpy = self.world_to_canvas(gx, gy)
        # Double gold goal ring
        draw.ellipse([(gpx - 22, gpy - 22), (gpx + 22, gpy + 22)], outline=(255, 215, 0), width=3)
        draw.ellipse([(gpx - 9, gpy - 9), (gpx + 9, gpy + 9)], fill=(255, 190, 0))
        # Place label cleanly below goal ring so it never collides with upper jetbridge pedestrians
        draw.text((gpx - 85, gpy + 26), "GOAL: Gate 12 (Boarding)", font=self.font_btn, fill=(230, 110, 0))

        # ----------------------------------------------------------------------
        # 4. RIGHT PANEL CONTAINER & BOX STRUCTURES (x: 1060 to 1584)
        # ----------------------------------------------------------------------
        panel_r_x0, panel_r_y0 = 1060, 64
        panel_r_x1, panel_r_y1 = 1584, 888
        draw.rectangle([(panel_r_x0, panel_r_y0), (panel_r_x1, panel_r_y1)], fill=PANEL_BG, outline=BORDER_COLOR, width=2)

        bx0, bx1 = 1072, 1572

        # BOX 1: Showcase Demonstration
        self._draw_titled_box(draw, bx0, 74, bx1, 142, "Showcase Demonstration")
        # Big Green Run Button
        draw.rounded_rectangle([(1084, 88), (1560, 120)], radius=4, fill=(40, 167, 69), outline=(52, 208, 88), width=1)
        btn_txt = "[>] RUN FULL DEMONSTRATION"
        t_bb = self.font_btn_lg.getbbox(btn_txt)
        draw.text(((1084 + 1560 - (t_bb[2] - t_bb[0])) // 2, 94), btn_txt, font=self.font_btn_lg, fill=WHITE_TEXT)
        draw.text((1086, 125), "sequence: View Terminal -> Navigate -> Yield & Reach Gate", font=self.font_small, fill=(130, 145, 165))

        # BOX 2: Modular Operations
        self._draw_titled_box(draw, bx0, 152, bx1, 226, "Modular Operations")
        # Row 1 Buttons
        draw.rounded_rectangle([(1084, 164), (1316, 190)], radius=3, fill=(24, 34, 48), outline=(42, 58, 78), width=1)
        draw.text((1140, 169), "Show Terminal", font=self.font_btn, fill=(160, 180, 205))

        draw.rounded_rectangle([(1326, 164), (1560, 190)], radius=3, fill=(13, 110, 253), outline=(56, 139, 253), width=1)
        draw.text((1375, 169), "Run Path Planning", font=self.font_btn, fill=WHITE_TEXT)

        # Row 2 Buttons (Static bases)
        draw.rounded_rectangle([(1084, 195), (1316, 221)], radius=3, fill=(25, 135, 84), outline=(32, 201, 151), width=1)
        draw.text((1140, 200), "Start Navigation", font=self.font_btn, fill=WHITE_TEXT)

        draw.rounded_rectangle([(1326, 195), (1438, 221)], radius=3, fill=(20, 26, 36), outline=(32, 40, 54), width=1)
        draw.text((1364, 200), "Pause", font=self.font_btn, fill=(90, 105, 120))

        draw.rounded_rectangle([(1448, 195), (1560, 221)], radius=3, fill=(26, 36, 50), outline=(43, 59, 80), width=1)
        draw.text((1486, 200), "Reset", font=self.font_btn, fill=(160, 180, 205))

        # BOX 3: Interactive Goal Selection & Replanning
        self._draw_titled_box(draw, bx0, 236, bx1, 328, "Interactive Goal Selection & Replanning")
        draw.rounded_rectangle([(1084, 248), (1334, 274)], radius=3, fill=(10, 54, 65), outline=(0, 168, 204), width=1)
        draw.text((1122, 253), "[+] SELECT NEW GOAL", font=self.font_btn, fill=(0, 210, 252))

        draw.rounded_rectangle([(1344, 248), (1560, 274)], radius=3, fill=(189, 58, 0), outline=(230, 81, 0), width=1)
        draw.text((1395, 253), "REPLAN ROUTE", font=self.font_btn, fill=(255, 175, 130))

        # Preset selection field
        draw.rounded_rectangle([(1084, 280), (1404, 304)], radius=3, fill=(18, 24, 34), outline=(34, 48, 66), width=1)
        draw.text((1094, 284), "Preset: Gate 12 - Boarding [v]", font=self.font_sub, fill=(210, 225, 240))

        draw.rounded_rectangle([(1414, 280), (1560, 304)], radius=3, fill=(22, 31, 43), outline=(37, 52, 72), width=1)
        draw.text((1432, 284), "Reset Original", font=self.font_sub, fill=(145, 168, 192))
        draw.text((1086, 309), "Route active: 20.69m. Click 'Start Navigation' or 'Resume'.", font=self.font_small, fill=ACCENT_CYAN)

        # BOX 4: Difficulty & Sim Pacing Radios
        self._draw_titled_box(draw, bx0, 338, bx1, 408, "Difficulty & Pacing")
        # Sub-headers
        draw.text((1086, 350), "Difficulty Level:", font=self.font_sub, fill=(160, 175, 190))
        draw.text((1340, 350), "Sim Pacing (Playback):", font=self.font_sub, fill=(160, 175, 190))

        # BOX 5: Live Telemetry & Navigation State
        self._draw_titled_box(draw, bx0, 418, bx1, 562, "Live Telemetry & Navigation State")

        # BOX 6: Navigation Policy Details (Replaces A* search box)
        self._draw_titled_box(draw, bx0, 572, bx1, 716, "Navigation Policy Details")

        # BOX 7: 20-Seed Quantitative Benchmark
        self._draw_titled_box(draw, bx0, 726, bx1, 874, "20-Seed Quantitative Benchmark")

        return img

    def _draw_titled_box(self, draw, x0, y0, x1, y1, title):
        """Draws a navy container box with a neat title label tab."""
        draw.rectangle([(x0, y0), (x1, y1)], fill=BOX_BG, outline=BOX_BORDER, width=1)
        # Title tab notch
        t_bb = self.font_hdr.getbbox(title)
        tw = t_bb[2] - t_bb[0]
        draw.rectangle([(x0 + 12, y0 - 8), (x0 + 16 + tw + 6, y0 + 6)], fill=PANEL_BG)
        draw.text((x0 + 15, y0 - 8), title, font=self.font_hdr, fill=(135, 155, 180))

    def render_frame(self,
                     step_num,
                     sim_time,
                     max_steps,
                     time_limit,
                     robot_pos,
                     robot_vel,
                     robot_trail,
                     humans,
                     goal_pos,
                     status,
                     action,
                     difficulty):
        """
        Renders a dynamic frame onto the cached static layer and returns RGB numpy array.
        """
        # Create a copy of the pre-rendered static layer
        frame = self.static_bg.copy()
        draw = ImageDraw.Draw(frame)

        diff = difficulty.upper()
        rx, ry, rth = robot_pos[0], robot_pos[1], robot_pos[2]
        vx, vy = robot_vel[0], robot_vel[1]
        spd = math.hypot(vx, vy)
        gx, gy = goal_pos[0], goal_pos[1]
        dist_to_goal = math.hypot(gx - rx, gy - ry)
        dx_goal, dy_goal = gx - rx, gy - ry

        # Find nearest human distance
        min_human_dist = float('inf')
        for h in humans:
            d = math.hypot(h["pos"][0] - rx, h["pos"][1] - ry)
            if d < min_human_dist:
                min_human_dist = d

        # Compute Controller State from real telemetry rules:
        # 1. READY (intro frames)
        # 2. CRUISING when speed >= 0.25 m/s and min_human_dist > 2.0 m
        # 3. YIELDING when speed < 0.25 m/s or min_human_dist <= 2.0 m
        # 4. Terminal outcome states: GOAL REACHED / COLLISION / TIMEOUT
        if status == "GOAL REACHED":
            ctrl_state = "GOAL REACHED (Docked at Gate 12)"
            ctrl_color = GREEN_SUCCESS
        elif status == "COLLISION":
            ctrl_state = "COLLISION (Obstacle / Pedestrian Hazard)"
            ctrl_color = RED_COLLISION
        elif status == "TIMEOUT":
            ctrl_state = "TIMEOUT (Time Limit Reached)"
            ctrl_color = ORANGE_TIMEOUT
        elif status == "READY":
            ctrl_state = "READY (System Initialized)"
            ctrl_color = (180, 195, 210)
        elif spd < 0.25 or min_human_dist <= 2.0:
            ctrl_state = "YIELDING (Pedestrian Collision Avoidance)"
            ctrl_color = AMBER_WARN
        else:
            ctrl_state = "CRUISING (Terminal Concourse Corridor)"
            ctrl_color = ACCENT_CYAN

        # ----------------------------------------------------------------------
        # A. TOP HEADER STATUS CHIPS
        # ----------------------------------------------------------------------
        # Chip 1: MODE: AIRPORT VIEW
        draw.rounded_rectangle([(930, 12), (1110, 46)], radius=4, fill=(14, 35, 51), outline=(0, 180, 216), width=2)
        draw.text((955, 21), "MODE: AIRPORT VIEW", font=self.font_chip, fill=(0, 229, 255))

        # Chip 2: DIFFICULTY: EASY / MEDIUM / HARD
        if diff == "EASY":
            draw.rounded_rectangle([(1125, 12), (1305, 46)], radius=4, fill=(18, 51, 38), outline=GREEN_SUCCESS, width=2)
            draw.text((1148, 21), "DIFFICULTY: EASY", font=self.font_chip, fill=GREEN_SUCCESS)
        elif diff == "MEDIUM":
            draw.rounded_rectangle([(1125, 12), (1305, 46)], radius=4, fill=(59, 40, 16), outline=ORANGE_TIMEOUT, width=2)
            draw.text((1140, 21), "DIFFICULTY: MEDIUM", font=self.font_chip, fill=AMBER_WARN)
        else:
            draw.rounded_rectangle([(1125, 12), (1305, 46)], radius=4, fill=(59, 24, 24), outline=RED_COLLISION, width=2)
            draw.text((1148, 21), "DIFFICULTY: HARD", font=self.font_chip, fill=(255, 107, 107))

        # Chip 3: STATUS CHIP
        status_chip_rect = [(1320, 12), (1580, 46)]
        if status == "READY":
            draw.rounded_rectangle(status_chip_rect, radius=4, fill=(34, 42, 54), outline=(85, 102, 119), width=2)
            draw.text((1380, 21), "STATUS: READY", font=self.font_chip, fill=(180, 195, 210))
        elif status == "NAVIGATING":
            draw.rounded_rectangle(status_chip_rect, radius=4, fill=(16, 43, 72), outline=(41, 128, 185), width=2)
            draw.text((1360, 21), "STATUS: NAVIGATING", font=self.font_chip, fill=ACCENT_CYAN)
        elif status == "GOAL REACHED":
            draw.rounded_rectangle(status_chip_rect, radius=4, fill=(18, 56, 36), outline=GREEN_SUCCESS, width=2)
            draw.text((1350, 21), "STATUS: GOAL REACHED", font=self.font_chip, fill=GREEN_SUCCESS)
        elif status == "COLLISION":
            draw.rounded_rectangle(status_chip_rect, radius=4, fill=(61, 20, 20), outline=RED_COLLISION, width=2)
            draw.text((1365, 21), "STATUS: COLLISION", font=self.font_chip, fill=(255, 107, 107))
        elif status == "TIMEOUT":
            draw.rounded_rectangle(status_chip_rect, radius=4, fill=(61, 38, 13), outline=ORANGE_TIMEOUT, width=2)
            draw.text((1375, 21), "STATUS: TIMEOUT", font=self.font_chip, fill=AMBER_WARN)

        # ----------------------------------------------------------------------
        # B. DYNAMIC FLOOR PLAN ELEMENTS
        # ----------------------------------------------------------------------
        # 1. Cyan Trajectory Trail
        if len(robot_trail) >= 2:
            pts = [self.world_to_canvas(p[0], p[1]) for p in robot_trail]
            for i in range(len(pts) - 1):
                draw.line([pts[i], pts[i + 1]], fill=(0, 245, 212), width=3)

        # 2. Dynamic Humans
        role_palette = {
            "passenger_luggage_trolley": ((155, 93, 229), "Trolley"),
            "passenger_hurried": ((0, 187, 249), "Hurried"),
            "family_parent": ((241, 91, 181), "Family"),
            "family_child": ((255, 112, 166), "Child"),
            "staff_crossing": ((251, 86, 7), "Staff"),
            "extra_crowd": ((46, 196, 182), "Traveler")
        }

        for idx, h in enumerate(humans, start=1):
            hpx, hpy = self.world_to_canvas(h["pos"][0], h["pos"][1])
            hr = max(int(h["radius"] * self.scale), 8)
            role = h.get("role", "extra_crowd")
            color, role_name = role_palette.get(role, ((46, 196, 182), "Traveler"))

            # Shadow
            draw.ellipse([(hpx - hr - 1, hpy - hr - 1), (hpx + hr + 1, hpy + hr + 1)], fill=(160, 150, 140))
            # Human circle
            draw.ellipse([(hpx - hr, hpy - hr), (hpx + hr, hpy + hr)], fill=color, outline=WHITE_TEXT, width=2)
            draw.ellipse([(hpx - 3, hpy - 3), (hpx + 3, hpy + 3)], fill=WHITE_TEXT)

            # Velocity Arrow
            hth = h["pos"][2]
            v_len = hr + 12
            vex = int(hpx + v_len * math.cos(hth))
            vey = int(hpy - v_len * math.sin(hth))
            draw.line([(hpx, hpy), (vex, vey)], fill=(255, 230, 80), width=2)

            # Label above
            lbl = "P{}: {}".format(idx, role_name)
            lbb = self.font_obs.getbbox(lbl)
            lw = lbb[2] - lbb[0]
            draw.text((hpx - lw // 2, hpy - hr - 15), lbl, font=self.font_obs, fill=(50, 45, 40))

        # 3. Dynamic Robot
        rpx, rpy = self.world_to_canvas(rx, ry)
        rr = int(self.cfg.robot_radius * self.scale) if hasattr(self.cfg, 'robot_radius') else 12
        # Shadow
        draw.ellipse([(rpx - rr - 2, rpy - rr - 2), (rpx + rr + 2, rpy + rr + 2)], fill=(130, 120, 110))
        # Blue Robot Disc
        draw.ellipse([(rpx - rr, rpy - rr), (rpx + rr, rpy + rr)], fill=(0, 120, 255), outline=WHITE_TEXT, width=3)
        draw.ellipse([(rpx - 4, rpy - 4), (rpx + 4, rpy + 4)], fill=(0, 229, 255))

        # Heading vector arrow
        arr_len = rr + 14
        rex = int(rpx + arr_len * math.cos(rth))
        rey = int(rpy - arr_len * math.sin(rth))
        draw.line([(rpx, rpy), (rex, rey)], fill=WHITE_TEXT, width=3)

        # ----------------------------------------------------------------------
        # C. DYNAMIC RIGHT PANEL ELEMENTS
        # ----------------------------------------------------------------------
        # BOX 4: Radios
        # Difficulty radios (x: 1086, y: 375)
        levels = ["EASY", "MEDIUM", "HARD"]
        rad_x = 1086
        for lvl in levels:
            is_active = (lvl == diff)
            if is_active:
                draw.ellipse([(rad_x, 376), (rad_x + 10, 386)], outline=ACCENT_CYAN, width=2)
                draw.ellipse([(rad_x + 3, 379), (rad_x + 7, 383)], fill=ACCENT_CYAN)
                draw.text((rad_x + 14, 374), lvl, font=self.font_body_bold, fill=WHITE_TEXT)
            else:
                draw.ellipse([(rad_x, 376), (rad_x + 10, 386)], outline=(90, 105, 120), width=1)
                draw.text((rad_x + 14, 374), lvl, font=self.font_body, fill=(120, 135, 150))
            rad_x += 75

        # Sim Pacing radios (x: 1340, y: 375)
        pacings = [("0.5x", False), ("1.0x", False), ("2.0x (Fast)", True)]
        p_x = 1340
        for p_lbl, p_active in pacings:
            if p_active:
                draw.ellipse([(p_x, 376), (p_x + 10, 386)], outline=ACCENT_CYAN, width=2)
                draw.ellipse([(p_x + 3, 379), (p_x + 7, 383)], fill=ACCENT_CYAN)
                draw.text((p_x + 14, 374), p_lbl, font=self.font_body_bold, fill=ACCENT_CYAN)
            else:
                draw.ellipse([(p_x, 376), (p_x + 10, 386)], outline=(90, 105, 120), width=1)
                draw.text((p_x + 14, 374), p_lbl, font=self.font_body, fill=(120, 135, 150))
            p_x += 72

        # BOX 5: Live Telemetry & Navigation State
        t_y = 435
        th_deg = math.degrees(rth)
        draw.text((1086, t_y), "Step: {:03d} / {:03d}  |  Sim Time: {:5.1f}s / {:4.1f}s".format(
            step_num, max_steps, sim_time, time_limit), font=self.font_body, fill=(225, 235, 245))
        draw.text((1086, t_y + 20), "Pose: ({:+5.2f}m, {:+5.2f}m)  |  th = {:+5.1f} deg".format(
            rx, ry, th_deg), font=self.font_body, fill=(225, 235, 245))
        draw.text((1086, t_y + 40), "Speed: |v| = {:4.2f} m/s  (vx={:+4.2f}, vy={:+4.2f})".format(
            spd, vx, vy), font=self.font_body, fill=(225, 235, 245))
        draw.text((1086, t_y + 60), "Dist to Goal: {:4.2f}m  (dx={:+4.2f}, dy={:+4.2f})".format(
            dist_to_goal, dx_goal, dy_goal), font=self.font_body, fill=(225, 235, 245))
        draw.text((1086, t_y + 80), "Goal Target: Gate 12 (+9.80m, +2.80m)", font=self.font_body_bold, fill=(255, 209, 102))
        draw.text((1086, t_y + 102), "Controller State: {}".format(ctrl_state), font=self.font_body_bold, fill=ctrl_color)

        # BOX 6: Navigation Policy Details
        p_y = 588
        draw.text((1086, p_y), "Algorithm: PPO2 + VAE-LSTM (VM) World-Model Policy", font=self.font_body_bold, fill=(162, 210, 255))
        draw.text((1086, p_y + 18), "Controller: AirportNavigationController (TCA Reactive)", font=self.font_body, fill=(220, 230, 240))
        draw.text((1086, p_y + 36), "Model: navreptrainencodedenv_PPO_VAE_LSTM_ckpt.zip", font=self.font_sub, fill=(175, 190, 205))
        draw.text((1086, p_y + 54), "Deterministic: True  |  Action: [{:+.2f}, {:+.2f}, {:+.2f}]".format(
            action[0], action[1], action[2] if len(action) > 2 else 0.0), font=self.font_body, fill=(220, 230, 240))
        draw.text((1086, p_y + 72), "Dynamic Pedestrians: {:d} agents in scene".format(len(humans)), font=self.font_body, fill=(220, 230, 240))
        h_color = AMBER_WARN if min_human_dist <= 2.0 else (220, 230, 240)
        draw.text((1086, p_y + 90), "Nearest Pedestrian Dist: {:4.2f} m".format(min_human_dist), font=self.font_body, fill=h_color)
        draw.text((1086, p_y + 108), "Time Limit: {:d} steps ({:.1f} s @ dt=0.2s)".format(max_steps, time_limit), font=self.font_body, fill=(220, 230, 240))

        # BOX 7: 20-Seed Quantitative Benchmark
        b_y = 742
        stats = BENCHMARK_STATS.get(diff, BENCHMARK_STATS['MEDIUM'])
        draw.text((1086, b_y), "Evaluation Ladder: {} (Seeds 100-119, N=20)".format(diff), font=self.font_body_bold, fill=ACCENT_CYAN)
        draw.text((1086, b_y + 20), "Success Rate:   {:5.1f}%  ({:2d}/20 episodes)".format(
            stats['success_rate'], stats['success_count']), font=self.font_body_bold, fill=GREEN_SUCCESS)
        draw.text((1086, b_y + 40), "Collision Rate: {:5.1f}%  ({:2d}/20 episodes)".format(
            stats['collision_rate'], stats['collision_count']), font=self.font_body_bold, fill=(255, 107, 107))
        draw.text((1086, b_y + 60), "Timeout Rate:   {:5.1f}%  ({:2d}/20 episodes)".format(
            stats['timeout_rate'], stats['timeout_count']), font=self.font_body_bold, fill=AMBER_WARN)
        draw.text((1086, b_y + 80), "Mean Time-to-Goal: {} s".format(stats['mean_time']), font=self.font_body_bold, fill=(0, 229, 255))
        draw.text((1086, b_y + 104), "Benchmark Status: Rigorous 20-Seed Evaluation Complete", font=self.font_small, fill=(127, 140, 155))

        # Convert PIL Image to RGB numpy array
        return np.array(frame, dtype=np.uint8)
