#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_hotel_gui_rendering.py
Verifies the luxury hotel rendering methods and generates a sample frame to inspect visually.
"""
from __future__ import print_function
import math
import numpy as np
import cv2

# Hotel Scenario High-Resolution Floor Canvas Colors (BGR for OpenCV)
HOTEL_FLOOR_MARBLE = (220, 228, 236)
HOTEL_FLOOR_GROUT = (195, 205, 215)
HOTEL_CARPET_BASE = (38, 16, 115)
HOTEL_CARPET_PATTERN = (55, 26, 140)
HOTEL_BRASS_TRIM = (55, 195, 230)
HOTEL_WALL_WOOD = (22, 40, 75)
HOTEL_WALL_BORDER = (12, 24, 48)
HOTEL_DOOR_MAHOGANY = (25, 48, 95)
HOTEL_DOOR_BRASS = (45, 185, 220)
HOTEL_PLAQUE_BG = (15, 30, 60)
HOTEL_PLAQUE_GOLD = (50, 215, 255)

HOTEL_RECEPTION_TOP = (24, 45, 88)
HOTEL_RECEPTION_BELL = (40, 190, 240)
HOTEL_MONITOR_BEZEL = (35, 35, 35)
HOTEL_MONITOR_SCREEN = (220, 140, 40)

HOTEL_SOFA_SAPPHIRE = (138, 52, 20)
HOTEL_SOFA_BORDER = (90, 32, 10)
HOTEL_COFFEE_TABLE = (35, 65, 120)
HOTEL_TABLE_GLASS = (210, 215, 200)
HOTEL_PLANT_GREEN = (40, 145, 60)
HOTEL_PLANT_POT = (45, 75, 160)

HOTEL_TOWEL_WHITE = (245, 245, 245)
HOTEL_TOWEL_TEAL = (180, 170, 40)
HOTEL_HAMPER_COL = (70, 95, 125)

HOTEL_ROBOT_SILVER = (195, 200, 205)
HOTEL_ROBOT_CLOCHE = (230, 235, 240)
HOTEL_ROBOT_HANDLE = (45, 190, 235)
HOTEL_ROBOT_AMBER = (30, 160, 240)

ROLE_GUEST = (190, 110, 35)
ROLE_HOUSEKEEPING = (45, 165, 95)
ROLE_SUITCASE = (35, 35, 85)
ROLE_LINEN_CART = (85, 95, 105)

HOTEL_SUITE_METADATA = {
    4: ("SUITE 201", "Executive King", True),
    5: ("SUITE 202", "Deluxe Double", True),
    6: ("SUITE 203", "Presidential Suite", True),
    7: ("ROOM 205", "Superior Suite", False),
    8: ("ROOM 206", "Deluxe Twin", False),
    9: ("ROOM 207", "Junior Suite", False),
}

canvas_width = 860
canvas_height = 640
world_min_x = -12.5
world_max_x = 12.5
world_min_y = -8.5
world_max_y = 8.5

scale_px = min(
    (canvas_width - 40) / (world_max_x - world_min_x),
    (canvas_height - 40) / (world_max_y - world_min_y)
)
orig_x = int(20 + (-world_min_x) * scale_px)
orig_y = int(canvas_height / 2.0)

def w2px(wx, wy):
    px = int(orig_x + wx * scale_px)
    py = int(orig_y - wy * scale_px)
    return px, py

def render_sample_hotel_frame():
    from navrep_custom.scenarios import get_scenario_config
    cfg = get_scenario_config('hotelroomservice')
    img = np.full((canvas_height, canvas_width, 3), HOTEL_FLOOR_MARBLE, dtype=np.uint8)

    # 1. Marble tile grout lines
    for gx in range(0, canvas_width, 36):
        cv2.line(img, (gx, 0), (gx, canvas_height), HOTEL_FLOOR_GROUT, 1)
    for gy in range(0, canvas_height, 36):
        cv2.line(img, (0, gy), (canvas_width, gy), HOTEL_FLOOR_GROUT, 1)

    # 2. Central plush runner carpet
    c_tl = w2px(-12.0, 2.7)
    c_br = w2px(12.0, -2.7)
    cv2.rectangle(img, (0, c_tl[1]), (canvas_width, c_br[1]), HOTEL_CARPET_BASE, -1)
    # Brass transition trim
    cv2.line(img, (0, c_tl[1]), (canvas_width, c_tl[1]), HOTEL_BRASS_TRIM, 4)
    cv2.line(img, (0, c_br[1]), (canvas_width, c_br[1]), HOTEL_BRASS_TRIM, 4)
    # Inner gold pinstripes
    cv2.line(img, (0, c_tl[1] + 6), (canvas_width, c_tl[1] + 6), (70, 210, 245), 1)
    cv2.line(img, (0, c_br[1] - 6), (canvas_width, c_br[1] - 6), (70, 210, 245), 1)
    # Repeating diamond pattern
    mid_y = int((c_tl[1] + c_br[1]) / 2.0)
    for dx in range(30, canvas_width, 60):
        d_pts = np.array([[dx, mid_y - 8], [dx + 12, mid_y], [dx, mid_y + 8], [dx - 12, mid_y]], dtype=np.int32)
        cv2.polylines(img, [d_pts], isClosed=True, color=HOTEL_CARPET_PATTERN, thickness=1)

    # 3. Outer architectural perimeter walls
    outer_poly = np.array([w2px(pt[0], pt[1]) for pt in cfg.static_obstacles[0]], dtype=np.int32)
    cv2.polylines(img, [outer_poly], isClosed=True, color=(20, 35, 60), thickness=6)

    # 4. Obstacles
    for idx, poly in enumerate(cfg.static_obstacles[1:], start=1):
        pts = np.array([w2px(pt[0], pt[1]) for pt in poly], dtype=np.int32)
        x_min, x_max = np.min(pts[:, 0]), np.max(pts[:, 0])
        y_min, y_max = np.min(pts[:, 1]), np.max(pts[:, 1])

        if idx == 1:
            # Reception Desk
            cv2.fillPoly(img, [pts + np.array([4, 4], dtype=np.int32)], (180, 190, 200))
            cv2.fillPoly(img, [pts], HOTEL_RECEPTION_TOP)
            cv2.polylines(img, [pts], isClosed=True, color=HOTEL_BRASS_TRIM, thickness=2)
            # Golden Call Bell at (-7.2, 4.8)
            b_px, b_py = w2px(-7.2, 4.8)
            cv2.circle(img, (b_px, b_py), 6, HOTEL_RECEPTION_BELL, -1)
            cv2.circle(img, (b_px, b_py), 7, (20, 120, 180), 1)
            cv2.line(img, (b_px, b_py - 6), (b_px, b_py - 9), (20, 120, 180), 2)
            # Computer monitor at (-8.7, 4.8)
            m_px, m_py = w2px(-8.7, 4.8)
            cv2.rectangle(img, (m_px - 8, m_py - 5), (m_px + 8, m_py + 5), HOTEL_MONITOR_BEZEL, -1)
            cv2.rectangle(img, (m_px - 6, m_py - 4), (m_px + 6, m_py + 4), HOTEL_MONITOR_SCREEN, -1)
            # Desk blotter at (-7.9, 4.8)
            bl_px, bl_py = w2px(-7.9, 4.8)
            cv2.rectangle(img, (bl_px - 6, bl_py - 4), (bl_px + 6, bl_py + 4), (40, 70, 120), -1)
            # Plaque above desk
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
                p_px, p_py = w2px(-7.0, -5.5)
                cv2.circle(img, (p_px, p_py), 8, HOTEL_PLANT_POT, -1)
                cv2.circle(img, (p_px, p_py), 6, HOTEL_PLANT_GREEN, -1)
            else:
                t_px, t_py = w2px(-2.85, -5.5)
                cv2.rectangle(img, (t_px - 8, t_py - 10), (t_px + 8, t_py + 10), HOTEL_COFFEE_TABLE, -1)
                cv2.rectangle(img, (t_px - 6, t_py - 8), (t_px + 6, t_py + 8), HOTEL_TABLE_GLASS, -1)
                cv2.circle(img, (t_px, t_py), 3, (180, 80, 220), -1)
                p_px, p_py = w2px(0.8, -5.5)
                cv2.circle(img, (p_px, p_py), 8, HOTEL_PLANT_POT, -1)
                cv2.circle(img, (p_px, p_py), 6, HOTEL_PLANT_GREEN, -1)
                # Lounge Label above table
                cv2.putText(img, "EXECUTIVE LOUNGE", (t_px - 50, y_min - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (30, 45, 80), 1, cv2.LINE_AA)

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

    # 5. Start & Goal Markers
    # Start: Kitchen Service Pantry (-10.0, 0.0)
    s_px, s_py = w2px(-10.0, 0.0)
    cv2.rectangle(img, (s_px - 14, s_py - 18), (s_px + 14, s_py + 18), (170, 175, 180), -1)
    cv2.line(img, (s_px, s_py - 18), (s_px, s_py + 18), (110, 115, 120), 2)
    cv2.circle(img, (s_px, s_py), 14, (60, 180, 75), 2, cv2.LINE_AA)
    cv2.circle(img, (s_px, s_py), 6, (60, 180, 75), -1, cv2.LINE_AA)
    cv2.putText(img, "START: Pantry", (s_px - 45, s_py - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 255, 230), 1, cv2.LINE_AA)

    # Goal: Suite 204 Delivery (+10.0, 0.0)
    g_px, g_py = w2px(10.0, 0.0)
    cv2.rectangle(img, (g_px - 12, g_py - 22), (g_px + 12, g_py + 22), HOTEL_DOOR_MAHOGANY, -1)
    cv2.rectangle(img, (g_px - 12, g_py - 22), (g_px + 12, g_py + 22), HOTEL_DOOR_BRASS, 1)
    cv2.circle(img, (g_px, g_py), 16, (0, 190, 250), 2, cv2.LINE_AA)
    cv2.circle(img, (g_px, g_py), 8, HOTEL_ROBOT_CLOCHE, -1, cv2.LINE_AA)
    cv2.circle(img, (g_px, g_py - 8), 2, HOTEL_ROBOT_HANDLE, -1, cv2.LINE_AA)
    cv2.putText(img, "GOAL: Suite 204", (g_px - 55, g_py - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (70, 220, 255), 2, cv2.LINE_AA)

    # 6. Robot AMR
    r_px, r_py = w2px(-5.0, 0.0)
    cv2.circle(img, (r_px - 12, r_py), 4, (30, 30, 30), -1)
    cv2.circle(img, (r_px + 12, r_py), 4, (30, 30, 30), -1)
    cv2.circle(img, (r_px, r_py), 14, HOTEL_ROBOT_SILVER, -1)
    cv2.circle(img, (r_px, r_py), 14, (140, 145, 150), 2)
    cv2.circle(img, (r_px, r_py), 8, HOTEL_ROBOT_CLOCHE, -1)
    cv2.circle(img, (r_px, r_py), 8, (160, 165, 170), 1)
    cv2.circle(img, (r_px, r_py), 2, HOTEL_ROBOT_HANDLE, -1)
    cv2.rectangle(img, (r_px + 12, r_py - 4), (r_px + 15, r_py + 4), HOTEL_ROBOT_AMBER, -1)
    cv2.line(img, (r_px, r_py), (r_px + 22, r_py), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "AMR: Service", (r_px - 36, r_py - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)

    # 7. Pedestrians
    h1_px, h1_py = w2px(0.0, 1.0)
    s_x, s_y = h1_px - 14, h1_py
    cv2.rectangle(img, (s_x - 5, s_y - 4), (s_x + 5, s_y + 4), ROLE_SUITCASE, -1)
    cv2.line(img, (s_x + 5, s_y), (h1_px, h1_py), (50, 50, 50), 1)
    cv2.circle(img, (h1_px, h1_py), 9, ROLE_GUEST, -1)
    cv2.circle(img, (h1_px, h1_py), 9, (20, 60, 100), 1)
    cv2.putText(img, "Guest", (h1_px - 14, h1_py - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)

    h2_px, h2_py = w2px(-2.0, -1.5)
    c_x, c_y = h2_px + 16, h2_py
    cv2.rectangle(img, (c_x - 8, c_y - 12), (c_x + 8, c_y + 12), ROLE_LINEN_CART, -1)
    cv2.rectangle(img, (c_x - 6, c_y - 8), (c_x - 1, c_y + 8), HOTEL_TOWEL_WHITE, -1)
    cv2.circle(img, (h2_px, h2_py), 10, ROLE_HOUSEKEEPING, -1)
    cv2.circle(img, (h2_px, h2_py), 10, (20, 90, 50), 1)
    cv2.putText(img, "Housekeeping", (h2_px - 34, h2_py - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)

    # 8. Legend
    cv2.rectangle(img, (20, canvas_height - 38), (560, canvas_height - 12), (25, 28, 34), -1)
    cv2.rectangle(img, (20, canvas_height - 38), (560, canvas_height - 12), (50, 55, 65), 1)
    cv2.circle(img, (32, canvas_height - 25), 6, HOTEL_ROBOT_SILVER, -1)
    cv2.circle(img, (32, canvas_height - 25), 3, HOTEL_ROBOT_CLOCHE, -1)
    cv2.putText(img, "AMR Cloche (0.6m)", (44, canvas_height - 21), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220, 225, 230), 1)
    cv2.circle(img, (155, canvas_height - 25), 6, ROLE_GUEST, -1)
    cv2.putText(img, "Guest + Luggage", (167, canvas_height - 21), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220, 225, 230), 1)
    cv2.circle(img, (275, canvas_height - 25), 6, ROLE_HOUSEKEEPING, -1)
    cv2.putText(img, "Housekeeping + Cart", (287, canvas_height - 21), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220, 225, 230), 1)
    cv2.rectangle(img, (415, canvas_height - 29), (427, canvas_height - 21), HOTEL_WALL_WOOD, -1)
    cv2.putText(img, "Suites", (433, canvas_height - 21), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220, 225, 230), 1)
    cv2.line(img, (485, canvas_height - 25), (505, canvas_height - 25), (40, 210, 80), 2)
    cv2.putText(img, "A* Route", (512, canvas_height - 21), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220, 225, 230), 1)

    cv2.imwrite('/workspace/test_hotel_render.png', img)
    print('Refined hotel frame rendered successfully.')

if __name__ == '__main__':
    render_sample_hotel_frame()
