# Task 3: Airport Passenger Assistance Scenario

## Scenario Narrative & Overview
An autonomous airport assistance robot guides an elderly passenger carrying heavy luggage from the terminal check-in hall entrance to boarding Gate 12. The robot must traverse a realistic 24m x 15m terminal hall featuring check-in counter islands, information kiosks, security inspection lanes, structural columns, lounge seating banks, duty-free retail, and boarding podiums. Along the concourse, the robot encounters heterogeneous dynamic human traffic including slow travelers pushing bulky luggage trolleys ($r=0.45\,\text{m}, v=0.40\,\text{m/s}$), hurried business travelers rushing to gates ($r=0.30\,\text{m}, v=0.95\,\text{m/s}$), a coordinated family group walking together ($v=0.55\,\text{m/s}$), and airport staff crossing the concourse ($v=0.60\,\text{m/s}$). On MEDIUM and HARD levels, a dynamic "Boarding Call" event redirects concourse travelers toward Gate 12, creating dynamic multi-agent interaction along the primary concourse aisle.

## Architectural Changes & Specifications

| Dimension | Hotel Room Service (`hotelroomservice`) | Airport Passenger Assistance (`airportpassengerassistance`) |
| :--- | :--- | :--- |
| **Story Context** | Room service tray delivery in a hotel corridor | Luggage assistance robot guiding passenger to Gate 12 |
| **World Bounds** | $[-12.0, 12.0] \times [-8.0, 8.0]\,\text{m}$ ($24\times 16\,\text{m}$) | $[-12.0, 12.0] \times [-7.5, 7.5]\,\text{m}$ ($24\times 15\,\text{m}$) |
| **Robot Start** | $(-10.0, 0.0, 0.0\,\text{rad})$ (Lobby / Service Kitchen) | $(-10.2, -2.5, 0.0\,\text{rad})$ (Terminal Check-in Entrance) |
| **Robot Goal** | $(10.0, 0.0)$ (Guest Suite 104) | $(9.8, 2.8)$ (Gate 12 Boarding Podium) |
| **Separation Dist** | $20.0\,\text{m}$ | $20.69\,\text{m}$ (Diagonal cross-concourse trajectory) |
| **Wall Topology** | Double outer wall (perimeter poly + border offset) | Double outer wall (perimeter poly + border offset) |
| **Obstacle Layout** | 6 rectangular guest room blocks along corridor | 10 rectangular islands: check-in desks, kiosk, security lanes, columns, seating, duty-free |
| **Dynamic Humans** | 4 base agents (hotel guests, trolley staff) | 5 base agents: luggage trolley, hurried traveler, family parent/child, staff + crowd |
| **Special Event** | None | Dynamic "Boarding Call" at step $T=75$ (MEDIUM) and $T=50$ (HARD) heading to Gate 12 |
| **Difficulty Ladder**| EASY (4), MEDIUM (7), HARD (11) | EASY (5 humans, 420 steps), MEDIUM (8 humans, 360 steps), HARD (12 humans, 290 steps) |
| **Navigation Policy**| `HotelNavigationController` ("HOTEL NAVIGATION") | `AirportNavigationController` ("AIRPORT NAVIGATION") |

> [!NOTE]
> **Policy Architecture**: Consistent with the hotel room service scenario reference implementation, `"POLICY: AIRPORT NAVIGATION"` in the video HUD telemetry refers to `AirportNavigationController` in `navrep_custom/airport_passenger_assistance.py`. It implements reactive waypoint tracking, dynamic Time-to-Closest-Approach (TCA) collision avoidance against pedestrians, lateral detour maneuvering if blocked, and terminal docking deceleration at Gate 12.

## Map Layout Preview

![Airport Passenger Assistance Map](maps/airport_passenger_assistance.png)

- **Green S Circle**: Check-in Entrance start position $(-10.2, -2.5)$ with $\ge 1.80\,\text{m}$ clearance to outer walls and $\ge 2.16\,\text{m}$ clearance to obstacles.
- **Gold Goal Ring**: Gate 12 boarding podium $(9.8, 2.8)$ with $\ge 1.87\,\text{m}$ clearance to obstacles.
- **Yellow Waypoint Path**: Collision-free concourse passage through check-in aisles, central concourse junction, and lounge corridors.
- **Coral Human Circles**: Dynamic human agents with velocity orientation arrows.

## Empirical Benchmark Results (20 Episodes per Level, Seeds 100–119)

Each level was evaluated across 20 fixed benchmark episodes (seeds 100–119) using `AirportNavRepEnv` and `AirportNavigationController`. Metrics strictly sum to $100\%$ across outcomes:

| Difficulty Level | Dynamic Humans | Time Limit (Steps / s) | Success Rate (%) | Collision Rate (%) | Timeout Rate (%) | Mean Time-to-Goal (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EASY** | 5 | 420 steps ($84.0\,\text{s}$) | **100.0%** (20/20) | 0.0% (0/20) | 0.0% (0/20) | 32.76 s |
| **MEDIUM** | 8 | 360 steps ($72.0\,\text{s}$) | **50.0%** (10/20) | 20.0% (4/20) | 30.0% (6/20) | 33.46 s |
| **HARD** | 12 | 290 steps ($58.0\,\text{s}$) | **15.0%** (3/20) | 50.0% (10/20) | 35.0% (7/20) | 32.87 s |

### Seed Calibration (Medium Search, Seeds 0–59)
- **Evaluated**: 60 episodes at MEDIUM difficulty.
- **Goal Reached Successes**: 35 / 60 (58.3%) — e.g. seeds `0`, `1`, `3`, `4`, `7`, `42`.
- **Collisions**: 9 / 60 (15.0%) — e.g. seeds `5`, `13`, `20`, `27`, `38`.
- **Timeouts**: 16 / 60 (26.7%) — e.g. seeds `2`, `6`, `8`, `11`, `14`.

## Demonstration Videos (Standard 640x640 Format)

All standard videos are rendered at $640\times 640$, 12 fps, H.264 `yuv420p` with top-down static camera, cyan path trail, real-time distance-to-goal tracking, and verified quality gates:

| File | Level | Seed | Outcome | Steps | Final Dist | Duration | Frozen Frames | Quality Gate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [`airport_passenger_assistance_easy_success.mp4`](videos/airport_passenger_assistance_easy_success.mp4) | EASY | 100 | GOAL_REACHED | 164 | 0.35 m | 14.50 s | 10 | **PASS** |
| [`airport_passenger_assistance_medium_success.mp4`](videos/airport_passenger_assistance_medium_success.mp4) | MEDIUM | 0 | GOAL_REACHED | 163 | 0.41 m | 14.42 s | 10 | **PASS** |
| [`airport_passenger_assistance_medium_failure.mp4`](videos/airport_passenger_assistance_medium_failure.mp4) | MEDIUM | 118 | COLLISION | 287 | 6.61 m | 24.75 s | 10 | **PASS** |
| [`airport_passenger_assistance_hard_bestprogress.mp4`](videos/airport_passenger_assistance_hard_bestprogress.mp4) | HARD | 111 | TIMEOUT | 290 | 1.30 m | 25.00 s | 10 | **PASS** |

---

## Interactive Mission-Control Dashboard UI (1600x900 Restyle)

To provide an executive-level interactive demonstration interface, the scenario videos were restyled into a high-resolution $1600\times 900$ mission-control dashboard replicating the professional layout of the institutional reference UI (`ui_reference/library_ui.jpeg`).

### 1. Canvas & Display Architecture
- **Dimensions**: $1600\times 900$ pixels, encoded in H.264 `yuv420p` with `macro_block_size=None` to ensure strict 16:9 aspect ratio preservation.
- **Frame Rate & Sim Pacing (2.0x Fast)**: Rendered at $10\,\text{fps}$. In the underlying NavRep physics engine, each simulation step represents $\Delta t = 0.20\,\text{s}$ ($5\,\text{Hz}$). At $10\,\text{fps}$ playback, 10 frames represent $1.0\,\text{s}$ of video time while advancing $10 \times 0.20\,\text{s} = 2.0\,\text{s}$ of simulation time. Thus, the playback speed is exactly $2.0\times$ real-time speed, correctly reflected by the active `(*) 2.0x (Fast)` radio selector in the Sim Pacing control widget.
- **Rendering Engine**: Custom Pillow-based dashboard renderer (`navrep_custom/airport_dashboard_renderer.py`) utilizing TrueType monospace typography (`Consolas`, `consolab.ttf`), pre-rendered static background caching, anti-aliased geometry, and dynamic metric overlays.

### 2. Header & Telemetry Badges
- **Header Title**: `AIRPORT PASSENGER ASSISTANCE ROBOT` in high-contrast monospace.
- **Subtitle**: `Task 3: Autonomous Mobile Navigation in an International Airport Terminal | Author: Ritu (SVNIT Surat)`.
- **Top-Right Status Chips**:
  - `MODE: AIRPORT VIEW` (Cyan background `#0d7377`).
  - `DIFFICULTY: EASY / MEDIUM / HARD` (Color-coded: Green `#1b5e20` for EASY, Amber `#e65100` for MEDIUM, Red `#b71c1c` for HARD).
  - `STATUS: <OUTCOME>`: Dynamically transitions from `READY` (Grey `#424242`, initial 5 briefing frames) $\to$ `NAVIGATING` (Blue `#1565c0`) $\to$ final outcome badge: `GOAL REACHED` (Emerald `#2e7d32`), `COLLISION` (Crimson `#c62828`), or `TIMEOUT` (Dark Gold `#ef6c00`).

### 3. Left Panel: Floor Plan & Terminal Environment
- **Title Bar**: `AIRPORT TERMINAL FLOOR PLAN (24x15 m) -- CHECK-IN HALL, SECURITY & GATE AREA`.
- **Visual Styling**: Light beige concourse floor (`#f0ece1`), dark brown double perimeter walls (`#4a2e18`), distinct concourse zone annotations (`CHECK-IN HALL`, `CENTRAL CONCOURSE`, `DEPARTURE LOUNGE`, `DUTY-FREE AISLE`, `JETBRIDGE ACCESS`, `GATE 12 AREA`).
- **Obstacle Layout**: Rendered with structural styling and labels:
  - Check-in desks: `CHECK-IN DESKS A & B` (wood-grain desks with luggage belt indicators).
  - Information kiosk: `INFO KIOSK`.
  - Security inspection lanes: `SECURITY CHECKPOINT`.
  - Structural columns: `PILLAR A` and `PILLAR B` (hexagonal concrete discs with cross-hatching).
  - Lounge seating: `LOUNGE SEATING A & B` (navy upholstered passenger seating rows).
  - Retail & boarding: `DUTY-FREE AISLE` and `GATE 12 AREA`.
- **Agents & Dynamic Traffic**:
  - **Robot**: Royal blue disc with white orientation vector arrow and cyan breadcrumb path trail.
  - **Dynamic Pedestrians**: Colored discs with velocity direction arrows and true role callouts derived from environment metadata: `P1: Trolley`, `P2: Hurried`, `P3: Family`, `P4: Child`, `P5: Staff`, etc.
  - **Start & Goal**: Green callout `START: Check-in Entrance` $(-10.2, -2.5)$ and dual-ring gold marker with orange callout `GOAL: Gate 12 (Boarding)` $(9.8, 2.8)$.

### 4. Right Panel: Control & Telemetry Deck
- **Showcase Demonstration**: High-visibility `[>] RUN FULL DEMONSTRATION` button and workflow sequence indicator (`View Terminal -> Route Plan -> Navigate & Assist Passenger`).
- **Modular Operations**: Status-sensitive operational buttons (`Show Airport`, `Run Path Planning`, `Start Navigation`, `Pause`, `Reset`).
- **Interactive Goal Selection & Replanning**: Active goal preset selector displaying `Gate 12 (Boarding Podium)` with calculated Euclidean route length ($20.69\,\text{m}$) and active waypoint sequence count.
- **Difficulty & Sim Pacing Radios**: Interactive-style radio selectors indicating current episode configuration.
- **Live Telemetry & Navigation State**:
  - `Step`: $k / K_{\max}$ | `Sim Time`: $t.t\,\text{s} / T_{\max}\,\text{s}$
  - `Pose`: $(x, y)$ coordinates and heading $\theta$ in degrees.
  - `Speed`: Instantaneous linear velocity $|v|$ and component vectors $(v_x, v_y)$.
  - `Dist to Goal`: Distance $d$ in meters and displacement vector $(dx, dy)$ captured prior to terminal environment reset.
  - `Active Waypoint`: Current target waypoint index and coordinates $\text{WP } i / N \to (x_w, y_w)$.
  - `Controller State`: High-level operational state dynamically derived from kinematic state and human proximity.

### 5. Derived Controller State Rules
The controller state telemetry is dynamically evaluated at each step using deterministic kinematic rules:
1. **`READY`**: Active during initial briefing/freeze frames prior to episode trajectory rollout (step 0, frame index $< 5$).
2. **`YIELDING (Pedestrian clearance)`**: Triggered when the robot decelerates below normal cruising speed ($|v| < 0.25\,\text{m/s}$) while an active dynamic pedestrian is within close collision buffer range ($d_{\text{ped}} < 2.2\,\text{m}$) or approaching along a conflicting trajectory, indicating active collision avoidance yielding.
3. **`CRUISING (<Zone>)`**: Triggered when the robot is actively translating ($|v| \ge 0.25\,\text{m/s}$ or clear pedestrian headway). The location tag is dynamically determined from spatial bounding boxes:
   - $x < -4.0$: `CRUISING (Check-in Hall)`
   - $-4.0 \le x < 2.0$: `CRUISING (Central Concourse)`
   - $2.0 \le x < 7.0$: `CRUISING (Departure Lounge)`
   - $x \ge 7.0$: `CRUISING (Gate 12 Area)`
4. **`GOAL REACHED`**: Triggered when Euclidean distance to Gate 12 satisfies docking tolerance ($d \le 1.0\,\text{m}$).
5. **`COLLISION`**: Triggered upon physical contact with a pedestrian or obstacle boundary.
6. **`TIMEOUT`**: Triggered if elapsed step count reaches maximum budget without reaching docking tolerance.

### 6. Navigation Policy Details (Replacement of A* Box)
The reference library UI displayed an "A* Search Metrics & Plan Details" box. However, the Airport Passenger Assistance robot operates using a continuous, learned policy framework / deterministic reactive waypoint tracking controller (`AirportNavigationController` and PPO2 + VAE-LSTM World-Model policy) operating in continuous velocity space $(v_x, v_y, \omega)$.

To adhere strictly to engineering truthfulness and avoid inventing fictitious A* open/closed lists or heuristic expansions, the section was replaced with **Navigation Policy Details**, displaying verified operational parameters:
- **Algorithm**: `PPO2 + VAE-LSTM (NavRep VM)`
- **Controller**: `AirportNavigationController`
- **Action Space**: Continuous Holonomic $(v_x, v_y, \omega)$
- **Dynamic Pedestrians**: Real-time count of active dynamic humans in the terminal
- **Nearest Pedestrian**: Euclidean distance to nearest pedestrian
- **Step Budget / Time Limit**: Strict difficulty-specific step budget ($420 / 360 / 290$ steps)

### 7. 20-Seed Quantitative Benchmark Box
Directly embeds the true empirical evaluation results from the 20-episode benchmark suite (`airport_metrics.csv`):
- **EASY**: 100.0% Success (20/20), 0.0% Collisions, 0.0% Timeouts, Mean Time: 32.76s (Target: $\ge 70\%$)
- **MEDIUM**: 50.0% Success (10/20), 20.0% Collisions, 30.0% Timeouts, Mean Time: 33.46s (Target: $\ge 40\%$)
- **HARD**: 15.0% Success (3/20), 50.0% Collisions, 35.0% Timeouts, Mean Time: 32.87s (Target: $\ge 15\%$)

### 8. Dashboard Video Verification Results

All 4 demonstration episodes were replayed using the exact same seeds, environment parameters, and controller logic as the baseline videos. Automated assertions confirmed bit-for-bit parity of step counts, final distances, and outcomes against the original sidecar metadata:

| File | Level | Seed | Outcome | Status Chip | Steps | Final Dist | Duration | Frozen Frames | Quality Gate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [`airport_passenger_assistance_easy_success_dashboard.mp4`](videos_dashboard/airport_passenger_assistance_easy_success_dashboard.mp4) | EASY | 100 | GOAL_REACHED | GOAL_REACHED (Green) | 164 | 0.35 m | 17.90 s | 8 | **PASS** |
| [`airport_passenger_assistance_medium_success_dashboard.mp4`](videos_dashboard/airport_passenger_assistance_medium_success_dashboard.mp4) | MEDIUM | 0 | GOAL_REACHED | GOAL_REACHED (Green) | 163 | 0.41 m | 17.80 s | 7 | **PASS** |
| [`airport_passenger_assistance_medium_failure_dashboard.mp4`](videos_dashboard/airport_passenger_assistance_medium_failure_dashboard.mp4) | MEDIUM | 118 | COLLISION | COLLISION (Red) | 287 | 6.61 m | 30.20 s | 9 | **PASS** |
| [`airport_passenger_assistance_hard_bestprogress_dashboard.mp4`](videos_dashboard/airport_passenger_assistance_hard_bestprogress_dashboard.mp4) | HARD | 111 | TIMEOUT | TIMEOUT (Amber) | 290 | 1.30 m | 30.50 s | 11 | **PASS** |

- **Frozen Frame Verification**: Ready briefing (~5 frames) + terminal hold (10 frames) produced 7–11 total frozen frames per video (strictly within the $\le 20$ limit).
- **Video Format Integrity**: All videos conform to $1600\times 900$, 10 fps, H.264 `yuv420p`, duration $17.8\,\text{s} - 30.5\,\text{s}$ (within the $8-40\,\text{s}$ requirement).
- **Visual Auditing**: Frame snapshots sampled at 5%, 50%, and 97% confirm crisp rendering, high-contrast text, clear obstacle styling, and clean text margins with zero edge clipping.

---

## Execution & Reproduction Commands

### 1. Run Scenario Unit Tests
```bash
python -m unittest tests/test_airport_scenario.py
```

### 2. Run Calibration & 20-Episode Benchmark
```bash
python scripts_custom/benchmark_airport.py
```

### 3. Record All 4 Standard Demonstration Videos (640x640)
```bash
python scripts_custom/record_airport_demo.py
```

### 4. Record & Verify All 4 Dashboard Demonstration Videos (1600x900)
```bash
python scripts_custom/record_airport_dashboard.py
```

### 5. Verify Video Quality Gates Only
```bash
# Verify standard 640x640 videos
python scripts_custom/record_airport_demo.py --verify-only

# Verify 1600x900 mission-control dashboard videos
python scripts_custom/record_airport_dashboard.py --verify-only
```
