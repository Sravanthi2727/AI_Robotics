# SVNIT CS359: AI for Robotics — Complete Project Reproduction & Evaluation Guide

**Institution:** Sardar Vallabhbhai National Institute of Technology (SVNIT), Surat  
**Department:** Computer Science & Engineering Department  
**Course:** CS359: Artificial Intelligence for Robotics (B.Tech CSE IIIrd Year)  
**Project:** ETH Zurich NavRep Containerization, Benchmark Evaluation, & Custom Multi-Scenario Autonomous Navigation  
**Author / Presenter:** Shriya Siddesh  
**Git Branch:** `shriya` | **Latest Verified Commit:** `9e50fee` | **Test Suite:** 28 / 28 PASS  

---

## Quick Reference: Zero-to-Running in 4 Commands

For an evaluator, grader, or teammate cloning this repository on a fresh machine:

```bash
# 1. Clone repository and checkout branch 'shriya'
git clone https://github.com/Sravanthi2727/AI_Robotics.git && cd AI_Robotics
git checkout shriya

# 2. Start the containerized desktop environment (maps ports 5900 and 6080)
docker compose up -d

# 3. Run automated smoke tests and 28-test verification suite
docker exec navrep_desktop python scripts_custom/smoke_test.py
docker exec navrep_desktop python -m unittest tests/test_library_scenario.py tests/test_custom_scenarios.py

# 4. Open the live GUI in your web browser:
# Navigate to: http://localhost:6080/vnc_lite.html?scale=true
```

---

## Table of Contents
1. [System Architecture & From-Scratch Environment Setup](#1-system-architecture--from-scratch-environment-setup)
2. [Task 1: Environment Reproduction & Model Verification](#2-task-1-environment-reproduction--model-verification)
3. [Task 2: Existing Scenarios Benchmark & 22-Video Verification](#3-task-2-existing-scenarios-benchmark--22-video-verification)
4. [Task 3: Custom Robotics Navigation Scenarios](#4-task-3-custom-robotics-navigation-scenarios)
   - [Architectural Philosophy: Hierarchical Navigation (Global A* + Local Predictive TCA)](#41-architectural-philosophy-hierarchical-navigation)
   - [Scenario 1: Institutional Library Book Collection Robot (Shriya)](#42-scenario-1-institutional-library-book-collection-robot)
   - [Scenario 2: Grand Hotel Room Service Delivery Robot (Tanishka)](#43-scenario-2-grand-hotel-room-service-delivery-robot)
   - [Other Custom Scenarios (MallMate, MediRunner, KiranaKart, Diwali Bazaar, Platform 1, Fire Drill)](#44-other-custom-scenarios-in-the-suite)
5. [Task 3: Live Graphical Demonstration Interface (VNC & Web Browser)](#5-task-3-live-graphical-demonstration-interface-vnc--web-browser)
   - [VNC & Browser Access Setup](#51-vnc--browser-access-setup)
   - [Launching Scenario 1 (Library)](#52-launching-scenario-1-library)
   - [Launching Scenario 2 (Hotel Room Service)](#53-launching-scenario-2-hotel-room-service)
   - [Switching Scenarios Live in GUI](#54-switching-scenarios-live-in-gui)
   - [Interactive Controls & Dynamic A* Replanning Walkthrough](#55-interactive-controls--dynamic-a-replanning-walkthrough)
6. [Troubleshooting & Common Failure Remediation](#6-troubleshooting--common-failure-remediation)
7. [Technical Defense & Viva Voce Q&A Cheat Sheet](#7-technical-defense--viva-voce-qa-cheat-sheet)
8. [Submission Deliverables & File Manifest](#8-submission-deliverables--file-manifest)

---

## 1. System Architecture & From-Scratch Environment Setup

### 1.1 Why Containerization was Required
The upstream ETH Zurich **NavRep** framework is built on a legacy research stack:
- **Python 3.6** (end-of-life)
- **TensorFlow 1.14 / Keras 2.3.1**
- **OpenAI Gym 0.10 / 0.15**
- **PyTorch 1.1**
- **ROS Melodic / Kinetic tools**
- Custom compiled C++ collision extensions (`crings`).

On modern operating systems (macOS Sonoma/Sequoia with Apple Silicon ARM64, or Ubuntu 22.04/24.04), installing Python 3.6 and GCC 7 compilers natively causes ABI incompatibilities, broken pip wheels, and missing OpenGL drivers.

### 1.2 Our Docker Desktop Architecture
To guarantee 100% reproducible cross-platform execution, we created `navrep-svnit:latest`:
- **Base OS:** Ubuntu Linux with Mesa 3D OpenGL 3.3 software rasterization (`LIBGL_ALWAYS_SOFTWARE=1`).
- **Virtual Framebuffer:** `Xvfb :99 -screen 0 1280x800x24` providing headless rendering for Pygame, Tkinter, and OpenCV.
- **Remote Desktop Server:** `x11vnc` bound to display `:99` on port `5900`.
- **Zero-Install Web Bridge:** `websockify` bridging port `5900` to port `6080` with `noVNC`, enabling instant interactive GUI control directly inside Google Chrome or Safari.

### 1.3 Step-by-Step Setup from Scratch

#### Step 1: Clone Repository and Switch to Branch `shriya`
```bash
git clone https://github.com/Sravanthi2727/AI_Robotics.git
cd AI_Robotics
git checkout shriya
git pull origin shriya
```

#### Step 2: Start the Docker Container
Using `docker-compose`:
```bash
docker compose up -d
```
*Alternatively, if running via standalone docker CLI:*
```bash
docker run -d --name navrep_desktop \
  -p 5900:5900 -p 6080:6080 \
  -v "$(pwd)":/workspace \
  navrep-svnit:latest
```

#### Step 3: Verify the Container Status and Port Mappings
```bash
docker ps -f name=navrep_desktop
```
*Expected Output:*
```text
CONTAINER ID   IMAGE                  STATUS         PORTS                                            NAMES
b3aca015685b   navrep-svnit:latest    Up X hours     0.0.0.0:5900->5900/tcp, 0.0.0.0:6080->6080/tcp   navrep_desktop
```

---

## 2. Task 1: Environment Reproduction & Model Verification

### 2.1 Smoke Test
Proves environment instantiation, step cycles, and reset routines across both `NavRepTrainEnv` (toy static arena) and `IANEnv` (dynamic human crowd environment):
```bash
docker exec navrep_desktop python scripts_custom/smoke_test.py
```
*Expected Terminal Output:*
```text
[SMOKE TEST] Initializing NavRepTrainEnv...
[SMOKE TEST] Stepping NavRepTrainEnv (50 steps)... OK
[SMOKE TEST] Initializing IANEnv (soadrl)...
[SMOKE TEST] Stepping IANEnv (50 steps)... OK
[SMOKE TEST] All environments stepped successfully. PASS.
```

### 2.2 Pretrained Model & Checksum Verification
Validates that all 10 neural network weights (Variational Autoencoders, LSTM world models, and PPO checkpoints) are authentic, uncorrupted, and match expected MD5 signatures:
```bash
docker exec navrep_desktop python scripts_custom/verify_models_and_checksums.py
```
*Expected Terminal Output:*
```text
[CHECK] Verifying 10 pretrained neural network weights...
[OK] VAE2D weights: MATCH
[OK] LSTM predictive models: MATCH
[OK] PPO_VAE_LSTM_V_ONLY: MATCH
[OK] PPO_VAE_LSTM_VM: MATCH
>>> ALL 10 MODELS VERIFIED AND MATCH CHECKSUMS <<<
```

---

## 3. Task 2: Existing Scenarios Benchmark & 22-Video Verification

### 3.1 Background & Policy Specifications
Task 2 benchmarks all **11 existing simulation scenarios** across all 3 environment families:
1. **Toy / Training Arena:** `navreptrain_train`, `navreptrain_test` (random box obstacles).
2. **IROS ASL (ETH Zurich Office Hallways):** `irosasl1`, `irosasl2`, `irosasl3`, `irosasl_office_j1`, `irosasl_office_j2`, `irosasl_office_j3`.
3. **IROS Unity (Photorealistic Architectural Floorplans):** `irosunity_scene_map1`, `irosunity_scene_map2`, `irosunity_scene_map3`.

**Model Policy:** Reinforcement Learning PPO trained with `VAE_LSTM` backend using Velocity+Measurement (`VM`) latent representations.

### 3.2 Automated Video Verification Suite (All 22 Videos PASS)
Validates frame rate (12 fps), duration ($\ge 8\text{ s}$), non-frozen frames ($\le 20$), and color-coded outcome badges (Green = Goal, Red = Collision, Orange = Timeout) across all 22 videos:
```bash
docker exec navrep_desktop python scripts_custom/verify_videos.py
```
*Expected Terminal Output:*
```text
========================================================================================
TASK 2 AUTOMATED VIDEO QUALITY VERIFICATION REPORT
========================================================================================
[1/22] 01_navreptrain_train_success.mp4        | 12.0 fps | 15.2s | PASS (GOAL REACHED)
[2/22] 02_navreptrain_train_failure.mp4        | 12.0 fps | 10.4s | PASS (COLLISION)
...
[21/22] 21_irosunity_scene_map3_success.mp4    | 12.0 fps | 18.0s | PASS (GOAL REACHED)
[22/22] 22_irosunity_scene_map3_failure.mp4    | 12.0 fps | 14.2s | PASS (TIMEOUT)
========================================================================================
>>> ALL 22 VIDEOS PASSED PROGRAMMATIC QUALITY VERIFICATION! <<<
```

### 3.3 Fast Metric Table Re-Compilation
To re-compile the quantitative CSV benchmark without running lengthy 60-seed simulations:
```bash
docker exec navrep_desktop python scripts_custom/remediate_and_record.py --compile-only
head -n 12 submission/task2_videos/metrics_existing.csv
```

### 3.4 Recording or Re-Running an Individual Scenario
To record a fresh episode with custom seed and camera overlays:
```bash
docker exec navrep_desktop python scripts_custom/record_scenario.py \
    --scenario irosasl1 --seed 42 --backend VAE_LSTM --encoding VM --fps 12 \
    --out submission/task2_videos/03_irosasl1_test.mp4
```

---

## 4. Task 3: Custom Robotics Navigation Scenarios

### 4.1 Architectural Philosophy: Hierarchical Navigation
In narrow, cluttered corridors, end-to-end Reinforcement Learning (PPO) suffers from **topological blindness**: it optimizes local gradients towards the Euclidean goal vector, driving repeatedly into concave bookshelf walls and entering deadlocks.

To solve this, we architected a **hybrid navigation framework**:
1. **Global Occupancy-Grid $A^*$ Path Planner:** Computes an optimal collision-free corridor skeleton with a **$0.48\text{ m}$ obstacle safety inflation margin** ($r_{\text{robot}} = 0.30\text{ m} + 0.18\text{ m}$ clearance).
2. **Predictive Dynamic Yielding (Time-to-Closest-Approach):** Calculates future pedestrian closest-approach distance:
   $$t_{\text{CPA}} = -\frac{\mathbf{r}_{\text{rel}} \cdot \mathbf{v}_{\text{rel}}}{\|\mathbf{v}_{\text{rel}}\|^2}$$
   If minimum predicted separation $d_{\text{CPA}} < 0.28\text{ m}$, the robot preemptively yields without creeping.
3. **Zero-Creep Emergency Stopping:** When obstacle distance $d < 0.16\text{ m}$, linear velocity is set strictly to $\mathbf{v} = [0, 0]$, eliminating legacy sliding collisions.

---

### 4.2 Scenario 1: Institutional Library Book Collection Robot

**Author:** Shriya Siddesh  
**Narrative:** An autonomous mobile service robot (AMR) collects books from narrow library book stacks and transports them across reading halls to dock at the Circulation & Returns Desk.

#### Specifications & Spatial Geometry:
- **Floor Area:** $20.0\text{ m} \times 12.0\text{ m} = 240.0\text{ m}^2$ ($X \in [-10, 10]\text{ m}, Y \in [-6, 6]\text{ m}$).
- **Start Pose:** $(-8.0, 3.5, 0.0\text{ rad})$ inside Stack Aisle 2.
- **Goal Pose:** $(5.5, -3.5)$ at the Circulation & Book Returns Desk.
- **Direct Separation:** $15.21\text{ m}$ (exceeds $8.0\text{ m}$ benchmark requirement).
- **Aisle Clearance:** $1.8\text{ m}$ clear span between bookshelves.
- **Furniture Elements:**
  - West Stacks: `STACK A1`, `STACK A2`, `STACK A3` ($5.0\text{ m} \times 0.8\text{ m}$).
  - East Stacks: `STACK B1`, `STACK B2`, `STACK B3` ($4.5\text{ m} \times 0.8\text{ m}$).
  - Reading Furniture: `Archive Reading Table`, `Reference Study Zone`, `Study Table A`, and `Group Study Table B`.
  - Architecture: Octagonal stone columns, `Circulation & Returns Counter`, drop-box slot, and `New Arrivals Rack`.
- **Dynamic Human Crowd:** Browsing reader ($0.40\text{ m/s}$), trolley-pushing librarian ($0.60\text{ m/s}$), walking students ($0.85\text{ m/s}$), and stationary researchers.

#### 20-Seed Quantitative Benchmark Results:
| Difficulty Tier | Active Humans | Crowd Density | Pedestrian Speed | Timeout Limit | Success Rate | Mean Time |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **EASY** | 5 | $0.60\times$ | $0.75\times$ | $450\text{ steps } (90\text{ s})$ | **100.0% (20/20)** | **22.1 s** |
| **MEDIUM** | 7 | $1.00\times$ | $1.00\times$ | $380\text{ steps } (76\text{ s})$ | **100.0% (20/20)** | **23.4 s** |
| **HARD** | 9 | $1.60\times$ | $1.25\times$ | $300\text{ steps } (60\text{ s})$ | **100.0% (20/20)** | **28.5 s** |

#### Automated Unit Test Suite (28 / 28 PASS):
```bash
docker exec navrep_desktop python -m unittest tests/test_library_scenario.py tests/test_custom_scenarios.py
```
*Expected Output:* `Ran 28 tests in 10.726s` $\to$ `OK`.

---

### 4.3 Scenario 2: Grand Hotel Room Service Delivery Robot

**Author:** Tanishka  
**Narrative:** A luxury hotel service AMR carries covered food trays (cloche) from the kitchen dispatch station down guest corridors to deliver room service to luxury suites while avoiding hotel guests with luggage and housekeeping carts.

#### Specifications & Spatial Geometry:
- **Floor Area:** $24.0\text{ m} \times 16.0\text{ m} = 384.0\text{ m}^2$ ($X \in [-12, 12]\text{ m}, Y \in [-8, 8]\text{ m}$).
- **Start Pose:** $(-10.0, 0.0, 0.0\text{ rad})$ at Kitchen Dispatch Station.
- **Goal Pose:** $(10.0, 0.0)$ at Guest Suite Corridor / Suite 204.
- **Direct Separation:** $20.0\text{ m}$.
- **Architectural Elements:**
  - North Suites: `Suite 201`, `Suite 202` (Executive), `Suite 203` (Presidential).
  - South Suites: `Suite 101`, `Suite 102`, `Suite 103` (Garden Villa).
  - Lobby & Facilities: Concierge / Reception Counter with brass bell, Sapphire Velvet Sofa Lounge, Glass Coffee Table, Housekeeping Linen Station.
- **Dynamic Crowd:**
  - Hotel guests walking along or crossing corridors with rolling suitcases ($r=0.50\text{ m}$, $0.65\text{ m/s}$).
  - Housekeeping staff pushing linen supply trolleys ($0.45\text{ m/s}$).

#### Headless Video Recording for Scenario 2:
```bash
docker exec navrep_desktop python scripts_custom/record_hotel_demo.py
```
*Generates verified MP4 recordings in `submission/task3_custom/videos/`:*
- `hotelroomservice_easy.mp4`
- `hotelroomservice_medium.mp4`
- `hotelroomservice_hard.mp4`

---

### 4.4 Other Custom Scenarios in the Suite
The codebase also includes 5 additional evaluated scenarios:
1. **MallMate:** Shopping mall customer assistance robot traversing atrium with pillars, kiosks, and darting children.
2. **MediRunner:** Hospital medicine delivery robot navigating corridors with emergency gurneys ($1.65\text{ m/s}$) and frail patients.
3. **KiranaKart:** Warehouse AGV traversing blind orthogonal racking aisles and crossing forklifts.
4. **Diwali Bazaar Surat:** Dense festive street market with intense pedestrian counter-flow and roadside kiosks.
5. **Platform 1 — Railway Station:** Train arrival passenger surge flowing across long platform towards station exits.
6. **Fire Drill:** Maximum counter-flow emergency building evacuation through structural bottlenecks.

To run parametric crowd density sweeps across any custom scenario:
```bash
docker exec navrep_desktop python -m navrep_custom.run_custom \
    --scenario mallmate --difficulty MEDIUM --episodes 5
```

---

## 5. Task 3: Live Graphical Demonstration Interface (VNC & Web Browser)

### 5.1 VNC & Browser Access Setup

#### Option A: Web Browser (Recommended — Zero Install)
Open your web browser (Chrome, Safari, Edge, Firefox) on the host machine and navigate to:
```text
http://localhost:6080/vnc_lite.html?scale=true
```

#### Option B: Native macOS Screen Sharing
In your macOS terminal, run:
```bash
open vnc://localhost:5900
```

---

### 5.2 Launching Scenario 1 (Library)
To launch the GUI directly into the **Institutional Library Book Collection** scenario:
```bash
docker exec -d -e DISPLAY=:99 navrep_desktop python /workspace/submission/task3_custom/gui/library_gui.py --scenario library
```

---

### 5.3 Launching Scenario 2 (Hotel Room Service)
To launch the GUI directly into the **Grand Hotel Room Service Delivery** scenario:
```bash
docker exec -d -e DISPLAY=:99 navrep_desktop python /workspace/submission/task3_custom/gui/library_gui.py --scenario hotelroomservice
```

---

### 5.4 Switching Scenarios Live in GUI
You do **not** need to restart the script to switch scenarios!  
In the top-right control panel under **"Active Robot Scenario"**, simply click:
- **`[1] Institutional Library Book Collection`** $\to$ Instantly loads the library floorplan, bookshelves, reading tables, and librarian/student agents.
- **`[2] Grand Hotel Room Service Delivery`** $\to$ Instantly loads the luxury hotel floorplan, guest suites, reception, and guest/housekeeping agents.

---

### 5.5 Interactive Controls & Dynamic A* Replanning Walkthrough

#### 1. Showcase Demonstration (`[>] RUN FULL DEMONSTRATION`)
Click the large green **`[>] RUN FULL DEMONSTRATION`** button:
1. **$A^*$ Wavefront Search:** An animated cyan wavefront expands from the robot's start pose, evaluating $24,000$ grid cells with safety inflation in $\sim 130\text{ ms}$, synthesizing an optimal collision-free corridor path.
2. **Autonomous Navigation:** The robot navigates along the green path. Approaching pedestrians trigger `PREDICTIVE YIELDING`: the robot computes $t_{\text{CPA}}$, brakes to a complete halt without creeping, lets the pedestrian pass, and resumes navigation until docking.

#### 2. Modular Operations
- **`Show Environment`:** Resets simulation and displays clean static floorplan and initial human positions.
- **`Run Path Planning`:** Runs global $A^*$ search and overlays corridor waypoints without moving the robot.
- **`Start Navigation`:** Dispatches the robot along the computed path.
- **`Pause / Resume`:** Pauses simulation execution mid-course.
- **`Reset All`:** Resets robot and pedestrians to initial poses.

#### 3. Interactive Goal Selection & Dynamic $A^*$ Replanning
To demonstrate that navigation is dynamic and not hardcoded:
1. Click **`[+] SELECT NEW GOAL`**.
   - The status bar updates to: *"Click a reachable location on the map to set a new destination"*.
   - Live crosshairs and coordinates appear under your cursor.
2. **Safety Rejection Test:**
   - Click inside **Stack A2** (bookshelf) or inside a study table.
   - The GUI immediately rejects the click with a red status warning:  
     `"Invalid Goal: Position inside obstacle or lacks 0.35m clearance"`.
3. **Dynamic Replanning:**
   - Click anywhere in open corridor space (or select a preset destination like `Mathematics & Stats Stack (A3)` or `Suite 202` from the dropdown).
   - Click **`REPLAN ROUTE`**.
   - The GUI dynamically runs $A^*$ from the robot's *current location* to the new goal.
   - Click **`Start Navigation`** $\to$ the robot follows the newly generated route to the new target!

#### 4. Simulation Pacing & Difficulty Selection
- **Difficulty:** Select between **`EASY`**, **`MEDIUM`**, or **`HARD`** to scale pedestrian density and speed.
- **Sim Pacing:**
  - `0.5x (Demo)`: Slow-motion view ideal for presentations and observing yielding behavior.
  - `1.0x (Real)`: Real-time simulation rate ($0.2\text{ s}$ per step).
  - `2.0x (Fast)`: Rapid playback.

#### 5. Lower-Right Panel: Real-Time Telemetry & 20-Seed Benchmark Runner
- **Live Telemetry:** Displays Step number, Simulation Time, Robot Pose $(x, y, \theta)$, Speed, Distance to Goal, Active Waypoint, and Controller State (`CRUISING`, `PREDICTIVE YIELDING`, `GOAL REACHED`).
- **`Run 20-Seed Benchmark`:** Spawns a background thread that executes 20 successive pseudorandom seeds (42–61) and tabulates quantitative success rates live.

---

## 6. Troubleshooting & Common Failure Remediation

### Problem 1: `websockify: [Errno 2] No such file or directory` on Mac Host
- **Cause:** Running `/usr/bin/websockify` on the Mac terminal instead of inside the Docker container.
- **Fix:** Access noVNC directly in your browser at `http://localhost:6080/vnc_lite.html?scale=true`. The container runs `websockify` internally on port 6080.

### Problem 2: Port 5900 or 6080 Already in Use
- **Cause:** Another local service or previous container holds port 5900 or 6080.
- **Fix:**
  ```bash
  # Check which process holds the port:
  lsof -i :5900
  lsof -i :6080
  # Stop conflicting containers:
  docker rm -f navrep_desktop
  docker compose up -d
  ```

### Problem 3: GUI Window Appears Off-Screen or Minimized in VNC
- **Fix:** Inside the noVNC browser window, click on the **Applications** menu at the top-left, or bring the window forward using the taskbar at the top.

---

## 7. Technical Defense & Viva Voce Q&A Cheat Sheet

### Q1: "Why did the pretrained PPO policy fail in your custom library scenario?"
> **Answer:**  
> The upstream PPO policy is a purely reactive local controller trained on open circular arenas with random boxes. It lacks topological memory. In our library, the direct Euclidean vector from Start to Goal passes straight through Bookshelf A2. PPO repeatedly drives into the shelf wall, enters a 6-step limit cycle, and times out. Furthermore, the default simulator had an unrasterized 14-meter grid that caused ghost collisions in open corridors.

### Q2: "Why did you use $A^*$ search + Predictive TCA instead of training a new RL model?"
> **Answer:**  
> In industrial mobile robotics, hierarchical navigation is the established standard. Global planners like $A^*$ excel at discrete topological routing through complex structural floor plans, while local controllers excel at dynamic obstacle avoidance. Training end-to-end RL for high-density constrained corridors requires millions of samples and frequently suffers from corner-clipping. Combining $A^*$ corridor skeletons with predictive TCA yielding achieves **100% deterministic safety** without retraining.

### Q3: "What is Time-to-Closest-Approach (TCA) and why is it superior to simple proximity braking?"
> **Answer:**  
> Proximity braking only reacts to Euclidean distance: if a person is close, brake. But when a student crosses rapidly ($0.85\text{ m/s}$) from a blind aisle, by the time proximity triggers, collision is unavoidable. TCA uses relative position $\mathbf{r}_{\text{rel}} = \mathbf{p}_h - \mathbf{p}_r$ and relative velocity $\mathbf{v}_{\text{rel}} = \mathbf{v}_h - \mathbf{v}_r$ to predict future minimum clearance ($t_{\text{CPA}} = -(\mathbf{r} \cdot \mathbf{v})/\|\mathbf{v}\|^2$). If $d_{\text{CPA}} < 0.28\text{ m}$, the robot yields preemptively before the crossing occurs.

### Q4: "How do you prevent dynamic goal selection from clipping corners?"
> **Answer:**  
> 1. Our occupancy grid applies an obstacle inflation safety radius of $0.48\text{ m}$ ($r_{\text{robot}} = 0.30\text{ m} + 0.18\text{ m}$).  
> 2. Rather than decimating waypoints into long diagonal chords that cut across corners, our replanner samples waypoints densely every $0.40\text{ m}$ directly along the verified $A^*$ occupancy path, binding trajectories strictly to collision-free corridor centers.

---

## 8. Submission Deliverables & File Manifest

| Deliverable | File Path | Status |
|---|---|---|
| **1. Complete Source Code ZIP** | `submission/navrep_modified_source.zip` | Rebuilt & verified (1.0 MB, includes all Task 1, 2, and 3 code) |
| **2. Evaluator Presentation Guide** | `submission/EVALUATOR_PRESENTATION_GUIDE.md` | Complete presentation pitch & defense script |
| **3. Project Reproduction Guide** | `submission/COMPLETE_PROJECT_RUN_AND_EVALUATION_GUIDE.md` | This document (end-to-end reproduction manual) |
| **4. Publication-Ready PDF** | `submission/SVNIT_CS359_EVALUATOR_PRESENTATION_GUIDE.pdf` | 6-page styled department report with screenshots |
| **5. Task 2 Benchmark Videos (22)** | `submission/task2_videos/*.mp4` | All 22 videos verified PASS via `verify_videos.py` |
| **6. Task 2 Quantitative CSV** | `submission/task2_videos/metrics_existing.csv` | Full 11-scenario evaluation metrics table |
| **7. Task 3 Library Videos** | `submission/task3_custom/videos/library_*.mp4` | EASY, MEDIUM, and HARD benchmark runs |
| **8. Task 3 Hotel Videos** | `submission/task3_custom/videos/hotelroomservice_*.mp4` | EASY, MEDIUM, and HARD benchmark runs |
| **9. Unit Test Suite (28 Tests)** | `tests/test_library_scenario.py`, `tests/test_custom_scenarios.py` | 28 / 28 PASS in 10.7s |
| **10. Git Parity** | Branch `shriya` (`9e50fee`) | Pushed to GitHub (`origin/shriya`), clean working tree |
