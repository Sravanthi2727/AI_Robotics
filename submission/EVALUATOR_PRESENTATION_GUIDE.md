# SVNIT CS359: AI for Robotics — Complete Evaluator Presentation & Live Demonstration Guide

**Institute:** Sardar Vallabhbhai National Institute of Technology, Surat  
**Department:** Computer Science & Engineering Department  
**Course:** CS359: Artificial Intelligence for Robotics (B.Tech CSE IIIrd Year)  
**Author / Presenter:** Shriya Siddesh  
**Git Branch:** `shriya` | **Latest Commit:** `02ef766` | **Total Test Suite:** 28 / 28 PASS  

---

## 0. Evaluator Presentation Strategy & Opening Pitch

### The 90-Second Opening Script (Spoken Directly to Sir)
> *"Good morning / afternoon, Sir.*  
> *In this project, we explored and extended the ETH Zurich **NavRep** framework for autonomous mobile robot navigation across three progressive tasks:*
> 1. *In **Task 1**, we containerized the entire legacy Python 3.6 / TensorFlow 1.14 / ROS stack using Docker and Xvfb virtual displays to achieve 100% reproducible cross-platform execution on modern macOS and Linux without dependency conflicts.*
> 2. *In **Task 2**, we benchmarked all 11 existing ETH Zurich simulation scenarios across 20-seed sweeps, diagnosed upstream rendering and policy limitations, and recorded 22 verified demonstration videos with color-coded HUD outcome badges.*
> 3. *In **Task 3**, I designed and implemented a realistic, safety-critical real-world scenario: the **Library Book Collection Robot**. The robot navigates a $20\text{ m} \times 12\text{ m}$ institutional library with narrow $1.8\text{ m}$ stack aisles, blind occluded corners, and moving human patrons. To solve the policy limit-cycles inherent to narrow corridors, I built a hybrid architecture combining **global occupancy-grid $A^*$ path planning** with **dynamic predictive collision avoidance (Time-to-Closest-Approach yielding)**, backed by a dedicated **graphical demonstration interface** with **live interactive goal selection and dynamic $A^*$ replanning**.*
> 
> *Sir, allow me to walk you through each task with live output proof and terminal execution."*

---

## 1. Task 1: Repository Setup & Execution

### 1.1 What Was Done & Why
- **The Challenge:** Upstream ETH Zurich `navrep` relies on Python 3.6, TensorFlow 1.14, OpenAI Gym 0.10, PyTorch 1.1, and custom C++ extensions (`crings`). On modern operating systems (Ubuntu 22.04+ or Apple Silicon macOS), installing these obsolete C-libraries natively causes severe compiler and ABI failures.
- **The Engineering Solution:** We containerized the complete stack using a standalone Docker architecture (`navrep-svnit:latest`), provisioning:
  - Ubuntu with Mesa 3D OpenGL 3.3 software rasterization (`LIBGL_ALWAYS_SOFTWARE=1`).
  - An isolated virtual framebuffer X-server (`Xvfb :99 -screen 0 1280x800x24`).
  - A lightweight XFCE4 desktop session with an `x11vnc` server on port 5900 and a `noVNC` WebSocket bridge on port 6080 for instant zero-install browser access.
- **Deliverable:** Comprehensive `submission/README_TASK1.md` (700+ lines covering OS details, dependency matrix, code modifications, and full troubleshooting log).

### 1.2 Exact Commands to Run in Front of Evaluator

#### Step 1: Prove the Docker Environment is Live
Show that the container runs with all necessary ports bound (5900 and 6080):
```bash
docker ps -f name=navrep_desktop
```
*Expected Output:* Container `navrep_desktop` status `Up`, with ports `0.0.0.0:5900->5900` and `0.0.0.0:6080->6080`.

#### Step 2: Run the Smoke Test
Proves environment instantiation and step/reset cycles across `NavRepTrainEnv` and `IANEnv`:
```bash
docker exec navrep_desktop python scripts_custom/smoke_test.py
```
*Expected Output:* `[SMOKE TEST] All environments stepped successfully. PASS`.

#### Step 3: Verify Pretrained Models & MD5 Checksums
Proves that all 10 neural network weights (VAE, LSTM, PPO checkpoints) are authentic and uncorrupted:
```bash
docker exec navrep_desktop python scripts_custom/verify_models_and_checksums.py
```
*Expected Output:* `ALL 10 MODELS VERIFIED AND MATCH CHECKSUMS`.

---

## 2. Task 2: Testing Existing Simulation Scenarios

### 2.1 What Was Done & Key Findings
- **Scope:** Tested all 11 existing scenarios across all 3 environment families:
  1. **Toy/Training:** `navreptrain_train`, `navreptrain_test` (random static box obstacles).
  2. **IROS ASL (ETH Hallways):** `irosasl1`, `irosasl2`, `irosasl3`, `irosasl_office_j1`, `irosasl_office_j2`, `irosasl_office_j3`.
  3. **IROS Unity (Office Floorplans):** `irosunity_scene_map1`, `irosunity_scene_map2`, `irosunity_scene_map3`.
- **Engineering Fixes Applied:**
  - Resolved headless Pygame/OpenGL buffer lockups in `IANEnv` during video frame capture.
  - Deployed the correct, high-capacity `VAE_LSTM` model with Velocity+Measurement (`VM`) latent encoding (`navreptrainencodedenv_..._PPO_VAE_LSTM_V_ONLY_V32M512_ckpt.zip`).
- **Deliverable:**
  - `submission/README_TASK2.md` and `submission/task2_videos/metrics_existing.csv`.
  - 22 MP4 simulation videos (1 success/best-progress + 1 failure for every scenario) equipped with color-coded HUD status banners:
    - **GREEN (GOAL REACHED)**: Successful navigation.
    - **RED (COLLISION)**: Collision with obstacle or pedestrian.
    - **ORANGE (TIMEOUT)**: Trajectory exceeded step budget.

### 2.2 Live Output Proof to Show Evaluator

Run the automated video verification suite live in the terminal:
```bash
docker exec navrep_desktop python scripts_custom/verify_videos.py
```
*Live Output:* A programmatic table evaluates all 22 videos against duration ($\ge 8\text{ s}$), frame rates (12 fps), frozen frame counts ($\le 20$), and HUD banner color consistency, concluding with:  
`>>> ALL 22 VIDEOS PASSED PROGRAMMATIC QUALITY VERIFICATION! <<<`

To show the quantitative CSV benchmark table:
```bash
head -n 12 submission/task2_videos/metrics_existing.csv
```

---

## 3. Task 3: Custom Scenario — "Library Book Collection Robot"

### 3.1 Scenario Design & Real-World Motivation
- **Narrative:** An autonomous mobile service robot (AMR) retrieves books from stack aisles in an institutional library and navigates through reading halls to dock at the Circulation & Book Returns Desk.
- **Physical Dimensions:**
  - Floor Area: $20.0\text{ m} \times 12.0\text{ m} = 240.0\text{ m}^2$ ($X \in [-10, 10]\text{ m}, Y \in [-6, 6]\text{ m}$).
  - Start Pose: $(-8.0, 3.5, 0.0\text{ rad})$ in Stack Aisle 2.
  - Goal Pose: $(5.5, -3.5)$ at Circulation Counter.
  - Direct Euclidean Distance: $d = 15.21\text{ m}$ (exceeds the required $8.0\text{ m}$ benchmark separation).
- **Static Obstacles (100% Free-Space Verified):**
  - West Stacks A1, A2, A3 ($5.0\text{ m} \times 0.8\text{ m}$ bookshelves forming narrow $1.8\text{ m}$ aisles).
  - East Archives B1, B2, B3 ($4.5\text{ m} \times 0.8\text{ m}$ reference racks).
  - Study Reading Tables ($3.6\text{ m} \times 1.6\text{ m}$) & 8-sided structural columns.
  - Circulation Desk ($1.2\text{ m} \times 3.0\text{ m}$) in South-East foyer.
- **Dynamic Pedestrian Crowd (ORCA Simulation):**
  - Browsing reader ($0.40\text{ m/s}$ along shelf), Librarian pushing a trolley ($0.60\text{ m/s}$ down cross-corridor), Walking student ($0.85\text{ m/s}$ crossing east), Study room researcher, and Circulation attendant.

### 3.2 Root-Cause Failure Analysis & Engineering Solutions

| # | Diagnosed Failure Symptom | Mechanical Cause in Simulator | Shriya's Architectural Fix |
|---|---|---|---|
| **1** | Premature 177-step Timeout | `CrowdSim.py` hardcoded `time_limit = 35s` ($176 \times 0.2\text{ s}$) | `sync_and_rasterize_sim()` overrides `sim.time_limit = raw_env.time_limit * dt`. |
| **2** | Ghost Obstacle Collisions | `soadrl_sim` had a default unrasterized $14\text{ m}$ grid with 3,418 stale obstacle cells | Synchronized $24\text{ m}$ rasterizer using exact metric transpose: `sim.map = img.T`. |
| **3** | PPO Oscillatory Deadlock | PPO local gradient policy blind to topological corners of Shelf A2 | Built global occupancy-grid $A^*$ path planner to synthesize collision-free corridor skeleton. |
| **4** | Creeping Collisions | Heuristic braking had a $5\%$ velocity floor (`brake = max(0.05, ...)`), creeping at $8\text{ mm/step}$ into humans | Implemented absolute zero-creep emergency stopping ($\mathbf{v} = [0, 0]$) when boundary distance $d_{\text{bound}} < 0.16\text{ m}$. |
| **5** | High-Speed Cross Collisions | Reactive distance checks had no lookahead for crossing students ($0.85\text{ m/s}$) | Formulated predictive Time-to-Closest-Approach ($t_{\text{CPA}}$) and Closest Point of Approach ($d_{\text{CPA}}$) yielding. |
| **6** | Human Corridor Blocking | Idle roaming humans blocked the $3.5\text{ m}$ central corridor | Smooth orthogonal detours ($\pm 0.70\text{ m}$) into open foyer space. |

### 3.3 Difficulty Ladder & Quantitative Benchmark Results

Evaluated across **20 identical pseudorandom seeds (42–61)** per difficulty tier:

| Difficulty Tier | Active Humans | Crowd Density | Pedestrian Speed | Timeout Limit | Success Rate | Mean Time (Success) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **EASY** | 5 | $0.60\times$ | $0.75\times$ | $450\text{ steps } (90\text{ s})$ | **100.0% (20/20)** | **22.1 s** |
| **MEDIUM** | 7 | $1.00\times$ | $1.00\times$ | $380\text{ steps } (76\text{ s})$ | **100.0% (20/20)** | **23.4 s** |
| **HARD** | 9 | $1.60\times$ | $1.25\times$ | $300\text{ steps } (60\text{ s})$ | **100.0% (20/20)** | **28.5 s** |

### 3.4 Live Unit Test Suite Execution (28 / 28 PASS)

Run both test suites in front of the evaluator:
```bash
docker exec navrep_desktop python -m unittest tests/test_library_scenario.py tests/test_custom_scenarios.py
```
*Expected Result:* `Ran 28 tests in 10.68s` $\to$ `OK`.

---

## 4. The Live VNC Demonstration Walkthrough

### 4.1 How to Open the Interface
In Google Chrome or Safari on your Mac, open:
```text
http://localhost:6080/vnc_lite.html?scale=true
```
*(Alternatively, from your Mac terminal run `open vnc://localhost:5900` to launch native macOS Screen Sharing).*

### 4.2 Interactive Walkthrough Script (Step-by-Step for Sir)

#### Step 1: Explain the Realistic Library Environment
> *"Sir, notice that rather than a generic robotics grid, the simulation environment visually models an academic library while strictly adhering to real obstacle footprints:*
> - *Light oak parquet hardwood flooring with directional wood planks.*
> - *Dark walnut bookshelves with individual, multi-colored book spines.*
> - *Dedicated library zones: **STACK A1**, **A2**, **A3**, **Archive Reading Table**, **Reference Study Zone**, and **Group Study Table B**.*
> - *Architectural stone columns with central medallions, and the official **Circulation & Returns Counter** with drop-slot and new-arrival racks.*
> - *The blue AMR robot with LiDAR heading indicator, and color-coded human avatars."*

#### Step 2: Run the Full Demonstration (`RUN FULL DEMONSTRATION`)
Click the green **`[>] RUN FULL DEMONSTRATION`** button:
1. **Watch the $A^*$ Wavefront:**  
   The cyan explored wavefront rapidly propagates from Start $(-8.0, 3.5)$ through the cross-corridor to the Circulation Counter $(5.5, -3.5)$.
   > *"Sir, the $A^*$ search evaluates $24,000$ occupancy grid cells with a safety inflation radius of $0.48\text{ m}$. In $131\text{ ms}$, it explores $5,110$ cells and synthesizes a $17.19\text{ m}$ collision-free corridor skeleton."*
2. **Watch the Autonomous Navigation:**  
   The robot departs Aisle 2 along the green path.
   > *"Notice how the robot stays centered in the $1.8\text{ m}$ aisle. When approaching human patrons, the telemetry state switches to `PREDICTIVE YIELDING`. The robot calculates Time-to-Closest-Approach, brakes to a complete halt without creeping, allows the pedestrian to pass, and smoothly resumes until docking at the Circulation Desk."*

#### Step 3: Interactive Goal Selection & Dynamic $A^*$ Replanning
> *"Sir, to prove that this is not a hardcoded trajectory, you can choose any new destination in the library right now."*

1. Click **`[+] SELECT NEW GOAL`**.
   - The status banner displays: *"Click a reachable location on the map to set a new destination"*.
   - Live crosshairs and coordinates hover under the cursor.
2. **Demonstrate Safety Validation:**
   - Click inside **Stack A2** or inside a table.
   - Show the red warning: `"Invalid Goal: Position inside obstacle or lacks 0.35m clearance"`. The system refuses unsafe coordinates!
3. **Set a New Goal & Replan:**
   - Click anywhere in open space (or select **`Mathematics & Statistics Stack (A3)`** or **`Archive Reading Table`** from the Preset dropdown).
   - Click **`REPLAN ROUTE`**.
   - Watch the animated $A^*$ search compute a brand-new path from the robot's *current location* to the new target.
   - Click **`Start Navigation`** $\to$ the robot follows the newly generated waypoints and successfully docks at the newly chosen destination!

#### Step 4: Show the 20-Seed Benchmark Table
Point to the lower-right panel:
- Click **`Run 20-Seed Benchmark`**:
  Executes in a background thread and demonstrates verified $100\%$ success across EASY, MEDIUM, and HARD tiers.

---

## 5. Defense & Viva Voce Q&A Cheat Sheet

### Q1: "Why did the pretrained PPO policy fail in your custom library scenario?"
> **Answer:**  
> *"The upstream PPO policy is a reactive local controller trained on open circular arenas with random boxes. It lacks topological memory. In our library, the direct Euclidean vector to the goal points through Shelf A2. PPO repeatedly drives into the shelf wall, repulses back, and enters an infinite 6-step limit cycle. Furthermore, the default CrowdSim environment had an unrasterized 14-meter grid map that caused phantom collisions in free space."*

### Q2: "Why did you combine $A^*$ search with local predictive control instead of training a new RL model?"
> **Answer:**  
> *"In real-world mobile robotics, hierarchical navigation is the industry standard. Global planners like $A^*$ excel at discrete topological routing through complex structural floor plans, while local controllers excel at dynamic obstacle avoidance. Training pure end-to-end RL for high-density constrained corridors is sample-inefficient and prone to corner clipping. By feeding $A^*$ corridor skeletons into a predictive TCA controller, we achieve 100% deterministic safety without retraining."*

### Q3: "What is Time-to-Closest-Approach (TCA) and why is it better than simple proximity braking?"
> **Answer:**  
> *"Proximity braking only measures Euclidean distance: if a person is close, brake. But this fails when a student crosses rapidly ($0.85\text{ m/s}$) from the side — by the time distance falls below the threshold, collision is inevitable. TCA uses relative position $\mathbf{r}_{\text{rel}} = \mathbf{p}_h - \mathbf{p}_r$ and relative velocity $\mathbf{v}_{\text{rel}} = \mathbf{v}_h - \mathbf{v}_r$ to predict the exact time and distance of closest approach ($t_{\text{CPA}} = -(\mathbf{r} \cdot \mathbf{v})/\|\mathbf{v}\|^2$). If minimum clearance at that time falls under $0.28\text{ m}$, the robot yields preemptively before the crossing trajectory occurs."*

### Q4: "How do you guarantee that dynamic goal selection doesn't clip shelf corners?"
> **Answer:**  
> *"First, our occupancy grid applies an obstacle inflation radius of $0.48\text{ m}$ (robot radius $0.30\text{ m}$ + safety margin $0.18\text{ m}$). Second, instead of decimating waypoints into long chords that cut corners diagonally, dynamic replanning samples waypoints densely every $0.40\text{ m}$ directly along the verified $A^*$ occupancy path with a waypoint reach tolerance of $0.35\text{ m}$. This strictly binds the robot trajectory to collision-free corridor centers."*

---

## 6. Submission Deliverables Summary Table

| Deliverable Required | Exact Submission File Path | Verification Status |
|---|---|---|
| **1. Modified Source Code in ZIP** | `submission/navrep_modified_source.zip` | **Rebuilt & Verified** (1.0 MB, includes all Task 1, 2, and 3 code) |
| **2. Scenario Documentation** | `submission/task3_custom/LIBRARY_BOOK_COLLECTION.md` | **Complete** (463 lines, full engineering report & math specs) |
| | `submission/task3_custom/SCENARIO_DESCRIPTIONS.md` | **Complete** (308 lines, multi-scenario specifications) |
| **3. Simulation Demonstration Videos** | `submission/task3_custom/videos/library_easy.mp4` | **Verified** (640x640 @ 12 fps, 21.5s, clean goal reach) |
| | `submission/task3_custom/videos/library_medium.mp4` | **Verified** (640x640 @ 12 fps, 44.7s, predictive yielding) |
| | `submission/task3_custom/videos/library_hard.mp4` | **Verified** (640x640 @ 12 fps, 12.8s, dense crowd avoidance) |
| **4. Task 2 Videos (22 Videos)** | `submission/task2_videos/*.mp4` | **All 22 PASS** via `scripts_custom/verify_videos.py` |
| **5. Git Repository & Checkpoint** | Branch `shriya` at Commit `02ef766` | **Pushed to GitHub** (`origin/shriya`), clean working tree |

### Final Closing Line for Your Evaluator
> *"Sir, all modified code is committed on branch `shriya` (commit `02ef766`), pushed to GitHub, packaged in `navrep_modified_source.zip`, and verified with 28 passing unit tests and a live demonstration interface. Thank you, and I am happy to answer any further questions."*

