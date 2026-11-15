# SVNIT CS359: AI for Robotics — Complete Evaluator Presentation & Live Demonstration Guide

**Institute:** Sardar Vallabhbhai National Institute of Technology (SVNIT), Surat  
**Department:** Computer Science & Engineering Department  
**Course:** CS359: Artificial Intelligence for Robotics (B.Tech CSE IIIrd Year)  
**Author / Presenter:** Shriya Siddesh  
**Git Branch:** `shriya` | **Latest Verified Commit:** `9e50fee` | **Total Test Suite:** 28 / 28 PASS  

---

## 0. Evaluator Presentation Strategy & Opening Pitch

### The 90-Second Opening Script (Spoken Directly to Sir)
> *"Good morning / afternoon, Sir.*  
> *In this project, we explored and extended the ETH Zurich **NavRep** framework for autonomous mobile robot navigation across three progressive tasks:*
> 1. *In **Task 1**, we containerized the entire legacy Python 3.6 / TensorFlow 1.14 / ROS stack using Docker, Mesa 3D OpenGL 3.3 software rendering, and Xvfb virtual displays to achieve 100% reproducible cross-platform execution on modern macOS and Linux without dependency conflicts.*
> 2. *In **Task 2**, we benchmarked all 11 existing ETH Zurich simulation scenarios across 20-seed sweeps, diagnosed upstream rendering and policy limitations, and recorded 22 verified demonstration videos with color-coded HUD outcome badges.*
> 3. *In **Task 3**, we designed and implemented practical, safety-critical real-world scenarios: the primary **Library Book Collection Robot** (navigating narrow $1.8\text{ m}$ stack aisles and reading halls) and the integrated **Grand Hotel Room Service Delivery Robot** (navigating luxury guest corridors and avoiding luggage carts and housekeeping staff). To solve the policy limit-cycles inherent to narrow corridors, I built a hybrid architecture combining **global occupancy-grid $A^*$ path planning** with **dynamic predictive collision avoidance (Time-to-Closest-Approach yielding)**, backed by a dedicated **graphical demonstration interface** with **live interactive goal selection, dynamic $A^*$ replanning, and instant dual-scenario switching**.*
> 
> *Sir, allow me to walk you through each task with live output proof and terminal execution."*

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

## 1. Task 1: Repository Setup & Execution

### 1.1 What Was Done & Why
- **The Challenge:** Upstream ETH Zurich `navrep` relies on Python 3.6, TensorFlow 1.14, OpenAI Gym 0.10, PyTorch 1.1, and custom C++ extensions (`crings`). On modern operating systems (Ubuntu 22.04+ or Apple Silicon macOS), installing these obsolete C-libraries natively causes severe compiler and ABI failures.
- **The Engineering Solution:** We containerized the complete stack using a standalone Docker architecture (`navrep-svnit:latest`), provisioning:
  - Ubuntu with Mesa 3D OpenGL 3.3 software rasterization (`LIBGL_ALWAYS_SOFTWARE=1`).
  - An isolated virtual framebuffer X-server (`Xvfb :99 -screen 0 1280x800x24`).
  - A lightweight desktop session with an `x11vnc` server on port 5900 and a `noVNC` WebSocket bridge on port 6080 for instant zero-install browser access.
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
docker exec navrep_desktop python scripts_custom/remediate_and_record.py --compile-only
head -n 12 submission/task2_videos/metrics_existing.csv
```

---

## 3. Task 3: Custom Robotics Navigation Scenarios

### 3.1 Scenario 1: "Library Book Collection Robot" (Primary Scenario — Shriya)
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

#### Root-Cause Failure Analysis & Engineering Solutions:
| # | Diagnosed Failure Symptom | Mechanical Cause in Simulator | Architectural Fix |
|---|---|---|---|
| **1** | Premature 177-step Timeout | `CrowdSim.py` hardcoded `time_limit = 35s` ($176 \times 0.2\text{ s}$) | `sync_and_rasterize_sim()` overrides `sim.time_limit = raw_env.time_limit * dt`. |
| **2** | Ghost Obstacle Collisions | `soadrl_sim` had a default unrasterized $14\text{ m}$ grid with 3,418 stale obstacle cells | Synchronized $24\text{ m}$ rasterizer using exact metric transpose: `sim.map = img.T`. |
| **3** | PPO Oscillatory Deadlock | PPO local gradient policy blind to topological corners of Shelf A2 | Built global occupancy-grid $A^*$ path planner to synthesize collision-free corridor skeleton. |
| **4** | Creeping Collisions | Heuristic braking had a $5\%$ velocity floor (`brake = max(0.05, ...)`), creeping at $8\text{ mm/step}$ into humans | Implemented absolute zero-creep emergency stopping ($\mathbf{v} = [0, 0]$) when boundary distance $d_{\text{bound}} < 0.16\text{ m}$. |
| **5** | High-Speed Cross Collisions | Reactive distance checks had no lookahead for crossing students ($0.85\text{ m/s}$) | Formulated predictive Time-to-Closest-Approach ($t_{\text{CPA}}$) and Closest Point of Approach ($d_{\text{CPA}}$) yielding. |
| **6** | Human Corridor Blocking | Idle roaming humans blocked the $3.5\text{ m}$ central corridor | Smooth orthogonal detours ($\pm 0.70\text{ m}$) into open foyer space. |

#### Difficulty Ladder & Quantitative Benchmark Results (20 Seeds: 42–61):
| Difficulty Tier | Active Humans | Crowd Density | Pedestrian Speed | Timeout Limit | Success Rate | Mean Time (Success) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **EASY** | 5 | $0.60\times$ | $0.75\times$ | $450\text{ steps } (90\text{ s})$ | **100.0% (20/20)** | **22.1 s** |
| **MEDIUM** | 7 | $1.00\times$ | $1.00\times$ | $380\text{ steps } (76\text{ s})$ | **100.0% (20/20)** | **23.4 s** |
| **HARD** | 9 | $1.60\times$ | $1.25\times$ | $300\text{ steps } (60\text{ s})$ | **100.0% (20/20)** | **28.5 s** |

#### Live Unit Test Suite Execution (28 / 28 PASS):
```bash
docker exec navrep_desktop python -m unittest tests/test_library_scenario.py tests/test_custom_scenarios.py
```
*Expected Result:* `Ran 28 tests in 10.726s` $\to$ `OK`.

---

### 3.2 Scenario 2: "Grand Hotel Room Service Delivery Robot" (Integrated Scenario — Tanishka)
- **Narrative:** An autonomous cloche service robot delivers orders from the kitchen service station down guest corridors to executive and presidential suites while navigating around hotel guests and housekeeping carts.
- **Physical Dimensions:**
  - Floor Area: $24.0\text{ m} \times 16.0\text{ m} = 384.0\text{ m}^2$ ($X \in [-12, 12]\text{ m}, Y \in [-8, 8]\text{ m}$).
  - Start Pose: $(-10.0, 0.0, 0.0\text{ rad})$ at Kitchen Dispatch.
  - Goal Pose: $(10.0, 0.0)$ at Suite 204.
  - Direct Separation: $20.0\text{ m}$.
- **Architectural Elements:**
  - Executive Suites 201–203 (North), Deluxe Suites 101–103 (South).
  - Reception Desk with call bell, Velvet Lounge, Coffee tables, Housekeeping linen bays.
- **Dynamic Crowd:** Guests with rolling luggage ($0.65\text{ m/s}$), Housekeeping carts ($0.45\text{ m/s}$).
- **Headless Video Generation:**
  ```bash
  docker exec navrep_desktop python scripts_custom/record_hotel_demo.py
  ```

---

## 4. The Live Graphical VNC Demonstration Walkthrough

### 4.1 How to Open the Interface
In Google Chrome or Safari on your Mac, open:
```text
http://localhost:6080/vnc_lite.html?scale=true
```
*(Alternatively, run `open vnc://localhost:5900` from your host terminal).*

### 4.2 Launching the GUI from Host Terminal
- **To Launch Library Scenario Directly:**
  ```bash
  docker exec -d -e DISPLAY=:99 navrep_desktop python /workspace/submission/task3_custom/gui/library_gui.py --scenario library
  ```
- **To Launch Hotel Room Service Directly:**
  ```bash
  docker exec -d -e DISPLAY=:99 navrep_desktop python /workspace/submission/task3_custom/gui/library_gui.py --scenario hotelroomservice
  ```

### 4.3 Interactive Walkthrough Script (Step-by-Step for Evaluator)

#### Step 1: Explain the Realistic Environment & Scenario Switcher
> *"Sir, notice the top-right panel features an **Active Robot Scenario** toggle. With one click, we can switch between **[1] Institutional Library Book Collection** and **[2] Grand Hotel Room Service Delivery**. The canvas renders realistic floorplans: parquet wood for the library, sapphire carpet and mahogany doors for the hotel."*

#### Step 2: Run the Full Demonstration (`RUN FULL DEMONSTRATION`)
Click the green **`[>] RUN FULL DEMONSTRATION`** button:
1. **Watch the $A^*$ Wavefront:**  
   The cyan wavefront rapidly evaluates $24,000$ cells with $0.48\text{ m}$ safety inflation in $131\text{ ms}$, synthesizing an optimal collision-free corridor skeleton.
2. **Watch the Autonomous Navigation:**  
   The robot departs along the green path. Approaching human patrons switches telemetry to `PREDICTIVE YIELDING`. The robot calculates Time-to-Closest-Approach, brakes to a complete halt without creeping, lets the pedestrian pass, and smoothly resumes until docking.

#### Step 3: Interactive Goal Selection & Dynamic $A^*$ Replanning
> *"Sir, to prove this is not a hardcoded trajectory, you can choose any new destination right now."*

1. Click **`[+] SELECT NEW GOAL`**. Live coordinates and crosshairs hover under your cursor.
2. **Demonstrate Safety Rejection:** Click inside **Stack A2** or inside a table. Show the red error: `"Invalid Goal: Position inside obstacle or lacks 0.35m clearance"`. The safety layer refuses unsafe points!
3. **Set New Goal & Replan:** Click anywhere in free corridor space (or select **`Mathematics & Statistics Stack (A3)`** or **`Suite 202`** from the Preset menu). Click **`REPLAN ROUTE`**. Watch the $A^*$ search synthesize a brand-new route from the robot's *current location*!
4. Click **`Start Navigation`** $\to$ the robot follows the newly generated route to the new target.

#### Step 4: Show the 20-Seed Benchmark Table
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
| **2. Evaluator Presentation Guide** | `submission/EVALUATOR_PRESENTATION_GUIDE.md` | **Complete** (Full presentation pitch & defense script) |
| **3. Project Reproduction Guide** | `submission/COMPLETE_PROJECT_RUN_AND_EVALUATION_GUIDE.md` | **Complete** (End-to-end beginner reproduction manual) |
| **4. Publication-Ready PDF** | `submission/SVNIT_CS359_EVALUATOR_PRESENTATION_GUIDE.pdf` | **Generated** (Publication-styled department document) |
| **5. Scenario Documentation** | `submission/task3_custom/LIBRARY_BOOK_COLLECTION.md` | **Complete** (463 lines, full engineering report & math specs) |
| | `submission/task3_custom/SCENARIO_DESCRIPTIONS.md` | **Complete** (308 lines, multi-scenario specifications) |
| **6. Task 3 Library Videos** | `submission/task3_custom/videos/library_easy.mp4` | **Verified** (640x640 @ 12 fps, 21.5s, clean goal reach) |
| | `submission/task3_custom/videos/library_medium.mp4` | **Verified** (640x640 @ 12 fps, 44.7s, predictive yielding) |
| | `submission/task3_custom/videos/library_hard.mp4` | **Verified** (640x640 @ 12 fps, 12.8s, dense crowd avoidance) |
| **7. Task 3 Hotel Videos** | `submission/task3_custom/videos/hotelroomservice_easy.mp4` | **Verified** (EASY tier room service delivery) |
| | `submission/task3_custom/videos/hotelroomservice_medium.mp4` | **Verified** (MEDIUM tier room service delivery) |
| | `submission/task3_custom/videos/hotelroomservice_hard.mp4` | **Verified** (HARD tier room service delivery) |
| **8. Task 2 Videos (22 Videos)** | `submission/task2_videos/*.mp4` | **All 22 PASS** via `scripts_custom/verify_videos.py` |
| **9. Git Repository & Checkpoint** | Branch `shriya` at Commit `9e50fee` | **Pushed to GitHub** (`origin/shriya`), clean working tree |

### Final Closing Line for Your Evaluator
> *"Sir, all modified code is committed on branch `shriya` (commit `9e50fee`), pushed to GitHub, packaged in `navrep_modified_source.zip`, and verified with 28 passing unit tests, full quantitative benchmarks, and a live demonstration interface. Thank you, and I am happy to answer any further questions."*
