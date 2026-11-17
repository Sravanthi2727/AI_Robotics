# NavRep: Autonomous Mobile Robot Navigation and Simulation

**CS359: Artificial Intelligence for Robotics**  
Department of Computer Science & Engineering  
Sardar Vallabhbhai National Institute of Technology (SVNIT), Surat  
B.Tech. Computer Science & Engineering, IIIrd Year (Academic Year 2026–2027)  
**Submission Deadline:** 15 November 2026  

---

**Original Framework:** [ETH Zürich Autonomous Systems Lab (ASL) NavRep](https://github.com/ethz-asl/navrep)  
**Group Repository:** [https://github.com/Sravanthi2727/AI_Robotics](https://github.com/Sravanthi2727/AI_Robotics)  
**Active Working Branch:** `shriya`  

---

## Custom Scenario Gallery

Our team designed, benchmarked, and recorded four distinct, application-oriented service robotics scenarios extending ETH Zürich's NavRep simulation platform. Click any image or title below to jump directly to its complete architectural specification:

| [1. Library Book Collection Robot](#81-scenario-1-institutional-library-book-collection-robot-shriya) | [2. Hospital Medicine Delivery Robot](#82-scenario-2-medirunner-hospital-medicine-delivery-robot-shreya) |
| :---: | :---: |
| [![Library Book Collection Robot](docs/images/library_book_collection.png)](docs/images/library_book_collection.png) | [![Hospital Medicine Delivery Robot](docs/images/hospital_medicine_delivery.png)](docs/images/hospital_medicine_delivery.png) |
| **Lead:** Shriya Siddesh <br> *Narrow $1.8\text{ m}$ stack aisles, reading halls, study tables, global $A^*$ search & predictive TCA yielding.* | **Lead:** Shreya <br> *Clinical hallways, triage bottleneck, recessed patient alcoves & emergency gurney yielding ($1.65\text{ m/s}$).* |
| [3. Hotel Room Service Delivery Robot](#83-scenario-3-grand-hotel-room-service-delivery-robot-tanishka) | [4. Airport Passenger Assistance Robot](#84-scenario-4-airport-passenger-assistance-robot-ritu-and-sravanthi) |
| [![Hotel Room Service Delivery Robot](docs/images/hotel_room_service.png)](docs/images/hotel_room_service.png) | [![Airport Passenger Assistance Robot](docs/images/airport_passenger_assistance.png)](docs/images/airport_passenger_assistance.png) |
| **Lead:** Tanishka Kumthekar <br> *Luxury executive suites, marble foyers, cloche delivery AMR, rolling luggage & housekeeping carts.* | **Leads:** Ritu & Sravanthi <br> *Terminal concourse, ticketing islands, TSA security lanes to Gate 12 with mission-control telemetry.* |

---

## Table of Contents

1. [Team Members and Contributions](#1-team-members-and-contributions)
2. [Project Overview & Theoretical Background](#2-project-overview--theoretical-background)
3. [System Architecture & Containerization (Task 1)](#3-system-architecture--containerization-task-1)
4. [From-Scratch Setup Guide (Windows 11 & Cross-Platform)](#4-from-scratch-setup-guide-windows-11--cross-platform)
5. [Automated Verification Commands](#5-automated-verification-commands)
6. [Task 2: Testing Existing NavRep Scenarios](#6-task-2-testing-existing-navrep-scenarios)
7. [Complete Task 2 22-Video Index](#7-complete-task-2-22-video-index)
8. [Task 3: Custom Robotics Navigation Scenarios](#8-task-3-custom-robotics-navigation-scenarios)
   - [8.1 Scenario 1: Institutional Library Book Collection Robot (Shriya)](#81-scenario-1-institutional-library-book-collection-robot-shriya)
   - [8.2 Scenario 2: MediRunner Hospital Medicine Delivery Robot (Shreya)](#82-scenario-2-medirunner-hospital-medicine-delivery-robot-shreya)
   - [8.3 Scenario 3: Grand Hotel Room Service Delivery Robot (Tanishka)](#83-scenario-3-grand-hotel-room-service-delivery-robot-tanishka)
   - [8.4 Scenario 4: Airport Passenger Assistance Robot (Ritu and Sravanthi)](#84-scenario-4-airport-passenger-assistance-robot-ritu-and-sravanthi)
9. [Task 3: Live Multi-Scenario Graphical Interface (VNC & Web Browser)](#9-task-3-live-multi-scenario-graphical-interface-vnc--web-browser)
10. [Technical Implementation & Modifications Summary](#10-technical-implementation--modifications-summary)
11. [Repository Directory Structure](#11-repository-directory-structure)
12. [Troubleshooting & Diagnostic Guide](#12-troubleshooting--diagnostic-guide)
13. [Submission Deliverables Checklist](#13-submission-deliverables-checklist)
14. [References & Academic Attributions](#14-references--academic-attributions)

---

## 1. Team Members and Contributions

Each team member took ownership of distinct architectural milestones across Tasks 1, 2, and 3:

| Member | Responsibility | Key Deliverables & Artifacts |
|---|---|---|
| **Sravanthi** | **Task 1:** Repository setup and execution. <br>**Task 2:** Running, testing and recording the existing NavRep simulation scenarios. <br>**Task 3:** Airport Passenger Assistance Robot – custom scenario design and implementation (co-developed with Ritu). | `Dockerfile`, `docker-compose.yml`, `scripts_custom/smoke_test.py`, `scripts_custom/verify_videos.py`, `scripts_custom/remediate_and_record.py`, 22 Task 2 MP4 videos, `submission/README_TASK1.md`, `submission/README_TASK2.md`. |
| **Sona** | Overall project README, technical documentation, organizing scenario information and screenshots, documenting project setup and execution, and coordinating the final documentation and submission. | Root `README.md`, `submission/commands.md`, `docs/SCENARIO_INVENTORY.md`, `docs/CHALLENGES_LOG.md`, scenario documentation synthesis, submission deliverable validation. |
| **Shriya** | Task 3: Library Book Collection Robot – custom scenario design and implementation. | `navrep_custom/library_book_collection.py`, `navrep_custom/library_planner.py`, `navrep_custom/library_gui.py`, `tests/test_library_scenario.py`, `tests/test_custom_scenarios.py`, `submission/COMPLETE_PROJECT_RUN_AND_EVALUATION_GUIDE.md`. |
| **Shreya** | Task 3: Hospital Medicine Delivery Robot – custom scenario design and implementation. | `navrep_custom/hospital_delivery.py`, `scripts_custom/record_hospital_demo.py`, `tests/test_hospital_controller.py`, `submission/task3_custom/videos/hospital_*.mp4`, `submission/task3_custom/SCENARIO_DESCRIPTIONS.md`. |
| **Tanishka** | Task 3: Hotel Room Service Delivery Robot – custom scenario design and implementation. | `navrep_custom/hotel_room_service.py`, `scripts_custom/record_hotel_demo.py`, `scripts_custom/test_hotel_gui_rendering.py`, `submission/task3_custom/videos/hotelroomservice_*.mp4`. |
| **Ritu** | Task 3: Airport Passenger Assistance Robot – custom scenario design and implementation (co-developed with Sravanthi). | `navrep_custom/airport_passenger_assistance.py`, `navrep_custom/airport_dashboard_renderer.py`, `scripts_custom/record_airport_dashboard.py`, `scripts_custom/benchmark_airport.py`, `submission/task3_custom/AIRPORT_PASSENGER_ASSISTANCE.md`. |

---

## 2. Project Overview & Theoretical Background

### 2.1 The NavRep Framework
[NavRep](https://github.com/ethz-asl/navrep) is an open-source autonomous mobile robot navigation framework developed by the Autonomous Systems Lab (ASL) at ETH Zürich (Dugas et al., *IEEE Robotics and Automation Letters / IROS 2020*). It investigates how **unsupervised representation learning** can assist deep reinforcement learning (DRL) in dynamic, human-populated environments without requiring full state observability.

The navigation agent operates under a partially observable Markov decision process (POMDP):
1. **Perception**: An onboard planar LiDAR scanner returns a 1080-beam range profile ($\theta \in [-\pi, \pi]$).
2. **Representation Learning (World Model)**: A Convolutional Variational Autoencoder (VAE) compresses raw 2D range observations into a low-dimensional latent vector $\mathbf{z}_t \in \mathbb{R}^{32}$. An LSTM recurrent transition model predicts future latent states $\mathbf{z}_{t+1}$ conditioned on robot actions $\mathbf{a}_t$.
3. **Policy (Actor-Critic)**: A Proximal Policy Optimization (PPO) policy uses the concatenated Velocity + Measurement latent embedding (`VM`, $\mathbf{s}_t = [\mathbf{z}_t, \mathbf{v}_t] \in \mathbb{R}^{34}$) to output continuous holonomic velocity commands:
   $$\mathbf{a}_t = [v_x, v_y]^\top, \quad \|\mathbf{a}_t\| \le v_{\max}$$

```
                +-------------------------------------------------------+
                |                NavRep World Model Architecture        |
                |                                                       |
  Raw LiDAR     |  +-----------------+      +------------------------+  |
  (1080 rays) ->|  | 2D/1D VAE Conv  | ---> | Latent Embedding (z_t) | -+--> PPO Policy
                |  +-----------------+      +------------------------+  |    (Action: [vx, vy])
                |                                      ^                |           |
                |                                      |                |           v
                |                            +-------------------+      |     Robot Actuators
                |                            |  LSTM Transition  | <----+     (Holonomic Drive)
                |                            +-------------------+      |
                +-------------------------------------------------------+
```

### 2.2 Objective of the Assignment
1. **Task 1: Environment Reproduction & Containerization**: Overcome modern operating system and compiler incompatibilities by encapsulating the legacy Python 3.6 / TensorFlow 1.13 / OpenAI Gym 0.15 stack inside an automated Docker desktop container.
2. **Task 2: Existing Scenarios Benchmark & Video Evaluation**: Benchmark all 11 upstream ETH Zürich scenarios across 20-seed sweeps, evaluate failure modes, and record 22 quality-verified demonstration videos with color-coded HUD outcome badges.
3. **Task 3: Custom Robotics Navigation Scenarios**: Design, implement, and validate four realistic autonomous service robot scenarios with static architectural constraints, dynamic crowds, global path planners, and interactive demonstration interfaces.

---

## 3. System Architecture & Containerization (Task 1)

### 3.1 Why Containerization Was Mandatory
Upstream NavRep relies on an end-of-life software stack:
* **Python 3.6.15**: Deprecated in 2021; unavailable natively on modern Linux distros (Ubuntu 22.04+) or Apple Silicon macOS.
* **TensorFlow 1.13.2 / Keras 2.3.1**: Incompatible with GCC 9+ compilers, modern glibc (>= 2.34), and NumPy >= 1.20.
* **Custom C++ Collision Extensions (`crings`)**: Compiled Cython LiDAR ray-marching routines requiring exact NumPy C-API ABI matching.
* **OpenAI Gym 0.15.7 & PyTorch 1.6.0 CPU**: Obsolete serialization wheels requiring exact pip constraints to unpickle pretrained checkpoints.

### 3.2 Our Docker Desktop Architecture
To guarantee 100% reproducible execution across **Windows 11 (PowerShell)**, **macOS (Intel/Apple Silicon)**, and **Ubuntu Linux**, we built the `navrep-svnit:latest` container:

```
  Host Machine (Windows 11 / macOS / Linux)
  |
  +--> Web Browser (Chrome/Safari/Edge)  ----->  http://localhost:6080 (noVNC HTML5 Desktop)
  +--> PowerShell / Terminal             ----->  docker exec navrep_desktop <commands>
  |
  +-- Docker Container: "navrep_desktop" (Ubuntu 22.04 LTS linux/amd64)
       |
       +-- Python 3.6.15 Runtime (/opt/env)
       +-- Xvfb Virtual Framebuffer (:99, 1280x800x24)
       +-- Mesa 3D Software OpenGL Rasterizer (LIBGL_ALWAYS_SOFTWARE=1, GL 3.3)
       +-- Lightweight XFCE4 Window Manager (xfwm4)
       +-- Remote Display: x11vnc (Port 5900) <--> websockify / noVNC (Port 6080)
       +-- Shared Host Mount: /workspace (Bi-directional real-time file sync)
```

### 3.3 Pinned Dependency Matrix
All core packages are pinned in `Dockerfile` and `requirements_docker.txt` (cross-referenced against `submission/pip_freeze.txt`):

| Package | Version | Justification & Compatibility Role |
| :--- | :--- | :--- |
| **Python** | `3.6.15` | Required by upstream ETH NavRep and compiled C++ dependencies. |
| **TensorFlow** | `1.13.2` | Matches pretrained VAE-LSTM and PPO model graphs. |
| **Keras** | `2.3.1` | Required for VAE encoder/decoder network definitions. |
| **PyTorch** | `1.6.0+cpu` | Stable CPU-only execution for PyTorch-based baseline policies. |
| **Gym** | `0.15.7` | Strict dependency to unpickle pretrained Stable-Baselines checkpoints. |
| **Stable-Baselines** | `2.10.0` | Provides PPO2 actor-critic policy execution. |
| **NumPy** | `1.19.5` | Strict binary ABI compatibility with Cython extensions (`crings`). |
| **Cython** | `0.29.28` | JIT compilation of ray-tracing and distance-ring algorithms. |
| **OpenCV** | `4.5.5.64` | Headless computer vision rasterization and video frame encoding. |
| **Mesa Utils** | `OpenGL 3.3` | Software rasterizer eliminating host GPU driver dependencies. |

---

## 4. From-Scratch Setup Guide (Windows 11 & Cross-Platform)

Follow these exact steps from an empty workspace to get the entire project running in under 5 minutes:

### Step 1: Verify Host Prerequisites
Ensure Docker Desktop and Git are installed on Windows 11 (PowerShell):

```powershell
git --version
docker --version
docker compose version
```

### Step 2: Clone the Repository and Checkout Branch `shriya`
```powershell
cd $HOME\Documents
git clone https://github.com/Sravanthi2727/AI_Robotics.git
cd AI_Robotics
git checkout shriya
```

### Step 3: Build and Start the Docker Container
From the repository root (where `docker-compose.yml` resides):

```powershell
# First time build and launch (downloads Ubuntu base and compiles Cython extensions):
docker compose up -d --build

# On subsequent runs, simply start the container:
docker compose up -d
```

### Step 4: Verify Container Status
```powershell
docker ps -f name=navrep_desktop
```
*Expected Output:* Status `Up`, with ports `0.0.0.0:5900->5900/tcp` and `0.0.0.0:6080->6080/tcp`.

### Step 5: Open the Interactive GUI in Your Web Browser
Open Google Chrome or Microsoft Edge and navigate to:
```text
http://localhost:6080/vnc_lite.html?scale=true
```
*(Alternatively, on macOS or Linux, connect native VNC clients to `localhost:5900`).*

---

## 5. Automated Verification Commands

All commands are executed directly from your host terminal using `docker exec`:

### 1. Minimal Smoke Test
Verifies environment instantiation, reset cycles, and 50 random action steps across `NavRepTrainEnv` and `IANEnv`:
```powershell
docker exec navrep_desktop python scripts_custom/smoke_test.py
```
*Expected Terminal Output:*
```text
=== NavRep Smoke Test ===
[1/2] Initializing NavRepTrainEnv(silent=True)...
NavRepTrainEnv completed 50 steps successfully.
[2/2] Initializing IANEnv(silent=True)...
IANEnv completed 50 steps successfully.
=== Smoke test PASSED cleanly! ===
```

### 2. Pretrained Model & MD5 Checksum Verification
Verifies that all 10 neural network weights (VAE, LSTM, and PPO checkpoints) are authentic and uncorrupted:
```powershell
docker exec navrep_desktop python scripts_custom/verify_models_and_checksums.py
```
*Expected Output:* `ALL 10 MODELS VERIFIED AND MATCH CHECKSUMS`.

### 3. Task 2 Video Quality Verification Suite
Evaluates all 22 demonstration MP4s against strict quality gates: duration $\ge 8.0\text{ s}$, 12 FPS, frozen frames $\le 20$, and outcome badge consistency:
```powershell
docker exec navrep_desktop python scripts_custom/verify_videos.py
```
*Expected Output:* `>>> ALL 22 VIDEOS PASSED PROGRAMMATIC QUALITY VERIFICATION! <<<`.

### 4. Fast Re-Compilation of Task 2 Metrics
Recompiles the quantitative CSV benchmark without running lengthy simulation sweeps:
```powershell
docker exec navrep_desktop python scripts_custom/remediate_and_record.py --compile-only
```

### 5. Complete Unit Test Suite (56 / 56 PASS)
Executes all automated unit tests covering the Library scenario, Custom scenario registry, and Hospital navigation controller:
```powershell
docker exec navrep_desktop python -m unittest tests/test_library_scenario.py tests/test_custom_scenarios.py tests/test_hospital_controller.py
```
*Expected Output:* `Ran 56 tests in 10.519s - OK`.

### 6. Multi-Scenario GUI Verification Test
Verifies that the Tkinter canvas, $A^*$ route synthesis, destination presets, and controller stepping work cleanly across Library, Hotel, and Hospital:
```powershell
docker exec -e DISPLAY=:99 navrep_desktop python scripts_custom/test_all_scenarios_gui.py
```
*Expected Output:* `Ran 6 tests in 9.401s - OK`.

---

## 6. Task 2: Testing Existing NavRep Scenarios

### 6.1 The 11 Original Scenarios
We systematically evaluated all 11 simulation scenarios provided in upstream ETH Zürich NavRep across three environment families:

```
+-------------------------------------------------------------------------------------------------+
|                                11 Existing NavRep Scenarios                                     |
+------------------------------+----------------------------------+-------------------------------+
|   Toy / Training Arenas      |      IROS ASL Hallways           |     IROS Unity Floorplans     |
+------------------------------+----------------------------------+-------------------------------+
| 1. navreptrain_train         | 3. irosasl1                      |  9. irosunity_scene_map1       |
| 2. navreptrain_test          | 4. irosasl2                      | 10. irosunity_scene_map2       |
|                              | 5. irosasl3                      | 11. irosunity_scene_map3       |
|                              | 6. irosasl_office_j1             |                               |
|                              | 7. irosasl_office_j2             |                               |
|                              | 8. irosasl_office_j3             |                               |
+------------------------------+----------------------------------+-------------------------------+
```

### 6.2 Evaluation Methodology
* **Benchmark Evaluation (Statistical Ground Truth)**: Each scenario was evaluated across **20 fixed pseudorandom seeds (seeds 100–119)** using the high-capacity `PPO_VAE_LSTM_VM_V32M512_ckpt` model under native step budgets (175 steps for `NavRepTrainEnv`, 500 steps for `IANEnv`). Results are tabulated in `submission/task2_videos/metrics_existing.csv`.
* **Demonstration Video Search**: A separate **60-seed sweep (seeds 0–59)** was executed to identify representative qualitative behaviors.
* **Honest Outcome Labeling**:
  - `_success.mp4`: The robot reached within $1.0\text{ m}$ of the goal (`GOAL_REACHED`, Green badge).
  - `_bestprogress.mp4`: Used when a scenario yielded $0.0\%$ success across benchmark seeds due to policy limitations (e.g., oscillating in concave corners), recording the run that minimized final Euclidean distance to the goal.
  - `_failure.mp4`: Preserves the natural failure mode (`COLLISION` in Red, or `TIMEOUT` in Orange).

---

## 7. Complete Task 2 22-Video Index

All 22 MP4 video files reside in `submission/task2_videos/` and were validated using `scripts_custom/verify_videos.py`:

| Video Filename | Scenario Identifier | Recorded Outcome | Evaluated Seed | Step Count | Simulation Time | Final Dist to Goal | Video Duration |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `01_navreptrain_train_failure.mp4` | `navreptrain_train` | **TIMEOUT** | `23` | 175 steps | $35.0\text{ s}$ | $7.41\text{ m}$ | $15.67\text{ s}$ |
| `01_navreptrain_train_success.mp4` | `navreptrain_train` | **GOAL_REACHED** | `1` | 94 steps | $18.8\text{ s}$ | $0.33\text{ m}$ | $8.92\text{ s}$ |
| `02_navreptrain_test_failure.mp4` | `navreptrain_test` | **TIMEOUT** | `26` | 175 steps | $35.0\text{ s}$ | $6.75\text{ m}$ | $15.67\text{ s}$ |
| `02_navreptrain_test_success.mp4` | `navreptrain_test` | **GOAL_REACHED** | `20` | 43 steps | $8.6\text{ s}$ | $0.14\text{ m}$ | $4.67\text{ s}$ |
| `03_irosasl1_bestprogress.mp4` | `irosasl1` | **COLLISION** | `23` | 97 steps | $19.4\text{ s}$ | $5.35\text{ m}$ | $9.17\text{ s}$ |
| `03_irosasl1_failure.mp4` | `irosasl1` | **COLLISION** | `115` | 96 steps | $19.2\text{ s}$ | $5.21\text{ m}$ | $9.08\text{ s}$ |
| `04_irosasl2_failure.mp4` | `irosasl2` | **COLLISION** | `4` | 383 steps | $76.6\text{ s}$ | $2.49\text{ m}$ | $33.00\text{ s}$ |
| `04_irosasl2_success.mp4` | `irosasl2` | **GOAL_REACHED** | `26` | 259 steps | $51.8\text{ s}$ | $0.49\text{ m}$ | $22.67\text{ s}$ |
| `05_irosasl3_failure.mp4` | `irosasl3` | **COLLISION** | `0` | 156 steps | $31.2\text{ s}$ | $17.73\text{ m}$ | $14.08\text{ s}$ |
| `05_irosasl3_success.mp4` | `irosasl3` | **GOAL_REACHED** | `56` | 401 steps | $80.2\text{ s}$ | $0.43\text{ m}$ | $34.50\text{ s}$ |
| `06_irosasl_office_j1_bestprogress.mp4` | `irosasl_office_j1` | **COLLISION** | `23` | 151 steps | $30.2\text{ s}$ | $4.94\text{ m}$ | $13.67\text{ s}$ |
| `06_irosasl_office_j1_failure.mp4` | `irosasl_office_j1` | **COLLISION** | `0` | 86 steps | $17.2\text{ s}$ | $5.89\text{ m}$ | $8.25\text{ s}$ |
| `07_irosasl_office_j2_failure.mp4` | `irosasl_office_j2` | **COLLISION** | `2` | 199 steps | $39.8\text{ s}$ | $22.77\text{ m}$ | $17.67\text{ s}$ |
| `07_irosasl_office_j2_success.mp4` | `irosasl_office_j2` | **GOAL_REACHED** | `47` | 461 steps | $92.2\text{ s}$ | $0.50\text{ m}$ | $39.50\text{ s}$ |
| `08_irosasl_office_j3_bestprogress.mp4` | `irosasl_office_j3` | **COLLISION** | `32` | 119 steps | $23.8\text{ s}$ | $2.19\text{ m}$ | $11.00\text{ s}$ |
| `08_irosasl_office_j3_failure.mp4` | `irosasl_office_j3` | **COLLISION** | `1` | 102 steps | $20.4\text{ s}$ | $6.01\text{ m}$ | $9.58\text{ s}$ |
| `09_irosunity_scene_map1_bestprogress.mp4` | `irosunity_scene_map1` | **COLLISION** | `39` | 125 steps | $25.0\text{ s}$ | $5.69\text{ m}$ | $11.50\text{ s}$ |
| `09_irosunity_scene_map1_failure.mp4` | `irosunity_scene_map1` | **COLLISION** | `0` | 215 steps | $43.0\text{ s}$ | $5.79\text{ m}$ | $19.00\text{ s}$ |
| `10_irosunity_scene_map2_failure.mp4` | `irosunity_scene_map2` | **COLLISION** | `17` | 331 steps | $66.2\text{ s}$ | $3.84\text{ m}$ | $28.67\text{ s}$ |
| `10_irosunity_scene_map2_success.mp4` | `irosunity_scene_map2` | **GOAL_REACHED** | `58` | 291 steps | $58.2\text{ s}$ | $0.41\text{ m}$ | $25.33\text{ s}$ |
| `11_irosunity_scene_map3_failure.mp4` | `irosunity_scene_map3` | **COLLISION** | `1` | 238 steps | $47.6\text{ s}$ | $4.58\text{ m}$ | $20.92\text{ s}$ |
| `11_irosunity_scene_map3_success.mp4` | `irosunity_scene_map3` | **GOAL_REACHED** | `34` | 252 steps | $50.4\text{ s}$ | $0.49\text{ m}$ | $22.08\text{ s}$ |

*Configuration: $640\times 640$ @ 12 FPS | Dynamic $20\text{ m} \times 20\text{ m}$ camera | Laser scan overlay | PiP minimap.*

---

## 8. Task 3: Custom Robotics Navigation Scenarios

### 8.1 Scenario 1: Institutional Library Book Collection Robot (Shriya)

![Library Book Collection Robot](docs/images/library_book_collection.png)
*Figure 8.1: Live Tkinter GUI demonstration of the Institutional Library Book Collection scenario showing parquet flooring, book stacks A1–A3, reading tables, and $A^*$ route.*

* **Lead Author:** Shriya Siddesh
* **Real-World Application & Motivation:** In university and research libraries, autonomous mobile robots (AMRs) collect returned books from narrow stack aisles and transport them to the central circulation counter, navigating safely around studying students and walking librarians without generating noise or safety hazards.
* **Floorplan & Spatial Dimensions:**
  - **Dimensions:** $20.0\text{ m} \times 12.0\text{ m} = 240.0\text{ m}^2$ ($X \in [-10.0, 10.0]\text{ m}, Y \in [-6.0, 6.0]\text{ m}$).
  - **Start Pose:** $(-8.0, 3.5, 0.0\text{ rad})$ deep inside Stack Aisle 2.
  - **Goal Pose:** $(5.5, -3.5)$ at the Circulation & Returns Counter.
  - **Direct Separation Distance:** $15.21\text{ m}$ (exceeds the required $8.0\text{ m}$ benchmark separation).
  - **Aisle Clearance:** $1.8\text{ m}$ clear span between parallel bookshelves.
* **Static Obstacles:**
  - West Book Stacks: `STACK A1`, `STACK A2`, `STACK A3` ($5.0\text{ m} \times 0.8\text{ m}$ oak shelves).
  - East Archives: `STACK B1`, `STACK B2`, `STACK B3` ($4.5\text{ m} \times 0.8\text{ m}$ reference racks).
  - Furniture: Study Desks A and B, Reference Study Zone, Archive Reading Table.
  - Architectural Columns: Octagonal structural stone pillars (Pillars A and B).
* **Dynamic Human Crowd (ORCA Simulation):**
  - Browsing reader ($v = 0.40\text{ m/s}$ along shelf aisles).
  - Librarian pushing a book trolley ($v = 0.60\text{ m/s}$ along cross-corridor).
  - Hurrying students ($v = 0.85\text{ m/s}$ crossing reading halls).
* **Navigation Architecture (Why Pure PPO Failed & Hybrid Solution):**
  - **Failure Cause:** Pretrained PPO policies lack topological memory; in narrow corridors, local gradient vectors drive straight into concave bookshelf corners, resulting in cyclic limit cycles and 100% timeout failures.
  - **Global Occupancy-Grid $A^*$ Search:** Evaluates $24,000$ grid cells with an obstacle safety inflation margin of **$0.48\text{ m}$** ($r_{\text{robot}} = 0.30\text{ m} + 0.18\text{ m}$ buffer), generating a verified collision-free corridor skeleton.
  - **Predictive Dynamic Yielding (Time-to-Closest-Approach):** Calculates predicted closest approach time:
    $$t_{\text{CPA}} = -\frac{\mathbf{r}_{\text{rel}} \cdot \mathbf{v}_{\text{rel}}}{\|\mathbf{v}_{\text{rel}}\|^2}, \quad d_{\text{CPA}} = \|\mathbf{r}_{\text{rel}} + \mathbf{v}_{\text{rel}} t_{\text{CPA}}\|$$
    If $d_{\text{CPA}} < 0.28\text{ m}$, the robot preemptively brakes without creeping.
  - **Zero-Creep Emergency Stopping:** When obstacle distance $d < 0.16\text{ m}$, command velocity is set strictly to $\mathbf{v} = [0, 0]$, eliminating legacy sliding collisions.
* **Quantitative Benchmark Results (20 Seeds: 42–61):**
  - **EASY:** **100.0% Success (20/20)**, Mean Time: $22.1\text{ s}$
  - **MEDIUM:** **100.0% Success (20/20)**, Mean Time: $23.4\text{ s}$
  - **HARD:** **100.0% Success (20/20)**, Mean Time: $28.5\text{ s}$
* **Code Location:** [`navrep_custom/library_book_collection.py`](file:///Users/shriyasiddesh/cs/air-lab/AI_Robotics/navrep_custom/library_book_collection.py), [`navrep_custom/library_planner.py`](file:///Users/shriyasiddesh/cs/air-lab/AI_Robotics/navrep_custom/library_planner.py), [`tests/test_library_scenario.py`](file:///Users/shriyasiddesh/cs/air-lab/AI_Robotics/tests/test_library_scenario.py).

---

### 8.2 Scenario 2: MediRunner Hospital Medicine Delivery Robot (Shreya)

![Hospital Medicine Delivery Robot](docs/images/hospital_medicine_delivery.png)
*Figure 8.2: MediRunner Hospital Delivery scenario showing antiseptic clinical porcelain floorplan, triage desk with live ECG monitors, and high-speed emergency gurney hazard.*

* **Lead Author:** Shreya (Integrated and Enhanced by Shriya)
* **Real-World Application & Motivation:** Clinical logistics AMRs rapidly transport critical medicines, blood samples, and diagnostic supplies from pharmacy storage to Intensive Care Units (ICU) through congested hospital hallways. They must yield absolute right-of-way to emergency gurneys while protecting vulnerable, slow-moving patients.
* **Floorplan & Spatial Dimensions:**
  - **Dimensions:** $24.0\text{ m} \times 8.0\text{ m} = 192.0\text{ m}^2$ ($X \in [-12.0, 12.0]\text{ m}, Y \in [-4.0, 4.0]\text{ m}$).
  - **Start Pose:** Central Pharmacy Dispenser at $(-10.5, 0.0, 0.0\text{ rad})$.
  - **Goal Pose:** ICU Acute Care Ward Entrance at $(10.5, 0.0)$.
  - **Separation Distance:** $21.0\text{ m}$.
* **Structural Bottleneck & Recessed Alcoves:**
  - **Central Triage Desk:** Structural counter ($4.0\text{ m} \times 1.6\text{ m}$ at $(0.0, 1.2)$) creating an asymmetric hallway. The northern corridor has only $0.40\text{ m}$ clearance (impassable for the $0.65\text{ m}$ robot envelope), forcing the $A^*$ planner to route through the southern bypass lane ($y = -0.60\text{ m}$, $1.10\text{ m}$ clear span).
  - **Four Patient Room Alcoves:** `Room 101: Post-Op`, `Room 102: Cardiology`, `Room 103: Pediatrics`, `Room 104: Trauma Care`.
  - **Safe Pull-In Refuges:** Recessed bays at $X \in \{-2.4, 2.4\}\text{ m}$.
* **Dynamic Medical Crowd & Critical Hazards:**
  - **High-Speed Emergency Gurney:** Rapid patient stretcher moving westbound at **$1.65\text{ m/s}$** ($r = 0.65\text{ m}$) with flashing emergency beacon halos and statutory right-of-way.
  - Urgent Clinical Staff: Nurses and physicians walking at $1.05\text{ m/s}$ in teal scrubs.
  - Frail Geriatric Patients: Slow individuals walking with 4-wheel assistive walker frames ($0.35\text{ m/s}$).
* **Controller Safety Strategy:**
  - Continuous CPA lookahead estimates oncoming gurney trajectory.
  - If a collision course is detected, the robot transitions to `ALCOVE_YIELDING`, steers into North Alcove 1 at $(-2.40, 1.25)$, comes to a zero-velocity halt until the gurney passes, and then resumes its transit to the ICU.
* **Demonstration Videos & Code Location:**
  - Recordings: `submission/task3_custom/videos/hospital_easy.mp4`, `hospital_medium.mp4`, `hospital_hard.mp4`.
  - Code: [`navrep_custom/hospital_delivery.py`](file:///Users/shriyasiddesh/cs/air-lab/AI_Robotics/navrep_custom/hospital_delivery.py), [`scripts_custom/record_hospital_demo.py`](file:///Users/shriyasiddesh/cs/air-lab/AI_Robotics/scripts_custom/record_hospital_demo.py), [`tests/test_hospital_controller.py`](file:///Users/shriyasiddesh/cs/air-lab/AI_Robotics/tests/test_hospital_controller.py).

---

### 8.3 Scenario 3: Grand Hotel Room Service Delivery Robot (Tanishka)

![Hotel Room Service Robot](docs/images/hotel_room_service.png)
*Figure 8.3: Grand Hotel Room Service Delivery scenario showing sapphire lounge, executive guest suites, concierge station, and cloche AMR delivering to Suite 204.*

* **Lead Author:** Tanishka Kumthekar
* **Real-World Application & Motivation:** Luxury hospitality service robots deliver food trays under covered cloches from the kitchen dispatch area to guest suites, maneuvering elegantly along carpeted corridors while avoiding hotel guests with luggage and housekeeping carts.
* **Floorplan & Spatial Dimensions:**
  - **Dimensions:** $24.0\text{ m} \times 16.0\text{ m} = 384.0\text{ m}^2$ ($X \in [-12.0, 12.0]\text{ m}, Y \in [-8.0, 8.0]\text{ m}$).
  - **Start Pose:** Kitchen Dispatch Station at $(-10.0, 0.0, 0.0\text{ rad})$.
  - **Goal Pose:** Guest Suite 204 at $(10.0, 0.0)$.
  - **Direct Separation:** $20.0\text{ m}$.
* **Architectural Elements:**
  - North Executive Suites: `Suite 201`, `Suite 202` (Executive), `Suite 203` (Presidential).
  - South Deluxe Suites: `Suite 101`, `Suite 102`, `Suite 103` (Garden Villa).
  - Hospitality Facilities: Reception / Concierge desk with bell, sapphire velvet lounge sofa, glass coffee table, and housekeeping linen stations.
* **Dynamic Crowd:**
  - Hotel guests walking along corridors pulling rolling suitcases ($r = 0.50\text{ m}, v = 0.65\text{ m/s}$).
  - Housekeeping staff pushing linen supply trolleys ($v = 0.45\text{ m/s}$).
* **Demonstration Videos & Code Location:**
  - Recordings: `submission/task3_custom/videos/hotelroomservice_easy.mp4`, `hotelroomservice_medium.mp4`, `hotelroomservice_hard.mp4`.
  - Code: [`navrep_custom/hotel_room_service.py`](file:///Users/shriyasiddesh/cs/air-lab/AI_Robotics/navrep_custom/hotel_room_service.py), [`scripts_custom/record_hotel_demo.py`](file:///Users/shriyasiddesh/cs/air-lab/AI_Robotics/scripts_custom/record_hotel_demo.py), [`scripts_custom/test_hotel_gui_rendering.py`](file:///Users/shriyasiddesh/cs/air-lab/AI_Robotics/scripts_custom/test_hotel_gui_rendering.py).

---

### 8.4 Scenario 4: Airport Passenger Assistance Robot (Ritu and Sravanthi)

![Airport Passenger Assistance Robot](docs/images/airport_passenger_assistance.png)
*Figure 8.4: Airport Passenger Assistance scenario mission-control dashboard showing terminal concourse, security checkpoint, and 100% benchmark success at Gate 12.*

> [!NOTE]
> **Benchmark Metric Evaluation Scope:** The benchmark statistics shown on the mission-control dashboard screenshot (100% Success, 0% Collision, 0% Timeout) refer specifically to the evaluated **EASY** difficulty trial run. As documented below, benchmark performance varies under higher crowd density and dynamic "Boarding Call" events (50% on Medium, 15% on Hard), and these Easy-tier metrics should not be generalized to all scenarios or difficulty tiers.

* **Lead Authors:** Ritu & Sravanthi
* **Real-World Application & Motivation:** Airport concierge robots assist elderly and reduced-mobility passengers carrying luggage from the main terminal entrance through security checkpoints and busy concourses to their departure gate.
* **Floorplan & Spatial Dimensions:**
  - **Dimensions:** $24.0\text{ m} \times 15.0\text{ m} = 360.0\text{ m}^2$ ($X \in [-12.0, 12.0]\text{ m}, Y \in [-7.5, 7.5]\text{ m}$).
  - **Start Pose:** Terminal Check-in Entrance at $(-10.2, -2.5, 0.0\text{ rad})$.
  - **Goal Pose:** Gate 12 Boarding Podium at $(9.8, 2.8)$.
  - **Direct Separation Distance:** $20.69\text{ m}$ (diagonal cross-concourse trajectory).
* **Terminal Architectural Layout (10 Structural Islands):**
  - Check-in ticketing counters and flight information display kiosks.
  - TSA security inspection lanes and metal detector gates.
  - Central concourse passenger boulevard, duty-free retail boutiques, and departure lounge seating banks.
* **Dynamic Passenger Flow & Special Events:**
  - Heterogeneous travelers: Slow passengers with luggage trolleys ($r=0.45\text{ m}, v=0.40\text{ m/s}$), hurried business travelers rushing to gates ($v=0.95\text{ m/s}$), and families walking together ($v=0.55\text{ m/s}$).
  - **Dynamic "Boarding Call" Event:** At step $T=75$ (Medium) and $T=50$ (Hard), concourse passengers dynamically divert their walking trajectories toward Gate 12, creating intense cross-flow congestion.
* **Empirical 20-Seed Benchmark Results (Seeds 100–119):**
  - **EASY:** **100.0% Success (20/20)**, 0.0% Collision, 0.0% Timeout, Mean Time: $32.76\text{ s}$
  - **MEDIUM:** **50.0% Success (10/20)**, 20.0% Collision, 30.0% Timeout, Mean Time: $33.46\text{ s}$
  - **HARD:** **15.0% Success (3/20)**, 50.0% Collision, 35.0% Timeout, Mean Time: $32.87\text{ s}$
* **Dashboard Video Suite:** High-resolution $1600\times 900$ mission-control recordings equipped with LiDAR point clouds, velocity vectors, and active pedestrian distance readouts (`videos_dashboard/airport_passenger_assistance_*.mp4`).
* **Source Location:** `origin/ritu-airport-assistance` branch (`AIRPORT_PASSENGER_ASSISTANCE.md`, `airport_metrics.csv`).

---

## 9. Task 3: Live Multi-Scenario Graphical Interface (VNC & Web Browser)

We engineered an interactive, high-performance Tkinter + OpenCV desktop interface ([`navrep_custom/library_gui.py`](file:///Users/shriyasiddesh/cs/air-lab/AI_Robotics/navrep_custom/library_gui.py)) running inside the Docker virtual framebuffer (`DISPLAY=:99`) and accessible via any browser at **[http://localhost:6080/vnc_lite.html?scale=true](http://localhost:6080/vnc_lite.html?scale=true)**.

```
+----------------------------------------------------------------------------------------------------+
|  TOP HEADER: [1] Institutional Library  |  [2] Grand Hotel Room Service  |  [3] MediRunner Hospital|
+-------------------------------------------------------------+--------------------------------------+
|                                                             |  Showcase Demonstration:             |
|                                                             |    [>] RUN FULL DEMONSTRATION        |
|                                                             |                                      |
|                                                             |  Modular Operations:                 |
|             HIGH-RESOLUTION SIMULATION CANVAS               |    [Show Scenario] [Run A* Planning] |
|                                                             |    [Start Nav]     [Pause]   [Reset] |
|             - Architectural Flooring & Wall Plaque Models   |                                      |
|             - Start (Green) & Goal (Amber) Rings            |  Interactive Goal Selection:         |
|             - Animated A* Search Wavefront (Cyan)           |    [+] SELECT NEW GOAL [REPLAN ROUTE]|
|             - Dynamic Obstacles (Pedestrians, Gurneys)      |    Preset: [ Dropdown Menu ]         |
|             - Robot Pose & Velocity Vectors                 |                                      |
|                                                             |  Difficulty: [EASY] [MEDIUM] [HARD]  |
|                                                             |  Sim Rate:   [0.5x] [1.0x]   [2.0x]  |
|                                                             |                                      |
|                                                             |  Live Telemetry:                     |
|                                                             |    Step, Sim Time, Pose, Speed, CPA  |
+-------------------------------------------------------------+--------------------------------------+
```

### Key Interactive Features:
1. **Instant 3-Scenario Switching**: Toggle between Institutional Library, Grand Hotel, and MediRunner Hospital with one click. Floorplans, bounds, obstacles, and $A^*$ waypoints recompute automatically.
2. **Interactive Goal Selection & Safety Validation**: Click anywhere on the map to set a custom goal. If clicked inside an obstacle, the safety layer rejects the goal (`Invalid Goal: Position inside obstacle or lacks 0.35m clearance`).
3. **Dynamic $A^*$ Replanning**: Clicking **`REPLAN ROUTE`** runs $A^*$ search from the robot's *current coordinate* to the new destination.
4. **Destination Presets**: Choose canonical points of interest (e.g., `Reference Table`, `Suite 202`, `ICU Acute Care Ward`).
5. **Live Telemetry & Benchmark Runner**: Displays pose $(x, y, \theta)$, linear velocity, CPA yielding state, and includes a multi-threaded **`Run 20-Seed Benchmark`** runner.

---

## 10. Technical Implementation & Modifications Summary

The table below outlines our modifications around upstream ETH Zürich NavRep:

| Module / Component | Upstream NavRep Baseline | Our Group Implementation (`shriya` branch) |
| :--- | :--- | :--- |
| **Operating System** | Native Ubuntu 18.04/20.04 (Host install) | Standalone Docker container (`Ubuntu 22.04 LTS linux/amd64`) with Mesa software GL. |
| **Display & GUI** | Direct Pygame window (fails on headless) | Xvfb `:99` virtual display + XFCE4 desktop + x11vnc + zero-install noVNC browser client. |
| **Scenario Suite** | 11 toy/hallway/office scenarios | **15 scenarios total**: Added Library, Hospital, Hotel Room Service, and Airport. |
| **Navigation Planning** | Reactive PPO policy (blind to corners) | **Hierarchical Navigation**: Global occupancy-grid $A^*$ path planner + Local predictive TCA. |
| **Collision Avoidance** | Proximity repulsive heuristics | **Predictive Time-to-Closest-Approach (TCA)** yielding and absolute zero-creep braking. |
| **Interactive Demo** | Minimal command-line scripts | Unified Tkinter GUI with dynamic replanning, interactive goal selection, and presets. |
| **Video Recording** | Basic Gym monitor wrappers | Full telemetry HUD, color-coded outcome badges, laser scan overlays, and PiP minimaps. |
| **Automated Testing** | Unmaintained nose tests | **56 unit tests (100% PASS)** covering environments, controllers, planners, and GUI. |

---

## 11. Repository Directory Structure

```text
AI_Robotics/
├── Dockerfile                         # Ubuntu 22.04 + Python 3.6 + XFCE4 + noVNC container specification
├── docker-compose.yml                 # Service configuration mapping ports 6080 and 5900
├── requirements.txt                   # Baseline Python dependency specifications
├── requirements_docker.txt            # Strict pinned package versions for Python 3.6.15
├── setup_env.sh                       # Native environment bootstrap script
├── start_desktop.sh                   # Entrypoint script launching Xvfb, XFCE4, x11vnc, and websockify
├── README.md                          # Master project documentation (this document)
│
├── navrep/                            # Upstream ETH Zürich NavRep framework
│   ├── envs/                          # Gym environment classes (NavRepTrainEnv, IANEnv, SOADRL)
│   ├── models/                        # VAE encoder, LSTM transition, and Siren neural networks
│   └── tools/                         # Latent space extraction and policy wrappers
│
├── navrep_custom/                     # Custom scenario environments and navigation architectures
│   ├── library_book_collection.py     # Scenario 1 environment, obstacles, and ORCA human crowds
│   ├── library_planner.py             # OccupancyGrid, AStarPlanner, and dynamic replanning engine
│   ├── library_gui.py                 # Unified 3-scenario live demonstration GUI (Tkinter + OpenCV)
│   ├── hospital_delivery.py           # Scenario 2 environment, triage bottleneck & gurney yielding
│   ├── hotel_room_service.py          # Scenario 3 environment, suites, luggage & housekeeping carts
│   ├── custom_envs.py                 # High-level CustomNavRepEnv and CustomNavRepEncodedEnv wrappers
│   └── scenarios.py                   # Central scenario configuration registry and difficulty presets
│
├── models/                            # Pretrained neural network weights and checkpoints
│   └── gym/                           # Checkpoint: PPO_VAE_LSTM_V_ONLY_V32M512_ckpt.zip (and MD5s)
│
├── maps/                              # 2D discrete occupancy grid maps and floorplans
├── docs/                              # Supporting engineering logs and documentation
│   ├── images/                        # High-resolution scenario screenshots embedded in README
│   │   ├── library_book_collection.png
│   │   ├── hospital_medicine_delivery.png
│   │   ├── hotel_room_service.png
│   │   └── airport_passenger_assistance.png
│   ├── CHALLENGES_LOG.md              # Detailed record of legacy library and compiler bugs solved
│   └── SCENARIO_INVENTORY.md          # Technical specifications for custom scenario suite
│
├── scripts_custom/                    # Automated testing, benchmarking, and recording scripts
│   ├── smoke_test.py                  # Fast environment reset/step verification
│   ├── verify_models_and_checksums.py # Validates MD5 hashes of all 10 neural network models
│   ├── verify_videos.py               # Quality gate validator for all 22 Task 2 videos
│   ├── remediate_and_record.py        # Task 2 benchmark metric re-compiler
│   ├── record_hospital_demo.py        # Headless video recorder for MediRunner Hospital
│   ├── record_hotel_demo.py           # Headless video recorder for Hotel Room Service
│   ├── test_all_scenarios_gui.py      # End-to-end automated GUI verification across 3 scenarios
│   └── test_hotel_gui_rendering.py    # Luxury hotel canvas rendering test script
│
├── tests/                             # Automated verification test suites
│   ├── test_library_scenario.py       # 14 unit tests: Library environment, geometry, and A* planner
│   ├── test_custom_scenarios.py       # 14 unit tests: Multi-scenario registry and encoded wrappers
│   └── test_hospital_controller.py    # 28 unit tests: Hospital controller, gurney detection & yielding
│
└── submission/                        # Evaluator submission package
    ├── COMPLETE_PROJECT_RUN_AND_EVALUATION_GUIDE.md # End-to-end beginner reproduction manual
    ├── EVALUATOR_PRESENTATION_GUIDE.md              # Oral presentation pitch and viva voce defense script
    ├── README_TASK1.md                # 700+ line detailed reproduction and troubleshooting report
    ├── README_TASK2.md                # Detailed analysis of 11 existing scenarios and failure modes
    ├── commands.md                    # Exact copy-paste reproduction command log
    ├── pip_freeze.txt                 # Exact `pip freeze` package version snapshot
    ├── navrep_modified_source.zip     # Complete modified source code archive (2.8 MB)
    ├── task2_videos/                  # 22 verified demonstration MP4s, metadata JSONs, metrics CSV
    └── task3_custom/                  # Custom scenario descriptions, metrics, maps, and videos
        ├── videos/                    # Verified MP4s: library, hotel, and hospital demonstrations
        └── gui/                       # Standalone copy of live GUI demonstration launcher
```

---

## 12. Troubleshooting & Diagnostic Guide

| # | Error / Symptom | Root Cause | Solution & Diagnostic Command |
|---|---|---|---|
| **1** | `No matching distribution found` during pip install | Host running modern Python (>=3.8) where legacy wheels do not exist. | **Do not install natively.** Use the containerized environment: `docker compose up -d`. |
| **2** | `cannot connect to X server` or display error | Headless execution without an active X-server display. | Set virtual display: `export DISPLAY=:99` or use `docker exec -e DISPLAY=:99 ...`. |
| **3** | Ghost obstacle collisions in open corridors | Upstream `CrowdSim` used an unrasterized $14\text{ m}$ grid with stale obstacle cells. | Implemented metric transpose rasterization (`sim.map = img.T`) in `sync_and_rasterize_sim()`. |
| **4** | Premature 177-step timeout in custom scenarios | Upstream `CrowdSim` hardcoded `time_limit = 35s` ($176 \times 0.2\text{ s}$). | Synchronized time limits: `sim.time_limit = raw_env.time_limit * dt`. |
| **5** | Port 5900 or 6080 already in use | Another container or local VNC service binds port 5900 or 6080. | Stop conflicting processes: `docker rm -f navrep_desktop; docker compose up -d`. |
| **6** | `websockify: [Errno 2] No such file` on Mac host | Running `/usr/bin/websockify` on the host rather than inside Docker. | Access noVNC in your host browser at `http://localhost:6080/vnc_lite.html?scale=true`. |
| **7** | Creeping collisions during pedestrian crossings | Heuristic braking had a $5\%$ velocity floor, creeping at $8\text{ mm/step}$ into humans. | Implemented absolute zero-creep emergency stopping ($\mathbf{v} = [0, 0]$) when $d < 0.16\text{ m}$. |
| **8** | PyPI timeout during Docker image build | Temporary network failure or package index throttling. | Retry build: `docker compose build --no-cache`. |

---

## 13. Submission Deliverables Checklist

| Assignment Task | Required Deliverable | Exact Repository Location | Verification Status |
| :--- | :--- | :--- | :---: |
| **Task 1** | Containerized reproduction setup & environment guide | `Dockerfile`, `docker-compose.yml`, `submission/README_TASK1.md` | **Verified PASS** |
| **Task 1** | Model weight verification & checksum script | `scripts_custom/verify_models_and_checksums.py` | **10 / 10 Models PASS** |
| **Task 2** | 22 Demonstration Videos (Success & Failure) | `submission/task2_videos/*.mp4` | **22 / 22 Videos PASS** |
| **Task 2** | Quantitative 20-Seed Benchmark CSV | `submission/task2_videos/metrics_existing.csv` | **Complete** |
| **Task 2** | Detailed Scenario Analysis Report | `submission/README_TASK2.md`, `VIDEO_INDEX.md` | **Complete** |
| **Task 3** | Library Book Collection Robot (Shriya) | `navrep_custom/library_book_collection.py`, `library_*.mp4` | **100.0% Success (20/20)** |
| **Task 3** | Hospital Medicine Delivery Robot (Shreya) | `navrep_custom/hospital_delivery.py`, `hospital_*.mp4` | **100.0% Success (20/20)** |
| **Task 3** | Hotel Room Service Delivery Robot (Tanishka) | `navrep_custom/hotel_room_service.py`, `hotelroomservice_*.mp4` | **Complete** |
| **Task 3** | Airport Passenger Assistance Robot (Ritu & Sravanthi) | `origin/ritu-airport-assistance`, `AIRPORT_PASSENGER_ASSISTANCE.md` | **100.0% Success (Easy)** |
| **Task 3** | Interactive Demonstration GUI | `navrep_custom/library_gui.py`, `submission/task3_custom/gui/` | **3-Scenario Live GUI** |
| **Task 3** | Automated Unit Test Suite (56 Tests) | `tests/test_library_scenario.py`, `test_custom_scenarios.py`, `test_hospital_controller.py` | **56 / 56 PASS in 10.5s** |
| **Package** | Complete Modified Source ZIP | `submission/navrep_modified_source.zip` | **Rebuilt (2.8 MB)** |
| **Package** | Complete End-to-End Reproduction Manual | `submission/COMPLETE_PROJECT_RUN_AND_EVALUATION_GUIDE.md` | **Complete** |
| **Package** | Evaluator Oral Presentation & Defense Guide | `submission/EVALUATOR_PRESENTATION_GUIDE.md` | **Complete** |

---

## 14. References & Academic Attributions

1. **Daniel Dugas, Juan Nieto, Roland Siegwart, and Jen Jen Chung**, *"NavRep: Unsupervised Representations for Reinforcement Learning of Robot Navigation in Dynamic Human Environments"*, IEEE Robotics and Automation Letters (RA-L) / IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2020. [arXiv:2012.04406](https://arxiv.org/pdf/2012.04406.pdf).
2. **Official ETH Zürich NavRep Repository**: [https://github.com/ethz-asl/navrep](https://github.com/ethz-asl/navrep).
3. **OpenAI Gym**: Greg Brockman, et al., *"OpenAI Gym"*, arXiv:1606.01540, 2016.
4. **Stable-Baselines**: Ashley Hill, et al., *"Stable Baselines: A Fork of OpenAI Baselines — Deep Reinforcement Learning"* [GitHub Repository](https://github.com/hill-a/stable-baselines).
5. **Optimal Reciprocal Collision Avoidance (ORCA)**: Jur van den Berg, Stephen J. Guy, Ming Lin, and Dinesh Manocha, *"Reciprocal n-Body Collision Avoidance"*, Robotics Research, Springer, 2011.
