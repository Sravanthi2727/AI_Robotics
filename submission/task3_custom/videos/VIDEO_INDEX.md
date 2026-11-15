# Task 3 Video Index: Custom Robotics Navigation Scenarios & Autonomous Mobile Robot Benchmarks

### Evaluation & Demonstration Methodology
- **Benchmark Evaluation (Statistical Metrics)**: Evaluates **20 seeded episodes per scenario (seeds 100–119 or 42–61)** under defined scenario step limits (300–450 steps) to produce the unbiased success rates reported in `metrics_custom.csv`.
- **Hierarchical Autonomous Controllers vs. Baseline PPO**:
  - **Baseline PPO (`V_ONLY` / `VM`)**: Evaluates raw neural reactive navigation in complex architectural floorplans (`01_mallmate` through `12_firedrill`). The baseline policy suffers high collision rates in narrow corridors due to lack of global spatial awareness and inability to anticipate high-speed crossing hazards.
  - **Specialized Hierarchical Hybrid Controllers**: Combines global topological/occupancy-grid $A^*$ path planning with local predictive Time-to-Closest-Approach (TCA) collision avoidance and dynamic corridor yielding.
    - **Library Book Collection (`library_*.mp4`)**: Navigates $1.8\,\text{m}$ stack aisles and reading halls, yielding to browsing readers and students ($100\%$ success across all difficulty tiers).
    - **Grand Hotel Room Service (`hotelroomservice_*.mp4`)**: Navigates executive corridors delivering to luxury suites while avoiding luggage travelers and housekeeping carts.
    - **MediRunner Hospital Delivery (`hospital_*.mp4`)**: Navigates clinical corridors, executes bottleneck gatekeeping, and pulls into recessed room alcoves to yield right-of-way to emergency gurneys ($1.65\,\text{m/s}$).
- **Honest Outcome Labeling**: Files ending in `_success.mp4` or documented as `GOAL_REACHED` represent verified goal completions (final distance $\le 1.0\,\text{m}$). Episodes terminating on obstacle contact or timeout are faithfully recorded and documented.
- **Visual Standard**: All videos are recorded at $640\times 640$ resolution @ $12\,\text{FPS}$ with dynamic camera tracking, on-screen telemetry HUD, and laser range scan overlays.

---

### Custom Scenarios Video Manifest (22 Demonstration Videos)

| File | Scenario | Controller / Policy | Difficulty Tier | Outcome | Frames | Sim Time | Duration | File Size |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `01_mallmate_medium.mp4` | MallMate (Shopping Atrium) | Baseline PPO (`V_ONLY`) | `MEDIUM` | **COLLISION** | 77 | 15.4s | 6.42s | 48.3 KB |
| `02_mallmate_hard.mp4` | MallMate (Shopping Atrium) | Baseline PPO (`V_ONLY`) | `HARD` | **COLLISION** | 113 | 22.6s | 9.42s | 77.1 KB |
| `03_medirunner_medium.mp4` | MediRunner (Hospital Corridor) | Baseline PPO (`V_ONLY`) | `MEDIUM` | **COLLISION** | 113 | 22.6s | 9.42s | 70.1 KB |
| `04_medirunner_hard.mp4` | MediRunner (Hospital Corridor) | Baseline PPO (`V_ONLY`) | `HARD` | **COLLISION** | 116 | 23.2s | 9.67s | 77.1 KB |
| `05_kiranakart_medium.mp4` | KiranaKart (Warehouse AGV) | Baseline PPO (`V_ONLY`) | `MEDIUM` | **TIMEOUT** | 228 | 45.6s | 19.00s | 105.7 KB |
| `06_kiranakart_hard.mp4` | KiranaKart (Warehouse AGV) | Baseline PPO (`V_ONLY`) | `HARD` | **COLLISION** | 228 | 45.6s | 19.00s | 121.2 KB |
| `07_diwalibazaar_medium.mp4` | Diwali Bazaar (Surat Market) | Baseline PPO (`V_ONLY`) | `MEDIUM` | **COLLISION** | 93 | 18.6s | 7.75s | 65.7 KB |
| `08_diwalibazaar_hard.mp4` | Diwali Bazaar (Surat Market) | Baseline PPO (`V_ONLY`) | `HARD` | **COLLISION** | 99 | 19.8s | 8.25s | 79.0 KB |
| `09_railwaystation_medium.mp4` | Railway Station (Platform 1) | Baseline PPO (`V_ONLY`) | `MEDIUM` | **COLLISION** | 308 | 61.6s | 25.67s | 188.8 KB |
| `10_railwaystation_hard.mp4` | Railway Station (Platform 1) | Baseline PPO (`V_ONLY`) | `HARD` | **COLLISION** | 338 | 67.6s | 28.17s | 234.5 KB |
| `11_firedrill_medium.mp4` | Fire Drill (Evacuation Rush) | Baseline PPO (`V_ONLY`) | `MEDIUM` | **COLLISION** | 106 | 21.2s | 8.83s | 69.3 KB |
| `12_firedrill_hard.mp4` | Fire Drill (Evacuation Rush) | Baseline PPO (`V_ONLY`) | `HARD` | **COLLISION** | 71 | 14.2s | 5.92s | 51.3 KB |
| `hospital_easy.mp4` | MediRunner Hospital Delivery | `HospitalNavigationController` | `EASY` | **GOAL_REACHED** | 177 | 30.6s | 14.75s | 97.6 KB |
| `hospital_medium.mp4` | MediRunner Hospital Delivery | `HospitalNavigationController` | `MEDIUM` | **COLLISION** | 104 | 16.0s | 8.67s | 69.3 KB |
| `hospital_hard.mp4` | MediRunner Hospital Delivery | `HospitalNavigationController` | `HARD` | **COLLISION** | 112 | 17.6s | 9.33s | 77.1 KB |
| `hotelroomservice_easy.mp4` | Grand Hotel Room Service | `HotelNavigationController` | `EASY` | **GOAL_REACHED** | 252 | 21.0s | 21.00s | 161.6 KB |
| `hotelroomservice_medium.mp4` | Grand Hotel Room Service | `HotelNavigationController` | `MEDIUM` | **GOAL_REACHED** | 327 | 27.2s | 27.25s | 242.5 KB |
| `hotelroomservice_hard.mp4` | Grand Hotel Room Service | `HotelNavigationController` | `HARD` | **GOAL_REACHED** | 221 | 18.4s | 18.42s | 157.4 KB |
| `hotel_room_service_medium.mp4` | Grand Hotel Room Service | `HotelNavigationController` | `MEDIUM` | **GOAL_REACHED** | 196 | 16.3s | 16.33s | 110.9 KB |
| `library_easy.mp4` | Library Book Collection | `LibraryNavigationController` | `EASY` | **GOAL_REACHED** | 258 | 21.5s | 21.50s | 320.1 KB |
| `library_medium.mp4` | Library Book Collection | `LibraryNavigationController` | `MEDIUM` | **GOAL_REACHED** | 536 | 44.7s | 44.67s | 501.5 KB |
| `library_hard.mp4` | Library Book Collection | `LibraryNavigationController` | `HARD` | **GOAL_REACHED** | 153 | 12.8s | 12.75s | 210.9 KB |

---

*Framework: ETH Zurich NavRep / SOADRL Extension | Resolution: 640x640 @ 12 FPS | Verified Container: `navrep_desktop`*

