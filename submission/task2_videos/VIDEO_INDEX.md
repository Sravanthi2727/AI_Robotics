# Task 2 Video Index: ETH NavRep Scenarios Benchmark

### Evaluation & Demonstration Methodology
- **Benchmark Evaluation (Statistical Metrics)**: Evaluates **20 seeded episodes per scenario (seeds 100–119)** under native/documented step limits (175 steps for `NavRepTrainEnv`, 500 steps for `IANEnv`) to produce the unbiased success rates reported in `metrics_existing.csv`.
- **Demonstration Video Search (Qualitative Evidence)**: Conducts a separate, exhaustive **60-seed sweep (seeds 0–59)** to find representative visual demonstrations.
- **Seed Sweep Distinction**: A scenario with an aggregate benchmark success rate of **0.0%** across seeds 100–119 can still yield a verified `_success.mp4` demonstration if the policy safely reached the goal on an isolated seed during the 60-seed search (e.g., `seed 56` for `irosasl3` and `seed 34` for `irosunity_scene_map3`).
- **Honest Labeling**: Files ending in `_success.mp4` represent true `GOAL_REACHED` episodes (final distance $\le 1.0\text{m}$). Scenarios where the policy could not reach the goal on any of the 60 candidate seeds are honestly labeled `_bestprogress.mp4` (selected by smallest final Euclidean distance to goal). Every video includes the dynamic 20m x 20m camera, top telemetry HUD, laser scan points, and PiP minimap, with static freeze frames strictly bounded ($\le 12$ frames).

---

| File | Scenario | Outcome | Seed | Steps | Sim Time | Final Distance | Duration |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `01_navreptrain_train_failure.mp4` | navreptrain_train | **TIMEOUT** | `23` | 175 | 35.0s | 7.41m | 15.67s |
| `01_navreptrain_train_success.mp4` | navreptrain_train | **GOAL_REACHED** | `1` | 94 | 18.8s | 0.33m | 8.92s |
| `02_navreptrain_test_failure.mp4` | navreptrain_test | **TIMEOUT** | `26` | 175 | 35.0s | 6.75m | 15.67s |
| `02_navreptrain_test_success.mp4` | navreptrain_test | **GOAL_REACHED** | `20` | 43 | 8.6s | 0.14m | 4.67s |
| `03_irosasl1_bestprogress.mp4` | irosasl1 | **COLLISION** | `23` | 97 | 19.4s | 5.35m | 9.17s |
| `03_irosasl1_failure.mp4` | irosasl1 | **COLLISION** | `115` | 96 | 19.2s | 5.21m | 9.08s |
| `04_irosasl2_failure.mp4` | irosasl2 | **COLLISION** | `4` | 383 | 76.6s | 2.49m | 33.0s |
| `04_irosasl2_success.mp4` | irosasl2 | **GOAL_REACHED** | `26` | 259 | 51.8s | 0.49m | 22.67s |
| `05_irosasl3_failure.mp4` | irosasl3 | **COLLISION** | `0` | 156 | 31.2s | 17.73m | 14.08s |
| `05_irosasl3_success.mp4` | irosasl3 | **GOAL_REACHED** | `56` | 401 | 80.2s | 0.43m | 34.5s |
| `06_irosasl_office_j1_bestprogress.mp4` | irosasl_office_j1 | **COLLISION** | `23` | 151 | 30.2s | 4.94m | 13.67s |
| `06_irosasl_office_j1_failure.mp4` | irosasl_office_j1 | **COLLISION** | `0` | 86 | 17.2s | 5.89m | 8.25s |
| `07_irosasl_office_j2_failure.mp4` | irosasl_office_j2 | **COLLISION** | `2` | 199 | 39.8s | 22.77m | 17.67s |
| `07_irosasl_office_j2_success.mp4` | irosasl_office_j2 | **GOAL_REACHED** | `47` | 461 | 92.2s | 0.5m | 39.5s |
| `08_irosasl_office_j3_bestprogress.mp4` | irosasl_office_j3 | **COLLISION** | `32` | 119 | 23.8s | 2.19m | 11.0s |
| `08_irosasl_office_j3_failure.mp4` | irosasl_office_j3 | **COLLISION** | `1` | 102 | 20.4s | 6.01m | 9.58s |
| `09_irosunity_scene_map1_bestprogress.mp4` | irosunity_scene_map1 | **COLLISION** | `39` | 125 | 25.0s | 5.69m | 11.5s |
| `09_irosunity_scene_map1_failure.mp4` | irosunity_scene_map1 | **COLLISION** | `0` | 215 | 43.0s | 5.79m | 19.0s |
| `10_irosunity_scene_map2_failure.mp4` | irosunity_scene_map2 | **COLLISION** | `17` | 331 | 66.2s | 3.84m | 28.67s |
| `10_irosunity_scene_map2_success.mp4` | irosunity_scene_map2 | **GOAL_REACHED** | `58` | 291 | 58.2s | 0.41m | 25.33s |
| `11_irosunity_scene_map3_failure.mp4` | irosunity_scene_map3 | **COLLISION** | `1` | 238 | 47.6s | 4.58m | 20.92s |
| `11_irosunity_scene_map3_success.mp4` | irosunity_scene_map3 | **GOAL_REACHED** | `34` | 252 | 50.4s | 0.49m | 22.08s |


*Evaluation Controller: `PPO_VAE_LSTM_VM_V32M512_ckpt` | Sim Resolution: 640x640 @ 12 FPS*
