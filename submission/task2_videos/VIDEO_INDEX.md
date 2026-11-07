# Task 2 Video Index: ETH NavRep Scenarios Benchmark

All 22 videos (11 success/best-progress and 11 honest failure/near-miss) were recorded using the robot-centric camera, top telemetry HUD, PiP minimap, dynamic human velocity vectors, and laser beam points. Every video has been programmatically verified via the frame quality gate (asserting standard deviation >= 15.0 at 10%, 25%, 50%, 75%, and 90% timestamps).

| File | Scenario | Outcome | Seed | Duration | What to Look For |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `01_navreptrain_train_success.mp4` | navreptrain_train | **SUCCESS** | `1` | 8.0s | Robot reaches goal marker smoothly while avoiding dynamic humans and walls |
| `01_navreptrain_train_failure.mp4` | navreptrain_train | **COLLISION** | `2` | 8.0s | Near-miss or collision with dynamic human/obstacle; robot damage and status badge clearly displayed |
| `02_navreptrain_test_success.mp4` | navreptrain_test | **SUCCESS** | `3` | 10.2s | Robot reaches goal marker smoothly while avoiding dynamic humans and walls |
| `02_navreptrain_test_failure.mp4` | navreptrain_test | **COLLISION** | `1` | 10.1s | Near-miss or collision with dynamic human/obstacle; robot damage and status badge clearly displayed |
| `03_irosasl1_success.mp4` | irosasl1 | **BEST PROGRESS (3.4m to goal, COLLISION)** | `39` | 8.1s | Best achievable trajectory: advanced to 3.4m of goal before collision |
| `03_irosasl1_failure.mp4` | irosasl1 | **COLLISION** | `1` | 8.0s | Near-miss or collision with dynamic human/obstacle; robot damage and status badge clearly displayed |
| `04_irosasl2_success.mp4` | irosasl2 | **BEST PROGRESS (1.2m to goal, TIMEOUT)** | `26` | 22.4s | Best achievable trajectory: advanced to 1.2m of goal before timeout |
| `04_irosasl2_failure.mp4` | irosasl2 | **TIMEOUT** | `1` | 22.4s | Near-miss or collision with dynamic human/obstacle; robot damage and status badge clearly displayed |
| `05_irosasl3_success.mp4` | irosasl3 | **BEST PROGRESS (7.6m to goal, COLLISION)** | `29` | 20.0s | Best achievable trajectory: advanced to 7.6m of goal before collision |
| `05_irosasl3_failure.mp4` | irosasl3 | **TIMEOUT** | `1` | 22.4s | Near-miss or collision with dynamic human/obstacle; robot damage and status badge clearly displayed |
| `06_irosasl_office_j1_success.mp4` | irosasl_office_j1 | **BEST PROGRESS (4.7m to goal, COLLISION)** | `29` | 8.0s | Best achievable trajectory: advanced to 4.7m of goal before collision |
| `06_irosasl_office_j1_failure.mp4` | irosasl_office_j1 | **COLLISION** | `1` | 14.1s | Near-miss or collision with dynamic human/obstacle; robot damage and status badge clearly displayed |
| `07_irosasl_office_j2_success.mp4` | irosasl_office_j2 | **BEST PROGRESS (22.0m to goal, TIMEOUT)** | `6` | 22.4s | Best achievable trajectory: advanced to 22.0m of goal before timeout |
| `07_irosasl_office_j2_failure.mp4` | irosasl_office_j2 | **COLLISION** | `1` | 8.0s | Near-miss or collision with dynamic human/obstacle; robot damage and status badge clearly displayed |
| `08_irosasl_office_j3_success.mp4` | irosasl_office_j3 | **BEST PROGRESS (2.7m to goal, COLLISION)** | `5` | 11.7s | Best achievable trajectory: advanced to 2.7m of goal before collision |
| `08_irosasl_office_j3_failure.mp4` | irosasl_office_j3 | **COLLISION** | `1` | 10.1s | Near-miss or collision with dynamic human/obstacle; robot damage and status badge clearly displayed |
| `09_irosunity_scene_map1_success.mp4` | irosunity_scene_map1 | **BEST PROGRESS (5.7m to goal, COLLISION)** | `6` | 8.0s | Best achievable trajectory: advanced to 5.7m of goal before collision |
| `09_irosunity_scene_map1_failure.mp4` | irosunity_scene_map1 | **COLLISION** | `1` | 9.9s | Near-miss or collision with dynamic human/obstacle; robot damage and status badge clearly displayed |
| `10_irosunity_scene_map2_success.mp4` | irosunity_scene_map2 | **BEST PROGRESS (11.6m to goal, TIMEOUT)** | `6` | 22.4s | Best achievable trajectory: advanced to 11.6m of goal before timeout |
| `10_irosunity_scene_map2_failure.mp4` | irosunity_scene_map2 | **TIMEOUT** | `1` | 22.4s | Near-miss or collision with dynamic human/obstacle; robot damage and status badge clearly displayed |
| `11_irosunity_scene_map3_success.mp4` | irosunity_scene_map3 | **BEST PROGRESS (7.2m to goal, TIMEOUT)** | `2` | 22.4s | Best achievable trajectory: advanced to 7.2m of goal before timeout |
| `11_irosunity_scene_map3_failure.mp4` | irosunity_scene_map3 | **COLLISION** | `1` | 21.4s | Near-miss or collision with dynamic human/obstacle; robot damage and status badge clearly displayed |


*Evaluation Controller: `PPO_VAE_LSTM_VM_V32M512_ckpt` | Sim Resolution: 640x640 @ 12 FPS*
