# Reproduction Commands Log

This document records the exact reproducible commands for all environment setups, tests, benchmarks, scenario recordings, and analyses.

---

## 1. Environment Setup & Smoke Testing (Task 1)

### Automated Setup Script
Run once from the repository root to create the Python 3.6 environment, compile ASL robotics packages, and establish asset links:
```bash
cd "$REPO_ROOT"
bash setup_env.sh
```

### Minimal Verification Smoke Test (Random Actions, 50 Steps)
Runs 50 steps on both `NavRepTrainEnv` and `IANEnv`:
```bash
MPLCONFIGDIR=/tmp/matplotlib ./env/bin/python scripts_custom/smoke_test.py
```

### Pretrained Model Verification & Checksums
Validates file sizes, MD5 checksums, and weights for all 10 pretrained models:
```bash
MPLCONFIGDIR=/tmp/matplotlib ./env/bin/python scripts_custom/verify_models_and_checksums.py
```

### Pretrained Policy Cross-Testing in IANEnv (Repo Native Script)
Runs the upstream cross-testing script on compatible PPO models:
```bash
MPLCONFIGDIR=/tmp/matplotlib ./env/bin/python navrep/scripts/cross_test_navreptrain_in_ianenv.py \
    --backend VAE_LSTM --encoding VM --no-gpu --dry-run --n 11
```

---

## 2. Existing Scenarios Benchmark and Video Recordings (Task 2)

### Full Batch Evaluation and Recording Suite (11 Scenarios)
Evaluates 20 seeded episodes per scenario (seeds 100-119), finds representative success and failure seeds, records verified 640x640 H.264 MP4 videos with dynamic robot-following camera and PiP minimap, and enforces the frame quality gate (asserting std >= 15.0 at 10%, 25%, 50%, 75%, and 90% timestamps):
```bash
MPLCONFIGDIR=/tmp/matplotlib ./env/bin/python scripts_custom/evaluate_and_record_existing.py
```
Outputs produced:
- `submission/task2_videos/metrics_existing.csv`
- `submission/task2_videos/VIDEO_INDEX.md`
- `submission/task2_videos/<NN>_<scenario>_success.mp4`
- `submission/task2_videos/<NN>_<scenario>_failure.mp4`

### Individual Scenario Video Recording CLI
Records an individual scenario episode with custom seed, backend, and camera overlay:
```bash
# Example 1: irosasl1
MPLCONFIGDIR=/tmp/matplotlib ./env/bin/python scripts_custom/record_scenario.py \
    --scenario irosasl1 --seed 42 --backend VAE_LSTM --encoding VM --fps 12 \
    --out submission/task2_videos/03_irosasl1_test.mp4

# Example 2: navreptrain_train
MPLCONFIGDIR=/tmp/matplotlib ./env/bin/python scripts_custom/record_scenario.py \
    --scenario navreptrain_train --seed 1 --backend VAE_LSTM --encoding VM --fps 12 \
    --out submission/task2_videos/01_navreptrain_train_test.mp4

# Example 3: irosunity_scene_map1
MPLCONFIGDIR=/tmp/matplotlib ./env/bin/python scripts_custom/record_scenario.py \
    --scenario irosunity_scene_map1 --seed 14 --backend VAE_LSTM --encoding VM --fps 12 \
    --out submission/task2_videos/09_irosunity_scene_map1_test.mp4
```

---

## 3. Custom Scenarios, Benchmark, and Crowd Density Sweep (Task 3)

### Unit Test Suite Execution
Runs test cases verifying registry loading, custom environment instantiation, step/reset cycles, and observation/action space alignment across all 6 custom scenarios:
```bash
MPLCONFIGDIR=/tmp/matplotlib ./env/bin/python -m unittest tests/test_custom_scenarios.py
```

### Full Batch Custom Evaluation, Recording & Density Analysis Suite
Evaluates 20 seeded episodes per scenario/difficulty combination, records MP4 videos to `submission/task3_custom/videos/`, outputs `submission/task3_custom/metrics_custom.csv`, and executes the parametric crowd density stress test sweep generating `submission/task3_custom/analysis/density_vs_success.png`:
```bash
MPLCONFIGDIR=/tmp/matplotlib ./env/bin/python scripts_custom/evaluate_and_record_custom.py
```

### Custom Scenario CLI (`navrep_custom.run_custom`)
```bash
# Run interactive benchmark evaluation:
MPLCONFIGDIR=/tmp/matplotlib ./env/bin/python -m navrep_custom.run_custom \
    --scenario mallmate --difficulty MEDIUM --episodes 5

# Record video:
MPLCONFIGDIR=/tmp/matplotlib ./env/bin/python -m navrep_custom.run_custom \
    --scenario mallmate --difficulty MEDIUM --episodes 2 --record submission/task3_custom/videos/01_mallmate_medium.mp4

# Export 2D blueprint map preview:
MPLCONFIGDIR=/tmp/matplotlib ./env/bin/python -m navrep_custom.run_custom \
    --scenario mallmate --export-map submission/task3_custom/maps/mallmate.png

# Parametric crowd density sweep:
MPLCONFIGDIR=/tmp/matplotlib ./env/bin/python -m navrep_custom.run_custom \
    --density-sweep diwalibazaar --out-plot submission/task3_custom/analysis/density_vs_success.png
```
