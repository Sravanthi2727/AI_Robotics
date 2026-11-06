# Reproduction Commands Log

This file records the exact reproducible commands for all tasks, benchmarks, tests, scenario recordings, and analysis.

---

## Phase 1 / Task 1: Environment Setup and Smoke Testing

### 1. Minimal Smoke Test (Random Actions, 50 Steps)
```bash
HOME=/home/sravanthi/Desktop MPLCONFIGDIR=/tmp/matplotlib /home/sravanthi/Desktop/navrep/env/bin/python scripts_custom/smoke_test.py
```

### 2. Pretrained Policy Cross-Testing in IANEnv
```bash
HOME=/home/sravanthi/Desktop MPLCONFIGDIR=/tmp/matplotlib /home/sravanthi/Desktop/navrep/env/bin/python -m navrep.scripts.cross_test_navreptrain_in_ianenv --backend VAE_LSTM --encoding V_ONLY --no-gpu --n 1
```

---

## Phase 2 / Task 2: Existing Scenarios Benchmark and Video Recordings

### 1. Full Batch Evaluation and Recording Suite (11 Scenarios)
Evaluates 20 seeded episodes per scenario (computing success, collision, timeout rates, mean time-to-goal) and writes `submission/task2_videos/metrics_existing.csv`. Generates high-definition telemetry-annotated MP4 recordings for all 11 scenarios in `submission/task2_videos/`:
```bash
HOME=/home/sravanthi/Desktop MPLCONFIGDIR=/tmp/matplotlib /home/sravanthi/Desktop/navrep/env/bin/python scripts_custom/evaluate_and_record_existing.py
```

### 2. Individual Scenario Video Recording (CLI)
Example recording 2 episodes of `navreptrain_train`:
```bash
HOME=/home/sravanthi/Desktop MPLCONFIGDIR=/tmp/matplotlib /home/sravanthi/Desktop/navrep/env/bin/python scripts_custom/record_scenario.py --scenario navreptrain_train --episodes 2 --seed 42 --fps 12 --out submission/task2_videos/01_navreptrain_train.mp4
```

Example recording 2 episodes of `irosasl_office_j1`:
```bash
HOME=/home/sravanthi/Desktop MPLCONFIGDIR=/tmp/matplotlib /home/sravanthi/Desktop/navrep/env/bin/python scripts_custom/record_scenario.py --scenario irosasl_office_j1 --episodes 2 --seed 55 --fps 12 --out submission/task2_videos/06_irosasl_office_j1.mp4
```

Example recording 2 episodes of `irosunity_scene_map1`:
```bash
HOME=/home/sravanthi/Desktop MPLCONFIGDIR=/tmp/matplotlib /home/sravanthi/Desktop/navrep/env/bin/python scripts_custom/record_scenario.py --scenario irosunity_scene_map1 --episodes 2 --seed 77 --fps 12 --out submission/task2_videos/09_irosunity_scene_map1.mp4
```

---

## Phase 3 / Task 3: Custom Scenarios, Benchmark, and Crowd Density Sweep

### 1. Unit Test Suite Execution
Runs test cases verifying registry loading, custom environment instantiation, step/reset cycle, and observation/action space alignment across all 6 scenarios and difficulty ladder tiers:
```bash
HOME=/home/sravanthi/Desktop MPLCONFIGDIR=/tmp/matplotlib /home/sravanthi/Desktop/navrep/env/bin/python -m unittest tests/test_custom_scenarios.py
```

### 2. Full Batch Custom Evaluation, Recording & Density Analysis Suite
Evaluates 20 seeded episodes per scenario/difficulty combination, records 12 MP4 videos (>5s duration each) to `submission/task3_custom/videos/`, outputs `submission/task3_custom/metrics_custom.csv`, and executes the parametric crowd density stress test sweep generating `submission/task3_custom/analysis/density_vs_success.png`:
```bash
HOME=/home/sravanthi/Desktop MPLCONFIGDIR=/tmp/matplotlib /home/sravanthi/Desktop/navrep/env/bin/python scripts_custom/evaluate_and_record_custom.py
```

### 3. Custom Scenario CLI (`navrep_custom.run_custom`)

#### Running an Interactive / Benchmark Evaluation of a Specific Scenario:
```bash
HOME=/home/sravanthi/Desktop MPLCONFIGDIR=/tmp/matplotlib /home/sravanthi/Desktop/navrep/env/bin/python -m navrep_custom.run_custom --scenario mallmate --difficulty MEDIUM --episodes 5
```

#### Recording Video for a Specific Scenario & Difficulty:
```bash
HOME=/home/sravanthi/Desktop MPLCONFIGDIR=/tmp/matplotlib /home/sravanthi/Desktop/navrep/env/bin/python -m navrep_custom.run_custom --scenario mallmate --difficulty MEDIUM --episodes 2 --record submission/task3_custom/videos/01_mallmate_medium.mp4
```

#### Exporting 2D Top-Down Blueprint Map Previews:
```bash
HOME=/home/sravanthi/Desktop MPLCONFIGDIR=/tmp/matplotlib /home/sravanthi/Desktop/navrep/env/bin/python -m navrep_custom.run_custom --scenario mallmate --export-map submission/task3_custom/maps/mallmate.png
```
*(Supports `--scenario` options: `mallmate`, `medirunner`, `kiranakart`, `diwalibazaar`, `railwaystation`, `firedrill`)*

#### Running the Parametric Crowd Density Sweep:
```bash
HOME=/home/sravanthi/Desktop MPLCONFIGDIR=/tmp/matplotlib /home/sravanthi/Desktop/navrep/env/bin/python -m navrep_custom.run_custom --density-sweep diwalibazaar --out-plot submission/task3_custom/analysis/density_vs_success.png
```
