# Reproduction Commands Log

This document records the exact reproducible commands for all environment setups, tests, benchmarks, scenario recordings, and analyses.

---

## 1. Environment Setup & Smoke Testing (Task 1)

### Option A: Standard Docker Workflow (Recommended for Cross-Platform Execution)
All team members (macOS, Windows, Linux) can run the pre-configured container without installing local Python 3.6:
```bash
# Start desktop container in background on port 6080
docker compose up -d

# Minimal verification smoke test (50 random steps on NavRepTrainEnv and IANEnv)
docker exec navrep_desktop python scripts_custom/smoke_test.py

# Pretrained model and checkpoint MD5 verification
docker exec navrep_desktop python scripts_custom/verify_models_and_checksums.py
```
*(Interactive desktop is accessible via browser at `http://localhost:6080/vnc.html`)*

### Option B: Native Standalone Setup Script
```bash
# Clone and enter repository
cd "$REPO_ROOT"
bash setup_env.sh

# Run smoke test in native virtualenv
HOME=/tmp/home MPLCONFIGDIR=/tmp/matplotlib ./env/bin/python scripts_custom/smoke_test.py

# Verify models and checksums
HOME=/tmp/home MPLCONFIGDIR=/tmp/matplotlib ./env/bin/python scripts_custom/verify_models_and_checksums.py
```

---

## 2. Existing Scenarios Benchmark and Video Recordings (Task 2)

### 2.1 Fast Verification & Table Reproduction (Reproduces Existing Results)
*These commands run instantaneously and do NOT perform long simulation rollouts.*

#### A. Automated Video Quality Verification Suite
Validates suffix/outcome consistency, frozen frames ($\le 20$), standard deviation $\ge 15.0$ at 10/25/50/75/90%, and duration $\ge 8.0\text{ s}$ across all 22 videos:
* **Docker**:
  ```bash
  docker exec navrep_desktop python scripts_custom/verify_videos.py
  ```
* **Native**:
  ```bash
  HOME=/tmp/home MPLCONFIGDIR=/tmp/matplotlib ./env/bin/python scripts_custom/verify_videos.py
  ```

#### B. Pretrained Model Verification & Checksums
Validates file sizes, MD5 checksums, and weights for all 10 pretrained models:
* **Docker**:
  ```bash
  docker exec navrep_desktop python scripts_custom/verify_models_and_checksums.py
  ```
* **Native**:
  ```bash
  HOME=/tmp/home MPLCONFIGDIR=/tmp/matplotlib ./env/bin/python scripts_custom/verify_models_and_checksums.py
  ```

#### C. Cross-Test Baseline Parser
Parses upstream evaluation pickle files and outputs `results/repo_script_cross_test.json`:
* **Docker**:
  ```bash
  docker exec navrep_desktop python scripts_custom/log_repo_cross_test.py
  ```
* **Native**:
  ```bash
  HOME=/tmp/home MPLCONFIGDIR=/tmp/matplotlib ./env/bin/python scripts_custom/log_repo_cross_test.py
  ```

#### D. Fast Metric & Video Index Re-Compilation
Re-compiles `submission/task2_videos/metrics_existing.csv` and `VIDEO_INDEX.md` directly from existing sidecars and per-scenario JSON files without re-running simulations:
* **Docker**:
  ```bash
  docker exec navrep_desktop python scripts_custom/remediate_and_record.py --compile-only
  ```
* **Native**:
  ```bash
  HOME=/tmp/home MPLCONFIGDIR=/tmp/matplotlib ./env/bin/python scripts_custom/remediate_and_record.py --compile-only
  ```

---

### 2.2 Full Simulation & Recording Suite (Generates New Recordings)
*WARNING: These commands run multi-seed simulations and video rendering. Do NOT run unless re-recording is required.*

#### A. Single Scenario 20-Episode Benchmark & 60-Seed Sweep Recording
Evaluates 20 seeded episodes (seeds 100–119), runs a 60-seed sweep (seeds 0–59), and records success/best-progress and failure MP4 videos:
* **Docker**:
  ```bash
  docker exec navrep_desktop python scripts_custom/remediate_and_record.py --scenario navreptrain_train
  ```
* **Native**:
  ```bash
  HOME=/tmp/home MPLCONFIGDIR=/tmp/matplotlib ./env/bin/python scripts_custom/remediate_and_record.py --scenario navreptrain_train
  ```

#### B. Parallel Orchestrator (All 11 Scenarios)
Runs 3 concurrent worker processes across all 11 scenarios:
* **Docker**:
  ```bash
  docker exec navrep_desktop python scripts_custom/dev/run_parallel_remediation.py
  ```
* **Native**:
  ```bash
  HOME=/tmp/home MPLCONFIGDIR=/tmp/matplotlib ./env/bin/python scripts_custom/dev/run_parallel_remediation.py
  ```

#### C. Individual Scenario Custom Video Recording CLI
Records an individual episode with custom seed, camera overlay, and output path:
* **Docker**:
  ```bash
  # Example: irosasl1
  docker exec navrep_desktop python scripts_custom/record_scenario.py \
      --scenario irosasl1 --seed 42 --backend VAE_LSTM --encoding VM --fps 12 \
      --out submission/task2_videos/03_irosasl1_test.mp4
  ```
* **Native**:
  ```bash
  # Example: irosasl1
  HOME=/tmp/home MPLCONFIGDIR=/tmp/matplotlib ./env/bin/python scripts_custom/record_scenario.py \
      --scenario irosasl1 --seed 42 --backend VAE_LSTM --encoding VM --fps 12 \
      --out submission/task2_videos/03_irosasl1_test.mp4
  ```

#### D. Upstream Policy Cross-Testing in IANEnv (ETH Zurich Reference Script)
Runs the upstream cross-testing script on compatible PPO models:
* **Docker**:
  ```bash
  docker exec navrep_desktop python -m navrep.scripts.cross_test_navreptrain_in_ianenv \
      --backend VAE_LSTM --encoding VM --no-gpu --n 20
  ```
* **Native**:
  ```bash
  HOME=/tmp/home MPLCONFIGDIR=/tmp/matplotlib ./env/bin/python -m navrep.scripts.cross_test_navreptrain_in_ianenv \
      --backend VAE_LSTM --encoding VM --no-gpu --n 20
  ```

---

## 3. Custom Scenarios, Benchmark, and Crowd Density Sweep (Task 3)

### Unit Test Suite Execution
Runs test cases verifying registry loading, custom environment instantiation, step/reset cycles, and observation/action space alignment across all 6 custom scenarios:
* **Docker**:
  ```bash
  docker exec navrep_desktop python -m unittest tests/test_custom_scenarios.py
  ```
* **Native**:
  ```bash
  HOME=/tmp/home MPLCONFIGDIR=/tmp/matplotlib ./env/bin/python -m unittest tests/test_custom_scenarios.py
  ```

### Full Batch Custom Evaluation, Recording & Density Analysis Suite
Evaluates 20 seeded episodes per scenario/difficulty combination, records MP4 videos to `submission/task3_custom/videos/`, outputs `submission/task3_custom/metrics_custom.csv`, and executes the parametric crowd density stress test sweep generating `submission/task3_custom/analysis/density_vs_success.png`:
* **Docker**:
  ```bash
  docker exec navrep_desktop python scripts_custom/evaluate_and_record_custom.py
  ```
* **Native**:
  ```bash
  HOME=/tmp/home MPLCONFIGDIR=/tmp/matplotlib ./env/bin/python scripts_custom/evaluate_and_record_custom.py
  ```

### Custom Scenario CLI (`navrep_custom.run_custom`)
* **Docker**:
  ```bash
  # Run interactive benchmark evaluation:
  docker exec navrep_desktop python -m navrep_custom.run_custom \
      --scenario mallmate --difficulty MEDIUM --episodes 5

  # Record video:
  docker exec navrep_desktop python -m navrep_custom.run_custom \
      --scenario mallmate --difficulty MEDIUM --episodes 2 --record submission/task3_custom/videos/01_mallmate_medium.mp4

  # Export 2D blueprint map preview:
  docker exec navrep_desktop python -m navrep_custom.run_custom \
      --scenario mallmate --export-map submission/task3_custom/maps/mallmate.png

  # Parametric crowd density sweep:
  docker exec navrep_desktop python -m navrep_custom.run_custom \
      --density-sweep diwalibazaar --out-plot submission/task3_custom/analysis/density_vs_success.png
  ```
* **Native**:
  ```bash
  # Run interactive benchmark evaluation:
  HOME=/tmp/home MPLCONFIGDIR=/tmp/matplotlib ./env/bin/python -m navrep_custom.run_custom \
      --scenario mallmate --difficulty MEDIUM --episodes 5
  ```
