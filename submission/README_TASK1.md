# Task 1: NavRep Environment Setup, Audit, and Execution Report

**Verified on**: November 7, 2026 (Ubuntu 24.04 LTS x86_64, Python 3.6.15)

---

## 1. Overview of the Project and Paper
ETH Zurich's **NavRep** (*NavRep: Unsupervised Representations for Reinforcement Learning of Robot Navigation in Dynamic Human Environments*) investigates self-supervised representation learning for reinforcement learning in dynamic robot navigation. Rather than mapping high-dimensional raw lidar streams directly to motor commands with end-to-end RL, NavRep decouples perception into world models and navigation policies:
1. **Perception & World Models (V & M modules)**:
   - **V-Model (ConvVAE 2D / 1D)**: Compresses raw 1080-beam lidar scans into a 32- or 64-dimensional latent state $z$.
   - **M-Model (MDN-RNN / LSTM & GPT)**: Projects sequences of latent codes and robot actions forward in time, estimating dynamic transitions and maintaining a recurrent belief state $h$ (512 dimensions).
2. **Controller (C-module)**:
   - A Proximal Policy Optimization (PPO2) policy receives the compressed state $(z, h, \text{robot\_state})$ to predict velocity commands $(v_x, v_y, \omega)$.
3. **Key Finding**:
   - In our zero-shot tests, the VM policy reached the goal in 45–65% of episodes in the NavRepTrain train/test environments and 0–50% in the IAN environments (see `metrics_existing.csv`); a sim-to-sim domain gap is a likely, untested explanation for the lower transfer performance observed in IAN environments.

---

## 2. System Hardware and Environment
- **Operating System**: Linux (Ubuntu 24.04.4 LTS, kernel `6.17.0-14-generic x86_64`)
- **CPU**: 12th Gen Intel(R) Core(TM) i5-12450H (4 vCPUs allocated, 2.50 GHz, AVX2 / FMA support)
- **RAM**: 7.8 GiB total (~3.8 GiB available)
- **GPU / Acceleration**: Headless CPU execution (`--no-gpu`; all models run on CPU)
- **Architecture**: x86_64 Little Endian

---

## 3. Pinned Dependency Versions
Key dependency versions verified against `submission/pip_freeze.txt` (running on Python 3.6.15 Conda-forge runtime):
- `python==3.6.15` (Conda-forge runtime)
- `tensorflow==1.13.2`
- `torch==1.6.0+cpu`, `torchvision==0.7.0+cpu`
- `Keras==2.3.1`
- `stable-baselines==2.10.0`
- `gym==0.15.7`
- `cloudpickle==1.6.0`
- `numpy==1.19.5`
- `scipy==1.5.4`
- `matplotlib==3.3.4`
- `opencv-python==4.5.5.64`
- `pyglet==1.5.0`
- `Cython==0.29.28`
- `numba==0.53.1`
- `pandas==1.1.0`
- `imageio==2.15.0`, `imageio-ffmpeg==0.5.1`
- `importlib-resources==5.4.0`
- `pyIAN==0.0.3`, `asl-pepper-2d-sim==0.0.5`, `soadrl-crowdnav==0.0.3`
- `pyniel==0.2`, `pymap2d==0.1.15`, `pylidar2d==0.1`
- `pyrvo2-danieldugas==0.0.2`, `pyrangelibc-danieldugas==0.0.2`
- `rospy==1.15.11`, `rosbag==1.15.11`, `tf==1.12.1.post3`, `tf2-ros==0.6.5`

---

## 4. Automated Installation (`setup_env.sh`)

A single standalone setup script [`setup_env.sh`](setup_env.sh) automates the complete installation from scratch. It requires no root privileges:

```bash
# 1. Clone repository and enter repository root
cd <path-to-cloned-repo>
export REPO_ROOT=$(pwd)

# 2. Run automated environment setup
bash setup_env.sh

# 3. Activate environment
export PATH="$REPO_ROOT/env/bin:$PATH"
export MPLCONFIGDIR=/tmp/matplotlib
```

### What `setup_env.sh` does step-by-step:
1. Downloads standalone `micromamba` into `./bin/` (isolated user-space package manager).
2. Creates `./env` with Python 3.6.15.
3. Installs `numpy==1.19.5` and `Cython==0.29.28` first (required to build ASL Cython extensions).
4. Installs the official CPU build of PyTorch (`torch==1.6.0+cpu`).
5. Installs pre-built `opencv-python==4.5.5.64` wheel.
6. Installs core ML stack: `tensorflow==1.13.2`, `keras==2.3.1`, `stable-baselines==2.10.0`, `cloudpickle==1.6.0`, `gym==0.15.7`, and `pyglet==1.5.0`.
7. Installs ETH Zurich ASL robotics libraries (`pyIAN`, `asl-pepper-2d-sim`, `soadrl-crowdnav`, `pyniel`, `pymap2d`, `pylidar2d`, `pyrvo2-danieldugas`, `pyrangelibc-danieldugas`).
8. Installs rospypi shims (`rospy`, `rosbag`, `tf`, `tf2_ros`) and NavRep in editable development mode (`pip install --no-build-isolation -e .`).
9. Copies bundled `external/frame_msgs` and `external/tf_bag` into site-packages.
10. Establishes the `~/navrep/models`, `~/maps`, `/tmp/home/navrep/models`, and `/tmp/home/maps` asset symlinks.
11. Checks FFmpeg and Xvfb availability (falls back to bundled `imageio-ffmpeg` binary and software offscreen rendering if absent, so Xvfb is not required), and executes the smoke test.

---

## 5. Model Asset Symlinks Convention
The upstream NavRep codebase hardcodes asset lookups using `os.path.expanduser("~/navrep/...")` and `os.path.expanduser("~/maps/...")`.

To ensure consistent resolution across all environments (including systems with read-only user directories or execution under `HOME=/tmp/home`):
```bash
# Ensure standard asset symlinks exist in your active HOME
mkdir -p "$HOME/navrep"
ln -sfn "$REPO_ROOT/models" "$HOME/navrep/models"
ln -sfn "$REPO_ROOT/maps" "$HOME/maps"

# Also create symlinks in /tmp/home for execution environments using HOME=/tmp/home (handled automatically by setup_env.sh)
mkdir -p /tmp/home/navrep
ln -sfn "$REPO_ROOT/models" /tmp/home/navrep/models
ln -sfn "$REPO_ROOT/maps" /tmp/home/maps
```

---

## 6. Modifications to Original Code or Configuration
As verified by `git diff --stat f824612 HEAD -- navrep crings external maps` (where `f824612` is the original upstream commit) and documented in `docs/CHANGES_LOG.md`:
**No original NavRep source files were modified.** Compatibility came entirely from pinned package versions, copying `external/frame_msgs` and `external/tf_bag` into site-packages, the `~/navrep/models` and `~/maps` symlinks, and new files only (`scripts_custom/`, `navrep_custom/`, `tests/`, `docs/`, `submission/`).

---

## 7. Challenges and Solutions

| # | Issue Encountered | Root Cause | Engineering Solution |
|---|---|---|---|
| 1 | Lack of root/sudo permissions for system packages | Standard machine without root privileges. | Deployed user-space `micromamba` binary into `$REPO_ROOT/bin/` to manage an isolated Python 3.6 environment. |
| 2 | Heavy network stall downloading default PyPI PyTorch | Default PyPI `torch==1.6.0` pulls 513 MB CUDA 10 binaries. | Pointed pip to official PyTorch CPU wheel (`torch==1.6.0+cpu` from `download.pytorch.org`), reducing download size to ~120 MB. |
| 3 | 40-minute source compilation for OpenCV | Modern PyPI releases lack Python 3.6 wheels, falling back to C++ source build. | Pinned and installed `opencv-python==4.5.5.64` pre-built binary wheel before installing dependencies. |
| 4 | `Syntax error in simple statement list (print "...")` in `RangeLibc.pyx` | Pip build isolation pulled modern Cython, rejecting Python 2 print statements in legacy Cython code. | Installed with `--no-build-isolation` to ensure the environment's pinned `Cython==0.29.28` was utilized. |
| 5 | `ModuleNotFoundError: No module named 'frame_msgs'` | `setup.py` maps `package_dir={'frame_msgs': ...}` which editable pip installs fail to link. | Explicitly copied `external/frame_msgs` and `external/tf_bag` directly into the environment's `site-packages`. |
| 6 | Serialized policy compatibility in Gym & Cloudpickle | Models serialized with Gym 0.15.7 failed on Gym >= 0.20; Cloudpickle 1.2.2 failed on `types.CellType`. | Pinned `gym==0.15.7` and `cloudpickle==1.6.0`. |
| 7 | `ModuleNotFoundError: No module named 'importlib.resources'` in `imageio_ffmpeg` | `importlib.resources` is Python 3.7+; Python 3.6 requires `importlib_resources`. | Dynamically mapped `sys.modules['importlib.resources'] = importlib_resources` in recording scripts. |
| 8 | IANEnv rendering bounds inverted & black screen | `pymap2d.get_extent_xy()` returns `[xmin, xmax, ymin, ymax]`; unpacking as `x_min, y_min, ...` inverted Y-span. Also, polygon index 0 is the bounding box. | Corrected coordinate unpacking, added dynamic robot-following 20m x 20m camera, PiP overview map, and laser points. |
| 9 | Robot stuck in place in initial recordings | Evaluating V_ONLY policy in deterministic mode led to local minima in symmetric arenas. | Upgraded evaluation to `VAE_LSTM VM` (`navreptrainencodedenv_2020_09_23__09_48_22_PPO_VAE_LSTM_VM_V32M512_ckpt.zip`), enabling recurrent temporal memory ($h$). |
| 10 | Artificial 250-step cap truncated valid navigation | A hardcoded 250-step limit prematurely timed out longer IANEnv scenarios (e.g., `irosasl2` requires ~55–60s = ~280–300 steps). | Removed artificial cap; aligned benchmarks with native limits: 175 steps (35.0s) for `NavRepTrainEnv` and 500 steps (100.0s, native 1000) for `IANEnv`. Success jumped from 0% to 45% on `irosasl2` and 50% on `irosunity_scene_map2`. |
| 11 | Misleading outcome labeling & early crash freezes | Initial videos contained mislabeled outcomes and long frozen-frame tails on early crashes. | Enforced strict labeling from sidecar metadata (`_success.mp4` strictly for `GOAL_REACHED`; `_bestprogress.mp4` for non-reaching sweeps). Constrained hold padding to $\le 12$ frames and selected failure runs lasting $\ge 30$ steps and $\ge 8.0\text{ s}$. |
| 12 | Frozen-frame false positives during slow motion | Scaling image diff by $255.0$ caused whole-frame mean intensity changes of ~0.003 to be flagged as frozen even during active robot motion. | Formulated unscaled mean absolute intensity difference: $\Delta_{\text{frame}} = \text{mean}(|\text{gray}_t - \text{gray}_{t-1}|) < 0.01$. Completely eliminated false positives while reliably catching static freeze padding. |

---

## 8. How to Run Benchmarks and Reproduce Results

### 8.1 Verification Smoke Test
Executes a 50-step rollout across both `NavRepTrainEnv` and `IANEnv`:
```bash
HOME=/tmp/home MPLCONFIGDIR=/tmp/matplotlib ./env/bin/python scripts_custom/smoke_test.py
```

### 8.2 Pretrained Policy Cross-Testing (Repo's Native Script)
Runs the original cross-test script on compatible PPO models:
```bash
HOME=/tmp/home MPLCONFIGDIR=/tmp/matplotlib ./env/bin/python -m navrep.scripts.cross_test_navreptrain_in_ianenv --backend VAE_LSTM --encoding VM --no-gpu --n 20
```

### 8.3 Full Benchmark and Video Recording (Task 2 Deliverables)
Evaluates 20 seeded episodes per scenario (seeds 100-119) under native/calibrated limits (175 steps for NavRepTrainEnv, 500 steps for IANEnv), conducts 60-seed sweeps, records 22 videos with dynamic robot-following camera, laser points, PiP minimap, and telemetry HUD, and validates quality gates:

```bash
# 1. Run full benchmark and video recording for all 11 scenarios:
HOME=/tmp/home MPLCONFIGDIR=/tmp/matplotlib ./env/bin/python scripts_custom/remediate_and_record.py

# 2. Or run a single scenario benchmark and recording:
HOME=/tmp/home MPLCONFIGDIR=/tmp/matplotlib ./env/bin/python scripts_custom/remediate_and_record.py --scenario irosasl1

# 3. Re-compile metrics_existing.csv and VIDEO_INDEX.md from existing results:
HOME=/tmp/home MPLCONFIGDIR=/tmp/matplotlib ./env/bin/python scripts_custom/remediate_and_record.py --compile-only

# 4. Automated quality gate verification:
HOME=/tmp/home MPLCONFIGDIR=/tmp/matplotlib ./env/bin/python scripts_custom/verify_videos.py
```

Outputs produced:
- `submission/task2_videos/metrics_existing.csv`: Aggregated benchmark metrics with `max_steps`, `repo_script_success_pct` (numeric), `repo_script_max_steps`, and `repo_script_episodes` columns.
- `submission/task2_videos/VIDEO_INDEX.md`: Generated strictly from sidecar JSON files in `submission/task2_videos/metadata/`.
- `submission/task2_videos/<NN>_<scenario>_success.mp4`: True `GOAL_REACHED` episodes (with terminal distance to goal <= 1.0m).
- `submission/task2_videos/<NN>_<scenario>_bestprogress.mp4`: Honest best-progress runs for scenarios without goal reaches.
- `submission/task2_videos/<NN>_<scenario>_failure.mp4`: Verified honest collision or timeout runs ($\ge 30$ steps, $\ge 8.0\text{ s}$).

Success videos come from a 60-seed sweep (seeds 0-59), benchmark rates come from seeds 100-119. Scenarios with 0% benchmark success but a success video (irosasl3, irosunity_scene_map3) reach the goal only in rare episodes.

**Programmatic Quality Gate (`scripts_custom/verify_videos.py`)**:
The quality gate programmatically checks HUD status badge color and sidecar/filename consistency across all 22 videos. For each video, it extracts the final frame, detects the HUD status badge by pixel color (green = `GOAL REACHED`, orange = `TIMEOUT`, red = `COLLISION`), and asserts strict consistency between the HUD badge, sidecar outcome, and filename suffix. It also verifies that the robot (cyan), goal (green ring), and obstacles (light grey) are actively present in frames at 25%, 50%, and 75%, enforces frozen frame bounds ($\le 20$), and verifies minimum duration ($\ge 8.0\text{ s}$ for failure/bestprogress).

### 8.4 Benchmark Methodology and Measured Comparison with Repo Script
The benchmark results reported in `metrics_existing.csv` compare our evaluation against the ETH Zurich repository's reference script `navrep/scripts/cross_test_navreptrain_in_ianenv.py`. Differences in measured numbers are governed strictly by the following measured code facts:
1. **Backend and Encoding (`backend`, `encoding`)**: Both evaluations use identical architecture and pretrained weights: `--backend VAE_LSTM --encoding VM` (`navreptrainencodedenv_2020_09_23__09_48_22_PPO_VAE_LSTM_VM_V32M512_ckpt.zip`).
2. **Episode Count (`episodes` vs `repo_script_episodes`)**: In our benchmark, exactly 20 episodes are evaluated per scenario (20 episodes $\times$ 11 scenarios = 220 episodes). In the repo script, `--n 20` executes 20 total episodes across the entire suite of IANEnv environments, randomly selecting scenarios on each `env.reset()`. In this measured 20-episode rollout, individual scenarios were sampled only 1 to 3 times (e.g., `irosasl6` (a scenario outside the 11 benchmarked): 3 episodes, `irosasl_office_j1`: 2 episodes, `irosasl_office_j3`: 2 episodes, `irosasl3`: 1 episode, `irosunity_scene_map3`: 1 episode), while unsampled scenarios cannot be measured and are recorded as `N/A`.
3. **Step Limits (`max_steps` vs `repo_script_max_steps`)**: Read directly from the repository code, `IANEnv` sets `max_episode_length = 1000` (`navrep/envs/encodedenv.py:317` and `navrep/envs/ianenv.py:29`), which our cross-test rollout verified (episodes reached up to 1000 steps). Our benchmark evaluates `IANEnv` scenarios with `max_steps = 500`. For `NavRepTrainEnv`, the code defines `time_limit = 35` and `time_step = 0.2` ($35 / 0.2 = 175$ steps in `crowd_sim/envs/crowd_sim.py:28-29`), but `cross_test_navreptrain_in_ianenv.py` only tests `IANEnv`, so repo script numbers for `navreptrain` scenarios cannot be measured and are recorded as `N/A`.
4. **Seed Distribution (`seeds`)**: Our benchmark strictly pins random seeds 100–119 using `np.random.seed(seed)` and `random.seed(seed)` before each reset for reproducible per-scenario evaluation. The repo script calls `env.reset()` without setting seeds, running unseeded stochastic episode resets.
5. **Deterministic Flag (`predict(deterministic=True)`)**: Read directly from code, both evaluations use `deterministic=True` during policy inference (`predict(obs, deterministic=True)` in `navrep/scripts/test_navrep.py:22` and `scripts_custom/remediate_and_record.py`).

### 8.5 Single Scenario Recording CLI
To record an individual scenario with custom parameters:
```bash
HOME=/tmp/home MPLCONFIGDIR=/tmp/matplotlib ./env/bin/python scripts_custom/record_scenario.py \
    --scenario irosasl1 \
    --seed 42 \
    --backend VAE_LSTM \
    --encoding VM \
    --fps 12 \
    --out submission/task2_videos/03_irosasl1_test.mp4
```

### 8.6 Development Scripts
Scratch, diagnostic, and remediation scripts used during investigation and debugging are organized in [`scripts_custom/dev/`](scripts_custom/dev/):
- `scripts_custom/dev/diagnose_collisions.py`: Analyzes collision boundaries and minimum obstacle clearances.
- `scripts_custom/dev/diagnose_ian_rendering.py`: Verifies ASL map bounding box coordinate extents and laser projection.
- `scripts_custom/dev/finish_task2.py`: Initial orchestration scratch script for video candidates.
- `scripts_custom/dev/rerecord_fixes.py`: Utility to re-record individual target scenarios.
- `scripts_custom/dev/run_parallel_remediation.py`: Scratch script for parallel seed evaluation.

---

## 9. Display and Video Encoding Notes
- **Headless Display**: In environments without an active X server or Xvfb display server, software vector rendering is natively handled through OpenCV buffer compositing directly to image arrays, eliminating X11 and Xvfb display server dependencies.
- **Video Encoding**: MP4 encoding uses `imageio` with the pinned `imageio-ffmpeg` binary backend, producing standard H.264 (`yuv420p`) video streams playable in any web browser and media player.
