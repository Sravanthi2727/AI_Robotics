# Task 1: NavRep Setup, Audit, and Execution Report

## 1. Overview of the Project and the Paper
NavRep (ETH Zurich Autonomous Systems Lab) investigates unsupervised representation learning for reinforcement learning in 2D simulated robot navigation among dynamic human crowds. Rather than feeding raw high-dimensional lidar point clouds directly into policy networks, NavRep modularizes navigation into representation learning (world models) and controller training (C-module). Unsupervised architectures—including Variational Autoencoders (ConvVAE 2D and 1D), Recurrent Neural Networks (MDN-RNN / LSTM), and autoregressive Transformers (GPT / minGPT)—compress temporal lidar streams into compact latent representations \(z\) and dynamic belief states \(h\). These latent representations are then provided to Proximal Policy Optimization (PPO2) controllers trained across simulated crowd environments (such as SOADRL CrowdSim and ASL Pepper 2D / IANEnv). The paper demonstrates that self-supervised world representations yield superior sample efficiency, generalization, and transferability across crowd navigation benchmarks compared to end-to-end RL baselines.

## 2. Operating System and Hardware
- **Operating System**: Linux (Ubuntu 24.04.4 LTS, kernel `6.17.0-14-generic #14~24.04.1-Ubuntu x86_64`)
- **CPU**: 12th Gen Intel(R) Core(TM) i5-12450H (4 vCPUs allocated, 2.50 GHz, AVX2 / FMA supported)
- **RAM**: 7.8 GiB total (~3.8 GiB available)
- **GPU**: Headless CPU execution (`nvidia-smi` not attached; CPU-only execution verified)
- **Architecture**: x86_64 Little Endian

## 3. Software Requirements and Key Dependencies
NavRep was authored for Python 3.6, TensorFlow 1.13.2 / 1.15.x, Stable-Baselines 2.10.0, Gym 0.15.7, Pyglet 1.5.0/1.5.22, PyTorch 1.6.0, and custom C++/Cython robotic libraries from ETH Zurich ASL.

Key pinned versions (from `pip freeze` in `submission/pip_freeze.txt`):
- `python==3.6.15` (Conda-forge)
- `tensorflow==1.13.2`
- `torch==1.6.0+cpu`, `torchvision==0.7.0+cpu`
- `keras==2.3.1`
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
- `pyIAN==0.0.3`
- `asl-pepper-2d-sim==0.0.5`
- `soadrl-crowdnav==0.0.3`
- `pyniel==0.2`
- `pymap2d==0.1.15`
- `pylidar2d==0.1`
- `pyrvo2-danieldugas==0.0.2`
- `pyrangelibc-danieldugas==0.0.2`
- `rospy==1.15.11`, `rosbag==1.15.11`, `tf==1.12.1.post3`, `tf2-ros==0.6.5`

## 4. Copy-Pasteable Installation Steps

To reproduce this exact working environment from scratch without requiring root or sudo privileges:

```bash
# 1. Navigate to the repository root
cd /home/sravanthi/Desktop/navrep

# 2. Download standalone micromamba binary for isolated user-space environment management
mkdir -p bin
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C bin --strip-components=1 bin/micromamba

# 3. Create isolated Python 3.6 environment
./bin/micromamba create -y -p ./env -c conda-forge python=3.6 pip

# 4. Install Cython and NumPy first (required for compiling C/C++ extensions)
./env/bin/pip install numpy==1.19.5 Cython==0.29.28

# 5. Install PyTorch CPU build from PyTorch wheel index
./env/bin/pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# 6. Install pinned opencv-python wheel (avoids compiling unpinned OpenCV C++ from source)
./env/bin/pip install opencv-python==4.5.5.64

# 7. Install core ML and RL dependencies with pinned Gym 0.15.7 and Cloudpickle 1.6.0
./env/bin/pip install "tensorflow==1.13.2" "keras==2.3.1" "stable-baselines==2.10.0" "cloudpickle==1.6.0" "gym==0.15.7" "pandas==1.1.0" "scipy==1.5.4" "matplotlib==3.3.4"

# 8. Install Pyglet 1.5.x, rendering, video recording, and ROS utility packages
./env/bin/pip install "pyglet==1.5.22" "numba==0.53.1" "rospkg" "catkin_pkg" "rich" "tqdm" "imageio" "imageio-ffmpeg"

# 9. Install ETH Zurich ASL robotics simulation packages
./env/bin/pip install "pyIAN==0.0.3" "asl-pepper-2d-sim==0.0.5" "soadrl-crowdnav==0.0.3" "pyniel==0.2" "pymap2d==0.1.15" "pylidar2d==0.1" "pyrvo2-danieldugas==0.0.2"
./env/bin/pip install --no-build-isolation "pyrangelibc-danieldugas==0.0.2"

# 10. Install dummy ROS packages for rospypi compatibility
./env/bin/pip install --extra-index-url https://rospypi.github.io/simple/ rospy rosbag tf tf2_ros

# 11. Install NavRep in editable development mode
./env/bin/pip install --no-build-isolation -e .

# 12. Copy bundled external modules into site-packages (required by ASL state estimator)
cp -r external/frame_msgs env/lib/python3.6/site-packages/
cp -r external/tf_bag env/lib/python3.6/site-packages/

# 13. Set HOME and maps symlinks (so ~/navrep and ~/maps resolve to repo paths)
ln -sf $(pwd) ~/navrep
ln -sf $(pwd)/maps ~/maps
```

## 5. Modifications Made to Original Code or Config
In accordance with Ground Rule 3, **ZERO original repository files were modified**. All behavior and compatibility fixes were achieved through pinned environment specification, site-packages module placement, and dedicated wrappers in `scripts_custom/` and `navrep_custom/`.

Refer to `docs/CHANGES_LOG.md` for the audit trail.

## 6. Challenges and Solutions Table

| Error | Root Cause | Fix Applied |
|---|---|---|
| `sudo: The "no new privileges" flag is set` | Container sandbox prohibits privilege escalation; sudo is disallowed. | Installed user-space micromamba environment inside `./env` without needing root. |
| Prolonged stall / network throttle downloading 513MB PyPI `torch==1.6.0` wheel | PyPI default package bundles CUDA 10 binaries; PyPI server was severely throttled for large binary downloads. | Switched to official PyTorch CPU wheel (`torch==1.6.0+cpu`) from `download.pytorch.org`, downloading at >10 MB/s. |
| `Building wheel for opencv-python (pyproject.toml) ...` | `stable-baselines` pulled unpinned `opencv-python`, which defaulted to modern PyPI release lacking Python 3.6 wheels and triggered a 40-minute C++ compilation. | Pinned and installed `opencv-python==4.5.5.64` pre-built wheel before installing `stable-baselines`. |
| `Syntax error in simple statement list (print "...")` in `RangeLibc.pyx` | Pip build isolation pulled modern Cython which defaults to Python 3 syntax, rejecting Python 2 print syntax. | Installed `pyrangelibc-danieldugas` with `--no-build-isolation` to use the pinned environment `Cython==0.29.28`. |
| `ModuleNotFoundError: No module named 'frame_msgs'` | `setup.py` maps `package_dir={'frame_msgs': ...}` which `pip install -e` fails to link into site-packages. | Copied `external/frame_msgs` and `external/tf_bag` into `env/lib/python3.6/site-packages/`. |
| `FileNotFoundError: No such file or directory: '~/navrep/models/gym'` | NavRep scripts hardcode `os.path.expanduser("~/navrep/models/gym")`. | Provided `HOME=/home/sravanthi/Desktop` (or symlinked `~/navrep -> repo`), ensuring paths resolve correctly. |
| `AttributeError: 'Box' object has no attribute '_shape'` | PPO2 model policies were serialized with Gym 0.15.7; Gym >=0.20.0 removed `_shape` attribute. | Downgraded gym to `gym==0.15.7`. |
| `AttributeError: module 'types' has no attribute 'CellType'` | `cloudpickle 1.2.2` accessed `types.CellType` which was introduced in Python 3.8. | Upgraded cloudpickle to `cloudpickle==1.6.0` (as specified in `requirements.txt`). |

Refer to `docs/CHALLENGES_LOG.md` for full timestamps and details.

## 7. How to Run the Repository and Simulation Scenarios

### Smoke Test
Runs a 50-step random-action rollout on both `NavRepTrainEnv` and `IANEnv`:
```bash
HOME=/home/sravanthi/Desktop MPLCONFIGDIR=/tmp/matplotlib ./env/bin/python scripts_custom/smoke_test.py
```
**Expected Output**:
```
=== NavRep Smoke Test ===
Python version: 3.6.15 ...
NavRep package path: /home/sravanthi/Desktop/navrep/navrep/__init__.py
[1/2] Initializing NavRepTrainEnv(silent=True)...
Reset successful. Observation shape: [(1080,), (5,)]
Running 50 random action steps...
NavRepTrainEnv completed 50 steps successfully.
[2/2] Initializing IANEnv(silent=True)...
Reset successful. Observation shape: [(1080,), (5,)]
Running 50 random action steps...
IANEnv completed 50 steps successfully.
=== Smoke test PASSED cleanly! ===
```

### Pretrained Policy Evaluation (Cross-Test)
Cross-tests the pretrained PPO2 policies (using ConvVAE representation) in IANEnv:
```bash
HOME=/home/sravanthi/Desktop MPLCONFIGDIR=/tmp/matplotlib ./env/bin/python -m navrep.scripts.cross_test_navreptrain_in_ianenv --backend VAE_LSTM --encoding V_ONLY --no-gpu --n 1
```
**Expected Output**:
Loads compatible models from `models/gym/`, loads world model from `models/V/navreptrainvae.json`, and evaluates across IANEnv scenarios with progress bars.

## 8. Troubleshooting Tips
1. **Always export environment variables**:
   ```bash
   export HOME=/home/sravanthi/Desktop
   export MPLCONFIGDIR=/tmp/matplotlib
   ```
   This prevents Matplotlib directory errors and ensures `~/navrep` and `~/maps` paths resolve properly.
2. **Offscreen Rendering / Headless mode**:
   On headless servers, rendering requires Xvfb or Pyglet offscreen windowing:
   ```bash
   xvfb-run -a -s "-screen 0 1024x768x24" ./env/bin/python ...
   ```
3. **CPU Execution**:
   Always pass `--no-gpu` when executing on systems without CUDA support to prevent TensorFlow CUDA init warnings.
