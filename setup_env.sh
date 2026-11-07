#!/usr/bin/env bash
# ==============================================================================
# setup_env.sh - ETH Zurich NavRep Automated Environment Setup
# ==============================================================================
# This script creates an isolated Python 3.6 environment and installs all
# dependencies required to run NavRep, evaluate existing scenarios, and execute
# custom simulation benchmarks without requiring root or sudo privileges.
# ==============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "=== [1/6] Environment Initialization ==="
echo "Repository Root: $REPO_ROOT"

# Ensure user-space bin directory exists
mkdir -p "$REPO_ROOT/bin"
export PATH="$REPO_ROOT/bin:$PATH"

# Download standalone micromamba binary if not already present
if [ ! -f "$REPO_ROOT/bin/micromamba" ]; then
    echo "Downloading standalone micromamba binary..."
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$REPO_ROOT/bin" --strip-components=1 bin/micromamba
    chmod +x "$REPO_ROOT/bin/micromamba"
fi

# Create isolated Python 3.6.15 environment
if [ ! -d "$REPO_ROOT/env" ]; then
    echo "Creating Python 3.6 environment at $REPO_ROOT/env..."
    "$REPO_ROOT/bin/micromamba" create -y -p "$REPO_ROOT/env" -c conda-forge python=3.6.15 pip

    PIP="$REPO_ROOT/env/bin/pip"
    PYTHON="$REPO_ROOT/env/bin/python"

    echo "=== [2/6] Installing Pinned Core Compilers & Pre-requisites ==="
    $PIP install numpy==1.19.5 Cython==0.29.28

    echo "=== [3/6] Installing PyTorch, ML and Reinforcement Learning Packages ==="
    $PIP install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
    $PIP install opencv-python==4.5.5.64
    $PIP install \
        tensorflow==1.13.2 \
        keras==2.3.1 \
        stable-baselines==2.10.0 \
        cloudpickle==1.6.0 \
        gym==0.15.7 \
        pandas==1.1.0 \
        scipy==1.5.4 \
        matplotlib==3.3.4 \
        pyglet==1.5.0 \
        numba==0.53.1 \
        imageio==2.15.0 \
        imageio-ffmpeg==0.5.1 \
        importlib_resources \
        rospkg \
        catkin_pkg \
        rich \
        tqdm

    echo "=== [4/6] Installing ETH ASL Simulation Libraries ==="
    $PIP install \
        pyIAN==0.0.3 \
        asl-pepper-2d-sim==0.0.5 \
        soadrl-crowdnav==0.0.3 \
        pyniel==0.2 \
        pymap2d==0.1.15 \
        pylidar2d==0.1 \
        pyrvo2-danieldugas==0.0.2

    $PIP install --no-build-isolation pyrangelibc-danieldugas==0.0.2
    $PIP install --extra-index-url https://rospypi.github.io/simple/ rospy rosbag tf tf2_ros

    echo "=== [5/6] Installing NavRep & Copying Bundled Extension Packages ==="
    $PIP install --no-build-isolation -e .
else
    echo "Python environment already exists at $REPO_ROOT/env."
    PIP="$REPO_ROOT/env/bin/pip"
    PYTHON="$REPO_ROOT/env/bin/python"
fi

# Copy bundled ROS message and transform bag utilities into site-packages
SP_DIR="$($PYTHON -c 'import site; print(site.getsitepackages()[0])')"
echo "Copying frame_msgs and tf_bag to $SP_DIR..."
cp -rn "$REPO_ROOT/external/frame_msgs" "$SP_DIR/" || true
cp -rn "$REPO_ROOT/external/tf_bag" "$SP_DIR/" || true

echo "=== [6/6] Establishing NavRep Asset Symlinks ==="
# Create clean symlinks in $HOME so ~/navrep and ~/maps resolve natively
mkdir -p "$HOME/navrep" 2>/dev/null || true
ln -sfn "$REPO_ROOT/models" "$HOME/navrep/models" 2>/dev/null || true
ln -sfn "$REPO_ROOT/maps" "$HOME/maps" 2>/dev/null || true

# Also create symlinks in /tmp/home for execution environments using HOME=/tmp/home
mkdir -p /tmp/home/navrep 2>/dev/null || true
ln -sfn "$REPO_ROOT/models" /tmp/home/navrep/models 2>/dev/null || true
ln -sfn "$REPO_ROOT/maps" /tmp/home/maps 2>/dev/null || true

# Also create local Desktop links if running inside sandboxed user homes
if [ "$HOME" != "$REPO_ROOT/.." ]; then
    mkdir -p "$(dirname "$REPO_ROOT")/navrep" 2>/dev/null || true
    ln -sfn "$REPO_ROOT/models" "$(dirname "$REPO_ROOT")/navrep/models" 2>/dev/null || true
    ln -sfn "$REPO_ROOT/maps" "$(dirname "$REPO_ROOT")/maps" 2>/dev/null || true
fi

echo "Checking headless display and video encoding utilities..."
which ffmpeg || echo "NOTE: ffmpeg not in PATH; imageio-ffmpeg bundled binary will be used."
which Xvfb || echo "NOTE: Xvfb not detected; software offscreen rendering will be used."

echo "Running verification smoke test..."
MPLCONFIGDIR=/tmp/matplotlib $PYTHON scripts_custom/smoke_test.py

echo "=============================================================================="
echo "Setup completed successfully! To activate this environment:"
echo "  export PATH=\"$REPO_ROOT/env/bin:\$PATH\""
echo "=============================================================================="
