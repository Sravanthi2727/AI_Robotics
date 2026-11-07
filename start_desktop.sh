#!/bin/bash
set -e

# ==============================================================================
# NavRep XFCE Desktop & Virtual Display Startup Script
# SVNIT CS359: AI for Robotics
# Matches start_gazebo.sh pattern for noVNC browser streaming on port 6080
# ==============================================================================

export DISPLAY=:99
export LIBGL_ALWAYS_SOFTWARE=1
export QT_X11_NO_MITSHM=1
export MESA_GL_VERSION_OVERRIDE=3.3
export MPLCONFIGDIR=/tmp/matplotlib
export HOME=/root
export PATH=/opt/env/bin:$PATH
export PYTHONPATH=/workspace:/workspace/external:$PYTHONPATH

# Ensure temporary directories exist and are writable
mkdir -p /tmp/matplotlib /tmp/run

# Symlinks for NavRep's hardcoded ~/navrep and ~/maps paths
if [ -d "/workspace" ]; then
    echo "Configuring home-directory symlinks for /workspace..."
    ln -sfn /workspace /root/navrep
    if [ -d "/workspace/maps" ]; then
        ln -sfn /workspace/maps /root/maps
    fi
fi

# Verify editable packages and compiled C extensions are available; if not or if mounted afresh, register with full visibility (fail-fast)
echo "Verifying NavRep package registration and Cython extensions..."
python -c "from crings import fast_lidar_to_rings; import navrep, navrep_custom, frame_msgs" 2>/dev/null || {
    echo "Registering NavRep in editable mode and compiling Cython extensions from /workspace..."
    pip install --no-build-isolation -e /workspace
    echo "Registering navrep_custom in editable mode..."
    pip install -e /workspace/navrep_custom
}

# Start DBus session bus
if [ -d /var/run/dbus ]; then
    rm -f /var/run/dbus/pid
fi
mkdir -p /var/run/dbus
dbus-daemon --system --fork >/dev/null 2>&1 || true
eval $(dbus-launch --sh-syntax)

echo "Starting virtual display (Xvfb :99, 1280x800x24)..."
Xvfb :99 -screen 0 1280x800x24 >/tmp/xvfb.log 2>&1 &
sleep 2

echo "Starting XFCE4 desktop..."
startxfce4 >/tmp/xfce.log 2>&1 &
sleep 3

echo "Starting x11vnc server on port 5900..."
x11vnc \
    -display :99 \
    -forever \
    -shared \
    -rfbport 5900 \
    -nopw \
    >/tmp/x11vnc.log 2>&1 &
sleep 2

echo "Starting noVNC websockify on port 6080..."
websockify \
    --web=/usr/share/novnc \
    6080 \
    localhost:5900 \
    >/tmp/novnc.log 2>&1 &
sleep 2

echo ""
echo "=========================================================="
echo "  NavRep Desktop Environment is Ready!"
echo "  Open your browser and navigate to:"
echo "  http://localhost:6080/vnc.html"
echo "=========================================================="
echo "  Project Root : /workspace"
echo "  Python Exec  : $(which python)"
echo "  Python Ver   : $(python --version)"
echo "  Display      : $DISPLAY (Mesa Software GL 3.3)"
echo "=========================================================="
echo ""

tail -f /dev/null
