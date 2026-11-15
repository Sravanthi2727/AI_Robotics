#!/bin/bash
# ==============================================================================
# launch_demo.sh
# ==============================================================================
# One-line demonstration launcher for Shriya's Task 3:
# Library Book Collection Robot GUI Demonstration.
#
# SVNIT Surat — CS359: AI for Robotics
# Author: Shriya (Branch: shriya)
# ==============================================================================

set -e

# Ensure display is directed to XFCE desktop
export DISPLAY=${DISPLAY:-:99}
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=3.3
export MPLCONFIGDIR=/tmp/matplotlib
export PYTHONPATH=/workspace:/workspace/external:$PYTHONPATH

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$WORKSPACE_ROOT"

echo "=========================================================="
echo "  Launching Autonomous Robot Navigation GUI Demo (SVNIT)"
echo "  Display : $DISPLAY"
echo "  Root    : $WORKSPACE_ROOT"
echo "  Scenarios: [1] Library  [2] Hotel Room Service  [3] MediRunner Hospital"
echo "=========================================================="

python "$SCRIPT_DIR/library_gui.py" "$@"

