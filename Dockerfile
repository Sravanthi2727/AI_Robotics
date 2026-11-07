# ==============================================================================
# NavRep Multi-Platform Docker Environment
# Base OS: Ubuntu 22.04 LTS (linux/amd64)
# Python: 3.6.15 (Hermetic Micromamba Environment)
# GUI Stack: XFCE4 Desktop + Xvfb Virtual Display + x11vnc + noVNC (Port 6080)
# ==============================================================================

FROM --platform=linux/amd64 ubuntu:22.04

LABEL maintainer="SVNIT AI for Robotics (CS359)"
LABEL description="Cross-platform Docker development environment for ETH NavRep"

ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash
ENV DISPLAY=:99
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV QT_X11_NO_MITSHM=1
ENV MESA_GL_VERSION_OVERRIDE=3.3
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV HOME=/root

# 1. Install Ubuntu system packages, XFCE desktop, virtual display, and rendering tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Core compilation & system tools
    build-essential \
    gcc \
    g++ \
    cmake \
    make \
    git \
    curl \
    wget \
    bzip2 \
    ca-certificates \
    # X11 virtual display & desktop
    xvfb \
    x11vnc \
    novnc \
    websockify \
    xfce4 \
    xfce4-terminal \
    dbus-x11 \
    # OpenGL software rendering & Mesa drivers
    mesa-utils \
    libgl1-mesa-dri \
    libgl1-mesa-glx \
    libglu1-mesa-dev \
    freeglut3-dev \
    libosmesa6-dev \
    # OpenCV / Pyglet dependencies & media tools
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libfontconfig1 \
    fonts-dejavu-core \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 2. Download and install standalone Micromamba binary for linux-64
RUN mkdir -p /tmp/micromamba && \
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /tmp/micromamba bin/micromamba && \
    mv /tmp/micromamba/bin/micromamba /usr/local/bin/micromamba && \
    rm -rf /tmp/micromamba

# 3. Create hermetic Python 3.6.15 environment via conda-forge
RUN micromamba create -y -p /opt/env -c conda-forge python=3.6 pip

# Set environment PATH so Python 3.6 is the default
ENV PATH=/opt/env/bin:$PATH

# 4. Install dependencies in strict compatibility order
# Step A: Cython and NumPy (required for C/Cython extension building)
RUN pip install numpy==1.19.5 Cython==0.29.28

# Step B: PyTorch CPU wheels from official PyTorch repository (avoids 500MB+ PyPI CUDA wheel)
RUN pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Step C: Pinned OpenCV wheel (MUST precede stable-baselines to avoid 40-min source build)
RUN pip install opencv-python==4.5.5.64

# Step D: Install remaining pinned ML, RL, ROS, and utility packages
COPY requirements_docker.txt /tmp/requirements_docker.txt
RUN pip install --extra-index-url https://rospypi.github.io/simple/ --use-deprecated=legacy-resolver --no-build-isolation -r /tmp/requirements_docker.txt

# Step E: Install RangeLibc with --no-build-isolation (forces use of Cython 0.29.28)
RUN pip install --no-build-isolation "pyrangelibc-danieldugas==0.0.2"

# 5. Set up workspace, pre-install NavRep packages and Cython extensions during build
WORKDIR /workspace
COPY . /workspace

ENV PYTHONPATH=/workspace:/workspace/external

# Build Cython extensions (crings.pyx) and register editable packages during build
RUN pip install --no-build-isolation -e /workspace && \
    pip install -e /workspace/navrep_custom

COPY start_desktop.sh /start_desktop.sh
RUN chmod +x /start_desktop.sh

EXPOSE 6080 5900

CMD ["/start_desktop.sh"]
