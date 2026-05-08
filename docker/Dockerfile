FROM ubuntu:22.04

ARG USERNAME=morph
ARG USER_UID=1000
ARG USER_GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace/src \
    MUJOCO_GL=glfw \
    QT_X11_NO_MITSHM=1 \
    OMPL_BRIDGE_MODE=native

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libglvnd0 \
    libglx0 \
    libglfw3 \
    libglew2.2 \
    libice6 \
    libosmesa6 \
    libsm6 \
    libx11-6 \
    libx11-xcb1 \
    libxcb1 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxi6 \
    libxinerama1 \
    libxrandr2 \
    libxrender1 \
    libxxf86vm1 \
    mesa-utils \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-venv \
    python3-wheel \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid "${USER_GID}" "${USERNAME}" && \
    useradd --uid "${USER_UID}" --gid "${USER_GID}" --create-home --shell /bin/bash "${USERNAME}" && \
    mkdir -p /workspace && \
    chown -R "${USERNAME}:${USERNAME}" /workspace

WORKDIR /workspace

# Use the project's single requirements.txt (at the project root) so there's
# one dependency source of truth for both Docker and native installs.  Build
# context is the motion-planning/ project root (see docker-compose.yml).
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r /tmp/requirements.txt

COPY . /workspace

USER ${USERNAME}

CMD ["bash"]
