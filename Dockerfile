# syntax=docker/dockerfile:1.7

# Base stage - common runtime libraries and environment variables.
FROM ubuntu:22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    D4RL_SUPPRESS_IMPORT_ERROR=1 \
    D4RL_DATASET_DIR=/home/devuser/.d4rl/datasets \
    MINARI_DATASETS_PATH=/home/devuser/.minari/datasets \
    MUJOCO_GL=osmesa \
    PYOPENGL_PLATFORM=osmesa \
    DISPLAY=""

# Runtime deps plus build deps required for mujoco-py's first import compile.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglew2.2 \
    libglew-dev \
    libgl1-mesa-dri \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libglfw3 \
    libglfw3-dev \
    libosmesa6 \
    libosmesa6-dev \
    patchelf \
    python3.10 \
    python3.10-dev \
    zsh && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


# Dev stage - full image with build tools and dependencies for interactive use.
FROM base AS dev

ENV UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    libglew-dev \
    libgl1-mesa-dev \
    libglfw3-dev \
    libosmesa6-dev \
    neovim \
    patchelf \
    pkg-config \
    python3-pip \
    python3.10-dev \
    swig \
    wget && \
    pip3 install uv && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create a non-root user matching the host user's UID/GID so that
# files written by the container are not owned by root on the host.
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} devuser && \
    useradd -m -u ${USER_ID} -g devuser -s /bin/zsh -d /home/devuser devuser && \
    mkdir -p /home/devuser/.cache /home/devuser/.d4rl /home/devuser/.minari /app && \
    touch /home/devuser/.zshrc && \
    chown -R devuser:devuser /home/devuser /app

ENV HOME=/home/devuser \
    PATH="/home/devuser/.local/bin:${PATH}"

USER devuser

# MuJoCo 2.1.0 is required by mujoco-py / d4rl.
RUN mkdir -p /home/devuser/.mujoco && \
    curl -fsSL https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz \
    | tar -xz -C /home/devuser/.mujoco

ENV MUJOCO_PY_MUJOCO_PATH=/home/devuser/.mujoco/mujoco210 \
    LD_LIBRARY_PATH="/home/devuser/.mujoco/mujoco210/bin"

# Activate the venv for all subsequent layers and at runtime.
# Placed in the home dir so the .:/app bind mount in Compose doesn't shadow it.
ENV VIRTUAL_ENV=/home/devuser/.venv \
    PATH="/home/devuser/.venv/bin:/home/devuser/.local/bin:${PATH}"

# d4rl requires Python < 3.11; Ubuntu 22.04 ships Python 3.10.
# d4rl and d3rlpy disagree on gym, so keep the final image on gym 0.26.2.
# prepare_data.py temporarily downgrades gym only inside its one-shot container.
# Keep later installs (uv and d3rlpy's pip3) from upgrading numpy past the lockfile pin.
RUN --mount=type=bind,src=requirements.txt,target=requirements.txt \
    --mount=type=cache,gid=${GROUP_ID},uid=${USER_ID},target=/home/devuser/.cache/uv \
    grep -oE '^numpy==[^ ]+' requirements.txt > /tmp/constraints.txt && \
    export UV_CONSTRAINT=/tmp/constraints.txt PIP_CONSTRAINT=/tmp/constraints.txt && \
    uv venv --seed --python /usr/bin/python3.10 /home/devuser/.venv && \
    uv pip install -r requirements.txt && \
    uv pip install d3rlpy==2.8.1 && \
    d3rlpy install minari && \
    uv pip install d4rl && \
    uv pip install git+https://github.com/aravindr93/mjrl.git && \
    uv pip install "gymnasium[all]==1.0.0" gym==0.26.2

CMD ["sleep", "infinity"]


# Run stage - minimal image for executing preparation and training.
FROM base AS run

WORKDIR /app

# Named the same as the dev stage to keep venv and cache paths consistent.
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} devuser && \
    useradd -m -u ${USER_ID} -g devuser -s /bin/zsh -d /home/devuser devuser && \
    mkdir -p /home/devuser/.cache /home/devuser/.d4rl /home/devuser/.minari /app && \
    chown -R devuser:devuser /home/devuser /app

ENV HOME=/home/devuser \
    PATH="/home/devuser/.local/bin:${PATH}"

USER devuser

COPY --from=dev /home/devuser/.venv /home/devuser/.venv
COPY --from=dev /home/devuser/.mujoco /home/devuser/.mujoco

ENV VIRTUAL_ENV=/home/devuser/.venv \
    PATH="/home/devuser/.venv/bin:/home/devuser/.local/bin:${PATH}" \
    MUJOCO_PY_MUJOCO_PATH=/home/devuser/.mujoco/mujoco210 \
    LD_LIBRARY_PATH="/home/devuser/.mujoco/mujoco210/bin"

ENTRYPOINT ["python"]
CMD ["--help"]
