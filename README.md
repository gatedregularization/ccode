# Gated Regularization for Offline Reinforcement Learning

Experimental code for gated regularization methods in offline RL on D4RL MuJoCo tasks.
Baseline implementations are adapted from `d3rlpy`.

The experiments were conducted on a workstation with an AMD Ryzen 9 5950X (16 cores), 128GB RAM, and an RTX 3090Ti, running Ubuntu 24.04 and. The code also runs on CPU-only systems, though experiments will be slower.


## Setup Experiments with Docker

Docker is the default workflow because this stack mixes legacy and modern RL dependencies that are difficult to keep consistent in a single local environment.

- `d4rl` + `mujoco-py` depend on older MuJoCo-era packages.
- Dataset conversion uses `gym==0.24.1`, while training is kept on newer `gym`.
- `d3rlpy` and `d4rl` constraints are not cleanly aligned for local installs.

### 1) Build Images

Create bind-mount directories first:

```bash
mkdir -p data/minari data/d4rl data/cache
```

Build containers:

```bash
docker compose build
```

If you use non-default UID/GID:

```bash
UID=$(id -u) GID=$(id -g) docker compose build
```

### 2) Prepare Datasets (One Time)

```bash
docker compose run --rm run prepare_data.py
```

This downloads D4RL data, reconstructs policy metadata, and writes Minari datasets.

### 3) Run Experiments

Baselines:

```bash
docker compose run --rm run-zsh ./without_gate.sh
```

Gated regularization:

```bash
docker compose run --rm run-zsh ./with_gate.sh
```

Constant-gate ($0.5$) and stochastic-gate ($\mathrm{Uniform}[0, 1]$) ablations:

```bash
docker compose run --rm run-zsh ./ablation_constant.sh
docker compose run --rm run-zsh ./ablation_stochastic.sh
```

Each script runs:
- 3 algorithms
- 3 datasets
- 5 seeds

Total: 45 runs per script.

### 4) Aggregate Results

```bash
docker compose run --rm run process_results.py --results_path data/d3rlpy_logdata --window_size 10
```

## Run Individual Experiments

```bash
docker compose run --rm run -m scs.experiments.cql_gr --dataset d4rl/hopper/medium-p-v2 --seed 0 --gpu --compile
```

## Dev Container

To start it:

```bash
docker compose up -d dev
```

Attach from a terminal:

```bash
docker exec -it grc-dev zsh
```

Detach by exiting the shell (`exit` or `Ctrl+D`).

To stop it:

```bash
docker compose stop dev
```

The `dev` environment can also be used with VS Code Dev Containers by attaching to `grc-dev`.

## Virtual Environment

Using this requires to setup and resolve dependnecies.

### Using uv

```bash
uv venv --python 3.10 .venv
source .venv/bin/activate
uv sync
```

### Using pip + venv

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Data Layout

Persistent data lives under `data/`:

- `data/minari/` Minari datasets from `prepare_data.py`
- `data/d4rl/` D4RL cache
- `data/cache/` shared tool cache
- `data/d3rlpy_logdata/` checkpoints and metrics
- `data/logs_without_gate/` batch logs from `without_gate.sh`
- `data/logs_with_gate/` batch logs from `with_gate.sh`
- `data/logs_ablation_constant/` batch logs from `ablation_constant.sh`
- `data/logs_ablation_stochastic/` batch logs from `ablation_stochastic.sh`

## Algorithms

| Algorithm | Module |
|---|---|
| CQL | `scs.experiments.cql` |
| CQLgR | `scs.experiments.cql_gr` |
| ReBRAC | `scs.experiments.rebrac` |
| ReBRACgR | `scs.experiments.rebrac_gr` |
| TD3+BC | `scs.experiments.td3_plus_bc` |
| TD3+BCgR | `scs.experiments.td3_plus_bc_gr` |
