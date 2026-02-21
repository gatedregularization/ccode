
# Gated Regularization for Offline Reinforcement Learning

This repository contains experimental code for evaluating gated regularization methods in offline reinforcement learning on [D4RL](https://github.com/Farama-Foundation/D4RL) MuJoCo benchmarks.
Baseline algorithms are from the [`d3rlpy`](https://github.com/takuseno/d3rlpy) library and adapted to incorporate the gating mechanism.

The experiments were conducted on a workstation with an AMD Ryzen 9 5950X (16 cores), 128GB RAM, and an RTX 3090Ti, running Ubuntu 24.04, CUDA 12.2, and Python 3.11 with package versions as specified in `requirements.txt`. The code also runs on CPU-only systems, though training will be significantly slower.

## Installation

### Prerequisites

Since the environments extracted from `d4rl` require `mujoco-py`, follow the [mujoco210 installation instructions](https://github.com/openai/mujoco-py) such that you have `~/.mujoco/mujoco210` and set the library path:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
```

> **Note:** Due to dependency conflicts between `d4rl` and `d3rlpy`, the order of package installation matters. `pip` will report some conflicts, but these can be safely ignored as they are only import health checks and do not affect functionality.

### Environment Setup

```bash
python3.11 -m venv .grc_venv
source .grc_venv/bin/activate
```

### Package Installation

Install the base requirements:

```bash
pip install -r requirements.txt
```

Then install `d3rlpy`, `minari` (via `d3rlpy`), `d4rl`, `mjrl` (a dependency required for the MuJoCo gym tasks), and the matching version of `gymnasium`:

```bash
pip install d3rlpy==2.8.1
d3rlpy install minari
pip install d4rl
python -m pip install git+https://github.com/aravindr93/mjrl.git
pip install "gymnasium[all]==1.0.0"
```

### Data Preparation

Before running the experiments, the datasets must be downloaded and prepared. This is where the dependency conflicts come into play: we need the [gym MuJoCo](https://github.com/Farama-Foundation/d4rl/wiki/Tasks#gym) tasks from the `d4rl` dataset, annotate them with their policies, and store them in the local `minari` storage.

This step only needs to be done once; all subsequent experiments can then use the prepared datasets.

```bash
python prepare_data.py
```

This processes the following D4RL datasets and stores them as Minari datasets with recovered behavioral policy mean and standard deviation annotations:

- `ant-medium-v2` → `d4rl/ant/medium-p-v2`
- `hopper-medium-v2` → `d4rl/hopper/medium-p-v2`
- `halfcheetah-medium-v2` → `d4rl/halfcheetah/medium-p-v2`
- `walker2d-medium-v2` → `d4rl/walker2d/medium-p-v2`

The script will also temporarily downgrade `gym` to `0.24.1` to ensure `d4rl` loads correctly, bypass the import health check of `d3rlpy`, create the datasets, and restore `gym` to `0.26.2`. Since `d4rl` won't be imported once the data is created and stored locally, this does not affect subsequent steps.

## Running the Experiments

The experiments follow a two-step workflow: training and results aggregation.

### Step 1: Run Experiments

Run the training scripts for experiments without and with gated regularization:

```bash
# Without gated regularization (CQL, ReBRAC, TD3+BC)
./without_gate.sh

# With gated regularization (CQLgR, ReBRACgR, TD3+BCgR)
./with_gate.sh
```

Each script runs all algorithm–dataset–seed combinations (3 algorithms × 4 datasets × 5 seeds = 60 runs per script). Training logs are stored in `data/d3rlpy_logdata/`.

#### Running Individual Experiments

You can also run individual experiments via the Python module interface:

```bash
python -m scs.experiments.cql_gr --dataset d4rl/hopper/medium-p-v2 --seed 0 --gpu --compile
```

Available flags:

- `--dataset`: Minari dataset identifier (default: see `scs/experiments/defaults.py`)
- `--seed`: Random seed (default: 1)
- `--gate_weight`: Weight for the gate term (default: 2.0, gated algorithms only)
- `--gpu`: Enable GPU acceleration
- `--compile`: Enable `torch.compile` for faster training

Default values can be modified in `scs/experiments/defaults.py`:

```python
DATASET: str = "test/dataset/test-v0"  # Default dataset
GATE_WEIGHT: float = 2.0               # Gate weight (ρ in the paper)
SEED: int = 1                          # Random seed
GPU: bool = True                       # Use GPU
COMPILE: bool = True                   # Use torch.compile
```

### Step 2: Process Results

Aggregate results across seeds and compute statistics:

```bash
python process_results.py
```

This generates CSV files with aggregated reward progressions and statistics (mean, std, standard error) for each algorithm–environment combination.

#### Options

```bash
python process_results.py --results_path data/d3rlpy_logdata --window_size 1
```

- `--results_path`: Directory containing experiment results (default: `data/d3rlpy_logdata`)
- `--window_size`: Window size for rolling statistics (default: 1)

## Project Structure

```
├── prepare_data.py          # Data preparation
├── with_gate.sh             # Run gated experiments
├── without_gate.sh          # Run baseline experiments
├── process_results.py       # Aggregate results
├── scs/
│   ├── gates.py             # Gate functions (Eq. 11)
│   ├── experiments/         # Experiment entry points
│   ├── algs/                # Algorithm configurations
│   └── datasets/            # Data loading and processing utilities
└── data/                    # Output directory (created automatically)
```

## Algorithms

| Algorithm | Module | Description |
|-----------|--------|-------------|
| CQL | `scs.experiments.cql` | Conservative Q-Learning |
| CQLgR | `scs.experiments.cql_gr` | CQL with gated regularization |
| ReBRAC | `scs.experiments.rebrac` | ReBRAC baseline |
| ReBRACgR | `scs.experiments.rebrac_gr` | ReBRAC with gated regularization |
| TD3+BC | `scs.experiments.td3_plus_bc` | TD3+BC baseline |
| TD3+BCgR | `scs.experiments.td3_plus_bc_gr` | TD3+BC with gated regularization |
