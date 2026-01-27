
# Gated Regularization for Offline Reinforcement Learning

This repository contains experimental code for evaluating gated regularization methods in offline reinforcement learning on [D4RL](https://github.com/Farama-Foundation/D4RL) MuJoCo benchmarks.
Baseline algorithms are from the [d3rlpy](https://github.com/takuseno/d3rlpy) library and adapted to incorporate the gating mechanism.

The experiments were conducted on a workstation with an AMD Ryzen 9 5950X (16 cores), 128GB RAM, and an RTX 3090Ti, running Ubuntu 24.04 and using Python 3.11 with package versions as specified in `requirements.txt`. The code also runs on CPU-only systems, though training will be significantly slower.

## Installation

```bash
pip install -r requirements.txt
```

After installing the requirements, install Minari via d3rlpy:

```bash
d3rlpy install minari
```

## Running the Experiments

The experiments follow a three-step workflow: data preparation, training, and results aggregation.

### Step 1: Prepare Data

First, convert D4RL datasets to Minari format with policy annotations:

```bash
python prepare_data.py
```

This processes the following D4RL datasets and stores them as Minari datasets with recovered behavioral policy mean and standard deviation annotations:
- `ant-medium-v2` → `d4rl/ant/medium-p-v2`
- `hopper-medium-v2` → `d4rl/hopper/medium-p-v2`
- `halfcheetah-medium-v2` → `d4rl/halfcheetah/medium-p-v2`
- `walker2d-medium-v2` → `d4rl/walker2d/medium-p-v2`

### Step 2: Run Experiments

Run the training scripts for experiments without and with gated regularization:

```bash
# Without gated regularization (CQL, ReBRAC, TD3+BC)
./without_gate.sh

# With gated regularization (CQLgR, ReBRACgR, TD3+BCgR)
./with_gate.sh
```

Each script runs all algorithm–dataset–seed combinations (3 algorithms × 4 datasets × 5 seeds = 60 runs per script), d3rlpy training logs are stored in `data/d3rlpy_logdata/`.

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
- `--compile`: Enable torch.compile for faster training

Default values can be modified in `scs/experiments/defaults.py`:

```python
DATASET: str = "test/dataset/test-v0"  # Default dataset
GATE_WEIGHT: float = 2.0               # Gate weight (ρ in the paper)
SEED: int = 1                          # Random seed
GPU: bool = True                       # Use GPU
COMPILE: bool = True                   # Use torch.compile
```

### Step 3: Process Results

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
├── prepare_data.py          # Step 1: Data preparation
├── with_gate.sh             # Step 2a: Run gated experiments
├── without_gate.sh          # Step 2b: Run baseline experiments
├── process_results.py       # Step 3: Aggregate results
├── scs/
│   ├── gates.py             # Gate functions (Eq. 11)
│   ├── experiments/         # Experiment entry points
│   ├── algs/                # Algorithm configurations
│   └── datasets/            # Data loading utilities
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
