# Reproducibility Guide

## Environment

- **OS:** macOS 26.3.1 (ARM64, Apple M1 Pro)
- **Python:** 3.12.4
- **Active virtual environment:** `CybORG/.venv/`

## Installation

```bash
git clone https://github.com/aliramadhan-bh/ACD-DRL-dissertation
cd ACD-DRL-dissertation/CybORG
pip install -e .
pip install stable-baselines3==2.7.1 sb3-contrib==2.7.1 gymnasium==1.2.3 torch==2.5.1 numpy==2.4.3
```

## Dependencies

| Package | Version |
|---|---|
| torch | 2.5.1 |
| stable-baselines3 | 2.7.1 |
| sb3-contrib | 2.7.1 |
| gymnasium | 1.2.3 |
| gym | 0.26.2 |
| numpy | 2.4.3 |
| scipy | 1.17.0 |
| pandas | 3.0.1 |
| matplotlib | 3.10.8 |
| networkx | 3.6.1 |
| PyYAML | 6.0.3 |
| paramiko | 4.0.0 |
| cryptography | 46.0.5 |
| bcrypt | 5.0.0 |
| invoke | 2.2.1 |
| prettytable | 3.17.0 |
| cloudpickle | 3.1.2 |
| CybORG | 2.1 (editable install) |

## Reproducing Evaluation Results

To reproduce the baseline result (mean reward -17.93):

```bash
cd CybORG
python evaluate_ppo_blue.py --model models/ppo_decoys_bline_10m.zip --episodes 30 --seed 200
```

To reproduce all approach results, run the corresponding `evaluate_*.py` script with `--episodes 30 --seed 200` and the relevant model file from the `models/` directory.

## Results

All verified evaluation results are stored in `CybORG/SEED200_*.csv` and `CybORG/FRESH_SEED200_TEST.csv`.
