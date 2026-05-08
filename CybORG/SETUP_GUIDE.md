# CAGE Challenge 2 - Setup Guide

## Quick Start

```bash
# 1. Navigate to CybORG directory
cd CybORG

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Install RL dependencies
pip install stable-baselines3 sb3-contrib

# 4. Verify setup
python verify_setup.py

# 5. Explore the challenge
python explore_challenge.py
```

## Verification

```bash
python verify_setup.py
```

Expected output: All checks should show `[PASS]`

## Demo Commands

```bash
# Quick import test
python -c "from CybORG import CybORG; print('CybORG works!')"

# Run exploration script
python explore_challenge.py

# Run training example
python -c "from CybORG.Agents.training_example import run_training_example; run_training_example('Scenario1b')"

# Train recurrent PPO (LSTM) blue agent
python train_ppo_blue.py \
  --algo recurrent_ppo \
  --red-agent bline \
  --timesteps 100000 \
  --model-out models/recurrent_ppo_blue_s1b_bline.zip

# Evaluate recurrent PPO model
python evaluate_ppo_blue.py \
  --algo recurrent_ppo \
  --model-path models/recurrent_ppo_blue_s1b_bline.zip \
  --episodes 30 \
  --max-steps 100 \
  --red-agent bline

# Run unit tests
pytest CybORG/Tests/test_sim/test_sim_Cyborg.py -v
```

## Environment Summary

- **Python:** 3.12.4
- **Virtual Env:** .venv (in this directory)
- **CybORG:** 2.1 (editable install)
- **ML Frameworks:** PyTorch, JAX
- **RL Libraries:** Gymnasium, Stable-Baselines3, sb3-contrib

## Troubleshooting

### Import errors
```bash
pip install -e .
```

### Virtual environment not active
```bash
source .venv/bin/activate
```

## Project Structure

```
CybORG/
├── .venv/                 # Virtual environment
├── SETUP_GUIDE.md         # This file
├── verify_setup.py        # Verification script
├── explore_challenge.py   # Learning demo
├── setup.py               # Package installer
└── CybORG/
    ├── Agents/            # Blue/Red agents
    ├── Shared/            # Scenarios, actions
    └── Tests/             # Unit tests
```
