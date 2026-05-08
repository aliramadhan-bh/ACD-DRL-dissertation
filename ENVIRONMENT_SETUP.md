# Environment Setup Guide

## Problem Summary

The models were trained with numpy 2.4.3, but the base anaconda environment had numpy 1.26.4 which caused a `ModuleNotFoundError: No module named 'numpy._core.numeric'` error when trying to load models.

## Solution

A fresh virtual environment `cyborg_eval_env` has been created with the exact package versions used during training.

## Package Versions (from model metadata)

- Python: 3.12.4
- Numpy: 2.4.3
- Stable-Baselines3: 2.7.1
- PyTorch: 2.5.1
- Gymnasium: 1.2.3
- Cloudpickle: 3.1.2
- OpenAI Gym: 0.26.2

## How to Use the Environment

### From Command Line

Always use the full path to the Python interpreter in `cyborg_eval_env`:

```bash
cd /Users/ali/Desktop/final_project/cage-challenge-2/CybORG

# Run evaluation
../cyborg_eval_env/bin/python evaluate_ppo_blue.py \
  --model-path /Users/ali/Desktop/final_project/cage-challenge-2/CybORG/models/ppo_decoys_bline_10m.zip \
  --episodes 30 \
  --seed 200 \
  --enhanced-obs \
  --reduced-actions \
  --use-decoys
```

### From VSCode

1. Open Command Palette (Cmd+Shift+P)
2. Select "Python: Select Interpreter"
3. Choose: `/Users/ali/Desktop/final_project/cage-challenge-2/cyborg_eval_env/bin/python`

### Test That It Works

```bash
cd /Users/ali/Desktop/final_project/cage-challenge-2/CybORG

# Quick test
../cyborg_eval_env/bin/python -c "
from stable_baselines3 import PPO
model = PPO.load('models/ppo_decoys_bline_10m.zip')
print('✓ Model loaded successfully!')
"
```

## Common Evaluation Scripts

All scripts should be run from the `CybORG` directory using the `cyborg_eval_env` Python:

```bash
cd /Users/ali/Desktop/final_project/cage-challenge-2/CybORG

# Standard evaluation
../cyborg_eval_env/bin/python evaluate_ppo_blue.py --model-path models/ppo_decoys_bline_10m.zip --episodes 30 --enhanced-obs --reduced-actions --use-decoys

# Hierarchical evaluation
../cyborg_eval_env/bin/python evaluate_hierarchical.py --model-path models/hierarchical_ppo_10m.zip --episodes 30

# Reactive decoys evaluation
../cyborg_eval_env/bin/python evaluate_reactive_decoys.py --model-path models/ppo_decoys_bline_10m.zip --episodes 30 --enhanced-obs --reduced-actions

# Test different decoy targets
../cyborg_eval_env/bin/python test_decoy_targets.py --model-path models/ppo_decoys_bline_10m.zip --episodes 30
```

## Important Notes

- **Do not** use the base anaconda Python (`/opt/anaconda3/bin/python`) - it has numpy 1.26.4 which is incompatible
- **Do not** use the `CybORG/.venv` environment - it has import issues
- **Always** use `cyborg_eval_env` for running evaluations
- The environment is located at: `/Users/ali/Desktop/final_project/cage-challenge-2/cyborg_eval_env`

## Verification

If you get import errors, verify you're using the correct environment:

```bash
which python  # Should show: .../cyborg_eval_env/bin/python

python -c "import numpy; print(numpy.__version__)"  # Should show: 2.4.3
```
