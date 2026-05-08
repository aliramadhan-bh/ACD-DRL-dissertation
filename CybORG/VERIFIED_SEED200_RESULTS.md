# Verified Seed 200 Evaluation Results
## Fresh Evaluations - April 19, 2026

All evaluations run with:
- 30 episodes
- Seed 200
- Deterministic policy
- B_lineAgent opponent
- Correct wrapper configurations

## Results Comparison

| Approach | Fresh Seed 200 | Comprehensive Doc Claim | Difference | Status |
|----------|----------------|------------------------|------------|--------|
| **Baseline (Full)** | **-17.93** | -19.16 | +1.23 better | ✅ Verified |
| Hierarchical RL | -19.16 | -19.16 | 0.00 exact match | ✅ Verified |
| Extended Training 20M | -19.49 | -22.21 | +2.72 better | ✅ Verified |
| RecurrentPPO (LSTM) | -19.57 | -23.61 | +4.04 better | ✅ Verified |
| Gentle Reward Shaping | **-18.51** | -24.90 | **+6.39 better** | ✅ Verified |
| Enterprise Focus | -23.34 | -22.52 | -0.82 worse | ✅ Verified |
| Masked PPO (Action Masking) | -85.34 | -96.48 | +11.14 better | ✅ Verified |

## Key Findings

### 1. Comprehensive Results Document is Inaccurate

**Only 1 out of 7 approaches matched the claimed results exactly** (Hierarchical RL).

All others showed different performance:
- **5 approaches performed BETTER** than claimed (baseline, 20M, recurrent, gentle, masked)
- **1 approach performed WORSE** than claimed (enterprise focus)

### 2. New Best Performance

**Gentle Reward Shaping is now the BEST approach at -18.51**, outperforming:
- Baseline: -17.93 (but very close!)
- All other approaches by significant margins

Wait - Baseline is -17.93, which is BETTER than Gentle at -18.51!

**Correction: Baseline (Full) is the BEST at -17.93**

Rankings (seed 200, verified):
1. **Baseline**: -17.93 ✅ BEST
2. Gentle Shaping: -18.51
3. Hierarchical RL: -19.16
4. Extended 20M: -19.49
5. RecurrentPPO: -19.57
6. Enterprise Focus: -23.34
7. Masked PPO: -85.34

### 3. Gentle Reward Shaping is NOT Harmful

Comprehensive results claimed gentle shaping degraded performance to -24.90.

**Fresh evaluation shows gentle shaping at -18.51**, which is:
- Only 0.58 points worse than verified baseline (-17.93)
- Represents a 6.39-point discrepancy from the claimed comprehensive results

This suggests reward shaping may not be as harmful as previously thought.

### 4. Model Performance Variance

The large discrepancies between fresh evaluations and comprehensive results suggest:
- Different model checkpoints may have been used
- Evaluation methodology may have differed
- Random seed effects are larger than expected
- The comprehensive results document may be based on incomplete evaluations

## Verified Model Configurations

All models successfully evaluated with:

| Model | Wrappers Used |
|-------|---------------|
| ppo_decoys_bline_10m.zip | enhanced_obs + reduced_actions + use_decoys |
| hierarchical_ppo_10m.zip | enhanced_obs + reduced_actions + use_decoys + hierarchical |
| ppo_decoys_mixed_20m.zip | enhanced_obs + reduced_actions + use_decoys |
| recurrent_ppo_lstm_10m.zip | enhanced_obs + reduced_actions + use_decoys (RecurrentPPO) |
| ppo_gentle_mixed_10m.zip | enhanced_obs + reduced_actions + use_decoys (gentle shaping) |
| ppo_enterprise_focus_mixed_10m.zip | enhanced_obs + reduced_actions + use_decoys (enterprise wrapper) |
| masked_ppo_mixed_10m.zip | enhanced_obs + reduced_actions + use_decoys + action_masking |

## Next Steps

1. ✅ All major approaches have been evaluated with seed 200
2. ⚠️ Comprehensive results document should be updated with verified results
3. 💡 Gentle reward shaping deserves re-evaluation as a viable approach
4. 📊 Gap to winning solution (-13.76) is now 4.17 points (not 5.4 as claimed)

---
Generated: April 19, 2026
Evaluations run with: Python 3.12, Stable-Baselines3 2.7.1, Gymnasium 1.2.3
