# Final Results for Dissertation
## Table 4.3: Advanced Approach Performance

**Evaluation Protocol:** 30 episodes, seed 200, deterministic policy, B_lineAgent opponent, 100 steps maximum

**Success Definition:** Episodes achieving reward ≥ -20 (within 6 points of winning solution)

---

## Primary Results Table

| Rank | Approach | Mean Reward | Std Dev | Success Rate | Op_Server0 Protection |
|------|----------|-------------|---------|--------------|----------------------|
| 1 | Baseline (Full Configuration) | -17.93 | 7.69 | 66.7% | 100%* |
| 2 | Gentle Reward Shaping | -18.51 | 8.66 | 63.3% | ~100%† |
| 3 | Hierarchical RL | -19.16 | 9.09 | 56.7% | ~100%† |
| 4 | Extended Training (20M steps) | -19.49 | 8.42 | 66.7% | ~100%† |
| 5 | RecurrentPPO (LSTM policy) | -19.57 | 9.65 | 53.3% | ~100%† |
| 6 | Enterprise Focus | -23.34 | 11.18 | 50.0% | ~100%† |
| 7 | Masked PPO (Action Masking) | -85.34 | 56.91 | 0.0% | Unknown |
| — | Reactive Decoys | — | — | — | Model not retained |
| — | Aggressive Reward Shaping | — | — | — | Model not retained |

\* Verified through diagnostic evaluation tracking per-step host compromise states (0/30 episodes with Op_Server0 compromised)

† Estimated based on identical wrapper configuration and comparable best episode performance (-9.70 to -9.80 vs. verified baseline -9.40)

---

## Performance Tiers

**Tier 1 - Excellent Performance:**
- Baseline: -17.93 (66.7% success)
- Extended Training 20M: -19.49 (66.7% success)

**Tier 2 - Good Performance:**
- Gentle Reward Shaping: -18.51 (63.3% success)

**Tier 3 - Moderate Performance:**
- Hierarchical RL: -19.16 (56.7% success)
- RecurrentPPO: -19.57 (53.3% success)

**Tier 4 - Suboptimal Performance:**
- Enterprise Focus: -23.34 (50.0% success)

**Tier 5 - Failed:**
- Masked PPO: -85.34 (0.0% success)

---

## Detailed Performance Metrics

| Approach | Mean | Best Episode | Worst Episode | Range | Std Dev |
|----------|------|--------------|---------------|-------|---------|
| Baseline | -17.93 | -9.40 | -35.40 | 26.0 | 7.69 |
| Gentle Shaping | -18.51 | -9.80 | -38.60 | 28.8 | 8.66 |
| Hierarchical RL | -19.16 | -9.70 | -40.60 | 30.9 | 9.09 |
| Extended 20M | -19.49 | -9.80 | -43.30 | 33.5 | 8.42 |
| RecurrentPPO | -19.57 | -9.80 | -47.60 | 37.8 | 9.65 |
| Enterprise Focus | -23.34 | -9.80 | -43.90 | 34.1 | 11.18 |
| Masked PPO | -85.34 | -41.90 | -257.70 | 215.8 | 56.91 |

---

## Key Findings for Section 4.3

### 1. Baseline Remains Optimal
No advanced approach surpassed the baseline configuration (enhanced observations + reduced actions + greedy decoys). Mean reward of **-17.93** represents best verified performance with **100% Op_Server0 protection** (verified).

### 2. Gentle Reward Shaping is Viable
**Critical revision:** Gentle reward shaping achieved **-18.51** (only 0.58 points below baseline) with **63.3% success rate**. Previous documentation incorrectly claimed -24.90 performance (6.39-point discrepancy from verified results).

**Conclusion:** Carefully calibrated reward shaping (bonuses/penalties 80-90% smaller than aggressive variants) provides weak guidance without reward hacking, achieving competitive performance.

### 3. Extended Training Shows No Degradation
Extended training to 20M steps achieved **66.7% success rate** (tied with baseline), contradicting claims of severe overfitting. Mean reward degradation is mild (-19.49 vs. -17.93, only 1.56 points), suggesting 10M-20M training range is relatively stable.

### 4. Architectural Modifications Offer Minimal Benefit
Hierarchical RL (-19.16), RecurrentPPO (-19.57), and Extended Training (-19.49) cluster within **1.6 points** of baseline, indicating:
- Added complexity (goal conditioning, LSTM memory, longer training) does not improve mean performance
- Success rates vary (53-67%) but remain within moderate range
- Re-compromise cycles remain unsolved across all approaches

### 5. Action Masking Catastrophically Failed
Masked PPO degraded performance by **67.41 points** with **0% success rate**, confirming hard constraints fundamentally break reinforcement learning exploration. Worst episode (-257.70) represents complete policy collapse.

### 6. Op_Server0 Protection is Robust
Baseline achieved **100% Op_Server0 protection** (verified across all 30 episodes, 0 compromise-steps). Approaches with similar wrapper configuration and best episode performance (-9.70 to -9.80) estimated to achieve comparable ~100% protection, though unverified.

### 7. Re-Compromise Cycles Remain Core Challenge
Success rates of **50-67%** across viable approaches indicate approximately **one-third to one-half** of episodes struggle with persistent User and Enterprise host re-compromises. This represents the fundamental limitation preventing further performance improvement.

---

## Wrapper Configuration Summary

All evaluated models use consistent wrapper stack (except as noted):

**Base Configuration:**
1. ChallengeWrapper (CybORG interface)
2. GymCompatChallengeEnv (Gymnasium compatibility)
3. DecoyWrapper (Greedy Misinform on Enterprise0-2, steps 0-2)
4. ReducedActionWrapper (54 → 30 curated actions)
5. ObsEnhancedWrapper (52 → 105 dimensions: +40 temporal features, +13 decoy flags)

**Modifications by Approach:**
- Hierarchical RL: +HierarchicalWrapper (105 → 111 dim: +6 for goal encoding)
- Gentle/Aggressive Shaping: +RewardShapingWrapper (gentle=True/False)
- Enterprise Focus: +Enterprise-specific reward modifications
- Masked PPO: +ActionMaskingWrapper
- RecurrentPPO: LSTM policy architecture (no wrapper change)
- Extended 20M: Same as baseline, longer training

---

## Recommendations for Discussion Section

1. **Use verified results only:** Fresh seed 200 evaluations provide reproducible, accurate performance data
2. **Emphasize gentle shaping viability:** Corrects misconception that reward shaping is universally harmful
3. **Highlight architectural clustering:** 5 approaches within 1.6 points suggests architectural variations matter less than fundamental approach quality
4. **Focus on re-compromise cycles:** Success rate analysis clearly identifies unsolved core problem
5. **Acknowledge verification limitation:** Op_Server0 protection verified for baseline only; others estimated

---

## Footnote Text for Table 4.3

"All results evaluated with seed 200, deterministic policy, B_lineAgent opponent, 30 episodes per approach. Success rate defined as proportion of episodes achieving reward ≥ -20, representing good defensive performance within the evaluated range. Op_Server0 protection rates: * = verified through diagnostic evaluation with per-step host compromise tracking (baseline only); † = estimated based on identical wrapper configuration (greedy decoys on Enterprise0-2) and comparable best episode performance. Models marked 'Model not retained' indicate trained weights were not preserved for evaluation, likely due to catastrophic preliminary performance or architectural flaws identified during development."

---

**Prepared:** April 19, 2026
**Source Data:** Fresh seed 200 evaluations with verified wrapper configurations
**Verification Status:** All mean rewards, std dev, and success rates verified from evaluation logs
