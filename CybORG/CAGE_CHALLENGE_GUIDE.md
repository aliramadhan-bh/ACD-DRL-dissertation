# CAGE Challenge 2 - Complete Reference Guide

## Overview

CAGE Challenge 2 is a cyber defense simulation where a **Blue agent** (defender) protects a network against a **Red agent** (attacker). Your goal is to build a Blue agent that minimizes damage.

---

## Network Topology

```
User Subnet (10.0.205.x)          Enterprise Subnet (10.0.250.x)       Operational Subnet (10.0.111.x)
┌─────────────────────┐           ┌─────────────────────┐              ┌─────────────────────┐
│ User0 (Red starts)  │           │ Enterprise0         │              │ Op_Host0            │
│ User1               │  ──────►  │ Enterprise1         │  ──────►     │ Op_Host1            │
│ User2               │           │ Enterprise2         │              │ Op_Host2            │
│ User3               │           │ Defender            │              │ Op_Server0 (TARGET) │
│ User4               │           └─────────────────────┘              └─────────────────────┘
└─────────────────────┘
```

### All Hosts

| Host | Subnet | Notes |
|------|--------|-------|
| User0 | User | Red's starting point (cannot be cleaned) |
| User1-4 | User | Entry points for Red |
| Enterprise0-2 | Enterprise | Middle tier |
| Defender | Enterprise | Blue's base |
| Op_Host0-2 | Operational | High value |
| Op_Server0 | Operational | **Critical target** - Red's goal |

---

## Red Agent Behavior

### B_lineAgent (Aggressive)

Follows a fixed **kill chain**:

1. **Discover** User subnet → find hosts
2. **Exploit** a User host → gain access
3. **Privilege Escalate** → get root
4. **Pivot** to Enterprise subnet
5. **Exploit** Enterprise hosts
6. **Pivot** to Operational subnet
7. **Exploit Op_Server0** → the target
8. **Impact** → cause damage (catastrophic for Blue)

**Key insight:** B_lineAgent is predictable. It follows the same path every time.

### Meander (Random)

- More unpredictable
- Randomly explores the network
- Harder to defend against

---

## Blue Agent Actions

| Action | What it does | Cost | When to use |
|--------|--------------|------|-------------|
| `Monitor` | Scan all hosts for suspicious activity | None | Every turn - your "eyes" |
| `Analyse` | Deep inspection of one host | None | When you suspect a specific host |
| `Remove` | Remove malware from a host | None | When you detect an intrusion |
| `Restore` | Full host restoration | Small | Nuclear option - removes everything |
| `Misinform` | Plant decoy information | None | Advanced deception |
| `Sleep` | Do nothing | None | Never useful |

### Action Limitations

- Blue can only perform **one action per turn**
- `Monitor` scans everything but may miss stealthy attacks
- `Remove` only works if malware is detected
- `Restore` always works but has a cost

---

## Rewards

| Event | Reward |
|-------|--------|
| Normal operation | 0 |
| Host compromised | -0.1 per host per turn |
| Op_Server0 impacted | -10 (catastrophic) |
| Using Restore | Small negative cost |

**Goal:** Minimize total negative reward over 100 steps.

**Typical scores:**
- Doing nothing: ~ -50 to -100
- Basic React agent: ~ -5 to -20
- Good agent: ~ 0 to -5

---

## Observations

When Blue calls `Monitor`, it receives:

```python
{
    'success': True,
    'User1': {
        'System info': {'Hostname': 'User1'},
        'Processes': [{'PID': 1234, ...}],  # Suspicious!
        'Interface': [{'IP Address': '10.0.205.137'}]
    },
    ...
}
```

### Detection Rule

If a host has `Processes` with `PID`, it's likely compromised:

```python
for host_name, host_info in observation.items():
    if host_name != 'success':
        if 'Processes' in host_info:
            if len([p for p in host_info['Processes'] if 'PID' in p]) > 0:
                # This host is compromised!
```

---

## Critical Constraints

1. **User0 cannot be cleaned** - Red always maintains a foothold there
2. **One action per turn** - Choose wisely
3. **Monitor doesn't see everything** - Red can be stealthy
4. **Restore has a cost** - Don't spam it
5. **100 steps per episode** - Plan accordingly

---

## Project Structure

```
CybORG/
├── .venv/                 # Virtual environment
├── SETUP_GUIDE.md         # Setup instructions
├── CAGE_CHALLENGE_GUIDE.md # This file
├── verify_setup.py        # Verification script
├── explore_challenge.py   # Learning demo
├── setup.py              # Package installer
│
└── CybORG/               # The main package
    ├── Agents/           # ⭐ Agent implementations
    │   ├── SimpleAgents/ # Simple rule-based agents
    │   │   ├── BaseAgent.py        # Base class (inherit from this)
    │   │   ├── B_line.py           # Red attacker (aggressive)
    │   │   ├── Meander.py          # Red attacker (random)
    │   │   ├── BlueReactAgent.py   # Example Blue agents
    │   │   └── BlueMonitorAgent.py # Simplest Blue agent
    │   └── Wrappers/     # Gym wrappers for RL
    │
    ├── CybORG.py         # ⭐ Main environment class
    │
    ├── Evaluation/       # ⭐ Scoring scripts
    │   └── evaluation.py
    │
    ├── Shared/           # Actions and scenarios
    │   ├── Actions/      # All available actions
    │   │   └── AbstractActions/  # Monitor, Remove, Restore, etc.
    │   └── Scenarios/    # Network definitions
    │       └── Scenario1b.yaml   # Challenge scenario
    │
    ├── Simulator/        # Internal (don't modify)
    ├── Tests/            # Unit tests
    └── Tutorial/         # Jupyter notebooks
```

---

## Building a Blue Agent

### Basic Structure

```python
from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Shared.Actions import Monitor, Remove, Restore, Analyse

class MyBlueAgent(BaseAgent):
    def __init__(self):
        # Initialize any state
        self.compromised_hosts = []

    def get_action(self, observation, action_space):
        # Your decision logic here
        session = list(action_space['session'].keys())[0]

        # Example: always monitor
        return Monitor(agent='Blue', session=session)

    def train(self, results):
        pass  # Empty for non-learning agents

    def end_episode(self):
        # Reset state between episodes
        self.compromised_hosts = []

    def set_initial_values(self, action_space, observation):
        pass
```

### Running Your Agent

```python
from CybORG import CybORG
from CybORG.Agents import B_lineAgent
import inspect

# Setup
path = str(inspect.getfile(CybORG))
path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'

# Create environment
cyborg = CybORG(path, 'sim', agents={'Red': B_lineAgent})

# Your agent
my_agent = MyBlueAgent()

# Run episode
results = cyborg.reset('Blue')
total_reward = 0

for step in range(100):
    action = my_agent.get_action(results.observation, results.action_space)
    results = cyborg.step(agent='Blue', action=action)
    total_reward += results.reward

print(f"Final score: {total_reward}")
```

---

## Example Strategies

### 1. React Strategy (Existing)
```
Monitor → Detect compromised host → Remove malware → Repeat
```

### 2. Proactive Strategy
```
Cycle through critical hosts → Restore preemptively → Focus on Op_Server0
```

### 3. Priority-Based Strategy
```
Protect Op_Server0 first → Then Enterprise hosts → Then User hosts
```

### 4. Predictive Strategy
```
Know Red's path → Intercept before they reach critical hosts
```

---

## Files to Study (In Order)

| Priority | File | Why |
|----------|------|-----|
| 1 | `Agents/SimpleAgents/BlueReactAgent.py` | Working Blue agent example |
| 2 | `Agents/SimpleAgents/B_line.py` | Understand Red's attack |
| 3 | `Agents/SimpleAgents/BaseAgent.py` | Interface you must implement |
| 4 | `Evaluation/evaluation.py` | How scoring works |
| 5 | `Shared/Scenarios/Scenario1b.yaml` | Network configuration |

---

## Quick Commands

```bash
# Activate environment
cd CybORG
source .venv/bin/activate

# Verify setup
python verify_setup.py

# Explore the challenge
python explore_challenge.py

# Run training example
python -c "from CybORG.Agents.training_example import run_training_example; run_training_example('Scenario1b')"

# Run tests
pytest CybORG/Tests/test_sim/test_sim_Cyborg.py -v
```

---

## Imports Cheatsheet

```python
# Environment
from CybORG import CybORG

# Base class for agents
from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent

# Blue actions
from CybORG.Shared.Actions import Monitor, Remove, Restore, Analyse, Sleep

# Red agents (opponents)
from CybORG.Agents import B_lineAgent
from CybORG.Agents.SimpleAgents.Meander import Meander

# Existing Blue agents (for reference)
from CybORG.Agents import BlueReactRemoveAgent, BlueReactRestoreAgent
```
