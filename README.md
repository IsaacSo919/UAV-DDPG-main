# UAV-DDPG: UAV-Assisted Mobile Edge Computing with Deep Reinforcement Learning

A Deep Reinforcement Learning system that optimises drone-assisted Mobile Edge Computing (MEC) using Deep Deterministic Policy Gradient (DDPG). The UAV intelligently decides where to fly, which user to serve, and how much computation to offload — balancing energy consumption against latency.

## Table of Contents
- [Overview](#overview)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Results](#results)
- [Technical Details](#technical-details)
- [References](#references)

## Overview
Mobile devices often lack the computation power and battery life to process complex tasks locally with low latency. While cloud offloading is an option, it introduces significant communication delays. This project explores using a UAV (drone) as a mobile edge computing server — flying directly to users to provide temporary, high-speed computation resources.

The drone must learn an optimal strategy for three coupled decisions:

| Decision | Description | Action Type |
| :--- | :--- | :--- |
| **Who to serve** | Which user's task to process | Discrete (via continuous mapping) |
| **Where to fly** | Flight angle and distance | Continuous |
| **How much to offload** | Ratio of task computed on UAV vs. locally | Continuous |

Because these decisions involve continuous action spaces and multi-objective trade-offs (energy vs. latency), the system uses DDPG — an Actor-Critic reinforcement learning algorithm designed for continuous control.

## How It Works

```text
┌─────────────────────────────────────────────────────────────┐
│                      DDPG Agent (DRL_algo.py)               │
│                                                             │
│   ┌──────────┐    actions     ┌──────────────────────┐      │
│   │  Actor   │───────────────►│                      │      │
│   │ Network  │                │    UAV Environment   │      │
│   └──────────┘   ◄────────── │   (UAV_environment.py)│      │
│   ┌──────────┐  state,reward │                      │      │
│   │  Critic  │◄──────────────│  • Physics simulation │      │
│   │ Network  │               │  • Channel model      │      │
│   └──────────┘               │  • Energy model       │      │
│                              │  • Delay calculation  │      │
│   ┌──────────────────┐       └──────────────────────┘      │
│   │  Replay Buffer   │                                      │
│   │  (10,000 trans.) │                                      │
│   └──────────────────┘                                      │
└─────────────────────────────────────────────────────────────┘
```

### Environment (`UAV_environment.py`)
A physics and communication simulator modelling a 100m × 100m area with:
- **1 UAV** at 100m altitude, starting at the centre (50, 50)
- **4 mobile users (UEs)** randomly placed on the ground
- Air-to-ground channel model with Line-of-Sight (LoS) and Non-Line-of-Sight (NLoS) conditions
- Energy models for propulsion, computation (DVFS), and communication
- Shannon capacity for uplink data rate calculation
- Parallel execution model: local and offloaded computation run simultaneously; total delay = `max(local, upload + edge)`

### Agent (`DRL_algo.py`)
A DDPG agent implemented in PyTorch with:
- **Actor Network**: State → Action (400 → 300 → 10 → action_dim, ReLU6/Tanh)
- **Critic Network**: (State, Action) → Q-value (separate state/action input layers, 400 → 300 → 10 → 1)
- **Target Networks**: Soft-updated via Polyak averaging (τ = 0.01)
- **Experience Replay**: Circular buffer with 10,000 capacity

### Reward Function
```text
R = -(α · E_total + β · Latency)
```
Both energy and latency are normalised to approximately the same scale. The agent learns to minimise this penalty over 1,000 episodes (40 time steps each).

## Project Structure

```text
UAV-DDPG/
├── Deep_Reinforcement_learning/
│   ├── UAV_environment.py          # Gymnasium-compatible MEC environment
│   ├── DRL_algo.py                 # DDPG agent and training loop (PyTorch)
│   ├── state_normalization.py      # State normalisation utility
│   ├── output.txt                  # Training logs (generated at runtime)
│   └── learning_curve.png          # Reward plot (generated after training)
└── README.md
```

## Getting Started

### Prerequisites
- Python 3.10 or 3.11 (recommended)
- `pip` package manager

### Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/UAV-DDPG.git
cd UAV-DDPG
```

2. Create and activate a virtual environment:
```bash
# Create
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install --upgrade pip
pip install torch numpy matplotlib gymnasium
```

### Run Training
Navigate to the source folder and start the script:
```bash
cd Deep_Reinforcement_learning
python DRL_algo.py
```

### Expected Output
```text
Episode:    0 | Steps: 39 | Reward:  -52.31 | Explore: 0.010
Episode:    1 | Steps: 39 | Reward:  -48.76 | Explore: 0.010
...
Episode:  999 | Steps: 22 | Reward:  -15.43 | Explore: 0.010
```
Training generates:
- `output.txt` — Per-step logs (UE served, task size, offloading ratio, delay, UAV position)
- `learning_curve.png` — Reward progression over episodes

## Configuration

Key hyperparameters in `DRL_algo.py`:
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `MAX_EPISODES` | 1000 | Total training episodes |
| `LR_A` | 0.001 | Actor learning rate |
| `LR_C` | 0.002 | Critic learning rate |
| `GAMMA` | 0.001 | Reward discount factor |
| `TAU` | 0.01 | Soft target update rate |
| `MEMORY_CAPACITY` | 10000 | Replay buffer size |
| `BATCH_SIZE` | 64 | Training batch size |
| `VAR_MIN` | 0.01 | Minimum exploration noise |

Key environment parameters in `UAV_environment.py`:
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `M` | 4 | Number of ground users |
| `height` | 100m | UAV flight altitude |
| `flight_speed` | 50 m/s | Maximum UAV speed |
| `f_uav` | 1.2 GHz | UAV CPU frequency |
| `f_ue` | 0.2 GHz | User device CPU frequency |
| `e_battery_uav` | 500 kJ | UAV battery capacity |
| `T` | 320s | Total mission time |
| `slot_num` | 40 | Decision steps per episode |

## Results
Over 1,000 episodes, the DDPG agent learns to reduce the combined energy-latency penalty:

*(Learning curve graph: `Deep_Reinforcement_learning/learning_curve.png`)*

The trained agent discovers strategies such as:
- Flying toward clusters of users to improve channel quality for multiple tasks
- Adjusting offloading ratios based on remaining battery
- Conserving flight energy when channel conditions are already favourable

## Technical Details

### State Space (20 dimensions)
| Component | Dims | Description |
| :--- | :--- | :--- |
| UAV battery | 1 | Remaining energy (J) |
| UAV position | 2 | (x, y) coordinates |
| Remaining tasks | 1 | Total bits left to compute |
| UE positions | 8 | 4 users × (x, y) |
| UE task sizes | 4 | Current task per user (bits) |
| UE block flags | 4 | LoS (0) or NLoS (1) per user |

### Action Space (4 dimensions, continuous)
| Index | Range | Maps To |
| :--- | :--- | :--- |
| 0 | `[-1, 1]` → `[0, 1]` | UE selection (discretised) |
| 1 | `[-1, 1]` → `[0, 2π]` | Flight angle |
| 2 | `[-1, 1]` → `[0, 50m]` | Flight distance |
| 3 | `[-1, 1]` → `[0, 1]` | Offloading ratio |

### Key Equations
**Shannon Capacity (Data Rate):**
```text
R = B × log₂(1 + P_tx × g / σ²)
```

**Channel Gain (Free-Space Path Loss):**
```text
g = α₀ / d²
```

**Computation Energy (DVFS Model):**
```text
E_comp = κ × f³ × t
```

**Delay Model (Parallel Execution):**
```text
D = max(t_upload + t_edge, t_local)
```

### Why DDPG?
| Algorithm | Action Space | Why Not? |
| :--- | :--- | :--- |
| DQN | Discrete only | Can't output continuous flight angles/distances |
| PPO | Continuous | On-policy; less sample-efficient with replay |
| TD3 | Continuous | Valid upgrade path; DDPG is simpler baseline |
| **DDPG** | **Continuous** | **Native continuous control with Actor-Critic** |

## References
- DDPG Algorithm: Lillicrap et al., "Continuous Control with Deep Reinforcement Learning", ICLR 2016
- UAV Propulsion Model: Jeong et al., "Mobile Edge Computing via a UAV-Mounted Cloudlet: Optimization of Bit Allocation and Path Planning", IEEE 2018
- Air-to-Ground Channel: Al-Hourani et al., "Modeling Air-to-Ground Path Loss for Low Altitude Platforms in Urban Environments", IEEE 2014

## License
This project is for educational and research purposes.
