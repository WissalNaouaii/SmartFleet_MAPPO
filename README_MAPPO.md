# SmartFleet MAPPO Training Module

## Multi-Agent Proximal Policy Optimization for Autonomous Delivery

This module implements **MAPPO (Multi-Agent PPO)** training for the SmartFleet delivery system, featuring:

- ✅ **Centralized Training, Decentralized Execution (CTDE)**
- ✅ **Parameter sharing** across agents (optional)
- ✅ **SUMO integration** for realistic traffic simulation
- ✅ **Multi-objective rewards**: on-time delivery, energy efficiency, fairness
- ✅ **Comprehensive logging** and visualization
- ✅ **Checkpoint management** with best model tracking

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_mappo.txt
```

Or manually:
```bash
pip install numpy gymnasium torch "ray[rllib]>=2.9.0" matplotlib
```

### 2. Run Training

```bash
# Basic training (100 iterations)
python smartfleet_mappo_training.py --mode train --iterations 100

# With more agents
python smartfleet_mappo_training.py --mode train --n_agents 8 --iterations 200

# Just run greedy baseline
python smartfleet_mappo_training.py --mode baseline --episodes 10
```

### 3. Evaluate Trained Model

```bash
# Evaluate best checkpoint
python smartfleet_mappo_training.py --mode evaluate --episodes 20

# Evaluate specific checkpoint
python smartfleet_mappo_training.py --mode evaluate --checkpoint ./smartfleet_checkpoints/checkpoint_000100

# Evaluate with SUMO visualization
python smartfleet_mappo_training.py --mode evaluate --use_sumo --episodes 5
```

### 4. Plot Training Curves

```bash
python smartfleet_mappo_training.py --mode plot
```

---

## Architecture

### MAPPO Implementation

```
┌─────────────────────────────────────────────────────────────┐
│                    CENTRALIZED CRITIC                        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Global State = [Agent₀_obs, Agent₁_obs, ..., Metrics]│    │
│  │                          ↓                           │    │
│  │              Value Function V(global_state)          │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              ↓ (training only)
┌─────────────────────────────────────────────────────────────┐
│                   DECENTRALIZED ACTORS                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ Agent 0  │  │ Agent 1  │  │ Agent 2  │  │ Agent 3  │    │
│  │ Policy π │  │ Policy π │  │ Policy π │  │ Policy π │    │
│  │  (local  │  │  (local  │  │  (local  │  │  (local  │    │
│  │   obs)   │  │   obs)   │  │   obs)   │  │   obs)   │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘    │
│       ↓             ↓             ↓             ↓          │
│   Action 0      Action 1      Action 2      Action 3       │
└─────────────────────────────────────────────────────────────┘
```

### Observation Space (58 dimensions per agent)

| Component | Dimensions | Description |
|-----------|------------|-------------|
| Self state | 10 | Position, battery, capacity, speed, heading, tasks |
| Current task | 6 | Relative position, deadline, priority, weight |
| Visible tasks | 30 | Top 5 tasks × 6 features each |
| Neighbors | 12 | Top 3 neighbors × 4 features each |

### Action Space (15 discrete actions)

| Action | ID | Description |
|--------|------|-------------|
| ACCEPT_TASK_0-4 | 0-4 | Accept one of top 5 visible tasks |
| MOVE_NORTH/EAST/SOUTH/WEST | 5-8 | Exploration movement |
| GO_TO_DEPOT | 9 | Return for charging |
| HANDOVER_TO_0-2 | 10-12 | Request task handover to neighbor |
| QUERY_5G | 13 | Query 5G for routing info |
| CONTINUE | 14 | Continue current task |

### Reward Function

```python
reward = -0.01  # Base time penalty

# Delivery rewards
if on_time_delivery:
    reward += 10.0
elif late_delivery < 5min:
    reward += 3.0
else:
    reward -= 8.0

# Other rewards
reward += 0.5 * pickup_completed
reward += 0.2 * task_accepted
reward += 1.0 * handover_accepted  # Cooperation
reward += 2.0 * received_help

# Penalties
reward -= 0.01 * energy_used
reward -= 0.5 * battery_critical
reward -= 5.0 * task_failed
```

---

## File Structure

```
smartfleet/
├── delivery_agent.py              # Standalone RL environment
├── delivery_agent_sumo.py         # SUMO-integrated environment
├── smartfleet_mappo_training.py   # MAPPO training module (this file)
├── requirements_mappo.txt         # Dependencies
├── README_MAPPO.md               # This documentation
│
├── smartfleet.sumocfg            # SUMO configuration
├── finished_map_cleaning.net.xml  # Tunis road network
├── smartfleet_zones.add.xml      # TAZ zones
├── smartfleet_scenario.json      # Verified connected locations
│
└── smartfleet_checkpoints/        # Training outputs
    ├── checkpoint_000010/
    ├── checkpoint_000020/
    ├── best_model/
    └── training_history.json
```

---

## Configuration

### MAPPOConfig Options

```python
@dataclass
class MAPPOConfig:
    # Environment
    n_agents: int = 5           # Number of delivery agents
    use_sumo: bool = True       # Use SUMO traffic simulation
    sim_duration: int = 600     # Simulation duration (seconds)
    
    # Training hyperparameters
    total_timesteps: int = 500_000
    learning_rate: float = 3e-4
    gamma: float = 0.99         # Discount factor
    gae_lambda: float = 0.95    # GAE parameter
    clip_param: float = 0.2     # PPO clipping
    entropy_coeff: float = 0.01 # Exploration bonus
    
    # Batch sizes
    train_batch_size: int = 4000
    sgd_minibatch_size: int = 256
    num_sgd_iter: int = 10
    
    # Multi-agent specific
    use_centralized_critic: bool = True
    share_policy: bool = True   # Parameter sharing across agents
    
    # Hardware
    num_workers: int = 2        # Parallel rollout workers
    num_gpus: float = 0         # GPU usage (set to 1 if available)
```

---

## Expected Results

After training, you should see improvements like:

| Metric | Greedy Baseline | MAPPO (100 iter) | MAPPO (500 iter) |
|--------|-----------------|------------------|------------------|
| Deliveries | 15-25 | 30-50 | 60-100+ |
| On-Time Rate | 60-75% | 80-90% | 95-100% |
| Energy/Delivery | 2.5 kWh | 1.8 kWh | 1.2 kWh |

---

## Programmatic Usage

```python
from smartfleet_mappo_training import MAPPOTrainer, MAPPOConfig

# Configure
config = MAPPOConfig(
    n_agents=5,
    total_timesteps=200_000,
    share_policy=True,
    num_workers=4,
)

# Train
trainer = MAPPOTrainer(config)
history = trainer.train(n_iterations=50)

# Evaluate
results = trainer.evaluate(n_episodes=10)
print(f"Mean deliveries: {results['mean_deliveries']}")
print(f"On-time rate: {results['mean_on_time_rate']:.1%}")

# Clean up
trainer.close()
```

---

## Troubleshooting

### Ray not finding workers
```bash
# Reduce workers or restart Ray
ray stop
python smartfleet_mappo_training.py --mode train
```

### SUMO not found
```bash
# Set SUMO_HOME environment variable
export SUMO_HOME=/path/to/sumo  # Linux/Mac
set SUMO_HOME=C:\Program Files\sumo-1.25.0  # Windows
```

### Out of memory
```python
# Reduce batch size in MAPPOConfig
config = MAPPOConfig(
    train_batch_size=2000,
    sgd_minibatch_size=128,
    num_workers=1,
)
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{smartfleet2025,
  author = {Maram Ouelhazi},
  title = {SmartFleet: Multi-Agent RL for Autonomous Delivery},
  year = {2025},
  publisher = {IEEE RAS x VTS SmartFleet Challenge},
}
```

---

## Author

**Maram Ouelhazi**
- IEEE RAS x VTS SmartFleet Challenge
- Faculty of Sciences of Tunis (FST)
- IoT & Embedded Systems

---

## License

MIT License - See LICENSE file for details.
