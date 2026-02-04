# SmartFleet: Multi-Agent Delivery System with MAPPO

A multi-agent reinforcement learning system for urban delivery fleet coordination, built for the **IEEE RAS x VTS SmartFleet Challenge**.


## 🎯 Project Overview

This project implements an intelligent delivery fleet management system using:
- **SUMO** (Simulation of Urban Mobility) for realistic traffic simulation
- **MAPPO** (Multi-Agent Proximal Policy Optimization) for cooperative learning
- **Ray RLlib** for scalable distributed training
- Real **Tunis city road network** integration

## 📊 Results

| Metric | Greedy Baseline | MAPPO Trained |
|--------|----------------|---------------|
| Deliveries | 112.7 | 110+ |
| On-Time Rate | 100% | 99.6% |
| Energy Efficiency | 13.78 kWh/delivery | Optimized |

## 🚗 Vehicle Types

- **Truck**: High capacity (500kg), long range
- **Van**: Medium capacity (200kg), balanced
- **Drone**: Fast, limited capacity (5kg), aerial routes
- **Robot**: Small capacity (20kg), last-mile delivery

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────┐
│              CENTRALIZED CRITIC (Training)          │
│         Global State → Value Function V(s)          │
└─────────────────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   Agent 0    │ │   Agent 1    │ │   Agent N    │
│  π(obs) → a  │ │  π(obs) → a  │ │  π(obs) → a  │
│   (Truck)    │ │    (Van)     │ │   (Drone)    │
└──────────────┘ └──────────────┘ └──────────────┘
        │                │                │
        └────────────────┴────────────────┘
                         │
              ┌──────────▼──────────┐
              │   SUMO Simulation   │
              │  (Tunis Road Net)   │
              └─────────────────────┘
```

## 🚀 Quick Start

### Installation
```bash
# Create virtual environment
python -m venv smartfleet_env
source smartfleet_env/bin/activate  # Linux/Mac
# or: .\smartfleet_env\Scripts\Activate  # Windows

# Install dependencies
pip install numpy gymnasium torch matplotlib
pip install "ray[rllib]"

# Install SUMO (optional, for full simulation)
# Download from: https://sumo.dlr.de/docs/Downloads.php
```

### Run Demo (Greedy Baseline)
```bash
python smartfleet_demo.py
```

### Train MAPPO
```bash
# Quick training (10 iterations)
python smartfleet_mappo_training.py --mode train --iterations 10

# Full training (100+ iterations recommended)
python smartfleet_mappo_training.py --mode train --iterations 100
```

### Evaluate Trained Model
```bash
python smartfleet_mappo_training.py --mode evaluate --episodes 10
```

## 📁 Project Structure
```
SmartFleet_Sim/
├── delivery_agent_sumo.py      # Main environment with SUMO integration
├── smartfleet_mappo_training.py # MAPPO training pipeline
├── smartfleet_demo.py          # Quick demo script
├── create_scenario.py          # Scenario generation
├── finished_map_cleaning.net.xml # Tunis road network
├── smartfleet_scenario.json    # Delivery zones configuration
├── vehicle_types.add.xml       # Vehicle specifications
├── smartfleet_checkpoints/     # Trained model weights
├── linkedin_visuals/           # Result visualizations
└── training_curves.png         # Training progress plot
```

## 🔧 Key Features

- **Multi-agent coordination**: 5 agents with different vehicle types
- **Realistic simulation**: SUMO traffic with real Tunis map data
- **Smart task allocation**: Priority-based with deadline awareness
- **Energy optimization**: Battery management and efficient routing
- **Scalable training**: Ray RLlib for distributed learning

## 📈 Action Space (15 discrete actions)

| Action | Description |
|--------|-------------|
| 0-4 | Accept visible task (by index) |
| 5-8 | Move (North/South/East/West) |
| 9 | Return to depot |
| 10-12 | Handover task to nearby agent |
| 13 | Query 5G for task info |
| 14 | Continue current action |


This project was developed for the **IEEE RAS x VTS SmartFleet Challenge** - a multi-agent reinforcement learning competition for sustainable urban logistics.
