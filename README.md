# GRU-MATD3 for Multi-Agent UAV Swarm RL

## Overview
This repository implements **GRU-enhanced Multi-Agent Twin Delayed Deep Deterministic Policy Gradient (GRU-MATD3)** for training UAV swarms in a **multi-agent reinforcement learning (MARL) environment**. The algorithm improves **coordination, decision-making, and exploration efficiency** in UAV-based flood assessment and disaster response scenarios.

### Key Features
- **Temporal Dependency Handling**: GRU captures sequential patterns and maintains memory for partially observable environments.
- **MATD3 for Multi-Agent Stability**: Reduces overestimation bias, improves sample efficiency, and stabilizes training.
- **UAV Swarm Exploration**: Ensures full coverage of a disaster-affected area while avoiding redundant paths and collisions.
- **CUDA-Optimized Training**: Uses GPU acceleration for faster training with PyTorch.

---

## Installation
### Prerequisites
Ensure you have the following dependencies installed:

```bash
pip install torch pybullet numpy gym matplotlib stable-baselines3
```

For **GPU acceleration**, install PyTorch with CUDA:
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```

---

## Usage
### 1. Training the UAV Swarm
Run the main training script:
```bash
python training.py --agents 3 --episodes 10000 --use-gru True
```
- `--agents`: Number of UAVs (default: 3)
- `--episodes`: Number of training episodes
- `--use-gru`: Enables GRU for handling temporal dependencies

### 2. Visualizing the Trained Model
```bash
python visualize.py 
```

---

## Algorithm Details
### **1. GRU for Temporal Memory**
- Each UAV receives **partial observations** from the environment.
- GRU processes historical observations, enabling **better decision-making** over time.
- Helps UAVs **remember visited areas** and optimize path planning.

### **2. MATD3 for Multi-Agent Learning**
- **Twin Delayed DDPG** reduces overestimation bias in Q-value estimation.
- **Multi-Agent Adaptation** ensures better credit assignment and stable training.
- Improves coordination among UAVs, reducing redundant coverage and collisions.

### **3. Reward Function**
The UAV swarm is rewarded based on:
- **Area Coverage**: Positive reward for discovering new grid cells.
- **Flood Severity Mapping**: Incentivizes capturing high-risk areas.
- **Battery Efficiency**: Penalizes inefficient movements.
- **Collision Avoidance**: Negative reward for overlapping paths.

---

## Results & Visualizations
During training, the system tracks:
- **Coverage Rate**: Percentage of the total area explored.
- **Reward Progression**: Learning curve over episodes.
- **Collision Rate**: Frequency of UAV collisions.

---


