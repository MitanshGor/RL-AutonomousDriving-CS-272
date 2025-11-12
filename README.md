# RL-AutonomousDriving-CS-272

Reinforcement Learning project for autonomous driving using HighwayEnv environments.

**Course:** CS 272  
**Team:** Mitansh Gor, Henry Ha, John Yun

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [RL Algorithm Setup](#rl-algorithm-setup)
- [Documentation](#documentation)
- [Requirements](#requirements)

---

## Overview

This project implements deep reinforcement learning (DRL) agents for autonomous driving in simulated highway environments. The project consists of two main tasks:

1. **Task 1:** Train DRL agents on existing HighwayEnv environments (Highway, Merge, Intersection)
2. **Task 2:** Design and evaluate a custom construction zone environment

The project uses the [HighwayEnv](https://highway-env.farama.org/) framework built on Gymnasium.

---

## Project Structure

```
RL-AutonomousDriving-CS-272/
├── src/                    # Source code
│   ├── __init__.py
│   ├── env_runner.py      # Environment wrapper class
│   └── config_loader.py    # Configuration loader utility
├── config/                 # Configuration files
│   └── env_config.json    # Environment configuration
├── scripts/               # Training scripts
│   ├── train_ppo.py       # PPO training script
│   ├── train_sac.py       # SAC training script
│   ├── train_dqn.py       # DQN training script
│   ├── train_all_models.py # Train all models for comparison
│   └── train_rl.py        # Template for custom RL setup
├── examples/              # Example scripts
│   └── example_usage.py  # Usage examples
├── docs/                  # Documentation
│   ├── ConstructionZoneEnv.md
│   └── requierements.md
├── models/                # Trained RL models (generated)
├── logs/                  # Training logs and TensorBoard data (generated)
├── plots/                 # Generated plots and visualizations (generated)
├── results/               # Evaluation results and metrics (generated)
├── requirements.txt       # Python dependencies
└── README.md              # This file
```


---

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (optional, for cloning)

### Step 1: Clone or Navigate to Project

```bash
# If cloning from git
git clone <repository-url>
cd RL-AutonomousDriving-CS-272

# Or navigate to existing project directory
cd /path/to/RL-AutonomousDriving-CS-272
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
# Test imports
python -c "from src import HighwayEnvRunner, ConfigLoader; print('✓ Installation successful!')"
```

---

## Quick Start

### Run Example Script

```bash
# Run the example usage script
python examples/example_usage.py
```

### Train Your First Model

```bash
# Train a PPO model on highway environment (quick test with 10k timesteps)
python scripts/train_ppo.py --env highway --timesteps 10000

# Train with observation type (for Task 1)
python scripts/train_ppo.py --env highway --obs lidar --timesteps 100000
```

### Basic Usage

```python
from src import HighwayEnvRunner

# Create environment
env = HighwayEnvRunner('highway', use_config=True)

# Reset environment
obs, info = env.reset()

# Take a step
action = env.sample_action()  # Random action
obs, reward, terminated, truncated, info = env.step(action)

# Close environment
env.close()
```

---

## Usage

### Available Environments

The project supports three HighwayEnv environments:

- **`highway`** - Highway driving environment
- **`merge`** - Highway merge scenario
- **`intersection`** - Intersection navigation

### Creating Environments

```python
from src import HighwayEnvRunner

# Without visualization (for training)
env = HighwayEnvRunner('highway', use_config=True)

# With visualization (for testing/evaluation)
env = HighwayEnvRunner('highway', render_mode='human', use_config=True)

# With observation type (for Task 1)
env_lidar = HighwayEnvRunner('highway', observation_type='lidar', use_config=True)
env_grayscale = HighwayEnvRunner('highway', observation_type='grayscale', use_config=True)
```

### Using Configuration

The environment automatically loads configuration from `config/env_config.json`:

```python
# Config is loaded by default
env = HighwayEnvRunner('highway', use_config=True)

# Access config values
thresholds = env.get_action_thresholds()
reward_config = env.get_reward_config()
safety_rules = env.get_safety_rules()
```

### Context Manager Usage

```python
# Automatically closes environment
with HighwayEnvRunner('highway', use_config=True) as env:
    obs, info = env.reset()
    action = env.sample_action()
    obs, reward, terminated, truncated, info = env.step(action)
```

---

## RL Algorithm Setup

### Ready-to-Use Training Scripts

The project includes ready-to-use training scripts for three popular RL algorithms:

1. **`scripts/train_ppo.py`** - Proximal Policy Optimization (works with discrete and continuous actions)
2. **`scripts/train_sac.py`** - Soft Actor-Critic (continuous actions only)
3. **`scripts/train_dqn.py`** - Deep Q-Network (discrete actions only)
4. **`scripts/train_all_models.py`** - Train all models for comparison

### Quick Start Training

```bash
# Train PPO on highway environment with lidar observation
python scripts/train_ppo.py --env highway --obs lidar --timesteps 100000

# Train SAC on merge environment
python scripts/train_sac.py --env merge --timesteps 100000

# Train DQN on intersection with grayscale observation
python scripts/train_dqn.py --env intersection --obs grayscale --timesteps 100000

# Train all models for comparison
python scripts/train_all_models.py --env highway --obs lidar --timesteps 50000
```

### Model Output

Trained models are automatically saved in the `models/` directory:
- `models/ppo_{env}_{obs}/final_model.zip` - Final trained model
- `models/ppo_{env}_{obs}/best/best_model.zip` - Best model during training
- `models/ppo_{env}_{obs}/checkpoints/` - Periodic checkpoints

Training logs (TensorBoard) are saved in `logs/`:
```bash
# View training progress
tensorboard --logdir logs/
```

### Output Directories

The project uses the following directories for outputs (created automatically):

- **`models/`** - Trained RL models saved by training scripts
- **`logs/`** - TensorBoard logs and training logs
- **`plots/`** - Generated plots (learning curves, performance tests)
- **`results/`** - Evaluation results and metrics

These directories are created automatically when you run training scripts. They are included in `.gitignore` to avoid committing large files.

### Custom RL Setup

For custom RL implementations, use `scripts/train_rl.py` as a template. See [scripts/README.md](scripts/README.md) for detailed documentation.

### Observation Types

For Task 1, you need to train with two observation types:
- **LidarObservation** (use `observation_type='lidar'`)
- **GrayscaleObservation** (use `observation_type='grayscale'`)

**Usage:**
```python
# Create environment with LidarObservation
env_lidar = HighwayEnvRunner('highway', observation_type='lidar', use_config=True)

# Create environment with GrayscaleObservation
env_grayscale = HighwayEnvRunner('highway', observation_type='grayscale', use_config=True)

# Check observation type
obs_type = env_lidar.get_observation_type()
print(f"Observation type: {obs_type}")  # Output: LidarObservation
```

See [docs/requierements.md](docs/requierements.md) for details.

### Experiment IDs

Organize your experiments using IDs 1-16:
- IDs 1-12: Task 1 (Highway, Merge, Intersection × 2 observation types × 2 plots)
- IDs 13-16: Task 2 (Custom environments × 2 plots)

See [docs/requierements.md](docs/requierements.md) for the complete experiment ID table.

---

## Documentation

- **[docs/requierements.md](docs/requierements.md)** - Project requirements and guidelines
- **[docs/ConstructionZoneEnv.md](docs/ConstructionZoneEnv.md)** - Custom environment specifications
- **[scripts/README.md](scripts/README.md)** - Training scripts documentation

---

## Requirements

### Python Packages

See `requirements.txt` for the complete list. Main dependencies:

- `gymnasium>=0.29.0` - RL environment framework
- `highway-env>=1.8.0` - Highway driving environments
- `numpy>=1.24.0` - Numerical computing

### Additional Packages

The `requirements.txt` already includes:
- `stable-baselines3>=2.0.0` - RL algorithms
- `tensorboard>=2.10.0` - Training visualization

For additional features, you may install:

```bash
# For extra stable-baselines3 features
pip install stable-baselines3[extra]

# For plotting and visualization
pip install matplotlib seaborn
```

---

## Project Tasks

### Task 1: DRL for Existing Environments

Train DRL agents on:
- Highway environment
- Merge environment
- Intersection environment

With two observation types:
- LidarObservation
- GrayscaleObservation

**Deliverables:** 12 plots (3 envs × 2 obs types × 2 plot types)

### Task 2: Custom Environment

Design and evaluate a custom construction zone environment.

**Deliverables:** 4 plots (2 envs × 2 plot types)

See [docs/requierements.md](docs/requierements.md) for complete requirements.

---

## Troubleshooting

### Import Errors

If you encounter import errors:

```bash
# Make sure you're in the project root directory
cd /path/to/RL-AutonomousDriving-CS-272

# Verify virtual environment is activated
which python  # Should point to venv/bin/python

# Reinstall dependencies
pip install -r requirements.txt
```

### Environment Creation Errors

If environment creation fails:

```bash
# Verify highway-env is installed
python -c "import highway_env; print('highway-env installed')"

# Check gymnasium version
python -c "import gymnasium; print(gymnasium.__version__)"
```

### Config File Not Found

The config file should be in `config/env_config.json`. If missing, check that the file exists:

```bash
ls config/env_config.json
```

### Output Directories

The output directories (`models/`, `logs/`, `plots/`, `results/`) are created automatically when you run training scripts. If they don't exist, the training scripts will create them.

To manually create them:
```bash
mkdir -p models logs plots results
```

---

## Contributing

This is a course project. For questions or issues, contact the team members.

---

## License

MIT

---

## Citation

If you use this project, please cite:

**HighwayEnv: A Flexible Gymnasium Environment for Autonomous Driving** — Farama Foundation  
https://github.com/Farama-Foundation/HighwayEnv

---

## Contact

**Team Members:**
- Mitansh Gor (gormitansh@gmail.com)
- Henry Ha (henryha4912@gmail.com)
- John Yun (John.yun2017@gmail.com)

**Course:** CS 272

