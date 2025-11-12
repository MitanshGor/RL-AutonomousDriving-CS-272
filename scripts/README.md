# Training Scripts

This directory contains scripts for training RL models using stable-baselines3.

## Available Training Scripts

### 1. `train_ppo.py` - Proximal Policy Optimization
- **Algorithm:** PPO (Policy Gradient)
- **Action Space:** Both discrete and continuous
- **Best for:** General-purpose RL, stable training

**Usage:**
```bash
# Basic training
python scripts/train_ppo.py --env highway --timesteps 100000

# With observation type
python scripts/train_ppo.py --env highway --obs lidar --timesteps 100000
python scripts/train_ppo.py --env highway --obs grayscale --timesteps 100000

# Different environments
python scripts/train_ppo.py --env merge --timesteps 100000
python scripts/train_ppo.py --env intersection --timesteps 100000
```

### 2. `train_sac.py` - Soft Actor-Critic
- **Algorithm:** SAC (Off-policy, Actor-Critic)
- **Action Space:** Continuous only
- **Best for:** Continuous control tasks

**Usage:**
```bash
# Basic training
python scripts/train_sac.py --env highway --timesteps 100000

# With observation type
python scripts/train_sac.py --env highway --obs lidar --timesteps 100000
```

**Note:** SAC requires continuous action spaces. If your environment has discrete actions, use PPO or DQN instead.

### 3. `train_dqn.py` - Deep Q-Network
- **Algorithm:** DQN (Value-based)
- **Action Space:** Discrete only
- **Best for:** Discrete action spaces, Q-learning

**Usage:**
```bash
# Basic training
python scripts/train_dqn.py --env highway --timesteps 100000

# With observation type
python scripts/train_dqn.py --env highway --obs lidar --timesteps 100000
```

**Note:** DQN requires discrete action spaces. If your environment has continuous actions, use PPO or SAC instead.

### 4. `train_all_models.py` - Train Multiple Models
- **Purpose:** Train all algorithms for comparison
- **Usage:** Trains PPO, SAC, and DQN on the same environment

**Usage:**
```bash
# Train all algorithms
python scripts/train_all_models.py --env highway --timesteps 50000

# Train specific algorithms
python scripts/train_all_models.py --env highway --algorithms ppo sac

# With observation type
python scripts/train_all_models.py --env highway --obs lidar --timesteps 50000
```

## Model Output Structure

Models are saved in the `models/` directory with the following structure:

```
models/
├── ppo_highway_lidar/
│   ├── final_model.zip          # Final trained model
│   ├── best/
│   │   └── best_model.zip       # Best model during training
│   ├── checkpoints/             # Periodic checkpoints
│   └── eval_logs/               # Evaluation logs
├── sac_highway_lidar/
│   └── ...
└── dqn_highway_lidar/
    └── ...
```

## Logs

Training logs (TensorBoard) are saved in the `logs/` directory:

```
logs/
├── ppo_highway_lidar/
├── sac_highway_lidar/
└── dqn_highway_lidar/
```

View logs with TensorBoard:
```bash
tensorboard --logdir logs/
```

## Command Line Arguments

All training scripts support the following arguments:

- `--env`: Environment type (`highway`, `merge`, `intersection`)
- `--obs`: Observation type (`lidar`, `grayscale`, or omit for default)
- `--timesteps`: Total training timesteps (default: 100000)
- `--name`: Custom model name (optional)

## Example: Training for Task 1

For Task 1, you need to train on 3 environments × 2 observation types = 6 combinations:

```bash
# Highway with LidarObservation
python scripts/train_ppo.py --env highway --obs lidar --timesteps 100000

# Highway with GrayscaleObservation
python scripts/train_ppo.py --env highway --obs grayscale --timesteps 100000

# Merge with LidarObservation
python scripts/train_ppo.py --env merge --obs lidar --timesteps 100000

# Merge with GrayscaleObservation
python scripts/train_ppo.py --env merge --obs grayscale --timesteps 100000

# Intersection with LidarObservation
python scripts/train_ppo.py --env intersection --obs lidar --timesteps 100000

# Intersection with GrayscaleObservation
python scripts/train_ppo.py --env intersection --obs grayscale --timesteps 100000
```

## Loading Trained Models

```python
from stable_baselines3 import PPO, SAC, DQN

# Load PPO model
model = PPO.load("models/ppo_highway_lidar/final_model")

# Load SAC model
model = SAC.load("models/sac_highway_lidar/final_model")

# Load DQN model
model = DQN.load("models/dqn_highway_lidar/final_model")
```

