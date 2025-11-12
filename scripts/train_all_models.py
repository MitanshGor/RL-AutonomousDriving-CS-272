"""
Train multiple models (PPO, SAC, DQN) for quick comparison.

This script trains all three algorithms on the same environment for comparison.
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import training functions
sys.path.insert(0, str(Path(__file__).parent))
from train_ppo import train_ppo
from train_sac import train_sac
from train_dqn import train_dqn


def train_all_models(
    env_type: str = 'highway',
    observation_type: str = None,
    total_timesteps: int = 50000,
    algorithms: list = None
):
    """
    Train multiple RL algorithms on the same environment.
    
    Args:
        env_type: Environment type ('highway', 'merge', 'intersection')
        observation_type: Observation type ('lidar', 'grayscale', None)
        total_timesteps: Total training timesteps per algorithm
        algorithms: List of algorithms to train ['ppo', 'sac', 'dqn']. If None, trains all.
    """
    if algorithms is None:
        algorithms = ['ppo', 'sac', 'dqn']
    
    print("=" * 60)
    print(f"Training Multiple Models on {env_type}")
    if observation_type:
        print(f"Observation Type: {observation_type}")
    print("=" * 60)
    
    results = {}
    
    # Train PPO
    if 'ppo' in algorithms:
        print("\n" + "=" * 60)
        print("Training PPO")
        print("=" * 60)
        try:
            model = train_ppo(
                env_type=env_type,
                observation_type=observation_type,
                total_timesteps=total_timesteps
            )
            results['ppo'] = 'success'
        except Exception as e:
            print(f"❌ PPO training failed: {e}")
            results['ppo'] = f'failed: {e}'
    
    # Train SAC
    if 'sac' in algorithms:
        print("\n" + "=" * 60)
        print("Training SAC")
        print("=" * 60)
        try:
            model = train_sac(
                env_type=env_type,
                observation_type=observation_type,
                total_timesteps=total_timesteps
            )
            results['sac'] = 'success'
        except Exception as e:
            print(f"❌ SAC training failed: {e}")
            results['sac'] = f'failed: {e}'
    
    # Train DQN
    if 'dqn' in algorithms:
        print("\n" + "=" * 60)
        print("Training DQN")
        print("=" * 60)
        try:
            model = train_dqn(
                env_type=env_type,
                observation_type=observation_type,
                total_timesteps=total_timesteps
            )
            results['dqn'] = 'success'
        except Exception as e:
            print(f"❌ DQN training failed: {e}")
            results['dqn'] = f'failed: {e}'
    
    # Summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    for algo, status in results.items():
        status_icon = "✓" if status == 'success' else "❌"
        print(f"{status_icon} {algo.upper()}: {status}")
    
    print(f"\nAll models saved in: models/")
    print(f"All logs saved in: logs/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multiple RL models for comparison")
    parser.add_argument(
        "--env",
        type=str,
        default="highway",
        choices=["highway", "merge", "intersection"],
        help="Environment type"
    )
    parser.add_argument(
        "--obs",
        type=str,
        default=None,
        choices=["lidar", "grayscale", None],
        help="Observation type (lidar, grayscale, or None for default)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50000,
        help="Total training timesteps per algorithm"
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="+",
        default=['ppo', 'sac', 'dqn'],
        choices=['ppo', 'sac', 'dqn'],
        help="Algorithms to train"
    )
    
    args = parser.parse_args()
    
    train_all_models(
        env_type=args.env,
        observation_type=args.obs,
        total_timesteps=args.timesteps,
        algorithms=args.algorithms
    )

