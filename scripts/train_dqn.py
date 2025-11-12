"""
Train DQN (Deep Q-Network) model on HighwayEnv environments.

DQN is a value-based method suitable for discrete action spaces.
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src import HighwayEnvRunner
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import os


def train_dqn(
    env_type: str = 'highway',
    observation_type: str = None,
    total_timesteps: int = 100000,
    model_name: str = None,
    save_path: str = None
):
    """
    Train a DQN model on the specified environment.
    
    Args:
        env_type: Environment type ('highway', 'merge', 'intersection')
        observation_type: Observation type ('lidar', 'grayscale', None for default)
        total_timesteps: Total training timesteps
        model_name: Name for the saved model
        save_path: Path to save the model (default: models/dqn_{env_type}_{obs_type})
    """
    print("=" * 60)
    print("Training DQN Model")
    print("=" * 60)
    
    # Create environment
    print(f"\nCreating {env_type} environment...")
    if observation_type:
        print(f"Observation type: {observation_type}")
        env = HighwayEnvRunner(env_type, observation_type=observation_type, use_config=True)
    else:
        env = HighwayEnvRunner(env_type, use_config=True)
    
    # Get environment info
    env_info = env.get_env_info()
    print(f"Environment: {env_info['env_id']}")
    print(f"Observation type: {env_info.get('observation_type', 'Default')}")
    print(f"Observation shape: {env_info['observation_shape']}")
    print(f"Action space: {env_info['action_space']}")
    print(f"Action space type: {'Discrete' if env_info['is_discrete'] else 'Continuous'}")
    
    # Check if action space is discrete (DQN requires discrete actions)
    if env_info['is_continuous']:
        print("\n⚠️  Warning: DQN requires discrete action space.")
        print("This environment has continuous actions. Consider using PPO or SAC instead.")
        env.close()
        return None
    
    # Create evaluation environment
    eval_env = HighwayEnvRunner(env_type, observation_type=observation_type, use_config=True)
    
    # Setup save path
    if save_path is None:
        obs_suffix = f"_{observation_type}" if observation_type else "_default"
        save_path = f"models/dqn_{env_type}{obs_suffix}"
    
    os.makedirs(save_path, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Setup logging
    log_dir = f"logs/dqn_{env_type}"
    if observation_type:
        log_dir += f"_{observation_type}"
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"\nModel will be saved to: {save_path}")
    print(f"Logs will be saved to: {log_dir}")
    
    # Initialize DQN model
    print("\nInitializing DQN model...")
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=1e-4,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        max_grad_norm=10.0,
    )
    
    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_path}/best",
        log_path=f"{save_path}/eval_logs",
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"{save_path}/checkpoints",
        name_prefix="dqn_model"
    )
    
    # Train the model
    print(f"\nStarting training for {total_timesteps} timesteps...")
    print("-" * 60)
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save final model
    final_model_path = f"{save_path}/final_model"
    model.save(final_model_path)
    print(f"\n✓ Training completed!")
    print(f"✓ Final model saved to: {final_model_path}")
    print(f"✓ Best model saved to: {save_path}/best/best_model")
    
    # Close environments
    env.close()
    eval_env.close()
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN model on HighwayEnv")
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
        default=100000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Custom model name"
    )
    
    args = parser.parse_args()
    
    train_dqn(
        env_type=args.env,
        observation_type=args.obs,
        total_timesteps=args.timesteps,
        model_name=args.name
    )

