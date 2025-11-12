"""
Alternative RL Training Script Location

This is an alternative location for your RL algorithm setup.
You can organize your training scripts in the scripts/ directory.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src import HighwayEnvRunner


def main():
    """
    Main training function.
    
    TODO: SETUP YOUR RL ALGORITHM HERE
    ===================================
    
    This is where you should:
    1. Create your environment
    2. Initialize your RL algorithm (PPO, SAC, DQN, etc.)
    3. Configure hyperparameters
    4. Set up logging (TensorBoard, wandb, etc.)
    5. Train the model
    6. Save the trained model
    7. Evaluate the model
    
    Example structure:
    
    # 1. Create environment
    env = HighwayEnvRunner('highway', use_config=True)
    
    # 2. Initialize RL algorithm
    # from stable_baselines3 import PPO
    # model = PPO("MlpPolicy", env, verbose=1)
    
    # 3. Train
    # model.learn(total_timesteps=100000)
    
    # 4. Save
    # model.save("models/highway_ppo")
    
    # 5. Evaluate
    # evaluate_model(model, env, num_episodes=10)
    """
    
    # ============================================================================
    # TODO: YOUR RL ALGORITHM SETUP GOES HERE
    # ============================================================================
    pass


if __name__ == "__main__":
    main()

