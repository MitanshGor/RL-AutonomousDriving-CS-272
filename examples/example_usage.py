"""
Example usage of HighwayEnvRunner with RL algorithms.

This demonstrates how to use the HighwayEnvRunner class
with different RL algorithms.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src import HighwayEnvRunner
import numpy as np


def example_random_agent():
    """Example: Using the environment with a random agent."""
    print("=" * 60)
    print("Example: Random Agent with Config")
    print("=" * 60)
    
    # Create environment with config (default: loads env_config.json)
    env = HighwayEnvRunner('highway', use_config=True)
    
    # Display config information
    if env.get_config() is not None:
        print("\nConfig loaded successfully!")
        print(f"Action thresholds: {env.get_action_thresholds()}")
        print(f"Reward config: {env.get_reward_config()}")
        print(f"Episode config: {env.get_episode_config()}")
    
    # Display observation type
    obs_type = env.get_observation_type()
    print(f"\nObservation type: {obs_type if obs_type else 'Default'}")
    
    # Get environment info
    info = env.get_env_info()
    print(f"\nEnvironment: {info['env_id']}")
    print(f"Observation shape: {info['observation_shape']}")
    print(f"Action space: {info['action_space']}")
    print(f"Is discrete: {info['is_discrete']}")
    print(f"Is continuous: {info['is_continuous']}")
    
    # Run a few episodes
    num_episodes = 3
    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated) and steps < 1000:
            # Sample random action
            action = env.sample_action()
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
        
        print(f"\nEpisode {episode + 1}:")
        print(f"  Steps: {steps}")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")
    
    env.close()
    print("\n" + "=" * 60)


def example_with_visualization(envType: str = 'highway'):
    """Example: Using the environment with visualization."""
    print("=" * 60)
    print("Example: With Visualization and Config")
    print("=" * 60)
    
    # Create environment with visualization and config
    env = HighwayEnvRunner(envType, render_mode='human', use_config=True)
    
    obs, info = env.reset()
    total_reward = 0
    steps = 0
    terminated = False
    truncated = False
    
    print("\nRunning 4 episodes with visualization...")
    print("Close the visualization window to stop.")

    num_episodes = 4
    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated) and steps < 500:
            # Sample random action
            action = env.sample_action()

            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            # Render (if render_mode='human', this happens automatically)
            env.render()

        print(f"\nEpisode {episode + 1} completed:")
        print(f"  Steps: {steps}")
        print(f"  Total Reward: {total_reward:.2f}")

    env.close()
    print("=" * 60)






if __name__ == "__main__":
    # Run examples
    # example_with_visualization('highway')
    # example_with_visualization('merge') 
    # example_with_visualization('intersection')
    # print("\n")
    

    
    # # Example: Using observation types (for Task 1)
    print("=" * 60)
    print("Example: Observation Types (Task 1)")
    print("=" * 60)
    
    obs_types = ['lidar', 'grayscale']
    for obs_type in obs_types:
        print(f"\nTesting with {obs_type} observation...")
        env = HighwayEnvRunner('highway', observation_type=obs_type, use_config=True)
        info = env.get_env_info()
        print(f"  Observation type: {info['observation_type']}")
        print(f"  Observation shape: {info['observation_shape']}")
        
        # Test one step
        obs, info = env.reset()
        action = env.sample_action()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  First step reward: {reward:.4f}")
        env.close()
    
    # print("\n" + "=" * 60)
    
    # Example: Using config without loading it
    # print("=" * 60)
    # print("Example: Environment without Config")
    # print("=" * 60)
    # env_no_config = HighwayEnvRunner('highway', use_config=False)
    # print(f"Config loaded: {env_no_config.get_config() is not None}")
    # env_no_config.close()
