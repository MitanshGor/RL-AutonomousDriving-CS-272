"""Test script for construction zone environment with lane merging."""
import gymnasium as gym
import highway_env

# Register and create the environment
env = gym.make('highway-with-obstacles-v0', render_mode='rgb_array')

# Reset and test
obs, info = env.reset()
print("Environment created successfully!")
print(f"Observation shape: {obs.shape}")
print(f"Construction zones: {env.unwrapped.construction_zones}")

# Take a few steps
for i in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {i+1}: reward={reward:.3f}, terminated={terminated}, truncated={truncated}")
    
    if terminated or truncated:
        break

env.close()
print("Test completed successfully!")
