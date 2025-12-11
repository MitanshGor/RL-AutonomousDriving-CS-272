import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import highway_env
from stable_baselines3 import PPO # Changed from custom PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Video
import os

# Create output directories
os.makedirs("highway_ppo", exist_ok=True)
os.makedirs("highway_ppo/logs", exist_ok=True)
os.makedirs("highway_ppo/checkpoints", exist_ok=True)
os.makedirs("highway_ppo/videos", exist_ok=True)

log_dir = "highway_ppo/logs"

# Create the environment
env = gym.make('highway-with-obstacles-v0', render_mode='rgb_array')

print("\n" + "="*60)
print("Environment Configuration:")
print(f"  Construction zones: {env.unwrapped.config['construction_zones_count']}")
print(f"  Zone length: {env.unwrapped.config['construction_zone_length']}m")
print(f"  Vehicles: {env.unwrapped.config['vehicles_count']}")
print(f"  Density: {env.unwrapped.config['vehicles_density']}")
print(f"  Duration: {env.unwrapped.config['duration']}s")
print("="*60 + "\n")
env = Monitor(env, log_dir)

# Test environment
obs, info = env.reset()
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")
print(f"\n✓ Environment created and configured!")

train = False # Set to True to train, False to load model

# Create the model
model = PPO(
    'MlpPolicy',
    env,
    policy_kwargs=dict(net_arch=[256,256,256]),
    n_steps=2048, 
    batch_size=64,
    n_epochs=20,
    gamma=0.99,
    ent_coef=0.01,
    learning_rate=3e-4,
    verbose=1,
    tensorboard_log=log_dir,
    device='cpu'
)

# Checkpoint callback - saves model periodically
eval_callback = EvalCallback(
    env,
    best_model_save_path='group10_custom_env_results/',
    log_path='./ppo_eval/',
    eval_freq=10000
)

print("✓ Callbacks configured!")

model_path = "group10_custom_env_results/best_model.zip"

if train:
    model.learn(total_timesteps=int(200000), callback=eval_callback)
    model.save("group10_custom_env_results/")
else:
    if os.path.exists(model_path):
        model = PPO.load(model_path, env=env)
        print("✓ Model loaded!")
    else:
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Set train=True to train first."
        )



# Run the model and record video
model = PPO.load(model_path, env=env)

env = RecordVideo(
    env, video_folder="group10_custom_env_results/videos", episode_trigger=lambda e: True
)
env.unwrapped.config["simulation_frequency"] = 15  # Higher FPS for rendering
env.unwrapped.set_record_video_wrapper(env)

print("\n" + "="*60)
print("Starting video recording...")
print("="*60)
returns = []
for videos in range(10):
    done = truncated = False
    obs, info = env.reset()

    # Print obstacle count for debugging
    print(f"\nEpisode {videos + 1}:")
    print(f"  Vehicles: {len(env.unwrapped.road.vehicles)}")
    print(f"  Total objects (cones+barriers+obstacles): {len(env.unwrapped.road.objects)}")
    r = 0
    while not (done or truncated):
        # Predict
        action, _states = model.predict(obs, deterministic=True)
        # Get reward
        obs, reward, done, truncated, info = env.step(action)
        r += reward
        # Render
        env.render()
    returns.append(r)

print(returns)
env.close()