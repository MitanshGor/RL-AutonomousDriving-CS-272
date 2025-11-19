import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN
import json
from highway_env.envs.highway_with_obstacles_env import HighwayWithObstaclesEnv
import highway_env 
TRAIN = False  # Set to False to test trained model, True to train first

if __name__ == "__main__":

    env = gym.make('highway-with-obstacles-v0', render_mode='rgb_array')
    
    env.unwrapped.config.update({
        "obstacles_count": 4,
        "obstacle_spacing": 20,
        "vehicles_count": 1,  
        "construction_zones_count": 2,  # Number of construction zones
        "construction_zone_length": 150,  # Length of each zone [m]
        "construction_zone_side": "random",  # "left", "right", or "random"
        "construction_zone_lanes": 3,  # Number of lanes the zone takes up
        "construction_cone_spacing": 5,  # Distance between cones [m]
    })
    
    print("\n" + "="*60)
    print("Environment Configuration:")
    print(f"  Construction zones: {env.unwrapped.config['construction_zones_count']}")
    print(f"  Zone length: {env.unwrapped.config['construction_zone_length']}m")
    print(f"  Obstacles: {env.unwrapped.config['obstacles_count']}")
    print(f"  Vehicles: {env.unwrapped.config['vehicles_count']}")
    print("="*60 + "\n")
    
    # Now reset with the updated config
    obs, info = env.reset()

    # Create the model
    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=5e-4,
        buffer_size=15000,
        learning_starts=200,
        batch_size=32,
        gamma=0.8,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=50,
        verbose=1,
        tensorboard_log="highway_dqn/",
    )

    # Train the model
    if TRAIN:
        model.learn(total_timesteps=int(2e4))
        model.save("highway_dqn/model")
    else:
        # Load existing trained model
        model = DQN.load("highway_dqn/model", env=env)

    # Run the model and record video
    env = RecordVideo(
        env, video_folder="highway_dqn/videos", episode_trigger=lambda e: True
    )
    env.unwrapped.config["simulation_frequency"] = 15  # Higher FPS for rendering
    env.unwrapped.set_record_video_wrapper(env)

    print("\n" + "="*60)
    print("Starting video recording...")
    print("="*60)
    
    for videos in range(10):
        done = truncated = False
        obs, info = env.reset()
        
        # Print obstacle count for debugging
        print(f"\nEpisode {videos + 1}:")
        print(f"  Vehicles: {len(env.unwrapped.road.vehicles)}")
        print(f"  Total objects (cones+barriers+obstacles): {len(env.unwrapped.road.objects)}")
        
        while not (done or truncated):
            # Predict
            action, _states = model.predict(obs, deterministic=True)
            # Get reward
            obs, reward, done, truncated, info = env.step(action)
            # Render
            env.render()
    env.close()