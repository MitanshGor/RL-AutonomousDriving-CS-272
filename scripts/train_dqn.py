import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
import json
from highway_env.envs.highway_with_obstacles_env import HighwayWithObstaclesEnv
import highway_env 
TRAIN = False  # Set to False to test trained model, True to train first

if __name__ == "__main__":

    env = gym.make('highway-with-obstacles-v0', render_mode='rgb_array')
    
    print("\n" + "="*60)
    print("Environment Configuration:")
    print(f"  Construction zones: {env.unwrapped.config['construction_zones_count']}")
    print(f"  Zone length: {env.unwrapped.config['construction_zone_length']}m")
    print(f"  Lanes: {env.unwrapped.config['lanes_count']} -> {env.unwrapped.config['lanes_count'] - env.unwrapped.config['construction_zone_closed_lanes']} (in construction zone)")
    print(f"  Vehicles: {env.unwrapped.config['vehicles_count']}")
    print(f"  Density: {env.unwrapped.config['vehicles_density']}")
    print(f"  Duration: {env.unwrapped.config['duration']}s")
    print("="*60 + "\n")
    
    # Now reset with the updated config
    obs, info = env.reset()

    # Create the model
    '''model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=5e-4,
        buffer_size=15000,
        learning_starts=2000,
        batch_size=32,
        gamma=0.9,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=50,
        verbose=1,
        tensorboard_log="highway_dqn/",
    )'''

    # A2C Training
    vec_env = make_vec_env('highway-with-obstacles-v0', n_envs=8)
    model = A2C('MlpPolicy', vec_env, verbose=1)
    # Train the model
    if TRAIN:
        model.learn(total_timesteps=int(1500))
        model.save("highway_a2c/model")
    else:
        # Load existing trained model
        model = A2C.load("highway_a2c/model", env=env)
    # # Run the model and record video
    env = RecordVideo(
        env, video_folder="highway_a2c/videos", episode_trigger=lambda e: True
    )

    # # for DQN training
    # model = DQN("MlpPolicy", env)
    # # Train the model
    # if TRAIN:
    #     model.learn(total_timesteps=int(1500))
    #     model.save("highway_dqn/model")
    # else:
    #     # Load existing trained model
    #     model = DQN.load("highway_dqn/model", env=env)

    # # Run the model and record video
    # env = RecordVideo(
    #     env, video_folder="highway_dqn/videos", episode_trigger=lambda e: True
    # )
    env.unwrapped.config["simulation_frequency"] = 15  # Higher FPS for rendering
    env.unwrapped.set_record_video_wrapper(env)

    print("\n" + "="*60)
    print("Starting video recording...")
    print("="*60)
    returns = []
    for videos in range(10):
        done = truncated = False
        obs, info = env.reset()
        
        # Print info for debugging
        print(f"\nEpisode {videos + 1}:")
        print(f"  Vehicles: {len(env.unwrapped.road.vehicles)}")
        if hasattr(env.unwrapped, 'construction_zones'):
            print(f"  Construction zones: {len(env.unwrapped.construction_zones)}")
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