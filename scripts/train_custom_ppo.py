import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.env_util import make_vec_env
import json
import highway_env
from PPO_Model.ppo import PPO
import os
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
import pandas as pd
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
TRAIN = True  # Set to False to test trained model, True to train first

if __name__ == "__main__":

    log_dir = './ppo_logs/'
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make('intersection-v0', 
                   render_mode='rgb_array',
                   config = {
                        "observation": {
                            "type": "GrayscaleObservation",
                            "observation_shape": (128, 64),
                            "stack_size": 4,
                            "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
                            "scaling": 1.75,
                        },
                        "policy_frequency": 2
                    })
    env = Monitor(env, log_dir)
    env.reset()

    #vec_env = make_vec_env('merge-v0', n_envs=4)
    model = PPO('CnnPolicy',
                env,
                policy_kwargs=dict(net_arch=[256,256,256]),
                batch_size=256,
                n_epochs=20,
                gamma=0.99,
                ent_coef=0.1,
                learning_rate=1e-4,
                kl_coef=1e-1,
                dual_clip=2,
                verbose=1,
                tensorboard_log=log_dir)
    
    eval_callback = EvalCallback(
            env,
            best_model_save_path='intersection_best_model/best_model/',
            log_path='./ppo_eval/',
            eval_freq=10000
        )

    # Train the model
    if TRAIN:
        model.learn(total_timesteps=int(100_000), callback=eval_callback, progress_bar=True)
        model.save("intersection_ppo/model")

        x, y = ts2xy(load_results(log_dir), 'timesteps')
        y = np.convolve(y, np.ones(1000)/1000, mode='valid')
        x = x[:len(y)]
        plt.plot(x,y)
        plt.xlabel('Timesteps')
        plt.ylabel('Reward')
        plt.title('PPO Intersection Grayscale Env Learning Curve')
        plt.savefig(log_dir+'plots/ID11_intersection_grayscale_learning_curve.png')
        plt.show()

        rewards, lengths = evaluate_policy(model, env, n_eval_episodes=500, return_episode_rewards=True)
        plt.violinplot(rewards, showmeans=True)
        plt.ylabel('Reward')
        plt.title('PPO Intersection Grayscale Env Performance Episodes')
        plt.savefig(log_dir+'plots/ID12_intersection_grayscale_performance_test.png')
        plt.show()

    else:
        # Load existing trained model
        model = PPO.load("highway_ppo/model", env=env)


    # Run the model and record video
    env = RecordVideo(
        env, video_folder="highway_ppo/videos", episode_trigger=lambda e: True
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