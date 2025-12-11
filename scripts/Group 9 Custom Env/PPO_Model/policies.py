# PPO Implementation from SB3:
# https://github.com/DLR-RM/stable-baselines3/tree/master/stable_baselines3/ppo

# This file is here just to define MlpPolicy/CnnPolicy
# that work for PPO
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, MultiInputActorCriticPolicy

MlpPolicy = ActorCriticPolicy
CnnPolicy = ActorCriticCnnPolicy
MultiInputPolicy = MultiInputActorCriticPolicy