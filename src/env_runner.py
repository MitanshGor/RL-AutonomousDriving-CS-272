"""
Environment Wrapper for HighwayEnv
Provides a clean interface for RL algorithms to interact with highway-env environments.

Usage:
    from env_runner import HighwayEnvRunner
    
    # Create environment (no visualization by default)
    env = HighwayEnvRunner('highway')
    
    # Or with visualization
    env = HighwayEnvRunner('highway', render_mode='human')
    
    # Use with RL algorithms
    obs, info = env.reset()
    action = agent.select_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Get environment properties
    obs_space = env.get_observation_space()
    action_space = env.get_action_space()
    
    env.close()

Requirements:
    - gymnasium
    - highway-env
    - numpy
    
    Install with: pip install -r requirements.txt
"""

import gymnasium as gym
import numpy as np
from typing import Optional, Tuple, Dict, Any, Union
import highway_env  # This import registers all highway-env environments with gymnasium
from .config_loader import ConfigLoader


class HighwayEnvRunner:
    """
    Wrapper class for HighwayEnv environments.
    Provides a clean interface for RL algorithms without visualization logic.
    """
    
    # Available environments
    AVAILABLE_ENVS = {
        'highway': 'highway-v0',
        'merge': 'merge-v0',
        'intersection': 'intersection-v0'
    }
    
    # Available observation types
    AVAILABLE_OBSERVATION_TYPES = {
        'lidar': 'LidarObservation',
        'lidarobservation': 'LidarObservation',
        'grayscale': 'GrayscaleObservation',
        'grayscaleobservation': 'GrayscaleObservation',
        'kinematics': 'Kinematics',
        'time_to_collision': 'TimeToCollision',
        'occupancy_grid': 'OccupancyGrid',
    }
    
    def __init__(self, env_type: str, render_mode: Optional[str] = None, 
                 observation_type: Optional[str] = None,
                 config_path: Optional[str] = None, use_config: bool = True, **kwargs):
        """
        Initialize the environment.
        
        Args:
            env_type: Type of environment ('highway', 'merge', 'intersection')
            render_mode: Optional render mode ('human', 'rgb_array', None)
            observation_type: Optional observation type ('lidar', 'grayscale', 'kinematics', etc.)
                            For Task 1, use 'lidar' or 'grayscale'
            config_path: Path to config file. If None, uses default 'env_config.json'
            use_config: Whether to load and use configuration from config file
            **kwargs: Additional arguments to pass to gym.make() or env.configure()
        """
        env_type = env_type.lower().strip()
        
        if env_type not in self.AVAILABLE_ENVS:
            raise ValueError(
                f"'{env_type}' is not a valid environment type. "
                f"Available environments: {list(self.AVAILABLE_ENVS.keys())}"
            )
        
        self.env_key = env_type
        self.env_id = self.AVAILABLE_ENVS[env_type]
        
        # Process observation type
        self.observation_type = None
        if observation_type:
            obs_type_lower = observation_type.lower().strip()
            if obs_type_lower in self.AVAILABLE_OBSERVATION_TYPES:
                self.observation_type = self.AVAILABLE_OBSERVATION_TYPES[obs_type_lower]
            else:
                # Allow direct specification of observation type name
                self.observation_type = observation_type
                print(f"Warning: Using observation type '{observation_type}'. "
                      f"Available types: {list(self.AVAILABLE_OBSERVATION_TYPES.values())}")
        
        # Load configuration
        self.config = None
        if use_config:
            try:
                self.config = ConfigLoader(config_path)
            except FileNotFoundError:
                print(f"Warning: Config file not found. Running without config.")
        
        # Apply config to environment kwargs if available
        env_kwargs = {'render_mode': render_mode} if render_mode else {}
        
        # Apply config-based environment configuration
        if self.config is not None:
            # Apply episode configuration
            episode_config = self.config.get_episode_config()
            if 'max_episode_steps' not in env_kwargs:
                env_kwargs['max_episode_steps'] = episode_config.get('max_episode_steps', 1000)
        
        env_kwargs.update(kwargs)
        
        # Create the environment
        self.env = gym.make(self.env_id, **env_kwargs)
        
        # Configure observation type if specified
        if self.observation_type:
            try:
                self.env.configure({
                    "observation": {
                        "type": self.observation_type
                    }
                })
                # Reset to apply observation type configuration
                self.env.reset()
            except Exception as e:
                print(f"Warning: Failed to configure observation type '{self.observation_type}': {e}")
                print("Continuing with default observation type.")
        
        # Track episode statistics
        self.episode_steps = 0
        self.episode_reward = 0.0
        self.total_steps = 0
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to an initial state.
        
        Args:
            seed: Optional random seed
            options: Optional dictionary of additional options
            
        Returns:
            observation: Initial observation
            info: Dictionary with additional information
        """
        # Reset episode statistics
        self.episode_steps = 0
        self.episode_reward = 0.0
        
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Add config info to info dict if available
        if self.config is not None:
            info['config'] = {
                'action_thresholds': self.config.get_action_thresholds(),
                'episode_config': self.config.get_episode_config(),
                'speed_config': self.config.get_speed_config()
            }
        
        return obs, info
    
    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Run one timestep of the environment's dynamics.
        Applies config-based action clipping and reward modification if config is loaded.
        
        Args:
            action: Action to take in the environment
            
        Returns:
            observation: Observation of the environment
            reward: Reward for the action (modified by config if available)
            terminated: Whether the episode has terminated
            truncated: Whether the episode was truncated
            info: Dictionary with additional information
        """
        # Clip action based on config thresholds if continuous action space
        if self.config is not None and self.is_continuous_action_space():
            action = self._clip_action(action)
        
        # Step the environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update episode statistics
        self.episode_steps += 1
        self.total_steps += 1
        self.episode_reward += reward
        
        # Apply config-based reward modifications
        if self.config is not None:
            reward = self._apply_reward_modifications(reward, info)
        
        # Add config-based info
        if self.config is not None:
            info['episode_steps'] = self.episode_steps
            info['episode_reward'] = self.episode_reward
            info['total_steps'] = self.total_steps
        
        return obs, reward, terminated, truncated, info
    
    def get_observation_space(self) -> gym.Space:
        """
        Get the observation space of the environment.
        
        Returns:
            Observation space
        """
        return self.env.observation_space
    
    def get_action_space(self) -> gym.Space:
        """
        Get the action space of the environment.
        
        Returns:
            Action space
        """
        return self.env.action_space
    
    def get_observation_shape(self) -> Tuple:
        """
        Get the shape of observations.
        
        Returns:
            Shape tuple of observations
        """
        if hasattr(self.env.observation_space, 'shape'):
            return self.env.observation_space.shape
        elif hasattr(self.env.observation_space, 'n'):
            return (self.env.observation_space.n,)
        else:
            return None
    
    def get_action_shape(self) -> Tuple:
        """
        Get the shape of actions.
        
        Returns:
            Shape tuple of actions
        """
        if hasattr(self.env.action_space, 'shape'):
            return self.env.action_space.shape
        elif hasattr(self.env.action_space, 'n'):
            return (self.env.action_space.n,)
        else:
            return None
    
    def is_discrete_action_space(self) -> bool:
        """
        Check if the action space is discrete.
        
        Returns:
            True if action space is discrete, False otherwise
        """
        return isinstance(self.env.action_space, gym.spaces.Discrete)
    
    def is_continuous_action_space(self) -> bool:
        """
        Check if the action space is continuous.
        
        Returns:
            True if action space is continuous, False otherwise
        """
        return isinstance(self.env.action_space, gym.spaces.Box)
    
    def sample_action(self) -> Union[int, np.ndarray]:
        """
        Sample a random action from the action space.
        
        Returns:
            Random action
        """
        return self.env.action_space.sample()
    
    def render(self):
        """
        Render the environment (if render_mode is set).
        """
        return self.env.render()
    
    def close(self):
        """Close the environment and free resources."""
        if self.env is not None:
            self.env.close()
    
    def get_env_info(self) -> Dict[str, Any]:
        """
        Get information about the environment.
        
        Returns:
            Dictionary containing environment information
        """
        return {
            "env_id": self.env_id,
            "env_key": self.env_key,
            "observation_type": self.observation_type,
            "observation_space": str(self.env.observation_space),
            "action_space": str(self.env.action_space),
            "observation_shape": self.get_observation_shape(),
            "action_shape": self.get_action_shape(),
            "is_discrete": self.is_discrete_action_space(),
            "is_continuous": self.is_continuous_action_space(),
        }
    
    def get_observation_type(self) -> Optional[str]:
        """
        Get the current observation type.
        
        Returns:
            Observation type name or None if using default
        """
        return self.observation_type
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        """
        Clip action based on config thresholds.
        
        Args:
            action: Action array [acceleration, steering]
            
        Returns:
            Clipped action array
        """
        if self.config is None:
            return action
        
        thresholds = self.config.get_action_thresholds()
        action = np.array(action, dtype=np.float32)
        
        # Clip acceleration
        if len(action) > 0:
            action[0] = np.clip(
                action[0],
                thresholds['acceleration']['min'],
                thresholds['acceleration']['max']
            )
        
        # Clip steering
        if len(action) > 1:
            action[1] = np.clip(
                action[1],
                thresholds['steering']['min'],
                thresholds['steering']['max']
            )
        
        return action
    
    def _apply_reward_modifications(self, reward: float, info: Dict) -> float:
        """
        Apply config-based reward modifications.
        
        Args:
            reward: Original reward from environment
            info: Info dictionary from environment step
            
        Returns:
            Modified reward
        """
        if self.config is None:
            return reward
        
        reward_config = self.config.get_reward_config()
        modified_reward = reward
        
        # Apply collision penalty if collision detected
        if info.get('crashed', False) or info.get('collision', False):
            if reward_config.get('collision_penalty') is not None:
                modified_reward += reward_config['collision_penalty']
        
        # Apply closed lane penalty if lane violation detected
        if info.get('lane_violation', False) or info.get('closed_lane', False):
            if reward_config.get('closed_lane_penalty') is not None:
                modified_reward += reward_config['closed_lane_penalty']
        
        # Note: Speed compliance rewards are typically handled by the environment
        # but can be added here if needed
        
        return modified_reward
    
    def get_config(self) -> Optional[ConfigLoader]:
        """
        Get the config loader instance.
        
        Returns:
            ConfigLoader instance or None if not loaded
        """
        return self.config
    
    def get_action_thresholds(self) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Get action space thresholds from config.
        
        Returns:
            Dictionary with thresholds or None if config not loaded
        """
        if self.config is None:
            return None
        return self.config.get_action_thresholds()
    
    def get_reward_config(self) -> Optional[Dict[str, Any]]:
        """
        Get reward configuration from config.
        
        Returns:
            Dictionary with reward config or None if config not loaded
        """
        if self.config is None:
            return None
        return self.config.get_reward_config()
    
    def get_safety_rules(self) -> Optional[Dict[str, Any]]:
        """
        Get safety rules from config.
        
        Returns:
            Dictionary with safety rules or None if config not loaded
        """
        if self.config is None:
            return None
        return self.config.get_safety_rules()
    
    def get_episode_config(self) -> Optional[Dict[str, Any]]:
        """
        Get episode configuration from config.
        
        Returns:
            Dictionary with episode config or None if config not loaded
        """
        if self.config is None:
            return None
        return self.config.get_episode_config()
    
    def get_speed_config(self) -> Optional[Dict[str, float]]:
        """
        Get speed configuration from config.
        
        Returns:
            Dictionary with speed config or None if config not loaded
        """
        if self.config is None:
            return None
        return self.config.get_speed_config()
    
    def get_constraints(self) -> Optional[Dict[str, Any]]:
        """
        Get constraints from config.
        
        Returns:
            Dictionary with constraints or None if config not loaded
        """
        if self.config is None:
            return None
        return self.config.get_constraints()
    
    def __repr__(self) -> str:
        config_status = "with config" if self.config is not None else "without config"
        obs_status = f", obs_type='{self.observation_type}'" if self.observation_type else ""
        return f"HighwayEnvRunner(env_type='{self.env_key}', env_id='{self.env_id}'{obs_status}, {config_status})"
