"""
Configuration Loader for ConstructionZoneEnv

Utility class to load and access environment configuration from JSON file.
"""

import json
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigLoader:
    """Load and manage environment configuration from JSON file."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the config loader.
        
        Args:
            config_path: Path to the JSON config file. If None, uses default 'config/env_config.json'
        """
        if config_path is None:
            # Look for config in config/ directory relative to project root
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config" / "env_config.json"
        
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.load_config()
    
    def load_config(self):
        """Load configuration from JSON file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a value from config using dot notation.
        
        Args:
            key_path: Dot-separated path to the config value (e.g., 'action_space.acceleration.min')
            default: Default value if key not found
            
        Returns:
            Config value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_action_thresholds(self) -> Dict[str, Dict[str, float]]:
        """
        Get action space thresholds.
        
        Returns:
            Dictionary with acceleration, steering, and brake thresholds
        """
        return {
            "acceleration": {
                "min": self.get("action_space.acceleration.min", -1.0),
                "max": self.get("action_space.acceleration.max", 1.0)
            },
            "steering": {
                "min": self.get("action_space.steering.min", -1.0),
                "max": self.get("action_space.steering.max", 1.0)
            },
            "brake": {
                "min": self.get("action_space.brake.min", -1.0),
                "max": self.get("action_space.brake.max", 0.0)
            }
        }
    
    def get_reward_config(self) -> Dict[str, Any]:
        """
        Get reward function configuration.
        
        Returns:
            Dictionary with reward parameters
        """
        return {
            "collision_penalty": self.get("reward.collision_penalty", -1.0),
            "closed_lane_penalty": self.get("reward.closed_lane_penalty", -1.0),
            "speed_compliance_reward": self.get("reward.speed_compliance.within_limit", 0.05),
            "speed_violation_penalty": self.get("reward.speed_violation.beyond_limit", -0.05)
        }
    
    def get_safety_rules(self) -> Dict[str, Any]:
        """
        Get safety rules configuration.
        
        Returns:
            Dictionary with safety rule parameters
        """
        return {
            "collision": {
                "enabled": self.get("safety_rules.collision.enabled", True),
                "terminates_episode": self.get("safety_rules.collision.terminates_episode", True),
                "penalty": self.get("safety_rules.collision.penalty", -1.0)
            },
            "closed_lane": {
                "enabled": self.get("safety_rules.closed_lane.enabled", True),
                "terminates_episode": self.get("safety_rules.closed_lane.terminates_episode", True),
                "penalty": self.get("safety_rules.closed_lane.penalty", -1.0)
            },
            "following_distance": {
                "time_to_collision_min": self.get("safety_rules.following_distance.time_to_collision_min", 1.2)
            },
            "speed_limit": {
                "enforced": self.get("safety_rules.speed_limit.enforced", True),
                "tolerance": self.get("safety_rules.speed_limit.tolerance", 5)
            }
        }
    
    def get_episode_config(self) -> Dict[str, Any]:
        """
        Get episode configuration.
        
        Returns:
            Dictionary with episode parameters
        """
        return {
            "min_steps": self.get("episode.duration.min_steps", 300),
            "max_steps": self.get("episode.duration.max_steps", 600),
            "max_episode_steps": self.get("episode.max_steps", 1000),
            "success_threshold_n": self.get("episode.success_threshold.n_size", 100)
        }
    
    def get_speed_config(self) -> Dict[str, float]:
        """
        Get speed configuration.
        
        Returns:
            Dictionary with speed limits and tolerances
        """
        return {
            "construction_zone_limit_mph": self.get("speed.construction_zone_limit_mph", 45),
            "construction_zone_limit_kmh": self.get("speed.construction_zone_limit_kmh", 72.42),
            "speed_tolerance_mph": self.get("speed.speed_tolerance_mph", 5),
            "speed_tolerance_kmh": self.get("speed.speed_tolerance_kmh", 8.05)
        }
    
    def get_randomization_config(self) -> Dict[str, Any]:
        """
        Get randomization parameters.
        
        Returns:
            Dictionary with randomization parameters
        """
        return {
            "lanes": {
                "min": self.get("randomization.lanes.min_count", 2),
                "max": self.get("randomization.lanes.max_count", 5)
            },
            "worker_carts": {
                "min": self.get("randomization.worker_carts.min_count", 0),
                "max": self.get("randomization.worker_carts.max_count", 3)
            },
            "worker_speed": {
                "min_kmh": self.get("randomization.worker_speed.min_kmh", 5),
                "max_kmh": self.get("randomization.worker_speed.max_kmh", 15)
            },
            "traffic_density": self.get("randomization.traffic_density.current", "medium"),
            "barrier_pattern": self.get("randomization.barrier_pattern.current", "solid")
        }
    
    def get_constraints(self) -> Dict[str, Any]:
        """
        Get constraint configuration.
        
        Returns:
            Dictionary with constraint parameters
        """
        return {
            "collision_avoidance": {
                "enabled": self.get("constraints.collision_avoidance.enabled", True),
                "strict": self.get("constraints.collision_avoidance.strict", True)
            },
            "lane_restriction": {
                "enabled": self.get("constraints.lane_restriction.enabled", True),
                "allowed": self.get("constraints.lane_restriction.allowed", "open_valid_lanes_only")
            },
            "speed_restriction": {
                "enabled": self.get("constraints.speed_restriction.enabled", True),
                "tolerance_mph": self.get("constraints.speed_restriction.tolerance_mph", 5)
            },
            "road_boundary": {
                "enabled": self.get("constraints.road_boundary.enabled", True)
            }
        }
    
    def update_config(self, key_path: str, value: Any):
        """
        Update a config value.
        
        Args:
            key_path: Dot-separated path to the config value
            value: New value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set the value
        config[keys[-1]] = value
    
    def save_config(self, output_path: Optional[str] = None):
        """
        Save configuration to JSON file.
        
        Args:
            output_path: Path to save the config. If None, overwrites original file.
        """
        save_path = Path(output_path) if output_path else self.config_path
        
        with open(save_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def __repr__(self) -> str:
        return f"ConfigLoader(config_path='{self.config_path}')"


# Example usage
if __name__ == "__main__":
    # Load config
    config = ConfigLoader()
    
    # Get specific values
    print("Action Thresholds:")
    print(config.get_action_thresholds())
    print("\nReward Config:")
    print(config.get_reward_config())
    print("\nSafety Rules:")
    print(config.get_safety_rules())
    print("\nEpisode Config:")
    print(config.get_episode_config())
    print("\nSpeed Config:")
    print(config.get_speed_config())
    
    # Get a specific value using dot notation
    print(f"\nCollision Penalty: {config.get('reward.collision_penalty')}")
    print(f"Min Acceleration: {config.get('action_space.acceleration.min')}")
