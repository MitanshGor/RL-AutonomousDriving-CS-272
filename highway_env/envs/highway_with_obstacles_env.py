from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.action import Action
from highway_env.envs.highway_env import HighwayEnv
from highway_env.road.lane import LineType, SineLane, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Obstacle


class HighwayWithObstaclesEnv(HighwayEnv):
    """
    A highway driving environment with static obstacles and construction zones on the road.

    The vehicle is driving on a straight highway with several lanes and static obstacles,
    and is rewarded for reaching a high speed, staying on the rightmost lanes and avoiding collisions.
    also should be rewarded for keeping speed limits within construction zones and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "vehicles_count": 40,
                "vehicles_density": 0.8,
                "construction_zones_count": 2,  # Number of construction zones
                "construction_zone_length": 150,  # Length of each zone [m]
                "construction_zone_taper_length": 50,  # Length of lane closures/reopenings [m]
                "construction_zone_closed_lanes": 2,  # Number of lanes closed (4 lanes -> 2 lanes)
                "lanes_count": 4,
                "duration": 75,

                "reward": {
                    "collision_penalty": -1.0,
                    "closed_lane_penalty": -1.0,
                    "progress_reward": {
                    "type": "percentage",
                    "description": "Distance covered as percentage (e.g., 0.73 if 73% covered)"
                    },
                    "speed_compliance": {
                    "within_limit": 0.05,
                    },
                    "speed_violation": {
                    "beyond_limit": -0.05,
                    }
                },

                "speed": {
                    "construction_zone_limit_ms": 20,  # m/s (matches lane speed_limit)
                    "normal_zone_limit_ms": 30,  # m/s
                    "speed_tolerance_ms": 2.5,  # m/s tolerance (±2.5 m/s)
                    "description": "Speed limits in meters per second"
                },
            }
        )
        return config
    
    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def step(self, action):
        """Override step to manage lane change restrictions in construction zones."""
        # Before stepping, disable lane changes for vehicles in construction zones
        for vehicle in self.road.vehicles:
            if hasattr(vehicle, 'enable_lane_change'):
                # Check if vehicle is in construction zone using global X coordinate
                try:
                    vehicle_x = vehicle.position[0]  # Global X coordinate
                    in_zone = self._is_in_construction_zone(vehicle_x)
                    
                    # Disable lane changes in construction zone
                    if in_zone:
                        vehicle.enable_lane_change = False
                    else:
                        # Re-enable lane changes outside construction zone
                        if not hasattr(vehicle, '_original_lane_change_enabled'):
                            vehicle._original_lane_change_enabled = True
                        vehicle.enable_lane_change = vehicle._original_lane_change_enabled
                except:
                    # If we can't determine position, allow lane changes
                    pass
        
        # Call parent step
        return super().step(action)


    def _create_road(self) -> None:
        """
        Create a highway with construction zones that reduce lane count.
        Uses road network topology with lane merges instead of obstacles.
        """
        zones_count = self.config["construction_zones_count"]
        zone_length = self.config["construction_zone_length"]
        taper_length = self.config["construction_zone_taper_length"]
        closed_lanes = self.config["construction_zone_closed_lanes"]
        total_lanes = self.config["lanes_count"]
        
        net = RoadNetwork()
        
        # Line types
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        
        if zones_count == 0:
            # Simple straight road without construction zones
            net = RoadNetwork.straight_road_network(
                self.config["lanes_count"], length=10000, speed_limit=30
            )
        else:
            # Build segmented road with construction zones
            # Segment lengths
            before_zone = 700  # Normal driving before first zone
            between_zones = 600  # Distance between zones if multiple (increased from 400)
            after_zone = 500  # Normal driving after last zone
            
            open_lanes = total_lanes - closed_lanes
            if open_lanes < 1:
                open_lanes = 1
                closed_lanes = total_lanes - 1
            
            lane_width = StraightLane.DEFAULT_WIDTH
            current_x = 0
            
            # Store construction zone info for reward calculation
            self.construction_zones = []
            
            # Segment a: Before first construction zone
            for lane_idx in range(total_lanes):
                y_pos = lane_idx * lane_width
                line_type_left = c if lane_idx == 0 else s
                line_type_right = c if lane_idx == total_lanes - 1 else n
                net.add_lane(
                    "a", "b",
                    StraightLane(
                        [current_x, y_pos],
                        [current_x + before_zone, y_pos],
                        line_types=[line_type_left, line_type_right],
                        speed_limit=30
                    )
                )
            
            current_x += before_zone
            node_from = "b"
            node_to = "c"
            
            # Track lanes that need obstacles at their end
            obstacle_positions = []
            
            for zone_idx in range(zones_count):
                zone_start_x = current_x
                
                # LANE CLOSURES: Lanes simply end, no merging/tapering
                # Close lanes sequentially - lane 4 ends, then lane 3 ends
                
                num_lanes_to_close = total_lanes - open_lanes
                closure_length = 50  # Short distance for each lane to end
                
                # Close lanes one by one
                for closure_step in range(num_lanes_to_close):
                    current_lane_count = total_lanes - closure_step
                    next_lane_count = current_lane_count - 1
                    closing_lane_idx = current_lane_count - 1  # The lane that is closing
                    
                    # Store obstacle position for the closing lane (at the start of this segment)
                    # The closing lane ends at current_x (where this segment begins)
                    y_pos_closing = closing_lane_idx * lane_width
                    obstacle_positions.append([current_x, y_pos_closing])
                    
                    # Continuing lanes go straight
                    for lane_idx in range(next_lane_count):
                        y_pos = lane_idx * lane_width
                        line_type_left = c if lane_idx == 0 else s
                        line_type_right = c if lane_idx == next_lane_count - 1 else n
                        
                        net.add_lane(
                            node_from, node_to,
                            StraightLane(
                                [current_x, y_pos],
                                [current_x + closure_length, y_pos],
                                line_types=[line_type_left, line_type_right],
                                speed_limit=30
                            )
                        )
                    
                    current_x += closure_length
                    node_from = node_to
                    node_to = chr(ord(node_to) + 1)
                
                # CONSTRUCTION ZONE: Only open_lanes available
                for lane_idx in range(open_lanes):
                    y_pos = lane_idx * lane_width
                    line_type_left = c if lane_idx == 0 else s
                    line_type_right = c if lane_idx == open_lanes - 1 else n
                    net.add_lane(
                        node_from, node_to,
                        StraightLane(
                            [current_x, y_pos],
                            [current_x + zone_length, y_pos],
                            line_types=[line_type_left, line_type_right],
                            speed_limit=20  # Reduced speed in construction zone
                        )
                    )
                
                # Store zone boundaries
                self.construction_zones.append({
                    'start': zone_start_x,
                    'end': current_x + zone_length + (num_lanes_to_close * closure_length),
                    'closure_start': zone_start_x,
                    'closure_end': zone_start_x + (num_lanes_to_close * closure_length),
                    'zone_start': current_x,
                    'zone_end': current_x + zone_length,
                    'reopening_start': current_x + zone_length,
                    'reopening_end': current_x + zone_length + (num_lanes_to_close * closure_length)
                })
                
                current_x += zone_length
                node_from = node_to
                node_to = chr(ord(node_to) + 1)
                
                # LANE REOPENINGS: Lanes reappear sequentially
                # Lane 3 opens first, then lane 4
                
                for reopening_step in range(num_lanes_to_close):
                    current_lane_count = open_lanes + reopening_step
                    next_lane_count = current_lane_count + 1
                    
                    # Continuing lanes go straight
                    for lane_idx in range(current_lane_count):
                        y_pos = lane_idx * lane_width
                        line_type_left = c if lane_idx == 0 else s
                        line_type_right = c if lane_idx == current_lane_count - 1 else n
                        
                        net.add_lane(
                            node_from, node_to,
                            StraightLane(
                                [current_x, y_pos],
                                [current_x + closure_length, y_pos],
                                line_types=[line_type_left, line_type_right],
                                speed_limit=30
                            )
                        )
                    
                    # Newly opening lane appears
                    opening_lane_idx = current_lane_count
                    y_pos = opening_lane_idx * lane_width
                    line_type_right = c if opening_lane_idx == total_lanes - 1 else n
                    
                    net.add_lane(
                        node_from, node_to,
                        StraightLane(
                            [current_x, y_pos],
                            [current_x + closure_length, y_pos],
                            line_types=[s, line_type_right],
                            speed_limit=30
                        )
                    )
                    
                    current_x += closure_length
                    node_from = node_to
                    node_to = chr(ord(node_to) + 1)
                
                # If more zones, add straight section between them
                if zone_idx < zones_count - 1:
                    for lane_idx in range(total_lanes):
                        y_pos = lane_idx * lane_width
                        line_type_left = c if lane_idx == 0 else s
                        line_type_right = c if lane_idx == total_lanes - 1 else n
                        net.add_lane(
                            node_from, node_to,
                            StraightLane(
                                [current_x, y_pos],
                                [current_x + between_zones, y_pos],
                                line_types=[line_type_left, line_type_right],
                                speed_limit=30
                            )
                        )
                    current_x += between_zones
                    node_from = node_to
                    node_to = chr(ord(node_to) + 1)
            
            # Final segment: After last construction zone
            for lane_idx in range(total_lanes):
                y_pos = lane_idx * lane_width
                line_type_left = c if lane_idx == 0 else s
                line_type_right = c if lane_idx == total_lanes - 1 else n
                net.add_lane(
                    node_from, chr(ord(node_from) + 1),
                    StraightLane(
                        [current_x, y_pos],
                        [current_x + after_zone, y_pos],
                        line_types=[line_type_left, line_type_right],
                        speed_limit=30
                    )
                )
        
        self.road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        
        # Add obstacles at the end of each closing lane to force merging vehicles to yield
        for obstacle_pos in obstacle_positions:
            self.road.objects.append(Obstacle(self.road, obstacle_pos))

    def _is_in_construction_zone(self, longitudinal_pos: float) -> bool:
        """
        Check if a position is inside a construction zone (including merge/diverge tapers).
        
        :param longitudinal_pos: longitudinal position along road [m]
        :return: True if position is inside a construction zone
        """
        if not hasattr(self, 'construction_zones') or not self.construction_zones:
            return False
        
        for zone in self.construction_zones:
            # Check if in the entire construction zone area (merge + zone + diverge)
            if zone['start'] <= longitudinal_pos <= zone['end']:
                return True
        
        return False

    def _create_vehicles(self) -> None:
        """Create vehicles on the road network, avoiding forbidden lanes."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = utils.near_split(
            self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]
        )

        self.controlled_vehicles = []
        for others in other_per_controlled:
            # Create ego vehicle at a fixed position before first construction zone
            # This ensures the player always starts in a safe location
            if hasattr(self, 'construction_zones') and self.construction_zones:
                # Spawn 400m before the first construction zone
                spawn_position_long = max(50, self.construction_zones[0]['start'] - 400)
            else:
                spawn_position_long = 50  # Default position if no construction zones
            
            # Choose a random non-forbidden lane
            for attempt in range(50):
                # Try lanes in the middle first (safer)
                lane_id = self.config["initial_lane_id"]
                if lane_id is None:
                    lane_id = self.road.np_random.choice(self.config["lanes_count"])
                
                # Get the first road segment (before construction zone)
                lane_index = ("a", "b", lane_id)
                try:
                    lane = self.road.network.get_lane(lane_index)
                    if not lane.forbidden:
                        # Create vehicle at specific position
                        position = lane.position(spawn_position_long, 0)
                        heading = lane.heading_at(spawn_position_long)
                        vehicle = Vehicle(self.road, position, heading, speed=25.0)
                        vehicle.lane_index = lane_index
                        break
                except:
                    continue
            
            vehicle = self.action_type.vehicle_class(
                self.road, vehicle.position, vehicle.heading, vehicle.speed
            )
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            # Create other vehicles, avoiding forbidden lanes and checking for collisions
            created = 0
            attempts = 0
            max_attempts = others * 20  # Allow more attempts per vehicle
            
            while created < others and attempts < max_attempts:
                attempts += 1
                try:
                    vehicle = other_vehicles_type.create_random(
                        self.road, spacing=1 / self.config["vehicles_density"]
                    )
                    
                    # Check if vehicle is on a forbidden lane
                    lane = self.road.network.get_lane(vehicle.lane_index)
                    if lane.forbidden:
                        continue
                    
                    # Check for collision with existing vehicles
                    collision = False
                    for other in self.road.vehicles:
                        if other != vehicle:
                            distance = np.linalg.norm(vehicle.position - other.position)
                            # Require at least 25m spacing (safe margin)
                            if distance < 25:
                                collision = True
                                break
                    
                    if not collision:
                        vehicle.randomize_behavior()
                        # Disable lane changes initially
                        if hasattr(vehicle, 'enable_lane_change'):
                            vehicle.enable_lane_change = False
                            vehicle._original_lane_change_enabled = True
                        self.road.vehicles.append(vehicle)
                        created += 1
                except:
                    # If vehicle creation fails, skip it
                    pass


    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        Additional rewards for compliance with construction zone speed limits.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = sum(v for k, v in rewards.items())
        return reward

    def _rewards(self, action: Action) -> dict[str, float]:
        total_rewards = {}

        # Get vehicle's longitudinal position
        try:
            longitudinal = self.vehicle.lane.local_coordinates(self.vehicle.position)[0]
        except:
            longitudinal = self.vehicle.position[0]

        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)

        if self._is_in_construction_zone(longitudinal):
            construction_min_speed = self.config['speed']['construction_zone_limit_ms'] - self.config['speed']['speed_tolerance_ms']
            construction_max_speed = self.config['speed']['construction_zone_limit_ms'] + self.config['speed']['speed_tolerance_ms']

            if construction_min_speed <= forward_speed <= construction_max_speed:
                total_rewards['speed_compliance'] = 0.25
            else:
                total_rewards['speed_compliance'] = -0.25
        else:
            scaled_speed = utils.lmap(
                forward_speed, self.config["reward_speed_range"], [0, 1]
            )
            total_rewards['high_speed_reward'] = np.clip(scaled_speed, 0, 0.5)
            total_rewards['efficiency'] = 0.025

        if self._is_terminated():
            if self.vehicle.crashed:
                total_rewards['collision_reward'] = -5

        if self._is_truncated():
            total_rewards['end'] = 5

        return total_rewards
            


class HighwayWithObstaclesEnvFast(HighwayWithObstaclesEnv):
    """
    A variant of highway-with-obstacles-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles and obstacles in the scene
        - fewer lanes, shorter episode duration
        - only check collision of controlled vehicles with others
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "simulation_frequency": 5,
                "lanes_count": 3,
                "vehicles_count": 20,
                "obstacles_count": 5,
                "duration": 30,  # [s]
                "ego_spacing": 1.5,
            }
        )
        return cfg

    def _create_vehicles(self) -> None:
        """Create vehicles using parent logic, then disable collision checks."""
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False
