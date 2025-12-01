"""Visualize the construction zone environment with lane merging."""
import gymnasium as gym
import highway_env

# Create environment with rendering
env = gym.make('highway-with-obstacles-v0', render_mode='human')

# Configure for better visualization
env.unwrapped.config.update({
    "construction_zones_count": 2,
    "construction_zone_length": 150,
    "construction_zone_taper_length": 50,
    "construction_zone_closed_lanes": 2,
    "lanes_count": 4,
    "duration": 70,
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "show_trajectories": True,
})

obs, info = env.reset()
print("Construction Zone Environment Loaded!")
print(f"Number of construction zones: {len(env.unwrapped.construction_zones)}")
for i, zone in enumerate(env.unwrapped.construction_zones):
    print(f"\nZone {i+1}:")
    print(f"  Full zone: {zone['start']:.0f}m to {zone['end']:.0f}m")
    print(f"  Lane closures: {zone['closure_start']:.0f}m to {zone['closure_end']:.0f}m")
    print(f"  Construction area: {zone['zone_start']:.0f}m to {zone['zone_end']:.0f}m")
    print(f"  Lane reopenings: {zone['reopening_start']:.0f}m to {zone['reopening_end']:.0f}m")

print("\nControls: Use arrow keys or WASD. Press ESC to exit.")
print("Watch how lanes close before construction zones and reopen after!")

# Run simulation
done = False
while not done:
    action = 1  # IDLE - let the vehicle drive
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    done = terminated or truncated

env.close()
print("\nSimulation ended.")
