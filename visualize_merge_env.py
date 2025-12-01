import gymnasium as gym
import highway_env

# Create the merge environment
env = gym.make('merge-v0', render_mode='rgb_array')

# Configure the environment
env.unwrapped.config.update({
    "observation": {
        "type": "Kinematics"
    },
    "action": {
        "type": "DiscreteMetaAction"
    },
    "duration": 40,  # seconds
    "simulation_frequency": 15,  # Hz
    "policy_frequency": 1,  # Hz
    "screen_width": 1200,
    "screen_height": 300,
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "show_trajectories": True,
    "render_agent": True
})

# Reset environment
obs, info = env.reset()

# Run simulation
done = False
truncated = False
step = 0

print("Starting merge environment visualization...")
print("Close the window to end the simulation.\n")

while not (done or truncated):
    # Render the environment
    env.render()
    
    # Take a random action (or you can use a trained model)
    action = env.action_space.sample()
    
    # Step the environment
    obs, reward, done, truncated, info = env.step(action)
    step += 1
    
    # Print some info every 15 steps (every second)
    if step % 15 == 0:
        print(f"Step {step}: Speed={env.vehicle.speed:.1f} m/s, "
              f"Lane={env.vehicle.lane_index}, Reward={reward:.3f}")

print(f"\nSimulation ended after {step} steps")
print(f"Crashed: {env.vehicle.crashed}")

env.close()
