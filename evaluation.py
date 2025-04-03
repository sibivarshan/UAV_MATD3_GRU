import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
import pybullet as p
import time
from environment.uav_env import UAVEnv
from models.actor_gru import GRUActor
from config import Config
from utils.visualization import plot_trajectory_3d, plot_obstacles_3d

def visualize_trajectories(episode, live_rendering=False):
    """Visualize drone trajectories and obstacles in 3D space"""
    # Load the trained models for evaluation
    actors = [GRUActor().to(Config.DEVICE) for _ in range(Config.NUM_AGENTS)]
    for i in range(Config.NUM_AGENTS):
        try:
            actors[i].load_state_dict(torch.load(f'models/actor_{i}_ep_{episode}.pth'))
            actors[i].eval()
        except FileNotFoundError:
            print(f"Model file not found for agent {i}, episode {episode}. Using untrained model.")
    
    # Create environment with visualization
    print("Creating environment with visualization...")
    env = UAVEnv(visualize=live_rendering)  # Enable GUI rendering when requested
    state, info = env.reset()  # Using Gymnasium API
    
    print(f"Loaded environment with {len(env.obstacles)} obstacles")
    
    # Store trajectories
    trajectories = [[] for _ in range(Config.NUM_AGENTS)]
    orientations = [[] for _ in range(Config.NUM_AGENTS)]
    for i in range(Config.NUM_AGENTS):
        uav_state = env.uavs[i].get_observation()
        trajectories[i].append(uav_state[:3].copy())
        orientations[i].append(uav_state[6:10].copy())
    
    hidden_states = [None] * Config.NUM_AGENTS
    
    if live_rendering:
        print("Running simulation with live PyBullet rendering...")
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # Disable PyBullet GUI panels
        p.resetDebugVisualizerCamera(
            cameraDistance=15,
            cameraYaw=45, 
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0]
        )
    
    # Run the simulation
    for step in range(Config.MAX_STEPS):
        # Select action (no exploration)
        with torch.no_grad():
            actions = []
            for i in range(Config.NUM_AGENTS):
                s = torch.FloatTensor(state[i]).unsqueeze(0).to(Config.DEVICE)
                a, hidden_states[i] = actors[i](s, hidden_states[i])
                actions.append(a.squeeze(0).cpu().numpy())
            actions = np.array(actions)
        
        # Take action
        next_state, rewards, dones, truncated, _ = env.step(actions)
        
        # Store positions and orientations
        for i in range(Config.NUM_AGENTS):
            uav_state = env.uavs[i].get_observation()
            trajectories[i].append(uav_state[:3].copy())
            orientations[i].append(uav_state[6:10].copy())
        
        # Force PyBullet to update visualization
        if live_rendering:
            env._update_visualization()
            time.sleep(0.02)  # Slow down for better visualization
            
        state = next_state
        
        if all(dones) or all(truncated):
            break
    
    # Create 3D matplotlib plot
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use the visualization utilities to plot trajectories and obstacles
    plot_trajectory_3d(ax, trajectories, env.goal_positions, env.obstacles, episode)
    
    # Save and display plot
    plt.tight_layout()
    plt.savefig(f'trajectories_episode_{episode}.png', dpi=300)
    
    if live_rendering:
        plt.show(block=False)  # Don't block execution
        plt.pause(5)  # Show for 5 seconds
        plt.close()
        print(f"Saved trajectory plot to trajectories_episode_{episode}.png")
    else:
        print(f"Saved trajectory plot to trajectories_episode_{episode}.png")
    
    env.close()
    
    return trajectories, env.obstacles

if __name__ == "__main__":
    # Visualize after 10 episodes
    visualize_trajectories(10)