import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from environment.uav_env import UAVEnv
from models.actor_lstm import LSTMActor
from config import Config
from utils.visualization import plot_trajectory_3d

def visualize_trajectories_lstm(episode, display=False):
    """Visualize drone trajectories using LSTM model and save to file"""
    # Create output directory
    os.makedirs('LSTM_details/visualizations', exist_ok=True)
    
    # Load the trained LSTM models for evaluation
    actors = [LSTMActor().to(Config.DEVICE) for _ in range(Config.NUM_AGENTS)]
    for i in range(Config.NUM_AGENTS):
        try:
            actors[i].load_state_dict(torch.load(f'LSTM_details/models/lstm_actor_{i}_ep_{episode}.pth'))
            actors[i].eval()
        except FileNotFoundError:
            print(f"LSTM Model file not found for agent {i}, episode {episode}. Using untrained model.")
    
    # Create environment without live visualization
    env = UAVEnv(visualize=False)
    state, info = env.reset()
    
    # Store trajectories
    trajectories = [[] for _ in range(Config.NUM_AGENTS)]
    orientations = [[] for _ in range(Config.NUM_AGENTS)]
    for i in range(Config.NUM_AGENTS):
        uav_state = env.uavs[i].get_observation()
        trajectories[i].append(uav_state[:3].copy())
        orientations[i].append(uav_state[6:10].copy())
    
    hidden_states = [None] * Config.NUM_AGENTS
    
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
        
        state = next_state
        
        if all(dones) or all(truncated):
            break
    
    # Create 3D matplotlib plot
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectories and obstacles
    plot_trajectory_3d(ax, trajectories, env.goal_positions, env.obstacles, episode)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(f'LSTM_details/visualizations/lstm_trajectories_episode_{episode}.png', dpi=300)
    
    # Optionally display
    if display:
        plt.show()
    else:
        plt.close()
    
    env.close()
    
    # Save trajectory data
    trajectory_data = {
        'trajectories': trajectories,
        'goal_positions': env.goal_positions,
        'obstacles': env.obstacles
    }
    np.save(f'LSTM_details/trajectories/lstm_trajectory_data_full_{episode}.npy', trajectory_data)
    
    return trajectories, env.obstacles

def generate_lstm_animations(episode='best', save=True):
    """Generate animations for LSTM model trajectories without displaying them"""
    from utils.visualization import create_trajectory_animation
    
    # Try to load trajectory data
    trajectory_data = None
    try:
        trajectory_data = np.load(f'LSTM_details/trajectories/lstm_trajectory_data_full_{episode}.npy', 
                                 allow_pickle=True).item()
        trajectories = trajectory_data['trajectories']
        obstacles = trajectory_data['obstacles']
        goal_positions = trajectory_data['goal_positions']
    except:
        # If data not found, run the simulation to get trajectories
        print(f"No saved trajectory data found for episode {episode}, running simulation...")
        trajectories, obstacles = visualize_trajectories_lstm(episode, display=False)
        # Get goal positions from the last run
        goal_positions = [traj[-1] for traj in trajectories]
    
    # Create the animation
    anim, fig = create_trajectory_animation(
        trajectories=trajectories,
        obstacles=obstacles,
        goal_positions=goal_positions,
        episode=f"LSTM-{episode}"
    )
    
    # Save animation if requested
    if save:
        os.makedirs('LSTM_details/animations', exist_ok=True)
        try:
            from matplotlib import animation
            writer = animation.FFMpegWriter(fps=15, metadata=dict(artist='LSTM-APF-MATD3'), bitrate=1800)
            anim.save(f'LSTM_details/animations/lstm_trajectory_animation_{episode}.mp4', writer=writer)
        except Exception as e:
            print(f"Error saving LSTM animation: {e}")
            print("Trying with PillowWriter instead...")
            writer = animation.PillowWriter(fps=10)
            anim.save(f'LSTM_details/animations/lstm_trajectory_animation_{episode}.gif', writer=writer)
    
    plt.close()
    
    return anim

if __name__ == "__main__":
    # Visualize best model
    visualize_trajectories_lstm('best', display=False)
    generate_lstm_animations('best')