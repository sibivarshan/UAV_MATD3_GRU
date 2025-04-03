import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, cm
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import time
import pybullet as p
from evaluation import visualize_trajectories
from config import Config
from utils.visualization import create_trajectory_animation

def create_trajectory_animation_from_episode(episode='best', save=True):
    """
    Create an animation showing UAVs moving through environment with obstacles
    
    Args:
        episode: Episode number or 'best' to visualize
        save: Whether to save the animation to file
    
    Returns:
        Animation object
    """
    print(f"Creating animation for episode {episode}...")
    
    # First get trajectories and obstacle data by running a simulation
    trajectories, obstacles = visualize_trajectories(episode, live_rendering=False)
    
    try:
        # Get trajectory data file if it exists
        trajectory_data_file = f'trajectory_data_{episode}.npy'
        if os.path.exists(trajectory_data_file):
            print(f"Loading trajectory data from {trajectory_data_file}")
            trajectory_data = np.load(trajectory_data_file, allow_pickle=True).item()
            if 'obstacles' in trajectory_data:
                obstacles = trajectory_data['obstacles']
    except Exception as e:
        print(f"Error loading trajectory data: {e}")
    
    # Create the animation
    anim, fig = create_trajectory_animation(
        trajectories=trajectories,
        obstacles=obstacles,
        goal_positions=[traj[-1] for traj in trajectories] if trajectories else [],
        episode=episode
    )
    
    # Save animation if requested
    if save:
        print(f"Saving animation to trajectory_animation_{episode}.mp4")
        try:
            writer = animation.FFMpegWriter(fps=15, metadata=dict(artist='APF-MATD3'), bitrate=1800)
            anim.save(f'trajectory_animation_{episode}.mp4', writer=writer)
            print("Animation saved successfully")
        except Exception as e:
            print(f"Error saving animation: {e}")
            print("Trying with PillowWriter instead...")
            writer = animation.PillowWriter(fps=10)
            anim.save(f'trajectory_animation_{episode}.gif', writer=writer)
            print("Animation saved as GIF")
    
    plt.tight_layout()
    if save:
        plt.pause(3)  # Brief pause to view
        plt.close()
    else:
        plt.show(block=False)
        plt.pause(5)  # Show for 5 seconds
        plt.close()
    
    return anim

def main():
    """Main function to handle visualization options"""
    # Handle command line arguments or use interactive input
    if len(sys.argv) > 1:
        episode = sys.argv[1]
        if episode.isdigit():
            episode = int(episode)
    else:
        episode = input("Enter episode number to visualize (or 'best' for best model): ")
        if episode.isdigit():
            episode = int(episode)
    
    # Ask which visualization to use
    print("\nVisualization options:")
    print("1. Static plot with PyBullet live rendering")
    print("2. Animated trajectory plot")
    print("3. Both")
    
    choice = input("Select option (1-3): ")
    
    if choice in ['1', '3']:
        print(f"\nRunning static visualization for episode {episode} with PyBullet...")
        visualize_trajectories(episode, live_rendering=True)
    
    if choice in ['2', '3']:
        print(f"\nCreating animated visualization for episode {episode}...")
        create_trajectory_animation_from_episode(episode=episode, save=True)
        
    print("\nVisualization complete.")

if __name__ == "__main__":
    main()