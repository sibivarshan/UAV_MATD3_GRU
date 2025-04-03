import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import pybullet as p
from config import Config

def plot_trajectory_3d(ax, trajectories, goal_positions, obstacles, episode_num=None):
    """
    Plot UAV trajectories, goal positions and obstacles in 3D
    
    Args:
        ax: Matplotlib 3D axis
        trajectories: List of trajectory arrays for each agent
        goal_positions: Goal positions for each agent
        obstacles: List of obstacle dictionaries
        episode_num: Episode number for title
    """
    # Plot trajectories
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    for i, traj in enumerate(trajectories):
        if len(traj) == 0:
            continue
            
        traj = np.array(traj)
        color = colors[i % len(colors)]
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                color=color, 
                marker='.', 
                markersize=2,
                label=f'UAV {i}')
        
        # Plot start and goal
        if len(traj) > 0:
            ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], 
                      color=color, marker='o', s=100, label=f"Start {i}")
        
        if i < len(goal_positions):
            goal = goal_positions[i]
            ax.scatter(goal[0], goal[1], goal[2], 
                      color=color, marker='*', s=200, label=f"Goal {i}")
    
    # Plot obstacles
    plot_obstacles_3d(ax, obstacles)
    
    # Set labels and title
    ax.set_xlabel('X (m)', fontsize=14)
    ax.set_ylabel('Y (m)', fontsize=14)
    ax.set_zlabel('Altitude (m)', fontsize=14)
    
    if episode_num is not None:
        ax.set_title(f'UAV Trajectories - Episode {episode_num}', fontsize=16)
    else:
        ax.set_title('UAV Trajectories', fontsize=16)
    
    # Set axis limits based on environment bounds
    ax.set_xlim([Config.WORLD_BOUNDS[0][0], Config.WORLD_BOUNDS[0][1]])
    ax.set_ylim([Config.WORLD_BOUNDS[1][0], Config.WORLD_BOUNDS[1][1]])
    ax.set_zlim([Config.WORLD_BOUNDS[2][0], Config.WORLD_BOUNDS[2][1]])
    
    ax.legend(loc='upper right')

def plot_obstacles_3d(ax, obstacles):
    """
    Plot obstacles as cylinders in 3D
    
    Args:
        ax: Matplotlib 3D axis
        obstacles: List of obstacle dictionaries with position, radius, height
    """
    for obs in obstacles:
        pos = obs['position']
        radius = obs['radius']
        height = obs['height']
        
        # Create cylinder with better rendering
        theta = np.linspace(0, 2*np.pi, 30)
        z_points = np.linspace(pos[2] - height/2, pos[2] + height/2, 15)
        theta_grid, z_grid = np.meshgrid(theta, z_points)
        x_grid = pos[0] + radius * np.cos(theta_grid)
        y_grid = pos[1] + radius * np.sin(theta_grid)
        
        # Plot cylinder surface with better appearance
        surf = ax.plot_surface(
            x_grid, y_grid, z_grid, 
            color='darkred', 
            alpha=0.7,
            linewidth=0, 
            antialiased=True
        )
        
        # Add top and bottom circles for clearer cylinder boundaries
        circle_top = np.array([
            pos[0] + radius * np.cos(theta),
            pos[1] + radius * np.sin(theta), 
            np.ones_like(theta) * (pos[2] + height/2)
        ])
        circle_bottom = np.array([
            pos[0] + radius * np.cos(theta), 
            pos[1] + radius * np.sin(theta), 
            np.ones_like(theta) * (pos[2] - height/2)
        ])
        
        ax.plot(circle_top[0], circle_top[1], circle_top[2], color='black', linewidth=2)
        ax.plot(circle_bottom[0], circle_bottom[1], circle_bottom[2], color='black', linewidth=2)

def create_trajectory_animation(trajectories, obstacles, goal_positions, episode='best'):
    """
    Create an animation showing UAVs moving through environment with obstacles
    
    Args:
        trajectories: List of trajectory arrays for each agent
        obstacles: List of obstacle dictionaries
        goal_positions: Goal positions for each agent
        episode: Episode identifier for title
    
    Returns:
        animation.FuncAnimation object
    """
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Initialize plot elements for each agent
    num_agents = len(trajectories)
    lines = []
    points = []
    history_trajectories = [[] for _ in range(num_agents)]
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    
    # Plot obstacles
    plot_obstacles_3d(ax, obstacles)
    
    # Plot goals
    for i, goal in enumerate(goal_positions):
        color = colors[i % len(colors)]
        ax.scatter(goal[0], goal[1], goal[2], color=color, marker='*', s=200)
    
    # Initialize empty lines and points for each agent
    for i in range(num_agents):
        color = colors[i % len(colors)]
        line, = ax.plot([], [], [], color=color, label=f'UAV {i} Path')
        point, = ax.plot([], [], [], color=color, marker='o', markersize=8)
        lines.append(line)
        points.append(point)
    
    # Set axis labels and title
    ax.set_xlabel('X (m)', fontsize=14)
    ax.set_ylabel('Y (m)', fontsize=14)
    ax.set_zlabel('Altitude (m)', fontsize=14)
    ax.set_title(f'UAV Trajectories Animation - Episode {episode}', fontsize=16)
    
    # Set axis limits based on environment bounds
    ax.set_xlim([Config.WORLD_BOUNDS[0][0], Config.WORLD_BOUNDS[0][1]])
    ax.set_ylim([Config.WORLD_BOUNDS[1][0], Config.WORLD_BOUNDS[1][1]])
    ax.set_zlim([Config.WORLD_BOUNDS[2][0], Config.WORLD_BOUNDS[2][1]])
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Add timestamp text
    timestamp_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes, fontsize=12)
    
    def init():
        """Initialize animation"""
        for line, point in zip(lines, points):
            line.set_data([], [])
            line.set_3d_properties([])
            point.set_data([], [])
            point.set_3d_properties([])
        timestamp_text.set_text("")
        return lines + points + [timestamp_text]
    
    def update(frame):
        """Update animation for each frame"""
        # Update timestamp
        timestamp_text.set_text(f"Time step: {frame}")
        
        for i in range(num_agents):
            # Get trajectory for this agent
            if i < len(trajectories) and frame < len(trajectories[i]):
                pos = trajectories[i][frame]
                history_trajectories[i].append(pos)
                
                # Convert trajectory to numpy array for plotting
                if history_trajectories[i]:
                    traj = np.array(history_trajectories[i])
                    lines[i].set_data(traj[:, 0], traj[:, 1])
                    lines[i].set_3d_properties(traj[:, 2])
                    
                    # Update current position
                    points[i].set_data([pos[0]], [pos[1]])
                    points[i].set_3d_properties([pos[2]])
        
        return lines + points + [timestamp_text]
    
    # Calculate number of frames based on longest trajectory
    max_frames = max(len(traj) for traj in trajectories) if trajectories else 0
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, update, frames=max_frames, init_func=init,
        interval=50, blit=True
    )
    
    return anim, fig
