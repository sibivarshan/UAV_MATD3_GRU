import torch
import numpy as np

class Config:
    # Environment
    STATE_DIM = 16  # [position(3), velocity(3), angular_vel(3), goal(3), apf_force(3), distance_to_goal(1)]
    ACTION_DIM = 4  # [thrust_cmd, roll_cmd, pitch_cmd, yaw_cmd]
    MAX_STEPS = 500
    NUM_AGENTS = 3
    WORLD_BOUNDS = [(-10, 10), (-10, 10), (0, 10)]
    
    # APF Parameters - Adjusted for better obstacle avoidance and goal seeking
    ATTRACTION_GAIN = 1.2    # Stronger attraction to goal
    REPULSION_GAIN = 0.8     # Balanced repulsion
    REPULSION_RADIUS = 3.5   # Larger detection radius for obstacles
    
    # Collision Parameters
    UAV_RADIUS = 0.3
    COLLISION_THRESHOLD = 0.3
    PROXIMITY_THRESHOLD = 1.0
    COLLISION_PENALTY = 5.0
    
    # TD3 Parameters
    GAMMA = 0.99
    TAU = 0.005
    POLICY_NOISE = 0.1   # Reduced for more stable actions
    NOISE_CLIP = 0.2     # Reduced for more stable actions
    POLICY_FREQ = 2
    
    # GRU Parameters
    HIDDEN_DIM = 128  # Increased for more complex dynamics
    GRU_LAYERS = 2
    
    # Training
    BATCH_SIZE = 256
    BUFFER_SIZE = int(1e6)
    MAX_EPISODES = 1000
    EVAL_FREQ = 10
    
    # Visualization
    VISUALIZATION_FREQ = 10
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
config = Config()