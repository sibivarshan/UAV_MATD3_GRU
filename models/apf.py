import numpy as np
from config import Config

class APF:
    def __init__(self):
        self.config = Config
    
    def compute_force(self, position, goal, obstacles_pos, obstacles_radius):
        """Compute artificial potential field force"""
        # Attractive force (to goal)
        direction_to_goal = goal - position
        distance_to_goal = np.linalg.norm(direction_to_goal)
        
        if distance_to_goal > 0:
            direction_to_goal = direction_to_goal / distance_to_goal
        
        # More balanced attractive force that smoothly decreases with proximity
        attractive_gain = self.config.ATTRACTION_GAIN
        if distance_to_goal < 1.0:
            # Reduce attraction when very close to avoid overshooting
            attractive_gain *= distance_to_goal
        attractive_force = attractive_gain * direction_to_goal
        
        # Repulsive forces (from obstacles)
        repulsive_force = np.zeros(3)
        
        for obs_pos, obs_radius in zip(obstacles_pos, obstacles_radius):
            direction_to_obs = position - obs_pos
            distance_to_obs = np.linalg.norm(direction_to_obs)
            
            # Skip obstacles that are too far away
            if distance_to_obs > self.config.REPULSION_RADIUS:
                continue
                
            if distance_to_obs > 0:
                direction_to_obs = direction_to_obs / distance_to_obs
            
            # Smoother repulsion that increases as agent gets closer to obstacle
            # Use a quadratic function for smoother gradient
            repulsion_factor = max(0, 1.0 - distance_to_obs / self.config.REPULSION_RADIUS)**2
            repulsion_magnitude = self.config.REPULSION_GAIN * repulsion_factor
            
            repulsive_force += repulsion_magnitude * direction_to_obs
        
        # Combine forces with better weighting
        # When close to obstacles, prioritize repulsion
        repulsive_norm = np.linalg.norm(repulsive_force)
        if repulsive_norm > 0 and repulsive_norm > attractive_gain * 0.5:
            # Reduce attractive force when strong repulsion is needed
            attractive_force *= (1.0 - min(1.0, repulsive_norm / (attractive_gain * 2)))
        
        total_force = attractive_force + repulsive_force
        
        # Smooth normalization to avoid sudden direction changes
        force_norm = np.linalg.norm(total_force)
        if force_norm > 1.0:
            total_force = total_force / force_norm  # Maintain direction but cap magnitude
        
        return total_force