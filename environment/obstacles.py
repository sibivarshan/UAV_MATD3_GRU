import numpy as np
import pybullet as p
from typing import List, Dict
import sys
import os

# Fix the relative import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

class ObstacleManager:
    def __init__(self, visualize: bool = False):
        """
        Initialize the obstacle manager
        
        Args:
            visualize: Whether to create visual representations in PyBullet
        """
        self.visualize = visualize
        self.config = Config
        self.obstacles: List[Dict] = []
        self.visual_objects: List[int] = []
        
    def generate_obstacles(self, num_obstacles: int, world_bounds: List[tuple]) -> List[Dict]:
        """
        Generate random obstacles in the environment
        
        Args:
            num_obstacles: Number of obstacles to generate
            world_bounds: List of tuples defining world boundaries [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
            
        Returns:
            List of obstacle dictionaries with position, radius, height
        """
        self.clear_obstacles()
        
        for _ in range(num_obstacles):
            # Generate random position within bounds (with some margin)
            margin = 1.0
            pos = np.array([
                np.random.uniform(world_bounds[0][0] + margin, world_bounds[0][1] - margin),
                np.random.uniform(world_bounds[1][0] + margin, world_bounds[1][1] - margin),
                np.random.uniform(world_bounds[2][0] + margin, world_bounds[2][1] - margin)
            ])
            
            # Random size parameters
            radius = np.random.uniform(0.5, 1.5)
            height = np.random.uniform(1.0, 3.0)
            
            # Create obstacle in PyBullet if visualizing
            visual_obj = None
            if self.visualize:
                col_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
                vis_shape = p.createVisualShape(
                    p.GEOM_CYLINDER, 
                    radius=radius, 
                    length=height, 
                    rgbaColor=[0.7, 0.2, 0.2, 0.7]  # Semi-transparent red
                )
                visual_obj = p.createMultiBody(
                    baseCollisionShapeIndex=col_shape,
                    baseVisualShapeIndex=vis_shape,
                    basePosition=pos
                )
                self.visual_objects.append(visual_obj)
            
            self.obstacles.append({
                'position': pos,
                'radius': radius,
                'height': height,
                'visual': visual_obj
            })
        
        return self.obstacles
    
    def check_collision(self, position: np.ndarray, radius: float = 0.3) -> bool:
        """
        Check if a position collides with any obstacle
        
        Args:
            position: 3D position to check (x,y,z)
            radius: Radius of the object to check collision for
            
        Returns:
            True if collision detected, False otherwise
        """
        for obs in self.obstacles:
            # Check horizontal distance
            horizontal_dist = np.linalg.norm(position[:2] - obs['position'][:2])
            
            # Check vertical overlap
            vertical_overlap = (abs(position[2] - obs['position'][2]) < 
                              (obs['height']/2 + radius))
            
            # Check if within collision radius
            if horizontal_dist < (obs['radius'] + radius) and vertical_overlap:
                return True
                
        return False
    
    def get_obstacle_data(self) -> List[Dict]:
        """
        Get list of obstacle data dictionaries
        
        Returns:
            List of obstacle dictionaries with position, radius, height
        """
        return [{
            'position': obs['position'],
            'radius': obs['radius'],
            'height': obs['height']
        } for obs in self.obstacles]
    
    def clear_obstacles(self):
        """Remove all obstacles from the environment"""
        if self.visualize:
            for obj in self.visual_objects:
                p.removeBody(obj)
            self.visual_objects = []
        self.obstacles = []
    
    def get_obstacle_count(self) -> int:
        """Get current number of obstacles"""
        return len(self.obstacles)

def create_obstacle_wall(start_pos: np.ndarray, end_pos: np.ndarray, 
                         num_obstacles: int, radius: float = 0.8, 
                         height: float = 2.0, visualize: bool = False) -> List[Dict]:
    """
    Create a wall of obstacles between two points
    
    Args:
        start_pos: Starting position of the wall (x,y,z)
        end_pos: Ending position of the wall (x,y,z)
        num_obstacles: Number of obstacles in the wall
        radius: Radius of each obstacle
        height: Height of each obstacle
        visualize: Whether to create visual representations
        
    Returns:
        List of obstacle dictionaries
    """
    obstacles = []
    visual_objects = []
    
    # Calculate direction vector
    direction = end_pos - start_pos
    length = np.linalg.norm(direction)
    direction = direction / length
    
    # Calculate spacing between obstacles
    spacing = length / (num_obstacles - 1) if num_obstacles > 1 else 0
    
    for i in range(num_obstacles):
        pos = start_pos + direction * (i * spacing)
        
        visual_obj = None
        if visualize:
            col_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
            vis_shape = p.createVisualShape(
                p.GEOM_CYLINDER, 
                radius=radius, 
                length=height, 
                rgbaColor=[0.7, 0.2, 0.2, 0.7]
            )
            visual_obj = p.createMultiBody(
                baseCollisionShapeIndex=col_shape,
                baseVisualShapeIndex=vis_shape,
                basePosition=pos
            )
            visual_objects.append(visual_obj)
        
        obstacles.append({
            'position': pos,
            'radius': radius,
            'height': height,
            'visual': visual_obj
        })
    
    return obstacles, visual_objects