import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
from typing import Tuple, List, Dict, Optional, Union, Any
from collections import deque
from config import Config
from models.apf import APF
import time
from environment.uav_dynamics import UAVDynamics
from utils.logger import get_logger

class UAVEnv(gym.Env):
    def __init__(self, num_agents=Config.NUM_AGENTS, visualize=False):
        super(UAVEnv, self).__init__()
        self.logger = get_logger()
        self.logger.info(f"Initializing UAV Environment with {num_agents} agents, visualization: {visualize}")
        self.num_agents = num_agents
        self.visualize = visualize
        self.config = Config
        self.apf = APF()
        self.obstacles = []
        self.uavs = [UAVDynamics() for _ in range(num_agents)]
        self.current_step = 0
        self.visual_objects = []
        self.num_obstacles = 0
        self.prev_distances = np.zeros(num_agents)
        
        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(num_agents, Config.ACTION_DIM), dtype=np.float32)
        
        # State shape is (num_agents, seq_len, state_dim)
        self.observation_space = spaces.Box(
            low=float('-inf'), 
            high=float('inf'), 
            shape=(num_agents, 5, Config.STATE_DIM),  # 5 is the history length
            dtype=np.float32
        )
        
        if self.visualize:
            self.logger.info("Connecting to PyBullet GUI")
            self.client = p.connect(p.GUI)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81)
            self.plane = p.loadURDF("plane.urdf")
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # Disable GUI panels
            p.resetDebugVisualizerCamera(cameraDistance=15, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0, 0, 0])
        else:
            self.logger.debug("Connecting to PyBullet DIRECT (headless)")
            self.client = p.connect(p.DIRECT)
        
        self.reset()

    def reset(self, *, seed=None, options=None):
        """Reset the environment and return initial state (Gymnasium API)"""
        super().reset(seed=seed)
        self.logger.debug(f"Resetting UAV environment (step {self.current_step})")
        
        # Clear previous visualization
        for obj in self.visual_objects:
            p.removeBody(obj)
        self.visual_objects = []
        
        if hasattr(self, 'agent_vis_objects'):
            for obj in self.agent_vis_objects:
                p.removeBody(obj)
        self.agent_vis_objects = []
        
        self.obstacles = []
        self.current_step = 0
        
        # Initialize observation history
        self.observation_history = [deque(maxlen=5) for _ in range(self.num_agents)]
        
        # Generate random start positions with increased minimum separation
        self.agent_positions = np.zeros((self.num_agents, 3))
        min_agent_separation = 3.0
        
        for i in range(self.num_agents):
            valid_position = False
            attempts = 0
            
            while not valid_position and attempts < 100:
                pos = np.random.uniform(
                    low=[-8, -8, 1], 
                    high=[8, 8, 5]
                )
                
                valid_position = True
                for j in range(i):
                    if np.linalg.norm(pos - self.agent_positions[j]) < min_agent_separation:
                        valid_position = False
                        break
                
                if valid_position:
                    self.agent_positions[i] = pos
                attempts += 1
            
            if not valid_position:
                pos = np.random.uniform(low=[-10, -10, 1], high=[10, 10, 5])
                self.agent_positions[i] = pos
                self.logger.warning(f"Agent {i} position not optimal after {attempts} attempts")
        
        # Generate goals with sufficient separation and minimum distance from start
        self.goal_positions = np.zeros((self.num_agents, 3))
        min_goal_distance = 6.0  # Minimum distance between start and goal
        
        for i in range(self.num_agents):
            valid_goal = False
            attempts = 0
            
            while not valid_goal and attempts < 100:
                goal = np.random.uniform(low=[-8, -8, 1], high=[8, 8, 5])
                start_dist = np.linalg.norm(goal - self.agent_positions[i])
                
                if start_dist < min_goal_distance:
                    attempts += 1
                    continue
                
                valid_goal = True
                for j in range(i):
                    if np.linalg.norm(goal - self.goal_positions[j]) < min_agent_separation:
                        valid_goal = False
                        break
                
                if valid_goal:
                    self.goal_positions[i] = goal
                attempts += 1
            
            if not valid_goal:
                # Force goal placement in random direction with minimum distance
                direction = np.random.uniform(-1, 1, 3)
                direction = direction / np.linalg.norm(direction)
                self.goal_positions[i] = self.agent_positions[i] + direction * min_goal_distance
                self.logger.warning(f"Goal {i}: Using directed placement after {attempts} failed attempts")
        
        # Reset UAV dynamics and store initial distances
        self.prev_distances = np.zeros(self.num_agents)
        for i in range(self.num_agents):
            self.uavs[i].reset(self.agent_positions[i])
            self.prev_distances[i] = np.linalg.norm(self.agent_positions[i] - self.goal_positions[i])
            self.logger.debug(f"Agent {i} start: {self.agent_positions[i]}, goal: {self.goal_positions[i]}")
        
        # Generate obstacles
        self._generate_obstacles(4)  # Generate exactly 4 obstacles as required
        
        # Get initial state
        state = self._get_state()
        
        # Update visualization
        if self.visualize:
            self._update_visualization()
        
        info = {
            "agent_positions": self.agent_positions.copy(),
            "goal_positions": self.goal_positions.copy(),
            "obstacles": self.obstacles.copy()
        }
        return state, info

    def _generate_obstacles(self, num_obstacles=4):
        """Generate exactly 4 obstacles in the environment"""
        self.logger.info(f"Generating {num_obstacles} obstacles")

        # Clear any existing obstacles
        for i in range(num_obstacles):
            valid_pos = False
            attempts = 0
            
            while not valid_pos and attempts < 50:
                # Generate obstacle position - keeping within a more central area
                obs_pos = np.random.uniform(low=[-7, -7, 0.5], high=[7, 7, 5])
                obs_radius = np.random.uniform(0.6, 1.2)  # Slightly smaller obstacles
                obs_height = np.random.uniform(1, 3)
                
                # Check if obstacle is too close to any agent start or goal
                # Using smaller margin to allow closer approach
                valid_pos = True
                for j in range(self.num_agents):
                    if (np.linalg.norm(obs_pos[:2] - self.agent_positions[j][:2]) < obs_radius + 1.0 or
                        np.linalg.norm(obs_pos[:2] - self.goal_positions[j][:2]) < obs_radius + 1.0):
                        valid_pos = False
                        break
                
                attempts += 1
            
            # Create obstacle visualization
            if self.visualize:
                obstacle_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=obs_radius, height=obs_height)
                obstacle_vis = p.createVisualShape(
                    p.GEOM_CYLINDER,
                    radius=obs_radius,
                    length=obs_height,
                    rgbaColor=[1.0, 0.0, 0.0, 0.8]
                )
                obstacle_body = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=obstacle_col,
                    baseVisualShapeIndex=obstacle_vis,
                    basePosition=obs_pos
                )
                self.visual_objects.append(obstacle_body)
            
            self.obstacles.append({
                'position': obs_pos,
                'radius': obs_radius,
                'height': obs_height
            })
        
        self.num_obstacles = len(self.obstacles)

    def _get_state(self) -> np.ndarray:
        """Get current state for all agents with APF guidance"""
        states = []
        
        # First compute APF forces for all agents
        apf_forces = []
        for i in range(self.num_agents):
            uav_state = self.uavs[i].get_observation()
            position = uav_state[:3]
            
            # Compute obstacles including other agents
            other_agents = [self.uavs[j].get_observation()[:3] for j in range(self.num_agents) if j != i]
            other_radii = [Config.UAV_RADIUS] * (self.num_agents - 1)
            
            obstacle_positions = [o['position'] for o in self.obstacles] + other_agents
            obstacle_radii = [o['radius'] for o in self.obstacles] + other_radii
            
            # Compute APF force
            apf_force = self.apf.compute_force(
                position,
                self.goal_positions[i], 
                obstacle_positions,
                obstacle_radii
            )
            
            apf_forces.append(apf_force)
            
            # Apply gentle APF nudge to help guide the agent
            apf_nudge_factor = 0.15  # Small value for subtle guidance
            self.uavs[i].state[3:6] += apf_force * apf_nudge_factor
        
        # Now build state representations
        for i in range(self.num_agents):
            uav_state = self.uavs[i].get_observation()
            
            # Compute distance to goal
            distance_to_goal = np.linalg.norm(uav_state[:3] - self.goal_positions[i])
            
            # Current state: position (3), velocity (3), angular_vel(3), goal(3), apf_force(3), distance_to_goal(1)
            state = np.concatenate([
                uav_state[:6],    # position and velocity
                uav_state[10:13], # angular velocity
                self.goal_positions[i],
                apf_forces[i],
                [distance_to_goal]
            ])
            
            # Add to history
            self.observation_history[i].append(state)
            
            # Stack last 5 observations for GRU
            while len(self.observation_history[i]) < 5:
                self.observation_history[i].appendleft(np.zeros_like(state))
            
            history_state = np.stack(self.observation_history[i])
            states.append(history_state)
        
        return np.array(states)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict, Dict]:
        # Clip actions to ensure they are within bounds
        actions = np.clip(actions, -1, 1)
        
        # Apply actions to UAV dynamics with smaller step for smoother motion
        dt = 0.04  # Reduced time step for finer control
        for i in range(self.num_agents):
            self.uavs[i].set_control(actions[i])
            self.uavs[i].step(dt)
            
            # Enforce environment boundaries with smoother approach
            position = self.uavs[i].state[:3]
            velocity = self.uavs[i].state[3:6]
            
            # Get world bounds
            for dim in range(3):
                min_bound = Config.WORLD_BOUNDS[dim][0]
                max_bound = Config.WORLD_BOUNDS[dim][1]
                
                # Distance to boundaries
                dist_to_min = position[dim] - min_bound
                dist_to_max = max_bound - position[dim]
                
                # Create boundary forces with smooth gradient
                boundary_margin = 2.0  # Larger margin for earlier correction
                
                # Apply corrective forces near boundaries
                if dist_to_min < boundary_margin:
                    # Force pushing away from min boundary
                    boundary_force = max(0, 1.0 - dist_to_min/boundary_margin) * 3.0
                    self.uavs[i].state[3+dim] += boundary_force * dt
                    # Additional damping when near boundaries to prevent oscillations
                    if velocity[dim] < 0:  # If moving toward boundary
                        self.uavs[i].state[3+dim] *= 0.8  # Apply damping
                        
                elif dist_to_max < boundary_margin:
                    # Force pushing away from max boundary
                    boundary_force = max(0, 1.0 - dist_to_max/boundary_margin) * 3.0
                    self.uavs[i].state[3+dim] -= boundary_force * dt
                    # Additional damping when near boundaries
                    if velocity[dim] > 0:  # If moving toward boundary
                        self.uavs[i].state[3+dim] *= 0.8  # Apply damping
                
                # Hard limit to prevent escaping the environment
                self.uavs[i].state[dim] = np.clip(
                    self.uavs[i].state[dim], 
                    min_bound + 0.01, 
                    max_bound - 0.01
                )
        
        self.current_step += 1
        
        # Get new state
        new_state = self._get_state()
        
        # Calculate rewards
        rewards = np.zeros(self.num_agents)
        dones = np.zeros(self.num_agents, dtype=bool)
        truncated = np.zeros(self.num_agents, dtype=bool)
        infos = [{} for _ in range(self.num_agents)]
        
        # Set truncated if max steps reached
        if self.current_step >= Config.MAX_STEPS:
            truncated = np.ones(self.num_agents, dtype=bool)
        
        for i in range(self.num_agents):
            uav_state = self.uavs[i].get_observation()
            position = uav_state[:3]
            velocity = uav_state[3:6]
            
            # Calculate current distance to goal
            distance_to_goal = np.linalg.norm(position - self.goal_positions[i])
            
            # Improved reward function
            
            # 1. Base reward component - small negative to encourage completion
            rewards[i] = -0.1
            
            # 2. Progress reward - stronger positive reinforcement for moving toward goal
            progress = self.prev_distances[i] - distance_to_goal
            if progress > 0:  # Moving toward goal
                rewards[i] += progress * 4.0  # Increased reward for moving toward goal
            else:  # Moving away from goal
                rewards[i] += progress * 1.5  # Increased penalty for moving away

            # Add distance-based reward component
            # Reward gets higher as distance decreases (max 2.0 when very close)
            rewards[i] += 2.0 / (1.0 + distance_to_goal)

            # Penalize being near environment boundaries
            for dim in range(3):
                min_bound = Config.WORLD_BOUNDS[dim][0]
                max_bound = Config.WORLD_BOUNDS[dim][1]
                margin = 1.5
                
                if position[dim] < min_bound + margin or position[dim] > max_bound - margin:
                    # Calculate how close to boundary (0 = at boundary, 1 = at margin)
                    boundary_factor = min(
                        (position[dim] - min_bound) / margin,
                        (max_bound - position[dim]) / margin
                    )
                    rewards[i] -= (1.0 - boundary_factor) * 1.0  # Penalty increases closer to boundary
            
            # 3. Speed efficiency reward - reward efficient movement
            optimal_speed = min(2.0, distance_to_goal)  # Optimal speed depends on distance
            current_speed = np.linalg.norm(velocity)
            speed_efficiency = 1.0 - min(1.0, abs(current_speed - optimal_speed) / optimal_speed)
            rewards[i] += speed_efficiency * 0.2
            
            # 4. Direction alignment reward - incentivize moving toward goal
            if distance_to_goal > 0.1 and current_speed > 0.1:
                dir_to_goal = (self.goal_positions[i] - position) / distance_to_goal
                vel_dir = velocity / current_speed
                alignment = np.dot(dir_to_goal, vel_dir)  # -1 to 1
                rewards[i] += max(0, alignment) * 0.3  # Only reward positive alignment
            
            # 5. Goal completion reward - major reward for reaching goal
            if distance_to_goal < 0.5:
                rewards[i] += 10.0  # Big reward for success
                dones[i] = True
                infos[i]['goal_reached'] = True
                infos[i]['success'] = True
            
            # 6. Obstacle avoidance reward
            for obs in self.obstacles:
                horizontal_dist = np.linalg.norm(position[:2] - obs['position'][:2])
                vertical_dist = abs(position[2] - obs['position'][2])
                
                # Proximity penalty with gradient
                if horizontal_dist < obs['radius'] * 2.0 and vertical_dist < obs['height']:
                    proximity = horizontal_dist / (obs['radius'] * 2.0)  # 0 to 1
                    rewards[i] -= (1.0 - proximity) * 1.5  # Stronger gradient
                    
                    # Collision penalty
                    if horizontal_dist < obs['radius'] and vertical_dist < obs['height']/2:
                        rewards[i] -= 5.0
                        dones[i] = True
                        infos[i]['collision'] = True
                        infos[i]['success'] = False
            
            # 7. Agent collision avoidance
            for j in range(self.num_agents):
                if i == j:
                    continue
                    
                uav_state_j = self.uavs[j].get_observation()
                position_j = uav_state_j[:3]
                
                distance = np.linalg.norm(position - position_j)
                
                # Proximity penalty
                if distance < 1.0:
                    proximity_factor = (1.0 - distance)
                    rewards[i] -= proximity_factor * 1.5
                    
                    # Collision penalty
                    if distance < 0.3:
                        rewards[i] -= 5.0
                        dones[i] = True
                        infos[i]['agent_collision'] = True
                        infos[i]['success'] = False
            
            # Store current distance for progress calculation in next step
            self.prev_distances[i] = distance_to_goal
        
        # Update visualization
        if self.visualize:
            self._update_visualization()
            time.sleep(0.02)
        
        # Combine per-agent info into a single dictionary
        combined_info = {f"agent_{i}": info for i, info in enumerate(infos)}
        return new_state, rewards, dones, truncated, combined_info

    def _update_visualization(self):
        """Update visualization elements"""
        if not self.visualize:
            return
        
        # Clear only agent and goal markers, not obstacle markers
        if hasattr(self, 'agent_vis_objects'):
            for obj in self.agent_vis_objects:
                p.removeBody(obj)
            self.agent_vis_objects = []
        else:
            self.agent_vis_objects = []
        
        # Update agent visualizations
        for i in range(self.num_agents):
            # Update UAV position directly from dynamics
            uav_state = self.uavs[i].get_observation()
            position = uav_state[:3]
            
            # Create UAV visualization
            agent = p.createCollisionShape(p.GEOM_SPHERE, radius=0.3)
            agent_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.3, rgbaColor=[0.2, 0.7, 0.2, 1])
            agent_body = p.createMultiBody(
                baseCollisionShapeIndex=agent, 
                baseVisualShapeIndex=agent_vis, 
                basePosition=position
            )
            self.agent_vis_objects.append(agent_body)
            
            # Add goal marker
            goal = p.createVisualShape(p.GEOM_SPHERE, radius=0.2, rgbaColor=[0.2, 0.2, 0.7, 0.7])
            goal_body = p.createMultiBody(
                baseVisualShapeIndex=goal, 
                basePosition=self.goal_positions[i]
            )
            self.agent_vis_objects.append(goal_body)
        
        # Debug text for obstacles
        for i, obs in enumerate(self.obstacles):
            p.addUserDebugText(
                f"Obstacle {i}", 
                [obs['position'][0], obs['position'][1], obs['position'][2] + obs['height']/2 + 0.5],
                [1, 0, 0],
                1.0
            )
    
    def close(self):
        p.disconnect()

    def get_visualization_data(self):
        """Return data needed for external visualization"""
        return {
            'agent_positions': [uav.get_observation()[:3] for uav in self.uavs],
            'goal_positions': self.goal_positions,
            'obstacles': self.obstacles,
            'num_obstacles': len(self.obstacles)
        }