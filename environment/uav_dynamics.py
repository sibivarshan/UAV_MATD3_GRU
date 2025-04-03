import numpy as np
from scipy.spatial.transform import Rotation
from config import Config

class UAVDynamics:
    def __init__(self):
        self.config = Config
        
        # UAV physical parameters
        self.mass = 1.0          # kg
        self.inertia = np.diag([0.01, 0.01, 0.02])  # kg·m²
        self.inv_inertia = np.linalg.inv(self.inertia)
        self.arm_length = 0.25    # m
        self.max_thrust = 15.0    # N (reduced for gentler movements)
        self.min_thrust = 2.0     # N (minimum thrust to stay airborne)
        self.gravity = 9.81       # m/s²
        
        # State: [position(3), velocity(3), orientation(quat 4), angular_vel(3)]
        self.state = np.zeros(13)
        self.state[6] = 1.0  # Initialize quaternion as identity
        
        # Control inputs: [throttle, roll, pitch, yaw] in [-1, 1]
        self.control = np.zeros(4)
        
        # Control limits and damping
        self.max_torque = 0.8     # Reduced for more stable rotation control
        self.linear_drag = 0.4    # Increased for better damping
        self.angular_drag = 0.15  # Increased for better damping
        self.prev_thrust = self.min_thrust  # For thrust smoothing
    
    def reset(self, position):
        """Reset UAV to initial state"""
        self.state = np.zeros(13)
        self.state[:3] = position
        self.state[6] = 1.0  # Identity quaternion
        self.control = np.zeros(4)
    
    def set_control(self, action):
        """
        Set control inputs from RL action [-1,1]^4
        action: [throttle_cmd, roll_cmd, pitch_cmd, yaw_cmd]
        """
        # Clip actions to [-1, 1] range
        self.control = np.clip(action, -1, 1)
    
    def _compute_motor_forces(self):
        """Convert control inputs to individual motor forces with smoother response"""
        # More balanced thrust computation with smoother response curve
        thrust_cmd = self.control[0]
        # Apply a sigmoid-like function for smoother response near zero
        thrust_factor = (np.tanh(1.5 * thrust_cmd) + 1) * 0.5  # Maps [-1,1] to [0,1] with a smoother curve
        
        total_thrust = (thrust_factor * (self.max_thrust - self.min_thrust) + self.min_thrust) * 1.5
        
        # Apply rate limiting to thrust changes
        max_thrust_change = 2.0  # Maximum change per step
        thrust_change = total_thrust - self.prev_thrust
        if abs(thrust_change) > max_thrust_change:
            total_thrust = self.prev_thrust + np.sign(thrust_change) * max_thrust_change
        
        self.prev_thrust = total_thrust
        
        # Smoother torque application
        roll_cmd = np.clip(self.control[1], -0.8, 0.8)  # Limit extreme commands
        pitch_cmd = np.clip(self.control[2], -0.8, 0.8)
        yaw_cmd = np.clip(self.control[3], -0.8, 0.8)
        
        roll_torque = roll_cmd * self.max_torque
        pitch_torque = pitch_cmd * self.max_torque
        yaw_torque = yaw_cmd * self.max_torque * 0.7  # Reduce yaw authority
        
        # Convert to individual motor forces (quadrotor X configuration)
        forces = np.array([
            total_thrust/4 - pitch_torque/(2*self.arm_length) - yaw_torque/(4*self.arm_length),  # FL
            total_thrust/4 - roll_torque/(2*self.arm_length) + yaw_torque/(4*self.arm_length),   # FR
            total_thrust/4 + pitch_torque/(2*self.arm_length) - yaw_torque/(4*self.arm_length),  # RR
            total_thrust/4 + roll_torque/(2*self.arm_length) + yaw_torque/(4*self.arm_length)    # RL
        ])
        
        # Ensure motors don't go below minimum thrust
        forces = np.maximum(forces, self.min_thrust/4)
        
        return forces
    
    def step(self, dt):
        """Integrate dynamics for time step dt"""
        # Unpack state
        pos = self.state[:3]
        vel = self.state[3:6]
        quat = self.state[6:10]
        ang_vel = self.state[10:13]
        
        # Normalize quaternion
        quat = quat / np.linalg.norm(quat)
        
        # Rotation matrix from quaternion
        rot = Rotation.from_quat(quat).as_matrix()
        
        # Compute motor forces
        motor_forces = self._compute_motor_forces()
        
        # Total thrust in body frame (z-axis)
        thrust_body = np.array([0, 0, np.sum(motor_forces)])
        
        # Thrust force in world frame
        thrust_world = rot @ thrust_body
        
        # Torques in body frame
        torque_body = np.array([
            self.arm_length * (motor_forces[3] - motor_forces[1]),  # Roll (about x-axis)
            self.arm_length * (motor_forces[0] - motor_forces[2]),  # Pitch (about y-axis)
            0.1 * (motor_forces[1] + motor_forces[3] - motor_forces[0] - motor_forces[2])  # Yaw (about z-axis)
        ])
        
        # Forces: thrust + gravity + drag
        forces = thrust_world - np.array([0, 0, self.gravity * self.mass]) - self.linear_drag * vel
        
        # Torques: control + drag
        torques = torque_body - self.angular_drag * ang_vel
        
        # Linear acceleration
        lin_acc = forces / self.mass
        
        # Angular acceleration
        ang_acc = self.inv_inertia @ (torques - np.cross(ang_vel, self.inertia @ ang_vel))
        
        # Integrate position and velocity (Euler integration)
        new_pos = pos + vel * dt
        new_vel = vel + lin_acc * dt
        
        # Integrate orientation using quaternion derivatives
        ang_vel_norm = np.linalg.norm(ang_vel)
        if ang_vel_norm > 1e-6:
            delta_angle = ang_vel * dt
            delta_quat = Rotation.from_rotvec(delta_angle).as_quat()
            new_quat = Rotation.from_quat(quat) * Rotation.from_quat(delta_quat)
            new_quat = new_quat.as_quat()
        else:
            new_quat = quat
        
        # Integrate angular velocity
        new_ang_vel = ang_vel + ang_acc * dt
        
        # Update state
        self.state[:3] = new_pos
        self.state[3:6] = new_vel
        self.state[6:10] = new_quat
        self.state[10:13] = new_ang_vel
        
        return self.state
    
    def get_observation(self):
        """Get observation vector for RL"""
        # Return full state: position, velocity, quaternion, angular velocity
        return self.state.copy()