"""
Artificial Potential Field (APF) Controller
Classic algorithm for navigation and obstacle avoidance
Obstacle source is LiDAR based;F_rep - loops over eaxh LiDAR ray and computes 
obstacles are whatever the robot’s simulated LiDAR “sees” at runtime.

More realistic; to generalize to arbitrary shapes and sensor-like behavior
"""

import numpy as np
from scipy.spatial.transform import Rotation


class APFController:
    """
    - The goal attracts the robot (attractive force)
    - Obstacles repel it (repulsive force)
    - The resulting force defines the movement direction
    """
    
    def __init__(self,
                 k_att=2.0,           # Attraction coefficient
                 k_rep=3.0,           # Repulsion coefficient
                 d_safe=0.9,          # Safe distance to obstacles (m)
                 max_repulsive_dist=3.0,  # Max obstacle influence distance
                 goal_tolerance=0.3,  # Goal reach radius (m)
                 max_linear_vel=1.5,  # Max linear velocity (m/s)
                 max_angular_vel=1.2, # Max angular velocity (rad/s)
                 ):
        
        self.k_att = k_att
        self.k_rep = k_rep
        self.d_safe = d_safe
        self.max_repulsive_dist = max_repulsive_dist
        self.goal_tolerance = goal_tolerance
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        
        # Debug info
        self.last_attractive_force = np.zeros(2)
        self.last_repulsive_force = np.zeros(2)
        self.last_total_force = np.zeros(2)
    
    def compute_attractive_force(self, robot_pos, goal_pos):
        """
        Compute attraction force to the goal
        
        F_att = k_att * (goal - robot_pos)
        """
        goal_vector = goal_pos - robot_pos
        distance_to_goal = np.linalg.norm(goal_vector)
        
        if distance_to_goal < self.goal_tolerance:
            # Already at goal
            return np.zeros(2)
        
        # Normalize and scale
        direction = goal_vector / distance_to_goal
        
        # Parabolic potential for smooth stop near goal
        if distance_to_goal < 2.0:
            magnitude = self.k_att * distance_to_goal
        else:
            magnitude = self.k_att * 2.0  # Constant far from goal
        
        F_attractive = magnitude * direction
        return F_attractive
    
    def compute_repulsive_force(self, robot_pos, robot_yaw, lidar_data):
        """
        Compute repulsive force from obstacles
        
        For each obstacle closer than d_safe:
        F_rep = k_rep * (1/d - 1/d_safe) * (1/d²) * direction_from_obstacle
        """
        if lidar_data is None or len(lidar_data['dist']) == 0:
            return np.zeros(2)
        
        distances = lidar_data['dist']
        n_rays = len(distances)
        
        # Lidar angles in robot frame
        lidar_angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
        
        F_repulsive = np.zeros(2)
        
        for i, dist in enumerate(distances):
            # Ignore far points
            if dist < 0 or dist > self.max_repulsive_dist:
                continue
            
            # Repulsion only from nearby obstacles
            if dist < self.d_safe:
                # Obstacle angle in robot frame
                angle_robot = lidar_angles[i]
                
                # Direction FROM obstacle (in robot frame)
                direction_robot = np.array([
                    -np.cos(angle_robot),
                    -np.sin(angle_robot)
                ])
                
                # Convert to world frame
                direction_world = self._rotate_to_world(direction_robot, robot_yaw)
                
                # Force inversely proportional to distance
                force_magnitude = self.k_rep * (1.0/dist - 1.0/self.d_safe) / (dist**2)
                
                # Limit max force for stability
                force_magnitude = min(force_magnitude, 5.0)
                
                F_repulsive += force_magnitude * direction_world
        
        return F_repulsive
    
    def _rotate_to_world(self, vector_robot, robot_yaw):
        """Rotate vector from robot frame to world frame"""
        cos_yaw = np.cos(robot_yaw)
        sin_yaw = np.sin(robot_yaw)
        
        rotation_matrix = np.array([
            [cos_yaw, -sin_yaw],
            [sin_yaw,  cos_yaw]
        ])
        
        return rotation_matrix @ vector_robot
    
    def _rotate_to_robot(self, vector_world, robot_yaw):
        """Rotate vector from world frame to robot frame"""
        cos_yaw = np.cos(robot_yaw)
        sin_yaw = np.sin(robot_yaw)
        
        rotation_matrix = np.array([
            [ cos_yaw, sin_yaw],
            [-sin_yaw, cos_yaw]
        ])
        
        return rotation_matrix @ vector_world
    
    def force_to_velocity(self, total_force, robot_yaw):
        """
        Convert total force to control commands (vx, vy, omega)
        """
        if np.linalg.norm(total_force) < 1e-6:
            # No movement
            return 0.0, 0.0, 0.0
        
        # Convert force to robot frame
        force_robot = self._rotate_to_robot(total_force, robot_yaw)
        
        # Linear velocities
        vx = np.clip(force_robot[0], -self.max_linear_vel, self.max_linear_vel)
        vy = np.clip(force_robot[1], -self.max_linear_vel * 0.5, self.max_linear_vel * 0.5)
        
        # Angular velocity (turn to force direction)
        desired_yaw = np.arctan2(total_force[1], total_force[0])
        yaw_error = self._normalize_angle(desired_yaw - robot_yaw)
        
        # Simple P-controller
        omega = np.clip(2.0 * yaw_error, -self.max_angular_vel, self.max_angular_vel)
        
        return vx, vy, omega
    
    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def control(self, robot, goal_pos):
        """
        Args:
           - robot: Go2Sim object
           - goal_pos: np.array([x, y]) - target position
        
        Returns:
            (vx, vy, omega): control commands
        """
        pos, quat = robot.getPose()
        robot_pos = pos[:2]
        
        # Get yaw from quaternion
        robot_yaw = self._quaternion_to_yaw(quat)
        
        lidar_data = robot.getLaserScan()
        
        # Compute forces
        F_attractive = self.compute_attractive_force(robot_pos, goal_pos)
        F_repulsive = self.compute_repulsive_force(robot_pos, robot_yaw, lidar_data)
        
        F_total = F_attractive + F_repulsive
        
        # Save for debug
        self.last_attractive_force = F_attractive
        self.last_repulsive_force = F_repulsive
        self.last_total_force = F_total
        
        # Convert to velocity commands
        vx, vy, omega = self.force_to_velocity(F_total, robot_yaw)
        
        return vx, vy, omega
    
    def _quaternion_to_yaw(self, quat):
        """Extract yaw angle from quaternion [w, x, y, z]"""
        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        euler = rot.as_euler('xyz', degrees=False)
        return euler[2]
    
    def is_goal_reached(self, robot, goal_pos):
        """Check if goal is reached"""
        pos, _ = robot.getPose()
        robot_pos = pos[:2]
        distance = np.linalg.norm(robot_pos - goal_pos)
        return distance < self.goal_tolerance
    
    def get_debug_info(self):
        """Return debug info"""
        return {
            'F_attractive': self.last_attractive_force,
            'F_repulsive': self.last_repulsive_force,
            'F_total': self.last_total_force,
            'F_att_magnitude': np.linalg.norm(self.last_attractive_force),
            'F_rep_magnitude': np.linalg.norm(self.last_repulsive_force),
        }
