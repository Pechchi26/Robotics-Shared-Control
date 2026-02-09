"""
Artificial Potential Field (APF) Controller
Classic algorithm for navigation and obstacle avoidance
Obstacle source is Map-based from MuJoCo XML;
Repulsion is computed per obstacle rather than per ray
Loads obstacles from the MuJoCo XML file by parsing and storing

When obstacles are known ----> simpler and more stable

When the robot is near obstacles (repulsion is active), 
we add a tangential component that “slides” along obstacles 
instead of getting stuck pushing directly away.

"""



import numpy as np
from scipy.spatial.transform import Rotation
import xml.etree.ElementTree as ET


class APFController:
    """
    - The goal attracts the robot (attractive force)
    - Obstacles repel it (repulsive force)
    - The resulting force defines the movement direction
    """
    
    def __init__(self,
                 xml_path=None,
                 k_att=2,           # Attraction coefficient
                 k_rep=3.0,           # Repulsion coefficient
                 d_safe=1.0,          # Safe distance to obstacles (m)
                 max_repulsive_dist=3.0,  # Max obstacle influence distance
                 goal_tolerance=0.3,  # Goal reach radius (m)
                 max_linear_vel=0.8,  # Max linear velocity (m/s)
                 max_angular_vel=1.0, # Max angular velocity (rad/s)

                 ):
        
        # Load obstacles from XML file
        if xml_path is not None:
            self.obstacles = self._load_obstacles_from_xml(xml_path)
        else:
            self.obstacles = []
        
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

        # Debug info
        self.last_attractive_force = np.zeros(2)
        self.last_repulsive_force = np.zeros(2)
        #self.last_tangential_force = np.zeros(2)  # >>> SOLUTION 1 (NEW)
        self.last_total_force = np.zeros(2)




     # ⭐ Load all <geom> with name="box_*" as obstacles
    def _load_obstacles_from_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        obstacles = []
        for geom in root.iter("geom"):
          name = geom.get("name", "")
          if name.startswith("box_"):
            pos_str = geom.get("pos")
            size_str = geom.get("size")

            x, y, _ = map(float, pos_str.split())
            sx, sy, _ = map(float, size_str.split())

            obstacles.append({
                "pos": np.array([x, y]),
                "half": np.array([sx, sy])
            })

      #  print(f"[APF] Loaded {len(obstacles)} obstacles (with size) from XML")

        return obstacles
    
    
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
    

    
    def get_min_obstacle_distance(self, robot_pos):

        min_dist = float("inf")

        for obs in self.obstacles:
           center = obs["pos"]
           half   = obs["half"]

        # closest point on AABB
           closest_x = np.clip(robot_pos[0], center[0] - half[0], center[0] + half[0])
           closest_y = np.clip(robot_pos[1], center[1] - half[1], center[1] + half[1])
           closest = np.array([closest_x, closest_y])

           direction = robot_pos - closest
           dist = np.linalg.norm(direction)

           if dist < min_dist:
              min_dist = dist

        return min_dist
            
         
    def compute_repulsive_force(self, robot_pos):
        F_rep = np.zeros(2)

        for obs in self.obstacles:
           center = obs["pos"]
           half = obs["half"]

        # point-to-AABB distance
           dx = max(0, abs(robot_pos[0] - center[0]) - half[0])
           dy = max(0, abs(robot_pos[1] - center[1]) - half[1])

           dist = np.sqrt(dx*dx + dy*dy)

           if dist > self.max_repulsive_dist or dist == 0:
              continue

           direction = (robot_pos - center)
           norm = np.linalg.norm(direction)
           if norm > 1e-6:
              direction = direction / norm

           if dist < self.d_safe:
              force = self.k_rep * (1/dist - 1/self.d_safe) / (dist**2)
              force = min(force, 5.0)
              F_rep += force * direction

        return F_rep
            
    
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
        
        
        # Compute forces
        F_attractive = self.compute_attractive_force(robot_pos, goal_pos)
        F_repulsive = self.compute_repulsive_force(robot_pos)
        
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
            #'F_tangential': self.last_tangential_force,  # >>> SOLUTION 1 (NEW)
            
            'F_total': self.last_total_force,
            'F_att_magnitude': np.linalg.norm(self.last_attractive_force),
            'F_rep_magnitude': np.linalg.norm(self.last_repulsive_force),
            #'F_tan_magnitude': np.linalg.norm(self.last_tangential_force),  # >>> SOLUTION 1 (NEW
        }