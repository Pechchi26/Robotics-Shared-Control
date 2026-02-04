"""
Force Visualization Module for APF Controller
"""

import numpy as np
import mujoco


class ForceVisualizer:
    """
    Visualizes force vectors in MuJoCo simulation
    """
    
    def __init__(self, arrow_scale=0.3, arrow_height=0.3):
        """
        Args:
            arrow_scale: Scale factor for arrow length
            arrow_height: Height above robot to draw arrows
        """
        self.arrow_scale = arrow_scale
        self.arrow_height = arrow_height
        
        self.color_attractive = np.array([0, 1, 0, 1])      # Green
        self.color_repulsive = np.array([1, 0, 0, 1])       # Red
        self.color_total = np.array([1, 1, 0, 1])         # Yellow
    
    def visualize_forces(self, robot, F_attractive, F_repulsive, F_total):
        """
        Draw force vectors in MuJoCo viewer
        
        Args:
            robot: Go2Sim object
            F_attractive: 2D attractive force vector
            F_repulsive: 2D repulsive force vector
            F_total: 2D total force vector
        """
        if not hasattr(robot, 'viewer') or robot.viewer is None:
            return
        
        pos, _ = robot.getPose()
        robot_pos_3d = pos.copy()
        robot_pos_3d[2] += self.arrow_height  # Lift arrows above robot
        
        # Draw attractive force
        if np.linalg.norm(F_attractive) > 0.01:
            end_att = robot_pos_3d.copy()
            end_att[0] += F_attractive[0] * self.arrow_scale
            end_att[1] += F_attractive[1] * self.arrow_scale
            
            self._draw_arrow(robot, robot_pos_3d, end_att, 
                           self.color_attractive, width=0.02)
        
        # Draw repulsive force
        if np.linalg.norm(F_repulsive) > 0.01:
            end_rep = robot_pos_3d.copy()
            end_rep[0] += F_repulsive[0] * self.arrow_scale
            end_rep[1] += F_repulsive[1] * self.arrow_scale
            
            self._draw_arrow(robot, robot_pos_3d, end_rep, 
                           self.color_repulsive, width=0.02)
        
        # Draw total force
        if np.linalg.norm(F_total) > 0.01:
            end_total = robot_pos_3d.copy()
            end_total[0] += F_total[0] * self.arrow_scale
            end_total[1] += F_total[1] * self.arrow_scale
            
            self._draw_arrow(robot, robot_pos_3d, end_total, 
                           self.color_total, width=0.03)
    
    def _draw_arrow(self, robot, start, end, color, width=0.02):
        """
        Draw an arrow from start to end position using MuJoCo viewer
        
        Args:
            robot: Go2Sim object
            start: 3D start position (numpy array)
            end: 3D end position (numpy array)
            color: RGBA color array
            width: Arrow width
        """
        viewer = robot.viewer
        
        # Initialize geometry
        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3),
            np.zeros(3),
            np.zeros(9),
            color
        )
        
        # Convert to numpy arrays
        start_arr = np.array(start, dtype=np.float64)
        end_arr = np.array(end, dtype=np.float64)
        
        # Set arrow from start to end
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            width,
            start_arr,
            end_arr
        )
        
        viewer.user_scn.ngeom += 1
    
    def reset_scene(self, robot):
        """
        Reset user scene geometries (call before drawing new frame)
        
        Args:
            robot: Go2Sim object
        """
        if hasattr(robot, 'viewer') and robot.viewer is not None:
            robot.viewer.user_scn.ngeom = 0