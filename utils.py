
import numpy as np
import mujoco
def get_goal_from_scene(robot):
    """
    Retrieve the navigation goal position from the MuJoCo XML scene. 
    """
    try:
        # Try to find goal_marker in the MuJoCo model
        site_id = mujoco.mj_name2id(
            robot.model, 
            mujoco.mjtObj.mjOBJ_SITE, 
            'goal_marker'
        )

        # Extract only the planar (x, y) coordinates of the goal
        goal_pos = robot.model.site_pos[site_id][:2]  

        print(
            f"Goal found in XML (goal_marker): "
            f"[{goal_pos[0]:.2f}, {goal_pos[1]:.2f}]"
        )
        return goal_pos
    
    except:
        pass
    
    try:
        # Try to find goal
        site_id = mujoco.mj_name2id(
            robot.model, 
            mujoco.mjtObj.mjOBJ_SITE, 
            'goal'
        )

        # Extract planar (x, y) coordinates
        goal_pos = robot.model.site_pos[site_id][:2]
        print(
            f"Goal found in XML (goal): "
            f"[{goal_pos[0]:.2f}, {goal_pos[1]:.2f}]"
        )
        
        return goal_pos
    except:
        pass
    
    # Default goal used when no goal site exists in the XML
    print("Goal not found in XML, using default: [5.0, 3.0]")
    return np.array([5.0, 3.0])

def calculate_path_length(trajectory):
    """
    Compute the total path length of a trajectory.

    The path length is calculated as the sum of Euclidean distances
    between consecutive trajectory points.
    """
    
    # If the trajectory has fewer than two points,
    # no distance can be computed
    if len(trajectory) < 2:
        return 0.0
    
    trajectory = np.array(trajectory)
    diffs = np.diff(trajectory, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    return np.sum(distances)