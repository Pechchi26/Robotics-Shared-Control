import sys
sys.path.append("/home/wolfie/Go2Py")
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'

import time
import numpy as np
from Go2Py.sim.mujoco import Go2Sim
from Go2Py import ASSETS_PATH
import sys
import mujoco

sys.path.append('.')
from apf_controller2 import APFController
from force_visualizer import ForceVisualizer  
from ai_apf_visualizer import AIAPFVisualizer

def set_viewer_camera(robot, azimuth=135, elevation=-25, distance=4.0, lookat=None):
    """
    Force MuJoCo viewer camera pose so you don't need to manually adjust every run.
    Call this once after robot has been reset (pose is valid) and viewer exists.

    Args:
        robot: Go2Sim instance
        azimuth: degrees, rotates around vertical axis (left/right orbit)
        elevation: degrees, up/down tilt
        distance: camera distance to lookat point
        lookat: optional [x,y,z]. If None, use robot base position.
    """
    # Go2Sim typically creates a viewer only when render=True
    if not hasattr(robot, "viewer") or robot.viewer is None:
        return

    v = robot.viewer

    # Viewer camera API can vary depending on Go2Py/MuJoCo viewer wrapper.
    # The common case is v.cam with fields: azimuth, elevation, distance, lookat.
    if not hasattr(v, "cam"):
        return

    # If lookat is not provided, center camera on the robot base position (x,y,z)
    if lookat is None:
        # qpos usually begins with base position [x, y, z, ...]
        # Fallback safely if qpos is shorter than expected.
        try:
            base_x = float(robot.data.qpos[0])
            base_y = float(robot.data.qpos[1])
            base_z = float(robot.data.qpos[2]) if len(robot.data.qpos) > 2 else 0.3
            lookat = [base_x, base_y, base_z]
        except Exception:
            lookat = [0.0, 0.0, 0.3]

    try:
        v.cam.azimuth = float(azimuth)
        v.cam.elevation = float(elevation)
        v.cam.distance = float(distance)

        # MuJoCo camera lookat is typically a length-3 array-like
        v.cam.lookat[0] = float(lookat[0])
        v.cam.lookat[1] = float(lookat[1])
        v.cam.lookat[2] = float(lookat[2])
    except Exception:
        # If the viewer wrapper uses a different camera interface, do nothing.
        return


def get_goal_from_scene(robot):
    """
    Get goal position from XML 
    """
    try:
        # Try to find goal_marker
        site_id = mujoco.mj_name2id(robot.model, 
                                     mujoco.mjtObj.mjOBJ_SITE, 
                                     'goal_marker')
        goal_pos = robot.model.site_pos[site_id][:2]  # take x, y only
        print(f"Goal found in XML (goal_marker): [{goal_pos[0]:.2f}, {goal_pos[1]:.2f}]")
        return goal_pos
    except:
        pass
    
    try:
        # Try to find goal
        site_id = mujoco.mj_name2id(robot.model, 
                                     mujoco.mjtObj.mjOBJ_SITE, 
                                     'goal')
        goal_pos = robot.model.site_pos[site_id][:2]
        print(f"Goal found in XML (goal): [{goal_pos[0]:.2f}, {goal_pos[1]:.2f}]")
        return goal_pos
    except:
        pass
    
    # Not found, use default
    print("Goal not found in XML, using default: [5.0, 3.0]")
    return np.array([5.0, 3.0])


def main():
    print("=" * 60)
    print("APF Navigation Test with Force Visualization")
    print("=" * 60)
    print("\nForce arrows legend:")
    print("GREEN  - Attractive force (pulls to goal)")
    print(" RED    - Repulsive force (pushes from obstacles)")
    print("BLUE   - Total/Net force (actual direction)")
    print("=" * 60)
    
    # Path to scene XML
    xml_filename = 'obstacle_scene.xml'
    xml_path = os.path.join(ASSETS_PATH, 'mujoco', xml_filename)
    
    try:
        robot = Go2Sim(
            mode='highlevel',
            render=True,
            xml_path=xml_path,
            dt=0.01
        )
    except Exception as e:
        print(f"\nScene load error: {e}")
        return
    
    # Get goal from XML 
    goal_position = get_goal_from_scene(robot)
    
    # Create APF controller
    controller = APFController(xml_path=xml_path)
    
    # Create force visualizer ← NEW
    visualizer = ForceVisualizer(
        arrow_scale=0.3,      # Adjust arrow length
        arrow_height=0.3      # Height above robot
    )
    
    print("\nAPF controller parameters:")
    print(f"  k_att: {controller.k_att}")
    print(f"  k_rep: {controller.k_rep}")
    print(f"  d_safe: {controller.d_safe}")
    print(f"  goal_tolerance: {controller.goal_tolerance}")
    print(f"  max_linear_vel: {controller.max_linear_vel}")
    print(f"  max_angular_vel: {controller.max_angular_vel}")
    print(f"  max_repulsive_dist: {controller.max_repulsive_dist}")
    
    # Stand up robot
    print("\nStanding up robot...")
    robot.standUpReset()
    time.sleep(6)

    # Set the viewer camera for the virtual interface
    set_viewer_camera(
        robot, azimuth=12, 
        elevation=-39, 
        distance=12.4, 
        lookat=None
    )
    
    # Initiate viz
    ai_viz = AIAPFVisualizer(
        title="AI / APF ",
        vx_max=controller.max_linear_vel,
        vy_max=controller.max_linear_vel * 0.5,  # force_to_velocity clipping 
        yaw_max=controller.max_angular_vel,
        invert_vy_display=False,
        invert_yaw_display=False,
    )
    ai_viz.start()


    # MAIN LOOP 
    
    print("\n" + "=" * 60)
    print("START NAVIGATION")
    print("=" * 60)
    
    start_time = time.time()
    step_count = 0
    max_steps = 10000  # max 100 sec (dt=0.01)
    
    # To record trajectory
    trajectory = []
    
    try:
        while step_count < max_steps:
            # Reset visualization scene ← NEW
            visualizer.reset_scene(robot)
            
            # Compute control commands
            vx, vy, omega = controller.control(robot, goal_position)

            # Visualize forces ← NEW
            debug = controller.get_debug_info()
            visualizer.visualize_forces(
                robot,
                debug['F_attractive'],
                debug['F_repulsive'],
                debug['F_total']
            )
            
            # Take simulation step
            robot.step(vx, vy, omega)
            
            # Update Viz
            ai_viz.update_from(robot, controller, goal_position, vx, vy, omega)

            # Record position
            pos, _ = robot.getPose()
            trajectory.append(pos[:2].copy())
            
            # Check if goal reached
            if controller.is_goal_reached(robot, goal_position):
                elapsed_time = time.time() - start_time
                path_length = calculate_path_length(trajectory)
                
                print("\n" + "=" * 60)
                print("GOAL REACHED!")
                print("=" * 60)
                print(f"Time: {elapsed_time:.2f} sec")
                print(f"Steps: {step_count}")
                print(f"Path length: {path_length:.2f} m")
                print("=" * 60)
                
                # Stand still 2 sec
                for _ in range(200):
                    robot.step(0, 0, 0)
                    time.sleep(0.01)
                
                break
            
            # Print status each 100 steps
            if step_count % 100 == 0:
                pos, _ = robot.getPose()
                robot_pos = pos[:2]
                distance = np.linalg.norm(robot_pos - goal_position)
                
                print(f"Step {step_count:4d} | "
                      f"Position: [{robot_pos[0]:5.2f}, {robot_pos[1]:5.2f}] | "
                      f"To goal: {distance:.2f} m | "
                      f"F_att: {debug['F_att_magnitude']:.2f} | "
                      f"F_rep: {debug['F_rep_magnitude']:.2f}")
            
            step_count += 1
            time.sleep(0.01)
        
        # Max steps reached
        if step_count >= max_steps:
            print("\n" + "=" * 60)
            print("MAX TIME REACHED")
            print("=" * 60)
            pos, _ = robot.getPose()
            robot_pos = pos[:2]
            distance = np.linalg.norm(robot_pos - goal_position)
            print(f"Final distance to goal: {distance:.2f} m")
            print("💡 Try adjusting APF parameters")
    
    except KeyboardInterrupt:
        print("\n\nStopped by user (Ctrl+C)")
    
    finally:
        # Stop robot
        print("\nStopping robot...")
        for _ in range(50):
            robot.step(0, 0, 0)
            time.sleep(0.01)
        
        try:
            ai_viz.close()
        except Exception:
            pass

        # Close simulation
        print("Closing simulation...")
        robot.close()


def calculate_path_length(trajectory):
    """Compute total path length"""
    if len(trajectory) < 2:
        return 0.0
    
    trajectory = np.array(trajectory)
    diffs = np.diff(trajectory, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    return np.sum(distances)


if __name__ == "__main__":
    main()