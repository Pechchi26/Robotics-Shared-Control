"""

human_controller_joystick.py

This script initializes a Go2 robot in high-level control mode and allows
a user to control the robot's movement using a joystick. The joystick inputs
are read in a separate process and sent to the main simulation loop via a
multiprocessing queue.

The robot can be commanded to move forward/backward, strafe left/right,
and rotate. Additionally, specific buttons on the joystick can trigger actions
such as resetting the robot's standing posture or stopping all movement.

No LiDAR. Added clean exit program.
Added: print + CSV logging of vx, vy, yaw.
"""

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append("/home/wolfie/Go2Py")

import time
import multiprocessing as mp
import pygame
from Go2Py import ASSETS_PATH
from Go2Py.sim.mujoco import Go2Sim
import os
import csv
import numpy as np
from utils0 import get_goal_from_scene
#from get_viewer_info import check_and_print_camera #uncomment if LiDAR required
from human_command_visualizer import HumanCommandVisualizer


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



def joystick_input(cmd_queue, stop_event):
    """
    Reads joystick input continuously and sends the latest command to the
    main process via a queue.
    """
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("Pads not found! Please connect a joystick.")
        return

    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print("Pads connected:", joystick.get_name())

    # DEBUG for PS5 controller: print axis info ONCE to check correct assignment
    #print("numaxes:", joystick.get_numaxes())
    #print("axes at rest:",
     #     [round(joystick.get_axis(i), 3)
      #     for i in range(joystick.get_numaxes())])

    try:
        while not stop_event.is_set():
            pygame.event.pump()

            # Read joystick axes
            vx = -joystick.get_axis(1) * 1.5   # forward/backward
            vy = -joystick.get_axis(0) * 1.5    # left/right
            yaw = -joystick.get_axis(3) * 2.0   # rotation - PS5 mapping

            # Read button states
            A = joystick.get_button(0)
            B = joystick.get_button(1)

            # Deadzone filtering to
            # Prevents small joystick noise from moving the robot
            if abs(vx) < 0.12: vx = 0
            if abs(vy) < 0.12: vy = 0
            if abs(yaw) < 0.12: yaw = 0

            # Clear old commands to keep only the latest one
            while not cmd_queue.empty():
                try:
                    cmd_queue.get_nowait()
                except:
                    break

            # Send command dictionary to main process
            cmd_queue.put({
                "vx": vx,
                "vy": vy, 
                "yaw": yaw, 
                "A": A, 
                "B": B
            })

            #Run at ~50 Hz
            time.sleep(0.02)

    finally:
        pygame.quit()
        print("Joystick process exited cleanly.")

# Main robot simulation loop
def run_robot(cmd_queue, stop_event):
    # Path to MuJoCo scene XML
    xml_filename = 'obstacle_scene.xml'
    xml_path = os.path.join(ASSETS_PATH, 'mujoco', xml_filename)
    
    # Initialize Go2 Simulation
    robot = Go2Sim(
            mode='highlevel',
            render=True,
            xml_path=xml_path,
            dt=0.01
        )
    
    robot.standUpReset()
    print("Simulation started...")

    # Set the viewer camera for the virtual interface
    set_viewer_camera(
        robot, azimuth=12, 
        elevation=-39, 
        distance=12.4, 
        lookat=None
    )
    
    #Initiate visualizer
    viz = HumanCommandVisualizer(
        title="HUMAN INPUT (JOYSTICK)",
        vx_max=1.5, vy_max=1.5, yaw_max=2.0
    )
    viz.start()

    # Read goal from MuJoCo XML
    goal_xy = get_goal_from_scene(robot)     # (x, y) goal position
    GOAL_TOL = 0.30                          # meters (tolerance to consider goal reached)

    # CSV logging
    log_path = "go2_velocity_log.csv"
    log_file = open(log_path, "w", newline="")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["sim_time", "vx", "vy", "yaw"])

    # For throttled printing
    last_print_t = 0.0

    try:
        while not stop_event.is_set():
            #check_and_print_camera(robot)
            if not cmd_queue.empty():
                cmd = cmd_queue.get()

                vx, vy, yaw = cmd["vx"], cmd["vy"], cmd["yaw"]
                if cmd["A"]:
                    robot.standUpReset()
                elif cmd["B"]:
                    robot.sitDownReset()

                robot.stepHighlevel(vx, vy, yaw)

                # Update visualizer
                viz.update(vx, vy, yaw)

                # Simulation
                sim_time = robot.data.time 

                # Goal reach check
                robot_xy = robot.data.qpos[:2]
                dist_to_goal = float(np.linalg.norm(robot_xy - goal_xy))

                # Logging
                csv_writer.writerow([sim_time, vx, vy, yaw])

                # Print at ~5 Hz (sim-time based)
                if sim_time - last_print_t >= 0.2:
                    print(
                        f"sim_t={sim_time:.2f} | "
                        f"vx={vx:.3f}, vy={vy:.3f}, yaw={yaw:.3f}"
                        f"dist={dist_to_goal:.2f} m"
                    )
                    last_print_t = sim_time

                # Stop if goal reached
                if dist_to_goal <= GOAL_TOL:
                    print(f"\n✅ Goal reached (dist={dist_to_goal:.2f} m ≤ {GOAL_TOL:.2f} m). Stopping robot...")

                    # Command zero velocity once to ensure robot stops
                    robot.stepHighlevel(0.0, 0.0, 0.0)

                    # Signal all processes to stop
                    stop_event.set()
                    break

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\nCtrl+C received. Stopping simulation...")

    finally:
        stop_event.set()
        log_file.close()

        try:
            viz.close()
        except Exception:
            pass

        print(f"Velocity log saved to {log_path}")

if __name__ == "__main__":
    """
    Program entry point.

    - Starts joystick input as a separate process
    - Runs the robot simulation in the main process
    - Ensures clean shutdown on normal exit or Ctrl+C
    """

    cmd_queue = mp.Queue(maxsize=1)
    stop_event = mp.Event()

    # Start Joystick input process
    p = mp.Process(
        target=joystick_input, 
        args=(cmd_queue, stop_event)
    )
    p.start()

    try:
        run_robot(cmd_queue, stop_event)
    finally:
        stop_event.set()
        p.join()
        print("Program exited cleanly.")





