"""
human_controller_keyboard.py

Robust keyboard teleop for Go2Sim high-level control using GLOBAL key capture.

Why global capture?
- pygame key reading only works when its window is focused.
- with MuJoCo viewer open, focus is usually not on pygame, so keys "don't work".
- pynput listens globally, so keys work regardless of focused window.

Controls (hold keys):
- W / S : +vx / -vx
- A / D : +vy / -vy
- Q / E : +yaw / -yaw
- SPACE : zero velocities (quick stop while held)
- R     : standUpReset()
- F     : sitDownReset()
- ESC   : exit

Stops:
- goal reached (distance <= GOAL_TOL)
- Ctrl+C (KeyboardInterrupt)
"""

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("/home/wolfie/Go2Py")

import time
import os
import csv
import numpy as np

from Go2Py import ASSETS_PATH
from Go2Py.sim.mujoco import Go2Sim

from utils0 import get_goal_from_scene
from human_command_visualizer import HumanCommandVisualizer


# Global keyboard listener
from pynput import keyboard


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
    if not hasattr(robot, "viewer") or robot.viewer is None:
        return

    v = robot.viewer
    if not hasattr(v, "cam"):
        return

    if lookat is None:
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
        v.cam.lookat[0] = float(lookat[0])
        v.cam.lookat[1] = float(lookat[1])
        v.cam.lookat[2] = float(lookat[2])
    except Exception:
        return


class KeyState:
    """Tracks which keys are currently held down (global)."""
    def __init__(self):
        self.down = set()
        self.exit_requested = False
        self.do_stand = False
        self.do_sit = False

    def on_press(self, key):
        try:
            # Character keys (w,a,s,d,q,e,r,f)
            if hasattr(key, "char") and key.char is not None:
                c = key.char.lower()
                self.down.add(c)
                if c == "r":
                    self.do_stand = True
                if c == "f":
                    self.do_sit = True
        except Exception:
            pass

        # Special keys
        if key == keyboard.Key.space:
            self.down.add("SPACE")
        if key == keyboard.Key.esc:
            self.exit_requested = True

    def on_release(self, key):
        try:
            if hasattr(key, "char") and key.char is not None:
                c = key.char.lower()
                if c in self.down:
                    self.down.remove(c)
        except Exception:
            pass

        if key == keyboard.Key.space:
            if "SPACE" in self.down:
                self.down.remove("SPACE")


def run_robot_keyboard_global():
    # --------------------------
    # Start simulation
    # --------------------------
    xml_filename = "obstacle_scene.xml"
    xml_path = os.path.join(ASSETS_PATH, "mujoco", xml_filename)

    robot = Go2Sim(
        mode="highlevel",
        render=True,
        xml_path=xml_path,
        dt=0.01
    )

    # Initiate Viz
    viz = HumanCommandVisualizer(
        title="HUMAN INPUT (KEYBOARD)",
        vx_max=1.5, vy_max=1.5, yaw_max=2.0
    )
    viz.start()

    robot.standUpReset()
    print("Simulation started (GLOBAL keyboard control).")

    # Set camera once after reset
    set_viewer_camera(robot, azimuth=12, elevation=-39, distance=12.4, lookat=None)

    # --------------------------
    # Goal reading + stop config
    # --------------------------
    goal_xy = get_goal_from_scene(robot)  # (x, y)
    GOAL_TOL = 0.30  # meters

    # --------------------------
    # Control tuning
    # --------------------------
    VX_SPEED = 1.5   # m/s
    VY_SPEED = 1.5   # m/s
    YAW_SPEED = 2.0  # rad/s

    # --------------------------
    # Logging
    # --------------------------
    log_path = "go2_velocity_log_keyboard.csv"
    log_file = open(log_path, "w", newline="")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["sim_time", "vx", "vy", "yaw", "dist_to_goal"])

    # --------------------------
    # Start global keyboard listener
    # --------------------------
    ks = KeyState()
    listener = keyboard.Listener(on_press=ks.on_press, on_release=ks.on_release)
    listener.start()

    last_print_t = 0.0

    try:
        while True:
            # Exit requested
            if ks.exit_requested:
                print("ESC pressed. Exiting...")
                break

            # One-shot posture actions
            if ks.do_stand:
                robot.standUpReset()
                #set_viewer_camera(robot, azimuth=135, elevation=-25, distance=4.0, lookat=None)
                ks.do_stand = False

            if ks.do_sit:
                robot.sitDownReset()
                #set_viewer_camera(robot, azimuth=135, elevation=-25, distance=4.0, lookat=None)
                ks.do_sit = False

            # --------------------------
            # Build vx, vy, yaw from held keys
            # --------------------------
            vx = 0.0
            vy = 0.0
            yaw = 0.0

            if "w" in ks.down:
                vx += VX_SPEED
            if "s" in ks.down:
                vx -= VX_SPEED

            # Strafe: if left/right is inverted in your frame, swap +/- here
            if "a" in ks.down:
                vy += VY_SPEED
            if "d" in ks.down:
                vy -= VY_SPEED

            if "q" in ks.down:
                yaw += YAW_SPEED
            if "e" in ks.down:
                yaw -= YAW_SPEED

            # Quick stop
            if "SPACE" in ks.down:
                vx = 0.0
                vy = 0.0
                yaw = 0.0

            # Apply command
            robot.stepHighlevel(vx, vy, yaw)

            # Update Viz
            viz.update(vx, vy, yaw)

            # Time + goal distance
            sim_time = robot.data.time
            robot_xy = robot.data.qpos[:2]
            dist_to_goal = float(np.linalg.norm(robot_xy - goal_xy))

            # Log
            csv_writer.writerow([sim_time, vx, vy, yaw, dist_to_goal])

            # Print ~5 Hz
            if sim_time - last_print_t >= 0.2:
                print(
                    f"sim_t={sim_time:.2f} | "
                    f"vx={vx:.3f}, vy={vy:.3f}, yaw={yaw:.3f} | "
                    f"dist={dist_to_goal:.2f} m"
                )
                last_print_t = sim_time

            # Stop if goal reached
            if dist_to_goal <= GOAL_TOL:
                print(f"\n✅ Goal reached (dist={dist_to_goal:.2f} m ≤ {GOAL_TOL:.2f} m). Stopping robot...")
                robot.stepHighlevel(0.0, 0.0, 0.0)
                break

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\nCtrl+C received. Stopping simulation...")

    finally:
        try:
            robot.stepHighlevel(0.0, 0.0, 0.0)
        except Exception:
            pass

        try:
            viz.close()
        except Exception:
            pass

        log_file.close()
        listener.stop()
        print(f"Velocity log saved to {log_path}")
        print("Program exited cleanly.")


if __name__ == "__main__":
    run_robot_keyboard_global()


    run_robot_keyboard_global()

