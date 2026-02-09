#!/usr/bin/env python3
"""
arbitration_switch.py

Outputs TWO clear plots (paper/poster style):
1) Mode over time (step graph) -> y: HUMAN / AI
2) Speed over time -> Human input speed vs AI(APF) input speed vs Final output speed

Based on your uploaded script :contentReference[oaicite:0]{index=0}
"""

import sys
sys.path.append("/home/wolfie/Go2Py")

import os
import time
import numpy as np
import multiprocessing as mp
from collections import deque

import matplotlib.pyplot as plt

from Go2Py.sim.mujoco import Go2Sim
from Go2Py import ASSETS_PATH

from utils import get_goal_from_scene  
from human_controller_for_arbitration import joystick_process  
from apf_controller2 import APFController  
from arbitration_switch_visualizer import SwitchArbitrationVisualizer 


def calculate_path_length(trajectory_xy):
    """Compute polyline length for a list of [x,y] points."""
    if trajectory_xy is None or len(trajectory_xy) < 2:
        return 0.0
    traj = np.asarray(trajectory_xy, dtype=float)
    diffs = traj[1:] - traj[:-1]
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def set_viewer_camera(robot, azimuth=135, elevation=-25, distance=4.0, lookat=None):
    """
    Force MuJoCo viewer camera pose so you don't need to manually adjust every run.
    Call this once after robot has been reset (pose is valid) and viewer exists.
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


# -------------------------
# Switch (Human vs AI)
# -------------------------
def arbitration_switch(vx_h, vy_h, w_h, A, B, vx_ai, vy_ai, w_ai, min_dist):
    """
    2-mode switching:
      - if obstacle too close -> AI
      - else if human active -> HUMAN
      - else -> AI
    """
    human_active = (
        abs(vx_h) > 0.01 or abs(vy_h) > 0.01 or abs(w_h) > 0.01 or A == 1 or B == 1
    )

    if min_dist < 0.65:
        return vx_ai, vy_ai, w_ai, "AI"

    if human_active:
        return vx_h, vy_h, w_h, "HUMAN"

    return vx_ai, vy_ai, w_ai, "AI"


# -------------------------
# Time + tolerance stuck detector
# -------------------------
class TimeStuckDetector:
    """
    Declares STUCK if conditions persist for >= stuck_time_s.

    Conditions (all must hold at once):
      1) drift over window < drift_eps
      2) commanded speed is low (final cmd small)
      3) robot is NOT at goal (dist_goal > goal_tol)
      4) (near obstacle OR repulsive force high)
    """
    def __init__(
        self,
        window_sec=1.0,
        stuck_time_s=1.0,
        drift_eps=0.05,
        cmd_speed_eps=0.05,
        yaw_eps=0.10,
        near_obs_thresh=0.75,
        rep_force_thresh=1.0,
        goal_tol=0.35,
    ):
        self.window_sec = float(window_sec)
        self.stuck_time_s = float(stuck_time_s)
        self.drift_eps = float(drift_eps)
        self.cmd_speed_eps = float(cmd_speed_eps)
        self.yaw_eps = float(yaw_eps)
        self.near_obs_thresh = float(near_obs_thresh)
        self.rep_force_thresh = float(rep_force_thresh)
        self.goal_tol = float(goal_tol)

        self.samples = deque()  # (t, x, y)
        self._stuck_timer = 0.0
        self._last_t = None

    def update(self, t_now, robot_xy, final_cmd, dist_goal, min_obs, F_rep_mag):
        self.samples.append((t_now, float(robot_xy[0]), float(robot_xy[1])))
        while self.samples and (t_now - self.samples[0][0] > self.window_sec):
            self.samples.popleft()

        if self._last_t is None:
            self._last_t = t_now
            return False, 0.0, 0.0

        dt = max(0.0, t_now - self._last_t)
        self._last_t = t_now

        if len(self.samples) >= 2:
            x0, y0 = self.samples[0][1], self.samples[0][2]
            x1, y1 = self.samples[-1][1], self.samples[-1][2]
            drift = float(np.hypot(x1 - x0, y1 - y0))
            window_progress = float(min(1.0, (t_now - self.samples[0][0]) / self.window_sec))
        else:
            drift = 999.0
            window_progress = 0.0

        vx_f, vy_f, w_f = final_cmd
        cmd_small = (
            abs(vx_f) < self.cmd_speed_eps
            and abs(vy_f) < self.cmd_speed_eps
            and abs(w_f) < self.yaw_eps
        )
        not_at_goal = (dist_goal > self.goal_tol)
        near_or_rep = (min_obs < self.near_obs_thresh) or (F_rep_mag > self.rep_force_thresh)
        low_drift = (drift < self.drift_eps)

        conditions = low_drift and cmd_small and not_at_goal and near_or_rep

        if conditions:
            self._stuck_timer += dt
        else:
            self._stuck_timer = max(0.0, self._stuck_timer - 2.0 * dt)

        is_stuck = self._stuck_timer >= self.stuck_time_s
        return is_stuck, window_progress, drift


def plot_results(log_t, log_mode, log_v_h, log_v_ai, log_v_f, save_dir=None):
    """
    Make a single figure with 2 subplots:

    1) Mode over time (step graph): HUMAN/AI only
    2) Speed over time: AI input vs Human input vs Final output

    mode encoding: 0=HUMAN, 1=AI (STUCK already mapped to AI in logging)
    """
    t = np.asarray(log_t, dtype=float)
    mode = np.asarray(log_mode, dtype=int)

    v_human = np.asarray(log_v_h, dtype=float)
    v_ai = np.asarray(log_v_ai, dtype=float)
    v_final = np.asarray(log_v_f, dtype=float)

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(10, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 2]}
    )

    # ================================================================
    # Plot 1: Mode over time (STEP GRAPH)
    # ================================================================
    ax1.set_title("1. Control Mode Over Time", fontsize=12, fontweight="bold")
    ax1.step(t, mode, where="post", linewidth=1, color="red")
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["HUMAN", "AI"])
    ax1.set_ylabel("Mode")
    ax1.grid(True, alpha=0.3)
    
    # ================================================================
    # Plot 2: Velocity Arbitration (Input vs. Output)
    # ================================================================
    ax2.set_title(
        "2. Velocity Arbitration (Input vs. Output)",
        fontsize=12,
        fontweight="bold"
    )

    # Always show BOTH inputs (dotted)
    ax2.plot(
        t,
        v_ai,
        color="yellow",
        #linestyle=":",
        linewidth=1.5,
        label="AI"
    )

    ax2.plot(
        t,
        v_human,
        color="blue",
        #linestyle=":",
        linewidth=1.5,
        label="Human"
    )

    # Final robot velocity (solid, emphasized)
    ax2.plot(
        t,
        v_final,
        color="green",
        linewidth=2.5,
        label="Robot"
    )

    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Speed (m/s)")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)


    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out_png = os.path.join(save_dir, "mode_and_speed.png")
        plt.savefig(out_png, dpi=300)
        print(f"[plots] saved: {out_png}")

    plt.show()


def main():
    xml_filename = "u_shape_scene.xml"
    xml_path = os.path.join(ASSETS_PATH, "mujoco", xml_filename)

    # Joystick process
    human_queue = mp.Queue(maxsize=1)
    p = mp.Process(target=joystick_process, args=(human_queue,))
    p.start()

    # IMPORTANT: load obstacles into APFController
    apf_ai = APFController(xml_path=xml_path)

    robot = Go2Sim(mode="highlevel", render=True, xml_path=xml_path, dt=0.01)
    goal = get_goal_from_scene(robot)
    robot.standUpReset()

    set_viewer_camera(
        robot, azimuth=12,
        elevation=-39,
        distance=12.4,
        lookat=None
    )

    viz = SwitchArbitrationVisualizer(refresh_ms=50, draw_forces_in_mujoco=True)

    stuck = TimeStuckDetector(
        window_sec=1.0,
        stuck_time_s=1.2,
        drift_eps=0.05,
        cmd_speed_eps=0.05,
        yaw_eps=0.10,
        near_obs_thresh=0.75,
        rep_force_thresh=1.0,
        goal_tol=max(0.35, getattr(apf_ai, "goal_tolerance", 0.3)),
    )

    start_time = time.time()
    trajectory = []
    step_count = 0
    loop_dt = 0.01

    # -------------------------
    # Logging buffers for plots
    # -------------------------
    log_t = []
    log_mode = []  # 0=HUMAN, 1=AI  (STUCK mapped to AI)
    log_v_h = []
    log_v_ai = []
    log_v_f = []

    try:
        while True:
            t_now = time.time()

            # latest human command
            cmd = {"A": 0, "B": 0, "vx": 0.0, "vy": 0.0, "yaw": 0.0}
            if not human_queue.empty():
                cmd = human_queue.get()

            vx_h, vy_h, w_h = cmd["vx"], cmd["vy"], cmd["yaw"]
            viz.set_buttons(cmd["A"], cmd["B"])


            if cmd["A"]:
                robot.standUpReset()
            if cmd["B"]:
                robot.sitDownReset()

            # AI APF command
            vx_ai, vy_ai, w_ai = apf_ai.control(robot, goal)

            # distances
            pos, _ = robot.getPose()
            robot_xy = np.array(pos[:2], dtype=float)
            dist_goal = float(np.linalg.norm(robot_xy - np.array(goal[:2], dtype=float)))
            min_obs = float(apf_ai.get_min_obstacle_distance(robot_xy))

            # select final cmd + base mode
            vx_f, vy_f, w_f, mode = arbitration_switch(
                vx_h, vy_h, w_h, cmd["A"], cmd["B"],
                vx_ai, vy_ai, w_ai,
                min_obs
            )

            # step robot
            robot.stepHighlevel(vx_f, vy_f, w_f)

            # debug forces
            debug = apf_ai.get_debug_info()
            F_rep_mag = float(debug.get("F_rep_magnitude", 0.0))

            # stuck state (time + tolerances)
            is_stuck, window_progress, drift = stuck.update(
                t_now=t_now,
                robot_xy=robot_xy,
                final_cmd=(vx_f, vy_f, w_f),
                dist_goal=dist_goal,
                min_obs=min_obs,
                F_rep_mag=F_rep_mag,
            )

            mode_str = "STUCK" if is_stuck else mode

            # visualization
            viz.update(
                mode=mode_str,
                human_cmd=(vx_h, vy_h, w_h),
                ai_cmd=(vx_ai, vy_ai, w_ai),
                final_cmd=(vx_f, vy_f, w_f),
                dist_goal=dist_goal,
                min_obs=min_obs,
                debug=debug
            )
            viz.draw_forces(robot)

            # -------------------------
            # Logging for plots
            # -------------------------
            t_rel = t_now - start_time
            log_t.append(t_rel)

            # Only HUMAN/AI in plot (STUCK -> AI)
            mode_for_plot = "AI" if mode_str == "STUCK" else mode
            log_mode.append(0 if mode_for_plot == "HUMAN" else 1)

            log_v_h.append(float(np.hypot(vx_h, vy_h)))
            log_v_ai.append(float(np.hypot(vx_ai, vy_ai)))
            log_v_f.append(float(np.hypot(vx_f, vy_f)))

            # console debug (not too frequent)
            if step_count % 150 == 0:
                print(
                    f"step={step_count:5d} mode={mode_str:5s} "
                    f"d_goal={dist_goal:.2f} min_obs={min_obs:.2f} "
                    f"|F_rep|={F_rep_mag:.2f} window={window_progress:.2f} drift={drift:.3f}"
                )

            # goal reached
            if apf_ai.is_goal_reached(robot, goal):
                robot.stepHighlevel(0.0, 0.0, 0.0)
                elapsed = time.time() - start_time
                path_len = calculate_path_length(trajectory)
                print("\nGOAL REACHED")
                print(f"Time: {elapsed:.2f}s | steps={step_count} | path={path_len:.2f}m")
                break

            trajectory.append(robot_xy.copy())
            step_count += 1
            time.sleep(loop_dt)

    except KeyboardInterrupt:
        print("\nCtrl+C -> exit")

    finally:
        try:
            robot.stepHighlevel(0.0, 0.0, 0.0)
        except Exception:
            pass
        try:
            viz.close()
        except Exception:
            pass
        try:
            p.terminate()
            p.join(timeout=0.2)
        except Exception:
            pass

    # -------------------------
    # Make plots at the end
    # -------------------------
    if len(log_t) > 10:
        save_dir = os.path.join(os.getcwd(), "plots")
        plot_results(
            log_t=log_t,
            log_mode=log_mode,
            log_v_h=log_v_h,
            log_v_ai=log_v_ai,
            log_v_f=log_v_f,
            save_dir=save_dir,
        )
        print(f"[plots] directory: {save_dir}")
    else:
        print("\nNot enough data logged to plot.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
