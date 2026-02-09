"""
ai_apf_visualizer.py

Standalone AI/APF visualizer that displays:
- Command: vx, vy, yaw (stick visualization + numeric)
- APF forces: F_att, F_rep, F_total (vector + magnitude)
- Distances: dist_to_goal, min_obstacle_dist

"""

import threading
import tkinter as tk
import numpy as np


class AIAPFVisualizer:
    def __init__(
        self,
        title="AI / APF ",
        vx_max=1.5,
        vy_max=1.5,
        yaw_max=2.0,
        window_size="500x430",
        refresh_ms=16,
        invert_vy_display=True,
        invert_yaw_display=False,
    ):
        from arbitration_visualizer_full import JoystickVisualizer  # UI reference :contentReference[oaicite:6]{index=6}
        self._JoystickVisualizer = JoystickVisualizer

        self.title = title
        self.vx_max = float(vx_max)
        self.vy_max = float(vy_max)
        self.yaw_max = float(yaw_max)
        self.window_size = window_size
        self.refresh_ms = int(refresh_ms)

        self.invert_vy_display = bool(invert_vy_display)
        self.invert_yaw_display = bool(invert_yaw_display)

        # latest values (written by update_from)
        self._vx = 0.0
        self._vy = 0.0
        self._yaw = 0.0

        self._F_att = np.zeros(2)
        self._F_rep = np.zeros(2)
        self._F_tot = np.zeros(2)

        self._dist_goal = float("nan")
        self._min_obs = float("nan")
        #self._d_safe = float("nan")

        self._running = False
        self._exit_requested = False

        self._root = None
        self._viz = None

        self._cmd_lbl = None
        self._force_lbl = None
        self._dist_lbl = None

        self._thread = None

    # -----------------------
    # Public API
    # -----------------------
    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_gui, daemon=True)
        self._thread.start()

    def close(self):
        self._exit_requested = True
        self._running = False
        try:
            if self._root is not None:
                self._root.after(0, self._root.destroy)
        except Exception:
            pass

    def exit_requested(self) -> bool:
        return bool(self._exit_requested)

    def update_from(self, robot, controller, goal_pos, vx, vy, yaw):
        """
        Call this ONCE per control step, right after controller.control(...)

        It reads:
          - controller.get_debug_info() for forces
          - robot pose + goal for distance-to-goal
          - controller.get_min_obstacle_distance() if exists, else uses robot.getLaserScan()
        """
        self._vx = float(vx)
        self._vy = float(vy)
        self._yaw = float(yaw)

        # Forces from controller
        dbg = controller.get_debug_info() if hasattr(controller, "get_debug_info") else {}
        self._F_att = np.array(dbg.get("F_attractive", [0.0, 0.0]), dtype=float)
        self._F_rep = np.array(dbg.get("F_repulsive", [0.0, 0.0]), dtype=float)
        self._F_tot = np.array(dbg.get("F_total", [0.0, 0.0]), dtype=float)

        # d_safe (nice to display)
        #self._d_safe = float(getattr(controller, "d_safe", float("nan")))

        # Dist to goal
        try:
            pos, _ = robot.getPose()
            robot_xy = np.array(pos[:2], dtype=float)
            g = np.array(goal_pos[:2], dtype=float)
            self._dist_goal = float(np.linalg.norm(robot_xy - g))
        except Exception:
            self._dist_goal = float("nan")

        # Min obstacle distance:
        # - XML controller provides get_min_obstacle_distance(robot_pos) 
        # - LiDAR controller: compute from robot.getLaserScan()['dist'] :contentReference[oaicite:8]{index=8}
        self._min_obs = self._compute_min_obstacle_distance(robot, controller)

    # -----------------------
    # Internals
    # -----------------------
    def _compute_min_obstacle_distance(self, robot, controller) -> float:
        # XML APF path
        if hasattr(controller, "get_min_obstacle_distance"):
            try:
                pos, _ = robot.getPose()
                robot_xy = np.array(pos[:2], dtype=float)
                return float(controller.get_min_obstacle_distance(robot_xy))
            except Exception:
                pass

        # LiDAR APF path
        try:
            scan = robot.getLaserScan()
            if scan is None:
                return float("nan")
            d = np.array(scan.get("dist", []), dtype=float)
            if d.size == 0:
                return float("nan")

            # Filter invalid / far readings
            d = d[(d > 0.0) & np.isfinite(d)]
            if d.size == 0:
                return float("nan")
            return float(np.min(d))
        except Exception:
            return float("nan")

    def _run_gui(self):
        self._root = tk.Tk()
        self._root.title(self.title)
        self._root.geometry(self.window_size)
        self._root.configure(bg="#2b2b2b")
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        title = tk.Label(
            self._root, text=self.title,
            font=("Arial", 18, "bold"),
            bg="#2b2b2b", fg="#ffffff"
        )
        title.pack(pady=8)

        self._viz = self._JoystickVisualizer(self._root, width=480, height=170)
        self._viz.pack(pady=6)

        self._cmd_lbl = tk.Label(
            self._root,
            text="cmd: vx=0.00  vy=0.00  yaw=0.00",
            font=("Courier", 15, "bold"),
            bg="#2b2b2b", fg="#00ff00"
        )
        self._cmd_lbl.pack(pady=6)

        self._force_lbl = tk.Label(
            self._root,
            text=(
                "F_att=[+0.00,+0.00] | |F_att|=0.00\n"
                "F_rep=[+0.00,+0.00] | |F_rep|=0.00\n"
                "F_tot=[+0.00,+0.00] | |F_tot|=0.00"
            ),
            justify="left",
            font=("Courier", 14, "bold"),
            bg="#2b2b2b", fg="#ffffff"
        )
        self._force_lbl.pack(pady=6)

        self._dist_lbl = tk.Label(
            self._root,
            text="dist_goal=nan   min_obs=nan ",
            font=("Courier", 12, "bold"),
            bg="#2b2b2b", fg="#ffcc66"
        )
        self._dist_lbl.pack(pady=6)

        self._tick()
        self._root.mainloop()

        self._running = False
        self._exit_requested = True

    def _on_close(self):
        self._exit_requested = True
        self._running = False
        try:
            self._root.destroy()
        except Exception:
            pass

    def _tick(self):
        if not self._running or self._root is None:
            return

        vx, vy, yaw = self._vx, self._vy, self._yaw

        vx_max = self.vx_max if self.vx_max > 1e-9 else 1.0
        vy_max = self.vy_max if self.vy_max > 1e-9 else 1.0
        yaw_max = self.yaw_max if self.yaw_max > 1e-9 else 1.0

        vy_disp = -vy if self.invert_vy_display else vy
        yaw_disp = -yaw if self.invert_yaw_display else yaw

        left_x = float(np.clip(vy_disp / vy_max, -1.0, 1.0))
        left_y = float(np.clip(vx / vx_max, -1.0, 1.0))
        right_x = float(np.clip(yaw_disp / yaw_max, -1.0, 1.0))

        if self._viz is not None:
            self._viz.update_joysticks(left_x=left_x, left_y=left_y, right_x=right_x, right_y=0.0)

        if self._cmd_lbl is not None:
            self._cmd_lbl.config(text=f"cmd: vx={vx:+.2f}  vy={vy:+.2f}  yaw={yaw:+.2f}")

        F_att = self._F_att
        F_rep = self._F_rep
        F_tot = self._F_tot

        if self._force_lbl is not None:
            self._force_lbl.config(
                text=(
                    f"F_att=[{F_att[0]:+6.2f},{F_att[1]:+6.2f}] | |F_att|={np.linalg.norm(F_att):6.2f}\n"
                    f"F_rep=[{F_rep[0]:+6.2f},{F_rep[1]:+6.2f}] | |F_rep|={np.linalg.norm(F_rep):6.2f}\n"
                    f"F_tot=[{F_tot[0]:+6.2f},{F_tot[1]:+6.2f}] | |F_tot|={np.linalg.norm(F_tot):6.2f}"
                )
            )

        if self._dist_lbl is not None:
            dg = self._dist_goal
            mo = self._min_obs
            #ds = self._d_safe
            self._dist_lbl.config(
                text=f"dist_goal={dg:6.2f}   min_obs={mo:6.2f}"
            )

        self._root.after(self.refresh_ms, self._tick)
