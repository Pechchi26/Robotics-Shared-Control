"""
arbitration_switch_visualizer.py

Switch (Human vs AI) visualizer in the SAME style as AIAPFVisualizer:
- Two joystick panels (Human + AI)
- Button state display (A/B) for human controller
- Mode (big, color-coded)
- Forces (F_att/F_rep/F_tot with magnitudes)
- Distances (dist_goal, min_obs)
- Draw force arrows in MuJoCo via ForceVisualizer
"""

import threading
import tkinter as tk
import numpy as np

from force_visualizer import ForceVisualizer


class SwitchArbitrationVisualizer:
    """
    GUI:
      - MODE (color-coded)
      - Human input: joystick + cmd numbers + A/B state
      - AI input: joystick + cmd numbers
      - Forces: F_att, F_rep, F_tot
      - Distances: dist_goal, min_obs

    MuJoCo overlay:
      - Force arrows via ForceVisualizer
    """

    def __init__(
        self,
        title="SWITCH ARBITRATION MONITOR",
        vx_max=1.5,
        vy_max=1.5,
        yaw_max=2.0,
        window_size="460x720",
        refresh_ms=50,
        draw_forces_in_mujoco=True,
        invert_vy_display=True,
        invert_yaw_display=True,
    ):
        # Joystick widget is used in your AI visualizer style
        from arbitration_visualizer_full import JoystickVisualizer
        self._JoystickVisualizer = JoystickVisualizer

        self.title = str(title)
        self.vx_max = float(vx_max)
        self.vy_max = float(vy_max)
        self.yaw_max = float(yaw_max)
        self.window_size = str(window_size)
        self.refresh_ms = int(refresh_ms)

        self.draw_forces_in_mujoco = bool(draw_forces_in_mujoco)
        self.invert_vy_display = bool(invert_vy_display)
        self.invert_yaw_display = bool(invert_yaw_display)

        self.force_viz = ForceVisualizer()

        # -------- Shared state (updated from main thread) --------
        self.mode = "INIT"

        self.hcmd = (0.0, 0.0, 0.0)  # vx, vy, yaw
        self.acmd = (0.0, 0.0, 0.0)  # vx, vy, yaw
        self.fcmd = (0.0, 0.0, 0.0)  # vx, vy, yaw (final)

        self.btnA = 0
        self.btnB = 0

        self.dist_goal = float("nan")
        self.min_obs = float("nan")

        self.F_att = np.zeros(2)
        self.F_rep = np.zeros(2)
        self.F_tot = np.zeros(2)

        # -------- GUI thread state --------
        self._running = False
        self._exit_requested = False
        self._root = None

        # Widgets
        self._mode_lbl = None

        self._human_viz = None
        self._ai_viz = None

        self._human_cmd_lbl = None
        self._human_btn_lbl = None

        self._ai_cmd_lbl = None

        self._force_lbl = None
        self._dist_lbl = None

        self._thread = None
        self.start()

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

    def update(self, mode, human_cmd, ai_cmd, final_cmd, dist_goal, min_obs, debug):
        # mode
        self.mode = str(mode)

        # commands
        self.hcmd = tuple(float(x) for x in human_cmd)
        self.acmd = tuple(float(x) for x in ai_cmd)
        self.fcmd = tuple(float(x) for x in final_cmd)

        # distances
        self.dist_goal = float(dist_goal)
        self.min_obs = float(min_obs)

        # forces from APF debug
        self.F_att = np.array(debug.get("F_attractive", [0.0, 0.0]), dtype=float)
        self.F_rep = np.array(debug.get("F_repulsive", [0.0, 0.0]), dtype=float)
        self.F_tot = np.array(debug.get("F_total", [0.0, 0.0]), dtype=float)

    def set_buttons(self, A: int, B: int):
        self.btnA = int(A)
        self.btnB = int(B)

    def draw_forces(self, robot):
        """Call every loop after update(); draws force arrows in MuJoCo."""
        if not self.draw_forces_in_mujoco:
            return
        self.force_viz.reset_scene(robot)
        self.force_viz.visualize_forces(robot, self.F_att, self.F_rep, self.F_tot)

    # -----------------------
    # Internals
    # -----------------------
    def _run_gui(self):
        self._root = tk.Tk()
        self._root.title(self.title)
        self._root.geometry(self.window_size)
        self._root.configure(bg="#2b2b2b")
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Title
        title = tk.Label(
            self._root,
            text=self.title,
            font=("Arial", 18, "bold"),
            bg="#2b2b2b",
            fg="#ffffff",
        )
        title.pack(pady=8)

        # MODE (big)
        self._mode_lbl = tk.Label(
            self._root,
            text="MODE: INIT",
            font=("Arial", 16, "bold"),
            bg="#2b2b2b",
            fg="#33aaff",
        )
        self._mode_lbl.pack(pady=6)

        # ------------ HUMAN PANEL ------------
        human_frame = tk.LabelFrame(
            self._root,
            text="HUMAN INPUT",
            font=("Arial", 12, "bold"),
            fg="#ffcc66",
            bg="#2b2b2b",
            bd=2,
            relief="groove",
            labelanchor="n",
        )
        human_frame.pack(fill="x", padx=10, pady=8)

        self._human_viz = self._JoystickVisualizer(human_frame, width=400, height=160)
        self._human_viz.pack(pady=6)

        self._human_cmd_lbl = tk.Label(
            human_frame,
            text="cmd: vx=+0.00  vy=+0.00  yaw=+0.00",
            font=("Courier", 14, "bold"),
            bg="#2b2b2b",
            fg="#00ff99",
        )
        self._human_cmd_lbl.pack(pady=4)

        self._human_btn_lbl = tk.Label(
            human_frame,
            text="Buttons: A=0  B=0",
            font=("Courier", 12, "bold"),
            bg="#2b2b2b",
            fg="#ffffff",
        )
        self._human_btn_lbl.pack(pady=2)

        # ------------ AI PANEL ------------
        ai_frame = tk.LabelFrame(
            self._root,
            text="AI (APF) INPUT",
            font=("Arial", 12, "bold"),
            fg="#66ccff",
            bg="#2b2b2b",
            bd=2,
            relief="groove",
            labelanchor="n",
        )
        ai_frame.pack(fill="x", padx=10, pady=8)

        self._ai_viz = self._JoystickVisualizer(ai_frame, width=400, height=160)
        self._ai_viz.pack(pady=6)

        self._ai_cmd_lbl = tk.Label(
            ai_frame,
            text="cmd: vx=+0.00  vy=+0.00  yaw=+0.00",
            font=("Courier", 14, "bold"),
            bg="#2b2b2b",
            fg="#00ff99",
        )
        self._ai_cmd_lbl.pack(pady=4)

        # ------------ FORCES + DISTANCES ------------
        info_frame = tk.Frame(self._root, bg="#2b2b2b")
        info_frame.pack(fill="x", padx=10, pady=8)

        self._force_lbl = tk.Label(
            info_frame,
            text=(
                "F_att=[+0.00,+0.00] | |F_att|=0.00\n"
                "F_rep=[+0.00,+0.00] | |F_rep|=0.00\n"
                "F_tot=[+0.00,+0.00] | |F_tot|=0.00"
            ),
            justify="left",
            font=("Courier", 13, "bold"),
            bg="#2b2b2b",
            fg="#ffffff",
        )
        self._force_lbl.pack(pady=6, anchor="w")

        self._dist_lbl = tk.Label(
            info_frame,
            text="dist_goal=nan   min_obs=nan",
            font=("Courier", 12, "bold"),
            bg="#2b2b2b",
            fg="#ffcc66",
        )
        self._dist_lbl.pack(pady=4, anchor="w")

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

    def _cmd_to_sticks(self, vx, vy, yaw):
        """Map (vx,vy,yaw) -> joystick normalized values for the widget."""
        vx_max = self.vx_max if self.vx_max > 1e-9 else 1.0
        vy_max = self.vy_max if self.vy_max > 1e-9 else 1.0
        yaw_max = self.yaw_max if self.yaw_max > 1e-9 else 1.0

        vy_disp = -vy if self.invert_vy_display else vy
        yaw_disp = -yaw if self.invert_yaw_display else yaw

        left_x = float(np.clip(vy_disp / vy_max, -1.0, 1.0))
        left_y = float(np.clip(vx / vx_max, -1.0, 1.0))
        right_x = float(np.clip(yaw_disp / yaw_max, -1.0, 1.0))
        return left_x, left_y, right_x

    def _tick(self):
        if not self._running or self._root is None:
            return

        # MODE color
        mode = self.mode.upper()
        if "STUCK" in mode:
            mode_color = "#ff4444"
        elif "HUMAN" in mode:
            mode_color = "#ff9933"
        else:
            mode_color = "#33aaff"

        if self._mode_lbl is not None:
            self._mode_lbl.config(text=f"MODE: {mode}", fg=mode_color)

        # HUMAN joystick + text
        hvx, hvy, hyaw = self.hcmd
        h_lx, h_ly, h_rx = self._cmd_to_sticks(hvx, hvy, hyaw)
        if self._human_viz is not None:
            self._human_viz.update_joysticks(left_x=h_lx, left_y=h_ly, right_x=h_rx, right_y=0.0)
        if self._human_cmd_lbl is not None:
            self._human_cmd_lbl.config(text=f"cmd: vx={hvx:+.2f}  vy={hvy:+.2f}  yaw={hyaw:+.2f}")
        if self._human_btn_lbl is not None:
            self._human_btn_lbl.config(text=f"Buttons: A={self.btnA:d}  B={self.btnB:d}")

        # AI joystick + text
        avx, avy, ayaw = self.acmd
        a_lx, a_ly, a_rx = self._cmd_to_sticks(avx, avy, ayaw)
        if self._ai_viz is not None:
            self._ai_viz.update_joysticks(left_x=a_lx, left_y=a_ly, right_x=a_rx, right_y=0.0)
        if self._ai_cmd_lbl is not None:
            self._ai_cmd_lbl.config(text=f"cmd: vx={avx:+.2f}  vy={avy:+.2f}  yaw={ayaw:+.2f}")

        # Forces
        F_att = self.F_att
        F_rep = self.F_rep
        F_tot = self.F_tot

        if self._force_lbl is not None:
            self._force_lbl.config(
                text=(
                    f"F_att=[{F_att[0]:+6.2f},{F_att[1]:+6.2f}] | |F_att|={np.linalg.norm(F_att):6.2f}\n"
                    f"F_rep=[{F_rep[0]:+6.2f},{F_rep[1]:+6.2f}] | |F_rep|={np.linalg.norm(F_rep):6.2f}\n"
                    f"F_tot=[{F_tot[0]:+6.2f},{F_tot[1]:+6.2f}] | |F_tot|={np.linalg.norm(F_tot):6.2f}"
                )
            )

        # Distances
        if self._dist_lbl is not None:
            self._dist_lbl.config(text=f"dist_goal={self.dist_goal:6.2f}   min_obs={self.min_obs:6.2f}")

        self._root.after(self.refresh_ms, self._tick)
