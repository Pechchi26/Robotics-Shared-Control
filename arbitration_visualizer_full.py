"""
Enhanced Arbitration Visualizer Module
Visualizes AI-Human control blending with:
- Integrated Nintendo Switch joystick visualization
- Alpha slider above robot (optional)
- Separate GUI window with real-time metrics
Uses existing ForceVisualizer for force vectors
"""

import numpy as np
import mujoco
from force_visualizer import ForceVisualizer
import tkinter as tk
from tkinter import ttk
import threading


class JoystickVisualizer(tk.Canvas):
    """
    Visualizes Nintendo Switch controller joysticks
    Shows left stick (movement) and right stick (rotation)
    """
    
    def __init__(self, parent, width=300, height=150, **kwargs):
        super().__init__(parent, width=width, height=height, 
                        bg='#1a1a1a', highlightthickness=0, **kwargs)
        
        self.width = width
        self.height = height
        
        # Joystick positions (normalized -1 to 1)
        self.left_x = 0.0
        self.left_y = 0.0
        self.right_x = 0.0
        self.right_y = 0.0
        
        # Visual parameters
        self.stick_radius = 45  # Base circle radius
        self.knob_radius = 15   # Moving knob radius
        self.deadzone_radius = 20  # Deadzone circle
        
        # Colors
        self.color_base = '#333333'
        self.color_deadzone = '#444444'
        self.color_knob_active = '#ff3333'
        self.color_knob_inactive = '#666666'
        self.color_border = '#555555'
        
    def update_joysticks(self, left_x=0.0, left_y=0.0, right_x=0.0, right_y=0.0):
        """
        Update joystick positions
        
        Args:
            left_x, left_y: Left stick position (-1 to 1)
            right_x, right_y: Right stick position (-1 to 1)
        """
        self.left_x = np.clip(left_x, -1, 1)
        self.left_y = np.clip(left_y, -1, 1)
        self.right_x = np.clip(right_x, -1, 1)
        self.right_y = np.clip(right_y, -1, 1)
        
        self.draw()
    
    def draw(self):
        """Draw both joysticks"""
        self.delete("all")
        
        # Left joystick (movement control)
        left_center_x = self.width // 4
        left_center_y = self.height // 2
        self._draw_joystick(left_center_x, left_center_y, 
                           self.left_x, self.left_y, 
                           label="MOVEMENT")
        
        # Right joystick (rotation control)
        right_center_x = 3 * self.width // 4
        right_center_y = self.height // 2
        self._draw_joystick(right_center_x, right_center_y, 
                           self.right_x, self.right_y,
                           label="ROTATION")
    
    def _draw_joystick(self, center_x, center_y, stick_x, stick_y, label=""):
        """
        Draw a single joystick
        
        Args:
            center_x, center_y: Center position on canvas
            stick_x, stick_y: Stick deflection (-1 to 1)
            label: Text label below joystick
        """
        # Base circle (outer boundary)
        self.create_oval(
            center_x - self.stick_radius,
            center_y - self.stick_radius,
            center_x + self.stick_radius,
            center_y + self.stick_radius,
            fill=self.color_base,
            outline=self.color_border,
            width=2
        )
        
        # Deadzone circle
        self.create_oval(
            center_x - self.deadzone_radius,
            center_y - self.deadzone_radius,
            center_x + self.deadzone_radius,
            center_y + self.deadzone_radius,
            fill=self.color_deadzone,
            outline='',
        )
        
        # Center crosshair
        crosshair_size = 8
        self.create_line(
            center_x - crosshair_size, center_y,
            center_x + crosshair_size, center_y,
            fill='#888888', width=1
        )
        self.create_line(
            center_x, center_y - crosshair_size,
            center_x, center_y + crosshair_size,
            fill='#888888', width=1
        )
        
        # Calculate knob position
        # Note: Y is inverted in canvas coordinates
        knob_x = center_x + stick_x * (self.stick_radius - self.knob_radius)
        knob_y = center_y - stick_y * (self.stick_radius - self.knob_radius)
        
        # Check if stick is active (outside deadzone)
        stick_magnitude = np.sqrt(stick_x**2 + stick_y**2)
        is_active = stick_magnitude > 0.15
        
        # Draw line from center to knob if active
        if is_active:
            self.create_line(
                center_x, center_y,
                knob_x, knob_y,
                fill='#ff6666', width=2, dash=(3, 3)
            )
        
        # Moving knob
        knob_color = self.color_knob_active if is_active else self.color_knob_inactive
        self.create_oval(
            knob_x - self.knob_radius,
            knob_y - self.knob_radius,
            knob_x + self.knob_radius,
            knob_y + self.knob_radius,
            fill=knob_color,
            outline='#ffffff' if is_active else '#444444',
            width=2
        )
        
        # Label
        if label:
            self.create_text(
                center_x,
                center_y + self.stick_radius + 15,
                text=label,
                fill='#aaaaaa',
                font=('Arial', 9, 'bold')
            )
        
        # Value display (only if active)
        if is_active:
            value_text = f"X:{stick_x:+.2f} Y:{stick_y:+.2f}"
            self.create_text(
                center_x,
                center_y + self.stick_radius + 30,
                text=value_text,
                fill='#00ff00',
                font=('Courier', 11, 'bold')
            )


class ArbitrationGUI:
    """
    Separate GUI window showing arbitration metrics in real-time
    Now includes integrated joystick visualization
    """
    
    def __init__(self):
        self.root = None
        self.running = False
        
        # Data storage
        self.alpha = 0.0
        self.risk = 0.0
        self.mode = "Initializing..."
        self.is_stuck = False
        self.min_dist = 0.0
        self.d_goal = 0.0
        self.window_progress = 0.0
        self.drift = 0.0
        self.cos_angle = 1.0
        self.F_att_mag = 0.0
        self.F_rep_mag = 0.0
        self.rescue_active = False
        
        # Human input data
        self.human_vx = 0.0
        self.human_vy = 0.0
        self.human_yaw = 0.0
        
        # UI elements
        self.alpha_label = None
        self.alpha_canvas = None
        self.risk_label = None
        self.mode_label = None
        self.metrics_labels = {}
        self.joystick_viz = None

        # ============================================================
        # FONT CONFIGURATION
        # ============================================================
        self.FONT_TITLE = ('Arial', 24, 'bold')
        self.FONT_SECTION = ('Arial', 18, 'bold')
        self.FONT_LABEL = ('Arial', 15, 'bold')
        self.FONT_VALUE = ('Courier', 15, 'bold')
        self.FONT_STATUS = ('Arial', 18, 'bold')
        self.FONT_MODE = ('Arial', 18, 'bold')
        self.FONT_ALPHA = ('Arial', 20, 'bold')
    
    def start(self):
        """Start GUI in separate thread"""
        if not self.running:
            self.running = True
            gui_thread = threading.Thread(target=self._run_gui, daemon=True)
            gui_thread.start()
    
    def _run_gui(self):
        """Run GUI main loop (in separate thread)"""
        self.root = tk.Tk()
        self.root.title("Arbitration Monitor + Controller")
        self.root.geometry("450x1000")  # Increased height for joysticks
        self.root.configure(bg='#2b2b2b')
        
        # Protocol for window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        self._create_widgets()
        self._update_display()
        
        self.root.mainloop()
    
    def _create_widgets(self):
        """Create all GUI widgets"""
        
        # Title
        title = tk.Label(self.root, text="ARBITRATION STATUS", 
                        font=self.FONT_TITLE,
                        bg='#2b2b2b', fg='#00ff00')
        title.pack(pady=15)
        
        # Separator
        ttk.Separator(self.root, orient='horizontal').pack(fill='x', padx=10, pady=5)
        
        # ===== JOYSTICK SECTION (NEW!) =====
        joystick_frame = tk.Frame(self.root, bg='#2b2b2b')
        joystick_frame.pack(fill='x', padx=20, pady=10)
        
        tk.Label(joystick_frame, text="Controller Input", 
                font=self.FONT_SECTION,
                bg='#2b2b2b', fg='#ffffff').pack(anchor='w')
        
        # Joystick visualizer
        self.joystick_viz = JoystickVisualizer(joystick_frame, width=400, height=160)
        self.joystick_viz.pack(pady=10)
        
        # Separator
        ttk.Separator(self.root, orient='horizontal').pack(fill='x', padx=10, pady=10)
        
        # ===== ALPHA SECTION =====
        alpha_frame = tk.Frame(self.root, bg='#2b2b2b')
        alpha_frame.pack(fill='x', padx=20, pady=15)
        
        tk.Label(alpha_frame, text="Control Authority (alpha)", 
                font=self.FONT_ALPHA,
                bg='#2b2b2b', fg='#ffffff').pack(anchor='w')
        
        self.alpha_label = tk.Label(alpha_frame, text="alpha = 0.000", 
                                    font=('Arial', 16, 'bold'),
                                    bg='#2b2b2b', fg='#00ff00')
        self.alpha_label.pack(anchor='w', pady=8)
        
        # Alpha slider canvas
        self.alpha_canvas = tk.Canvas(alpha_frame, width=410, height=50, 
                                      bg='#1a1a1a', highlightthickness=0)
        self.alpha_canvas.pack(pady=8)
        
        # Labels for Human and AI
        labels_frame = tk.Frame(alpha_frame, bg='#2b2b2b')
        labels_frame.pack(fill='x', pady=5)
        
        tk.Label(labels_frame, text="HUMAN", font=('Arial', 11, 'bold'),
                bg='#2b2b2b', fg='#ff9933').pack(side='left')
        
        tk.Label(labels_frame, text="AI", font=('Arial', 11, 'bold'),
                bg='#2b2b2b', fg='#3399ff').pack(side='right')
        
        # ===== RISK SECTION =====
        ttk.Separator(self.root, orient='horizontal').pack(fill='x', padx=10, pady=10)
        
        risk_frame = tk.Frame(self.root, bg='#2b2b2b')
        risk_frame.pack(fill='x', padx=20, pady=15)
        
        tk.Label(risk_frame, text="Risk Level", 
                font=self.FONT_STATUS,
                bg='#2b2b2b', fg='#ffffff').pack(anchor='w')
        
        self.risk_label = tk.Label(risk_frame, text="Risk = 0.000 | Status: OK", 
                                   font=('Arial', 14, 'bold'),
                                   bg='#2b2b2b', fg='#00ff00')
        self.risk_label.pack(anchor='w', pady=8)
        
        # ===== MODE SECTION =====
        ttk.Separator(self.root, orient='horizontal').pack(fill='x', padx=10, pady=10)
        
        mode_frame = tk.Frame(self.root, bg='#2b2b2b')
        mode_frame.pack(fill='x', padx=20, pady=15)
        
        tk.Label(mode_frame, text="Current Mode", 
                font=self.FONT_MODE,
                bg='#2b2b2b', fg='#ffffff').pack(anchor='w')
        
        self.mode_label = tk.Label(mode_frame, text="Initializing...", 
                                   font=self.FONT_LABEL,
                                   bg='#2b2b2b', fg='#ffff00',
                                   wraplength=400, justify='left')
        self.mode_label.pack(anchor='w', pady=8)
        
        # ===== METRICS SECTION (Compact) =====
        ttk.Separator(self.root, orient='horizontal').pack(fill='x', padx=10, pady=10)
        
        metrics_frame = tk.Frame(self.root, bg='#2b2b2b')
        metrics_frame.pack(fill='x', padx=20, pady=10)
        
        tk.Label(metrics_frame, text="Metrics", 
                font=self.FONT_SECTION,
                bg='#2b2b2b', fg='#ffffff').pack(anchor='w', pady=(0, 10))
        
        # Create metric labels (compact version)
        metric_names = [
            ('d_goal', 'Goal:'),
            ('min_dist', 'Obstacle:'),
            ('f_att', 'F_att:'),
            ('f_rep', 'F_rep:'),
        ]
        
        for key, label_text in metric_names:
            row = tk.Frame(metrics_frame, bg='#2b2b2b')
            row.pack(fill='x', pady=2)
            
            tk.Label(row, text=label_text,
                    font=self.FONT_LABEL,
                    bg='#2b2b2b', fg='#aaaaaa',
                    width=12, anchor='w').pack(side='left')
            
            value_label = tk.Label(row, text="0.000", 
                                  font=self.FONT_VALUE,
                                  bg='#2b2b2b', fg='#00ff00',
                                  anchor='e')
            value_label.pack(side='right', padx=10)
            self.metrics_labels[key] = value_label
    
    def _update_display(self):
        """Update all display elements (called periodically)"""
        if not self.running or not self.root:
            return
        
        # Update joystick visualization
        if self.joystick_viz:
            # Convert controller coordinates to joystick display
            # Left stick: vx (forward/back), vy (left/right)
            # Right stick: yaw (rotation)
            self.joystick_viz.update_joysticks(
                left_x=-(self.human_vy / 1.2),   # Normalize by max velocity
                left_y=(self.human_vx / 1.2),
                right_x=-(self.human_yaw / 1.5),
                right_y=0.0  # Y-axis not used for rotation
            )
        
        # Update alpha display
        self.alpha_label.config(text=f"alpha = {self.alpha:.3f}")
        
        # Update alpha bar on canvas
        self._draw_alpha_bar()
        
        # Update risk
        status = "STUCK" if self.is_stuck else "OK"
        risk_color = '#ff0000' if self.risk > 0.7 else ('#ffaa00' if self.risk > 0.3 else '#00ff00')
        
        self.risk_label.config(text=f"Risk = {self.risk:.3f} | Status: {status}", fg=risk_color)
        
        # Update mode
        mode_color = '#00ff00' if 'AI-Auto' in self.mode else ('#ff9933' if 'Human' in self.mode else '#ffff00')
        self.mode_label.config(text=self.mode, fg=mode_color)
        
        # Update metrics (compact)
        self.metrics_labels['d_goal'].config(text=f"{self.d_goal:.2f} m")
        self.metrics_labels['min_dist'].config(text=f"{self.min_dist:.2f} m")
        self.metrics_labels['f_att'].config(text=f"{self.F_att_mag:.2f}")
        self.metrics_labels['f_rep'].config(text=f"{self.F_rep_mag:.2f}")
        
        # Schedule next update
        if self.root:
            self.root.after(50, self._update_display)  # Update 20 times per second
    
    def _draw_alpha_bar(self):
        """Draw alpha bar with moving sections"""
        if not self.alpha_canvas:
            return
        
        self.alpha_canvas.delete("all")
        
        # Bar dimensions
        canvas_width = 410
        canvas_height = 50
        bar_height = 30
        bar_width = 380
        
        bar_left = 15
        bar_right = bar_left + bar_width
        bar_top = (canvas_height - bar_height) // 2
        bar_bottom = bar_top + bar_height
        mid_x = (bar_left + bar_right) // 2
        
        # Background (gray base)
        self.alpha_canvas.create_rectangle(bar_left, bar_top, bar_right, bar_bottom,
                                          fill='#1a1a1a', outline='#555555', width=2)
        
        # DYNAMIC Human section (left, orange) - shrinks as alpha increases
        human_width = int(bar_width * (1.0 - self.alpha))
        if human_width > 0:
            self.alpha_canvas.create_rectangle(bar_left, bar_top, bar_left + human_width, bar_bottom,
                                              fill='#ff9933', outline='')
        
        # DYNAMIC AI section (right, blue) - grows with alpha
        ai_width = int(bar_width * self.alpha)
        if ai_width > 0:
            self.alpha_canvas.create_rectangle(bar_right - ai_width, bar_top, bar_right, bar_bottom,
                                              fill='#3399ff', outline='')
        
        # Center divider line
        self.alpha_canvas.create_line(mid_x, bar_top-3, mid_x, bar_bottom+3,
                                      fill='#888888', width=2, dash=(3, 3))
        
        # MOVING alpha indicator (white circle)
        indicator_x = bar_left + int(bar_width * self.alpha)
        indicator_y = (bar_top + bar_bottom) // 2
        indicator_r = 12
        
        self.alpha_canvas.create_oval(indicator_x - indicator_r, indicator_y - indicator_r,
                                      indicator_x + indicator_r, indicator_y + indicator_r,
                                      fill='#ffffff', outline='#000000', width=3)
    
    def update_data(self, stuck_state, mode, debug, min_dist, human_cmd=None):
        """
        Update data from main simulation thread
        
        Args:
            stuck_state: Dictionary with arbitration state
            mode: Current control mode string
            debug: Debug info from APF controller
            min_dist: Distance to nearest obstacle
            human_cmd: Human controller command dict (optional)
        """
        self.alpha = stuck_state.get('alpha', 0.0)
        self.risk = stuck_state.get('risk_level', 0.0)
        self.mode = mode
        self.is_stuck = stuck_state.get('is_stuck', False)
        self.min_dist = min_dist
        self.d_goal = stuck_state.get('d_goal', 0.0)
        self.window_progress = stuck_state.get('window_progress', 0.0)
        self.drift = stuck_state.get('drift', 0.0)
        self.cos_angle = stuck_state.get('cos_angle', 1.0)
        self.F_att_mag = debug.get('F_att_magnitude', 0.0)
        self.F_rep_mag = debug.get('F_rep_magnitude', 0.0)
        self.rescue_active = stuck_state.get('human_rescue_active', False)
        
        # Update human input data
        if human_cmd:
            self.human_vx = human_cmd.get('vx', 0.0)
            self.human_vy = human_cmd.get('vy', 0.0)
            self.human_yaw = human_cmd.get('yaw', 0.0)
    
    def _on_closing(self):
        """Handle window close"""
        self.running = False
        if self.root:
            self.root.destroy()
    
    def stop(self):
        """Stop GUI"""
        self.running = False
        if self.root:
            self.root.quit()


class ArbitrationVisualizer:
    """
    Visualizes arbitration state with:
    - Integrated joystick visualization in GUI
    - Alpha slider above robot (optional, can be disabled)
    - Separate GUI window (real-time metrics)
    - Force vectors
    """
    
    def __init__(self, show_3d_slider=True, slider_height=0.6, slider_width=1.2):
        """
        Args:
            show_3d_slider: If True, show slider above robot in 3D. If False, only GUI window.
            slider_height: Height above robot to place slider
            slider_width: Width of the slider bar
        """
        # Use existing ForceVisualizer
        self.force_viz = ForceVisualizer()
        
        # 3D Slider settings
        self.show_3d_slider = show_3d_slider
        self.slider_height = slider_height
        self.slider_width = slider_width
        self.bar_thickness = 0.06
        
        # Colors for 3D visualization
        self.color_human = np.array([1.0, 0.6, 0.2, 0.85])     # Orange
        self.color_ai = np.array([0.2, 0.6, 1.0, 0.85])        # Blue
        self.color_bg = np.array([0.25, 0.25, 0.25, 0.7])      # Dark gray
        self.color_indicator = np.array([1.0, 1.0, 1.0, 1.0])  # White
        self.color_divider = np.array([1.0, 1.0, 1.0, 0.9])    # White
        
        # GUI window
        self.gui = ArbitrationGUI()
        self.gui.start()
    
    def visualize_all(self, robot, apf_controller, stuck_state, mode, human_cmd=None):
        """
        Main visualization function
        
        Args:
            robot: Go2Sim object
            apf_controller: APFController object
            stuck_state: Dictionary with arbitration state
            mode: Current control mode string
            human_cmd: Human controller command dict (optional)
        """
        if not hasattr(robot, 'viewer') or robot.viewer is None:
            return
        
        viewer = robot.viewer
        
        # Reset scene
        self.force_viz.reset_scene(robot)
        
        # Get debug info and draw forces
        debug = apf_controller.get_debug_info()
        F_attractive = debug.get('F_attractive', np.zeros(2))
        F_repulsive = debug.get('F_repulsive', np.zeros(2))
        F_total = debug.get('F_total', np.zeros(2))
        
        self.force_viz.visualize_forces(robot, F_attractive, F_repulsive, F_total)
        
        # Draw 3D slider if enabled
        if self.show_3d_slider:
            alpha = stuck_state.get('alpha', 0.0)
            self._draw_alpha_slider(robot, viewer, alpha)
        
        # Update GUI window (including joystick visualization)
        pos, _ = robot.getPose()
        min_dist = apf_controller.get_min_obstacle_distance(pos[:2])
        self.gui.update_data(stuck_state, mode, debug, min_dist, human_cmd)
    
    def _draw_alpha_slider(self, robot, viewer, alpha):
        """
        Draw FIXED alpha slider above robot
        Only white indicator sphere moves
        """
        # Get robot position
        pos, _ = robot.getPose()
        
        # Slider center (above robot)
        slider_center = pos.copy()
        slider_center[2] += self.slider_height
        
        # Background bar (gray base)
        bg_size = np.array([self.slider_width, self.bar_thickness, self.bar_thickness])
        self._draw_box(viewer, slider_center, bg_size, self.color_bg)
        
        # DYNAMIC Human section (left, orange) - width = (1-alpha) * total_width
        human_width = self.slider_width * (1.0 - alpha)
        if human_width > 0.01:
            human_center = slider_center.copy()
            human_center[0] = slider_center[0] - (self.slider_width / 2.0) + (human_width / 2.0)
            human_size = np.array([human_width, self.bar_thickness * 0.9, self.bar_thickness * 0.9])
            self._draw_box(viewer, human_center, human_size, self.color_human)
        
        # DYNAMIC AI section (right, blue) - width = alpha * total_width
        ai_width = self.slider_width * alpha
        if ai_width > 0.01:
            ai_center = slider_center.copy()
            ai_center[0] = slider_center[0] + (self.slider_width / 2.0) - (ai_width / 2.0)
            ai_size = np.array([ai_width, self.bar_thickness * 0.9, self.bar_thickness * 0.9])
            self._draw_box(viewer, ai_center, ai_size, self.color_ai)
        
        # Center reference line (subtle, shows 50/50 point)
        divider_pos = slider_center.copy()
        divider_size = np.array([0.01, self.bar_thickness * 1.1, self.bar_thickness * 1.1])
        divider_color = np.array([0.7, 0.7, 0.7, 0.5])
        self._draw_box(viewer, divider_pos, divider_size, divider_color)
        
        # MOVING indicator (white sphere) - ONLY THIS MOVES
        indicator_pos = slider_center.copy()
        indicator_pos[0] = slider_center[0] - (self.slider_width / 2.0) + (self.slider_width * alpha)
        indicator_pos[2] += self.bar_thickness * 0.8
        self._draw_sphere(viewer, indicator_pos, 0.06, self.color_indicator)
    
    def _draw_box(self, viewer, position, size, color):
        """Draw a 3D box"""
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            return
        
        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_BOX,
            size * 0.5,
            position,
            np.eye(3).flatten(),
            color
        )
        
        viewer.user_scn.ngeom += 1
    
    def _draw_sphere(self, viewer, position, radius, color):
        """Draw a sphere"""
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            return
        
        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        size = np.array([radius, radius, radius])
        
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            size,
            position,
            np.eye(3).flatten(),
            color
        )
        
        viewer.user_scn.ngeom += 1
    
    def stop(self):
        """Stop GUI window"""
        self.gui.stop()