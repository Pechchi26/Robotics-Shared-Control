"""
arbitration_blend_controller.py

Determines the final robot velocity by blending Human inputs 
and AI planning commands based on the current system risk and operation state.
        
"""

import sys
sys.path.append("/home/wolfie/Go2Py")

from human_controller_for_arbitration import joystick_process
from apf_controller2 import APFController
import numpy as np
import time
import os
import multiprocessing as mp
from Go2Py.sim.mujoco import Go2Sim
from Go2Py import ASSETS_PATH
from utils import get_goal_from_scene, plot_debug
import matplotlib
matplotlib.use("Agg")   
from arbitration_visualizer_full import ArbitrationVisualizer
import matplotlib.pyplot as plt

def set_viewer_camera(robot, azimuth=135, elevation=-25, distance=4.0, lookat=None):
    """Viewer camera preset (NO effect on control)."""
    if not hasattr(robot, "viewer") or robot.viewer is None:
        return
    v = robot.viewer

    if lookat is None:
        pos, _ = robot.getPose()
        x = float(pos[0]); y = float(pos[1])
        z = float(pos[2]) if len(pos) >= 3 else 0.4
        lookat = np.array([x, y, z], dtype=float)
    else:
        lookat = np.array(lookat, dtype=float)

    try:
        v.cam.azimuth = float(azimuth)
        v.cam.elevation = float(elevation)
        v.cam.distance = float(distance)
        v.cam.lookat[:] = lookat
    except Exception as e:
        print(f"[WARN] Could not set viewer camera: {e}")

def update_stuck_state(goal, robot, apf_ai, stuck_state):
    """
    Risk Evaluation:
    Identifies 3 distinct trap mechanisms:
    A. Physical Stall (Immediate hardware blockage)
    B. Predictive Trap (Future geometric blockage)
    C. Temporal Stagnation (Algorithm local minima) 
    """
    
    
    pos, _ = robot.getPose() 
    robot_pos = pos[:2]
    
    try:
        current_yaw = float(pos[2])
    except IndexError:
        current_yaw = 0.0 
        
    d_goal = float(np.linalg.norm(robot_pos - goal))
    stuck_state["d_goal"] = d_goal

   
    current_vel = robot.getLinVel()[:2]
    actual_speed = float(np.linalg.norm(current_vel))

   
    debug = apf_ai.get_debug_info()
    F_total = np.array(debug.get("F_total", np.zeros(2)), dtype=float)
    F_total_mag = float(np.linalg.norm(F_total))
    
   
    min_dist = apf_ai.get_min_obstacle_distance(robot_pos)

    # ---------- Progress Monitoring (Stagnation) ----------
    prev = stuck_state.get("prev_goal_dist")
    progress = 0.0 if prev is None else float(prev - d_goal)
    stuck_state["prev_goal_dist"] = d_goal
    stuck_state["progress"] = progress

    prog_window = int(stuck_state.get("prog_window", 80))
    prog_eps_m  = float(stuck_state.get("prog_window_eps_m", 0.05))

    d_hist = stuck_state.get("d_goal_hist") or []
    d_hist.append(d_goal)
    if len(d_hist) > prog_window: d_hist.pop(0)
    stuck_state["d_goal_hist"] = d_hist

    if len(d_hist) >= 2:
        window_progress = float(d_hist[0] - d_hist[-1])
    else:
        window_progress = 0.0
    
    stuck_state["window_progress"] = window_progress
    no_progress_window = abs(window_progress) < prog_eps_m

    # ---------- Drift Calculation (Position & Rotation) ----------
    drift_window = int(stuck_state.get("drift_window", 30))
    
    # Position History
    pos_hist = stuck_state.get("pos_hist") or []
    pos_hist.append(robot_pos.copy())
    if len(pos_hist) > drift_window: pos_hist.pop(0)
    stuck_state["pos_hist"] = pos_hist

    drift = float(np.linalg.norm(pos_hist[-1] - pos_hist[0])) if len(pos_hist) >= 2 else 0.0
    stuck_state["drift"] = drift
    
    # Yaw History for Rotation Detection
    yaw_hist = stuck_state.get("yaw_hist") or []
    yaw_hist.append(current_yaw)
    if len(yaw_hist) > drift_window: yaw_hist.pop(0)
    stuck_state["yaw_hist"] = yaw_hist
    
    # Yaw Drift
    if len(yaw_hist) >= 2:
        yaw_drift = float(np.max(yaw_hist) - np.min(yaw_hist))
    else:
        yaw_drift = 0.0
    stuck_state["yaw_drift"] = yaw_drift

    # ---------- Mechanism Detection ----------
    
    # Mechanism A: Physical Stall 
    is_physical_stall = (F_total_mag > 0.5) and \
                        (actual_speed < 0.02) and \
                        (min_dist < 0.5)

    # Mechanism B: Predictive Trap 
    is_predictive_trap = False
    if F_total_mag > 0.1 and min_dist < 0.8: 
        direction = F_total / F_total_mag
        lookahead_point = robot_pos + direction * 0.6 
        if hasattr(apf_ai, 'is_point_blocked'):
            if apf_ai.is_point_blocked(lookahead_point, safety_margin=0.15):
                is_predictive_trap = True

    # Mechanism C: Temporal Stagnation 
    
    is_position_stuck = (drift < 0.2)
    
    is_rotation_stuck = (yaw_drift < 0.3)
    
    is_temporal_stagnation = no_progress_window and is_position_stuck and is_rotation_stuck
    
    is_rotating_recovery = no_progress_window and is_position_stuck and not is_rotation_stuck

    # --- Risk Scoring & Cause Identification ---
    risk_raw = 0.0
    current_cause = "NONE"

    if is_physical_stall:
        risk_raw = 0.95
        current_cause = "STALL" 
        
    elif is_predictive_trap:
        risk_raw = max(risk_raw, 0.85)
        if risk_raw == 0.85: 
            current_cause = "TRAP" 
        
    elif is_temporal_stagnation:
        risk_raw = max(risk_raw, 0.6)
        if risk_raw == 0.6: 
            current_cause = "STAGNATION"
            
    elif is_rotating_recovery:
        current_cause = "ROTATING" 

    stuck_state["cause"] = current_cause

    # ---------- Risk Smoothing ----------
    risk_prev   = float(stuck_state.get("risk_level", 0.0))
    rise_gain   = float(stuck_state.get("risk_rise_gain", 0.15))
    decay_gain  = float(stuck_state.get("risk_decay_gain", 0.03))

    if risk_raw > risk_prev:
        risk = risk_prev + rise_gain * (risk_raw - risk_prev)
    else:
        risk = risk_prev + decay_gain * (risk_raw - risk_prev)

    risk = float(np.clip(risk, 0.0, 1.0))
    stuck_state["risk_level"] = risk

    # ---------- High Risk Timer ----------
    high_risk_timer = stuck_state.get("high_risk_timer", 0.0)
    dt = 0.01

    if risk > 0.75:
        high_risk_timer += dt
    else:
        high_risk_timer = max(0.0, high_risk_timer - dt * 3.0)

    stuck_state["high_risk_timer"] = high_risk_timer

    # 8. Stuck Flag
    if high_risk_timer > 2:
        stuck_state["is_stuck"] = True
    elif high_risk_timer < 0.5:
        stuck_state["is_stuck"] = False


# ================================
#  ARBITRATION FUNCTION 
# ================================
def arbitration_blend(
    vx_h, vy_h, w_h,    # Human inputs
    A, B,               # Button inputs
    vx_ai, vy_ai, w_ai, # AI inputs
    min_dist,           # Distance to nearest obstacle
    stuck_state,        # State dictionary
):
    # ---------- Detect Human Presence ----------
    human_active = (
        abs(vx_h) > 0.05 or
        abs(vy_h) > 0.05 or
        abs(w_h)  > 0.05 or
        A == 1 or B == 1
    )

    risk = float(stuck_state.get("risk_level", 0.0))
    timer = float(stuck_state.get("high_risk_timer", 0.0))
    critical_lock = stuck_state.get("critical_lock", False)
    
    cause_str = stuck_state.get("cause", "NONE")

    # =======================================================
    # Critical Lock 
    # =======================================================
    if critical_lock:
        if human_active:
            stuck_state["critical_lock"] = False
            stuck_state["high_risk_timer"] = 0.0
            stuck_state["human_rescue_active"] = True
            stuck_state["rescue_until"] = time.time() + 3.0
            return vx_h, vy_h, w_h, "[UNLOCK] Resetting..."
        else:
            return 0.0, 0.0, 0.0, f"[LOCKED] Cause: {cause_str}"

    # =======================================================
    # Auto-Reverse 
    # =======================================================
    if timer > 2.0 and not human_active:
        if timer > 4.0:
            stuck_state["critical_lock"] = True
            return 0.0, 0.0, 0.0, "[GIVE UP] Locking..."
        
        return -0.8, 0.0, 0.0, f"[REFLEX] Rev ({cause_str})"

    # =======================================================
    # Rescue Mode 
    # =======================================================
    now = time.time()
    rescue_active = bool(stuck_state.get("human_rescue_active", False))
    rescue_until  = float(stuck_state.get("rescue_until", 0.0))
    
    if risk > 0.7 and human_active:
        rescue_active = True
        rescue_until = now + 3.0 
    
    if rescue_active and now > rescue_until:
        if risk < 0.3 and min_dist > 0.6:
            rescue_active = False
        else:
            rescue_until = now + 0.5 

    stuck_state["human_rescue_active"] = rescue_active
    stuck_state["rescue_until"]        = rescue_until

    # =======================================================
    # Alpha 
    # =======================================================
    far_dist  = float(stuck_state.get("far_dist_no_obstacle", 1.5))
    near_dist = float(stuck_state.get("near_dist_strong_ai", 0.7))

    if min_dist > far_dist:
        base_ai = 0.0 if human_active else 1.0
    elif min_dist < near_dist:
        base_ai = 0.9 
    else:
        ratio = (far_dist - min_dist) / (far_dist - near_dist)
        ratio = float(np.clip(ratio, 0.0, 1.0))
        base_ai = 0.5 * ratio if human_active else 1.0

    # Risk Penalty
    alpha_target = base_ai * (1.0 - risk)

    # Rescue Clamp
    if rescue_active:
        alpha_target = min(alpha_target, 0.1) 

    alpha_target = float(np.clip(alpha_target, 0.0, 1.0))

    # Smoothing
    alpha_prev  = float(stuck_state.get("alpha_prev", alpha_target))
    alpha_gain  = float(stuck_state.get("alpha_smooth_gain", 0.1))
    alpha = alpha_prev + alpha_gain * (alpha_target - alpha_prev)
    alpha = float(np.clip(alpha, 0.0, 1.0))

    stuck_state["alpha_prev"] = alpha
    stuck_state["alpha"] = alpha 

    # Blend
    vx = alpha * vx_ai + (1.0 - alpha) * vx_h
    vy = alpha * vy_ai + (1.0 - alpha) * vy_h
    w  = alpha * w_ai  + (1.0 - alpha) * w_h

    # Logging Mode Update
    if rescue_active:
        mode = f"[RESCUE] {cause_str} (α={alpha:.2f})"
    else:
        if min_dist > far_dist: zone = "Far"
        elif min_dist < near_dist: zone = "Near"
        else: zone = "Mid"
        
        if risk > 0.3:
            mode = f"[{zone}] Risk:{cause_str} (α={alpha:.2f})"
        elif alpha > 0.7:
            mode = f"[{zone}] AI-Auto (α={alpha:.2f})"
        else:
            mode = f"[{zone}] Human (α={alpha:.2f})"

    return vx, vy, w, mode






# ================================
#               MAIN
# ================================


def main():

    debug_log = {
        "t": [],
        "alpha": [],
        "risk_level": [],
        "min_dist": [],
        "F_total_mag": [],
        "high_risk_timer": [],
        
        "vel_ai": [],       
        "vel_human": [],    
        "vel_final": [],    
        
        "mode_raw": [],   
        "cause_raw": []   
    }

    xml_filename = 'obstacle_scene.xml'
    xml_path = os.path.join(ASSETS_PATH, 'mujoco', xml_filename)

    human_queue = mp.Queue(maxsize=1)
    p = mp.Process(target=joystick_process, args=(human_queue,))
    p.start()

    apf_ai = APFController(xml_path=xml_path)
    
    visualizer = ArbitrationVisualizer() 

    robot = Go2Sim(mode='highlevel', render=True, xml_path=xml_path, dt=0.01)

    goal = get_goal_from_scene(robot)
    robot.standUpReset()
    set_viewer_camera(robot, azimuth=15, elevation=-30, distance=10.0)

    step_count = 0
    risk_timer_accumulator = 0.0

    stuck_state = {
        "prev_goal_dist": None, "pos_hist": None, "d_goal_hist": None,
        "drift_eps": 0.3, "drift_window": 30,
        "prog_window": 80, "prog_window_eps_m": 0.05,
        "alpha_prev": 1.0, "alpha_smooth_gain": 0.10,
        "cos_window": 80, "cos_thr_neg": -0.4,
        "risk_rise_gain": 0.15, "risk_decay_gain": 0.03,
        "is_stuck": False, "cause": "NONE", # Init cause
        "human_rescue_active": False, "rescue_until": 0.0, 
        "critical_lock": False 
    }

    try:
        while True:
            cmd = {"A": 0, "B": 0, "vx": 0.0, "vy": 0.0, "yaw": 0.0}

            if not human_queue.empty():
                cmd = human_queue.get()
                vx_h, vy_h, w_h = cmd["vx"], cmd["vy"], cmd["yaw"]
                if cmd["A"]: robot.standUpReset()
                if cmd["B"]: robot.sitDownReset()
            else:
                vx_h = vy_h = w_h = 0.0

            # AI Control
            vx_ai, vy_ai, w_ai = apf_ai.control(robot, goal)

            # Sensors
            pos, _ = robot.getPose()
            min_dist = apf_ai.get_min_obstacle_distance(pos[:2])
            
            # Logic Update
            update_stuck_state(goal, robot, apf_ai, stuck_state)

            # Arbitration
            vx_final, vy_final, w_final, mode = arbitration_blend(
                vx_h, vy_h, w_h, cmd["A"], cmd["B"],
                vx_ai, vy_ai, w_ai, min_dist, stuck_state
            )

            # Visuals
            visualizer.visualize_all(robot, apf_ai, stuck_state, mode, human_cmd=cmd)

            # Act
            robot.stepHighlevel(vx_final, vy_final, w_final)
            
            # === LOGGING ===
            
            v_ai_mag = np.sqrt(vx_ai**2 + vy_ai**2)
            v_h_mag = np.sqrt(vx_h**2 + vy_h**2)
            v_final_mag = np.sqrt(vx_final**2 + vy_final**2)
            
            debug_info = apf_ai.get_debug_info()
            f_att = np.array(debug_info.get("F_attractive", [0.0, 0.0]))
            f_rep = np.array(debug_info.get("F_repulsive", [0.0, 0.0]))
            f_total_val = np.linalg.norm(f_att + f_rep)

            current_risk = stuck_state.get("risk_level", 0.0)
            current_timer = stuck_state.get("high_risk_timer", 0.0)

 
            debug_log["t"].append(step_count * 0.01)
            debug_log["alpha"].append(stuck_state.get("alpha", 0.0))
            debug_log["risk_level"].append(current_risk)
            debug_log["min_dist"].append(min_dist)
            debug_log["F_total_mag"].append(f_total_val)
            debug_log["high_risk_timer"].append(current_timer)
            
            debug_log["vel_ai"].append(v_ai_mag)
            debug_log["vel_human"].append(v_h_mag)
            debug_log["vel_final"].append(v_final_mag)
            

            debug_log["mode_raw"].append(str(mode))
            debug_log["cause_raw"].append(str(stuck_state.get("cause", "NONE")))

            # Check Goal
            if apf_ai.is_goal_reached(robot, goal):
                print("\nGOAL REACHED!")
                break

            if step_count % 100 == 0:
                print(f"Step {step_count:4d} | Risk: {current_risk:.2f} | Mode: {mode}")

            step_count += 1
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nExiting...")
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if 'p' in locals() and p.is_alive(): p.terminate()
        if 'visualizer' in locals(): visualizer.stop()
        
        # Plot only if we have data
        if len(debug_log["t"]) > 0:
            print(f"[INFO] Plotting data to arbitration_debug.png...")
            plot_debug(debug_log)




if __name__ == "__main__":
    main()