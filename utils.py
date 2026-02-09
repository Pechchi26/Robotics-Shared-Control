import numpy as np
import mujoco
import matplotlib.pyplot as plt


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




def plot_debug(debug_log):
    """
    plots:
      1. Arbitration State & Risk Causes
      2. Velocity Arbitration (Input vs. Output)
      3. Physical Stall Diagnosis (High Force + Low Speed)
      4. Trigger Logic (Timer > 2s -> Reflex)
    """
    

    t = np.array(debug_log["t"])
    alpha = np.array(debug_log["alpha"])
    risk = np.array(debug_log["risk_level"])
    

    modes = np.array(debug_log["mode_raw"])  
    causes = np.array(debug_log["cause_raw"]) 
    

    is_reflex = np.array(["REFLEX" in str(m) for m in modes])
    is_locked = np.array(["LOCKED" in str(m) or "GIVE UP" in str(m) for m in modes])
    is_rescue = np.array(["RESCUE" in str(m) for m in modes])
    
  
    v_ai = np.array(debug_log["vel_ai"])
    v_human = np.array(debug_log["vel_human"])
    v_final = np.array(debug_log["vel_final"])
    f_total = np.array(debug_log["F_total_mag"])
    timer = np.array(debug_log["high_risk_timer"])


    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(16, 18), sharex=True)

    # =================================================================
    # Plot 1: System State & Risk Causes 
    # =================================================================
    ax1.set_title("1. Arbitration State & Risk Factors", fontsize=14, fontweight='bold')
    
 
    ax1.plot(t, alpha, label="Alpha (Human Authority)", color='blue', linewidth=2.5)
    ax1.plot(t, risk, label="Risk Level", color='red', linestyle='--', linewidth=2)
    

    ax1.fill_between(t, 0, 1.2, where=is_reflex, color='#9b59b6', alpha=0.5, label="[REFLEX] Auto-Reverse")

    ax1.fill_between(t, 0, 1.2, where=is_rescue, color='#2ecc71', alpha=0.3, label="[RESCUE] Human Active")

    ax1.fill_between(t, 0, 1.2, where=is_locked, color='#e74c3c', alpha=0.5, label="[LOCKED] System Halt")


    prev_cause = "NONE"
    last_text_x = -100 
    
    for i in range(len(t)):
        curr_cause = str(causes[i])
        curr_risk = risk[i]
        
        if curr_risk > 0.5 and curr_cause != "NONE":
            if (curr_cause != prev_cause) or (t[i] - last_text_x > 2.0):
                ax1.text(t[i], 1.05, f"{curr_cause}", 
                         fontsize=10, color='darkred', fontweight='bold', 
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
                last_text_x = t[i]
        
        prev_cause = curr_cause

    ax1.set_ylim(-0.1, 1.3)
    ax1.set_ylabel("State / Alpha")
    ax1.legend(loc="upper left", ncol=4, fontsize=10, frameon=True)
    ax1.grid(True, alpha=0.3)

    # =================================================================
    # Plot 2: Velocity Analysis (Who is driving?)
    # =================================================================
    ax2.set_title("2. Velocity Arbitration (Input vs. Output)", fontsize=12, fontweight='bold')
    
    ax2.plot(t, v_ai, color='blue', linestyle=':', label="AI Input Speed", alpha=0.6)
    ax2.plot(t, v_human, color='orange', linestyle='--', label="Human Input Speed", alpha=0.8)
    ax2.plot(t, v_final, color='black', linewidth=2, label="Final Output Speed")
    

    ax2.fill_between(t, -0.5, 1.5, where=is_reflex, color='#9b59b6', alpha=0.2, label="Reflex Active")
    
    ax2.set_ylabel("Speed (m/s)")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    # =================================================================
    # Plot 3: Force Diagnosis (Why did we Stall?)
    # =================================================================
    ax3.set_title("3. Physical Stall Diagnosis (High Force + Low Speed)", fontsize=12, fontweight='bold')
    
    color_f = 'tab:pink'
    ax3.set_ylabel('Total Force', color=color_f, fontweight='bold')
    ax3.plot(t, f_total, color=color_f, label="Force Magnitude", linewidth=1.5)
    ax3.axhline(0.5, color=color_f, linestyle=':', label="Stall Threshold (0.5)")
    ax3.tick_params(axis='y', labelcolor=color_f)

    ax3b = ax3.twinx()
    color_v = 'tab:cyan'
    ax3b.set_ylabel('Actual Speed', color=color_v, fontweight='bold')
    ax3b.plot(t, v_final, color=color_v, label="Robot Speed", linewidth=1.5)
    ax3b.tick_params(axis='y', labelcolor=color_v)
    

    is_stall = np.array(["STALL" in str(c) for c in causes])
    ax3.fill_between(t, 0, 5, where=is_stall, color='red', alpha=0.1, label="STALL Detected")
    
    lines3, labels3 = ax3.get_legend_handles_labels()
    lines3b, labels3b = ax3b.get_legend_handles_labels()
    ax3.legend(lines3 + lines3b, labels3 + labels3b, loc="upper left")
    ax3.grid(True, alpha=0.3)

    # =================================================================
    # Plot 4: Timer Logic (The Trigger)
    # =================================================================
    ax4.set_title("4. Trigger Logic (Timer > 2s -> Reflex)", fontsize=12, fontweight='bold')
    
    ax4.plot(t, timer, color='orange', linewidth=2.5, label="High Risk Timer")
    
    ax4.axhline(2.0, color='purple', linestyle='--', linewidth=2, label="Threshold: 2.0s (Trigger Reflex)")
    ax4.axhline(4.0, color='red', linestyle='--', linewidth=2, label="Threshold: 4.0s (Trigger Lock)")
    

    ax4.fill_between(t, 0, 5, where=(timer > 2.0), color='purple', alpha=0.1)

    ax4.set_ylabel("Timer (s)")
    ax4.set_xlabel("Time (s)", fontsize=12)
    ax4.legend(loc="upper left")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("arbitration_debug.png", dpi=200) 
    print("[INFO] Plot saved. Check arbitration_debug.png")
    plt.close()





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

    # Pick lookat target
    if lookat is None:
        pos, _ = robot.getPose()
        # pos is usually [x,y,z] or [x,y,theta]; adapt defensively
        x = float(pos[0])
        y = float(pos[1])
        z = float(pos[2]) if len(pos) >= 3 else 0.4
        lookat = np.array([x, y, z], dtype=float)
    else:
        lookat = np.array(lookat, dtype=float)

    # Apply camera
    try:
        v.cam.azimuth = float(azimuth)
        v.cam.elevation = float(elevation)
        v.cam.distance = float(distance)
        v.cam.lookat[:] = lookat
    except Exception as e:
        # If the viewer/cam API differs in your Go2Py version, this prevents crashing
        print(f"[WARN] Could not set viewer camera: {e}")


    return np.sum(distances)
