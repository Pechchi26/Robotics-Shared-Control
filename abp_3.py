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
from utils import get_goal_from_scene, calculate_path_length
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from arbitration_visualizer_full import ArbitrationVisualizer


# ============================================================
# Encoders for discrete plots (NO effect on control)
# ============================================================
CAUSE_TO_ID = {"NONE": 0, "STALL": 1, "TRAP": 2, "STAGNATION": 3, "ROTATING": 4}
MODE_TO_ID  = {"HUMAN": 0, "AI": 1, "RISK": 2, "RESCUE": 3, "REFLEX": 4, "LOCKED": 5}

def mode_string_to_id(mode_str: str) -> int:
    """Convert your mode string into a numeric id for plotting (NO effect on control)."""
    if not isinstance(mode_str, str):
        return MODE_TO_ID["HUMAN"]
    s = mode_str.upper()
    if "LOCKED" in s or "GIVE UP" in s:
        return MODE_TO_ID["LOCKED"]
    if "REFLEX" in s or "REV" in s:
        return MODE_TO_ID["REFLEX"]
    if "RESCUE" in s:
        return MODE_TO_ID["RESCUE"]
    if "RISK" in s:
        return MODE_TO_ID["RISK"]
    if "AI-AUTO" in s:
        return MODE_TO_ID["AI"]
    return MODE_TO_ID["HUMAN"]


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


# ============================================================
#  CORE: RISK / STUCK DETECTION (UNCHANGED)
# ============================================================
def update_stuck_state(goal, robot, apf_ai, stuck_state):
    """
    Risk Evaluation (Refined v2.1):
    Identifies 3 distinct trap mechanisms:
    A. Physical Stall
    B. Predictive Trap
    C. Temporal Stagnation (vs rotating recovery)
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

    prev = stuck_state.get("prev_goal_dist")
    progress = 0.0 if prev is None else float(prev - d_goal)
    stuck_state["prev_goal_dist"] = d_goal
    stuck_state["progress"] = progress

    prog_window = int(stuck_state.get("prog_window", 80))
    prog_eps_m = float(stuck_state.get("prog_window_eps_m", 0.05))

    d_hist = stuck_state.get("d_goal_hist") or []
    d_hist.append(d_goal)
    if len(d_hist) > prog_window:
        d_hist.pop(0)
    stuck_state["d_goal_hist"] = d_hist

    window_progress = float(d_hist[0] - d_hist[-1]) if len(d_hist) >= 2 else 0.0
    stuck_state["window_progress"] = window_progress
    no_progress_window = abs(window_progress) < prog_eps_m

    drift_window = int(stuck_state.get("drift_window", 30))

    pos_hist = stuck_state.get("pos_hist") or []
    pos_hist.append(robot_pos.copy())
    if len(pos_hist) > drift_window:
        pos_hist.pop(0)
    stuck_state["pos_hist"] = pos_hist

    drift = float(np.linalg.norm(pos_hist[-1] - pos_hist[0])) if len(pos_hist) >= 2 else 0.0
    stuck_state["drift"] = drift

    yaw_hist = stuck_state.get("yaw_hist") or []
    yaw_hist.append(current_yaw)
    if len(yaw_hist) > drift_window:
        yaw_hist.pop(0)
    stuck_state["yaw_hist"] = yaw_hist

    yaw_drift = float(np.max(yaw_hist) - np.min(yaw_hist)) if len(yaw_hist) >= 2 else 0.0
    stuck_state["yaw_drift"] = yaw_drift

    is_physical_stall = (F_total_mag > 0.5) and (actual_speed < 0.02) and (min_dist < 0.5)

    is_predictive_trap = False
    if F_total_mag > 0.1 and min_dist < 0.8:
        direction = F_total / (F_total_mag + 1e-9)
        lookahead_point = robot_pos + direction * 0.6
        if hasattr(apf_ai, 'is_point_blocked'):
            if apf_ai.is_point_blocked(lookahead_point, safety_margin=0.15):
                is_predictive_trap = True

    is_position_stuck = (drift < 0.2)
    is_rotation_stuck = (yaw_drift < 0.3)

    is_temporal_stagnation = no_progress_window and is_position_stuck and is_rotation_stuck
    is_rotating_recovery   = no_progress_window and is_position_stuck and not is_rotation_stuck

    risk_raw = 0.0
    current_cause = "NONE"

    if is_physical_stall:
        risk_raw = 0.90
        current_cause = "STALL"
    elif is_predictive_trap:
        risk_raw = max(risk_raw, 0.75)
        if risk_raw == 0.75:
            current_cause = "TRAP"
    elif is_temporal_stagnation:
        risk_raw = max(risk_raw, 0.5)
        if risk_raw == 0.5:
            current_cause = "STAGNATION"
    elif is_rotating_recovery:
        current_cause = "ROTATING"

    stuck_state["cause"] = current_cause

    risk_prev = float(stuck_state.get("risk_level", 0.0))
    rise_gain = float(stuck_state.get("risk_rise_gain", 0.15))
    decay_gain = float(stuck_state.get("risk_decay_gain", 0.03))

    if risk_raw > risk_prev:
        risk = risk_prev + rise_gain * (risk_raw - risk_prev)
    else:
        risk = risk_prev + decay_gain * (risk_raw - risk_prev)

    risk = float(np.clip(risk, 0.0, 1.0))
    stuck_state["risk_level"] = risk

    high_risk_timer = stuck_state.get("high_risk_timer", 0.0)
    dt = 0.01

    if risk > 0.75:
        high_risk_timer += dt
    else:
        high_risk_timer = max(0.0, high_risk_timer - dt * 3.0)

    stuck_state["high_risk_timer"] = high_risk_timer

    if high_risk_timer > 2.0:
        stuck_state["is_stuck"] = True
    elif high_risk_timer < 0.5:
        stuck_state["is_stuck"] = False


# ============================================================
#  CORE: ARBITRATION (UNCHANGED)
# ============================================================
def arbitration_blend(
    vx_h, vy_h, w_h,
    A, B,
    vx_ai, vy_ai, w_ai,
    min_dist,
    stuck_state,
):
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

    if critical_lock:
        if human_active:
            stuck_state["critical_lock"] = False
            stuck_state["high_risk_timer"] = 0.0
            stuck_state["human_rescue_active"] = True
            stuck_state["rescue_until"] = time.time() + 3.0
            return vx_h, vy_h, w_h, "[UNLOCK] Resetting..."
        else:
            return 0.0, 0.0, 0.0, f"[LOCKED] Cause: {cause_str}"

    if timer > 2.0 and not human_active:
        if timer > 4.0:
            stuck_state["critical_lock"] = True
            return 0.0, 0.0, 0.0, "[GIVE UP] Locking..."
        return -0.8, 0.0, 0.0, f"[REFLEX] Rev ({cause_str})"

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

    alpha_target = base_ai * (1.0 - risk)

    if rescue_active:
        alpha_target = min(alpha_target, 0.1)

    alpha_target = float(np.clip(alpha_target, 0.0, 1.0))

    alpha_prev  = float(stuck_state.get("alpha_prev", alpha_target))
    alpha_gain  = float(stuck_state.get("alpha_smooth_gain", 0.1))
    alpha = alpha_prev + alpha_gain * (alpha_target - alpha_prev)
    alpha = float(np.clip(alpha, 0.0, 1.0))

    stuck_state["alpha_prev"] = alpha
    stuck_state["alpha"] = alpha

    vx = alpha * vx_ai + (1.0 - alpha) * vx_h
    vy = alpha * vy_ai + (1.0 - alpha) * vy_h
    w  = alpha * w_ai  + (1.0 - alpha) * w_h

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


# ============================================================
# Poster plot 1:
#   Mode timeline (AI/HUMAN/RISK/RESCUE/REFLEX/LOCKED)
#   + highlight critical lock, rescue, auto reverse
# ============================================================
def poster_plot_modes(debug_log, out_png="poster_modes_timeline.png"):
    """
    Poster plot 1:
      - y-axis ONLY: HUMAN vs AI (0/1)
      - overlays (shaded): RESCUE / AUTO-REVERSE(REFLEX) / CRITICAL LOCK
    """
    t = np.asarray(debug_log["t"], dtype=float)

    # Current numeric mode id from your logging
    mode_id = np.asarray(debug_log["mode_id"], dtype=float)

    # Convert the full mode_id into a 2-state timeline:
    #   HUMAN = 0
    #   AI    = 1
    # Rule:
    #   - If mode is AI or RISK or REFLEX or LOCKED -> treat as AI-side (1)
    #   - If mode is HUMAN or RESCUE -> treat as HUMAN-side (0)
    #
    # Why: "RISK" means blending but still AI-influenced; REFLEX/LOCKED are AI safety behaviors.
    #      RESCUE is explicitly "human rescue" -> show as HUMAN on the main y-axis.
    ai_like = (mode_id == MODE_TO_ID["AI"]) | (mode_id == MODE_TO_ID["RISK"]) | \
              (mode_id == MODE_TO_ID["REFLEX"]) | (mode_id == MODE_TO_ID["LOCKED"])
    human_like = (mode_id == MODE_TO_ID["HUMAN"]) | (mode_id == MODE_TO_ID["RESCUE"])

    # default to human if unknown
    human_ai = np.zeros_like(mode_id)
    human_ai[ai_like] = 1.0
    human_ai[human_like] = 0.0

    rescue = np.asarray(debug_log["rescue_active"], dtype=float)
    lock   = np.asarray(debug_log["critical_lock"], dtype=float)
    reflex = np.asarray(debug_log["reflex_active"], dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(16, 4))

    ax.set_title("Mode timeline: Human vs AI (with Rescue / Auto-Reverse / Critical Lock overlays)")
    ax.step(t, human_ai, where="post", linewidth=2.5)

    # y-axis ONLY: HUMAN and AI
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["HUMAN", "AI"])
    ax.set_ylim(-0.2, 1.2)

    # Overlays for special phases (do not affect y-axis categories)
    ax.fill_between(t, -0.2, 1.2, where=(rescue > 0.5), alpha=0.10, label="RESCUE active")
    ax.fill_between(t, -0.2, 1.2, where=(reflex > 0.5), alpha=0.10, label="AUTO-REVERSE (REFLEX)")
    ax.fill_between(t, -0.2, 1.2, where=(lock > 0.5),   alpha=0.15, label="CRITICAL LOCK")

    ax.set_xlabel("time [s]")
    ax.set_ylabel("controller")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[SUCCESS] Saved {out_png}")



# ============================================================
# Poster plot 2:
#   Alpha + Risk time series
#   + highlight causes: TRAP / STALL / STAGNATION
# ============================================================
def poster_plot_alpha_risk_causes(debug_log, out_png="poster_alpha_risk_causes.png"):
    t = np.asarray(debug_log["t"], dtype=float)
    alpha = np.asarray(debug_log["alpha"], dtype=float)
    risk  = np.asarray(debug_log["risk_level"], dtype=float)

    cause_id = np.asarray(debug_log["cause_id"], dtype=float)

    # Boolean masks per cause
    stall = (cause_id == CAUSE_TO_ID["STALL"])
    trap  = (cause_id == CAUSE_TO_ID["TRAP"])
    stag  = (cause_id == CAUSE_TO_ID["STAGNATION"])

    fig, ax = plt.subplots(1, 1, figsize=(16, 4))

    ax.set_title("Control factors: alpha (AI authority) vs risk, with detection causes (stall/trap/stagnation)")
    ax.plot(t, alpha, label="alpha (AI authority)", linewidth=2.5)
    ax.plot(t, risk,  label="risk factor r", linestyle="--", linewidth=2.0)

    # Highlight cause intervals (background bands)
    ax.fill_between(t, 0, 1, where=stall, alpha=0.12, label="STALL detected")
    ax.fill_between(t, 0, 1, where=trap,  alpha=0.10, label="TRAP detected")
    ax.fill_between(t, 0, 1, where=stag,  alpha=0.10, label="STAGNATION detected")

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("0..1")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[SUCCESS] Saved {out_png}")


def save_debug_csv(debug_log, path="shared_control_log.csv"):
    import csv
    if "t" not in debug_log or len(debug_log["t"]) == 0:
        print("[WARN] No debug data to save.")
        return
    keys = list(debug_log.keys())
    n = len(debug_log["t"])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(n):
            w.writerow([debug_log[k][i] for k in keys])
    print(f"[SUCCESS] Wrote {path}")


# ================================
#               MAIN
# ================================
def main():
    # Debug log (ONLY for plots)
    debug_log = {
        "t": [],
        "alpha": [],
        "risk_level": [],
        "min_dist": [],
        "is_stuck": [],
        "cause_id": [],
        "mode_id": [],
        "mode_str": [],

        # flags for poster plot 1
        "rescue_active": [],
        "critical_lock": [],
        "reflex_active": [],

        # optional (not required for the two plots, but useful)
        "vx_h": [], "vy_h": [], "w_h": [],
        "vx_ai": [], "vy_ai": [], "w_ai": [],
        "vx_final": [], "vy_final": [], "w_final": [],
    }

    xml_filename = 'u_shape_scene.xml'
    xml_path = os.path.join(ASSETS_PATH, 'mujoco', xml_filename)

    human_queue = mp.Queue(maxsize=1)
    p = mp.Process(target=joystick_process, args=(human_queue,))
    p.start()

    apf_ai = APFController(xml_path=xml_path)
    visualizer = ArbitrationVisualizer()

    robot = Go2Sim(
        mode='highlevel',
        render=True,
        xml_path=xml_path,
        dt=0.01
    )

    goal = get_goal_from_scene(robot)
    start_time = time.time()

    robot.standUpReset()

    set_viewer_camera(robot, azimuth=15, elevation=-30, distance=10.0)

    step_count = 0

    stuck_state = {
        "prev_goal_dist": None,
        "pos_hist": None,
        "d_goal_hist": None,

        "drift_eps": 0.3,
        "drift_window": 30,
        "prog_window": 80,
        "prog_window_eps_m": 0.05,

        "alpha_prev": 1.0,
        "alpha_smooth_gain": 0.10,

        "risk_rise_gain": 0.15,
        "risk_decay_gain": 0.03,

        "cos_hist": None,
        "is_stuck": False,

        # rescue latch
        "human_rescue_active": False,
        "rescue_until": 0.0,

        # lock
        "critical_lock": False,
    }

    try:
        while True:
            cmd = {"A": 0, "B": 0, "vx": 0.0, "vy": 0.0, "yaw": 0.0}

            if not human_queue.empty():
                cmd = human_queue.get()
                vx_h, vy_h, w_h = cmd["vx"], cmd["vy"], cmd["yaw"]

                if cmd["A"]:
                    robot.standUpReset()
                if cmd["B"]:
                    robot.sitDownReset()
            else:
                vx_h = vy_h = w_h = 0.0

            vx_ai, vy_ai, w_ai = apf_ai.control(robot, goal)

            pos, _ = robot.getPose()
            robot_pos = pos[:2]
            min_dist = apf_ai.get_min_obstacle_distance(robot_pos)

            # CORE update (unchanged)
            update_stuck_state(goal, robot, apf_ai, stuck_state)

            # CORE arbitration (unchanged)
            vx_final, vy_final, w_final, mode = arbitration_blend(
                vx_h, vy_h, w_h,
                cmd["A"], cmd["B"],
                vx_ai, vy_ai, w_ai,
                min_dist,
                stuck_state
            )

            # ===== Logging ONLY (NO effect on core) =====
            sim_t = step_count * 0.01
            debug_log["t"].append(sim_t)

            debug_log["alpha"].append(float(stuck_state.get("alpha", 0.0)))
            debug_log["risk_level"].append(float(stuck_state.get("risk_level", 0.0)))
            debug_log["min_dist"].append(float(min_dist))
            debug_log["is_stuck"].append(int(bool(stuck_state.get("is_stuck", False))))

            cause = stuck_state.get("cause", "NONE")
            debug_log["cause_id"].append(int(CAUSE_TO_ID.get(cause, 0)))

            debug_log["mode_str"].append(str(mode))
            m_id = int(mode_string_to_id(mode))
            debug_log["mode_id"].append(m_id)

            # Poster plot flags
            debug_log["rescue_active"].append(int(bool(stuck_state.get("human_rescue_active", False))))
            debug_log["critical_lock"].append(int(bool(stuck_state.get("critical_lock", False))))
            debug_log["reflex_active"].append(1 if m_id == MODE_TO_ID["REFLEX"] else 0)

            # Optional command logs (not required for the 2 poster plots)
            debug_log["vx_h"].append(float(vx_h)); debug_log["vy_h"].append(float(vy_h)); debug_log["w_h"].append(float(w_h))
            debug_log["vx_ai"].append(float(vx_ai)); debug_log["vy_ai"].append(float(vy_ai)); debug_log["w_ai"].append(float(w_ai))
            debug_log["vx_final"].append(float(vx_final)); debug_log["vy_final"].append(float(vy_final)); debug_log["w_final"].append(float(w_final))
            # ===========================================

            visualizer.visualize_all(robot, apf_ai, stuck_state, mode, human_cmd=cmd)

            robot.stepHighlevel(vx_final, vy_final, w_final)
            time.sleep(0.02)

            if apf_ai.is_goal_reached(robot, goal):
                print("\n[INFO] Goal reached. Holding. Ctrl+C to exit.")
                while True:
                    robot.stepHighlevel(0, 0, 0)
                    time.sleep(0.02)

            if step_count % 100 == 0:
                print(f"Step {step_count:4d} | Risk: {stuck_state.get('risk_level',0.0):.2f} | Mode: {mode}")

            step_count += 1
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C received. Exiting...")

    except Exception as e:
        print(f"\n[ERROR] Runtime error: {e}")

    finally:
        print("\n" + "="*30)
        print("       CLOSING SESSION")
        print("="*30)

        if 'p' in locals() and p.is_alive():
            p.terminate()
            p.join()

        if 'visualizer' in locals():
            try:
                visualizer.stop()
            except:
                pass

        if 'debug_log' in locals() and len(debug_log["t"]) > 0:
            try:
                # Save CSV (optional)
                save_debug_csv(debug_log, path="shared_control_log.csv")

                # Poster plots (YOUR REQUIREMENT)
                poster_plot_modes(debug_log, out_png="poster_modes_timeline.png")
                poster_plot_alpha_risk_causes(debug_log, out_png="poster_alpha_risk_causes.png")

            except Exception as e:
                print(f"[ERROR] Failed to plot: {e}")
                import traceback
                traceback.print_exc()

        print("[INFO] Done.")


if __name__ == "__main__":
    main()
