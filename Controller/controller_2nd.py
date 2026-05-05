WIND = True
# WIND FLAG – set to True above to enable wind handling during marking

# =============================================================================
# AERO60492 Coursework 3 – Feedback Control
# Advanced method: Q-learning reinforcement learning with PID actuation
# =============================================================================

import atexit
import csv
import math
import os
import random
import threading
from datetime import datetime

# FIX: Use "Agg" (file-only) backend instead of "TkAgg" (GUI).
# Root cause of crash:
#   _PathTracker spawned a daemon thread that called plt.figure() and plt.pause().
#   On Windows, matplotlib GUI windows MUST be created on the main thread.
#   TkAgg attempted to create a Tk window from the daemon thread → crash.
# Solution:
#   Agg renders entirely in memory and writes to PNG files.  No GUI window,
#   no thread restrictions, no crash.  The live plot is replaced by a PNG
#   saved to logs/ alongside the CSV when the simulator exits.
# Ref: matplotlib docs – backend selection; Windows GUI threading restrictions.
try:
    import matplotlib
    matplotlib.use("Agg")   # File-only backend: no GUI, no thread, no crash.
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    _MATPLOTLIB_OK = True
except Exception:
    _MATPLOTLIB_OK = False

# -----------------------------------------------------------------------------
# Gain profiles
# Each row: (kp_xy, kp_z, kp_yaw, ki_xy, ki_z, ki_yaw, kd_xy, kd_z,
#            v_xy_max, v_z_max, yaw_rate_max)
# Naming convention: *_max = hard velocity cap; kp/ki/kd = PID gains.
# This matches the column comment above – consistent throughout the file.
# -----------------------------------------------------------------------------
_ACTION_PROFILES = (
    # -- Coarse approach (far from target) -------------------------------------
    # FIX: kp_xy raised (0.55→1.20) to saturate the ±1 m/s hardware cap quickly.
    # FIX: kd_xy lowered (0.50→0.10) – high kd was braking the drone during fast
    #      approach (large d_ex_b when closing at speed), cutting velocity in half
    #      before the drone was anywhere near the target.
    (1.20, 1.10, 0.80, 0.00, 0.00, 0.03, 0.10, 0.10, 2.50, 1.50, 1.40),  # 0
    (1.10, 1.00, 0.75, 0.00, 0.00, 0.03, 0.10, 0.10, 2.00, 1.20, 1.20),  # 1
    (1.00, 0.90, 0.70, 0.00, 0.00, 0.04, 0.10, 0.10, 1.50, 1.00, 1.00),  # 2
    (0.90, 0.80, 0.65, 0.00, 0.00, 0.04, 0.10, 0.10, 1.20, 0.80, 0.90),  # 3
    # -- Mid-range (0.5 m – 1.5 m) --------------------------------------------
    # FIX: kp_xy raised (0.40→0.85), kd_xy lowered (0.60→0.20).
    #      Drone now stays at full speed until 1.5 m, then begins braking.
    (0.85, 0.75, 1.20, 0.00, 0.00, 0.05, 0.20, 0.15, 0.80, 0.60, 1.40),  # 4
    (0.75, 0.65, 1.00, 0.00, 0.00, 0.05, 0.20, 0.15, 0.70, 0.55, 1.20),  # 5
    (0.65, 0.55, 0.85, 0.00, 0.00, 0.06, 0.20, 0.15, 0.60, 0.50, 1.00),  # 6
    (0.55, 0.45, 0.70, 0.00, 0.00, 0.06, 0.20, 0.15, 0.50, 0.45, 0.85),  # 7
    # -- Fine approach (< 0.5 m) -----------------------------------------------
    # kd raised here to give a smooth stop in the last 0.5 m.
    (0.45, 0.45, 1.50, 0.01, 0.01, 0.09, 0.40, 0.30, 0.25, 0.20, 1.50),  # 8
    (0.40, 0.40, 1.40, 0.01, 0.01, 0.10, 0.38, 0.28, 0.20, 0.15, 1.40),  # 9
    (0.35, 0.35, 1.30, 0.01, 0.01, 0.11, 0.35, 0.25, 0.15, 0.12, 1.30),  # 10
    (0.30, 0.30, 1.20, 0.01, 0.01, 0.12, 0.32, 0.22, 0.12, 0.10, 1.20),  # 11
    # -- Hover-hold (at target) ------------------------------------------------
    (0.35, 0.35, 0.50, 0.02, 0.02, 0.00, 0.55, 0.45, 0.10, 0.10, 0.25),  # 12
    (0.30, 0.30, 0.45, 0.02, 0.02, 0.00, 0.50, 0.42, 0.08, 0.08, 0.20),  # 13
    (0.25, 0.25, 0.40, 0.02, 0.02, 0.00, 0.45, 0.40, 0.06, 0.06, 0.15),  # 14
    (0.20, 0.20, 0.35, 0.02, 0.02, 0.00, 0.40, 0.38, 0.04, 0.04, 0.10),  # 15
)

_POS_TOL    = 0.08   # m   - position tolerance for dwell-hold phase
_YAW_TOL    = 0.06   # rad - yaw tolerance for dwell-hold phase
_HOLD_STEPS = 20     # consecutive in-tolerance steps to trigger hold bonus


# =============================================================================
# Data logger
# =============================================================================
class _FlightLogger:
    """Lightweight per-call data accumulator with atexit CSV + PNG export."""

    _LOG_DIR = "logs"

    def __init__(self):
        self._rows  = []
        self._t     = 0.0
        self._fname = None
        # Position history lists for the path PNG saved on exit.
        self._xs, self._ys, self._zs = [], [], []
        self._last_target_xyz = (2.0, 2.0, 2.0)   # updated each call
        atexit.register(self._export)

    def log(self, dt, pos, euler, vel_cmd, target, wind, profile_idx):
        # FIX (naming): parameter was 'vel_world' – renamed to 'vel_cmd'
        # throughout because we log the velocity COMMAND (vx,vy,vz outputs),
        # not the actual world velocity from pybullet.  Using 'vel_cmd' makes
        # the distinction clear and matches how it is described in the CSV header.
        if self._fname is None:
            os.makedirs(self._LOG_DIR, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._fname = os.path.join(self._LOG_DIR, f"flight_{stamp}.csv")

        self._t += dt
        x,  y,  z        = (float(v) for v in pos)
        roll, pitch, yaw = (float(v) for v in euler)
        vx, vy, vz       = (float(v) for v in vel_cmd)     # vel_cmd unpacked here
        tx, ty, tz, tyaw = (float(v) for v in target)

        dist    = math.sqrt((tx-x)**2 + (ty-y)**2 + (tz-z)**2)
        yaw_err = math.atan2(math.sin(tyaw - yaw), math.cos(tyaw - yaw))
        speed   = math.sqrt(vx**2 + vy**2 + vz**2)

        self._rows.append((
            round(self._t,   4),
            round(x,    4), round(y,     4), round(z,       4),
            round(roll, 4), round(pitch, 4), round(yaw,     4),
            round(vx,   4), round(vy,    4), round(vz,      4),
            round(tx,   4), round(ty,    4), round(tz,      4), round(tyaw, 4),
            round(dist, 4), round(yaw_err, 4), round(speed, 4),
            int(wind), profile_idx,
        ))

        self._xs.append(x); self._ys.append(y); self._zs.append(z)
        self._last_target_xyz = (tx, ty, tz)

    def _export(self):
        """Write CSV and, if matplotlib is available, save a 3-D path PNG."""
        if not self._rows or self._fname is None:
            return

        _COLS = [
            "time_s", "x", "y", "z",
            "roll_rad", "pitch_rad", "yaw_rad",
            "vx_cmd", "vy_cmd", "vz_cmd",          # renamed: vel_cmd not vel_world
            "target_x", "target_y", "target_z", "target_yaw",
            "dist_m", "yaw_err_rad", "speed_ms",
            "wind_enabled", "rl_profile",
        ]
        try:
            with open(self._fname, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(_COLS)
                w.writerows(self._rows)
            print(f"\nINFO: Flight log saved -> {self._fname}  ({len(self._rows)} rows)")
        except Exception as exc:
            print(f"\nWARN: Could not save flight log: {exc}")

        if _MATPLOTLIB_OK and len(self._xs) > 1:
            try:
                png_path = self._fname.replace(".csv", "_path.png")
                fig = plt.figure(figsize=(7, 6))
                ax  = fig.add_subplot(111, projection="3d")
                ax.plot(self._xs, self._ys, self._zs, "b-", lw=1, alpha=0.6, label="Path")
                ax.plot([self._xs[-1]], [self._ys[-1]], [self._zs[-1]],
                        "ro", ms=8, label="Final pos")
                ax.plot([self._last_target_xyz[0]],
                        [self._last_target_xyz[1]],
                        [self._last_target_xyz[2]],
                        "g*", ms=14, label="Target")
                ax.plot([self._xs[-1], self._xs[-1]],
                        [self._ys[-1], self._ys[-1]],
                        [0, self._zs[-1]], "r--", lw=0.5, alpha=0.3)
                ax.plot([self._last_target_xyz[0], self._last_target_xyz[0]],
                        [self._last_target_xyz[1], self._last_target_xyz[1]],
                        [0, self._last_target_xyz[2]], "g--", lw=0.5, alpha=0.3)
                ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
                ax.set_title("Flight Path – RL+PID Controller")
                ax.legend(loc="upper left", fontsize=8)
                fig.tight_layout()
                fig.savefig(png_path, dpi=120)
                plt.close(fig)
                print(f"INFO: Path plot saved  -> {png_path}")
            except Exception as exc:
                print(f"WARN: Could not save path PNG: {exc}")


_logger = _FlightLogger()


# =============================================================================
# Helper Functions – Mathematical Foundations
# =============================================================================
def _get_errors(state, target):
    """
    Computes control errors in both World and Body frames.
    Returns a dict named 'errs' at every call site (consistent with controller()).
    Ref: Lecture 6 (Coordinates & Rotations).
    """
    x, y, z, _, _, yaw = (float(v) for v in state)
    tx, ty, tz, tyaw   = (float(v) for v in target)

    # 1. World Frame Errors
    ex_w = tx - x
    ey_w = ty - y
    ez   = tz - z
    # Wrapped Yaw Error (Ref Lecture 10: shortest turn)
    eyaw = math.atan2(math.sin(tyaw - yaw), math.cos(tyaw - yaw))

    # 2. Body Frame Transformation
    # Ref Lecture 6, Slide 14: R_z(yaw) matrix application.
    cy, sy = math.cos(yaw), math.sin(yaw)
    ex_b =  cy * ex_w + sy * ey_w   # "Forward" error (Body X)
    ey_b = -sy * ex_w + cy * ey_w   # "Sideways" error (Body Y)

    # 3. Reward/Cost Function (Ref Lecture 10: Reward Shaping)
    dist = math.sqrt(ex_w**2 + ey_w**2 + ez**2)
    prec = 5.0 * max(0.0, 0.10 - dist) ** 2
    cost = dist**2 + 0.60 * eyaw**2 + prec

    return {
        "dist": dist, "cost": cost, "eyaw": eyaw, "ez": ez,
        "ex_b": ex_b, "ey_b": ey_b
    }


def _update_learning_agent(mem, errs, bonus, key):
    # FIX (naming): parameter renamed from 'errors' → 'errs' to match the
    # variable name used at every call site in controller().
    """
    Performs the Q-learning update (Temporal Difference learning).
    Ref: Lecture 10 (Advanced Control).
    """
    if mem["prev_key"] is None:
        return

    overshoot_penalty = (
        1500.0 * (errs["cost"] - mem["prev_cost"])
        if (errs["dist"] < 0.6 and errs["cost"] > mem["prev_cost"])
        else 0.0
    )
    reward = (
        (mem["prev_cost"] - errs["cost"])
        + bonus
        - 0.015 * mem["prev_effort"]
        - 2.0 * errs["dist"] ** 2
        - overshoot_penalty
    )

    if mem["step_count"] % 5 == 0:
        _td_update(mem, mem["prev_key"], mem["prev_action"], reward, key)
        mem["replay"].append((mem["prev_key"], mem["prev_action"], reward, key))
        if len(mem["replay"]) > mem["replay_cap"]:
            mem["replay"].pop(0)
        _replay_update(mem)

    mem["epsilon"] = max(mem["eps_floor"], mem["epsilon"] * mem["eps_decay"])


def _handle_dwell(mem, errs, target_pos):
    # FIX (naming): parameter renamed from 'errors' → 'errs' to match the
    # variable name used at every call site in controller().
    """
    Determines if the mission objective (hold phase) has been met.
    Ref: Coursework spec requirements.
    """
    is_within = errs["dist"] < _POS_TOL and abs(errs["eyaw"]) < _YAW_TOL
    bonus = 0.0

    if is_within:
        mem["dwell_steps"] += 1
        if mem["dwell_steps"] == 1:
            bonus = 20.0
        elif mem["dwell_steps"] >= _HOLD_STEPS:
            bonus = 30.0
            mem["dwell_steps"] = 0
            mem["integral"] = [0.0, 0.0, 0.0]
            mem["integral_yaw"] = 0.0
            if target_pos is None:
                mem["wp_idx"] = (mem["wp_idx"] + 1) % len(_CSV_TARGETS)
    else:
        mem["dwell_steps"] = 0

    return bonus


def _td_update(mem, s, a, r, s_next):
    # FIX (naming): unified internal variable names to q_curr / q_next.
    # The file previously had TWO definitions of _td_update:
    #   first  used  q_s / q_sn
    #   second used  q_curr / q_next
    # Python silently uses only the LAST definition, making the first dead
    # code.  Removed the duplicate; kept q_curr / q_next as they are more
    # descriptive (current-state Q-values vs next-state Q-values).
    """Ref: Lecture 10. Q(s,a) = Q(s,a) + alpha * (r + gamma * maxQ_next - Q(s,a))"""
    q_curr = mem["q_table"].setdefault(s,      [0.0] * len(_ACTION_PROFILES))
    q_next = mem["q_table"].setdefault(s_next, [0.0] * len(_ACTION_PROFILES))
    q_curr[a] += mem["alpha"] * (r + mem["gamma"] * max(q_next) - q_curr[a])


def _replay_update(mem):
    """Batch training to improve RL stability. Ref: Lecture 10, p.21."""
    if len(mem["replay"]) < mem["replay_batch"]:
        return
    for s, a, r, s_next in mem["rng"].sample(mem["replay"], mem["replay_batch"]):
        # FIX (naming): loop variable renamed from 'sn' → 's_next' to match
        # the parameter name used in _td_update.
        _td_update(mem, s, a, r, s_next)


# =============================================================================
# CSV waypoint loader – fallback when target_pos is None
# =============================================================================
def _load_csv_targets(csv_path="targets.csv"):
    _DEFAULTS = (
        (2.0,  2.0,  2.0, 0.0),
        (-2.0, 2.0,  2.0, 1.57),
        (-2.0, -2.0, 2.0, 3.14),
        (2.0,  -2.0, 2.0, 4.71),
    )
    if not os.path.isfile(csv_path):
        return _DEFAULTS
    try:
        targets = []
        with open(csv_path, newline="") as fh:
            reader = csv.DictReader(fh)
            fields = [f.strip() for f in (reader.fieldnames or [])]

            def _col(axis):
                for c in (f"target_{axis}", axis):
                    if c in fields:
                        return c
                raise ValueError(f"No '{axis}' column in {fields}")

            cx, cy, cz = _col("x"), _col("y"), _col("z")
            cyaw = next(
                (c for c in ("target_yaw", "yaw", "heading", "psi") if c in fields),
                None,
            )
            if cyaw is None:
                raise ValueError("No yaw column found")

            for i, raw in enumerate(reader):
                row = {k.strip(): v.strip() for k, v in raw.items()}
                try:
                    targets.append((float(row[cx]), float(row[cy]),
                                    float(row[cz]), float(row[cyaw])))
                except (KeyError, ValueError) as e:
                    raise ValueError(f"Bad row {i+2}: {row}") from e

        return tuple(targets) if targets else _DEFAULTS
    except Exception:
        return _DEFAULTS


_CSV_TARGETS = _load_csv_targets()


# =============================================================================
# Helper functions
# =============================================================================
def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def _wrap(angle):
    """Wrap angle to (-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def _state_key(dist, abs_yaw_err, wind):
    """
    Discretise continuous state into a hashable key for the Q-table.
    8 distance bands x 6 yaw-error bands x 2 wind modes = 96 states.
    """
    if   dist < 0.08: d = 0
    elif dist < 0.20: d = 1
    elif dist < 0.40: d = 2
    elif dist < 0.70: d = 3
    elif dist < 1.10: d = 4
    elif dist < 1.80: d = 5
    elif dist < 2.80: d = 6
    else:             d = 7

    if   abs_yaw_err < 0.04: y = 0
    elif abs_yaw_err < 0.10: y = 1
    elif abs_yaw_err < 0.25: y = 2
    elif abs_yaw_err < 0.50: y = 3
    elif abs_yaw_err < 1.00: y = 4
    else:                    y = 5

    return (d, y, int(wind))


# =============================================================================
# RL agent helpers
# =============================================================================
def _choose_action(mem, key, force_exploit=False, mask_coarse=False, mask_fine=False):
    """
    Action selection via e-greedy policy with Safety Masking.
    Ref: Lecture 10 (Cascade & Advanced Control), p.18-22.
    """
    # FIX (naming): 'q_vals' holds the raw Q-values; 'q_eff' holds the
    # masked version.  Both names are now used consistently: q_vals for
    # the table lookup, q_eff for the (possibly masked) effective values
    # passed to max() and the tie-breaking loop.
    q_vals = mem["q_table"].setdefault(key, [0.0] * len(_ACTION_PROFILES))

    if mask_coarse:
        q_eff = [v if i >= 4 else -1e9 for i, v in enumerate(q_vals)]
    elif mask_fine:
        q_eff = [v if i < 8  else -1e9 for i, v in enumerate(q_vals)]
    else:
        q_eff = q_vals   # No mask: q_eff is the same object as q_vals.

    eps = 0.0 if force_exploit else mem["epsilon"]
    if mem["rng"].random() < eps:
        if mask_coarse: return mem["rng"].randrange(4, 16)
        if mask_fine:   return mem["rng"].randrange(0, 8)
        return mem["rng"].randrange(0, 16)

    best_q = max(q_eff)
    ties   = [i for i, v in enumerate(q_eff) if v == best_q]
    return ties[mem["rng"].randrange(len(ties))]


def _init_memory():
    """Initialises persistent state across controller calls."""
    return {
        "wp_idx":       0,
        "q_table":      {},
        "alpha":        0.20,
        "gamma":        0.94,
        "epsilon":      0.25,
        "eps_decay":    0.9985,
        "eps_floor":    0.04,
        "replay":       [],
        "replay_cap":   400,
        "replay_batch": 16,
        "prev_key":     None,
        "prev_action":  None,
        "prev_cost":    None,
        "prev_effort":  0.0,
        "integral":     [0.0, 0.0, 0.0],
        "integral_yaw": 0.0,
        "prev_err":     [0.0, 0.0, 0.0],
        "dwell_steps":  0,
        "step_count":   0,
        "rng":          random.Random(42),
    }


# =============================================================================
# Main controller – DO NOT modify inputs/outputs
# =============================================================================
def controller(state, target_pos, dt, wind_enabled=False):
    """
    Modularised UAV Position Controller.

    Architecture:
    1. Error Transformation  (Lecture 6)
    2. RL Gain Selection     (Lecture 10)
    3. PID Actuation         (Lecture 10)
    """
    if not hasattr(controller, "_mem"):
        controller._mem = _init_memory()
    mem = controller._mem
    mem["step_count"] += 1

    # 1. State Estimation & Frame Transformation – Ref: Lecture 6
    # FIX (naming): the resolved setpoint is named 'active_target' throughout
    # to clearly distinguish it from the raw 'target_pos' argument.
    # Previously controller() used a local variable also called 'target' which
    # shadowed the outer meaning, and _handle_dwell received 'target_pos' while
    # using the already-resolved value internally.  Now:
    #   target_pos   = raw argument from simulator (may be None in dev mode).
    #   active_target = resolved 4-tuple (x, y, z, yaw) used for all computation.
    if target_pos is not None:
        active_target = tuple(float(v) for v in target_pos)
    else:
        active_target = _CSV_TARGETS[mem["wp_idx"]]

    errs = _get_errors(state, active_target)
    key  = _state_key(errs["dist"], abs(errs["eyaw"]), wind_enabled)

    # 2. Learning & Logic – Ref: Lecture 10
    # Pass target_pos (not active_target) to _handle_dwell so it can detect
    # whether we are in simulator mode (target_pos provided) or free-fly mode
    # (target_pos is None) when deciding whether to advance the waypoint index.
    bonus = _handle_dwell(mem, errs, target_pos)
    _update_learning_agent(mem, errs, bonus, key)

    # 3. Action Selection (Gain Scheduling) – Ref: Lecture 10, p.18-22
    # FIX: Removed the separate 'precision zone' block that forced profiles 12-15
    # at dist < 0.08m. This was causing premature deceleration – the drone slowed
    # to hover-hold speed (0.04 m/s cap) while still 8 cm away. The mask_coarse
    # at dist < 0.1m already prevents aggressive profiles near the target, so the
    # RL agent naturally selects fine/hold profiles without needing a hard override.
    action = _choose_action(
        mem, key,
        force_exploit=(errs["dist"] < 0.5),
        mask_coarse=(errs["dist"] < 0.1),
        mask_fine=(errs["dist"] > 1.5)
    )

    # FIX (naming): unpacked tuple variables renamed from *_lim → *_max to
    # match the column comment in _ACTION_PROFILES above.  Consistent naming:
    #   v_xy_max, v_z_max, yaw_rate_max  – everywhere in this file.
    (kp_xy, kp_z, kp_yaw,
     ki_xy, ki_z, ki_yaw,
     kd_xy, kd_z,
     v_xy_max, v_z_max, yaw_rate_max) = _ACTION_PROFILES[action]

    # 4. PID Law – Discrete Implementation – Ref: Lecture 10, Slide 41
    if dt > 0.0:
        # FIX (naming): single 'integ_limit' replaces the previous 'i_lim'
        # (ambiguous abbreviation) – one limit value applied to all three
        # positional axes and a separate ±0.30 limit for yaw.
        integ_limit = (
            0.40 if errs["dist"] < _POS_TOL   # Tight near target
            else 1.00 if wind_enabled           # Wide under wind disturbance
            else 0.60                           # Standard approach
        )
        mem["integral"][0]  = _clip(mem["integral"][0]  + errs["ex_b"] * dt, -integ_limit, integ_limit)
        mem["integral"][1]  = _clip(mem["integral"][1]  + errs["ey_b"] * dt, -integ_limit, integ_limit)
        mem["integral"][2]  = _clip(mem["integral"][2]  + errs["ez"]   * dt, -integ_limit, integ_limit)
        mem["integral_yaw"] = _clip(mem["integral_yaw"] + errs["eyaw"] * dt, -0.30, 0.30)

    # Derivative: finite difference (e[k] - e[k-1]) / dt – Ref: Lecture 10, Slide 31
    d_ex_b = _clip((errs["ex_b"] - mem["prev_err"][0]) / dt, -3.0, 3.0) if dt > 0 else 0.0
    d_ey_b = _clip((errs["ey_b"] - mem["prev_err"][1]) / dt, -3.0, 3.0) if dt > 0 else 0.0
    d_ez   = _clip((errs["ez"]   - mem["prev_err"][2]) / dt, -3.0, 3.0) if dt > 0 else 0.0
    # FIX (naming): derivative variables renamed from d_x/d_y/d_z → d_ex_b/d_ey_b/d_ez
    # to make clear these are derivatives of the body-frame errors (ex_b, ey_b)
    # and altitude error (ez), matching the error variable names they are derived from.

    # 5. Output Velocity Commands: v = Kp·e + Ki·∫e·dt + Kd·(de/dt)
    # Ref: Lecture 10, Slide 34
    vx = _clip(kp_xy * errs["ex_b"] + ki_xy * mem["integral"][0] + kd_xy * d_ex_b, -v_xy_max, v_xy_max)
    vy = _clip(kp_xy * errs["ey_b"] + ki_xy * mem["integral"][1] + kd_xy * d_ey_b, -v_xy_max, v_xy_max)
    vz = _clip(kp_z  * errs["ez"]   + ki_z  * mem["integral"][2] + kd_z  * d_ez,   -v_z_max,  v_z_max)
    yr = _clip(kp_yaw * errs["eyaw"] + ki_yaw * mem["integral_yaw"], -yaw_rate_max, yaw_rate_max)

    # 6. Bookkeeping for next step
    mem["prev_key"]    = key
    mem["prev_action"] = action
    mem["prev_cost"]   = errs["cost"]
    mem["prev_effort"] = abs(vx) + abs(vy) + abs(vz) + abs(yr)
    mem["prev_err"]    = [errs["ex_b"], errs["ey_b"], errs["ez"]]

    # 7. Data Logging – Ref: Tutorial 1 – Data Frames
    _logger.log(
        dt=dt,
        pos=state[0:3],
        euler=state[3:6],
        vel_cmd=(vx, vy, vz),          # vel_cmd: velocity command output
        target=active_target,           # active_target: resolved setpoint
        wind=wind_enabled,
        profile_idx=action,
    )

    return (vx, vy, vz, yr)
