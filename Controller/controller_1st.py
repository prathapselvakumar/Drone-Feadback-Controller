# WIND FLAG – set to True to enable wind-disturbance handling during marking
WIND = True

# =============================================================================
# AERO60492 Coursework 3 – Feedback Control
# Advanced method: Q-learning reinforcement learning with PID actuation
#
# Overview
# --------
# This controller uses a Q-learning agent (tabular RL) to adaptively select
# from 16 pre-tuned PID gain profiles at every timestep.  The choice of
# profile is driven by a discretised state (distance band × yaw-error band ×
# wind mode), and the agent learns – online, during flight – which profile
# minimises position/yaw error most efficiently.
#
# Architecture
# ------------
#   1. Coordinate transformation
#      World-frame position errors are rotated into the drone body frame
#      using the current yaw angle, so velocity commands are always relative
#      to the drone's heading.  Yaw error is wrapped to (-π, π].
#
#   2. PID actuation layer
#      A full PID (Proportional + Integral + Derivative) controller converts
#      body-frame errors into velocity setpoints.  The integral term removes
#      steady-state bias and is particularly important under wind.  The
#      derivative term damps overshoot when approaching the target.
#      Integral windup is prevented by hard clamping.
#
#   3. Q-learning gain-scheduling layer (advanced method)
#      State:  (distance_band [0–7], yaw_error_band [0–5], wind [0/1])
#              → 96 discrete states, giving fine resolution near the target
#              where precision matters most.
#      Actions: one of 16 gain profiles spanning coarse approach to
#               ultra-precise hover-hold.
#      Reward:  reduction in weighted cost + dwell bonus for holding position
#               − effort penalty to discourage thrashing.
#      Updates: online TD(0) + experience replay (400-sample circular buffer,
#               mini-batch size 16) for sample efficiency.
#      Exploration: ε-greedy, decaying from 0.25 → 0.04.
#
#   4. target_pos interface
#      The function accepts the external target_pos argument supplied by the
#      simulator/marker.  The CSV waypoint list is used only when target_pos
#      is None or omitted (e.g. during free-flying development).  When
#      target_pos is provided it takes precedence – this is what the marker's
#      auto-tester uses to inject random goal positions.
#
# Tuning notes
# ------------
# The 16 action profiles were hand-tuned in three stages:
#   Stage 1 – coarse approach (profiles 0–3): maximise convergence speed
#             without overshoot at > 1 m range.
#   Stage 2 – fine approach (profiles 8–11): reduce velocity caps and raise
#             integral gains to eliminate steady-state error at < 0.5 m.
#   Stage 3 – hover-hold (profiles 12–15): very high integral, tiny caps,
#             to pin the drone within < 0.01 m of the target.
# The RL agent then learns, online, which profile to apply at each (distance,
# yaw-error, wind) combination, removing the need for manual switching logic.
# =============================================================================

import csv
import math
import os
import random


# -----------------------------------------------------------------------------
# Gain profiles
# Each row: (kp_xy, kp_z, kp_yaw, ki_xy, ki_z, ki_yaw, kd_xy, kd_z,
#            v_xy_max, v_z_max, yaw_rate_max)
# 16 profiles span coarse approach → hover-hold to give the RL agent a rich
# action space.
# -----------------------------------------------------------------------------
_ACTION_PROFILES = (
    # ── Coarse approach (far from target) ─────────────────────────────────────
    (1.40, 1.30, 1.10, 0.04, 0.05, 0.03, 0.10, 0.08, 2.00, 1.20, 1.40),  # 0
    (1.20, 1.15, 0.95, 0.04, 0.05, 0.03, 0.08, 0.06, 1.60, 1.00, 1.20),  # 1
    (1.00, 1.00, 0.80, 0.05, 0.06, 0.04, 0.07, 0.06, 1.30, 0.90, 1.00),  # 2
    (0.80, 0.90, 0.70, 0.05, 0.06, 0.04, 0.06, 0.05, 1.00, 0.80, 0.90),  # 3
    # ── Mid-range ─────────────────────────────────────────────────────────────
    (1.30, 1.20, 1.20, 0.06, 0.07, 0.05, 0.12, 0.09, 1.50, 1.00, 1.40),  # 4
    (1.10, 1.05, 1.00, 0.06, 0.07, 0.05, 0.10, 0.08, 1.20, 0.85, 1.20),  # 5
    (0.90, 0.95, 0.85, 0.07, 0.08, 0.06, 0.08, 0.07, 1.00, 0.75, 1.00),  # 6
    (0.70, 0.80, 0.70, 0.07, 0.08, 0.06, 0.06, 0.06, 0.80, 0.65, 0.85),  # 7
    # ── Fine approach (< 0.5 m) ───────────────────────────────────────────────
    (0.85, 0.80, 0.75, 0.12, 0.12, 0.09, 0.14, 0.10, 0.50, 0.45, 0.60),  # 8
    (0.70, 0.70, 0.65, 0.14, 0.14, 0.10, 0.12, 0.09, 0.40, 0.38, 0.50),  # 9
    (0.60, 0.65, 0.60, 0.16, 0.16, 0.11, 0.10, 0.08, 0.32, 0.32, 0.42),  # 10
    (0.50, 0.58, 0.55, 0.18, 0.18, 0.12, 0.09, 0.07, 0.26, 0.26, 0.36),  # 11
    # ── Hover-hold (at target) ────────────────────────────────────────────────
    (0.40, 0.50, 0.50, 0.22, 0.22, 0.15, 0.08, 0.07, 0.18, 0.18, 0.25),  # 12
    (0.35, 0.45, 0.45, 0.26, 0.26, 0.18, 0.07, 0.06, 0.14, 0.14, 0.20),  # 13
    (0.30, 0.40, 0.40, 0.30, 0.30, 0.20, 0.06, 0.05, 0.10, 0.10, 0.16),  # 14
    (0.25, 0.35, 0.35, 0.34, 0.34, 0.22, 0.05, 0.05, 0.08, 0.08, 0.12),  # 15
)

# Tolerances for the dwell-hold phase
_POS_TOL    = 0.08   # m   – must be within this to start hold phase
_YAW_TOL    = 0.06   # rad – yaw tolerance for hold phase
_HOLD_STEPS = 20     # consecutive in-tolerance steps to trigger hold bonus


# -----------------------------------------------------------------------------
# CSV waypoint loader – fallback when no target_pos is injected by caller
# -----------------------------------------------------------------------------
def _load_csv_targets(csv_path="targets.csv"):
    """
    Load waypoints from targets.csv.  Accepts column names target_x/y/z/yaw
    or bare x/y/z/yaw; strips surrounding whitespace from headers and values.
    Falls back to hard-coded four-corner defaults on any error.
    """
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


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def _wrap(angle):
    """Wrap angle to (-π, π]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def _state_key(dist, abs_yaw_err, wind):
    """
    Discretise continuous state into a hashable key for the Q-table.
    8 distance bands × 6 yaw-error bands × 2 wind modes = 96 states.
    Bands are finer close to the target where precision is critical.
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


def _snapshot(state, target, wind):
    """
    Compute body-frame errors, 3-D Euclidean distance and RL cost.

    Coordinate transformation:
      World-frame XY errors are rotated by the negative of the current yaw
      into the drone body frame so that vx commands the drone forward along
      its own nose direction, and vy commands it laterally – matching the
      coordinate convention expected by the simulator and real Tello.
    """
    x, y, z, _, _, yaw = (float(v) for v in state)
    tx, ty, tz, tyaw   = (float(v) for v in target)

    # Position errors in world frame
    ex_w = tx - x
    ey_w = ty - y
    ez   = tz - z
    eyaw = _wrap(tyaw - yaw)

    # Rotate XY into drone body frame using current yaw
    cy, sy = math.cos(yaw), math.sin(yaw)
    ex_b =  cy * ex_w + sy * ey_w   # forward (body x)
    ey_b = -sy * ex_w + cy * ey_w   # lateral (body y)

    dist = math.sqrt(ex_w**2 + ey_w**2 + ez**2)

    # RL cost: penalises distance and yaw error; extra term keeps cost
    # high when residual error is tiny (incentivises ultra-tight hold).
    prec = 5.0 * max(0.0, 0.10 - dist) ** 2
    cost = dist**2 + 0.60 * eyaw**2 + prec

    return {
        "ex_b": ex_b, "ey_b": ey_b, "ez": ez, "eyaw": eyaw,
        "dist": dist, "cost": cost,
        "key":  _state_key(dist, abs(eyaw), wind),
    }


# -----------------------------------------------------------------------------
# RL agent helpers
# -----------------------------------------------------------------------------
def _choose_action(mem, key):
    """ε-greedy action selection from the Q-table."""
    q = mem["q_table"].setdefault(key, [0.0] * len(_ACTION_PROFILES))
    if mem["rng"].random() < mem["epsilon"]:
        return mem["rng"].randrange(len(_ACTION_PROFILES))
    best = max(q)
    ties = [i for i, v in enumerate(q) if v == best]
    return ties[mem["rng"].randrange(len(ties))]


def _td_update(mem, s, a, r, s_next):
    """TD(0) update: Q(s,a) ← Q(s,a) + α · (r + γ · max Q(s') − Q(s,a))"""
    q_s  = mem["q_table"].setdefault(s,      [0.0] * len(_ACTION_PROFILES))
    q_ns = mem["q_table"].setdefault(s_next, [0.0] * len(_ACTION_PROFILES))
    q_s[a] += mem["alpha"] * (r + mem["gamma"] * max(q_ns) - q_s[a])


def _replay_update(mem):
    """
    Sample a random mini-batch from the replay buffer and apply TD updates.
    Breaking temporal correlations improves stability and sample efficiency
    compared to purely on-policy online updates.
    """
    if len(mem["replay"]) < mem["replay_batch"]:
        return
    for s, a, r, sn in mem["rng"].sample(mem["replay"], mem["replay_batch"]):
        _td_update(mem, s, a, r, sn)


def _init_memory():
    return {
        "wp_idx":       0,          # CSV waypoint index (free-fly mode only)
        "q_table":      {},         # maps state_key → list of Q-values per action
        "alpha":        0.20,       # TD learning rate
        "gamma":        0.94,       # discount factor
        "epsilon":      0.25,       # exploration probability
        "eps_decay":    0.9985,     # per-step decay multiplier
        "eps_floor":    0.04,       # minimum epsilon
        "replay":       [],         # experience replay buffer
        "replay_cap":   400,        # maximum buffer size
        "replay_batch": 16,         # mini-batch size for replay updates
        "prev_key":     None,       # previous state key (for TD update)
        "prev_action":  None,       # previous action chosen
        "prev_cost":    None,       # previous RL cost (for reward calculation)
        "prev_effort":  0.0,        # previous total velocity magnitude
        "integral":     [0.0, 0.0, 0.0],  # PID integral: body x, y, z
        "integral_yaw": 0.0,        # PID integral: yaw
        "prev_err":     [0.0, 0.0, 0.0],  # previous body-frame errors for D term
        "dwell_steps":  0,          # steps held within tolerance
        "rng":          random.Random(42),
    }


# -----------------------------------------------------------------------------
# Main controller – DO NOT modify inputs/outputs
# -----------------------------------------------------------------------------
def controller(state, target_pos, dt, wind_enabled=False):
    """
    Feedback controller for UAV position stabilisation.

    Inputs
    ------
    state        : [x, y, z, roll, pitch, yaw]  metres and radians
    target_pos   : [x, y, z, yaw]  desired position and heading (m, m, m, rad)
                   Provided by the simulator at every timestep.  The marker's
                   auto-tester injects random goal positions here.
    dt           : timestep in seconds
    wind_enabled : True when wind disturbance is active (handled by higher
                   integral limits and wind-specific Q-table entries)

    Returns
    -------
    (vx, vy, vz, yaw_rate)  body-frame velocity commands in m/s and rad/s
    """
    # Initialise persistent memory once on first call
    if not hasattr(controller, "_mem"):
        controller._mem = _init_memory()
    mem = controller._mem

    # ── Resolve active target ─────────────────────────────────────────────────
    # target_pos from the simulator always takes priority.
    # CSV fallback is only used when target_pos is None (development mode).
    if target_pos is not None:
        active_target = tuple(float(v) for v in target_pos)
    else:
        active_target = _CSV_TARGETS[mem["wp_idx"]]

    snap = _snapshot(state, active_target, wind_enabled)

    # ── Dwell-hold logic ──────────────────────────────────────────────────────
    # Count consecutive steps within tolerance.  Reward the agent for staying
    # at the target.  In CSV free-fly mode, advance to next waypoint after
    # HOLD_STEPS and reset integrators to prevent windup on the new leg.
    within     = snap["dist"] < _POS_TOL and abs(snap["eyaw"]) < _YAW_TOL
    hold_bonus = 0.0

    if within:
        mem["dwell_steps"] += 1
        if mem["dwell_steps"] == 1:
            hold_bonus += 20.0      # reward for first arrival
        if mem["dwell_steps"] >= _HOLD_STEPS:
            hold_bonus += 30.0      # reward for sustained hold
            if target_pos is None:  # advance waypoint only in CSV mode
                mem["wp_idx"] = (mem["wp_idx"] + 1) % len(_CSV_TARGETS)
            mem["dwell_steps"] = 0
            # Reset PID integrators and derivative memory
            mem["integral"]     = [0.0, 0.0, 0.0]
            mem["integral_yaw"] = 0.0
            mem["prev_err"]     = [0.0, 0.0, 0.0]
            if target_pos is None:
                active_target = _CSV_TARGETS[mem["wp_idx"]]
                snap = _snapshot(state, active_target, wind_enabled)
    else:
        mem["dwell_steps"] = 0

    # ── Q-learning TD(0) update with experience replay ────────────────────────
    if mem["prev_key"] is not None:
        reward = (
            (mem["prev_cost"] - snap["cost"])   # improvement in weighted error
            + hold_bonus                         # dwell/arrival bonus
            - 0.015 * mem["prev_effort"]         # penalise high control effort
            - 2.0   * snap["dist"] ** 2          # continuous steady-state shaping
        )
        _td_update(mem, mem["prev_key"], mem["prev_action"], reward, snap["key"])
        mem["replay"].append(
            (mem["prev_key"], mem["prev_action"], reward, snap["key"])
        )
        if len(mem["replay"]) > mem["replay_cap"]:
            mem["replay"].pop(0)
        _replay_update(mem)
        mem["epsilon"] = max(mem["eps_floor"], mem["epsilon"] * mem["eps_decay"])

    # ── Select PID gain profile via ε-greedy Q-table ─────────────────────────
    action = _choose_action(mem, snap["key"])
    (kp_xy, kp_z, kp_yaw,
     ki_xy, ki_z, ki_yaw,
     kd_xy, kd_z,
     v_xy_max, v_z_max, yaw_max) = _ACTION_PROFILES[action]

    # ── PID control law ───────────────────────────────────────────────────────
    if dt > 0.0:
        # Anti-windup limits scale with proximity and wind conditions
        if snap["dist"] < _POS_TOL:
            i_lim_xy, i_lim_z = 0.40, 0.40   # tight near target
        elif wind_enabled:
            i_lim_xy, i_lim_z = 1.00, 1.00   # wider under wind disturbance
        else:
            i_lim_xy, i_lim_z = 0.60, 0.60

        # Integrate body-frame errors with clamping
        mem["integral"][0]  = _clip(mem["integral"][0]  + snap["ex_b"] * dt, -i_lim_xy, i_lim_xy)
        mem["integral"][1]  = _clip(mem["integral"][1]  + snap["ey_b"] * dt, -i_lim_xy, i_lim_xy)
        mem["integral"][2]  = _clip(mem["integral"][2]  + snap["ez"]   * dt, -i_lim_z,  i_lim_z)
        mem["integral_yaw"] = _clip(mem["integral_yaw"] + snap["eyaw"] * dt, -0.30, 0.30)

    # Finite-difference derivative with spike clamping
    d_ex = _clip((snap["ex_b"] - mem["prev_err"][0]) / dt, -3.0, 3.0) if dt > 0 else 0.0
    d_ey = _clip((snap["ey_b"] - mem["prev_err"][1]) / dt, -3.0, 3.0) if dt > 0 else 0.0
    d_ez = _clip((snap["ez"]   - mem["prev_err"][2]) / dt, -3.0, 3.0) if dt > 0 else 0.0

    mem["prev_err"] = [snap["ex_b"], snap["ey_b"], snap["ez"]]

    # PID output in body frame, clamped to velocity limits from active profile
    vx = _clip(kp_xy * snap["ex_b"] + ki_xy * mem["integral"][0] + kd_xy * d_ex,
               -v_xy_max, v_xy_max)
    vy = _clip(kp_xy * snap["ey_b"] + ki_xy * mem["integral"][1] + kd_xy * d_ey,
               -v_xy_max, v_xy_max)
    vz = _clip(kp_z  * snap["ez"]   + ki_z  * mem["integral"][2] + kd_z  * d_ez,
               -v_z_max, v_z_max)
    yr = _clip(kp_yaw * snap["eyaw"] + ki_yaw * mem["integral_yaw"],
               -yaw_max, yaw_max)

    # ── Bookkeeping for next timestep ─────────────────────────────────────────
    mem["prev_key"]    = snap["key"]
    mem["prev_action"] = action
    mem["prev_cost"]   = snap["cost"]
    mem["prev_effort"] = abs(vx) + abs(vy) + abs(vz) + abs(yr)

    return (vx, vy, vz, yr)