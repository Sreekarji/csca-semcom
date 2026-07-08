"""
Shaped Reward Function for HDM Training.

FIX (2026-07-08): Added inequality bonus to reward non-uniform bandwidth
allocation. Without this, even after fixing BW allocation to softmax, the
policy has no incentive to output non-uniform logits — the reward signal
from ISR alone is symmetric across tasks.

The inequality bonus rewards the policy when high-urgency tasks receive
more bandwidth than low-urgency tasks, by comparing the actual BW spread
to the intent urgency spread. This creates the gradient signal that teaches
HDM to prioritise urgent tasks.

Returns: float in [-2.0, 2.0] range (unchanged from original).
"""

import numpy as np
from cscqi import compute_cscqi, is_intent_satisfied


# ---------------------------------------------------------------------------
# Inequality bonus helpers
# ---------------------------------------------------------------------------

def _compute_urgency(task: dict) -> float:
    """
    Per-task urgency in [0, 1].
    High urgency = tight delay intent AND high quality intent.
    """
    delay_tightness   = 1.0 - min(task["tau_S_int"] / 5.0, 1.0)   # tight delay → high
    quality_tightness = task["vartheta_S_int"]                      # high quality → high
    return 0.5 * delay_tightness + 0.5 * quality_tightness


def _bw_inequality_bonus(tasks: list) -> float:
    """
    Reward that encourages the policy to allocate more BW to urgent tasks.

    Method: compute Spearman-rank correlation between per-task urgency and
    per-task BW allocation. Perfect positive correlation (urgent tasks get
    more BW) → bonus = +INEQUALITY_SCALE. No correlation or wrong direction
    → bonus ≈ 0 or negative.

    Falls back to 0 if bw_alloc_hz is not present in the task dict (e.g.
    during baseline evaluation — baselines don't use this field).
    """
    if len(tasks) < 2:
        return 0.0
    if "bw_alloc_hz" not in tasks[0]:
        return 0.0

    urgencies = np.array([_compute_urgency(t) for t in tasks])
    bw_values = np.array([t["bw_alloc_hz"] for t in tasks], dtype=np.float64)

    # Normalise so scale doesn't matter
    bw_range = bw_values.max() - bw_values.min()
    if bw_range < 1e3:
        # All tasks got essentially the same BW → no bonus (penalise slightly)
        return -0.05

    # Spearman rank correlation
    urgency_ranks = np.argsort(np.argsort(urgencies)).astype(float)
    bw_ranks      = np.argsort(np.argsort(bw_values)).astype(float)

    n = len(tasks)
    d_sq = np.sum((urgency_ranks - bw_ranks) ** 2)
    rho  = 1.0 - (6.0 * d_sq) / (n * (n ** 2 - 1) + 1e-8)   # Spearman formula

    # Scale to a small bonus range so it guides but doesn't dominate ISR reward
    INEQUALITY_SCALE = 0.15
    return float(np.clip(rho * INEQUALITY_SCALE, -0.15, 0.15))


# ---------------------------------------------------------------------------
# Main reward functions
# ---------------------------------------------------------------------------

def compute_shaped_reward(
    tasks: list,
    intent_vector: list = None,
) -> float:
    """
    Shaped reward for HDM Actor-Critic training.

    Components (weighted sum):
      1. Delay score     (0.6 weight) — sigmoid reward for meeting delay intent
      2. Quality score   (0.4 weight) — sigmoid reward for meeting quality intent
      3. Intent bonus    (+0.3)       — flat bonus when both constraints satisfied
      4. Intent vector   (0.3 blend)  — alignment with explicit delay/quality prefs
      5. Inequality bonus             — reward for non-uniform urgency-aligned BW

    The inequality bonus (component 5) is the key addition. Without it, the
    policy has no gradient signal to output non-uniform logits, so it settles
    into uniform output — making softmax BW allocation behave like static.

    Returns: float in [-2.0, 2.0]
    """
    if not tasks:
        return 0.0

    task_rewards = []
    for t in tasks:
        tau_S        = t["tau_S"]
        vartheta_S   = t["vartheta_S"]
        tau_int      = t["tau_S_int"]
        vartheta_int = t["vartheta_S_int"]

        # --- Delay component ---
        delay_ratio = tau_int / max(tau_S, 1e-6)
        if tau_S > tau_int:
            # Penalise proportionally to overshoot
            delay_score = -1.0 * (tau_S - tau_int) / max(tau_int, 1e-6)
            delay_score = max(delay_score, -2.0)
        else:
            # Sigmoid reward — saturates at 1.0 as slack grows
            delay_score = 1.0 / (1.0 + np.exp(-5.0 * (delay_ratio - 1.0)))

        # --- Quality component ---
        quality_ratio = vartheta_int / max(vartheta_S, 1e-6)
        if vartheta_S > vartheta_int:
            quality_score = -0.5 * (vartheta_S - vartheta_int) / max(vartheta_int, 1e-6)
            quality_score = max(quality_score, -1.0)
        else:
            quality_score = 1.0 / (1.0 + np.exp(-5.0 * (quality_ratio - 1.0)))

        # --- Satisfaction bonus ---
        bonus = 0.3 if is_intent_satisfied(
            tau_S, vartheta_S, tau_int, vartheta_int
        ) else 0.0

        task_reward = 0.6 * delay_score + 0.4 * quality_score + bonus
        task_rewards.append(task_reward)

    base_reward = float(np.mean(task_rewards))

    # --- Intent vector alignment ---
    if intent_vector is not None and len(intent_vector) >= 2:
        delay_pref   = intent_vector[0]
        quality_pref = intent_vector[1]
        delay_rewards   = []
        quality_rewards = []
        for t in tasks:
            dr = t["tau_S_int"]      / max(t["tau_S"],      1e-6)
            qr = t["vartheta_S_int"] / max(t["vartheta_S"], 1e-6)
            delay_rewards.append(min(dr, 2.0))
            quality_rewards.append(min(qr, 2.0))
        intent_reward = (
            delay_pref   * float(np.mean(delay_rewards)) +
            quality_pref * float(np.mean(quality_rewards))
        )
        base_reward = 0.7 * base_reward + 0.3 * float(intent_reward)

    # --- Inequality bonus (KEY FIX) ---
    # Rewards the policy for allocating more BW to more urgent tasks.
    # This is the gradient signal that breaks the symmetry causing HDM ≈ baselines.
    ineq_bonus = _bw_inequality_bonus(tasks)
    base_reward = base_reward + ineq_bonus

    return float(np.clip(base_reward, -2.0, 2.0))


def compute_isr_reward(tasks: list) -> float:
    """
    Intent Satisfaction Rate reward (for logging, not training).
    Returns fraction of tasks where delay <= intent AND quality <= intent.
    """
    if not tasks:
        return 0.0
    satisfied = sum(
        1 for t in tasks
        if t["tau_S"] <= t["tau_S_int"] and t["vartheta_S"] <= t["vartheta_S_int"]
    )
    return satisfied / len(tasks)
