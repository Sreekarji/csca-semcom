"""
Shaped Reward Function for HDM Training.

FIX (2026-07-08): Stronger per-task delay signal + BW-urgency alignment bonus.
Rewards HDM for giving more bandwidth to more urgent tasks.

Returns: float in [-3.0, 3.0] range.
"""

import numpy as np
from cscqi import compute_cscqi, is_intent_satisfied


def compute_shaped_reward(tasks: list, intent_vector: list = None,
                          actions: dict = None) -> float:
    """
    Shaped reward with explicit bandwidth-urgency alignment bonus.
    Rewards HDM for giving more bandwidth to more urgent tasks.

    Components:
      1. Per-task delay score (0.7 weight) — strong penalty for missing deadline
      2. Per-task quality score (0.3 weight) — penalty for missing quality
      3. Intent satisfaction bonus (+0.5) — flat bonus when both met
      4. BW-urgency alignment bonus (+0.3 * correlation) — rewards correct prioritization
    """
    if not tasks:
        return 0.0

    task_rewards = []
    urgency_scores = []

    for t in tasks:
        tau_S = t["tau_S"]
        vartheta_S = t["vartheta_S"]
        tau_int = t["tau_S_int"]
        vartheta_int = t["vartheta_S_int"]

        # Urgency: how tight is the delay constraint?
        urgency = 1.0 / max(tau_int, 0.1)
        urgency_scores.append(urgency)

        # Delay score: STRONG penalty for missing deadline
        if tau_S <= tau_int:
            delay_score = 1.0 + (tau_int - tau_S) / max(tau_int, 1e-6)
        else:
            overshoot = (tau_S - tau_int) / max(tau_int, 1e-6)
            delay_score = -2.0 * overshoot

        delay_score = float(np.clip(delay_score, -3.0, 2.0))

        # Quality score
        if vartheta_S <= vartheta_int:
            quality_score = 1.0
        else:
            overshoot_q = (vartheta_S - vartheta_int) / max(vartheta_int, 1e-6)
            quality_score = -1.0 * overshoot_q

        quality_score = float(np.clip(quality_score, -2.0, 1.0))

        # Intent satisfaction bonus
        if is_intent_satisfied(tau_S, vartheta_S, tau_int, vartheta_int):
            bonus = 0.5
        else:
            bonus = 0.0

        task_reward = 0.7 * delay_score + 0.3 * quality_score + bonus
        task_rewards.append(task_reward)

    base_reward = float(np.mean(task_rewards))

    # BW-urgency alignment bonus
    if actions is not None and "bandwidth" in actions:
        bw = actions["bandwidth"]
        if hasattr(bw, "cpu"):
            bw = bw.cpu().numpy().flatten()
        n = min(len(bw), len(urgency_scores))
        if n > 1:
            bw_n = np.array(bw[:n])
            urg_n = np.array(urgency_scores[:n])
            if bw_n.std() > 1e-6 and urg_n.std() > 1e-6:
                correlation = np.corrcoef(bw_n, urg_n)[0, 1]
                alignment_bonus = 0.3 * float(correlation)
            else:
                alignment_bonus = -0.1  # Penalize uniform BW
            base_reward += alignment_bonus

    return float(np.clip(base_reward, -3.0, 3.0))


def compute_isr_reward(tasks: list) -> float:
    """Intent Satisfaction Rate reward (for logging, not training)."""
    if not tasks:
        return 0.0
    satisfied = sum(
        1 for t in tasks
        if t["tau_S"] <= t["tau_S_int"] and t["vartheta_S"] <= t["vartheta_S_int"]
    )
    return satisfied / len(tasks)
