"""
FIX 20: Final experiment suite for Dr. Joshi.
Runs HDM vs baselines across tpc values, generates CSVs + plots + SUMMARY.txt.

Run: python code/experiments/final_results.py
Output: results/final/
"""
import os
import sys
import csv
import torch
import numpy as np
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for sub in ["code/hdm", "code/channel", "code/evaluation", "code/utils", "code/experiments"]:
    sys.path.insert(0, os.path.join(BASE, sub))

from reproducibility import set_seed
from train_han_mlp import (
    HANMLPTrainer, evaluate_policy, evaluate_static, evaluate_baseline_actor,
    sample_eval_state, intents_from_state, parse_action, DEVICE, POLICY,
    train_ac_baseline, train_ppo_baseline, train_sac_baseline, PerTaskGaussianActor,
    MultiCSCAEnvironment, compute_isr, compute_cscqi,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(BASE, "results", "final")
os.makedirs(RESULTS_DIR, exist_ok=True)

TGC_LIST = [1, 2, 4, 10]
N_EVAL = 200


def timestamp():
    return datetime.now().strftime("%H:%M:%S")


def run_one_tpc(tpc):
    """Train HDM + baselines for one tpc value. Returns dict of results."""
    print(f"\n{'='*60}")
    print(f"[{timestamp()}] tpc={tpc} ({tpc*5} tasks)")
    print(f"{'='*60}")

    # Train HDM
    set_seed(42)
    trainer = HANMLPTrainer(tasks_per_csca=tpc, difficulty="medium")
    best_isr = trainer.train(max_episodes=1000)

    # Load best checkpoint
    ckpt_path = os.path.join(BASE, "checkpoints", f"han_{POLICY}_tpc{tpc}_best.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        trainer.han.load_state_dict(ckpt["han"])
        trainer.actor.load_state_dict(ckpt["actor"])
        print(f"  Loaded best checkpoint (ISR={ckpt['isr']:.3f}, ep {ckpt['episode']})")

    # Evaluate HDM
    hdm_mean, hdm_std = evaluate_policy(trainer, N_EVAL)

    # Create eval environment
    eval_env = MultiCSCAEnvironment(
        n_cscas=trainer.n_cscas, n_relays=trainer.n_relays,
        difficulty="medium", tasks_per_csca=tpc,
    )

    # Train baselines
    set_seed(42)
    print(f"[{timestamp()}] Training AC baseline...")
    ac_actor = train_ac_baseline(
        trainer.han, eval_env, trainer.n_tasks, trainer.n_relays, trainer.n_mcs,
        trainer.action_dim, n_episodes=1000,
    )
    set_seed(42)
    print(f"[{timestamp()}] Training PPO baseline...")
    ppo_actor = train_ppo_baseline(
        trainer.han, eval_env, trainer.n_tasks, trainer.n_relays, trainer.n_mcs,
        trainer.action_dim, n_episodes=1000,
    )
    set_seed(42)
    print(f"[{timestamp()}] Training SAC baseline...")
    sac_actor = train_sac_baseline(
        trainer.han, eval_env, trainer.n_tasks, trainer.n_relays, trainer.n_mcs,
        trainer.action_dim, n_episodes=1000,
    )

    # Evaluate baselines
    ac_mean, ac_std = evaluate_baseline_actor(
        ac_actor, trainer.han, eval_env, trainer.n_tasks, trainer.n_relays, trainer.n_mcs, N_EVAL)
    ppo_mean, ppo_std = evaluate_baseline_actor(
        ppo_actor, trainer.han, eval_env, trainer.n_tasks, trainer.n_relays, trainer.n_mcs, N_EVAL)
    sac_mean, sac_std = evaluate_baseline_actor(
        sac_actor, trainer.han, eval_env, trainer.n_tasks, trainer.n_relays, trainer.n_mcs, N_EVAL)
    static_mean, static_std = evaluate_static(
        eval_env, trainer.n_tasks, trainer.n_relays, trainer.n_mcs, N_EVAL)

    results = {
        "HDM": (hdm_mean, hdm_std),
        "AC": (ac_mean, ac_std),
        "PPO": (ppo_mean, ppo_std),
        "SAC": (sac_mean, sac_std),
        "Static": (static_mean, static_std),
    }

    print(f"\n  tpc={tpc} results:")
    for name, (m, s) in results.items():
        print(f"    {name:<10} ISR={m:.4f} +/- {s:.4f}")

    return results, trainer.history


def main():
    print(f"{'='*60}")
    print(f"CSCA Final Experiments ({POLICY.upper()} policy)")
    print(f"tpc values: {TGC_LIST}")
    print(f"Eval episodes: {N_EVAL}")
    print(f"Started: {timestamp()}")
    print(f"{'='*60}")

    all_results = {}
    all_history = {}

    for tpc in TGC_LIST:
        results, history = run_one_tpc(tpc)
        all_results[tpc] = results
        all_history[tpc] = history

    # Write CSV
    csv_path = os.path.join(RESULTS_DIR, "isr_vs_tpc.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tpc", "policy", "isr_mean", "isr_std", "n_eval"])
        for tpc in TGC_LIST:
            for policy in ["HDM", "AC", "PPO", "SAC", "Static"]:
                m, s = all_results[tpc][policy]
                w.writerow([tpc, policy, f"{m:.4f}", f"{s:.4f}", N_EVAL])
    print(f"\nWrote {csv_path}")

    # Write convergence CSVs
    for tpc in TGC_LIST:
        conv_path = os.path.join(RESULTS_DIR, f"convergence_tpc{tpc}.csv")
        with open(conv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["episode", "train_cscqi", "train_isr", "critic_loss", "actor_loss"])
            for ep, cscqi, isr, closs, aloss in all_history[tpc]:
                w.writerow([ep, f"{cscqi:.4f}", f"{isr:.3f}", f"{closs:.4f}", f"{aloss:.4f}"])
        print(f"Wrote {conv_path}")

    # Plot 1: ISR vs tpc
    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {
        "HDM": {"color": "blue", "marker": "o", "ls": "-", "lw": 2},
        "AC": {"color": "red", "marker": "s", "ls": "--"},
        "PPO": {"color": "green", "marker": "^", "ls": "--"},
        "SAC": {"color": "orange", "marker": "D", "ls": "--"},
        "Static": {"color": "black", "marker": "x", "ls": ":", "lw": 2},
    }
    for policy in ["HDM", "AC", "PPO", "SAC", "Static"]:
        means = [all_results[tpc][policy][0] for tpc in TGC_LIST]
        stds = [all_results[tpc][policy][1] for tpc in TGC_LIST]
        ax.errorbar(TGC_LIST, means, yerr=stds, label=policy,
                     capsize=3, **styles[policy])
    ax.set_xlabel("Tasks per CSCA (tpc)")
    ax.set_ylabel("ISR (Intent Satisfaction Rate)")
    ax.set_title("ISR vs Congestion Level")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(TGC_LIST)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "isr_vs_tpc.png"), dpi=150)
    plt.close()
    print(f"Wrote isr_vs_tpc.png")

    # Plot 2: Convergence at tpc=4
    fig, ax = plt.subplots(figsize=(8, 5))
    hist4 = all_history[4]
    eps = [h[0] for h in hist4]
    train_isr = [h[2] for h in hist4]
    ax.scatter(eps, train_isr, alpha=0.3, s=10, color="gray", label="Train ISR (per-step)")
    # Smooth
    if len(train_isr) > 5:
        window = min(10, len(train_isr) // 3)
        smoothed = np.convolve(train_isr, np.ones(window)/window, mode='valid')
        eps_smooth = eps[window-1:]
        ax.plot(eps_smooth, smoothed, color="blue", lw=2, label="Train ISR (smoothed)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("ISR")
    ax.set_title("HDM Convergence (tpc=4)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "convergence_tpc4.png"), dpi=150)
    plt.close()
    print(f"Wrote convergence_tpc4.png")

    # Plot 3: Improvement bar chart at tpc=4
    fig, ax = plt.subplots(figsize=(8, 5))
    policies = ["HDM", "AC", "PPO", "SAC", "Static"]
    isrs = [all_results[4][p][0] for p in policies]
    colors = ["blue", "red", "green", "orange", "gray"]
    bars = ax.bar(policies, isrs, color=colors, alpha=0.7)
    static_isr = all_results[4]["Static"][0]
    for bar, isr in zip(bars, isrs):
        pct = (isr - static_isr) / max(static_isr, 1e-6) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{pct:+.0f}%", ha="center", fontsize=9)
    ax.set_ylabel("ISR")
    ax.set_title(f"ISR Comparison at tpc=4 ({POLICY.upper()} policy)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "improvement_bar.png"), dpi=150)
    plt.close()
    print(f"Wrote improvement_bar.png")

    # Write SUMMARY.txt
    summary_path = os.path.join(RESULTS_DIR, "SUMMARY.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("CSCA-SemCom Final Results\n")
        f.write(f"Policy: {POLICY.upper()}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        for tpc in TGC_LIST:
            f.write(f"--- tpc={tpc} ({tpc*5} tasks) ---\n")
            for policy in ["HDM", "AC", "PPO", "SAC", "Static"]:
                m, s = all_results[tpc][policy]
                flag = " <-- HDM" if policy == "HDM" else ""
                flag = " <-- baseline" if policy == "Static" else flag
                f.write(f"  {policy:<10} ISR={m:.4f} +/- {s:.4f}{flag}\n")
            hdm = all_results[tpc]["HDM"][0]
            static = all_results[tpc]["Static"][0]
            best_bl = max(all_results[tpc][p][0] for p in ["AC", "PPO", "SAC"])
            f.write(f"  HDM vs Static: {(hdm-static)/max(static,1e-6)*100:+.1f}%\n")
            f.write(f"  HDM vs best RL baseline: {(hdm-best_bl)/max(best_bl,1e-6)*100:+.1f}%\n\n")

        f.write("=" * 60 + "\n")
        hdm4 = all_results[4]["HDM"][0]
        static4 = all_results[4]["Static"][0]
        best_bl4 = max(all_results[4][p][0] for p in ["AC", "PPO", "SAC"])
        f.write(f"HDM improvement over Static at tpc=4: {(hdm4-static4)/max(static4,1e-6)*100:+.1f}%\n")
        f.write(f"HDM improvement over best RL baseline at tpc=4: {(hdm4-best_bl4)/max(best_bl4,1e-6)*100:+.1f}%\n")
        f.write("=" * 60 + "\n")

    print(f"Wrote {summary_path}")
    print(f"\nDone at {timestamp()}. All results in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
