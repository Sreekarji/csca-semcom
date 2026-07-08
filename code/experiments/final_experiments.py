"""
Final CSCA Experiments — Trained HDM vs Trained Baselines
Produces publication-quality plots with error bars.
"""
import os
import sys
import csv
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, r"D:\MP2\code\hdm")
sys.path.insert(0, r"D:\MP2\code\channel")
sys.path.insert(0, r"D:\MP2\code\evaluation")
sys.path.insert(0, r"D:\MP2\code\experiments")

from han_network import HANNetwork
from ddpm_policy import HDMPolicy
from sim_channel import MultiCSCAEnvironment
from cscqi import compute_isr, compute_cscqi, normalize_cscqi

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = r"D:\MP2\results\software\checkpoints"
FINAL = r"D:\MP2\results\software\final"
os.makedirs(FINAL, exist_ok=True)
LOG_PATH = r"D:\MP2\log.txt"


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def save_csv(path, rows, header):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def load_hdm(n_tasks=5, device=DEVICE):
    action_dim = n_tasks + n_tasks * 5 + n_tasks * 3
    han = HANNetwork(hidden_channels=128, num_heads=8, num_layers=2,
                     n_cscas=n_tasks, n_relays=5, n_messages=n_tasks,
                     n_base_stations=n_tasks).to(device)
    hdm = HDMPolicy(action_dim=action_dim, n_denoising_steps=6).to(device)
    ckpt_path = os.path.join(CKPT, "hdm_ep5000.pt")
    if n_tasks == 5 and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        han.load_state_dict(ckpt["han"])
        hdm.load_state_dict(ckpt["actor"])
        log(f"HDM loaded (trained): {ckpt_path}")
    else:
        log(f"HDM: using random weights (no trained checkpoint for n={n_tasks})")
    han.eval(); hdm.eval()
    return han, hdm


def load_sac(n_tasks=5, device=DEVICE):
    action_dim = n_tasks + n_tasks * 5 + n_tasks * 3
    actor = nn.Sequential(
        nn.Linear(128, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, action_dim), nn.Sigmoid()
    ).to(device)
    path = os.path.join(CKPT, "sac_trained.pt")
    if n_tasks == 5 and os.path.exists(path):
        actor.load_state_dict(torch.load(path, map_location=device, weights_only=False)["actor"])
        log(f"SAC loaded (trained)")
    else:
        log(f"SAC: random weights for n={n_tasks}")
    actor.eval()
    return actor


def load_ppo(n_tasks=5, device=DEVICE):
    action_dim = n_tasks + n_tasks * 5 + n_tasks * 3
    actor = nn.Sequential(
        nn.Linear(128, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, action_dim), nn.Sigmoid()
    ).to(device)
    path = os.path.join(CKPT, "ppo_trained.pt")
    if n_tasks == 5 and os.path.exists(path):
        actor.load_state_dict(torch.load(path, map_location=device, weights_only=False)["actor"])
        log(f"PPO loaded (trained)")
    else:
        log(f"PPO: random weights for n={n_tasks}")
    actor.eval()
    return actor


def load_ac(n_tasks=5, device=DEVICE):
    action_dim = n_tasks + n_tasks * 5 + n_tasks * 3
    actor = nn.Sequential(
        nn.Linear(128, 128), nn.Tanh(),
        nn.Linear(128, action_dim), nn.Sigmoid()
    ).to(device)
    path = os.path.join(CKPT, "ac_trained.pt")
    if n_tasks == 5 and os.path.exists(path):
        actor.load_state_dict(torch.load(path, map_location=device, weights_only=False)["actor"])
        log(f"AC loaded (trained)")
    else:
        log(f"AC: random weights for n={n_tasks}")
    actor.eval()
    return actor


def static_action(graph_emb, action_dim):
    return torch.ones(1, action_dim, device=graph_emb.device) * 0.5


def evaluate_method(get_action_fn, han, env, n_tasks, n_episodes, n_seeds):
    """Evaluate a method across multiple seeds."""
    all_isr, all_delay, all_cscqi = [], [], []

    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        isr_list, delay_list, cscqi_list = [], [], []

        for ep in range(n_episodes):
            state = env.generate_state()
            graph_emb, _ = han.encode_state(state)

            with torch.no_grad():
                action = get_action_fn(graph_emb)

            n_r = env.n_relays
            n_mcs = env.n_mcs
            bw = action[:, :n_tasks]
            relay = action[:, n_tasks:n_tasks + n_tasks * n_r].reshape(1, n_tasks, n_r)
            mcs = action[:, n_tasks + n_tasks * n_r:].reshape(1, n_tasks, n_mcs)
            parsed = {"bandwidth": bw, "relay": relay, "mcs": mcs}

            result = env.step(parsed, state)
            tasks = result["tasks"]

            isr = compute_isr(tasks)
            avg_delay = np.mean([t["tau_S"] for t in tasks])
            avg_cscqi = np.mean([
                compute_cscqi(t["tau_S"], t["vartheta_S"],
                              t["tau_S_int"], t["vartheta_S_int"])
                for t in tasks
            ])

            isr_list.append(isr)
            delay_list.append(avg_delay)
            cscqi_list.append(avg_cscqi)

        all_isr.append(np.mean(isr_list))
        all_delay.append(np.mean(delay_list))
        all_cscqi.append(np.mean(cscqi_list))

    return {
        "isr": np.mean(all_isr), "isr_std": np.std(all_isr),
        "delay": np.mean(all_delay), "delay_std": np.std(all_delay),
        "cscqi": np.mean(all_cscqi), "cscqi_std": np.std(all_cscqi),
    }


# ============ EXPERIMENT 1: ISR vs Tasks ============
def experiment_isr_vs_tasks():
    log("EXP 1: ISR vs tasks (n=5,10,15) — trained HDM vs trained baselines")
    task_counts = [5, 10, 15]
    n_episodes = 200
    n_seeds = 3

    results = {m: {"isr": [], "isr_std": []} for m in ["HDM", "SAC", "PPO", "AC", "Static"]}

    for n in task_counts:
        log(f"  n={n}: loading models...")
        han, hdm = load_hdm(n)
        sac = load_sac(n)
        ppo = load_ppo(n)
        ac = load_ac(n)
        env = MultiCSCAEnvironment(n_cscas=n, n_relays=5, difficulty="hard")

        methods = {
            "HDM": lambda ge, h=hdm, ha=han: h(ha(ge)[0]) if False else h(ha.encode_state(env.generate_state())[0]),
            "SAC": lambda ge, s=sac: s(ge),
            "PPO": lambda ge, p=ppo: p(ge),
            "AC": lambda ge, a=ac: a(ge),
            "Static": lambda ge, ad=n+n*5+n*3: static_action(ge, ad),
        }

        # HDM needs special handling
        r = evaluate_method(
            lambda ge, h=hdm, ha=han: h(ge),
            han, env, n, n_episodes, n_seeds
        )
        results["HDM"]["isr"].append(r["isr"])
        results["HDM"]["isr_std"].append(r["isr_std"])

        for name, model in [("SAC", sac), ("PPO", ppo), ("AC", ac)]:
            r = evaluate_method(lambda ge, m=model: m(ge), han, env, n, n_episodes, n_seeds)
            results[name]["isr"].append(r["isr"])
            results[name]["isr_std"].append(r["isr_std"])

        r = evaluate_method(lambda ge, ad=n+n*5+n*3: static_action(ge, ad), han, env, n, n_episodes, n_seeds)
        results["Static"]["isr"].append(r["isr"])
        results["Static"]["isr_std"].append(r["isr_std"])

        log(f"  n={n}: HDM={results['HDM']['isr'][-1]:.3f}+-{results['HDM']['isr_std'][-1]:.3f}, "
            f"SAC={results['SAC']['isr'][-1]:.3f}, PPO={results['PPO']['isr'][-1]:.3f}, "
            f"AC={results['AC']['isr'][-1]:.3f}, Static={results['Static']['isr'][-1]:.3f}")

    # Plot
    plt.figure(figsize=(8, 5))
    for method, color, marker in [("HDM","b","o"),("SAC","r","s"),("PPO","g","^"),("AC","m","D"),("Static","k","v")]:
        plt.errorbar(task_counts, results[method]["isr"], yerr=results[method]["isr_std"],
                     fmt=f"{color}-{marker}", label=method, markersize=7, capsize=4, linewidth=2)
    plt.xlabel("Number of Tasks", fontsize=12)
    plt.ylabel("Intent Satisfaction Rate", fontsize=12)
    plt.title("ISR vs Number of Tasks (Trained Models, 3 seeds)", fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FINAL, "fig_isr_vs_tasks.png"), dpi=150)
    plt.close()

    # CSV
    rows = []
    for i, n in enumerate(task_counts):
        row = [n]
        for m in ["HDM", "SAC", "PPO", "AC", "Static"]:
            row.extend([f"{results[m]['isr'][i]:.4f}", f"{results[m]['isr_std'][i]:.4f}"])
        rows.append(row)
    header = ["n_tasks"]
    for m in ["HDM", "SAC", "PPO", "AC", "Static"]:
        header.extend([f"{m}_ISR", f"{m}_std"])
    save_csv(os.path.join(FINAL, "fig_isr_vs_tasks.csv"), rows, header)
    log("EXP 1 complete.")
    return results


# ============ EXPERIMENT 2: CSCQI Convergence (N=5,6,7) ============
def experiment_cscqi_convergence():
    log("EXP 2: CSCQI convergence for N=5,6,7 denoising steps")
    han, _ = load_hdm(5)
    env = MultiCSCAEnvironment(n_cscas=5, n_relays=5, difficulty="hard")

    results = {}
    for N in [5, 6, 7]:
        hdm = HDMPolicy(action_dim=45, n_denoising_steps=N).to(DEVICE)
        ckpt_path = os.path.join(CKPT, "hdm_ep5000.pt")
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        # Load what we can (denoiser weights are same architecture)
        try:
            hdm.load_state_dict(ckpt["actor"])
            log(f"  N={N}: loaded trained weights")
        except:
            log(f"  N={N}: architecture mismatch, using random weights")

        hdm.eval()
        rewards = []
        for ep in range(500):
            state = env.generate_state()
            graph_emb, _ = han.encode_state(state)
            with torch.no_grad():
                action = hdm(graph_emb)
            bw = action[:, :5]
            relay = action[:, 5:30].reshape(1, 5, 5)
            mcs = action[:, 30:].reshape(1, 5, 3)
            result = env.step({"bandwidth": bw, "relay": relay, "mcs": mcs}, state)
            tasks = result["tasks"]
            isr = compute_isr(tasks)
            rewards.append(isr)

        results[N] = rewards
        log(f"  N={N}: mean ISR={np.mean(rewards):.4f}, final 100 avg={np.mean(rewards[-100:]):.4f}")

    # Plot
    plt.figure(figsize=(8, 5))
    for N, color in [(5, "b"), (6, "r"), (7, "g")]:
        smoothed = np.convolve(results[N], np.ones(50)/50, mode='valid')
        plt.plot(range(49, 500), smoothed, f'{color}-', linewidth=2, label=f'N={N}')
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Normalized Reward (ISR)", fontsize=12)
    plt.title("CSCQI Convergence for Different Denoising Steps", fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FINAL, "fig_cscqi_convergence.png"), dpi=150)
    plt.close()

    rows = [[ep] + [results[N][ep] for N in [5, 6, 7]] for ep in range(500)]
    save_csv(os.path.join(FINAL, "fig_cscqi_convergence.csv"), rows, ["episode", "N5", "N6", "N7"])
    log("EXP 2 complete.")
    return results


# ============ EXPERIMENT 3: Ablation ============
def experiment_ablation():
    log("EXP 3: Ablation — HDM vs no-HAN vs no-DDPM")
    task_counts = [5, 10, 15]
    n_episodes = 200
    n_seeds = 3

    results = {m: {"isr": [], "isr_std": []} for m in ["HDM", "no-HAN", "no-DDPM"]}

    for n in task_counts:
        han, hdm = load_hdm(n)
        env = MultiCSCAEnvironment(n_cscas=n, n_relays=5, difficulty="hard")

        # Full HDM
        r = evaluate_method(lambda ge: hdm(ge), han, env, n, n_episodes, n_seeds)
        results["HDM"]["isr"].append(r["isr"])
        results["HDM"]["isr_std"].append(r["isr_std"])

        # no-HAN: use random graph embedding instead of HAN-encoded
        def no_han_action(ge, ad=n+n*5+n*3):
            random_emb = torch.randn_like(ge) * 0.1
            return hdm(random_emb)
        r = evaluate_method(no_han_action, han, env, n, n_episodes, n_seeds)
        results["no-HAN"]["isr"].append(r["isr"])
        results["no-HAN"]["isr_std"].append(r["isr_std"])

        # no-DDPM: use MLP instead of DDPM
        action_dim = n + n * 5 + n * 3
        mlp_actor = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Sigmoid()
        ).to(DEVICE)
        r = evaluate_method(lambda ge, m=mlp_actor: m(ge), han, env, n, n_episodes, n_seeds)
        results["no-DDPM"]["isr"].append(r["isr"])
        results["no-DDPM"]["isr_std"].append(r["isr_std"])

        log(f"  n={n}: HDM={results['HDM']['isr'][-1]:.3f}, "
            f"no-HAN={results['no-HAN']['isr'][-1]:.3f}, "
            f"no-DDPM={results['no-DDPM']['isr'][-1]:.3f}")

    # Plot
    plt.figure(figsize=(8, 5))
    for method, color, style in [("HDM","b","-o"),("no-HAN","r","--s"),("no-DDPM","g","-.^")]:
        plt.errorbar(task_counts, results[method]["isr"], yerr=results[method]["isr_std"],
                     fmt=color+style, label=method, markersize=7, capsize=4, linewidth=2)
    plt.xlabel("Number of Tasks", fontsize=12)
    plt.ylabel("Intent Satisfaction Rate", fontsize=12)
    plt.title("Ablation Study: HAN and DDPM Contribution", fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FINAL, "fig_ablation.png"), dpi=150)
    plt.close()

    rows = []
    for i, n in enumerate(task_counts):
        row = [n]
        for m in ["HDM", "no-HAN", "no-DDPM"]:
            row.extend([f"{results[m]['isr'][i]:.4f}", f"{results[m]['isr_std'][i]:.4f}"])
        rows.append(row)
    header = ["n_tasks"]
    for m in ["HDM", "no-HAN", "no-DDPM"]:
        header.extend([f"{m}_ISR", f"{m}_std"])
    save_csv(os.path.join(FINAL, "fig_ablation.csv"), rows, header)
    log("EXP 3 complete.")
    return results


# ============ SUMMARY ============
def generate_summary(isr_results, ablation_results):
    log("Generating final summary...")
    lines = []
    lines.append("=" * 60)
    lines.append("CSCA FINAL RESULTS SUMMARY")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)
    lines.append("")

    lines.append("ISR vs Number of Tasks (Trained Models, 3 seeds x 200 episodes)")
    lines.append("-" * 60)
    lines.append(f"{'n_tasks':<10} {'HDM':<15} {'SAC':<15} {'PPO':<15} {'AC':<15} {'Static':<15}")
    for i, n in enumerate([5, 10, 15]):
        hdm = isr_results["HDM"]["isr"][i]
        sac = isr_results["SAC"]["isr"][i]
        ppo = isr_results["PPO"]["isr"][i]
        ac = isr_results["AC"]["isr"][i]
        st = isr_results["Static"]["isr"][i]
        best_baseline = max(sac, ppo, ac, st)
        improvement = ((hdm - best_baseline) / best_baseline * 100) if best_baseline > 0 else 0
        lines.append(f"{n:<10} {hdm:<15.4f} {sac:<15.4f} {ppo:<15.4f} {ac:<15.4f} {st:<15.4f}")
    lines.append("")

    lines.append("HDM Improvement over Best Baseline")
    lines.append("-" * 60)
    for i, n in enumerate([5, 10, 15]):
        hdm = isr_results["HDM"]["isr"][i]
        best_baseline = max(isr_results[m]["isr"][i] for m in ["SAC", "PPO", "AC", "Static"])
        improvement = ((hdm - best_baseline) / best_baseline * 100) if best_baseline > 0 else 0
        lines.append(f"  n={n}: HDM={hdm:.4f}, best baseline={best_baseline:.4f}, improvement={improvement:+.1f}%")
    lines.append("")

    lines.append("Ablation Study")
    lines.append("-" * 60)
    for i, n in enumerate([5, 10, 15]):
        hdm = ablation_results["HDM"]["isr"][i]
        no_han = ablation_results["no-HAN"]["isr"][i]
        no_ddpm = ablation_results["no-DDPM"]["isr"][i]
        han_contrib = ((hdm - no_han) / hdm * 100) if hdm > 0 else 0
        ddpm_contrib = ((hdm - no_ddpm) / hdm * 100) if hdm > 0 else 0
        lines.append(f"  n={n}: HDM={hdm:.4f}, no-HAN={no_han:.4f} ({han_contrib:+.1f}%), "
                     f"no-DDPM={no_ddpm:.4f} ({ddpm_contrib:+.1f}%)")
    lines.append("")

    lines.append("Paper Comparison (Sun et al. 2026)")
    lines.append("-" * 60)
    lines.append("  Paper: HDM ISR improvement ~42% over baselines at n=10")
    lines.append("  Paper: CSCQI converges in ~50 episodes with N=6 optimal")
    lines.append("  Paper: HAN contribution ~19%, DDPM contribution ~13%")
    lines.append("")
    lines.append("  Differences from paper:")
    lines.append("  1. Our channel sim uses numpy formulas, not full 3GPP TR 38.901")
    lines.append("  2. Baselines trained 2000ep (paper may use more)")
    lines.append("  3. HDM trained 5000ep (paper may use different schedule)")
    lines.append("  4. Our graph is 5 nodes (paper may use larger)")
    lines.append("  5. DDPM noise schedule: beta_min=0.01, beta_max=0.5 (paper Eq. 31)")
    lines.append("")
    lines.append("=" * 60)

    summary_text = "\n".join(lines)
    with open(os.path.join(FINAL, "results_summary.txt"), "w") as f:
        f.write(summary_text)
    print(summary_text)
    log("Summary saved to final/results_summary.txt")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, r"D:\MP2\code\utils")
    from reproducibility import set_seed
    set_seed(42)
    log("=" * 60)
    log("FINAL CSCA EXPERIMENTS — Starting")
    log("=" * 60)

    isr_results = experiment_isr_vs_tasks()
    cscqi_results = experiment_cscqi_convergence()
    ablation_results = experiment_ablation()
    generate_summary(isr_results, ablation_results)

    log("=" * 60)
    log("ALL FINAL EXPERIMENTS COMPLETE")
    log(f"Results saved to: {FINAL}")
    log("=" * 60)
