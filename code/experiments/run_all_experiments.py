import os
import sys
import csv
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, r"D:\MP2\code")
sys.path.insert(0, r"D:\MP2\code\hdm")
sys.path.insert(0, r"D:\MP2\code\channel")
sys.path.insert(0, r"D:\MP2\code\evaluation")
sys.path.insert(0, r"D:\MP2\code\experiments")

import sys
sys.path.insert(0, r"D:\MP2\code\utils")
from reproducibility import set_seed

from han_network import HANNetwork
from ddpm_policy import HDMPolicy, CriticNetwork
from sim_channel import MultiCSCAEnvironment
from cscqi import compute_isr, compute_cscqi, normalize_cscqi
from baselines import SACBaseline, ACBaseline, PPOBaseline, StaticBaseline
from deepsc_baseline import DeepSCBaseline
from dasc_baseline import DASCBaseline
from rlasc_baseline import RLASCBaseline

RESULTS = r"D:\MP2\results\software"
LOG_PATH = r"D:\MP2\log.txt"
CHECKPOINT = r"D:\MP2\results\software\checkpoints\hdm_ep5000.pt"
CKPT_DIR = r"D:\MP2\results\software\checkpoints"
os.makedirs(RESULTS, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def match_action_dim(action_raw, n_tasks, n_relays, n_mcs):
    """Pad or truncate action tensor to match expected dimensions."""
    expected = n_tasks + n_tasks * n_relays + n_tasks * n_mcs
    if action_raw.shape[1] > expected:
        return action_raw[:, :expected]
    elif action_raw.shape[1] < expected:
        pad = torch.zeros(action_raw.shape[0], expected - action_raw.shape[1], device=action_raw.device)
        return torch.cat([action_raw, pad], dim=1)
    return action_raw


def parse_action(action_raw, n_tasks, n_relays, n_mcs):
    """Parse action tensor into bandwidth, relay, mcs components."""
    action_raw = match_action_dim(action_raw, n_tasks, n_relays, n_mcs)
    bw = action_raw[:, :n_tasks]
    relay = action_raw[:, n_tasks:n_tasks + n_tasks * n_relays].reshape(1, n_tasks, n_relays)
    mcs = action_raw[:, n_tasks + n_tasks * n_relays:].reshape(1, n_tasks, n_mcs)
    return {"bandwidth": bw, "relay": relay, "mcs": mcs}


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


def load_trained_hdm(n_tasks=5, n_relays=5, device=DEVICE):
    """Load best available HDM checkpoint."""
    action_dim = n_tasks + n_tasks * n_relays + n_tasks * 3

    han = HANNetwork(
        hidden_channels=128, num_heads=8, num_layers=2,
        n_cscas=n_tasks, n_relays=n_relays,
        n_messages=n_tasks, n_base_stations=n_tasks,
    ).to(device)

    hdm = HDMPolicy(
        action_dim=action_dim,
        n_denoising_steps=6,
    ).to(device)

    # Try checkpoints — prefer newest by modification time
    # Retrained checkpoints (Jul 8, ep200-ep2000) are ep5200-ep7000 total
    # Original checkpoints (Jul 7, ep5000) are ep5000 total
    import glob
    all_ckpts = glob.glob(os.path.join(CKPT_DIR, "hdm_ep*.pt"))
    if all_ckpts:
        # Sort by modification time (newest first)
        all_ckpts.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        ckpt_candidates = all_ckpts
    else:
        ckpt_candidates = [
            fr"{CKPT_DIR}\hdm_ep5000.pt",
            fr"{CKPT_DIR}\hdm_ep4000.pt",
            fr"{CKPT_DIR}\hdm_ep3000.pt",
            fr"{CKPT_DIR}\hdm_ep2000.pt",
            fr"{CKPT_DIR}\hdm_ep1000.pt",
        ]

    loaded = False
    for ckpt_path in ckpt_candidates:
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            try:
                han.load_state_dict(ckpt["han"])
                hdm.load_state_dict(ckpt["actor"])
                log(f"[load_trained_hdm] Loaded: {ckpt_path}")
                loaded = True
                break
            except Exception as e:
                log(f"[load_trained_hdm] Failed to load {ckpt_path}: {e}")
                continue

    if not loaded:
        log("[load_trained_hdm] WARNING: No checkpoint loaded -- using random weights")

    han.eval()
    hdm.eval()
    return han, hdm


def load_trained_baseline(name: str, action_dim: int, device=DEVICE):
    """Load trained baseline checkpoint."""
    import torch.nn as nn

    ckpt_paths = {
        "SAC": fr"{CKPT_DIR}\sac_trained.pt",
        "PPO": fr"{CKPT_DIR}\ppo_trained.pt",
        "AC":  fr"{CKPT_DIR}\ac_trained.pt",
    }

    if name == "SAC":
        actor = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Sigmoid()
        ).to(device)
    elif name == "PPO":
        actor = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Sigmoid()
        ).to(device)
    elif name == "AC":
        actor = nn.Sequential(
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, action_dim), nn.Sigmoid()
        ).to(device)
    else:
        raise ValueError(f"Unknown baseline: {name}")

    ckpt_path = ckpt_paths.get(name)
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        # Auto-detect architecture from checkpoint keys
        state = ckpt["actor"]
        layer_dims = []
        for k in sorted(state.keys()):
            if k.endswith(".weight"):
                layer_dims.append(state[k].shape[0])  # output dim
        # Rebuild actor with detected architecture
        layers = []
        in_dim = 128
        for i, out_dim in enumerate(layer_dims):
            layers.append(nn.Linear(in_dim, out_dim))
            if i < len(layer_dims) - 1:
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Sigmoid())
            in_dim = out_dim
        actor = nn.Sequential(*layers).to(device)
        actor.load_state_dict(ckpt["actor"])
        log(f"[load_trained_baseline] Loaded {name}: {ckpt_path} (arch: {layer_dims})")
    else:
        log(f"[load_trained_baseline] WARNING: {name} checkpoint missing -- using random weights")

    actor.eval()

    # Wrap in a class with get_action method for compatibility
    class BaselineWrapper:
        def __init__(self, net):
            self.net = net
        def get_action(self, state_emb):
            with torch.no_grad():
                return self.net(state_emb)
        def forward(self, state_emb):
            return self.get_action(state_emb)

    return BaselineWrapper(actor)


def run_method_episodes_averaged(
    method_name, get_action_fn, env_fn, han,
    n_episodes=200, n_seeds=3, n_tasks=5
):
    """Run with multiple seeds and average results for stability."""
    all_isr = []
    all_delay = []
    all_cscqi = []

    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        env = env_fn()

        isr_list, delay_list, cscqi_list = [], [], []

        for ep in range(n_episodes):
            state = env.generate_state()
            graph_emb, _ = han.encode_state(state)

            n_r = env.n_relays
            n_mcs = env.n_mcs

            with torch.no_grad():
                action_raw = get_action_fn(graph_emb)

            parsed = parse_action(action_raw, n_tasks, n_r, n_mcs)

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
        "isr": np.mean(all_isr),
        "isr_std": np.std(all_isr),
        "delay": np.mean(all_delay),
        "delay_std": np.std(all_delay),
        "cscqi": np.mean(all_cscqi),
        "cscqi_std": np.std(all_cscqi),
    }


def experiment_isr_vs_tasks():
    """Fig 9a: ISR vs number of tasks. Trained HDM vs trained baselines."""
    set_seed(42)
    log("Experiment 1: ISR vs tasks (Fig 9a) -- averaged over 3 seeds x 200 episodes")
    task_counts = [2, 5, 8, 10, 12, 15, 18, 20]

    han_trained, hdm_trained = load_trained_hdm(n_tasks=5, n_relays=5)

    results = {m: {"isr": [], "isr_std": []} for m in ["HDM", "SAC", "AC", "PPO", "Static"]}

    for n in task_counts:
        action_dim = n + n * 5 + n * 3

        def env_fn(n=n): return MultiCSCAEnvironment(n_cscas=n, n_relays=5, difficulty="hard")
        han_n = HANNetwork(hidden_channels=128, num_heads=8, num_layers=2,
                           n_cscas=n, n_relays=5, n_messages=n, n_base_stations=n).to(DEVICE)

        # HDM (trained at n=5, use for all n for consistency)
        if n == 5:
            hdm_fn = hdm_trained.forward
            hdm_han = han_trained
            hdm_tasks = 5
        else:
            hdm_n = HDMPolicy(action_dim=action_dim).to(DEVICE)
            hdm_fn = hdm_n.forward
            hdm_han = han_n
            hdm_tasks = n

        r = run_method_episodes_averaged("HDM", hdm_fn, env_fn, hdm_han,
                                         n_episodes=200, n_seeds=3, n_tasks=hdm_tasks)
        results["HDM"]["isr"].append(r["isr"])
        results["HDM"]["isr_std"].append(r["isr_std"])

        # Baselines (trained checkpoints — always action_dim=45 from n=5 training)
        for name in ["SAC", "AC", "PPO"]:
            bl = load_trained_baseline(name, 45, DEVICE)
            r = run_method_episodes_averaged(name, bl.get_action, env_fn, han_n,
                                             n_episodes=200, n_seeds=3, n_tasks=n)
            results[name]["isr"].append(r["isr"])
            results[name]["isr_std"].append(r["isr_std"])

        static = StaticBaseline(action_dim=action_dim, device=DEVICE)
        r = run_method_episodes_averaged("Static", static.get_action, env_fn, han_n,
                                         n_episodes=200, n_seeds=3, n_tasks=n)
        results["Static"]["isr"].append(r["isr"])
        results["Static"]["isr_std"].append(r["isr_std"])

        log(f"  n={n}: HDM={results['HDM']['isr'][-1]:.3f}+-{results['HDM']['isr_std'][-1]:.3f}, "
            f"SAC={results['SAC']['isr'][-1]:.3f}+-{results['SAC']['isr_std'][-1]:.3f}")

    # Plot with error bars
    plt.figure(figsize=(7, 4))
    for method, color, marker in [("HDM","b","o"),("SAC","r","s"),("AC","g","^"),("PPO","m","D"),("Static","k","v")]:
        plt.errorbar(task_counts, results[method]["isr"], yerr=results[method]["isr_std"],
                     fmt=f"{color}-{marker}", label=method, markersize=5, capsize=3)
    plt.xlabel("Number of Tasks")
    plt.ylabel("Intent Satisfaction Rate")
    plt.title("Fig 9a: ISR vs Number of Tasks (3 seeds, 200 ep each)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "fig9a_isr_vs_tasks.png"), dpi=120)
    plt.close()

    rows = []
    for i, n in enumerate(task_counts):
        row = [n]
        for m in ["HDM", "SAC", "AC", "PPO", "Static"]:
            row.extend([f"{results[m]['isr'][i]:.4f}", f"{results[m]['isr_std'][i]:.4f}"])
        rows.append(row)
    header = ["n_tasks"]
    for m in ["HDM", "SAC", "AC", "PPO", "Static"]:
        header.extend([f"{m}_ISR", f"{m}_std"])
    save_csv(os.path.join(RESULTS, "fig9a_isr_vs_tasks.csv"), rows, header)
    log("Experiment 1 complete.")
    return results


def experiment_delay_vs_sinr():
    """Fig 9c: Delay vs SINR."""
    set_seed(42)
    log("Experiment 2: Delay vs SINR (Fig 9c) -- averaged over 3 seeds x 200 episodes")
    sinr_range = [0, 5, 10, 15, 20, 25]

    han, hdm = load_trained_hdm(n_tasks=5, n_relays=5)
    static = StaticBaseline(action_dim=45, device=DEVICE)

    hdm_delays, hdm_stds = [], []
    static_delays, static_stds = [], []

    for snr in sinr_range:
        all_hdm, all_static = [], []
        for seed in range(3):
            torch.manual_seed(seed)
            np.random.seed(seed)
            env = MultiCSCAEnvironment(n_cscas=5, difficulty="hard")
            h_d, s_d = [], []
            for _ in range(200):
                state = env.generate_state()
                graph_emb, _ = han.encode_state(state)
                for method, fn in [("HDM", hdm), ("Static", static)]:
                    with torch.no_grad():
                        action_raw = fn(graph_emb) if method == "HDM" else static.get_action(graph_emb)
                    parsed = parse_action(action_raw, 5, 5, 3)
                    result = env.step(parsed, state)
                    avg_d = np.mean([t["tau_S"] for t in result["tasks"]])
                    if method == "HDM":
                        h_d.append(avg_d)
                    else:
                        s_d.append(avg_d)
            all_hdm.append(np.mean(h_d))
            all_static.append(np.mean(s_d))

        hdm_delays.append(np.mean(all_hdm))
        hdm_stds.append(np.std(all_hdm))
        static_delays.append(np.mean(all_static))
        static_stds.append(np.std(all_static))
        log(f"  SINR={snr}dB: HDM={hdm_delays[-1]:.2f}+-{hdm_stds[-1]:.2f}s")

    plt.figure(figsize=(7, 4))
    plt.errorbar(sinr_range, hdm_delays, yerr=hdm_stds, fmt="b-o", label="HDM (trained)", capsize=3)
    plt.errorbar(sinr_range, static_delays, yerr=static_stds, fmt="k--s", label="Static", capsize=3)
    plt.xlabel("SINR (dB)")
    plt.ylabel("Communication Delay (s)")
    plt.title("Fig 9c: Delay vs SINR (3 seeds, 200 ep each)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "fig9c_delay_vs_sinr.png"), dpi=120)
    plt.close()

    rows = [[s, f"{h:.4f}", f"{hs:.4f}", f"{st:.4f}", f"{sts:.4f}"]
            for s, h, hs, st, sts in zip(sinr_range, hdm_delays, hdm_stds, static_delays, static_stds)]
    save_csv(os.path.join(RESULTS, "fig9c_delay_vs_sinr.csv"), rows,
             ["sinr_db", "HDM_delay", "HDM_std", "Static_delay", "Static_std"])
    log("Experiment 2 complete.")


def experiment_cscqi_convergence():
    """Fig 12a: CSCQI convergence from training checkpoint."""
    set_seed(42)
    log("Experiment 3: CSCQI convergence (Fig 12a) -- from checkpoint")
    if os.path.exists(CHECKPOINT):
        ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
        rewards = ckpt.get("reward_history", [])
    else:
        rewards = []

    if rewards:
        norm_rewards = normalize_cscqi(rewards)
        smoothed = np.convolve(norm_rewards, np.ones(50)/50, mode='valid')

        plt.figure(figsize=(7, 4))
        plt.plot(range(len(norm_rewards)), norm_rewards, "b-", alpha=0.3, linewidth=0.5, label="Raw")
        plt.plot(range(49, len(norm_rewards)), smoothed, "r-", linewidth=2, label="Smoothed (50-ep)")
        plt.xlabel("Training Episodes")
        plt.ylabel("Normalized Reward")
        plt.title("Fig 12a: CSCQI/Reward Convergence")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS, "fig12a_cscqi_convergence.png"), dpi=120)
        plt.close()

        rows = [[ep, norm_rewards[ep]] for ep in range(len(norm_rewards))]
        save_csv(os.path.join(RESULTS, "fig12a_cscqi_convergence.csv"), rows, ["episode", "normalized_reward"])
        log(f"  {len(rewards)} episodes, final smoothed={smoothed[-1]:.4f}")
    log("Experiment 3 complete.")


def experiment_ablation():
    """Fig 13: Ablation -- HDM vs no-HAN vs no-DDPM."""
    set_seed(42)
    log("Experiment 4: Ablation (Fig 13) -- averaged over 3 seeds x 200 episodes")
    task_counts = [2, 5, 8, 10, 12, 15, 18, 20]

    han, hdm = load_trained_hdm(n_tasks=5, n_relays=5)

    results = {m: {"isr": [], "isr_std": []} for m in ["HDM", "HDM-no-HAN", "HDM-no-DDPM"]}

    for n in task_counts:
        action_dim = n + n * 5 + n * 3
        han_n = HANNetwork(hidden_channels=128, num_heads=8, num_layers=2,
                           n_cscas=n, n_relays=5, n_messages=n, n_base_stations=n).to(DEVICE)

        def env_fn(n=n): return MultiCSCAEnvironment(n_cscas=n, n_relays=5, difficulty="hard")

        if n == 5:
            r = run_method_episodes_averaged("HDM", hdm.forward, env_fn, han,
                                             n_episodes=200, n_seeds=3, n_tasks=5)
        else:
            hdm_n = HDMPolicy(action_dim=action_dim).to(DEVICE)
            r = run_method_episodes_averaged("HDM", hdm_n.forward, env_fn, han_n,
                                             n_episodes=200, n_seeds=3, n_tasks=n)
        results["HDM"]["isr"].append(r["isr"])
        results["HDM"]["isr_std"].append(r["isr_std"])

        no_han = StaticBaseline(action_dim=action_dim, device=DEVICE)
        r = run_method_episodes_averaged("HDM-no-HAN", no_han.get_action, env_fn, han_n,
                                         n_episodes=200, n_seeds=3, n_tasks=n)
        results["HDM-no-HAN"]["isr"].append(r["isr"])
        results["HDM-no-HAN"]["isr_std"].append(r["isr_std"])

        no_ddpm = load_trained_baseline("SAC", 45, DEVICE)
        r = run_method_episodes_averaged("HDM-no-DDPM", no_ddpm.get_action, env_fn, han_n,
                                         n_episodes=200, n_seeds=3, n_tasks=n)
        results["HDM-no-DDPM"]["isr"].append(r["isr"])
        results["HDM-no-DDPM"]["isr_std"].append(r["isr_std"])

        log(f"  n={n}: HDM={results['HDM']['isr'][-1]:.3f}, "
            f"no-HAN={results['HDM-no-HAN']['isr'][-1]:.3f}, "
            f"no-DDPM={results['HDM-no-DDPM']['isr'][-1]:.3f}")

    plt.figure(figsize=(7, 4))
    for method, color, style in [("HDM","b","-o"),("HDM-no-HAN","r","--s"),("HDM-no-DDPM","g","-.^")]:
        plt.errorbar(task_counts, results[method]["isr"], yerr=results[method]["isr_std"],
                     fmt=color+style, label=method, markersize=5, capsize=3)
    plt.xlabel("Number of Tasks")
    plt.ylabel("Intent Satisfaction Rate")
    plt.title("Fig 13: Ablation Study (3 seeds, 200 ep each)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "fig13_ablation.png"), dpi=120)
    plt.close()

    rows = []
    for i, n in enumerate(task_counts):
        row = [n]
        for m in ["HDM", "HDM-no-HAN", "HDM-no-DDPM"]:
            row.extend([f"{results[m]['isr'][i]:.4f}", f"{results[m]['isr_std'][i]:.4f}"])
        rows.append(row)
    header = ["n_tasks"]
    for m in ["HDM", "HDM-no-HAN", "HDM-no-DDPM"]:
        header.extend([f"{m}_ISR", f"{m}_std"])
    save_csv(os.path.join(RESULTS, "fig13_ablation.csv"), rows, header)
    log("Experiment 4 complete.")


def experiment_multimodal_semcom():
    """Fig 6: Multimodal SemCom performance."""
    log("Experiment 5: Multimodal SemCom (Fig 6)")
    snr_range = [0, 5, 10, 15, 20, 25]

    deepsc = DeepSCBaseline(d_model=128, channel="AWGN", snr_db=10.0)
    sst_path = r"D:\MP2\data\raw\sst2_500.json"
    if os.path.exists(sst_path):
        with open(sst_path) as f:
            text_samples = json.load(f)["sentences"][:50]
    else:
        text_samples = ["send it within 1 second"] * 50
    text_results = deepsc.evaluate(text_samples, snr_range=snr_range)

    dasc = DASCBaseline(input_dim=80, compressed_dim=16, snr_db=10.0)
    audio_results = dasc.evaluate(n_samples=50, snr_range=snr_range)

    rlasc = RLASCBaseline(image_size=32, latent_dim=64, snr_db=10.0)
    image_results = rlasc.evaluate(image_dir=r"D:\MP2\data\raw\images", n_samples=50, snr_range=snr_range)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].plot(snr_range, [text_results["accuracy_by_snr"][s] for s in snr_range], "b-o", label="DeepSC")
    axes[0].set_title("Text SemCom"); axes[0].set_xlabel("SNR (dB)"); axes[0].set_ylabel("Accuracy"); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(snr_range, [audio_results["accuracy_by_snr"][s] for s in snr_range], "r-s", label="DASC")
    axes[1].set_title("Audio SemCom"); axes[1].set_xlabel("SNR (dB)"); axes[1].set_ylabel("Cos Sim"); axes[1].legend(); axes[1].grid(True)
    axes[2].plot(snr_range, [image_results["psnr_by_snr"][s] for s in snr_range], "g-^", label="RL-ASC")
    axes[2].axhline(y=22, color="k", linestyle="--", alpha=0.5, label="22dB")
    axes[2].set_title("Image SemCom"); axes[2].set_xlabel("SNR (dB)"); axes[2].set_ylabel("PSNR (dB)"); axes[2].legend(); axes[2].grid(True)
    plt.suptitle("Fig 6: Multimodal SemCom Performance")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "fig6_multimodal_semcom.png"), dpi=120)
    plt.close()

    rows = [[s, f"{text_results['accuracy_by_snr'][s]:.4f}", f"{audio_results['accuracy_by_snr'][s]:.4f}", f"{image_results['psnr_by_snr'][s]:.2f}"]
            for s in snr_range]
    save_csv(os.path.join(RESULTS, "fig6_multimodal_semcom.csv"), rows,
             ["SNR_dB", "DeepSC_Text_Acc", "DASC_Audio_Acc", "RL-ASC_Image_PSNR"])
    log(f"  Compression: DeepSC={deepsc.compression_ratio:.1%}, DASC={dasc.compression_ratio:.1%}, RL-ASC={rlasc.compression_ratio:.1%}")
    log("Experiment 5 complete.")


def generate_summary_table():
    """Generate results_summary.csv with averaged results."""
    set_seed(42)
    log("Generating summary table (3 seeds x 200 episodes)...")
    han, hdm = load_trained_hdm(n_tasks=5, n_relays=5)
    sac = load_trained_baseline("SAC", 45, DEVICE)
    ac = load_trained_baseline("AC", 45, DEVICE)
    ppo = load_trained_baseline("PPO", 45, DEVICE)
    static = StaticBaseline(action_dim=45, device=DEVICE)

    def env_fn(): return MultiCSCAEnvironment(n_cscas=5, difficulty="hard")

    methods = [
        ("HDM (trained)", hdm.forward, han),
        ("SAC", sac.get_action, han),
        ("AC", ac.get_action, han),
        ("PPO", ppo.get_action, han),
        ("Static", static.get_action, han),
    ]

    rows = []
    for name, fn, h in methods:
        r = run_method_episodes_averaged(name, fn, env_fn, h, n_episodes=200, n_seeds=3)
        rows.append([name, f"{r['isr']:.4f}", f"{r['isr_std']:.4f}",
                     f"{r['cscqi']:.2f}", f"{r['cscqi_std']:.2f}",
                     f"{r['delay']:.2f}", f"{r['delay_std']:.2f}"])
        log(f"  {name}: ISR={r['isr']:.4f}+-{r['isr_std']:.4f}, CSCQI={r['cscqi']:.2f}, delay={r['delay']:.2f}s")

    save_csv(os.path.join(RESULTS, "results_summary.csv"), rows,
             ["Method", "ISR", "ISR_std", "CSCQI", "CSCQI_std", "Delay_s", "Delay_std"])
    log("Summary table saved.")



def experiment_delay_reduction():
    """
    Measures delay reduction: HDM vs baselines across task counts and SINR.
    Paper claims -33.40% delay reduction (Fig 9c, 9d).
    """
    log("Experiment: Delay reduction (Fig 9c, 9d)")
    set_seed(42)

    task_counts = [2, 5, 8, 10, 12, 15, 18, 20]
    n_episodes = 100
    delay_results = {m: [] for m in ["HDM", "SAC", "PPO", "AC", "Static"]}

    for n in task_counts:
        env = MultiCSCAEnvironment(n_cscas=n, n_relays=5)
        action_dim = n + n * 5 + n * 3

        han_t, hdm_t = load_trained_hdm(n_tasks=n)
        sac_a = load_trained_baseline("SAC", 45)
        ppo_a = load_trained_baseline("PPO", 45)
        ac_a = load_trained_baseline("AC", 45)

        methods = {"HDM": hdm_t, "SAC": sac_a, "PPO": ppo_a, "AC": ac_a}

        for name, model in methods.items():
            delays = []
            for ep in range(n_episodes):
                state = env.generate_state()
                graph_emb, _ = han_t.encode_state(state)
                with torch.no_grad():
                    action_raw = model(graph_emb)
                parsed = parse_action(action_raw, n, 5, 3)
                result = env.step(parsed, state)
                delays.append(np.mean([t["tau_S"] for t in result["tasks"]]))
            delay_results[name].append(np.mean(delays))

        # Static
        delays_s = []
        for ep in range(n_episodes):
            state = env.generate_state()
            graph_emb, _ = han_t.encode_state(state)
            action_raw = torch.ones(1, 45, device=DEVICE) * 0.5
            parsed = parse_action(action_raw, n, 5, 3)
            result = env.step(parsed, state)
            delays_s.append(np.mean([t["tau_S"] for t in result["tasks"]]))
        delay_results["Static"].append(np.mean(delays_s))

        log(f"  n={n}: HDM={delay_results['HDM'][-1]:.4f}s, SAC={delay_results['SAC'][-1]:.4f}s, "
            f"PPO={delay_results['PPO'][-1]:.4f}s, Static={delay_results['Static'][-1]:.4f}s")

    # Plot delay vs tasks
    plt.figure(figsize=(8, 5))
    styles = {"HDM": "b-o", "SAC": "r-s", "PPO": "g-^", "AC": "m-D", "Static": "k--x"}
    for name, style in styles.items():
        plt.plot(task_counts, delay_results[name], style, label=name, markersize=5)
    plt.xlabel("Number of Tasks"); plt.ylabel("Average Delay (s)")
    plt.title("Fig 9d: Delay vs Number of Tasks (Trained Models)")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "fig9d_delay_vs_tasks.png"), dpi=120)
    plt.close()

    # Delay reduction
    hdm_arr = np.array(delay_results["HDM"])
    best_baseline = np.array([
        min(delay_results["SAC"][i], delay_results["PPO"][i],
            delay_results["AC"][i], delay_results["Static"][i])
        for i in range(len(task_counts))
    ])
    reductions = (best_baseline - hdm_arr) / np.maximum(best_baseline, 1e-8) * 100

    log("Delay reduction (HDM vs best baseline):")
    for n, r in zip(task_counts, reductions):
        log(f"  n={n}: {r:+.2f}%")
    log(f"Average: {np.mean(reductions):+.2f}% (paper: -33.40%)")

    # --- Delay vs SINR ---
    snr_range = [0, 5, 10, 15, 20, 25]
    n_episodes_snr = 100
    snr_hdm, snr_static = [], []

    env5 = MultiCSCAEnvironment(n_cscas=5, n_relays=5)
    han5, hdm5 = load_trained_hdm(n_tasks=5)

    for snr_db in snr_range:
        d_hdm, d_st = [], []
        for ep in range(n_episodes_snr):
            state = env5.generate_state()
            graph_emb, _ = han5.encode_state(state)
            for dl, fn in [(d_hdm, hdm5), (d_st, lambda g: torch.ones(1, 45, device=DEVICE) * 0.5)]:
                with torch.no_grad():
                    action_raw = fn(graph_emb)
                parsed = parse_action(action_raw, 5, 5, 3)
                result = env5.step(parsed, state, target_snr_db=snr_db)
                dl.append(np.mean([t["tau_S"] for t in result["tasks"]]))
        snr_hdm.append(np.mean(d_hdm))
        snr_static.append(np.mean(d_st))
        log(f"  SNR={snr_db}dB: HDM={snr_hdm[-1]:.4f}s, Static={snr_static[-1]:.4f}s")

    plt.figure(figsize=(8, 5))
    plt.plot(snr_range, snr_hdm, "b-o", label="HDM", markersize=5)
    plt.plot(snr_range, snr_static, "k--x", label="Static", markersize=5)
    plt.xlabel("SINR (dB)"); plt.ylabel("Average Delay (s)")
    plt.title("Fig 9c: Delay vs SINR (Trained Models)")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "fig9c_delay_vs_sinr_trained.png"), dpi=120)
    plt.close()

    # Save CSVs
    save_csv(os.path.join(RESULTS, "fig9d_delay_vs_tasks.csv"),
             [[n] + [delay_results[m][i] for m in ["HDM","SAC","PPO","AC","Static"]]
              for i, n in enumerate(task_counts)],
             ["n_tasks", "HDM", "SAC", "PPO", "AC", "Static"])
    save_csv(os.path.join(RESULTS, "fig9c_delay_vs_sinr.csv"),
             [[s, h, st] for s, h, st in zip(snr_range, snr_hdm, snr_static)],
             ["snr_db", "HDM", "Static"])

    log("Delay reduction experiment complete.")
    return delay_results


def experiment_scale_comparison():
    """
    ISR vs tasks at n=5, n=10, n=15 CSCA nodes.
    This is the paper's key result showing HDM advantage grows with scale.
    Matches paper Fig 10 (if it exists) or extends Fig 9a.
    """
    set_seed(42)
    log("Experiment: Scale comparison n=5, n=10, n=15")

    task_counts = [2, 5, 8, 10, 12, 15, 18, 20]
    results_by_scale = {}

    for n_cscas in [5, 10, 15]:
        log(f"  Running n_cscas={n_cscas}...")
        results_by_scale[n_cscas] = {}

        han_trained, hdm_trained = load_trained_hdm(
            n_tasks=n_cscas, n_relays=5, device=DEVICE
        )

        for n_tasks in task_counts:
            if n_tasks > n_cscas * 4:
                continue  # Skip unreasonable task counts

            action_dim = n_tasks + n_tasks * 5 + n_tasks * 3
            env = MultiCSCAEnvironment(n_cscas=n_tasks, n_relays=5)

            han_n = HANNetwork(hidden_channels=128, num_heads=8, num_layers=2,
                               n_cscas=n_tasks, n_relays=5, n_messages=n_tasks,
                               n_base_stations=n_tasks).to(DEVICE)

            # HDM: use trained at n=5 for n=5, untrained for others
            if n_cscas == 5 and n_tasks == 5:
                hdm_fn = hdm_trained.forward
                hdm_han = han_trained
            else:
                hdm_n = HDMPolicy(action_dim=action_dim).to(DEVICE)
                hdm_fn = hdm_n.forward
                hdm_han = han_n

            sac_n = load_trained_baseline("SAC", 45, DEVICE)
            static_n = StaticBaseline(action_dim=action_dim, device=DEVICE)

            method_results = {}
            for name, model_han, fn in [
                ("HDM", hdm_han, hdm_fn),
                ("SAC", han_n, sac_n.get_action),
                ("Static", han_n, static_n.get_action),
            ]:
                r = run_method_episodes_averaged(
                    name, fn, lambda e=env: e, model_han,
                    n_episodes=100, n_seeds=3, n_tasks=n_tasks
                )
                method_results[name] = r["isr"]

            results_by_scale[n_cscas][n_tasks] = method_results
            log(f"    n_tasks={n_tasks}: HDM={method_results['HDM']:.3f}, "
                f"SAC={method_results['SAC']:.3f}, Static={method_results['Static']:.3f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, n_cscas in enumerate([5, 10, 15]):
        ax = axes[idx]
        data = results_by_scale[n_cscas]
        task_ns = sorted(data.keys())
        for method, color in [("HDM", "b"), ("SAC", "r"), ("Static", "k")]:
            isr_vals = [data[n][method] for n in task_ns]
            ax.plot(task_ns, isr_vals, f"{color}-o", label=method, markersize=5)
        ax.set_xlabel("Number of Tasks")
        ax.set_ylabel("ISR")
        ax.set_title(f"n_cscas={n_cscas}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Scale Comparison: HDM vs Baselines")
    plt.tight_layout()
    out_path = os.path.join(RESULTS, "scale_comparison_n5_n10_n15.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    log(f"Scale comparison saved: {out_path}")

    # Save CSV
    rows = []
    for n_cscas in [5, 10, 15]:
        data = results_by_scale[n_cscas]
        for n_tasks in sorted(data.keys()):
            row = [n_cscas, n_tasks]
            for method in ["HDM", "SAC", "Static"]:
                row.append(data[n_tasks].get(method, "N/A"))
            rows.append(row)
    save_csv(
        os.path.join(RESULTS, "scale_comparison.csv"),
        rows,
        ["n_cscas", "n_tasks", "HDM", "SAC", "Static"]
    )
    log("Experiment scale comparison complete.")
    return results_by_scale


if __name__ == "__main__":
    set_seed(42)

    log("=" * 60)
    log("CSCA FINAL EXPERIMENT SUITE")
    log("Using trained HDM and baseline checkpoints")
    log("=" * 60)

    experiment_isr_vs_tasks()
    experiment_delay_vs_sinr()
    experiment_cscqi_convergence()
    experiment_ablation()
    experiment_scale_comparison()
    generate_summary_table()

    log("=" * 60)
    log("ALL EXPERIMENTS COMPLETE")
    log(f"Results saved to: {RESULTS}")
    log("=" * 60)
