"""
Oracle Gap Test: proves uniform BW allocation is suboptimal at high task counts.
If oracle >> uniform, there IS a learning problem worth solving.
If oracle ≈ uniform, the environment is fundamentally degenerate.
"""
import sys, os
import numpy as np
import torch
import csv

sys.path.insert(0, r"D:\MP2\code\channel")
sys.path.insert(0, r"D:\MP2\code\evaluation")
from sim_channel import MultiCSCAEnvironment
from cscqi import compute_isr, compute_cscqi

def oracle_allocation(state, env, n_trials=20):
    """
    Oracle: try n_trials random BW allocations, pick the one that maximizes ISR.
    This upper-bounds what any learned policy can achieve.
    """
    best_isr = 0.0
    best_action = None
    for _ in range(n_trials):
        # Random non-uniform allocation (Dirichlet gives valid distribution)
        bw_raw = np.random.dirichlet(np.ones(env.n_cscas))
        bw_t = torch.tensor(bw_raw, dtype=torch.float).unsqueeze(0)
        relay_t = torch.rand(1, env.n_cscas * env.n_relays).reshape(1, env.n_cscas, env.n_relays)
        mcs_t = torch.rand(1, env.n_cscas * 3).reshape(1, env.n_cscas, 3)
        action = {"bandwidth": bw_t, "relay": relay_t, "mcs": mcs_t}
        result = env.step(action, state)
        isr = compute_isr(result["tasks"])
        if isr > best_isr:
            best_isr = isr
            best_action = action
    return best_isr

def uniform_allocation(state, env):
    """Static uniform: equal BW to all CSCAs."""
    bw_t = torch.ones(1, env.n_cscas) * (1.0 / env.n_cscas)
    relay_t = torch.ones(1, env.n_cscas, env.n_relays) * 0.5
    mcs_t = torch.ones(1, env.n_cscas, 3) * 0.5
    action = {"bandwidth": bw_t, "relay": relay_t, "mcs": mcs_t}
    result = env.step(action, state)
    return compute_isr(result["tasks"])

print("="*60)
print("ORACLE GAP TEST")
print("="*60)

results = []
n_episodes = 100
tasks_per_csca_list = [1, 2, 4, 6, 8, 10, 12, 15, 20]

for tpc in tasks_per_csca_list:
    env = MultiCSCAEnvironment(
        n_cscas=5, n_relays=5, bandwidth_total_hz=5e6,
        difficulty="hard", tasks_per_csca=tpc
    )
    oracle_isrs = []
    uniform_isrs = []
    for ep in range(n_episodes):
        np.random.seed(ep)
        state = env.generate_state()
        oracle_isrs.append(oracle_allocation(state, env, n_trials=50))
        np.random.seed(ep)
        state = env.generate_state()
        uniform_isrs.append(uniform_allocation(state, env))
    
    mean_oracle = np.mean(oracle_isrs)
    mean_uniform = np.mean(uniform_isrs)
    gap = mean_oracle - mean_uniform
    gap_pct = gap / max(mean_uniform, 1e-8) * 100
    
    print(f"tasks_per_csca={tpc:2d} | Oracle={mean_oracle:.4f} | Uniform={mean_uniform:.4f} | Gap={gap:+.4f} ({gap_pct:+.1f}%)")
    results.append([tpc, tpc*5, mean_oracle, mean_uniform, gap, gap_pct])

print("\nKey diagnostic:")
print("- If Gap > 0.05 at tpc>=10: learning problem exists, fix the training")
print("- If Gap ≈ 0 at all tpc: environment is fundamentally degenerate")

os.makedirs(r"D:\MP2\results\software", exist_ok=True)
with open(r"D:\MP2\results\software\oracle_gap_test.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["tasks_per_csca", "total_tasks", "oracle_isr", "uniform_isr", "gap", "gap_pct"])
    w.writerows(results)
print("Saved: D:\\MP2\\results\\software\\oracle_gap_test.csv")
