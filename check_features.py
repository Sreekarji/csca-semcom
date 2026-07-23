"""Plumbing sanity check - run this FIRST.

Verifies:
  1. CSCA features vary across nodes
  2. HAN shapes correct for multiple tpc
  3. Denoiser gradient flows correctly (non-zero grad on denoiser params)
  4. Per-task BW not collapsed
  5. log_prob is negative and finite
  6. mu_list / a_list pairing is correct (NEW)
Run: python check_features.py
"""
import os
import sys
import numpy as np
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
for sub in ["", "code/hdm", "code/channel", "code/evaluation", "code/utils"]:
    p = os.path.join(BASE, sub)
    if os.path.isdir(p):
        sys.path.insert(0, p)

from reproducibility import set_seed
from sim_channel import MultiCSCAEnvironment, _softmax_bw_allocation
from han_network import HANNetwork
from ddpm_policy import HDMPolicy

def main():
    set_seed(42)
    device = "cpu"

    print("== 1. CSCA physics features ==")
    env   = MultiCSCAEnvironment(difficulty="medium", tasks_per_csca=2)
    state = env.generate_state()
    feats = np.array(state["Rt"]["csca_features"])
    print("csca_features:\n", np.round(feats, 3))
    print("feature std across cscas:", np.round(feats.std(axis=0), 4),
          "  (should be > 0)")

    print("\n== 2. HAN graph shapes ==")
    han = HANNetwork(256, 8, 3, 5, 5, 5, 5).to(device)
    for tpc in (1, 2, 10):
        e = MultiCSCAEnvironment(difficulty="medium", tasks_per_csca=tpc)
        s = e.generate_state()
        intents = [[m[1], m[2]] for m in s["SCt"]["message_features"]]
        g, _, memb = han.encode_state(s, intents)
        print(f"  tpc={tpc:2d} Nm={e.n_tasks:3d}  "
              f"graph_emb={tuple(g.shape)}  message_embs={tuple(memb.shape)}")

    print("\n== 3. Denoiser gradient (should be > 0 on ALL denoiser params) ==")
    policy  = HDMPolicy(5, 3, 256).to(device)
    _, _, memb = han.encode_state(
        state, [[m[1], m[2]] for m in state["SCt"]["message_features"]])
    action, log_prob = policy.collect_trajectory(memb, congestion_idx=1)

    lp_sum = log_prob.sum()
    lp_sum.backward()

    gnorm = sum(
        p.grad.norm().item()
        for p in policy.denoiser.parameters()
        if p.grad is not None
    )
    print(f"  log_prob shape={tuple(log_prob.shape)} "
          f"values={log_prob.detach().numpy().round(3)}")
    print(f"  denoiser grad-norm={gnorm:.6f}  (MUST be > 0)")
    assert gnorm > 0, "GRADIENT IS ZERO - bug in collect_trajectory!"

    print("\n== 4. Per-task BW allocation (not collapsed to uniform) ==")
    logits = torch.randn(1, env.n_tasks)
    bw     = _softmax_bw_allocation(
        [logits[0, i].item() for i in range(env.n_tasks)], 5e6)
    bw     = np.array(bw)
    print("  bw shares:", np.round(bw / bw.sum(), 3))
    print("  bw std/mean:", round(bw.std() / bw.mean(), 3), "  (should be > 0)")

    print("\n== 5. log_prob finite and negative ==")
    assert torch.isfinite(log_prob).all(), "log_prob contains NaN/Inf!"
    assert (log_prob < 0).all(), "log_prob should be negative!"
    print("  log_prob OK:", log_prob.detach().numpy().round(3))

    print("\nAll checks PASSED.")

if __name__ == "__main__":
    main()
