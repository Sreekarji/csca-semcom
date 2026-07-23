"""One-shot orchestration: evaluate, write CSV, optional plot.
Run: python run_all_experiments.py
"""
import os
import sys
import csv
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
for sub in ["", "code/experiments", "code/utils"]:
    p = os.path.join(BASE, sub)
    if os.path.isdir(p):
        sys.path.insert(0, p)

from joshi_eval_v2 import evaluate_all, TPC_LIST

RESULTS = os.path.join(BASE, "results")

def main():
    os.makedirs(RESULTS, exist_ok=True)
    results = evaluate_all()

    csv_path = os.path.join(RESULTS, "isr_by_tpc.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"]
                   + [f"tpc{t}_mean" for t in TPC_LIST]
                   + [f"tpc{t}_std"  for t in TPC_LIST])
        for name, per_tpc in results.items():
            means = [np.mean(per_tpc[t]) for t in TPC_LIST]
            stds  = [np.std(per_tpc[t])  for t in TPC_LIST]
            w.writerow([name]
                       + [f"{m:.4f}" for m in means]
                       + [f"{s:.4f}" for s in stds])
    print(f"Wrote {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 5))
        for name, per_tpc in results.items():
            means = [np.mean(per_tpc[t]) for t in TPC_LIST]
            stds  = [np.std(per_tpc[t])  for t in TPC_LIST]
            plt.errorbar(TPC_LIST, means, yerr=stds,
                         marker="o", capsize=3, label=name)
        plt.xlabel("tasks per CSCA (congestion)")
        plt.ylabel("Intent Satisfaction Rate")
        plt.title("ISR vs congestion")
        plt.legend()
        plt.grid(alpha=0.3)
        png = os.path.join(RESULTS, "isr_by_tpc.png")
        plt.savefig(png, dpi=140, bbox_inches="tight")
        print(f"Wrote {png}")
    except Exception as e:
        print(f"(plot skipped: {e})")

if __name__ == "__main__":
    main()
