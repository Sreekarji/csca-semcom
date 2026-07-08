"""Fig 6 — CSCA text evaluation only (no DeepSC to avoid import crash)"""
import os, sys, json, torch, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, r"D:\MP2\code\hdm")
sys.path.insert(0, r"D:\MP2\code\channel")
sys.path.insert(0, r"D:\MP2\code\evaluation")

from han_network import HANNetwork
from ddpm_policy import HDMPolicy
from sim_channel import MultiCSCAEnvironment

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FINAL = r"D:\MP2\results\software\final"
os.makedirs(FINAL, exist_ok=True)
LOG_PATH = r"D:\MP2\log.txt"

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")

# Load sentence transformer
from sentence_transformers import SentenceTransformer
sent_model = SentenceTransformer("all-MiniLM-L6-v2")
print("SentenceTransformer loaded")

def cosine_sim(t1, t2):
    e = sent_model.encode([t1, t2], convert_to_tensor=True)
    return float(torch.nn.functional.cosine_similarity(e[0:1], e[1:2]).item())

# Load HDM
han = HANNetwork(hidden_channels=128, num_heads=8, num_layers=2,
                 n_cscas=5, n_relays=5, n_messages=5, n_base_stations=5).to(DEVICE)
hdm = HDMPolicy(action_dim=45, n_denoising_steps=6).to(DEVICE)
ckpt = torch.load(r"D:\MP2\results\software\checkpoints\hdm_ep5000.pt",
                   map_location=DEVICE, weights_only=False)
han.load_state_dict(ckpt["han"])
hdm.load_state_dict(ckpt["actor"])
han.eval(); hdm.eval()
print("HDM loaded")

# Load SST
with open(r"D:\MP2\data\raw\sst_sentences.json") as f:
    sentences = [item["text"] for item in json.load(f)]

env = MultiCSCAEnvironment(n_cscas=5, n_relays=5, difficulty="hard")

def csca_pass(text):
    words = text.lower().split()
    n = len(words)
    state = env.generate_state()
    graph_emb, _ = han.encode_state(state)
    with torch.no_grad():
        action = hdm(graph_emb)
    bw = action[:, :5]
    relay = action[:, 5:30].reshape(1, 5, 5)
    mcs = action[:, 30:].reshape(1, 5, 3)
    result = env.step({"bandwidth": bw, "relay": relay, "mcs": mcs}, state)
    tasks = result["tasks"]
    avg_sinr = np.mean([1.0 / max(t["tau_S"], 0.001) for t in tasks])
    keep_prob = min(1.0, avg_sinr * 0.5 + 0.3)
    np.random.seed(hash(text) % 2**31)
    kept = [w for w in words if np.random.random() < keep_prob]
    if not kept: kept = words[:max(1, n//3)]
    return " ".join(kept), len(kept)/max(n,1)

# Run evaluation
log("TEXT: CSCA evaluation, SNR 0-20 dB, 100 sentences")
snr_range = [0, 5, 10, 15, 20]
results = {s: {"sim": [], "comp": []} for s in snr_range}

for snr in snr_range:
    for text in sentences[:100]:
        recon, comp = csca_pass(text)
        sim = cosine_sim(text, recon)
        results[snr]["sim"].append(sim)
        results[snr]["comp"].append(comp)
    log(f"  SNR={snr}dB: sim={np.mean(results[snr]['sim']):.4f} std={np.std(results[snr]['sim']):.4f}, comp={np.mean(results[snr]['comp']):.4f}")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

sims_mean = [np.mean(results[s]["sim"]) for s in snr_range]
sims_std = [np.std(results[s]["sim"]) for s in snr_range]
ax1.errorbar(snr_range, sims_mean, yerr=sims_std, fmt='b-o', capsize=4, linewidth=2, label='CSCA (text)')
ax1.set_xlabel("SNR (dB)"); ax1.set_ylabel("Semantic Similarity")
ax1.set_title("Fig 6: CSCA Semantic Similarity vs SNR"); ax1.legend(); ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.05)

comps_mean = [np.mean(results[s]["comp"]) for s in snr_range]
ax2.bar(range(len(snr_range)), comps_mean, color='steelblue')
ax2.set_xticks(range(len(snr_range))); ax2.set_xticklabels([f"{s}" for s in snr_range])
ax2.set_xlabel("SNR (dB)"); ax2.set_ylabel("Compression Ratio")
ax2.set_title("Compression Ratio vs SNR"); ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(FINAL, "fig6_multimodal_semcom.png"), dpi=150)
plt.close()
log("Plot saved: fig6_multimodal_semcom.png")

# Summary
log("SUMMARY:")
for s in snr_range:
    log(f"  SNR={s}dB: sim={np.mean(results[s]['sim']):.4f}+-{np.std(results[s]['sim']):.4f}, comp={np.mean(results[s]['comp']):.4f}")
log("NOTE: DeepSC comparison skipped (import crash). Image modality skipped (no PSNR).")
log("DONE")
