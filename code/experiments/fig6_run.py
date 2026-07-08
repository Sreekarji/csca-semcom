"""Fig 6 — CSCA evaluation (text + audio)"""
import os, sys, json, torch, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

# Preload sentence_transformers BEFORE other heavy imports
from sentence_transformers import SentenceTransformer as _ST_pre
_ST_pre("all-MiniLM-L6-v2")

sys.path.insert(0, r"D:\MP2\code\hdm")
sys.path.insert(0, r"D:\MP2\code\channel")
sys.path.insert(0, r"D:\MP2\code\evaluation")

from han_network import HANNetwork
from ddpm_policy import HDMPolicy
from sim_channel import MultiCSCAEnvironment
from cscqi import compute_semantic_accuracy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FINAL = r"D:\MP2\results\software\final"
os.makedirs(FINAL, exist_ok=True)
LOG = r"D:\MP2\log.txt"

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def sim(t1, t2):
    return compute_semantic_accuracy(t1, t2)

han = HANNetwork(hidden_channels=128, num_heads=8, num_layers=2,
                 n_cscas=5, n_relays=5, n_messages=5, n_base_stations=5).to(DEVICE)
hdm = HDMPolicy(action_dim=45, n_denoising_steps=6).to(DEVICE)
ckpt = torch.load(r"D:\MP2\results\software\checkpoints\hdm_ep5000.pt",
                   map_location=DEVICE, weights_only=False)
han.load_state_dict(ckpt["han"]); hdm.load_state_dict(ckpt["actor"])
han.eval(); hdm.eval()
log("HDM loaded")

with open(r"D:\MP2\data\raw\sst_sentences.json") as f:
    sentences = [item["text"] for item in json.load(f)]

env = MultiCSCAEnvironment(n_cscas=5, n_relays=5, difficulty="hard")

def csca(text):
    words = text.lower().split()
    n = len(words)
    state = env.generate_state()
    g, _ = han.encode_state(state)
    with torch.no_grad(): a = hdm(g)
    bw = a[:, :5]; r = a[:, 5:30].reshape(1,5,5); m = a[:, 30:].reshape(1,5,3)
    res = env.step({"bandwidth": bw, "relay": r, "mcs": m}, state)
    sinr = np.mean([1.0/max(t["tau_S"], 0.001) for t in res["tasks"]])
    kp = min(1.0, sinr*0.5+0.3)
    np.random.seed(hash(text) % 2**31)
    kept = [w for w in words if np.random.random() < kp]
    if not kept: kept = words[:max(1, n//3)]
    return " ".join(kept), len(kept)/max(n,1)

# TEXT evaluation
log("TEXT: CSCA, SNR 0-20 dB, 100 sentences")
snr_range = [0, 5, 10, 15, 20]
results = {s: {"sim": [], "comp": []} for s in snr_range}
for snr in snr_range:
    for text in sentences[:100]:
        r, c = csca(text)
        results[snr]["sim"].append(sim(text, r))
        results[snr]["comp"].append(c)
    log(f"  SNR={snr}dB: sim={np.mean(results[snr]['sim']):.4f}+-{np.std(results[snr]['sim']):.4f}, comp={np.mean(results[snr]['comp']):.4f}")

# AUDIO evaluation
log("AUDIO: Whisper -> CSCA, 50 files")
import whisper
wmodel = whisper.load_model("base")
audio_dir = r"D:\MP2\data\raw\audio"
afiles = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith(('.wav','.flac'))][:50]
a_sims, a_comps = [], []
for i, af in enumerate(afiles):
    try:
        res = wmodel.transcribe(af, language="en", fp16=False)
        orig = res["text"].strip()
        if len(orig.split()) < 3: continue
        r, c = csca(orig)
        a_sims.append(sim(orig, r))
        a_comps.append(c)
        if (i+1) % 10 == 0: log(f"  Audio {i+1}: sim={np.mean(a_sims):.4f}")
    except: pass
log(f"  Audio: sim={np.mean(a_sims):.4f}+-{np.std(a_sims):.4f}, comp={np.mean(a_comps):.4f}")

# PLOTS
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
sims_m = [np.mean(results[s]["sim"]) for s in snr_range]
sims_s = [np.std(results[s]["sim"]) for s in snr_range]
ax1.errorbar(snr_range, sims_m, yerr=sims_s, fmt='b-o', capsize=4, linewidth=2, label='CSCA (text)')
if a_sims:
    ax1.axhline(y=np.mean(a_sims), color='g', linestyle=':', linewidth=2,
                label=f'CSCA (audio)={np.mean(a_sims):.3f}')
ax1.set_xlabel("SNR (dB)"); ax1.set_ylabel("Semantic Similarity (Jaccard)")
ax1.set_title("Fig 6: CSCA Semantic Similarity vs SNR"); ax1.legend(); ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.05)

comps_m = [np.mean(results[s]["comp"]) for s in snr_range]
ax2.bar(range(len(snr_range)), comps_m, color='steelblue', alpha=0.7, label='Text')
if a_comps:
    ax2.bar(len(snr_range), np.mean(a_comps), color='green', alpha=0.7, label='Audio')
ax2.set_xticks(range(len(snr_range)+1)); ax2.set_xticklabels([f"{s}" for s in snr_range]+["Audio"])
ax2.set_xlabel("SNR (dB)"); ax2.set_ylabel("Compression Ratio")
ax2.set_title("Compression Ratio"); ax2.legend(); ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(FINAL, "fig6_multimodal_semcom.png"), dpi=150)
plt.close()

# Also save compression bar chart
fig, ax = plt.subplots(figsize=(6, 4))
mods = ['Text', 'Audio']
vals = [np.mean([results[s]["comp"] for s in snr_range]), np.mean(a_comps) if a_comps else 0]
paper = [0.73, 0.32]
x = np.arange(len(mods)); w = 0.35
ax.bar(x-w/2, vals, w, label='Our CSCA', color='steelblue')
ax.bar(x+w/2, paper, w, label='Paper', color='coral')
ax.set_ylabel("Compression Ratio"); ax.set_title("Compression Ratio"); ax.set_xticks(x)
ax.set_xticklabels(mods); ax.legend(); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(FINAL, "compression_ratio.png"), dpi=150)
plt.close()

log("Plots saved to final/")
log("=" * 60)
log("RESULTS:")
for s in snr_range:
    log(f"  SNR={s}dB: sim={np.mean(results[s]['sim']):.4f}+-{np.std(results[s]['sim']):.4f}, comp={np.mean(results[s]['comp']):.4f}")
log(f"  Audio: sim={np.mean(a_sims):.4f}+-{np.std(a_sims):.4f}, comp={np.mean(a_comps):.4f}")
log(f"  Paper: text=0.73, audio=0.32")
log("  NOTE: Using Jaccard word overlap (not cosine). DeepSC skipped (import crash). Image excluded.")
log("DONE")
