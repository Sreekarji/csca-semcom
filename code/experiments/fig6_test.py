import torch, sys, json, numpy as np
print("step1", flush=True)
sys.path.insert(0, r"D:\MP2\code\hdm")
sys.path.insert(0, r"D:\MP2\code\channel")
from han_network import HANNetwork
from ddpm_policy import HDMPolicy
from sim_channel import MultiCSCAEnvironment
print("step2", flush=True)

DEVICE = torch.device("cuda")
han = HANNetwork(hidden_channels=128, num_heads=8, num_layers=2,
                 n_cscas=5, n_relays=5, n_messages=5, n_base_stations=5).to(DEVICE)
hdm = HDMPolicy(action_dim=45, n_denoising_steps=6).to(DEVICE)
ckpt = torch.load(r"D:\MP2\results\software\checkpoints\hdm_ep5000.pt",
                   map_location=DEVICE, weights_only=False)
han.load_state_dict(ckpt["han"])
hdm.load_state_dict(ckpt["actor"])
han.eval(); hdm.eval()
print("step3", flush=True)

with open(r"D:\MP2\data\raw\sst_sentences.json") as f:
    sentences = [item["text"] for item in json.load(f)]
print(f"loaded {len(sentences)} sentences", flush=True)

env = MultiCSCAEnvironment(n_cscas=5, n_relays=5, difficulty="hard")

for text in sentences[:3]:
    words = text.lower().split()
    state = env.generate_state()
    g, _ = han.encode_state(state)
    with torch.no_grad():
        a = hdm(g)
    bw = a[:, :5]
    relay = a[:, 5:30].reshape(1, 5, 5)
    mcs = a[:, 30:].reshape(1, 5, 3)
    result = env.step({"bandwidth": bw, "relay": relay, "mcs": mcs}, state)
    tasks = result["tasks"]
    sinr = np.mean([1.0 / max(t["tau_S"], 0.001) for t in tasks])
    kp = min(1.0, sinr * 0.5 + 0.3)
    np.random.seed(hash(text) % 2**31)
    kept = [w for w in words if np.random.random() < kp]
    if not kept:
        kept = words[:max(1, len(words) // 3)]
    print(f"sim=comp={len(kept)/len(words):.2f}, keep_prob={kp:.2f}", flush=True)

print("DONE", flush=True)
