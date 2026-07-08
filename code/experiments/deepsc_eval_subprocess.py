"""
DeepSC evaluation using subprocess isolation.
Runs DeepSC in a separate Python process to avoid import conflicts.
"""
import os
import sys
import json
import subprocess
import tempfile

PYTHON_EXE = r"D:\MP2\.venv\Scripts\python.exe"
DEEPSC_PATH = r"D:\MP2\repos\DeepSC"
RESULTS_PATH = r"D:\MP2\results\software\final"
os.makedirs(RESULTS_PATH, exist_ok=True)


def run_in_subprocess(script_code, timeout=120):
    """Run Python code in an isolated subprocess."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir=r'D:\MP2') as f:
        f.write(script_code)
        script_path = f.name
    try:
        result = subprocess.run(
            [PYTHON_EXE, script_path],
            capture_output=True, text=True, timeout=timeout,
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "timeout", "returncode": -1}
    finally:
        try:
            os.unlink(script_path)
        except:
            pass


# Script 1: DeepSC import check
IMPORT_CHECK = f'''
import sys, json
sys.path.insert(0, r"{DEEPSC_PATH}")
try:
    from models.transceiver import DeepSC
    with open(r"{DEEPSC_PATH}/europarl/vocab.json") as f:
        vocab = json.load(f)
    num_vocab = len(vocab["token_to_idx"])
    import torch
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepSC(4, num_vocab, num_vocab, num_vocab, num_vocab, 128, 8, 512, 0.1).to(DEVICE)
    state = torch.load(r"D:\\MP2\\models\\deepsc\\text\\best_model.pth", map_location=DEVICE, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    print(json.dumps({{"status": "ok", "vocab": num_vocab, "device": str(DEVICE)}}))
except Exception as e:
    print(json.dumps({{"status": "error", "message": str(e), "type": type(e).__name__}}))
'''


# Script 2: DeepSC forward pass test
FORWARD_TEST = f'''
import sys, json, torch
sys.path.insert(0, r"{DEEPSC_PATH}")
from models.transceiver import DeepSC

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open(r"{DEEPSC_PATH}/europarl/vocab.json") as f:
    vocab = json.load(f)
tok2idx = vocab["token_to_idx"]
idx2tok = {{v: k for k, v in tok2idx.items()}}
num_vocab = len(tok2idx)

model = DeepSC(4, num_vocab, num_vocab, num_vocab, num_vocab, 128, 8, 512, 0.1).to(DEVICE)
model.load_state_dict(torch.load(r"D:\\MP2\\models\\deepsc\\text\\best_model.pth", map_location=DEVICE, weights_only=False))
model.eval()

# Test forward pass
text = "send the image quickly"
words = text.lower().split()
tokens = ["<START>"] + words + ["<END>"]
ids = [tok2idx.get(t, tok2idx.get("<UNK>", 3)) for t in tokens]
src = torch.tensor([ids], dtype=torch.long, device=DEVICE)
pad_idx = tok2idx.get("<PAD>", 0)

trg_inp = src[:, :-1]

    enc = model.encoder(src, None)
ce = model.channel_encoder(enc)
p = torch.mean(ce**2)
tx = ce / torch.sqrt(p + 1e-8)
n_var = 0.1
rx = tx + torch.randn_like(tx) * n_var
cd = model.channel_decoder(rx)
dec = model.decoder(trg_inp, cd, None, None)
logits = model.dense(dec)
pred = logits.argmax(dim=-1)[0].cpu().tolist()
decoded = [idx2tok.get(i, "<UNK>") for i in pred if idx2tok.get(i) not in ("<START>", "<PAD>")]
if "<END>" in decoded:
    decoded = decoded[:decoded.index("<END>")]
recon = " ".join(decoded) if decoded else text

# Similarity
from collections import Counter
def cosine_words(t1, t2):
    c1, c2 = Counter(t1.split()), Counter(t2.split())
    dot = sum(c1[k]*c2[k] for k in set(c1)|set(c2))
    n1 = sum(v**2 for v in c1.values())**0.5
    n2 = sum(v**2 for v in c2.values())**0.5
    return dot/(n1*n2) if n1*n2 > 0 else 0

sim = cosine_words(text, recon)
print(json.dumps({{
    "status": "ok",
    "input": text,
    "output": recon,
    "similarity": round(sim, 4),
    "compression": round(len(decoded)/len(words), 4)
}}))
'''


# Script 3: Full SNR sweep evaluation
SNR_SWEEP = f'''
import sys, json, torch, numpy as np
sys.path.insert(0, r"{DEEPSC_PATH}")
from models.transceiver import DeepSC

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open(r"{DEEPSC_PATH}/europarl/vocab.json") as f:
    vocab = json.load(f)
tok2idx = vocab["token_to_idx"]
idx2tok = {{v: k for k, v in tok2idx.items()}}
num_vocab = len(tok2idx)

model = DeepSC(4, num_vocab, num_vocab, num_vocab, num_vocab, 128, 8, 512, 0.1).to(DEVICE)
model.load_state_dict(torch.load(r"D:\\MP2\\models\\deepsc\\text\\best_model.pth", map_location=DEVICE, weights_only=False))
model.eval()

with open(r"D:\\MP2\\data\\raw\\sst_sentences.json") as f:
    sentences = [item["text"] for item in json.load(f)][:50]

def deepsc_forward(text, n_var):
    words = text.lower().split()
    tokens = ["<START>"] + words + ["<END>"]
    ids = [tok2idx.get(t, tok2idx.get("<UNK>", 3)) for t in tokens]
    src = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    pad_idx = tok2idx.get("<PAD>", 0)
    trg_inp = src[:, :-1]
    enc = model.encoder(src, None)
    ce = model.channel_encoder(enc)
    p = torch.mean(ce**2)
    tx = ce / torch.sqrt(p + 1e-8)
    rx = tx + torch.randn_like(tx) * n_var
    cd = model.channel_decoder(rx)
    dec = model.decoder(trg_inp, cd, None, None)
    logits = model.dense(dec)
    pred = logits.argmax(dim=-1)[0].cpu().tolist()
    decoded = [idx2tok.get(i, "<UNK>") for i in pred if idx2tok.get(i) not in ("<START>", "<PAD>")]
    if "<END>" in decoded: decoded = decoded[:decoded.index("<END>")]
    return " ".join(decoded) if decoded else text, len(decoded)/max(len(words),1)

from collections import Counter
def cosine_words(t1, t2):
    c1, c2 = Counter(t1.split()), Counter(t2.split())
    dot = sum(c1[k]*c2[k] for k in set(c1)|set(c2))
    n1 = sum(v**2 for v in c1.values())**0.5
    n2 = sum(v**2 for v in c2.values())**0.5
    return dot/(n1*n2) if n1*n2 > 0 else 0

snr_range = [0, 5, 10, 15, 20]
results = {{}}
for snr in snr_range:
    n_var = 10 ** (-snr / 10)
    sims, comps = [], []
    for text in sentences:
        recon, comp = deepsc_forward(text, n_var)
        sims.append(cosine_words(text, recon))
        comps.append(comp)
    results[snr] = {{"sim": round(float(np.mean(sims)), 4), "comp": round(float(np.mean(comps)), 4)}}

print(json.dumps(results))
'''


if __name__ == "__main__":
    print("=" * 60)
    print("DEEPSC SUBPROCESS EVALUATION")
    print("=" * 60)

    # Task 1: Import check
    print("\n[1/3] Import check...")
    r = run_in_subprocess(IMPORT_CHECK)
    print(f"  Return code: {r['returncode']}")
    if r['stdout'].strip():
        print(f"  Output: {r['stdout'].strip()}")
    if r['stderr'].strip():
        print(f"  Stderr: {r['stderr'][:200]}")

    # Task 2: Forward pass test
    print("\n[2/3] Forward pass test...")
    r = run_in_subprocess(FORWARD_TEST)
    print(f"  Return code: {r['returncode']}")
    if r['stdout'].strip():
        print(f"  Output: {r['stdout'].strip()}")

    # Task 3: SNR sweep
    print("\n[3/3] SNR sweep (50 sentences x 5 SNR values)...")
    r = run_in_subprocess(SNR_SWEEP, timeout=300)
    print(f"  Return code: {r['returncode']}")
    if r['stdout'].strip():
        try:
            data = json.loads(r['stdout'].strip())
            print("  Results:")
            for snr, vals in sorted(data.items(), key=lambda x: int(x[0])):
                print(f"    SNR={snr}dB: sim={vals['sim']}, comp={vals['comp']}")
            # Save results
            with open(os.path.join(RESULTS_PATH, "deepsc_snr_sweep.json"), "w") as f:
                json.dump(data, f, indent=2)
            print(f"  Saved to {RESULTS_PATH}\\deepsc_snr_sweep.json")
        except json.JSONDecodeError:
            print(f"  Raw output: {r['stdout'][:300]}")
    if r['stderr'].strip():
        print(f"  Stderr: {r['stderr'][:200]}")

    print("\n" + "=" * 60)
    print("DONE")
