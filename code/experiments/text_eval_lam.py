"""
Text evaluation using LAM-generated descriptions.
Matches the paper's methodology: LAM descriptions evaluated through MSS + channel.

Uses triple subprocess isolation: LAM / MSS(BERT) / Similarity(ST).
"""

import os, sys, json, subprocess, numpy as np

LOG_PATH = r"D:\MP2\log.txt"
RESULTS = r"D:\MP2\results\software\final"
PYTHON = sys.executable


def log(msg):
    from datetime import datetime
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def run_subprocess(script, label="subprocess", timeout=300):
    """Run Python script in isolated subprocess."""
    log(f"  Running {label}...")
    result = subprocess.run(
        [PYTHON, "-c", script],
        capture_output=True, text=True, timeout=timeout,
        cwd=r"D:\MP2",
        env={**os.environ, "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
    )
    if result.stdout.strip():
        for line in result.stdout.strip().split("\n")[-5:]:
            print(f"    {line}")
    if result.returncode != 0:
        log(f"  {label} FAILED (exit {result.returncode})")
        if result.stderr:
            print(f"    ERR: {result.stderr[-300:]}")
        return False
    return True


if __name__ == "__main__":
    log("=" * 60)
    log("TEXT EVALUATION (LAM descriptions, triple subprocess isolation)")

    # ---- Step 1: Generate LAM descriptions (subprocess) ----
    desc_path = os.path.join(RESULTS, "_lam_desc.json")
    step1 = f'''
import sys, os, json
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
sys.path.insert(0, r"D:\\MP2\\code\\lam")
sys.path.insert(0, r"D:\\MP2\\code\\utils")
from reproducibility import set_seed; set_seed(42)
from intent_parser import IntentParser

intents = [
    "Send this image to user B within 1 second with high resolution",
    "Transmit the audio file to the base station with minimal latency",
    "Forward the video stream to all connected devices accurately",
    "Send the sensor data to the cloud server within 500 milliseconds",
    "Deliver the message to receiver C with maximum quality preservation",
    "Transmit the document to the relay node as quickly as possible",
    "Send the multimodal data packet with 90 percent quality requirement",
    "Forward the image to the destination with low delay and high fidelity",
    "Transmit the voice message reliably within two seconds",
    "Send the data stream to base station B with high semantic accuracy",
] * 2

parser = IntentParser()
descs = []
for intent in intents:
    r = parser.parse(intent)
    desc = f"Communication intent: {{intent}}. Required delay: {{r['delay_intent']:.2f}}s. Quality: {{r['quality_intent']:.2f}}"
    descs.append({{"original": intent, "description": desc}})
    print(f"  parsed: delay={{r['delay_intent']:.2f}} quality={{r['quality_intent']:.2f}}")

json.dump(descs, open(r"{desc_path}", "w"), indent=2)
print(f"Saved {{len(descs)}} descriptions")
'''
    log("Step 1: LAM intent parsing...")
    run_subprocess(step1, "LAM parser")

    # ---- Step 2: MSS compression (subprocess, BERT only) ----
    mss_path = os.path.join(RESULTS, "_mss_results.json")
    step2 = f'''
import sys, os, json
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
sys.path.insert(0, r"D:\\MP2\\code\\lam")
from source_simplifier import SourceSimplifier

descs = json.load(open(r"{desc_path}"))
simplifier = SourceSimplifier()
results = []
for item in descs:
    desc = item["description"]
    ms = simplifier.find_mss(desc, eta=0.70)
    simp = ms["simplified"]
    comp = len(simp.split()) / max(len(desc.split()), 1)
    results.append({{"description": desc, "simplified": simp, "compression": comp}})
    print(f"  comp={{comp:.3f}} words: {{len(desc.split())}} -> {{len(simp.split())}}")

mean_comp = sum(r["compression"] for r in results) / len(results)
print(f"Mean compression: {{mean_comp:.3f}}")
json.dump(results, open(r"{mss_path}", "w"), indent=2)
'''
    log("Step 2: MSS compression (BERT subprocess)...")
    run_subprocess(step2, "MSS/BERT")

    # ---- Step 3: Similarity (subprocess, SentenceTransformer only) ----
    sim_path = os.path.join(RESULTS, "_sim_results.json")
    step3 = f'''
import sys, os, json, numpy as np
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
sys.path.insert(0, r"D:\\MP2\\code\\evaluation")
from cscqi import compute_semantic_accuracy

mss_results = json.load(open(r"{mss_path}"))
sims = []
for item in mss_results:
    sim = compute_semantic_accuracy(item["description"], item["simplified"])
    sims.append(sim)
    print(f"  sim={{sim:.4f}}")

mean_sim = float(np.mean(sims))
mean_comp = float(np.mean([r["compression"] for r in mss_results]))
print(f"MEAN: similarity={{mean_sim:.4f}}, compression={{mean_comp:.3f}}")
json.dump({{"mean_similarity": mean_sim, "mean_compression": mean_comp,
            "similarities": sims}}, open(r"{sim_path}", "w"))
'''
    log("Step 3: Semantic similarity (SentenceTransformer subprocess)...")
    run_subprocess(step3, "Similarity/ST")

    # ---- Read results ----
    if os.path.exists(sim_path):
        data = json.load(open(sim_path))
        log(f"\nText evaluation (LAM descriptions):")
        log(f"  Mean similarity: {data['mean_similarity']:.4f}")
        log(f"  Mean compression: {data['mean_compression']:.3f}")
        log(f"  (Paper target: similarity ~0.85, compression ~0.73)")
    else:
        log("Text evaluation FAILED — no results produced")

    # Cleanup
    for f in [desc_path, mss_path, sim_path]:
        if os.path.exists(f):
            os.remove(f)

    log("=" * 60)
