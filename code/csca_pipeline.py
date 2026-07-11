import os
import sys
import json
import torch
import numpy as np
from datetime import datetime

sys.path.insert(0, r"D:\MP2\code\lam")
sys.path.insert(0, r"D:\MP2\code\hdm")
sys.path.insert(0, r"D:\MP2\code\channel")
sys.path.insert(0, r"D:\MP2\code\evaluation")

from intent_parser import IntentParser
from han_network import HANNetwork
from ddpm_policy import HDMPolicy, CriticNetwork
from cscqi import compute_cscqi, compute_isr, compute_semantic_accuracy
from sim_channel import MultiCSCAEnvironment
from source_simplifier import SourceSimplifier, compute_mim, select_mcs

RESULTS_PATH = r"D:\MP2\results\software"
LOG_PATH = r"D:\MP2\log.txt"
os.makedirs(RESULTS_PATH, exist_ok=True)


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


class CSCAPipeline:
    """
    Full CSCA pipeline implementing Sun et al. 2026.
    Layer 1: LAM (Qwen2-VL-7B) — intent parsing + modality alignment
    Layer 2: HDM (HAN + DDPM) — policy generation
    Layer 3: Channel simulation — metric computation
    """

    def __init__(
        self,
        n_cscas: int = 5,
        n_relays: int = 5,
        n_mcs: int = 3,
        n_denoising_steps: int = 6,
        load_lam: bool = True,
        device: str = None,
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.n_cscas = n_cscas
        self.n_relays = n_relays
        self.n_mcs = n_mcs
        action_dim = n_cscas + n_cscas * n_relays + n_cscas * n_mcs

        log(f"Initializing CSCA pipeline on {self.device}")

        # Layer 1: LAM
        if load_lam:
            log("Loading Qwen2-VL-7B intent parser...")
            self.intent_parser = IntentParser()
        else:
            self.intent_parser = None
            log("LAM skipped (load_lam=False)")

        # Layer 2: HDM
        log("Initializing HAN network...")
        self.han = HANNetwork(
            hidden_channels=256,
            num_heads=8,
            num_layers=2,
            n_cscas=n_cscas,
            n_relays=n_relays,
            n_messages=n_cscas,
            n_base_stations=n_cscas,
        ).to(self.device)

        log("Initializing DDPM policy...")
        self.policy = HDMPolicy(
            action_dim=action_dim,
            graph_emb_dim=256,
            n_denoising_steps=n_denoising_steps,
        ).to(self.device)

        self.critic = CriticNetwork(
            state_dim=256,
            action_dim=action_dim,
        ).to(self.device)

        # Layer 3: Channel
        log("Initializing channel environment...")
        self.env = MultiCSCAEnvironment(
            n_cscas=n_cscas,
            n_relays=n_relays,
            n_base_stations=n_cscas,
            n_mcs=n_mcs,
        )

        # Source simplifier (Algorithm 1: MSS selection)
        log("Initializing source simplifier (BERT)...")
        self.simplifier = SourceSimplifier()

        log("CSCA pipeline ready.")

    def parse_intent(self, user_text: str) -> dict:
        if self.intent_parser is None:
            return {
                "intent_text": user_text,
                "intent_vector": [0.9, 0.8],
                "delay_intent": 0.9,
                "quality_intent": 0.8,
            }
        return self.intent_parser.parse(user_text)

    def generate_policy(self, intent_result: dict, system_state: dict = None):
        intent_vector = intent_result.get("intent_vector", [0.9, 0.8])
        intent_vectors = [intent_vector] * self.n_cscas

        graph_emb, node_embs, message_embs = self.han.encode_state(
            system_state, intent_vectors=intent_vectors
        )
        action = self.policy(graph_emb, message_embs=message_embs)
        parsed = self.policy.parse_action(
            action, self.n_cscas, self.n_relays, self.n_mcs
        )
        return action, parsed, graph_emb

    def run_episode(self, intents: list, system_state: dict = None) -> dict:
        if system_state is None:
            system_state = self.env.generate_state()

        intent_results = []
        for text in intents:
            ir = self.parse_intent(text)
            intent_results.append(ir)
            system_state["SCt"]["delay_intents"][
                min(len(intent_results) - 1, self.n_cscas - 1)
            ] = max(ir["delay_intent"] * 5.0, 0.1)
            system_state["SCt"]["quality_intents"][
                min(len(intent_results) - 1, self.n_cscas - 1)
            ] = ir["quality_intent"]

        # Simplify LAM-generated descriptions (Algorithm 1: MSS)
        # Only for non-original-text modality descriptions
        simplified_results = []
        for text in intents:
            mss_result = self.simplifier.find_mss(text, eta=0.85)
            simplified_results.append(mss_result)

        # Compute MIM and select MCS for each simplified description
        mcs_selections = []
        for mss in simplified_results:
            mim = compute_mim(mss["simplified"])
            mcs = select_mcs(mim, epsilon_weight=0.1)
            mcs_selections.append(mcs)

        action, parsed_action, graph_emb = self.generate_policy(
            intent_results[0], system_state
        )
        channel_result = self.env.step(parsed_action, system_state)
        tasks = channel_result["tasks"]

        cscqi_values = [
            compute_cscqi(
                t["tau_S"], t["vartheta_S"],
                t["tau_S_int"], t["vartheta_S_int"]
            )
            for t in tasks
        ]
        isr = compute_isr(tasks)
        avg_cscqi = np.mean(cscqi_values)
        avg_delay = np.mean([t["tau_S"] for t in tasks])

        return {
            "intents": intents,
            "intent_vectors": [ir["intent_vector"] for ir in intent_results],
            "graph_embedding_shape": list(graph_emb.shape),
            "action_shape": list(action.shape),
            "tasks": tasks,
            "cscqi_values": cscqi_values,
            "avg_cscqi": avg_cscqi,
            "isr": isr,
            "avg_delay": avg_delay,
            "simplified_texts": simplified_results,
            "mcs_selections": mcs_selections,
        }


def end_to_end_demo():
    pipeline = CSCAPipeline(n_cscas=5, load_lam=True)

    test_intents = [
        "Send it within 1 second",
        "Send it with high resolution",
        "Send the data reliably even if slow",
    ]

    log("Running end-to-end demo...")
    results = []

    for intent_text in test_intents:
        result = pipeline.run_episode([intent_text])

        print("\n" + "=" * 50)
        print(f"Intent: {intent_text!r}")
        print(f"Intent vector: {result['intent_vectors'][0]}")
        print(f"Graph embedding shape: {result['graph_embedding_shape']}")
        print(f"Action shape: {result['action_shape']}")
        print(f"Avg CSCQI: {result['avg_cscqi']:.4f}")
        print(f"ISR: {result['isr']:.3f}")
        print(f"Avg delay: {result['avg_delay']:.4f}s")
        satisfied = "YES" if result["isr"] > 0.5 else "NO"
        print(f"Majority intent satisfied: {satisfied}")
        print("=" * 50)

        results.append({
            "intent": intent_text,
            "intent_vector": result["intent_vectors"][0],
            "avg_cscqi": result["avg_cscqi"],
            "isr": result["isr"],
            "avg_delay": result["avg_delay"],
        })

    out_path = os.path.join(RESULTS_PATH, "demo_output.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    log(f"Demo complete. Results saved to {out_path}")
    return results


if __name__ == "__main__":
    end_to_end_demo()
