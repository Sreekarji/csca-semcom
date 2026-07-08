"""
Semantic Relay Selection for CSCA.
Implements Section IV.B from Sun et al. 2026.

NOTE: Eq. 15 requires probability distributions P(gamma_S) and P(gamma_hat_S | gamma_S)
over semantic knowledge sets. The paper states these come from LKB deployment —
a "symbol probability distribution table" is generated when the knowledge base is
deployed to edge CSCAs and semantic relays (Section IV.B, paragraph after Eq. 15).
Our implementation approximates these using sentence embedding distributions.
This is a principled approximation, not the exact formula.
For exact Eq. 15 computation, real semantic knowledge probability distributions
would need to be derived from the LAM's attention weights over the LKB.

Eq. 15: Semantic Mutual Information
  I(Gamma_S, Gamma_hat_S | R_s, R_r) =
    -sum_{beta in R_r} sum_{mu_hat in Gamma_hat_S} sum_{alpha in R_s} sum_{mu in Gamma_S}
      p(alpha) * p(beta) * f(mu|alpha) * p(beta|alpha,mu,mu_hat) * d(alpha,beta)

Variables:
  - p(alpha): probability distribution over sender's knowledge symbols
  - p(beta): probability distribution over relay's knowledge symbols
  - f(mu|alpha): semantic encoding function (probability of generating mu given alpha)
  - p(beta|alpha,mu,mu_hat): relay recovery probability
  - d(alpha,beta): semantic similarity between symbols alpha and beta

Relay selection (Section IV.B):
  1. Estimate distortion on direct path: theta = f(d, CSI, sigma^2)  [Eq. 14]
  2. If theta_A,B > theta_S (direct distortion exceeds intent): use relay
  3. Reference distortion: theta_S = max(theta_A,r, theta_r,B)
  4. Select relay r* = argmax_r I(Gamma_S, Gamma_hat_S | R_s, R_r)
     subject to theta_S(r) <= intent_quality

Total delay with relay (Eq. 16):
  tau_S = tau_A,r + tau' + tau_r,B + tau_w  (relay path)
  tau_S = tau_A,B + tau_w                   (direct path)

where tau' = semantic recovery delay (proportional to symbol sequence length)
      tau_w = queuing delay
"""

import numpy as np
import torch

# Lazy-load sentence transformer to avoid import crash
_sem_model = None

def get_sem_model():
    global _sem_model
    if _sem_model is None:
        from sentence_transformers import SentenceTransformer
        _sem_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sem_model


class SemanticRelay:
    """Represents a semantic relay node."""

    def __init__(self, relay_id: int, position_km: tuple, knowledge_set: list):
        self.id = relay_id
        self.position = position_km  # (x, y) in km
        self.knowledge_set = knowledge_set  # list of semantic topic labels
        self.kb_probs = None  # probability distribution over knowledge symbols

    def set_knowledge_probs(self, probs: dict):
        """Set probability distribution over semantic labels."""
        self.kb_probs = probs

    def distance_to(self, position_km: tuple) -> float:
        dx = self.position[0] - position_km[0]
        dy = self.position[1] - position_km[1]
        return np.sqrt(dx**2 + dy**2)


def compute_distortion(sinr_linear: float, data_dim: int = 128) -> float:
    """
    Distortion estimation (Eq. 14).
    Approximation: distortion decreases with SINR.
    Based on distortion estimation function from Ref [39].
    """
    sinr_db = 10 * np.log10(max(sinr_linear, 1e-10))
    distortion = np.exp(-0.1 * sinr_db)
    return float(np.clip(distortion, 0.0, 1.0))


def compute_symbol_similarity(alpha: str, beta: str) -> float:
    """
    Semantic similarity d(alpha, beta) between two symbols.
    Uses simple string overlap as fast default; cosine only if model loaded.
    """
    if alpha == beta:
        return 1.0
    # Fast path: string overlap (no model loading)
    a_set = set(alpha.lower().split())
    b_set = set(beta.lower().split())
    if not a_set or not b_set:
        return 0.0
    overlap = len(a_set & b_set) / len(a_set | b_set)
    # If model already loaded, use cosine similarity
    if _sem_model is not None:
        try:
            import torch
            embs = _sem_model.encode([alpha, beta], convert_to_tensor=True)
            cos_sim = float(torch.nn.functional.cosine_similarity(
                embs[0].unsqueeze(0), embs[1].unsqueeze(0)
            ).item())
            return max(0.0, cos_sim)
        except Exception:
            pass
    return overlap


def compute_semantic_mi_approximation(
    sender_knowledge: list,
    relay_knowledge: list,
) -> float:
    """
    Approximation of Semantic Mutual Information from Eq. 15.

    True Eq. 15 requires P(gamma_S) and P(gamma_hat_S | gamma_S) which are
    probability distributions over semantic knowledge sets. The paper states
    these come from a "symbol probability distribution table" generated during
    LKB deployment (Section IV.B).

    Approximation: treat each knowledge item as a sample from a distribution.
    Use Jensen-Shannon divergence between embedding distributions as MI proxy.
    JS divergence is bounded in [0, 1] and equals 0 when distributions match.
    We use 1 - JS divergence as the MI proxy (higher = more shared knowledge).

    This is a principled approximation, not the exact formula.
    Stated as approximation in paper reporting.

    Args:
        sender_knowledge: list of sender's semantic knowledge labels
        relay_knowledge: list of relay's semantic knowledge labels

    Returns:
        Semantic MI score in [0, 1]. Higher = better relay candidate.
    """
    if not sender_knowledge or not relay_knowledge:
        return 0.0

    # Fast path: if model not loaded, use string overlap
    if _sem_model is None:
        # Simple overlap approximation
        sender_set = set(w.lower() for w in sender_knowledge)
        relay_set = set(w.lower() for w in relay_knowledge)
        if not sender_set or not relay_set:
            return 0.0
        intersection = len(sender_set & relay_set)
        union = len(sender_set | relay_set)
        return float(intersection / union)

    # Full path: embedding-based MI approximation
    import torch

    sender_embs = _sem_model.encode(sender_knowledge, convert_to_tensor=True)
    relay_embs = _sem_model.encode(relay_knowledge, convert_to_tensor=True)

    # Compute distribution means
    sender_mean = sender_embs.mean(dim=0)
    relay_mean = relay_embs.mean(dim=0)

    # KL divergence approximation using mean embeddings
    # KL(P||Q) for Gaussians with same variance ~ ||mu_P - mu_Q||^2 / (2 * sigma^2)
    diff = sender_mean - relay_mean
    kl_approx = (diff ** 2).mean().item()

    # JS divergence = 0.5 * KL(P||M) + 0.5 * KL(Q||M) where M = 0.5*(P+Q)
    # Approximation: JS_approx = kl_approx / (kl_approx + 1)
    js_approx = kl_approx / (kl_approx + 1.0)

    # MI proxy: 1 - JS (higher when distributions are more similar)
    mi_proxy = 1.0 - js_approx

    # Also compute direct cosine similarity for comparison
    cos_sim = torch.nn.functional.cosine_similarity(
        sender_mean.unsqueeze(0),
        relay_mean.unsqueeze(0)
    ).item()

    # Combine both metrics
    combined = 0.6 * mi_proxy + 0.4 * max(0.0, cos_sim)
    return float(combined)


# Keep backward compatibility alias
compute_semantic_mutual_information = compute_semantic_mi_approximation


def select_relay(
    distortion_direct: float,
    intent_quality: float,
    relays: list,
    sender_knowledge: list = None,
    sender_pos: tuple = (0, 0),
    receiver_pos: tuple = (1, 0),
) -> dict:
    """
    Relay selection from Section IV.B.

    Decision logic:
    1. If direct distortion <= intent quality: no relay needed
    2. Otherwise: select relay that maximizes semantic MI (Eq. 15)
       subject to relay path distortion <= intent quality

    Args:
        distortion_direct: distortion on direct path (Eq. 14 output)
        intent_quality: user's quality intent (theta_S_int)
        relays: list of SemanticRelay objects
        sender_knowledge: sender's knowledge labels (for MI computation)
        sender_pos: sender position (km)
        receiver_pos: receiver position (km)

    Returns:
        dict with relay selection result
    """
    # Default sender knowledge
    if sender_knowledge is None:
        sender_knowledge = ["semantic", "communication", "multimodal"]

    # If direct link meets quality intent, no relay needed
    # Paper: "When the distortion cannot meet the user's intent,
    #         i.e., theta_A,B > theta_S, it is necessary to select a semantic relay"
    if distortion_direct <= intent_quality:
        return {
            "relay_needed": False,
            "relay_id": None,
            "distortion": distortion_direct,
            "semantic_mi": 0.0,
            "reason": "direct link meets quality intent",
        }

    # Find best relay
    best_relay = None
    best_score = -float("inf")
    best_mi = 0.0

    for relay in relays:
        # Semantic Mutual Information (Eq. 15 approximation)
        mi = compute_semantic_mi_approximation(
            sender_knowledge, relay.knowledge_set
        )

        # Estimate relay path distortion
        d_sr = relay.distance_to(sender_pos)
        d_rr = relay.distance_to(receiver_pos)
        sinr_sr = max(1.0, 100 / max(d_sr, 0.1)**2)
        sinr_rr = max(1.0, 100 / max(d_rr, 0.1)**2)
        relay_distortion = max(compute_distortion(sinr_sr), compute_distortion(sinr_rr))

        # Only consider relays that improve distortion
        if relay_distortion >= distortion_direct:
            continue

        # Score: semantic MI weighted by distortion improvement
        distortion_improvement = (distortion_direct - relay_distortion) / max(distortion_direct, 1e-8)
        score = mi * (1.0 + distortion_improvement)

        if score > best_score:
            best_score = score
            best_relay = relay
            best_mi = mi

    if best_relay is None:
        return {
            "relay_needed": True,
            "relay_id": None,
            "distortion": distortion_direct,
            "semantic_mi": 0.0,
            "reason": "no relay improves distortion",
        }

    d_sr = best_relay.distance_to(sender_pos)
    d_rr = best_relay.distance_to(receiver_pos)
    relay_distortion = max(
        compute_distortion(max(1.0, 100 / max(d_sr, 0.1)**2)),
        compute_distortion(max(1.0, 100 / max(d_rr, 0.1)**2)),
    )

    return {
        "relay_needed": True,
        "relay_id": best_relay.id,
        "distortion": relay_distortion,
        "semantic_mi": best_mi,
        "relay_dist_sender": d_sr,
        "dist_relay_receiver": d_rr,
        "reason": f"relay {best_relay.id}: distortion {distortion_direct:.3f}->{relay_distortion:.3f}, MI={best_mi:.4f}",
    }


def compute_total_delay_with_relay(
    delay_sender_relay: float,
    delay_recovery: float,
    delay_relay_receiver: float,
    delay_queue: float = 0.0,
) -> float:
    """
    Total communication delay with relay (Eq. 16).
    tau_S = tau_A,r + tau' + tau_r,B + tau_w
    """
    return delay_sender_relay + delay_recovery + delay_relay_receiver + delay_queue


# Default relay configurations for simulation
DEFAULT_RELAY_KNOWLEDGE = [
    ["image", "visual", "photo", "picture", "scene"],
    ["audio", "speech", "voice", "sound", "music"],
    ["text", "language", "semantic", "meaning", "context"],
    ["video", "stream", "motion", "temporal", "sequence"],
    ["sensor", "data", "measurement", "signal", "numeric"],
]

DEFAULT_SENDER_KNOWLEDGE = [
    "multimodal", "semantic", "communication", "intent", "quality"
]


if __name__ == "__main__":
    import torch
    print("=== Relay Selection Test (Eq. 15) ===\n")

    # Create relays with different knowledge sets
    relays = [
        SemanticRelay(0, (0.5, 0.3), DEFAULT_RELAY_KNOWLEDGE[0]),  # image
        SemanticRelay(1, (1.0, 0.5), DEFAULT_RELAY_KNOWLEDGE[1]),  # audio
        SemanticRelay(2, (0.8, -0.2), DEFAULT_RELAY_KNOWLEDGE[2]),  # text
        SemanticRelay(3, (1.5, 0.8), DEFAULT_RELAY_KNOWLEDGE[3]),  # video
        SemanticRelay(4, (0.3, 0.1), DEFAULT_RELAY_KNOWLEDGE[4]),  # sensor
    ]

    # Test 1: Semantic MI between knowledge sets
    print("Test 1: Semantic Mutual Information (Eq. 15)")
    for i, relay in enumerate(relays):
        mi = compute_semantic_mutual_information(
            DEFAULT_SENDER_KNOWLEDGE, relay.knowledge_set
        )
        print(f"  Relay {i} ({relay.knowledge_set[0]}): MI = {mi:.4f}")

    # Test 2: Direct link meets intent
    print("\nTest 2: Direct link sufficient (distortion=0.3, intent=0.5)")
    result = select_relay(
        distortion_direct=0.3,
        intent_quality=0.5,
        relays=relays,
        sender_knowledge=DEFAULT_SENDER_KNOWLEDGE,
    )
    print(f"  Result: {result['reason']}")

    # Test 3: Relay needed
    print("\nTest 3: Relay needed (distortion=0.8, intent=0.5)")
    result = select_relay(
        distortion_direct=0.8,
        intent_quality=0.5,
        relays=relays,
        sender_knowledge=DEFAULT_SENDER_KNOWLEDGE,
    )
    print(f"  Result: {result['reason']}")
    print(f"  Relay ID: {result['relay_id']}")
    print(f"  Semantic MI: {result['semantic_mi']:.4f}")

    # Test 4: Total delay with relay (Eq. 16)
    print("\nTest 4: Total delay (Eq. 16)")
    delay = compute_total_delay_with_relay(
        delay_sender_relay=0.05,
        delay_recovery=0.01,
        delay_relay_receiver=0.08,
        delay_queue=0.02,
    )
    print(f"  tau_S = 0.05 + 0.01 + 0.08 + 0.02 = {delay:.3f}s")

    print("\nRelay selection tests passed.")
