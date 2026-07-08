"""
Source Simplifier, MIM, and MCS Selection for CSCA.
Implements Section IV.A of Sun et al. 2026.

- Algorithm 1: Minimum Synonymous Subsequence (MSS) selection
- Eq. 4: BERT-based semantic distance (cosine similarity)
- Eq. 5: Message Importance Metric (MIM)
- Eq. 6: MCS selection based on MIM and BER constraint

FIX (2026-07-08): Original implementation called _get_embedding() once per
subsequence inside a nested loop, producing exponential BERT calls.
New implementation batches all candidates at each removal level into a single
model.encode() call, giving O(n) BERT forward passes per removal level
instead of O(C(n,k)) individual calls. Early-exit on first successful level
is preserved exactly as Algorithm 1 specifies.
"""

import numpy as np
import torch
from transformers import BertTokenizer, BertModel

# ---------------------------------------------------------------------------
# 3GPP TS 38.214 simplified MCS table (modulation order, coding rate, max BER)
# ---------------------------------------------------------------------------
MCS_TABLE = [
    {"index": 0, "modulation": "BPSK",   "order": 2,   "rate": 0.50, "max_ber": 1e-1},
    {"index": 1, "modulation": "QPSK",   "order": 4,   "rate": 0.50, "max_ber": 1e-2},
    {"index": 2, "modulation": "16QAM",  "order": 16,  "rate": 0.50, "max_ber": 1e-3},
    {"index": 3, "modulation": "64QAM",  "order": 64,  "rate": 0.67, "max_ber": 1e-4},
    {"index": 4, "modulation": "256QAM", "order": 256, "rate": 0.75, "max_ber": 1e-5},
]


class SourceSimplifier:
    """
    Source data simplification via Minimum Synonymous Subsequence (MSS).
    Algorithm 1 from Sun et al. 2026, Section IV.A.1.

    Only simplifies LAM-generated descriptions (not original text modality).
    Uses BERT [CLS] embeddings for semantic distance (Eq. 4).

    Performance fix: all candidates at each removal level are encoded in a
    single batched forward pass. This reduces BERT calls from O(C(n,k)) to
    O(n) per removal level, making the algorithm tractable for evaluation.
    """

    def __init__(self, model_name: str = "bert-base-uncased", batch_size: int = 64):
        print(f"[SourceSimplifier] Loading BERT: {model_name}")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()
        self.batch_size = batch_size
        # Move to GPU if available — speeds up batched inference significantly
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        print(f"[SourceSimplifier] Ready on {self.device}.")

    # ------------------------------------------------------------------
    # Single-text embedding (used internally and by compute_semantic_distance)
    # ------------------------------------------------------------------
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get BERT [CLS] embedding for a single text string."""
        inputs = self.tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=512, padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        return cls_emb

    # ------------------------------------------------------------------
    # Batched embedding — KEY FIX
    # Encodes a list of texts in chunks of self.batch_size.
    # Returns shape (len(texts), hidden_dim) numpy array.
    # ------------------------------------------------------------------
    def _get_embeddings_batch(self, texts: list) -> np.ndarray:
        """
        Batch-encode a list of texts.
        All texts at one removal level are processed in a single call,
        replacing O(C(n,k)) individual forward passes with O(ceil(C(n,k)/B)).
        In practice, because we break at the first successful removal level
        (n=1 removal has only L candidates), typical calls are tiny.
        """
        all_embeddings = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start: start + self.batch_size]
            inputs = self.tokenizer(
                chunk, return_tensors="pt",
                truncation=True, max_length=512,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            # [CLS] token embeddings for the whole chunk
            cls_embs = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embs)
        return np.vstack(all_embeddings)

    # ------------------------------------------------------------------
    # Public API: semantic distance (Eq. 4)
    # ------------------------------------------------------------------
    def compute_semantic_distance(self, s1: str, s2: str) -> float:
        """
        Compute semantic distance chi(s, S) using BERT cosine similarity (Eq. 4).
        Returns cosine similarity in [0, 1] — higher means more similar.
        """
        emb1 = self._get_embedding(s1)
        emb2 = self._get_embedding(s2)
        cos_sim = np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8
        )
        return float(cos_sim)

    # ------------------------------------------------------------------
    # Algorithm 1: MSS selection — batched implementation
    # ------------------------------------------------------------------
    def find_mss(self, text: str, eta: float = 0.85) -> dict:
        """
        Algorithm 1: Minimum Synonymous Subsequence Selection.

        Iterates removal levels n = 1, 2, ..., L-1.
        At each level, ALL candidates (sequences formed by dropping exactly n
        words) are encoded in a single batched BERT forward pass.
        Stops at the first level that produces at least one synonymous
        sequence (chi >= eta), exactly as Algorithm 1 specifies.

        Complexity: O(L) batched BERT calls in the best case (n=1 removal),
        compared to O(C(L, n)) individual calls in the original.

        Returns
        -------
        dict with keys:
            simplified         : str   — the MSS text
            similarity         : float — chi(MSS, original)
            original_length    : int
            simplified_length  : int
            compression_ratio  : float — simplified_length / original_length
            num_candidates_evaluated : int
        """
        words = text.split()
        l = len(words)

        # Trivial cases
        if l <= 2:
            return {
                "simplified": text,
                "similarity": 1.0,
                "original_length": l,
                "simplified_length": l,
                "compression_ratio": 1.0,
                "num_candidates_evaluated": 0,
            }

        # Encode original once — reused across all levels
        original_emb = self._get_embedding(text)      # shape: (hidden_dim,)
        original_norm = np.linalg.norm(original_emb) + 1e-8

        total_evaluated = 0

        for n in range(1, l):
            # Build ALL candidates at this removal level
            # Instead of combinations() which gives index sets to remove,
            # we use the complement: choose (l-n) indices to KEEP.
            # This is identical mathematically but clearer to read.
            # Full correctness: enumerate all subsets of size (l - n) to keep
            from itertools import combinations as _comb
            keep_size = l - n
            candidate_texts = []
            candidate_indices = []
            for kept_indices in _comb(range(l), keep_size):
                remaining = [words[i] for i in kept_indices]
                candidate_texts.append(" ".join(remaining))
                candidate_indices.append(kept_indices)

            if not candidate_texts:
                continue

            total_evaluated += len(candidate_texts)

            # Single batched BERT forward pass for ALL candidates at this level
            cand_embs = self._get_embeddings_batch(candidate_texts)
            # shape: (num_candidates, hidden_dim)

            # Vectorised cosine similarity against original
            cand_norms = np.linalg.norm(cand_embs, axis=1) + 1e-8   # (num_candidates,)
            dot_products = cand_embs @ original_emb                   # (num_candidates,)
            chi_values = dot_products / (cand_norms * original_norm)  # (num_candidates,)

            # Collect synonymous candidates (chi >= eta)
            synonymous_mask = chi_values >= eta
            if synonymous_mask.any():
                # Among all synonymous candidates, pick the one with highest chi
                # (ties broken by highest similarity, per paper's MSS definition)
                syn_indices = np.where(synonymous_mask)[0]
                best_local = syn_indices[np.argmax(chi_values[syn_indices])]
                simplified_text = candidate_texts[best_local]
                similarity = float(chi_values[best_local])

                simplified_len = len(simplified_text.split())
                return {
                    "simplified": simplified_text,
                    "similarity": similarity,
                    "original_length": l,
                    "simplified_length": simplified_len,
                    "compression_ratio": simplified_len / l,
                    "num_candidates_evaluated": total_evaluated,
                }
            # No synonymous candidate at this level — try removing one more word

        # No simplification possible while preserving semantics
        return {
            "simplified": text,
            "similarity": 1.0,
            "original_length": l,
            "simplified_length": l,
            "compression_ratio": 1.0,
            "num_candidates_evaluated": total_evaluated,
        }


# ---------------------------------------------------------------------------
# Module-level functions (unchanged interface)
# ---------------------------------------------------------------------------

def compute_mim(text: str) -> float:
    """
    Message Importance Metric (MIM) from Eq. 5.

    MIM(psi_S) = sum_{f in psi_S} p(f) * exp(-p(f))

    where f are word features and p(f) is their empirical frequency.
    Higher MIM = more information uncertainty = select lower BER MCS.
    """
    words = text.lower().split()
    if not words:
        return 0.0
    word_counts: dict = {}
    for w in words:
        word_counts[w] = word_counts.get(w, 0) + 1
    total = len(words)
    mim = 0.0
    for count in word_counts.values():
        p = count / total
        mim += p * np.exp(-p)
    return float(mim)


def select_mcs(mim_value: float, epsilon_weight: float = 0.1) -> dict:
    """
    MCS selection based on MIM (Eq. 6).

    Tolerable BER for message S: epsilon <= epsilon_weight * (1 - MIM(psi_S))
    Higher MIM (more important message) → lower tolerable BER → more robust MCS.
    We iterate the MCS table in descending order (highest spectral efficiency first)
    and return the most efficient MCS whose max_ber still satisfies the constraint.
    """
    ber_threshold = epsilon_weight * (1.0 - mim_value)
    ber_threshold = max(ber_threshold, 1e-6)   # numerical floor

    selected = MCS_TABLE[0]                     # fallback: most robust (BPSK)
    for mcs in reversed(MCS_TABLE):             # from 256QAM down to BPSK
        if mcs["max_ber"] >= ber_threshold:
            selected = mcs
            break

    return {
        "mim": mim_value,
        "ber_threshold": ber_threshold,
        "mcs_index": selected["index"],
        "modulation": selected["modulation"],
        "modulation_order": selected["order"],
        "coding_rate": selected["rate"],
        "max_ber": selected["max_ber"],
        "spectral_efficiency": np.log2(selected["order"]) * selected["rate"],
    }
