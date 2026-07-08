"""
Wireless Channel Simulation for CSCA.
Implements Section IV.A from Sun et al. 2026 with exact 3GPP parameters.

FIX (2026-07-08): Replaced proportional bandwidth normalisation with
softmax-temperature allocation in MultiCSCAEnvironment.step().

Root cause of HDM ≈ baselines:
  OLD: bw_alloc = bw_frac / sum(all_fracs) * total_bw
  When HDM outputs all-similar sigmoid values (~0.5–0.8), the ratio
  bw_frac/sum cancels to 1/N for every task → identical to static
  equal allocation → zero gradient signal between HDM and baseline.

Fix: softmax with temperature tau=BW_SOFTMAX_TEMP amplifies small
differences in HDM logits into meaningful BW differences.
  bw_alloc[i] = softmax(logits / tau)[i] * total_bw
At tau=0.3 a logit difference of 0.2 becomes a ~50% BW difference.
At tau=1.0 (original sigmoid range) the same difference is ~5%.

The temperature is exposed as a module constant so it can be tuned
without touching training code.
"""

import numpy as np
import torch
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mcs_table import select_mcs_for_sinr, compute_rate_from_mcs
from relay_selection import (
    SemanticRelay, select_relay as relay_select,
    compute_distortion as relay_compute_distortion,
    DEFAULT_RELAY_KNOWLEDGE, DEFAULT_SENDER_KNOWLEDGE,
)

# ---------------------------------------------------------------------------
# 3GPP physical-layer constants
# ---------------------------------------------------------------------------
NOISE_PSD_DBM_HZ   = -174.0
SIGMA_S            = 8.0        # shadow fading std dev (dB), 3GPP TR 38.901
N_CLUSTERS_LOS     = 12         # LOS clusters, 3GPP TR 36.873 Table 7.3-6
N_CLUSTERS_NLOS    = 19         # NLOS clusters
N_RAYS             = 20         # rays per cluster
BLOCK_LENGTH       = 1000
DEFAULT_TX_POWER_DBM = 23.0
DEFAULT_BANDWIDTH_HZ = 10e6
INTERFERENCE_CELLS = 6          # top-K interferers, Section IV.A.3

# ---------------------------------------------------------------------------
# BW allocation constant — controls how sharply HDM output maps to BW
# ---------------------------------------------------------------------------
BW_SOFTMAX_TEMP = 0.3   # lower = more differentiation; 1.0 = near-proportional


class WirelessChannel:
    """
    Single-link wireless channel following Sun et al. 2026 Section IV.A.

    Models:
      - Path loss:     128.1 + 37.6 * log10(d_km)  [Eq. 48 reference]
      - Shadow fading: N(0, sigma_s^2)              [3GPP TR 38.901]
      - RSRP:          tx_power - PL + SF + multipath_gain
      - Interference:  top-INTERFERENCE_CELLS RSRP values [Eq. 8]
      - SINR:          Eq. 9
      - Rate:          finite-blocklength [Eq. 10] or MCS table
      - Delay:         D_S / nu  [Eq. 12]
      - Distortion:    exp(-0.1 * SINR_dB)  [Eq. 14 proxy]
    """

    def __init__(
        self,
        bandwidth_hz: float = DEFAULT_BANDWIDTH_HZ,
        block_length: int   = BLOCK_LENGTH,
        noise_psd_dbm_hz: float = NOISE_PSD_DBM_HZ,
        sigma_s: float = SIGMA_S,
    ):
        self.bandwidth        = bandwidth_hz
        self.block_length     = block_length
        self.noise_psd        = noise_psd_dbm_hz
        self.sigma_s          = sigma_s
        noise_power_dbm = noise_psd_dbm_hz + 10 * np.log10(bandwidth_hz)
        self.noise_power_linear = 10 ** ((noise_power_dbm - 30) / 10)

    # ------------------------------------------------------------------
    # Path loss and channel gain
    # ------------------------------------------------------------------
    def compute_path_loss_db(self, d_km: float) -> float:
        """128.1 + 37.6 * log10(d_km) — exact 3GPP model 128.1."""
        d_km = max(d_km, 1e-4)
        return 128.1 + 37.6 * np.log10(d_km)

    def compute_shadow_fading_db(self) -> float:
        """N(0, sigma_s^2) — 3GPP TR 38.901, sigma_s = 8 dB."""
        return np.random.normal(0, self.sigma_s)

    def compute_rsrp(
        self, tx_power_dbm: float, d_km: float, los: bool = True
    ) -> float:
        """Reference Signal Received Power (Eq. 7)."""
        pl  = self.compute_path_loss_db(d_km)
        sf  = self.compute_shadow_fading_db()
        n_clusters = N_CLUSTERS_LOS if los else N_CLUSTERS_NLOS
        multipath_gain_db = 10 * np.log10(n_clusters * N_RAYS)
        return tx_power_dbm - pl + sf + multipath_gain_db

    # ------------------------------------------------------------------
    # Interference and SINR
    # ------------------------------------------------------------------
    def compute_interference(
        self, interferer_powers_dbm: list, top_k: int = INTERFERENCE_CELLS
    ) -> float:
        """
        Inter-cell interference power (Eq. 8).
        Sum of the top_k strongest interferers in linear scale.
        """
        if not interferer_powers_dbm:
            return 0.0
        sorted_powers = sorted(interferer_powers_dbm, reverse=True)[:top_k]
        return sum(10 ** ((p - 30) / 10) for p in sorted_powers)

    def compute_sinr(
        self, rx_power_dbm: float, interference_linear: float = 0.0
    ) -> float:
        """SINR (Eq. 9): phi / (sigma^2 + rho)."""
        rx_linear = 10 ** ((rx_power_dbm - 30) / 10)
        sinr = rx_linear / (self.noise_power_linear + interference_linear)
        return max(sinr, 1e-10)

    # ------------------------------------------------------------------
    # Rate, delay, distortion
    # ------------------------------------------------------------------
    def compute_transmission_rate(
        self, sinr: float, ber: float = 1e-3, use_mcs_table: bool = True
    ) -> float:
        """
        Data transmission rate nu (Eq. 10).
        Uses MCS table lookup by default; finite-blocklength fallback.
        """
        from scipy.stats import norm
        sinr = max(sinr, 1e-10)
        sinr_db = 10 * np.log10(sinr)
        if use_mcs_table:
            mcs  = select_mcs_for_sinr(sinr_db)
            rate = compute_rate_from_mcs(mcs, self.bandwidth)
            return max(rate, 1e3)
        C     = self.bandwidth * np.log2(1 + sinr)
        V     = sinr / (1 + sinr) ** 2 * np.log2(np.e) ** 2
        Q_inv = norm.ppf(1 - ber)
        rate  = C - np.sqrt(V / self.block_length) * Q_inv
        return max(rate, 1e3)

    def compute_delay(self, data_size_bits: float, rate: float) -> float:
        """tau_A_B = D_S / nu (Eq. 12)."""
        return data_size_bits / max(rate, 1.0)

    def compute_distortion(self, sinr: float) -> float:
        """
        Proxy for distortion vartheta_S (Eq. 14 approximation).
        Exponential decay with SINR_dB; clipped to [0, 1].
        """
        sinr_db = 10 * np.log10(max(sinr, 1e-10))
        distortion = np.exp(-0.1 * sinr_db)
        return float(np.clip(distortion, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Full single-link simulation
    # ------------------------------------------------------------------
    def simulate_channel(
        self,
        tx_power_dbm: float = DEFAULT_TX_POWER_DBM,
        distance_km: float  = 0.1,
        data_size_bits: float = 1e6,
        interference_dbm: float = None,
        interferer_powers_dbm: list = None,
        los: bool = True,
        use_mcs_table: bool = True,
        target_snr_db: float = None,
    ) -> dict:
        """
        Simulate one transmission link.

        If target_snr_db is given, the SINR is set directly (used for
        controlled SNR sweep in experiments). Otherwise, SINR is computed
        from the stochastic channel model.
        """
        if target_snr_db is not None:
            sinr_db = float(target_snr_db)
            sinr    = 10 ** (sinr_db / 10.0)
            rsrp    = tx_power_dbm
        else:
            rsrp = self.compute_rsrp(tx_power_dbm, distance_km, los)
            if interferer_powers_dbm is not None:
                interference_linear = self.compute_interference(interferer_powers_dbm)
            elif interference_dbm is not None:
                interference_linear = 10 ** ((interference_dbm - 30) / 10)
            else:
                interference_linear = 0.0
            sinr    = self.compute_sinr(rsrp, interference_linear)
            sinr_db = 10 * np.log10(sinr)

        rate      = self.compute_transmission_rate(sinr, use_mcs_table=use_mcs_table)
        delay     = self.compute_delay(data_size_bits, rate)
        distortion = self.compute_distortion(sinr)

        result = {
            "rx_power_dbm":  rsrp,
            "sinr_linear":   sinr,
            "sinr_db":       sinr_db,
            "rate_bps":      rate,
            "delay_s":       delay,
            "distortion":    distortion,
        }
        if use_mcs_table:
            mcs = select_mcs_for_sinr(sinr_db)
            result.update({
                "mcs_index":           mcs["mcs_index"],
                "modulation":          mcs["modulation"],
                "code_rate":           mcs["code_rate"],
                "spectral_efficiency": mcs["spectral_efficiency"],
            })
        return result


# ---------------------------------------------------------------------------
# Softmax BW allocation helper
# ---------------------------------------------------------------------------

def _softmax_bw_allocation(
    raw_logits: list,
    total_bw: float,
    temperature: float = BW_SOFTMAX_TEMP,
) -> list:
    """
    Convert raw HDM bandwidth logits to actual bandwidth allocations via
    temperature-scaled softmax.

    At low temperature (0.3), a logit difference of 0.2 maps to ~50% BW
    difference, giving the policy a strong gradient signal to differentiate
    tasks. At temperature 1.0 the allocation is nearly proportional (same
    as the old scheme).

    Parameters
    ----------
    raw_logits : list of float — HDM output for each CSCA (any range)
    total_bw   : float         — total available bandwidth in Hz
    temperature: float         — softmax sharpness; lower = more differentiation

    Returns
    -------
    list of float — per-CSCA bandwidth in Hz, summing to total_bw
    """
    logits = np.array(raw_logits, dtype=np.float64)
    # Numerical stability: subtract max before exp
    scaled = (logits - logits.max()) / max(temperature, 1e-6)
    weights = np.exp(scaled)
    weights = weights / (weights.sum() + 1e-12)
    return (weights * total_bw).tolist()


class MultiCSCAEnvironment:
    """
    Multi-CSCA wireless communication environment.

    Generates random states (channel conditions + task intents) and evaluates
    communication policies. Used by HDM trainer (Algorithm 2) and experiment
    runner.

    Key fix: step() uses softmax-temperature BW allocation instead of
    proportional normalisation. This gives HDM a meaningful gradient —
    outputting higher logits for urgent tasks now actually results in more
    bandwidth for those tasks.
    """

    def __init__(
        self,
        n_cscas: int           = 5,
        n_relays: int          = 5,
        n_base_stations: int   = 5,
        n_mcs: int             = 3,
        bandwidth_total_hz: float = 50e6,
        difficulty: str        = "hard",
    ):
        self.n_cscas         = n_cscas
        self.n_relays        = n_relays
        self.n_bs            = n_base_stations
        self.n_mcs           = n_mcs
        self.bandwidth_total = bandwidth_total_hz
        self.difficulty      = difficulty

        # Per-CSCA channel object — bandwidth updated each step
        self.channel = WirelessChannel(bandwidth_hz=bandwidth_total_hz / n_cscas)

        # Fixed random positions for this environment instance
        self.csca_positions  = [
            (np.random.uniform(0, 2), np.random.uniform(0, 2))
            for _ in range(n_cscas)
        ]
        self.bs_positions = [
            (np.random.uniform(0, 3), np.random.uniform(0, 3))
            for _ in range(n_base_stations)
        ]
        self.relay_positions = [
            (np.random.uniform(0, 2), np.random.uniform(0, 2))
            for _ in range(n_relays)
        ]

    # ------------------------------------------------------------------
    # State generation
    # ------------------------------------------------------------------
    def generate_state(self) -> dict:
        """
        Generate a random system state s_t = {R_t, SC_t}.

        Difficulty controls intent tightness:
          hard   — tight delay/quality intents, large data sizes
          medium — moderate intents
          easy   — relaxed intents, small data sizes
        """
        Rt = {
            "csca_features":      np.random.rand(self.n_cscas, 3).tolist(),
            "relay_features":     np.random.rand(self.n_relays, 3).tolist(),
            "bs_features":        np.random.rand(self.n_bs, 3).tolist(),
            "distortion":         np.random.rand(self.n_cscas).tolist(),
            "bandwidth_remaining": (np.random.rand(self.n_bs) * 0.5 + 0.1).tolist(),
            "positions": {
                "cscas":   self.csca_positions,
                "bs":      self.bs_positions,
                "relays":  self.relay_positions,
            },
        }

        if self.difficulty == "hard":
            delay_intents   = (np.random.rand(self.n_cscas) * 1.4 + 0.1).tolist()
            quality_intents = (np.random.rand(self.n_cscas) * 0.35 + 0.6).tolist()
            data_sizes      = (np.random.rand(self.n_cscas) * 5e6 + 1e6).tolist()
        elif self.difficulty == "medium":
            delay_intents   = (np.random.rand(self.n_cscas) * 2.0 + 0.5).tolist()
            quality_intents = (np.random.rand(self.n_cscas) * 0.4 + 0.4).tolist()
            data_sizes      = (np.random.rand(self.n_cscas) * 10e6 + 2e6).tolist()
        else:  # easy
            delay_intents   = (np.random.rand(self.n_cscas) * 3.0 + 1.0).tolist()
            quality_intents = (np.random.rand(self.n_cscas) * 0.5 + 0.3).tolist()
            data_sizes      = (np.random.rand(self.n_cscas) * 15e6 + 5e6).tolist()

        # Message features encode real task information for CSC graph
        msg_feats = []
        for i in range(self.n_cscas):
            ds_norm = min(data_sizes[i] / 1e7, 1.0)
            di      = delay_intents[i] / 5.0
            qi      = quality_intents[i]
            urgency = (1.0 - di) * 0.5 + (1.0 - qi) * 0.5
            msg_feats.append([ds_norm, di, qi, urgency])

        SCt = {
            "message_features": msg_feats,
            "data_sizes":       data_sizes,
            "delay_intents":    delay_intents,
            "quality_intents":  quality_intents,
        }
        return {"Rt": Rt, "SCt": SCt}

    def generate_state_with_params(self, params: dict) -> dict:
        """Generate state with explicit curriculum params."""
        delay_min, delay_max = params["delay_range"]
        quality_min, quality_max = params["quality_range"]
        data_min, data_max = params["data_size_range"]

        Rt = {
            "csca_features": np.random.rand(self.n_cscas, 3).tolist(),
            "relay_features": np.random.rand(self.n_relays, 3).tolist(),
            "bs_features": np.random.rand(self.n_bs, 3).tolist(),
            "distortion": np.random.rand(self.n_cscas).tolist(),
            "bandwidth_remaining": np.random.rand(self.n_bs).tolist(),
            "positions": {
                "cscas": self.csca_positions,
                "bs": self.bs_positions,
                "relays": self.relay_positions,
            },
        }
        data_sizes = (np.random.rand(self.n_cscas) * (data_max - data_min) + data_min).tolist()
        delay_intents = (np.random.rand(self.n_cscas) * (delay_max - delay_min) + delay_min).tolist()
        quality_intents = (np.random.rand(self.n_cscas) * (quality_max - quality_min) + quality_min).tolist()

        msg_feats = []
        for i in range(self.n_cscas):
            ds_norm = min(data_sizes[i] / 1e7, 1.0)
            di = delay_intents[i] / 5.0
            qi = quality_intents[i]
            urgency = (1.0 - di) * 0.5 + (1.0 - qi) * 0.5
            msg_feats.append([ds_norm, di, qi, urgency])

        SCt = {
            "message_features": msg_feats,
            "data_sizes": data_sizes,
            "delay_intents": delay_intents,
            "quality_intents": quality_intents,
        }
        return {"Rt": Rt, "SCt": SCt}

    # ------------------------------------------------------------------
    # Environment step
    # ------------------------------------------------------------------
    def step(
        self,
        action: dict,
        state: dict  = None,
        target_snr_db: float = None,
    ) -> dict:
        """
        Execute one environment step given a policy action.

        action dict keys:
            "bandwidth"  : torch.Tensor shape (1, n_cscas) — raw HDM logits
            "relay"      : torch.Tensor shape (1, n_cscas, n_relays)
            "mcs"        : torch.Tensor shape (1, n_cscas, n_mcs)

        FIX: BW allocation now uses softmax with temperature BW_SOFTMAX_TEMP.
        This means non-uniform HDM outputs produce non-uniform BW allocations,
        giving the policy a real gradient to learn task prioritisation.

        Returns dict with:
            "tasks" : list of per-CSCA result dicts
            "state" : the state used (generated if not provided)
        """
        if state is None:
            state = self.generate_state()
        SCt = state["SCt"]

        # Build relay objects
        relays = []
        for r_idx in range(self.n_relays):
            pos = (
                self.relay_positions[r_idx]
                if r_idx < len(self.relay_positions)
                else (1.0, 1.0)
            )
            kb = DEFAULT_RELAY_KNOWLEDGE[r_idx % len(DEFAULT_RELAY_KNOWLEDGE)]
            relays.append(SemanticRelay(r_idx, pos, kb))

        # ----------------------------------------------------------------
        # SOFTMAX BW ALLOCATION  (replaces proportional normalisation)
        # ----------------------------------------------------------------
        raw_logits = [
            float(action["bandwidth"][0, i].item())
            for i in range(self.n_cscas)
        ]
        bw_allocations = _softmax_bw_allocation(
            raw_logits, self.bandwidth_total, temperature=BW_SOFTMAX_TEMP
        )
        # bw_allocations[i] is the bandwidth in Hz for CSCA i
        # Sum is guaranteed to equal self.bandwidth_total
        # ----------------------------------------------------------------

        results = []
        for i in range(self.n_cscas):
            bw_alloc = max(bw_allocations[i], 1e3)  # floor at 1 kHz

            # TX power scales mildly with BW fraction (unchanged from original)
            bw_frac  = bw_alloc / self.bandwidth_total
            tx_power = 10 + bw_frac * 13

            distance = np.random.uniform(0.3, 2.0)

            # Inter-cell interference from other base stations
            interferer_powers = []
            if target_snr_db is None:
                for j in range(self.n_bs):
                    if j != i % self.n_bs:
                        d_int = np.random.uniform(0.5, 3.0)
                        rsrp_int = self.channel.compute_rsrp(tx_power, d_int, los=True)
                        interferer_powers.append(rsrp_int)

            # Update channel bandwidth to CSCA i's allocated BW
            self.channel.bandwidth = bw_alloc

            # Recompute noise power for the new bandwidth
            noise_power_dbm = (
                NOISE_PSD_DBM_HZ + 10 * np.log10(bw_alloc)
            )
            self.channel.noise_power_linear = 10 ** ((noise_power_dbm - 30) / 10)

            metrics = self.channel.simulate_channel(
                tx_power_dbm       = tx_power,
                distance_km        = distance,
                data_size_bits     = SCt["data_sizes"][i],
                interferer_powers_dbm = interferer_powers if interferer_powers else None,
                use_mcs_table      = True,
                target_snr_db      = target_snr_db,
            )

            # Relay selection (Eq. 15 approximation)
            relay_info = relay_select(
                distortion_direct  = metrics["distortion"],
                intent_quality     = SCt["quality_intents"][i],
                relays             = relays,
                sender_knowledge   = DEFAULT_SENDER_KNOWLEDGE,
                sender_pos         = (0, 0),
                receiver_pos       = (distance, 0),
            )

            delay      = metrics["delay_s"]
            distortion = metrics["distortion"]
            if relay_info["relay_needed"] and relay_info["relay_id"] is not None:
                distortion    = relay_info["distortion"]
                recovery_delay = (
                    SCt["data_sizes"][i] / max(metrics["rate_bps"], 1e3) * 0.1
                )
                delay = delay + recovery_delay

            results.append({
                "tau_S":          delay,
                "vartheta_S":     distortion,
                "tau_S_int":      SCt["delay_intents"][i],
                "vartheta_S_int": SCt["quality_intents"][i],
                "sinr_db":        metrics["sinr_db"],
                "rate_bps":       metrics["rate_bps"],
                "bw_alloc_hz":    bw_alloc,                  # for logging/debug
                "mcs_index":      metrics.get("mcs_index", 0),
                "modulation":     metrics.get("modulation", "QPSK"),
                "relay_used":     relay_info["relay_needed"] and relay_info["relay_id"] is not None,
                "relay_id":       relay_info.get("relay_id"),
                "semantic_mi":    relay_info.get("semantic_mi", 0.0),
            })

        return {"tasks": results, "state": state}
