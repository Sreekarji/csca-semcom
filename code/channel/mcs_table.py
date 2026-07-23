"""
3GPP TS 38.214 MCS Tables for CSCA.
Tables 5.1.3.1-1, 5.1.3.1-2, 5.1.3.1-3.

Each entry: (mcs_index, modulation_order, target_code_rate_x1024, spectral_efficiency)
"""

import numpy as np

# Table 5.1.3.1-1: QPSK, 16QAM, 64QAM (general use)
MCS_TABLE_1 = [
    # (index, mod_order, code_rate_x1024, spectral_eff)
    (0,  2,   120, 0.2344),   # QPSK
    (1,  2,   157, 0.3066),   # QPSK
    (2,  2,   193, 0.3770),   # QPSK
    (3,  2,   251, 0.4902),   # QPSK
    (4,  2,   308, 0.6016),   # QPSK
    (5,  2,   379, 0.7402),   # QPSK
    (6,  2,   449, 0.8770),   # QPSK
    (7,  2,   526, 1.0273),   # QPSK
    (8,  2,   602, 1.1758),   # QPSK
    (9,  2,   679, 1.3262),   # QPSK
    (10, 4,   340, 1.3281),   # 16QAM
    (11, 4,   378, 1.4766),   # 16QAM
    (12, 4,   434, 1.6953),   # 16QAM
    (13, 4,   490, 1.9141),   # 16QAM
    (14, 4,   553, 2.1602),   # 16QAM
    (15, 4,   616, 2.4063),   # 16QAM
    (16, 4,   658, 2.5703),   # 16QAM
    (17, 6,   438, 2.5664),   # 64QAM
    (18, 6,   466, 2.7305),   # 64QAM
    (19, 6,   517, 3.0293),   # 64QAM
    (20, 6,   567, 3.3223),   # 64QAM
    (21, 6,   616, 3.6094),   # 64QAM
    (22, 6,   666, 3.9023),   # 64QAM
    (23, 6,   719, 4.2129),   # 64QAM
    (24, 6,   772, 4.5234),   # 64QAM
    (25, 6,   822, 4.8164),   # 64QAM
    (26, 6,   873, 5.1152),   # 64QAM
    (27, 6,   910, 5.3320),   # 64QAM
    (28, 6,   948, 5.5547),   # 64QAM
]

# Table 5.1.3.1-2: QPSK, 16QAM, 64QAM (higher spectral efficiency)
MCS_TABLE_2 = [
    (0,  2,   120, 0.2344),
    (1,  2,   193, 0.3770),
    (2,  2,   308, 0.6016),
    (3,  2,   449, 0.8770),
    (4,  2,   602, 1.1758),
    (5,  4,   378, 1.4766),
    (6,  4,   434, 1.6953),
    (7,  4,   490, 1.9141),
    (8,  4,   553, 2.1602),
    (9,  4,   616, 2.4063),
    (10, 4,   658, 2.5703),
    (11, 6,   466, 2.7305),
    (12, 6,   517, 3.0293),
    (13, 6,   567, 3.3223),
    (14, 6,   616, 3.6094),
    (15, 6,   666, 3.9023),
    (16, 6,   719, 4.2129),
    (17, 6,   772, 4.5234),
    (18, 6,   822, 4.8164),
    (19, 6,   873, 5.1152),
    (20, 8,   682.5, 5.3320),
    (21, 8,   711, 5.5547),
    (22, 8,   754, 5.8906),
    (23, 8,   797, 6.2266),
    (24, 8,   841, 6.5703),
    (25, 8,   885, 6.9141),
    (26, 8,   916.5, 7.1602),
    (27, 8,   948, 7.4063),
]

# Table 5.1.3.1-3: 256QAM (ultra-high throughput)
MCS_TABLE_3 = [
    (0,  2,   120, 0.2344),
    (1,  2,   193, 0.3770),
    (2,  2,   308, 0.6016),
    (3,  2,   449, 0.8770),
    (4,  2,   602, 1.1758),
    (5,  4,   378, 1.4766),
    (6,  4,   434, 1.6953),
    (7,  4,   490, 1.9141),
    (8,  4,   553, 2.1602),
    (9,  4,   616, 2.4063),
    (10, 4,   658, 2.5703),
    (11, 6,   466, 2.7305),
    (12, 6,   517, 3.0293),
    (13, 6,   567, 3.3223),
    (14, 6,   616, 3.6094),
    (15, 6,   666, 3.9023),
    (16, 6,   719, 4.2129),
    (17, 6,   772, 4.5234),
    (18, 6,   822, 4.8164),
    (19, 6,   873, 5.1152),
    (20, 8,   682.5, 5.3320),
    (21, 8,   711, 5.5547),
    (22, 8,   754, 5.8906),
    (23, 8,   797, 6.2266),
    (24, 8,   841, 6.5703),
    (25, 8,   885, 6.9141),
    (26, 8,   916.5, 7.1602),
    (27, 8,   948, 7.4063),
    (28, 10,  948, 8.3262),   # 256QAM
    (29, 10,  985.14, 8.6523), # 256QAM
]

# Modulation name lookup
MOD_NAMES = {2: "QPSK", 4: "16QAM", 6: "64QAM", 8: "256QAM", 10: "1024QAM"}


def get_mcs_entry(mcs_index: int, table: int = 1) -> dict:
    """Get MCS entry by index from specified table."""
    tables = {1: MCS_TABLE_1, 2: MCS_TABLE_2, 3: MCS_TABLE_3}
    tbl = tables.get(table, MCS_TABLE_1)
    if mcs_index < 0 or mcs_index >= len(tbl):
        mcs_index = min(max(mcs_index, 0), len(tbl) - 1)
    idx, mod_order, code_rate_x1024, spec_eff = tbl[mcs_index]
    return {
        "mcs_index": idx,
        "modulation_order": mod_order,
        "modulation": MOD_NAMES.get(mod_order, f"{mod_order}-QAM"),
        "code_rate": code_rate_x1024 / 1024.0,
        "code_rate_x1024": code_rate_x1024,
        "spectral_efficiency": spec_eff,
    }


def select_mcs_for_sinr(sinr_db: float, table: int = 1) -> dict:
    """
    Select highest MCS achievable at given SINR.
    Uses Shannon capacity as upper bound and picks MCS whose
    spectral efficiency is below the bound.
    """
    sinr_linear = 10 ** (sinr_db / 10)
    shannon_eff = np.log2(1 + sinr_linear)  # bits/symbol

    tables = {1: MCS_TABLE_1, 2: MCS_TABLE_2, 3: MCS_TABLE_3}
    tbl = tables.get(table, MCS_TABLE_1)

    best = tbl[0]
    for entry in tbl:
        if entry[3] <= shannon_eff * 0.85:  # 85% of Shannon (practical margin)
            best = entry

    return {
        "mcs_index": best[0],
        "modulation_order": best[1],
        "modulation": MOD_NAMES.get(best[1], f"{best[1]}-QAM"),
        "code_rate": best[2] / 1024.0,
        "spectral_efficiency": best[3],
        "sinr_db": sinr_db,
        "shannon_efficiency": shannon_eff,
    }


def compute_rate_from_mcs(mcs_entry: dict, bandwidth_hz: float) -> float:
    """Compute data rate from MCS entry and bandwidth."""
    return mcs_entry["spectral_efficiency"] * bandwidth_hz


def compute_mim(token_probs: list) -> float:
    """
    Message Importance Measure (Eq. 5).
    MIM(psi_S) = sum over f: p(f) * exp(-p(f))
    Higher MIM = message is more predictable/certain = can tolerate higher BER.
    """
    if not token_probs:
        return 0.5
    total = sum(token_probs)
    if total <= 0:
        return 0.5
    norm_probs = [p / total for p in token_probs]
    return float(sum(p * np.exp(-p) for p in norm_probs if p > 0))


def select_mcs_for_mim_and_sinr(
    mim: float,
    sinr_db: float,
    bw_weight: float = 0.5,
    table: int = 1,
) -> dict:
    """
    Select MCS satisfying BOTH:
    1. Spectral efficiency <= Shannon capacity (channel constraint)
    2. Implied BER <= max_ber = bw_weight * (1 - MIM) (Eq. 6 constraint C3)
    """
    max_ber = bw_weight * (1.0 - mim)
    max_ber = float(np.clip(max_ber, 1e-6, 0.5))

    if max_ber >= 0.05:
        max_mod_order = 2   # QPSK only
    elif max_ber >= 0.005:
        max_mod_order = 4   # up to 16QAM
    elif max_ber >= 0.0005:
        max_mod_order = 6   # up to 64QAM
    else:
        max_mod_order = 8   # up to 256QAM

    sinr_linear = 10 ** (sinr_db / 10)
    shannon_eff = np.log2(1 + sinr_linear) * 0.85

    tables = {1: MCS_TABLE_1, 2: MCS_TABLE_2, 3: MCS_TABLE_3}
    tbl = tables.get(table, MCS_TABLE_1)

    best = tbl[0]
    for entry in tbl:
        idx, mod_order, code_rate_x1024, spec_eff = entry
        if mod_order > max_mod_order:
            continue
        if spec_eff > shannon_eff:
            continue
        best = entry

    return {
        "mcs_index": best[0],
        "modulation_order": best[1],
        "modulation": MOD_NAMES.get(best[1], f"{best[1]}-QAM"),
        "code_rate": best[2] / 1024.0,
        "spectral_efficiency": best[3],
        "sinr_db": sinr_db,
        "mim": mim,
        "max_ber": max_ber,
    }


if __name__ == "__main__":
    print("3GPP TS 38.214 MCS Tables")
    print("=" * 60)
    print(f"Table 1: {len(MCS_TABLE_1)} entries (QPSK to 64QAM)")
    print(f"Table 2: {len(MCS_TABLE_2)} entries (QPSK to 256QAM)")
    print(f"Table 3: {len(MCS_TABLE_3)} entries (QPSK to 256QAM)")

    print("\nSample MCS selections:")
    for sinr in [0, 5, 10, 15, 20, 25]:
        mcs = select_mcs_for_sinr(sinr)
        rate = compute_rate_from_mcs(mcs, 10e6)
        print(f"  SINR={sinr:2d}dB: MCS={mcs['mcs_index']:2d} {mcs['modulation']:6s} "
              f"rate={mcs['code_rate']:.3f} eff={mcs['spectral_efficiency']:.2f} "
              f"-> {rate/1e6:.1f} Mbps")
