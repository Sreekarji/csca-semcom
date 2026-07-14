import torch
from torch_geometric.data import HeteroData

class CSCGraphBuilder:
    """
    Builds the Cognitive SemCom (CSC) graph Gt = (Vt, Et)
    as defined in Sun et al. 2026, Section V.B.

    Node types: csca, relay, message, base_station, init
    Edge types:
      (csca, comm_conn, base_station)
      (message, comm_req, csca)
      (message, semantic_conn, relay)
      (init, init_conn, csca)
      (init, init_conn, relay)
      (init, init_conn, base_station)
      (init, init_conn, message)
    """

    def __init__(
        self,
        n_cscas: int = 5,
        n_relays: int = 5,
        n_messages: int = 5,
        n_base_stations: int = 5,
        csca_feat_dim: int = 3,
        relay_feat_dim: int = 3,
        message_feat_dim: int = 4,
        bs_feat_dim: int = 3,
    ):
        self.n_cscas = n_cscas
        self.n_relays = n_relays
        self.n_messages = n_messages
        self.bs_feat_dim = bs_feat_dim
        self.n_bs = n_base_stations
        self.csca_feat_dim = csca_feat_dim
        self.relay_feat_dim = relay_feat_dim
        self.message_feat_dim = message_feat_dim

    def build(self, system_state: dict = None,
              intent_vectors: list = None) -> HeteroData:
        """
        Build CSC graph.
        intent_vectors: list of [delay_urgency, quality_req] per task.
                       If provided, these override system_state delay/quality intents.
                       This ensures HAN sees real LAM-parsed intent, not random values.
        """
        data = HeteroData()
        n_c = self.n_cscas
        n_r = self.n_relays
        n_m = self.n_messages
        n_b = self.n_bs

        if system_state is not None:
            Rt = system_state.get("Rt", {})
            SCt = system_state.get("SCt", {})

            raw_csca = Rt.get("csca_features", torch.randn(n_c, self.csca_feat_dim).tolist())
            raw_relay = Rt.get("relay_features", torch.randn(n_r, self.relay_feat_dim).tolist())
            raw_bs = Rt.get("bs_features", torch.randn(n_b, self.bs_feat_dim).tolist())

            def _fit(feat_list, target_n, feat_dim):
                t = torch.tensor(feat_list, dtype=torch.float)
                if t.shape[0] >= target_n:
                    return t[:target_n]
                else:
                    pad = torch.randn(target_n - t.shape[0], feat_dim)
                    return torch.cat([t, pad], dim=0)

            csca_feats = _fit(raw_csca, n_c, self.csca_feat_dim)
            relay_feats = _fit(raw_relay, n_r, self.relay_feat_dim)
            bs_feats = _fit(raw_bs, n_b, self.bs_feat_dim)

            # Normalize data sizes to [0,1] using fixed max (10MB)
            data_sizes = torch.tensor(
                SCt.get("data_sizes", [1e6] * n_m), dtype=torch.float
            )
            data_sizes_norm = torch.clamp(data_sizes / 10e6, 0.0, 1.0)

            # Build message features with REAL intent vectors if provided
            if intent_vectors is not None:
                intent_t = torch.tensor(intent_vectors[:n_m], dtype=torch.float)
                if intent_t.shape[0] < n_m:
                    pad = intent_t[-1:].expand(n_m - intent_t.shape[0], -1)
                    intent_t = torch.cat([intent_t, pad], dim=0)
            else:
                delay_intents = torch.tensor(
                    SCt.get("delay_intents", [1.0] * n_m), dtype=torch.float
                )
                quality_intents = torch.tensor(
                    SCt.get("quality_intents", [0.8] * n_m), dtype=torch.float
                )
                delay_norm = 1.0 - torch.clamp(delay_intents / 10.0, 0.0, 1.0)
                intent_t = torch.stack([delay_norm, quality_intents], dim=1)

            # Message features: [data_size_norm, semantic_type, delay_urgency, quality_req]
            # Deterministic semantic type: text=0.0, audio=0.5, image=1.0
            type_map = [0.0, 0.5, 1.0]  # text, audio, image
            semantic_type = torch.tensor([type_map[i % 3] for i in range(n_m)], dtype=torch.float)
            message_feats = torch.cat([
                data_sizes_norm.unsqueeze(1),
                semantic_type.unsqueeze(1),
                intent_t,
            ], dim=1)  # shape: [n_m, 4]

        else:
            csca_feats = torch.randn(n_c, self.csca_feat_dim)
            relay_feats = torch.randn(n_r, self.relay_feat_dim)
            bs_feats = torch.randn(n_b, self.bs_feat_dim)
            message_feats = torch.randn(n_m, self.message_feat_dim)

        init_feat = torch.zeros(1, self.csca_feat_dim)

        data["csca"].x = csca_feats
        data["relay"].x = relay_feats
        data["message"].x = message_feats
        data["base_station"].x = bs_feats
        data["init"].x = init_feat

        # csca -> base_station
        csca_idx = torch.arange(n_c)
        bs_idx = torch.arange(n_c) % n_b
        data["csca", "comm_conn", "base_station"].edge_index = torch.stack(
            [csca_idx, bs_idx], dim=0
        )

        # message -> csca
        msg_idx = torch.arange(n_m)
        csca_assign = torch.arange(n_m) % n_c
        data["message", "comm_req", "csca"].edge_index = torch.stack(
            [msg_idx, csca_assign], dim=0
        )

        # message -> relay
        msg_idx2 = torch.arange(n_m)
        relay_assign = torch.arange(n_m) % n_r
        data["message", "semantic_conn", "relay"].edge_index = torch.stack(
            [msg_idx2, relay_assign], dim=0
        )

        # init -> all other node types
        for node_type, count in [
            ("csca", n_c), ("relay", n_r),
            ("base_station", n_b), ("message", n_m)
        ]:
            src = torch.zeros(count, dtype=torch.long)
            dst = torch.arange(count, dtype=torch.long)
            data["init", "init_conn", node_type].edge_index = torch.stack(
                [src, dst], dim=0
            )

        # Validate all edge indices are within bounds
        for store in data.edge_stores:
            edge_index = store.edge_index
            src_type = store._key[0]
            dst_type = store._key[2]

            n_src = data[src_type].x.shape[0]
            n_dst = data[dst_type].x.shape[0]

            if edge_index.numel() > 0:
                src_idx = edge_index[0]
                dst_idx = edge_index[1]

                if src_idx.min() < 0 or src_idx.max() >= n_src:
                    print(f"WARNING: {store._key} src index out of bounds: "
                          f"min={src_idx.min()}, max={src_idx.max()}, n_src={n_src}")
                    edge_index[0] = src_idx.clamp(0, n_src - 1)

                if dst_idx.min() < 0 or dst_idx.max() >= n_dst:
                    print(f"WARNING: {store._key} dst index out of bounds: "
                          f"min={dst_idx.min()}, max={dst_idx.max()}, n_dst={n_dst}")
                    edge_index[1] = dst_idx.clamp(0, n_dst - 1)

                store.edge_index = edge_index

        return data

    def get_metadata(self):
        node_types = ["csca", "relay", "message", "base_station", "init"]
        edge_types = [
            ("csca", "comm_conn", "base_station"),
            ("message", "comm_req", "csca"),
            ("message", "semantic_conn", "relay"),
            ("init", "init_conn", "csca"),
            ("init", "init_conn", "relay"),
            ("init", "init_conn", "base_station"),
            ("init", "init_conn", "message"),
        ]
        return node_types, edge_types
