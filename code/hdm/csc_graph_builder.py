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

    def build(self, system_state: dict = None) -> HeteroData:
        data = HeteroData()
        n_c = self.n_cscas
        n_r = self.n_relays
        n_m = self.n_messages
        n_b = self.n_bs

        if system_state is not None:
            Rt = system_state.get("Rt", {})
            SCt = system_state.get("SCt", {})
            # Read features but pad/trim to match this builder's counts
            raw_csca = Rt.get("csca_features", torch.randn(n_c, self.csca_feat_dim).tolist())
            raw_relay = Rt.get("relay_features", torch.randn(n_r, self.relay_feat_dim).tolist())
            raw_msg = SCt.get("message_features", torch.randn(n_m, self.message_feat_dim).tolist())
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
            message_feats = _fit(raw_msg, n_m, self.message_feat_dim)
            bs_feats = _fit(raw_bs, n_b, self.bs_feat_dim)
        else:
            csca_feats = torch.randn(n_c, self.csca_feat_dim)
            relay_feats = torch.randn(n_r, self.relay_feat_dim)
            message_feats = torch.randn(n_m, self.message_feat_dim)
            bs_feats = torch.randn(n_b, self.bs_feat_dim)

        init_feat = torch.zeros(1, self.csca_feat_dim)

        data["csca"].x = csca_feats
        data["relay"].x = relay_feats
        data["message"].x = message_feats
        data["base_station"].x = bs_feats
        data["init"].x = init_feat

        # csca -> base_station (each CSCA connected to one BS)
        csca_idx = torch.arange(n_c)
        bs_idx = torch.arange(n_c) % n_b
        data["csca", "comm_conn", "base_station"].edge_index = torch.stack(
            [csca_idx, bs_idx], dim=0
        )

        # message -> csca (each message assigned to one CSCA)
        msg_idx = torch.arange(n_m)
        csca_assign = torch.arange(n_m) % n_c
        data["message", "comm_req", "csca"].edge_index = torch.stack(
            [msg_idx, csca_assign], dim=0
        )

        # message -> relay (each message can use any relay with matching knowledge)
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


if __name__ == "__main__":
    builder = CSCGraphBuilder()
    graph = builder.build()
    print("CSC Graph built successfully")
    print(f"Node types: {graph.node_types}")
    print(f"Edge types: {graph.edge_types}")
    for nt in graph.node_types:
        print(f"  {nt}: {graph[nt].x.shape}")
