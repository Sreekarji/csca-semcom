import torch
import torch.nn as nn
from torch_geometric.nn import HANConv
from csc_graph_builder import CSCGraphBuilder

class HANNetwork(nn.Module):
    """
    Heterogeneous Graph Attention Network (HAN) as defined in
    Sun et al. 2026, Section V.B.
    Implements NLAN (node-level) + SLAN (semantic-level) attention
    via PyG HANConv.
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        n_cscas: int = 5,
        n_relays: int = 5,
        n_messages: int = 5,
        n_base_stations: int = 5,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        self.graph_builder = CSCGraphBuilder(
            n_cscas=n_cscas,
            n_relays=n_relays,
            n_messages=n_messages,
            n_base_stations=n_base_stations,
        )

        node_types, edge_types = self.graph_builder.get_metadata()
        metadata = (node_types, edge_types)

        # Input projection per node type
        in_channels = {
            "csca": 3,
            "relay": 3,
            "message": 4,
            "base_station": 3,
            "init": 3,
        }

        self.input_proj = nn.ModuleDict({
            nt: nn.Linear(in_channels[nt], hidden_channels)
            for nt in node_types
        })

        self.han_layers = nn.ModuleList([
            HANConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                metadata=metadata,
                heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        # Project all node types to hidden_channels
        x_dict = {
            nt: self.dropout(torch.relu(self.input_proj[nt](data[nt].x)))
            for nt in data.node_types
            if nt in self.input_proj
        }

        edge_index_dict = {
            et: data[et].edge_index
            for et in data.edge_types
        }

        # L layers of HAN
        for layer in self.han_layers:
            x_dict_new = layer(x_dict, edge_index_dict)
            # Residual connection (skip None outputs for isolated nodes)
            for nt in x_dict:
                if nt in x_dict_new and x_dict_new[nt] is not None:
                    x_dict[nt] = self.norm(x_dict_new[nt] + x_dict[nt])
                # else: keep x_dict[nt] unchanged (init node, etc.)

        # Graph embedding GL_t: mean pool all node embeddings
        all_embeddings = torch.cat(
            [x_dict[nt] for nt in x_dict], dim=0
        )
        graph_embedding = all_embeddings.mean(dim=0, keepdim=True)

        return graph_embedding, x_dict

    def encode_state(self, system_state: dict = None):
        data = self.graph_builder.build(system_state)
        device = next(self.parameters()).device
        data = data.to(device)
        return self.forward(data)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    han = HANNetwork(
        hidden_channels=128,
        num_heads=8,
        num_layers=2,
        n_cscas=5,
        n_relays=5,
        n_messages=5,
        n_base_stations=5,
    ).to(device)

    graph_emb, node_embs = han.encode_state()
    print(f"Graph embedding shape: {graph_emb.shape}")
    print(f"Expected: torch.Size([1, 128])")
    for nt, emb in node_embs.items():
        print(f"  {nt} embeddings: {emb.shape}")
    print("HAN test passed.")
