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
        hidden_channels: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
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

        # Graph embedding GL_t: mean pool all node embeddings
        all_embeddings = torch.cat(
            [x_dict[nt] for nt in x_dict], dim=0
        )
        graph_embedding = all_embeddings.mean(dim=0, keepdim=True)

        return graph_embedding, x_dict

    def encode_state(self, system_state: dict = None,
                     intent_vectors: list = None):
        """
        Encode system state into graph embedding + per-message embeddings.
        intent_vectors: list of [delay_urgency, quality_req] per task.
        Returns: graph_emb [1, 128], node_embs dict, message_embs [n_tasks, 128]
        """
        data = self.graph_builder.build(
            system_state, intent_vectors=intent_vectors
        )
        device = next(self.parameters()).device
        data = data.to(device)
        graph_emb, node_embs = self.forward(data)

        # Return per-message embeddings for task-specific policy
        if "message" in node_embs:
            message_embs = node_embs["message"]  # [n_messages, 128]
        else:
            message_embs = graph_emb.expand(
                self.graph_builder.n_messages, -1
            )

        return graph_emb, node_embs, message_embs
