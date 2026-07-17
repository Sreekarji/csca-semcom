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
        
        # Learnable weights for multi-layer aggregation (Eq. 26)
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)

    def forward(self, data):
        # Project all node types to hidden_channels
        x_dict_current = {
            nt: self.dropout(torch.relu(self.input_proj[nt](data[nt].x)))
            for nt in data.node_types
            if nt in self.input_proj
        }

        edge_index_dict = {
            et: data[et].edge_index
            for et in data.edge_types
        }

        # Collect graph embeddings from each HAN layer
        layer_embeddings = []
        
        # L layers of HAN with residual connections
        for layer in self.han_layers:
            x_dict_new = layer(x_dict_current, edge_index_dict)
            # Residual connection (skip None outputs for isolated nodes)
            for nt in x_dict_new:
                if nt in x_dict_current and x_dict_new[nt] is not None:
                    x_dict_current[nt] = self.norm(x_dict_new[nt] + x_dict_current[nt])
                elif x_dict_new[nt] is not None:
                    x_dict_current[nt] = self.norm(x_dict_new[nt])
            
            # Collect this layer's graph embedding (Eq. 26)
            # Focus on message + csca nodes (not all 21 nodes)
            focused_embs = []
            if "message" in x_dict_current and x_dict_current["message"] is not None:
                focused_embs.append(x_dict_current["message"])
            if "csca" in x_dict_current and x_dict_current["csca"] is not None:
                focused_embs.append(x_dict_current["csca"])
            if focused_embs:
                layer_focused = torch.cat(focused_embs, dim=0)
                layer_graph_emb = layer_focused.mean(dim=0, keepdim=True)
            else:
                layer_all_embs = torch.cat([x_dict_current[nt] for nt in x_dict_current], dim=0)
                layer_graph_emb = layer_all_embs.mean(dim=0, keepdim=True)
            layer_embeddings.append(layer_graph_emb)

        # Weighted sum across layers (Eq. 26: GL_t = sum w_l * H_l)
        layer_weights_norm = torch.softmax(self.layer_weights, dim=0)
        graph_embedding = sum(w * e for w, e in zip(layer_weights_norm, layer_embeddings))

        return graph_embedding, x_dict_current

    def encode_state(self, system_state: dict = None,
                     intent_vectors: list = None):
        """
        Encode system state into graph embedding + per-message embeddings.
        intent_vectors: list of [delay_urgency, quality_req] per task.
        Returns: graph_emb [1, 256], node_embs dict, message_embs [n_tasks, 256]
        """
        # Detect actual number of tasks from system_state
        if system_state is not None:
            n_tasks_actual = len(system_state.get("SCt", {}).get("data_sizes",
                             [None] * self.graph_builder.n_messages))
        else:
            n_tasks_actual = self.graph_builder.n_messages

        # Rebuild graph builder if task count changed
        if n_tasks_actual != self.graph_builder.n_messages:
            from csc_graph_builder import CSCGraphBuilder
            temp_builder = CSCGraphBuilder(
                n_cscas=self.graph_builder.n_cscas,
                n_relays=self.graph_builder.n_relays,
                n_messages=n_tasks_actual,
                n_base_stations=self.graph_builder.n_bs,
            )
            data = temp_builder.build(system_state, intent_vectors=intent_vectors)
        else:
            data = self.graph_builder.build(system_state, intent_vectors=intent_vectors)

        device = next(self.parameters()).device
        data = data.to(device)
        graph_emb, node_embs = self.forward(data)

        if "message" in node_embs:
            message_embs = node_embs["message"]
        else:
            message_embs = graph_emb.expand(n_tasks_actual, -1)

        return graph_emb, node_embs, message_embs
