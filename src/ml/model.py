"""B4 - Spatio-Temporal Graph Neural Network (ST-GNN) in plain PyTorch.

How it works (in simple words):
1. SPATIAL STEP (Graph Convolution):
   Every 15 minutes, each street/junction (node) asks its immediate neighbours:
   "How much rain and water do you have?"
   We mix each node's features with its neighbours using the city's pipe/street map (adjacency matrix A_hat).
2. TEMPORAL STEP (GRU):
   A recurrent neural network reads the 8-step timeline (the last 2 hours of history)
   for each node to learn whether water is building up or receding.
3. PREDICTION HEAD:
   Outputs the forecasted street flood depth for the next 12 steps (3 hours).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    """Simple graph convolution: Z = ReLU(A_hat @ X @ W + b)."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        """
        x   : [B, N, in_features]
        adj : [N, N] normalized adjacency matrix
        """
        # torch.matmul with 2D adj and 3D x:
        # [N, N] @ [B, N, F] -> we can do (adj @ x) by broadcasting
        ax = torch.matmul(adj, x)  # [B, N, in_features]
        return F.relu(self.linear(ax))


class STGNN(nn.Module):
    """Spatio-Temporal Graph Neural Network.

    Parameters
    ----------
    num_nodes : int
        Number of nodes N in the urban graph.
    in_channels : int
        Number of input channels per node (4: rain, fill, river, pump_on).
    hidden_dim : int
        Dimension of node embeddings (e.g. 48).
    history_steps : int
        Number of input time steps (8).
    horizon_steps : int
        Number of output prediction steps (12).
    """

    def __init__(
        self,
        num_nodes: int,
        in_channels: int = 4,
        hidden_dim: int = 48,
        history_steps: int = 8,
        horizon_steps: int = 12,
        adj: torch.Tensor = None,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.history_steps = history_steps
        self.horizon_steps = horizon_steps
        self.hidden_dim = hidden_dim

        # Register normalized adjacency as a buffer so it moves with .to(device)
        if adj is not None:
            if not isinstance(adj, torch.Tensor):
                adj = torch.from_numpy(adj).float()
            self.register_buffer("adj", adj)
        else:
            self.register_buffer("adj", torch.eye(num_nodes))

        # Spatial layers (GCN)
        self.gcn1 = GraphConv(in_channels, hidden_dim)
        self.gcn2 = GraphConv(hidden_dim, hidden_dim)

        # Temporal layer (GRU over the 8 time steps)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=1,
        )

        # Output head (predicts depth for next 12 steps)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, horizon_steps),
            nn.ReLU(),  # water depth in metres cannot be negative
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape [B, 8, N, 4] (batch, history_steps, nodes, features)

        Returns
        -------
        torch.Tensor
            Shape [B, 12, N] predicted flood depth in metres
        """
        B, H, N, C = x.shape

        # Step 1: Spatial mixing at each time step
        # Process each step t in [0..H-1]
        spatial_steps = []
        for t in range(H):
            xt = x[:, t, :, :]           # [B, N, C]
            h1 = self.gcn1(xt, self.adj) # [B, N, hidden]
            h2 = self.gcn2(h1, self.adj) # [B, N, hidden]
            spatial_steps.append(h2)

        # Stack over time -> [B, H, N, hidden]
        spatial_seq = torch.stack(spatial_steps, dim=1)

        # Step 2: Temporal sequence reading with GRU
        # Rearrange to treat each node as an independent sequence of length H:
        # [B, H, N, hidden] -> [B, N, H, hidden] -> [B * N, H, hidden]
        node_seq = spatial_seq.permute(0, 2, 1, 3).reshape(B * N, H, self.hidden_dim)
        gru_out, _ = self.gru(node_seq)  # [B * N, H, hidden]

        # Use the summary representation at the last history step
        last_hidden = gru_out[:, -1, :]  # [B * N, hidden]

        # Step 3: Predict future 12 steps
        out = self.head(last_hidden)     # [B * N, 12]

        # Reshape back to [B, N, 12] -> permute to [B, 12, N]
        out = out.view(B, N, self.horizon_steps).permute(0, 2, 1)
        return out
