import torch
import torch.nn as nn

class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index):
        agg = torch.zeros_like(x)
        src, dst = edge_index
        agg.index_add_(0, dst, self.lin(x)[src])
        degree = torch.zeros(x.size(0), device=x.device)
        degree.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
        out = self.norm(agg / degree.clamp_min(1.0).unsqueeze(-1))
        return torch.relu(out)

class StreamPrior(nn.Module):
    def __init__(self, in_dim=12, hidden=64, layers=3):
        super().__init__()
        self.input = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList([GraphConv(hidden, hidden) for _ in range(layers)])
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, edge_index):
        h = torch.relu(self.input(x))
        for layer in self.layers:
            h = layer(h, edge_index) + h
        return torch.sigmoid(self.head(h)).squeeze(-1)
