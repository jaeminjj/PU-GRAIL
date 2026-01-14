#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PU-GRAIL: Graph Neural Network Model for Protective Antigen Prediction
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import softmax as pyg_softmax


def sinusoidal_positional_encoding(n_positions: int, dim: int, device=None) -> torch.Tensor:
    """
    Sinusoidal positional encoding (Transformer-style).
    Returns: (n_positions, dim) tensor
    """
    if device is None:
        device = torch.device("cpu")
    
    position = torch.arange(n_positions, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / dim))
    
    pe = torch.zeros(n_positions, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    if dim > 1:
        pe[:, 1::2] = torch.cos(position * div_term[:dim//2]) if dim % 2 == 0 else torch.cos(position * div_term)
    
    return pe


class AttnReadout(nn.Module):
    """Attention-based graph readout with top-k pooling."""
    
    def __init__(self, hid, topk_frac=0.05, topk_min=5, topk_max=64):
        super().__init__()
        self.topk_frac = topk_frac
        self.topk_min = topk_min
        self.topk_max = topk_max
        
        self.attn_net = nn.Sequential(
            nn.Linear(hid, hid//2), nn.ReLU(),
            nn.Linear(hid//2, 1)
        )
        self.fuse = nn.Sequential(nn.Linear(4*hid, hid), nn.ReLU())

    def forward(self, x, batch, return_w=False):
        attn_scores = self.attn_net(x).squeeze(-1)
        w = pyg_softmax(attn_scores, batch)

        xw = x * w.unsqueeze(-1)
        x_attn = global_add_pool(xw, batch)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)

        topk_list = []
        for g_id in batch.unique():
            mask = (batch == g_id)
            xg = x[mask]
            wg = w[mask]
            n = int(wg.numel())
            if n == 0:
                topk_list.append(x.new_zeros((1, x.size(1))).squeeze(0))
                continue
            
            k = max(self.topk_min, int(math.ceil(self.topk_frac * n)))
            if self.topk_max > 0:
                k = min(k, self.topk_max)
            k = min(k, n)
            
            sel = torch.topk(wg, k=k, largest=True).indices
            topk_list.append(xg[sel].mean(dim=0))
        x_topk = torch.stack(topk_list, dim=0)

        g = self.fuse(torch.cat([x_mean, x_attn, x_max, x_topk], dim=1))

        if return_w:
            return g, w
        return g


class ESMProjector(nn.Module):
    """2-layer MLP projector for ESM embeddings."""
    
    def __init__(self, esm_dim: int, proj_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        hid = max(proj_dim, 128)
        self.net = nn.Sequential(
            nn.LayerNorm(esm_dim),
            nn.Linear(esm_dim, hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid, proj_dim),
        )
        self.out_dim = proj_dim

    def forward(self, esm):
        return self.net(esm)


class PUGRAIL(nn.Module):
    """
    PU-GRAIL: Graph Neural Network for Protective Antigen Prediction
    
    Args:
        in_dim: Input feature dimension
        hid: Hidden dimension (default: 128)
        out_dim: Output dimension (default: 1)
        gnn_layers: Number of GNN layers (default: 3)
        use_edge_separation: Whether to use edge type separation (default: True)
        use_positional_encoding: Whether to use positional encoding (default: True)
        pos_encoding_dim: Positional encoding dimension (default: 32)
        esm_proj_dim: ESM projection dimension (default: 256)
        aa_dim: Amino acid one-hot dimension (default: 20)
    """
    
    def __init__(
        self,
        in_dim,
        hid=128,
        out_dim=1,
        gnn_layers=3,
        use_edge_separation=True,
        use_positional_encoding=True,
        pos_encoding_dim=32,
        esm_proj_dim=256,
        aa_dim=20,
        esm_proj_dropout=0.1,
    ):
        super().__init__()
        self.hid = hid
        self.epoch = 0
        self.use_edge_separation = use_edge_separation
        self.use_positional_encoding = use_positional_encoding
        self.pos_encoding_dim = pos_encoding_dim
        self.aa_dim = aa_dim

        self.raw_in_dim = int(in_dim)
        
        # Calculate ESM dimension
        if use_positional_encoding:
            self.has_esm = (self.raw_in_dim > aa_dim + pos_encoding_dim)
            if self.has_esm:
                self.esm_dim = self.raw_in_dim - aa_dim - pos_encoding_dim
            else:
                self.esm_dim = 0
        else:
            self.has_esm = (self.raw_in_dim > aa_dim)
            if self.has_esm:
                self.esm_dim = self.raw_in_dim - aa_dim
            else:
                self.esm_dim = 0

        # ESM Projector
        if self.has_esm:
            self.esm_proj = ESMProjector(self.esm_dim, proj_dim=esm_proj_dim, dropout=esm_proj_dropout)
            model_in_dim = aa_dim + self.esm_proj.out_dim
        else:
            self.esm_proj = None
            model_in_dim = aa_dim
        
        if use_positional_encoding:
            model_in_dim += pos_encoding_dim

        # Edge Type Separation: separate GNNs for seq and structure
        if use_edge_separation:
            self.convs_seq = nn.ModuleList([GCNConv(model_in_dim, hid)])
            self.norms_seq = nn.ModuleList([nn.LayerNorm(hid)])
            for _ in range(gnn_layers-1):
                self.convs_seq.append(GCNConv(hid, hid))
                self.norms_seq.append(nn.LayerNorm(hid))
            
            self.convs_str = nn.ModuleList([GCNConv(model_in_dim, hid)])
            self.norms_str = nn.ModuleList([nn.LayerNorm(hid)])
            for _ in range(gnn_layers-1):
                self.convs_str.append(GCNConv(hid, hid))
                self.norms_str.append(nn.LayerNorm(hid))
            
            self.edge_alpha = nn.Parameter(torch.tensor(0.5))
            self.convs = None
            self.norms = None
        else:
            self.convs = nn.ModuleList([GCNConv(model_in_dim, hid)])
            self.norms = nn.ModuleList([nn.LayerNorm(hid)])
            for _ in range(gnn_layers-1):
                self.convs.append(GCNConv(hid, hid))
                self.norms.append(nn.LayerNorm(hid))
            
            self.convs_seq = None
            self.convs_str = None
            self.edge_alpha = None

        self.readout_attn = AttnReadout(hid)

        self.head = nn.Sequential(
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hid, out_dim),
        )

    def _apply_projector(self, x_raw: torch.Tensor) -> torch.Tensor:
        """Apply ESM projector and handle positional encoding."""
        if self.use_positional_encoding:
            pos_enc = x_raw[:, -self.pos_encoding_dim:]
            x_feat = x_raw[:, :-self.pos_encoding_dim]
        else:
            pos_enc = None
            x_feat = x_raw
        
        if (not self.has_esm) or (self.esm_proj is None):
            result = x_feat
        else:
            aa = x_feat[:, :self.aa_dim]
            esm = x_feat[:, self.aa_dim:]
            esm_p = self.esm_proj(esm)
            result = torch.cat([aa, esm_p], dim=1)
        
        if self.use_positional_encoding and pos_enc is not None:
            result = torch.cat([result, pos_enc], dim=1)
        
        return result

    def forward(self, data, return_node_info=False):
        x_raw, base_ei, pos, batch = data.x, data.edge_index, data.pos, data.batch
        edge_type = getattr(data, 'edge_type', None)

        h = self._apply_projector(x_raw)
        
        # Compute edge weights
        from .utils import gaussian_dist_weight
        base_w = gaussian_dist_weight(pos, base_ei, sigma=5.0)

        if self.use_edge_separation and edge_type is not None and self.convs_seq is not None:
            seq_mask = (edge_type == 1)
            str_mask = (edge_type == 0)
            
            ei_seq = base_ei[:, seq_mask] if seq_mask.any() else torch.empty(2, 0, dtype=torch.long, device=base_ei.device)
            ew_seq = base_w[seq_mask] if seq_mask.any() else torch.empty(0, device=base_w.device)
            
            ei_str = base_ei[:, str_mask] if str_mask.any() else torch.empty(2, 0, dtype=torch.long, device=base_ei.device)
            ew_str = base_w[str_mask] if str_mask.any() else torch.empty(0, device=base_w.device)
            
            h_seq = h.clone()
            h_str = h.clone()
            
            for conv, ln in zip(self.convs_seq, self.norms_seq):
                if ei_seq.numel() > 0:
                    h_seq = conv(h_seq, ei_seq, ew_seq)
                    h_seq = ln(F.relu(h_seq))
            
            for conv, ln in zip(self.convs_str, self.norms_str):
                if ei_str.numel() > 0:
                    h_str = conv(h_str, ei_str, ew_str)
                    h_str = ln(F.relu(h_str))
            
            alpha = torch.sigmoid(self.edge_alpha)
            h = alpha * h_seq + (1 - alpha) * h_str
        else:
            for conv, ln in zip(self.convs, self.norms):
                h = conv(h, base_ei, base_w)
                h = ln(F.relu(h))

        g, w = self.readout_attn(h, batch, return_w=True)
        logit = self.head(g).view(-1)

        if return_node_info:
            return logit, h, w, batch, g
        return logit

