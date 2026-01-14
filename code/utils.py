#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PU-GRAIL: Utility functions
"""

import math
import numpy as np
import torch
import torch.nn.functional as F

from Bio.PDB import PDBParser
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

from .model import sinusoidal_positional_encoding


# Amino acid constants
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {a: i for i, a in enumerate(AA_LIST)}
THREE_TO_ONE = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}


def three_to_one(resname):
    """Convert 3-letter amino acid code to 1-letter code."""
    return THREE_TO_ONE.get(resname, 'X')


def aa_onehot(aa):
    """Create one-hot encoding for amino acid."""
    v = np.zeros(len(AA_LIST), dtype=np.float32)
    if aa in AA_TO_IDX:
        v[AA_TO_IDX[aa]] = 1.0
    return v


def get_ca_coord(residue):
    """Get Cα coordinate from residue."""
    if "CA" in residue:
        return residue["CA"].get_coord()
    coords = []
    for name in ["N", "C", "O", "CB"]:
        if name in residue:
            coords.append(residue[name].get_coord())
    if not coords:
        return None
    return np.stack(coords, 0).mean(0)


def gaussian_dist_weight(pos, edge_index, sigma=5.0):
    """Compute Gaussian distance-based edge weights."""
    src, dst = edge_index
    d2 = (pos[src] - pos[dst]).pow(2).sum(dim=1)
    w = torch.exp(-d2 / (2.0 * (sigma**2)))
    return w.clamp(min=1e-6)


def parse_pdb_as_graph(
    pdb_path,
    esm_residue=None,
    dist_thr=8.0,
    use_positional_encoding=True,
    pos_encoding_dim=32
):
    """
    Parse PDB file and create a graph representation.
    
    Args:
        pdb_path: Path to PDB file
        esm_residue: ESM embedding array (L, D)
        dist_thr: Distance threshold for structure edges (Angstrom)
        use_positional_encoding: Whether to add positional encoding
        pos_encoding_dim: Dimension of positional encoding
    
    Returns:
        torch_geometric.data.Data object
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", pdb_path)

    residues = []
    for model in structure:
        for chain in model:
            for res in chain:
                if res.id[0] != " ":
                    continue
                ca = get_ca_coord(res)
                if ca is None:
                    continue
                residues.append(res)
    
    if len(residues) == 0:
        return None

    pos = []
    aa_feats = []
    for res in residues:
        pos.append(get_ca_coord(res))
        aa = three_to_one(res.get_resname())
        aa_feats.append(aa_onehot(aa))
    pos = np.asarray(pos, dtype=np.float32)
    aa_feats = np.asarray(aa_feats, dtype=np.float32)

    # Concat ESM residue embedding if provided
    if esm_residue is not None and esm_residue.ndim == 2:
        L, D = esm_residue.shape
        N = pos.shape[0]
        # Remove CLS and EOS tokens from ESM embedding
        if L >= 2:
            esm_seq = esm_residue[1:-1]
            L_actual = esm_seq.shape[0]
        else:
            esm_seq = esm_residue
            L_actual = L
        M = min(N, L_actual)
        pad = np.zeros((N, D), dtype=np.float32)
        pad[:M] = esm_seq[:M].astype(np.float32)
        x = np.concatenate([aa_feats, pad], axis=1)
    else:
        x = aa_feats

    coords = pos
    N = coords.shape[0]
    src, dst = [], []

    # Build structure edges based on distance threshold
    thr2 = dist_thr * dist_thr
    for i in range(N):
        di = coords[i] - coords
        d2 = np.einsum("ij,ij->i", di, di)
        idx = np.where((d2 <= thr2) & (d2 > 0))[0]
        for j in idx:
            src.append(i)
            dst.append(j)

    # Edge type tracking (0=structure, 1=sequence)
    edge_type = [0] * len(src)
    n_structure_edges = len(src)

    # Sequential edges
    for i in range(N-1):
        src.append(i)
        dst.append(i+1)
        src.append(i+1)
        dst.append(i)
    
    edge_type.extend([1] * (len(src) - n_structure_edges))

    edge_index = np.vstack([np.array(src), np.array(dst)]).astype(np.int64)
    edge_type = np.array(edge_type, dtype=np.int64)

    # Add positional encoding
    if use_positional_encoding:
        pe = sinusoidal_positional_encoding(N, pos_encoding_dim)
        x = np.concatenate([x, pe.cpu().numpy()], axis=1)

    return Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        pos=torch.tensor(pos, dtype=torch.float32),
        edge_type=torch.tensor(edge_type, dtype=torch.long),
    )


# =====================
# PU Learning Losses
# =====================

def puauc_pairwise_loss(logits, y, k=1.0, margin=0.0, max_pairs=200000):
    """
    PU-AUC pairwise ranking loss.
    
    Args:
        logits: Model predictions (B,)
        y: Labels (1=positive, 0=unlabeled)
        k: Weight multiplier
        margin: Ranking margin
        max_pairs: Maximum number of pairs for memory efficiency
    """
    y = y.view(-1).long()
    idx_p = (y == 1)
    idx_u = (y == 0)
    
    if not idx_p.any() or not idx_u.any():
        return (logits * 0).sum()
    
    pos_scores = logits[idx_p]
    neg_scores = logits[idx_u]
    
    P, N = pos_scores.numel(), neg_scores.numel()
    total_pairs = P * N
    
    if total_pairs <= max_pairs:
        diff = (pos_scores[:, None] - neg_scores[None, :]) - margin
        loss_mat = F.softplus(-diff)
        loss = loss_mat.mean()
    else:
        device = pos_scores.device
        idx = torch.randint(0, total_pairs, (max_pairs,), device=device)
        pos_idx = idx // N
        neg_idx = idx % N
        diff = (pos_scores[pos_idx] - neg_scores[neg_idx]) - margin
        loss = F.softplus(-diff).mean()
    
    return k * loss


def nnpu_loss_from_logits(logits, y, pi, loss_fn=F.softplus):
    """
    Non-negative PU loss.
    
    Args:
        logits: Model predictions
        y: Labels (1=positive, 0=unlabeled)
        pi: Class prior (proportion of positives)
        loss_fn: Base loss function
    """
    y = y.view(-1)
    idx_p = (y == 1)
    idx_u = (y == 0)
    
    if not idx_p.any() and not idx_u.any():
        return (logits * 0).sum()

    if idx_p.any():
        sp = logits[idx_p]
        R_p_pos = loss_fn(-sp).mean()
        R_p_neg = loss_fn(sp).mean()
    else:
        R_p_pos = logits.new_zeros(())
        R_p_neg = logits.new_zeros(())

    if idx_u.any():
        su = logits[idx_u]
        R_u_neg = loss_fn(su).mean()
    else:
        R_u_neg = logits.new_zeros(())

    R_n_tilde = R_u_neg - pi * R_p_neg
    R_n = torch.clamp(R_n_tilde, min=0.0)
    return pi * R_p_pos + R_n


def estimate_class_prior(y_tensor):
    """Estimate class prior from labels."""
    y = y_tensor.float().clamp(0, 1)
    n = float(len(y))
    p = float(y.sum())
    eps = 1e-8
    return max(min(p / max(n, eps), 0.99), 1e-3)


# =====================
# Auxiliary Losses
# =====================

def graph_contrastive_loss(g, y_graph, gamma=0.5):
    """
    Graph-level contrastive loss.
    
    Args:
        g: Graph embeddings (B, D)
        y_graph: Graph labels (B,)
        gamma: Weight for negative pairs
    """
    if g is None or g.size(0) < 2:
        return g.new_zeros(()) if g is not None else torch.tensor(0.0)

    g_norm = F.normalize(g, p=2, dim=1)
    sim = g_norm @ g_norm.t()

    y = y_graph.view(-1).long()
    mask_self = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)

    n_pos = (y == 1).sum().item()
    n_neg = (y == 0).sum().item()

    mask_pos = (y.unsqueeze(1) == 1) & (y.unsqueeze(0) == 1)
    mask_pos = mask_pos & (~mask_self)

    mask_neg = ((y.unsqueeze(1) == 1) & (y.unsqueeze(0) == 0)) | \
               ((y.unsqueeze(1) == 0) & (y.unsqueeze(0) == 1))

    pos_pairs = sim[mask_pos] if mask_pos.sum() > 0 else g.new_zeros(())
    neg_pairs = sim[mask_neg] if mask_neg.sum() > 0 else g.new_zeros(())
    
    pos_mean = pos_pairs.mean() if pos_pairs.numel() > 0 else g.new_zeros(())
    neg_mean = torch.abs(neg_pairs).mean() if neg_pairs.numel() > 0 else g.new_zeros(())

    if n_pos >= 2 and n_neg >= 1:
        loss = (1.0 - pos_mean) + gamma * neg_mean
    elif n_pos >= 2:
        loss = (1.0 - pos_mean)
    elif n_neg >= 1:
        loss = gamma * neg_mean
    else:
        loss = g.new_zeros(())

    return loss


def node_separation_loss(h, w, batch, y_graph, pos_frac=0.1, neg_frac=0.1, min_pos=3, min_neg=3):
    """
    Node-level separation loss for positive graphs.
    
    Args:
        h: Node embeddings
        w: Attention weights
        batch: Batch assignment
        y_graph: Graph labels
        pos_frac: Fraction of top nodes as pseudo-positives
        neg_frac: Fraction of bottom nodes as pseudo-negatives
        min_pos: Minimum number of pseudo-positives
        min_neg: Minimum number of pseudo-negatives
    """
    if w is None or h is None:
        return h.new_zeros(())

    z = F.normalize(h, p=2, dim=1)
    losses = []

    B = int(y_graph.numel())
    for g in range(B):
        yg = float(y_graph[g].item())
        if yg < 0.5:
            continue

        idx = (batch == g).nonzero(as_tuple=True)[0]
        n = idx.numel()
        if n < (min_pos + min_neg + 2):
            continue

        wg = w[idx]
        wg0 = (wg - wg.mean()) / (wg.std() + 1e-6)
        w_score = torch.sigmoid(wg0)

        k_pos = max(min_pos, int(math.ceil(pos_frac * n)))
        k_neg = max(min_neg, int(math.ceil(neg_frac * n)))

        order = torch.argsort(w_score)
        neg_local = order[:k_neg]
        pos_local = order[-k_pos:]
        
        pos_idx = idx[pos_local]
        neg_idx = idx[neg_local]

        if pos_idx.numel() < min_pos or neg_idx.numel() < min_neg:
            continue

        Zp = z[pos_idx]
        Zn = z[neg_idx]

        sim_pp = Zp @ Zp.t()
        mask_self = torch.eye(Zp.size(0), device=Zp.device, dtype=torch.bool)
        sim_pp_masked = sim_pp.masked_fill(mask_self, 0.0)
        n_pos_pairs = Zp.size(0) * (Zp.size(0) - 1)
        pos_pairs_mean = sim_pp_masked.sum() / max(n_pos_pairs, 1)

        sim_pn = Zp @ Zn.t()
        neg_pairs_mean = torch.abs(sim_pn).mean()
        
        loss_g = (1.0 - pos_pairs_mean) + 0.5 * neg_pairs_mean
        
        # Orthogonal loss
        orthogonal_loss = (sim_pn ** 2).mean()
        loss_g = loss_g + 0.2 * orthogonal_loss
        
        losses.append(loss_g)

    if len(losses) == 0:
        return h.new_zeros(())

    return torch.stack(losses).mean()

