#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PU-GRAIL: Training script for Protective Antigen Prediction

Usage:
    python -m pugrail.train \
        --csv_path /path/to/data.csv \
        --pdb_dir /path/to/pdb/ \
        --esm_dir /path/to/esm/ \
        --output_dir /path/to/output/
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedShuffleSplit

from .model import PUGRAIL
from .dataset import ProteinGraphDataset
from .utils import (
    puauc_pairwise_loss,
    nnpu_loss_from_logits,
    estimate_class_prior,
    graph_contrastive_loss,
    node_separation_loss,
)


def get_args():
    parser = argparse.ArgumentParser(
        description="PU-GRAIL: Graph Neural Network for Protective Antigen Prediction"
    )
    
    # Data paths
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to CSV file with columns: id_short, label, fold, split")
    parser.add_argument("--pdb_dir", type=str, required=True,
                        help="Directory containing PDB structure files")
    parser.add_argument("--esm_dir", type=str, required=True,
                        help="Directory containing ESM embedding files (.npy)")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Output directory for results and models")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                        help="Cache directory for processed datasets")
    
    # Model architecture
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="Hidden dimension")
    parser.add_argument("--gnn_layers", type=int, default=3,
                        help="Number of GNN layers")
    parser.add_argument("--esm_proj_dim", type=int, default=256,
                        help="ESM projection dimension")
    parser.add_argument("--pos_encoding_dim", type=int, default=32,
                        help="Positional encoding dimension")
    parser.add_argument("--use_edge_separation", action="store_true", default=True,
                        help="Use edge type separation")
    parser.add_argument("--no_edge_separation", action="store_false", dest="use_edge_separation")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximum number of epochs")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    
    # Loss weights
    parser.add_argument("--task_loss_weight", type=float, default=1.0,
                        help="Weight for task loss")
    parser.add_argument("--node_loss_weight", type=float, default=0.3,
                        help="Weight for node separation loss")
    parser.add_argument("--contrastive_loss_weight", type=float, default=0.1,
                        help="Weight for graph contrastive loss")
    
    # PU Learning
    parser.add_argument("--pu_loss", type=str, default="puauc",
                        choices=["puauc", "nnpu", "bce"],
                        help="PU loss type")
    parser.add_argument("--pu_auc_margin", type=float, default=0.0,
                        help="Margin for PU-AUC loss")
    parser.add_argument("--pu_auc_max_pairs", type=int, default=200000,
                        help="Maximum pairs for PU-AUC loss")
    
    # Evaluation
    parser.add_argument("--early_metric", type=str, default="aupr",
                        choices=["aupr", "auroc"],
                        help="Metric for early stopping")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="Validation ratio for train/val split")
    
    # Misc
    parser.add_argument("--tag", type=str, default="pugrail",
                        help="Experiment tag for output files")
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device, threshold=0.5):
    """Evaluate model on a data loader."""
    model.eval()
    ys, ps = [], []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            prob = torch.sigmoid(model(data))
            ys.append(data.y.cpu().numpy().ravel())
            ps.append(prob.cpu().numpy().ravel())

    if not ys:
        return dict(AUC=np.nan, AUPRC=np.nan, F1=np.nan, MCC=np.nan)

    y = np.concatenate(ys)
    p = np.concatenate(ps)

    try:
        auc = roc_auc_score(y, p)
    except:
        auc = float("nan")
    try:
        ap = average_precision_score(y, p)
    except:
        ap = float("nan")

    pred = (p >= threshold).astype(int)
    try:
        f1 = f1_score(y, pred)
    except:
        f1 = float("nan")
    try:
        mcc = matthews_corrcoef(y, pred) if len(np.unique(pred)) > 1 else 0.0
    except:
        mcc = float("nan")

    return dict(AUC=auc, AUPRC=ap, F1=f1, MCC=mcc)


def find_best_threshold(y_true, p, metric="f1"):
    """Find optimal classification threshold."""
    if len(y_true) == 0:
        return 0.5, 0.0
    
    thresholds = np.linspace(0.05, 0.95, 19)
    best_t, best_score = 0.5, -1.0
    
    for t in thresholds:
        pred = (p >= t).astype(int)
        if metric == "mcc" and len(np.unique(pred)) < 2:
            continue
        try:
            score = f1_score(y_true, pred) if metric == "f1" else matthews_corrcoef(y_true, pred)
        except:
            continue
        if score > best_score:
            best_score = score
            best_t = t
    
    if best_score < 0:
        return 0.5, 0.0
    return best_t, best_score


def stratified_split(df, val_ratio=0.2, seed=42):
    """Stratified train/validation split."""
    uniq = df.drop_duplicates(subset=["id_short"]).loc[:, ["id_short", "label"]].reset_index(drop=True)
    X = uniq["id_short"].values
    y = uniq["label"].astype(int).values

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    try:
        train_idx, valid_idx = next(sss.split(X, y))
        train_ids = set(X[train_idx])
        valid_ids = set(X[valid_idx])
    except ValueError:
        valid_ids = []
        for cls in np.unique(y):
            cls_ids = X[y == cls]
            take = min(len(cls_ids), max(1, int(round(len(cls_ids) * val_ratio))))
            valid_ids.extend(list(cls_ids[:take]))
        valid_ids = set(valid_ids)
        train_ids = set([i for i in X if i not in valid_ids])

    df_tr = df[df["id_short"].isin(train_ids)].reset_index(drop=True)
    df_va = df[df["id_short"].isin(valid_ids)].reset_index(drop=True)

    return df_tr, df_va


def train_fold(args, df_train, df_valid, df_test, fold_tag, device):
    """Train model for one fold."""
    
    # Create datasets
    ds_train = ProteinGraphDataset(
        df_train, args.pdb_dir, args.esm_dir,
        cache_dir=args.cache_dir,
        use_positional_encoding=True,
        pos_encoding_dim=args.pos_encoding_dim,
    )
    ds_valid = ProteinGraphDataset(
        df_valid, args.pdb_dir, args.esm_dir,
        cache_dir=args.cache_dir,
        use_positional_encoding=True,
        pos_encoding_dim=args.pos_encoding_dim,
    )
    ds_test = ProteinGraphDataset(
        df_test, args.pdb_dir, args.esm_dir,
        cache_dir=args.cache_dir,
        use_positional_encoding=True,
        pos_encoding_dim=args.pos_encoding_dim,
    )
    
    # Create data loaders
    loader_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    loader_valid = DataLoader(ds_valid, batch_size=args.batch_size, shuffle=False)
    loader_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    in_dim = ds_train[0].x.shape[1]
    model = PUGRAIL(
        in_dim=in_dim,
        hid=args.hidden_dim,
        gnn_layers=args.gnn_layers,
        use_edge_separation=args.use_edge_separation,
        pos_encoding_dim=args.pos_encoding_dim,
        esm_proj_dim=args.esm_proj_dim,
    ).to(device)
    
    # Estimate class prior for nnPU
    if args.pu_loss == "nnpu":
        ys_all = torch.stack([d.y for d in ds_train]).view(-1).float()
        pi = estimate_class_prior(ys_all)
        print(f"[{fold_tag}] nnPU prior pi={pi:.4f}")
    else:
        pi = None
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Training loop
    best = {"score": -1e18, "state": None, "epoch": -1}
    no_improve = 0
    warmup_epochs = 2
    
    for ep in range(1, args.epochs + 1):
        model.epoch = ep
        model.train()
        
        task_sum, node_sum, contrastive_sum, n_graphs = 0.0, 0.0, 0.0, 0
        
        for data in loader_train:
            data = data.to(device)
            logits, h, w, batch, g = model(data, return_node_info=True)
            y = data.y.view(-1)
            
            # Task loss
            if args.pu_loss == "puauc":
                task_loss = puauc_pairwise_loss(
                    logits, y, margin=args.pu_auc_margin, max_pairs=args.pu_auc_max_pairs
                )
            elif args.pu_loss == "nnpu":
                task_loss = nnpu_loss_from_logits(logits, y, pi)
            else:
                task_loss = F.binary_cross_entropy_with_logits(logits, y)
            
            # Node separation loss
            node_loss = task_loss.new_zeros(())
            if ep >= warmup_epochs and args.node_loss_weight > 0 and w is not None:
                node_loss = node_separation_loss(h, w, batch, y)
            
            # Contrastive loss
            contrastive_loss = task_loss.new_zeros(())
            if ep >= warmup_epochs and args.contrastive_loss_weight > 0:
                contrastive_loss = graph_contrastive_loss(g, y)
            
            # Total loss
            loss = (args.task_loss_weight * task_loss + 
                    args.node_loss_weight * node_loss + 
                    args.contrastive_loss_weight * contrastive_loss)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            task_sum += task_loss.detach().item() * data.num_graphs
            node_sum += node_loss.detach().item() * data.num_graphs
            contrastive_sum += contrastive_loss.detach().item() * data.num_graphs
            n_graphs += data.num_graphs
        
        # Validation
        val_m = evaluate(model, loader_valid, device)
        
        if args.early_metric == "aupr":
            cur = val_m["AUPRC"]
        else:
            cur = val_m["AUC"]
        
        if cur > best["score"] + 1e-6:
            best["score"] = float(cur)
            best["state"] = {k: v.cpu() for k, v in model.state_dict().items()}
            best["epoch"] = ep
            no_improve = 0
        else:
            no_improve += 1
        
        print(
            f"[{fold_tag}] Ep {ep:03d} "
            f"Task={task_sum/max(n_graphs,1):.4f} "
            f"Node={node_sum/max(n_graphs,1):.4f} "
            f"Contr={contrastive_sum/max(n_graphs,1):.4f} "
            f"| Val AUPRC={val_m['AUPRC']:.3f} AUC={val_m['AUC']:.3f} "
            f"F1={val_m['F1']:.3f} MCC={val_m['MCC']:.3f} "
            f"(best={best['score']:.4f}, no_improve={no_improve}/{args.patience})"
        )
        
        if ep <= 8:
            no_improve = 0
        if ep > 8 and no_improve >= args.patience:
            print(f"[{fold_tag}] Early stopping at ep={ep}")
            break
    
    # Load best model
    if best["state"] is not None:
        model.load_state_dict({k: v.to(device) for k, v in best["state"].items()})
    
    # Save model
    model_path = Path(args.output_dir) / "models" / f"{fold_tag}_best.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best["state"], model_path)
    
    # Find best threshold on validation
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for data in loader_valid:
            data = data.to(device)
            prob = torch.sigmoid(model(data))
            ys.append(data.y.cpu().numpy().ravel())
            ps.append(prob.cpu().numpy().ravel())
    y_val = np.concatenate(ys)
    p_val = np.concatenate(ps)
    thr_f1, _ = find_best_threshold(y_val, p_val, metric="f1")
    
    # Test evaluation
    test_m = evaluate(model, loader_test, device, threshold=thr_f1)
    print(
        f"[{fold_tag}] TEST @Ep{best['epoch']} (thr={thr_f1:.3f}): "
        f"AUPRC={test_m['AUPRC']:.3f} AUC={test_m['AUC']:.3f} "
        f"F1={test_m['F1']:.3f} MCC={test_m['MCC']:.3f}"
    )
    
    test_m["thr_used"] = float(thr_f1)
    return test_m


def main():
    args = get_args()
    
    # Setup
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "models").mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = pd.read_csv(args.csv_path)
    folds = sorted(df["fold"].unique().tolist())
    
    results = []
    
    for f in folds:
        df_fold = df[df["fold"] == f].copy().reset_index(drop=True)
        
        # Normalize split column
        if "split" in df_fold.columns:
            s = df_fold["split"].astype(str).str.lower().str.strip()
            s = s.replace({
                "validation": "valid", "val": "valid", "dev": "valid",
                "training": "train", "tr": "train",
                "testing": "test", "te": "test",
            })
            df_fold["split_norm"] = s
        else:
            df_fold["split_norm"] = "train"
        
        df_train_full = df_fold[df_fold["split_norm"] == "train"].reset_index(drop=True)
        df_test = df_fold[df_fold["split_norm"] == "test"].reset_index(drop=True)
        
        df_valid = df_fold[df_fold["split_norm"] == "valid"].reset_index(drop=True)
        if len(df_valid) == 0:
            df_train, df_valid = stratified_split(df_train_full, val_ratio=args.val_ratio, seed=args.seed)
        else:
            df_train = df_train_full
        
        print("\n" + "="*80)
        print(f"[Fold {f}] train={len(df_train)} valid={len(df_valid)} test={len(df_test)}")
        print("="*80)
        
        fold_tag = f"{args.tag}_fold{f}"
        m = train_fold(args, df_train, df_valid, df_test, fold_tag, device)
        results.append({"fold": f, **m})
    
    # Summary
    res_df = pd.DataFrame(results)
    out_csv = Path(args.output_dir) / f"{args.tag}_summary.csv"
    res_df.to_csv(out_csv, index=False)
    
    print("\n===== Summary =====")
    print(res_df)
    
    summ = {
        "AUPRC_mean": float(res_df["AUPRC"].mean()),
        "AUPRC_std": float(res_df["AUPRC"].std(ddof=1)),
        "AUC_mean": float(res_df["AUC"].mean()),
        "AUC_std": float(res_df["AUC"].std(ddof=1)),
        "F1_mean": float(res_df["F1"].mean()),
        "F1_std": float(res_df["F1"].std(ddof=1)),
        "MCC_mean": float(res_df["MCC"].mean()),
        "MCC_std": float(res_df["MCC"].std(ddof=1)),
    }
    
    with open(Path(args.output_dir) / f"{args.tag}_summary.json", "w") as f:
        json.dump(summ, f, indent=2)
    
    print(f"\nResults saved to {args.output_dir}")
    print("Summary:", summ)


if __name__ == "__main__":
    main()

