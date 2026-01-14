#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PU-GRAIL: Prediction script for inference on new proteins

Usage:
    python -m pugrail.predict \
        --model_path /path/to/model.pt \
        --pdb_path /path/to/protein.pdb \
        --esm_path /path/to/protein.npy \
        --output_path /path/to/output.json
        
    # Or batch prediction:
    python -m pugrail.predict \
        --model_path /path/to/model.pt \
        --input_csv /path/to/proteins.csv \
        --pdb_dir /path/to/pdb/ \
        --esm_dir /path/to/esm/ \
        --output_path /path/to/predictions.csv
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch

from .model import PUGRAIL
from .utils import parse_pdb_as_graph


def get_args():
    parser = argparse.ArgumentParser(
        description="PU-GRAIL: Predict protective antigenicity"
    )
    
    # Model
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    
    # Single protein prediction
    parser.add_argument("--pdb_path", type=str, default=None,
                        help="Path to PDB file (single protein)")
    parser.add_argument("--esm_path", type=str, default=None,
                        help="Path to ESM embedding file (single protein)")
    
    # Batch prediction
    parser.add_argument("--input_csv", type=str, default=None,
                        help="CSV file with id_short column for batch prediction")
    parser.add_argument("--pdb_dir", type=str, default=None,
                        help="Directory containing PDB files")
    parser.add_argument("--esm_dir", type=str, default=None,
                        help="Directory containing ESM embeddings")
    
    # Output
    parser.add_argument("--output_path", type=str, default="predictions.json",
                        help="Output path for predictions")
    
    # Model architecture (should match training)
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
    
    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold")
    
    return parser.parse_args()


def predict_single(model, pdb_path, esm_path, device, pos_encoding_dim=32):
    """Predict antigenicity for a single protein."""
    
    # Load ESM embedding
    esm_emb = np.load(esm_path, allow_pickle=False)
    
    # Parse PDB and create graph
    graph = parse_pdb_as_graph(
        pdb_path,
        esm_residue=esm_emb,
        use_positional_encoding=True,
        pos_encoding_dim=pos_encoding_dim,
    )
    
    if graph is None:
        return None
    
    # Add batch dimension
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    graph = graph.to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        logit = model(graph)
        prob = torch.sigmoid(logit).item()
    
    return prob


def predict_batch(model, df, pdb_dir, esm_dir, device, pos_encoding_dim=32):
    """Predict antigenicity for multiple proteins."""
    
    results = []
    
    for _, row in df.iterrows():
        id_short = str(row["id_short"])
        pdb_path = os.path.join(pdb_dir, f"{id_short}.pdb")
        esm_path = os.path.join(esm_dir, f"{id_short}.npy")
        
        if not os.path.exists(pdb_path):
            print(f"Warning: PDB file not found for {id_short}")
            results.append({"id_short": id_short, "probability": None, "error": "missing_pdb"})
            continue
        
        if not os.path.exists(esm_path):
            print(f"Warning: ESM file not found for {id_short}")
            results.append({"id_short": id_short, "probability": None, "error": "missing_esm"})
            continue
        
        try:
            prob = predict_single(model, pdb_path, esm_path, device, pos_encoding_dim)
            if prob is not None:
                results.append({"id_short": id_short, "probability": prob, "error": None})
            else:
                results.append({"id_short": id_short, "probability": None, "error": "parse_error"})
        except Exception as e:
            print(f"Error processing {id_short}: {e}")
            results.append({"id_short": id_short, "probability": None, "error": str(e)})
    
    return results


def main():
    args = get_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Determine input dimension based on architecture
    # 20 (AA) + ESM_proj (256) + PE (32) = 308 (approximate, actual depends on ESM dim)
    # This will be overridden by the actual data
    
    # Load model checkpoint to get in_dim
    checkpoint = torch.load(args.model_path, map_location="cpu")
    
    # Try to infer in_dim from checkpoint
    if "esm_proj.net.1.weight" in checkpoint:
        esm_dim = checkpoint["esm_proj.net.1.weight"].shape[1]
        in_dim = 20 + esm_dim + args.pos_encoding_dim
    else:
        # Fallback: typical ESM-2 650M dimension
        in_dim = 20 + 1280 + args.pos_encoding_dim
    
    # Create model
    model = PUGRAIL(
        in_dim=in_dim,
        hid=args.hidden_dim,
        gnn_layers=args.gnn_layers,
        use_edge_separation=args.use_edge_separation,
        pos_encoding_dim=args.pos_encoding_dim,
        esm_proj_dim=args.esm_proj_dim,
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint)
    model.eval()
    print("Model loaded successfully")
    
    # Predict
    if args.pdb_path and args.esm_path:
        # Single protein prediction
        print(f"\nPredicting for: {args.pdb_path}")
        prob = predict_single(model, args.pdb_path, args.esm_path, device, args.pos_encoding_dim)
        
        if prob is not None:
            prediction = "Protective Antigen" if prob >= args.threshold else "Non-Protective"
            result = {
                "pdb_path": args.pdb_path,
                "probability": prob,
                "threshold": args.threshold,
                "prediction": prediction,
            }
            print(f"\n{'='*50}")
            print(f"Probability: {prob:.4f}")
            print(f"Prediction: {prediction}")
            print(f"{'='*50}")
            
            # Save result
            output_path = Path(args.output_path)
            if output_path.suffix == ".json":
                with open(output_path, "w") as f:
                    json.dump(result, f, indent=2)
            else:
                with open(output_path, "w") as f:
                    f.write(f"probability\tprediction\n{prob:.4f}\t{prediction}\n")
            print(f"\nResult saved to: {output_path}")
        else:
            print("Error: Could not parse input files")
    
    elif args.input_csv and args.pdb_dir and args.esm_dir:
        # Batch prediction
        print(f"\nBatch prediction from: {args.input_csv}")
        df = pd.read_csv(args.input_csv)
        
        results = predict_batch(model, df, args.pdb_dir, args.esm_dir, device, args.pos_encoding_dim)
        
        # Add predictions
        results_df = pd.DataFrame(results)
        results_df["prediction"] = results_df["probability"].apply(
            lambda x: "Protective Antigen" if x is not None and x >= args.threshold else 
                      ("Non-Protective" if x is not None else "Error")
        )
        
        # Save results
        output_path = Path(args.output_path)
        if output_path.suffix == ".csv":
            results_df.to_csv(output_path, index=False)
        else:
            results_df.to_csv(output_path.with_suffix(".csv"), index=False)
        
        # Summary
        n_total = len(results_df)
        n_success = results_df["probability"].notna().sum()
        n_positive = (results_df["probability"] >= args.threshold).sum()
        
        print(f"\n{'='*50}")
        print(f"Total proteins: {n_total}")
        print(f"Successfully processed: {n_success}")
        print(f"Predicted protective antigens: {n_positive}")
        print(f"{'='*50}")
        print(f"\nResults saved to: {output_path}")
    
    else:
        print("Error: Please provide either:")
        print("  1. --pdb_path and --esm_path for single protein prediction")
        print("  2. --input_csv, --pdb_dir, and --esm_dir for batch prediction")


if __name__ == "__main__":
    main()

