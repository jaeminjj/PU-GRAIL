#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PU-GRAIL: Dataset classes
"""

import os
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

from torch_geometric.data import InMemoryDataset

from .utils import parse_pdb_as_graph


class ProteinGraphDataset(InMemoryDataset):
    """
    Dataset for protein graph data.
    
    Args:
        df: DataFrame with columns ['id_short', 'label', 'fold']
        pdb_dir: Directory containing PDB files
        esm_dir: Directory containing ESM embedding files (.npy)
        cache_dir: Directory for caching processed data
        use_positional_encoding: Whether to use positional encoding
        pos_encoding_dim: Dimension of positional encoding
        dist_thr: Distance threshold for structure edges
    """
    
    def __init__(
        self,
        df,
        pdb_dir,
        esm_dir,
        cache_dir="./cache",
        use_positional_encoding=True,
        pos_encoding_dim=32,
        dist_thr=8.0,
    ):
        self.df = df.reset_index(drop=True)
        self.pdb_dir = pdb_dir
        self.esm_dir = esm_dir
        self.cache_dir = cache_dir
        self.use_positional_encoding = use_positional_encoding
        self.pos_encoding_dim = pos_encoding_dim
        self.dist_thr = dist_thr
        
        super().__init__(self.processed_dir)

        try:
            self.data, self.slices = torch.load(
                self.processed_paths[0], 
                weights_only=False, 
                map_location="cpu"
            )
        except Exception as e:
            print(f"[Dataset] Cache load failed ({e}). Rebuilding...")
            if os.path.exists(self.processed_paths[0]):
                try:
                    os.remove(self.processed_paths[0])
                except:
                    pass
            self.process()
            self.data, self.slices = torch.load(
                self.processed_paths[0], 
                weights_only=False, 
                map_location="cpu"
            )

    @property
    def processed_dir(self) -> str:
        cache_root = Path(self.cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        return str(cache_root)

    @property
    def processed_file_names(self):
        key = (
            f"folds_{'-'.join(sorted(map(str, self.df['fold'].unique())))}_"
            f"n{len(self.df)}_h{abs(hash(tuple(self.df['id_short'].values))) % 10**8}"
        )
        return [f"pugrail_cached_{key}.pt"]

    def process(self):
        data_list = []
        skipped = 0
        reason = defaultdict(int)
        examples = defaultdict(list)
        preview_n = 5

        for _, row in self.df.iterrows():
            id_short = Path(str(row["id_short"])).stem
            label = int(row["label"])

            pdb_path = os.path.join(self.pdb_dir, f"{id_short}.pdb")
            esm_path = os.path.join(self.esm_dir, f"{id_short}.npy")

            if not os.path.exists(pdb_path):
                skipped += 1
                reason["missing_pdb"] += 1
                if len(examples["missing_pdb"]) < preview_n:
                    examples["missing_pdb"].append((id_short, pdb_path))
                continue
            
            if not os.path.exists(esm_path):
                skipped += 1
                reason["missing_esm"] += 1
                if len(examples["missing_esm"]) < preview_n:
                    examples["missing_esm"].append((id_short, esm_path))
                continue

            try:
                esm = np.load(esm_path, allow_pickle=False)
                if esm.ndim != 2 or esm.size == 0:
                    skipped += 1
                    reason["bad_esm_shape"] += 1
                    if len(examples["bad_esm_shape"]) < preview_n:
                        examples["bad_esm_shape"].append((id_short, getattr(esm, "shape", None)))
                    continue

                data = parse_pdb_as_graph(
                    pdb_path,
                    esm_residue=esm,
                    dist_thr=self.dist_thr,
                    use_positional_encoding=self.use_positional_encoding,
                    pos_encoding_dim=self.pos_encoding_dim,
                )
                
                if data is None or getattr(data, "num_nodes", 0) == 0:
                    skipped += 1
                    reason["empty_graph"] += 1
                    if len(examples["empty_graph"]) < preview_n:
                        examples["empty_graph"].append((id_short, pdb_path))
                    continue

                data.y = torch.tensor([label], dtype=torch.float32)
                data.id_short = id_short
                data.fold = int(row["fold"])
                data_list.append(data)

            except Exception as e:
                skipped += 1
                reason["exception"] += 1
                if len(examples["exception"]) < preview_n:
                    examples["exception"].append((id_short, str(e)))
                continue

        if skipped > 0:
            print(f"[Dataset] built={len(data_list)} skipped={skipped} total={len(self.df)}")
            if reason:
                print("  reasons:", dict(reason))
                for k, v in examples.items():
                    if v:
                        print(f"  - {k} examples:")
                        for it in v[:preview_n]:
                            print("    ", it)

        if len(data_list) == 0:
            raise RuntimeError(f"[Dataset] No valid graphs. Reasons={dict(reason)}")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

