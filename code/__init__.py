#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PU-GRAIL: Graph Neural Network for Protective Antigen Prediction

A graph neural network framework that integrates protein language model 
embeddings with predicted 3D structures under a positive-unlabeled (PU) 
learning paradigm for protective antigen prediction.
"""

from .model import PUGRAIL, AttnReadout, ESMProjector
from .dataset import ProteinGraphDataset
from .utils import (
    parse_pdb_as_graph,
    puauc_pairwise_loss,
    nnpu_loss_from_logits,
    graph_contrastive_loss,
    node_separation_loss,
)

__version__ = "1.0.0"
__author__ = "Jaemin Jeon"

__all__ = [
    "PUGRAIL",
    "AttnReadout",
    "ESMProjector",
    "ProteinGraphDataset",
    "parse_pdb_as_graph",
    "puauc_pairwise_loss",
    "nnpu_loss_from_logits",
    "graph_contrastive_loss",
    "node_separation_loss",
]

